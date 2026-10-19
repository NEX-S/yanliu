use super::Result;
use ::openvino::{Core, DeviceType, ElementType, InferRequest, RwPropertyKey, Shape, Tensor};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Mutex,
};

const BATCH: usize = 9;
pub(super) const CONTEXT: usize = 64;

/// The IR prepared from one exact source model.
#[derive(Deserialize)]
struct Metadata {
    format: String,
    source_sha256: String,
    xml_sha256: String,
    bin_sha256: String,
}

pub(super) struct Ranker {
    request: InferRequest,
    tokens: Tensor,
    mask: Tensor,
    index: HashMap<char, i64>,
}

fn checked_bytes(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(format!("{}: SHA-256 does not match ranker metadata", path.display()).into());
    }
    Ok(bytes)
}

impl Ranker {
    pub(super) fn load(source: &Path, runtime_dir: &Path, device: &str) -> Result<Self> {
        let device = match device {
            "gpu" => DeviceType::GPU,
            "npu" => DeviceType::NPU,
            _ => return Err("accelerator device must be gpu or npu".into()),
        };
        // Verify the selected source and its fixed IR before loading the device.
        let path = source.with_extension("openvino.json");
        let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let info: Metadata = serde_json::from_slice(&bytes)?;
        if info.format != "yanliu-ranker-ir-v1" {
            return Err(format!("{}: unsupported OpenVINO ranker metadata", path.display()).into());
        }
        let model = checked_bytes(source, &info.source_sha256)?;
        let length = u64::from_le_bytes(
            model
                .get(..8)
                .ok_or("truncated safetensors header")?
                .try_into()?,
        );
        let header: Value = serde_json::from_slice(
            model
                .get(8..)
                .and_then(|rest| rest.get(..usize::try_from(length).ok()?))
                .ok_or("truncated safetensors header")?,
        )?;
        let vocab: Vec<String> = serde_json::from_str(
            header["__metadata__"]["vocab"]
                .as_str()
                .ok_or("source model has no vocabulary")?,
        )?;
        // Same mapping as the CPU model: reserved tokens are spelled `<pad>` and never match a character.
        let index = vocab
            .iter()
            .enumerate()
            .filter_map(|(token, text)| {
                let mut chars = text.chars();
                match (chars.next(), chars.next()) {
                    (Some(ch), None) => Some((ch, token as i64)),
                    _ => None,
                }
            })
            .collect();
        let xml = checked_bytes(&source.with_extension("xml"), &info.xml_sha256)?;
        let bin = checked_bytes(&source.with_extension("bin"), &info.bin_sha256)?;
        let library = runtime_dir.join("libopenvino_c.so");
        let library = library
            .canonicalize()
            .map_err(|error| format!("{}: {error}", library.display()))?;
        // The official bindings load one runtime for the process; reject a different explicit path.
        static LOADED: Mutex<Option<(PathBuf, Vec<libloading::Library>)>> = Mutex::new(None);
        let mut loaded = LOADED.lock().expect("OpenVINO loader lock poisoned");
        if loaded.as_ref().is_some_and(|(path, _)| path != &library) {
            return Err("a different OpenVINO runtime is already loaded in this process".into());
        }
        if loaded.is_none() {
            let runtime_dir = library
                .parent()
                .expect("canonical library path has a parent");
            let mut dependencies = Vec::new();
            for name in ["libtbb.so.12", "libopenvino.so.2600"] {
                // Official archive libraries have no RUNPATH; preload their explicit siblings.
                dependencies.push(unsafe { libloading::Library::new(runtime_dir.join(name))? });
            }
            openvino_sys::library::load_from(library.clone()).map_err(std::io::Error::other)?;
            *loaded = Some((library, dependencies));
        }
        drop(loaded);
        let mut core = Core::new()?;
        core.set_property(&device, &RwPropertyKey::HintPerformanceMode, "LATENCY")?;
        if device == DeviceType::GPU {
            core.set_property(&device, &RwPropertyKey::HintInferencePrecision, "f32")?;
        }
        let mut weights = Tensor::new(ElementType::U8, &Shape::new(&[bin.len().try_into()?])?)?;
        weights.get_data_mut::<u8>()?.copy_from_slice(&bin);
        let model = core.read_model_from_buffer(&xml, Some(&weights))?;
        if model.get_inputs_len()? != 2 || model.get_outputs_len()? != 1 {
            return Err("ranker IR must have two inputs and one output".into());
        }
        for (i, name, element, shape) in [
            (
                0,
                "tokens",
                ElementType::I64,
                [BATCH as i64, CONTEXT as i64],
            ),
            (
                1,
                "target_mask",
                ElementType::F32,
                [BATCH as i64, CONTEXT as i64 - 1],
            ),
        ] {
            let port = model.get_input_by_index(i)?;
            if port.get_name()? != name
                || port.get_element_type()? != element
                || port.get_shape()?.get_dimensions() != shape
            {
                return Err(
                    format!("ranker IR input {i} must be {name} {element:?} {shape:?}").into(),
                );
            }
        }
        let mut compiled = core.compile_model(&model, device)?;
        // Inference reads the output buffer as f32, so check the artifact the device compiled.
        let output = compiled.get_output()?;
        if output.get_element_type()? != ElementType::F32
            || output.get_shape()?.get_dimensions() != [BATCH as i64]
        {
            return Err("ranker IR output must be f32[9]".into());
        }
        let mut request = compiled.create_infer_request()?;
        let tokens = Tensor::new(
            ElementType::I64,
            &Shape::new(&[BATCH as i64, CONTEXT as i64])?,
        )?;
        let mask = Tensor::new(
            ElementType::F32,
            &Shape::new(&[BATCH as i64, CONTEXT as i64 - 1])?,
        )?;
        request.set_tensor("tokens", &tokens)?;
        request.set_tensor("target_mask", &mask)?;
        Ok(Self {
            request,
            tokens,
            mask,
            index,
        })
    }

    pub(super) fn log_probabilities(
        &mut self,
        context: &str,
        candidates: &[&str],
    ) -> Result<Vec<f32>> {
        encode(
            &self.index,
            context,
            candidates,
            self.tokens.get_data_mut::<i64>()?,
            self.mask.get_data_mut::<f32>()?,
        );
        self.request.infer()?;
        Ok(self.request.get_output_tensor()?.get_data::<f32>()?[..candidates.len()].to_vec())
    }
}

/// Callers pass at most nine nonempty candidates shorter than the window; only context is trimmed.
fn encode(
    index: &HashMap<char, i64>,
    context: &str,
    candidates: &[&str],
    tokens: &mut [i64],
    mask: &mut [f32],
) {
    let longest = candidates
        .iter()
        .map(|text| text.chars().count())
        .max()
        .unwrap_or(0);
    let context: Vec<_> = context.chars().collect();
    let context = &context[context.len().saturating_sub(CONTEXT - longest - 1)..];
    tokens.fill(0);
    mask.fill(0.0);
    for (row, candidate) in candidates.iter().enumerate() {
        let row_tokens = &mut tokens[row * CONTEXT..(row + 1) * CONTEXT];
        row_tokens[0] = 2;
        for (position, ch) in context.iter().copied().chain(candidate.chars()).enumerate() {
            row_tokens[position + 1] = index.get(&ch).copied().unwrap_or(1);
            if position >= context.len() {
                mask[row * (CONTEXT - 1) + position] = 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_keeps_complete_candidates_and_trims_only_context() {
        let index = HashMap::from([('甲', 3), ('乙', 4), ('丙', 5)]);
        let mut tokens = [0; BATCH * CONTEXT];
        let mut mask = [0.0; BATCH * (CONTEXT - 1)];
        encode(
            &index,
            &"甲".repeat(100),
            &["乙丙", "乙"],
            &mut tokens,
            &mut mask,
        );
        assert_eq!(&tokens[61..64], &[3, 4, 5]);
        assert_eq!(&tokens[CONTEXT + 61..CONTEXT + 64], &[3, 4, 0]);
        assert_eq!(&mask[61..63], &[1.0, 1.0]);
        assert_eq!(&mask[63 + 61..63 + 63], &[1.0, 0.0]);
    }

    #[test]
    fn selected_model_source_hash_is_checked() {
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors");
        assert!(checked_bytes(&source, &"0".repeat(64)).is_err());
    }

    #[test]
    #[ignore = "requires local GPU/NPU and pinned runtime; set YANLIU_TEST_DEVICE"]
    fn actual_accelerator_matches_cpu_candidate_scores() {
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors");
        let runtime = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/runtime");
        let device =
            std::env::var("YANLIU_TEST_DEVICE").expect("set YANLIU_TEST_DEVICE=gpu or npu");
        let mut ranker = Ranker::load(&source, &runtime, &device).unwrap();
        let mut cpu = chinese_ime_lm::Reranker::new(std::sync::Arc::new(
            chinese_ime_lm::SentenceModel::load(&std::fs::read(&source).unwrap()).unwrap(),
        ));
        for (context, candidates) in [
            ("我去", vec!["现", "西安"]),
            ("请检查", vec!["编译错误", "边译错误", "编译", "变异"]),
        ] {
            let actual = ranker.log_probabilities(context, &candidates).unwrap();
            let expected = cpu.log_probabilities(context, &candidates);
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < if device == "gpu" { 0.0001 } else { 0.03 },
                    "{device}: {actual} != {expected}"
                );
            }
        }
    }
}

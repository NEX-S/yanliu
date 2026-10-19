use clap::Parser;
use serde::{Deserialize, Deserializer};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const DEVICES: [&str; 3] = ["cpu", "gpu", "npu"];

#[derive(Parser)]
#[command(name = "yanliu-ibus", about = "Offline GNOME Chinese input service")]
pub struct Args {
    // IBus component command lines pass --ibus; it selects nothing.
    #[arg(long)]
    pub ibus: bool,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub ibus_address: Option<String>,
    #[arg(long, requires = "ibus_address")]
    pub register: bool,
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, value_parser = DEVICES)]
    pub device: Option<String>,
    #[arg(long)]
    pub runtime_dir: Option<PathBuf>,
    #[arg(long)]
    pub user_dictionary: Option<PathBuf>,
    #[arg(long)]
    pub contextual_learning: bool,
    #[arg(long)]
    pub no_learning: bool,
}

// A missing optional field is allowed; explicit null is a type error.
fn supplied<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    de: D,
) -> std::result::Result<Option<T>, D::Error> {
    T::deserialize(de).map(Some)
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default, deserialize_with = "supplied")]
    data_dir: Option<PathBuf>,
    #[serde(default, deserialize_with = "supplied")]
    learning_enabled: Option<bool>,
    #[serde(default, deserialize_with = "supplied")]
    model: Option<PathBuf>,
    #[serde(default, deserialize_with = "supplied")]
    device: Option<String>,
    #[serde(default, deserialize_with = "supplied")]
    runtime_dir: Option<PathBuf>,
    #[serde(default, deserialize_with = "supplied")]
    user_dictionary: Option<PathBuf>,
    #[serde(default, deserialize_with = "supplied")]
    contextual_learning: Option<bool>,
}

/// Paths in the file are relative to its directory.
fn read_config(path: &Path) -> Result<FileConfig> {
    let bytes = std::fs::read(path)?;
    // Structs also deserialize from JSON arrays, field by field.
    if !serde_json::from_slice::<serde_json::Value>(&bytes)?.is_object() {
        return Err("config must be an object".into());
    }
    let file: FileConfig = serde_json::from_slice(&bytes)?;
    if file
        .device
        .as_deref()
        .is_some_and(|d| !DEVICES.contains(&d))
    {
        return Err("device must be cpu, gpu or npu".into());
    }
    let base = path.parent().expect("a readable file has a parent");
    Ok(FileConfig {
        data_dir: file.data_dir.map(|p| base.join(p)),
        model: file.model.map(|p| base.join(p)),
        runtime_dir: file.runtime_dir.map(|p| base.join(p)),
        user_dictionary: file.user_dictionary.map(|p| base.join(p)),
        ..file
    })
}

#[derive(Debug)]
pub struct Settings {
    pub data_dir: PathBuf,
    pub model: PathBuf,
    pub device: String,
    /// Present exactly when `device` is an accelerator.
    pub runtime_dir: Option<PathBuf>,
    pub user_dictionary: Option<PathBuf>,
    pub learning_enabled: bool,
    pub contextual_learning: bool,
}
impl Settings {
    /// Defaults, then the config file, then explicit command-line values.
    pub fn resolve(args: &Args) -> Result<Self> {
        let file = match &args.config {
            Some(path) => {
                read_config(path).map_err(|cause| format!("{}: {cause}", path.display()))?
            }
            None => FileConfig::default(),
        };
        let data_dir = args
            .data_dir
            .clone()
            .or(file.data_dir)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
        let device = args
            .device
            .clone()
            .or(file.device)
            .unwrap_or_else(|| "cpu".into());
        let runtime_dir = args.runtime_dir.clone().or(file.runtime_dir);
        if (device == "cpu") == runtime_dir.is_some() {
            return Err("runtime_dir is required for gpu/npu and must be absent for cpu".into());
        }
        let learning_enabled = !args.no_learning && file.learning_enabled.unwrap_or(true);
        Ok(Self {
            model: args
                .model
                .clone()
                .or(file.model)
                .unwrap_or_else(|| data_dir.join("models/sentence-model.safetensors")),
            data_dir,
            device,
            runtime_dir,
            user_dictionary: args.user_dictionary.clone().or(file.user_dictionary),
            learning_enabled,
            contextual_learning: learning_enabled
                && (args.contextual_learning || file.contextual_learning.unwrap_or(false)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn resolve(args: &[&str]) -> Result<Settings> {
        let args = std::iter::once("yanliu-ibus").chain(args.iter().copied());
        Settings::resolve(&Args::try_parse_from(args)?)
    }
    fn with_config(body: &str, check: impl FnOnce(&Path)) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "yanliu-config-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("config.json");
        std::fs::write(&path, body).unwrap();
        check(&path);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn defaults_and_command_line_without_config() {
        let defaults = resolve(&[]).unwrap();
        assert_eq!(
            defaults.model,
            defaults.data_dir.join("models/sentence-model.safetensors")
        );
        assert_eq!(defaults.device, "cpu");
        assert_eq!(
            (defaults.runtime_dir, defaults.user_dictionary),
            (None, None)
        );
        assert!(defaults.learning_enabled && !defaults.contextual_learning);
        let contextual = resolve(&["--contextual-learning"]).unwrap();
        assert!(contextual.learning_enabled && contextual.contextual_learning);
        let disabled = resolve(&["--contextual-learning", "--no-learning"]).unwrap();
        assert!(!disabled.learning_enabled && !disabled.contextual_learning);
        let npu = resolve(&["--device", "npu", "--runtime-dir", "runtime"]).unwrap();
        assert_eq!(npu.device, "npu");
        assert_eq!(npu.runtime_dir, Some(PathBuf::from("runtime")));
        for args in [
            &["--device", "gpu"][..],
            &["--runtime-dir", "runtime"],
            &["--device", "auto", "--runtime-dir", "runtime"],
            &["--register"],
        ] {
            assert!(resolve(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn config_paths_follow_its_directory_and_command_line_wins() {
        let body = r#"{"data_dir":"assets","user_dictionary":"words.dict.yaml","device":"gpu","runtime_dir":"runtime"}"#;
        with_config(body, |path| {
            let base = path.parent().unwrap();
            let config = path.to_str().unwrap();
            let settings = resolve(&["--config", config]).unwrap();
            assert_eq!(settings.data_dir, base.join("assets"));
            assert_eq!(
                settings.model,
                base.join("assets/models/sentence-model.safetensors")
            );
            assert_eq!(settings.user_dictionary, Some(base.join("words.dict.yaml")));
            assert_eq!(settings.device, "gpu");
            assert_eq!(settings.runtime_dir, Some(base.join("runtime")));
            let cli = resolve(&[
                "--config",
                config,
                "--data-dir",
                "cli",
                "--user-dictionary",
                "cli-words",
            ])
            .unwrap();
            assert_eq!(cli.data_dir, PathBuf::from("cli"));
            assert_eq!(
                cli.model,
                PathBuf::from("cli/models/sentence-model.safetensors")
            );
            assert_eq!(cli.user_dictionary, Some(PathBuf::from("cli-words")));
            let model = resolve(&["--config", config, "--model", "cli-model"]).unwrap();
            assert_eq!(model.model, PathBuf::from("cli-model"));
        });
        with_config(r#"{"model":"weights"}"#, |path| {
            let settings = resolve(&["--config", path.to_str().unwrap(), "--data-dir", "cli"]);
            assert_eq!(
                settings.unwrap().model,
                path.parent().unwrap().join("weights")
            );
        });
    }

    #[test]
    fn config_learning_switches_yield_to_no_learning() {
        for (body, learning_enabled, contextual_learning) in [
            (r#"{"learning_enabled":false}"#, false, false),
            (r#"{"learning_enabled":true}"#, true, false),
            (r#"{"contextual_learning":true}"#, true, true),
            (
                r#"{"learning_enabled":false,"contextual_learning":true}"#,
                false,
                false,
            ),
        ] {
            with_config(body, |path| {
                let config = path.to_str().unwrap();
                let settings = resolve(&["--config", config]).unwrap();
                assert_eq!(
                    (settings.learning_enabled, settings.contextual_learning),
                    (learning_enabled, contextual_learning),
                    "{body}"
                );
                let disabled = resolve(&["--config", config, "--no-learning"]).unwrap();
                assert!(!disabled.learning_enabled && !disabled.contextual_learning);
            });
        }
    }

    #[test]
    fn invalid_config_reports_path_and_cause() {
        let mut cases = vec![
            ("{".to_owned(), "EOF while parsing an object"),
            ("[]".into(), "config must be an object"),
            (r#"{"unknown":true}"#.into(), "unknown field `unknown`"),
            (
                r#"{"data_dir":false}"#.into(),
                "invalid type: boolean `false`",
            ),
            (
                r#"{"learning_enabled":"false"}"#.into(),
                "invalid type: string",
            ),
            (
                r#"{"user_dictionary":"a","user_dictionary":"b"}"#.into(),
                "duplicate field `user_dictionary`",
            ),
            (
                r#"{"device":"auto","runtime_dir":"runtime"}"#.into(),
                "device must be cpu, gpu or npu",
            ),
        ];
        for field in [
            "data_dir",
            "learning_enabled",
            "model",
            "device",
            "runtime_dir",
            "user_dictionary",
            "contextual_learning",
        ] {
            cases.push((format!(r#"{{"{field}":null}}"#), "invalid type: null"));
        }
        for (body, cause) in cases {
            with_config(&body, |path| {
                let error = resolve(&["--config", path.to_str().unwrap()])
                    .unwrap_err()
                    .to_string();
                let prefix = format!("{}: ", path.display());
                assert!(
                    error.starts_with(&prefix) && error.contains(cause),
                    "{error}"
                );
            });
        }
        let missing = "/nonexistent/yanliu-config-test/config.json";
        let error = resolve(&["--config", missing]).unwrap_err().to_string();
        assert!(error.starts_with(&format!("{missing}: ")), "{error}");
    }
}

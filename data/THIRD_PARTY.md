# Third-party assets

`manifest.json` pins the URL, revision, license, size and SHA-256 of every
upstream file. `python3 scripts/fetch_assets.py` downloads missing files and
rejects mismatched bytes; `--verify-only` fails if any local file is missing or
differs. For a locally patched file the manifest records the upstream hash, the
patch hash and the effective hash; upstream bytes are verified before the patch
is applied.

## Rime Ice dictionaries

Source: https://github.com/iDvel/rime-ice/tree/da1fbe602e38f26db846fa10120ee64c2b0324c0/cn_dicts

`8105.dict.yaml`, `base.dict.yaml` and the upstream rows of `ext.dict.yaml` keep
their weights and headers. The headers carry further attribution and provenance,
including THUOCL and Tencent word vectors; these notices must travel with the
files. The upstream GPL-3.0 license is kept as `dictionaries/LICENSE`.

`ext.dict.local.patch` appends computer, Linux and science headwords that the
decoder did not return for their full pinyin, each with weight 100 and a reading
present in `8105.dict.yaml`. The first group came from local Codex, Claude and
Grok transcripts, which are not stored in this repository; the source document,
revision, hash and license of each public-source addition are recorded in
`ext.dict.domain-sources.json`. The appendix is not part of upstream rime-ice.

## Rust inference library

Source: https://github.com/metasequoiaime/chinese-ime-lm/tree/ed030178ebf7f361a08b57c6b179d512ea47042f/reference

`vendor/chinese-ime-lm` keeps the upstream library files (`Cargo.toml`,
`src/lib.rs`, `src/weights.rs`) and their Apache-2.0 `LICENSE`; examples and
integration fixtures are omitted. Local changes to `src/lib.rs` are kept in
`yanliu.patch`. The GPL-3.0 license of the main upstream repository is distinct
from the Apache-2.0 license granted to `reference/`.

## Default language model

The weight is `sentence-model.safetensors`, asset 572758937 of the
metasequoiaime/chinese-ime-lm GitHub release `model-v1`. Release assets can be
replaced upstream, so the manifest pins its SHA-256 and size. The matching model
card contains metadata only:
https://huggingface.co/houko/pinyin-ime-reranker-4M/tree/615ce69d84b938de61915d95386000df59637d5a

The weights are Apache-2.0. The intact safetensors metadata carries that license
and the training-corpus attribution, Chinese C4 (ODC-BY,
https://huggingface.co/datasets/allenai/c4) and LCCC (MIT,
https://github.com/thu-coai/CDial-GPT), which must be kept with redistributed
weights. The upstream `NOTICE` is kept as `vendor/chinese-ime-training/NOTICE`.

## Model definition for accelerator export

Source: https://github.com/metasequoiaime/chinese-ime-lm/tree/ed030178ebf7f361a08b57c6b179d512ea47042f

`vendor/chinese-ime-training` keeps the unmodified upstream `neural/model.py`
with the repository's GPL-3.0 `LICENSE` and `NOTICE`. `neural/initialize.py` is
Yanliu's own loader that reads a safetensors artifact into that model for
`scripts/export_ranker.py`. `UPSTREAM.json` records the original and local
hashes.

## Tatoeba replay diagnostic v3

Source: https://downloads.tatoeba.org/exports/per_language/cmn/cmn_sentences_detailed.tsv.bz2,
2026-10-10 export, SHA-256
`a8c53945a8806ce7b6e220585294bf2df4c021f2adfe9d0efc49b6db7e9c0af5`, licensed
CC BY 2.0 FR.

`eval/replay-v3-dev.jsonl` and `eval/replay-v3-test.jsonl` adapt those
sentences: automatic pinyin input, punctuation removed from targets and sentence
prefixes as context. Every case keeps its author, sentence URL and license in
`source`; `eval/replay-v3-annotations.jsonl` keeps each original sentence,
author and timestamps. `eval/replay-v3.NOTICE` states the attribution,
`eval/replay-v3.lock.json` the selection rules and source hashes, and
`eval/replay-v3-artifacts.json` the generated file hashes.

## Optional native acceleration

The optional `accelerator` Cargo feature uses Intel's Apache-2.0
[openvino-rs](https://github.com/intel/openvino-rs) bindings (versions in
`Cargo.lock`). The default CPU build does not load OpenVINO.
`scripts/export_ranker.py` derives a fixed FP32 IR from the explicitly selected
safetensors through the vendored model definition; each output sidecar binds
the original model, config, vocabulary, license, attribution and XML/BIN
SHA-256. Exporting does not train or change source weights.

`scripts/prepare_accelerator_runtime.py` verifies the official OpenVINO 2026.0.0
archive before extracting the native core/C API, IR frontend, TBB/HWLOC, GPU/NPU
plugins and their license files. The output `runtime-source.json` keeps the
official archive URL/hash and every extracted file hash. The installer verifies
and copies only the chosen device libraries and licenses; it omits GenAI,
tokenizers and the CPU plugin. These optional generated assets are prepared
explicitly and are not fetched by `scripts/fetch_assets.py`.

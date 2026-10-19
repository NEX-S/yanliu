use clap::{Parser, Subcommand};
use std::{error::Error, path::PathBuf, sync::Arc};
use yanliu::{Decoder, Dictionary, Session, model::Model};
#[path = "yanliu/evaluate.rs"]
mod evaluate;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Parser)]
#[command(about = "砚流: offline pinyin and learning diagnostics")]
struct Args {
    #[arg(long, default_value = "data/dictionaries")]
    dictionary_dir: PathBuf,
    /// Explicit user-managed Rime subset, read once at startup.
    #[arg(long)]
    user_dictionary: Option<PathBuf>,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "gpu", "npu"])]
    device: String,
    #[arg(long)]
    runtime_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Decode pinyin with the dictionary and optional local language model.
    Decode {
        input: String,
        #[arg(long, default_value = "")]
        context: String,
    },
    /// Check JSONL cases through the same incremental Session used by IBus.
    Evaluate {
        /// JSONL file, or - for stdin. Fields: id, input, context, gold (or golds), bucket.
        #[arg(long)]
        data: PathBuf,
        /// New JSON report file, or - for stdout.
        #[arg(long, default_value = "-")]
        output: PathBuf,
    },
    /// Inspect or clear system selection preferences without loading assets.
    Learning {
        #[command(subcommand)]
        command: LearningCommand,
    },
}

#[derive(Subcommand)]
enum LearningCommand {
    List,
    Clear,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let assets = || -> Result<(Arc<Decoder>, Option<Model>)> {
        let decoder = Arc::new(Decoder::new(Dictionary::load_dir_with_user_dictionary(
            &args.dictionary_dir,
            args.user_dictionary.as_deref(),
        )?));
        let model = match &args.model {
            Some(path) => Some(Model::load(
                path,
                &args.device,
                args.runtime_dir.as_deref(),
            )?),
            None if args.device != "cpu" || args.runtime_dir.is_some() => {
                return Err("--device gpu/npu and --runtime-dir require --model".into());
            }
            None => None,
        };
        Ok((decoder, model))
    };
    match args.command {
        Command::Decode { input, context } => {
            let (decoder, mut model) = assets()?;
            let mut session = Session::new(decoder, None);
            session.insert(&input)?;
            session.set_context(&context);
            if let Some(model) = &mut model {
                promote(model, &mut session)?;
            }
            for (i, c) in session.candidates().iter().enumerate() {
                println!(
                    "{}\t{}\tconsumed={}\t{:?}\t{:.4}",
                    i + 1,
                    c.text,
                    c.consumed,
                    c.source,
                    c.score
                );
            }
        }
        Command::Evaluate { data, output } => {
            let (decoder, model) = assets()?;
            evaluate::run(&data, &output, &decoder, model)?;
        }
        Command::Learning { command } => {
            let directory = yanliu::learning::state_directory()?;
            if let LearningCommand::Clear = command {
                yanliu::learning::clear(&directory)?;
            }
            let state = yanliu::learning::list(&directory)?;
            println!("{}", serde_json::to_string_pretty(&state)?);
        }
    }
    Ok(())
}

fn promote(model: &mut Model, session: &mut Session) -> Result<()> {
    let snapshot = session.snapshot();
    if let Some(index) = model.promotion(&snapshot)? {
        session.apply_promotion(snapshot.revision, index);
    }
    Ok(())
}

#[path = "yanliu-ibus/config.rs"]
mod config;
use std::sync::Arc;
use tokio::signal::unix::{SignalKind, signal};
use yanliu::{
    Decoder, Dictionary, ibus,
    learning::{self, LearningWriter},
    model::Model,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?
        .block_on(service())
}

async fn service() -> Result<()> {
    let args = <config::Args as clap::Parser>::parse();
    let settings = config::Settings::resolve(&args)?;
    eprintln!(
        "yanliu config: {}; {} candidate promotion; learning_enabled={}, contextual_learning={}",
        args.config
            .as_ref()
            .map_or_else(|| "defaults + CLI".to_owned(), |p| p.display().to_string()),
        settings.device,
        settings.learning_enabled,
        settings.contextual_learning
    );
    let decoder = Arc::new(Decoder::new(Dictionary::load_dir_with_user_dictionary(
        settings.data_dir.join("dictionaries"),
        settings.user_dictionary.as_deref(),
    )?));
    let model = Model::load(
        &settings.model,
        &settings.device,
        settings.runtime_dir.as_deref(),
    )?;
    let (failures, mut failed) = tokio::sync::mpsc::unbounded_channel();
    let report = move |cause: String| {
        // Only failures after shutdown has begun find no listener.
        let _ = failures.send(cause);
    };
    let mut learning_writer = if settings.learning_enabled {
        let directory = learning::state_directory()?;
        Some(LearningWriter::start(
            &directory,
            settings.contextual_learning,
            report.clone(),
        )?)
    } else {
        None
    };
    let learning = learning_writer
        .as_ref()
        .map(|writer| writer.learning.clone());
    let worker = ibus::RankingWorker::start(model, decoder.clone(), report)?;
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    let connection = ibus::serve(
        args.ibus_address.as_deref(),
        args.register,
        decoder,
        learning,
        &worker,
    )
    .await?;
    eprintln!("Yanliu IBus factory ready");
    let failure = tokio::select! {
        cause = failed.recv() => cause,
        _ = connection.closed() => Some("IBus connection closed".to_owned()),
        _ = terminate.recv() => None,
        _ = interrupt.recv() => None,
    };
    // Stop ranking, then end D-Bus input before the final learning flush.
    drop(worker);
    let closed = connection.close().await;
    let flushed = learning_writer
        .as_mut()
        .map_or(Ok(()), LearningWriter::finish);
    if let Some(cause) = failure {
        return Err(cause.into());
    }
    closed?;
    Ok(flushed?)
}

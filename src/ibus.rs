//! The GNOME IBus boundary. Candidate windows are owned by GNOME Shell.
use crate::{
    Candidate, CandidateSource, Decoder, Session, learning::Learning, model::Model,
    session::SessionSnapshot,
};
use std::{
    collections::{HashMap, VecDeque},
    panic::AssertUnwindSafe,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use zbus::{
    Connection, ObjectServer, interface,
    object_server::SignalEmitter,
    zvariant::{OwnedObjectPath, OwnedValue, Structure, StructureBuilder, Value},
};

const BUS_NAME: &str = "org.freedesktop.IBus.Yanliu";
const FACTORY_PATH: &str = "/org/freedesktop/IBus/Factory";
const PAGE_SIZE: usize = 9;
type Attachments = HashMap<String, OwnedValue>;

/// IBus.PreeditFocusMode.CLEAR: drop the preedit when focus leaves.
const PREEDIT_MODE_CLEAR: u32 = 0;
/// IBus.ModifierType bits.
const MODIFIER_LOCK: u32 = 1 << 1;
const MODIFIER_CONTROL: u32 = 1 << 2;
const MODIFIER_MOD1: u32 = 1 << 3;
const MODIFIER_MOD2: u32 = 1 << 4;
const MODIFIER_MOD4: u32 = 1 << 6;
const MODIFIER_SUPER: u32 = 1 << 26;
const MODIFIER_HYPER: u32 = 1 << 27;
const MODIFIER_META: u32 = 1 << 28;
const MODIFIER_RELEASE: u32 = 1 << 30;
/// GTK InputPurpose values relayed by the IBus ContentType property.
const INPUT_PURPOSE_PASSWORD: u32 = 8;
const INPUT_PURPOSE_PIN: u32 = 9;
/// IBus InputHint bits (1 << 11, 1 << 12) that mark password-style entry.
const SENSITIVE_INPUT_HINTS: u32 = (1 << 11) | (1 << 12);
/// X keysyms; values below 0x100 are their Latin-1 code points.
const KEYSYM_SPACE: u32 = 0x20;
const KEYSYM_APOSTROPHE: u32 = 0x27;
const KEYSYM_DIGIT_0: u32 = 0x30;
const KEYSYM_DIGIT_1: u32 = 0x31;
const KEYSYM_DIGIT_9: u32 = 0x39;
const KEYSYM_UPPER_A: u32 = 0x41;
const KEYSYM_UPPER_Z: u32 = 0x5a;
const KEYSYM_LOWER_A: u32 = 0x61;
const KEYSYM_LOWER_H: u32 = 0x68;
const KEYSYM_LOWER_J: u32 = 0x6a;
const KEYSYM_LOWER_K: u32 = 0x6b;
const KEYSYM_LOWER_L: u32 = 0x6c;
const KEYSYM_LOWER_Z: u32 = 0x7a;
const KEYSYM_BACKSPACE: u32 = 0xff08;
const KEYSYM_RETURN: u32 = 0xff0d;
const KEYSYM_ESCAPE: u32 = 0xff1b;
const KEYSYM_HOME: u32 = 0xff50;
const KEYSYM_LEFT: u32 = 0xff51;
const KEYSYM_UP: u32 = 0xff52;
const KEYSYM_RIGHT: u32 = 0xff53;
const KEYSYM_DOWN: u32 = 0xff54;
const KEYSYM_PAGE_UP: u32 = 0xff55; // XK_Prior
const KEYSYM_PAGE_DOWN: u32 = 0xff56; // XK_Next
const KEYSYM_END: u32 = 0xff57;
const KEYSYM_DELETE: u32 = 0xffff;
/// GTK mouse button number reported by CandidateClicked.
const MOUSE_BUTTON_LEFT: u32 = 1;

fn text_value(text: &str) -> Value<'_> {
    let attributes = Value::new(("IBusAttrList", Attachments::new(), Vec::<Value<'_>>::new()));
    Value::new(("IBusText", Attachments::new(), text, attributes))
}

fn lookup_value(candidates: &[Candidate], cursor: usize) -> Value<'_> {
    let texts: Vec<_> = candidates.iter().map(|c| text_value(&c.text)).collect();
    Value::new((
        "IBusLookupTable",
        Attachments::new(),
        PAGE_SIZE as u32,
        cursor as u32,
        true,  // cursor visible
        false, // no wrap-around
        0_i32, // IBus.Orientation.HORIZONTAL
        texts,
        Vec::<Value<'_>>::new(), // default labels
    ))
}

/// Own the IBus bus name and serve the engine factory. `register` announces the
/// component on a private bus whose daemon did not start this service.
pub async fn serve(
    address: Option<&str>,
    register: bool,
    decoder: Arc<Decoder>,
    learning: Option<Arc<Learning>>,
    worker: &RankingWorker,
) -> zbus::Result<Connection> {
    let builder = match address {
        Some(address) => zbus::connection::Builder::address(address)?,
        None => zbus::connection::Builder::ibus()?,
    };
    let factory = Factory {
        decoder,
        learning,
        queue: worker.queue.clone(),
        next_id: 0,
    };
    let connection = builder
        .name(BUS_NAME)?
        .serve_at(FACTORY_PATH, factory)?
        .serve_at(FACTORY_PATH, Service)?
        .build()
        .await?;
    if register {
        register_component(&connection).await?;
    }
    Ok(connection)
}

async fn register_component(connection: &Connection) -> zbus::Result<()> {
    let engines = vec![Value::from(
        StructureBuilder::new()
            .add_field("IBusEngineDesc")
            .add_field(Attachments::new())
            .add_field("yanliu")
            .add_field("砚流")
            .add_field("Local Mandarin input")
            .add_field("zh_CN")
            .add_field("GPL-3.0-or-later")
            .add_field("Yanliu contributors")
            .add_field("")
            .add_field("us")
            .add_field(80_u32)
            .add_field("")
            .add_field("砚")
            .add_field("")
            .add_field("")
            .add_field("")
            .add_field("0.1.0")
            .add_field("yanliu")
            .add_field("")
            .build()?,
    )];
    let component = Value::new((
        "IBusComponent",
        Attachments::new(),
        BUS_NAME,
        "Yanliu local Mandarin input",
        "0.1.0",
        "GPL-3.0-or-later",
        "Yanliu contributors",
        "",
        "",
        "yanliu",
        Vec::<Value<'_>>::new(),
        engines,
    ));
    connection
        .call_method(
            Some("org.freedesktop.IBus"),
            "/org/freedesktop/IBus",
            Some("org.freedesktop.IBus"),
            "RegisterComponent",
            &(component,),
        )
        .await?;
    Ok(())
}

struct RankJob {
    emitter: SignalEmitter<'static>,
    generation: Arc<AtomicU64>,
    serial: u64,
    snapshot: SessionSnapshot,
}
impl RankJob {
    fn current(&self) -> bool {
        self.generation.load(Ordering::Acquire) == self.serial
    }
    async fn publish(&self, index: usize) -> zbus::Result<()> {
        let server = self.emitter.connection().object_server();
        // Only Destroy removes an engine, and it made this job stale first.
        let interface = match server.interface::<_, Engine>(self.emitter.path()).await {
            Ok(interface) => interface,
            Err(zbus::Error::InterfaceNotFound) if !self.current() => return Ok(()),
            Err(error) => return Err(error),
        };
        let mut engine = interface.get_mut().await;
        if self.current()
            && engine
                .session
                .apply_promotion(self.snapshot.revision, index)
        {
            engine.publish(&self.emitter).await?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct RankQueue {
    state: Mutex<QueueState>,
    ready: Condvar,
}
#[derive(Default)]
struct QueueState {
    pending: VecDeque<RankJob>,
    stopped: bool,
}
const MAX_PENDING_JOBS: usize = 32;

impl RankQueue {
    /// Keep only each engine's latest job, at that engine's place in line.
    fn submit(&self, job: RankJob) {
        let mut state = self.state.lock().expect("ranking queue poisoned");
        state
            .pending
            .retain(|pending| pending.emitter.path() == job.emitter.path() || pending.current());
        match state
            .pending
            .iter_mut()
            .find(|pending| pending.emitter.path() == job.emitter.path())
        {
            Some(pending) => *pending = job,
            None => {
                if state.pending.len() == MAX_PENDING_JOBS {
                    state.pending.pop_front();
                }
                state.pending.push_back(job);
            }
        }
        self.ready.notify_one();
    }
}

/// The service's single model thread. It ranks one job at a time and publishes
/// the promotion itself; dropping the worker stops it.
pub struct RankingWorker {
    queue: Arc<RankQueue>,
}
impl RankingWorker {
    /// `report` receives the cause when ranking fails; the thread then exits.
    pub fn start(
        mut model: Model,
        report: impl Fn(String) + Send + 'static,
    ) -> std::io::Result<Self> {
        let queue = Arc::new(RankQueue::default());
        let jobs = queue.clone();
        let runtime = tokio::runtime::Handle::current();
        std::thread::Builder::new()
            .name("yanliu-model".into())
            .spawn(move || {
                let run = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    loop {
                        let job = {
                            let state = jobs.state.lock().expect("ranking queue poisoned");
                            let mut state = jobs
                                .ready
                                .wait_while(state, |state| {
                                    state.pending.is_empty() && !state.stopped
                                })
                                .expect("ranking queue poisoned");
                            if state.stopped {
                                return Ok(());
                            }
                            state.pending.pop_front().expect("woken for a pending job")
                        };
                        if !job.current() {
                            continue;
                        }
                        let index = match model.promotion(&job.snapshot) {
                            Ok(Some(index)) => index,
                            Ok(None) => continue,
                            Err(error) => return Err(format!("ranking inference failed: {error}")),
                        };
                        if let Err(error) = runtime.block_on(job.publish(index)) {
                            let path = job.emitter.path();
                            return Err(format!("ranking publisher at {path}: {error}"));
                        }
                    }
                }));
                let cause = match run {
                    Ok(Ok(())) => return,
                    Ok(Err(cause)) => cause,
                    Err(panic) => format!(
                        "ranking worker panicked: {}",
                        panic
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| panic.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string panic payload")
                    ),
                };
                report(cause);
            })?;
        Ok(Self { queue })
    }
}
impl Drop for RankingWorker {
    // Signal instead of joining: an in-flight inference must not block the runtime.
    fn drop(&mut self) {
        self.queue
            .state
            .lock()
            .expect("ranking queue poisoned")
            .stopped = true;
        self.queue.ready.notify_all();
    }
}

struct Factory {
    decoder: Arc<Decoder>,
    learning: Option<Arc<Learning>>,
    queue: Arc<RankQueue>,
    next_id: u64,
}
#[interface(name = "org.freedesktop.IBus.Factory")]
impl Factory {
    async fn create_engine(
        &mut self,
        name: &str,
        #[zbus(object_server)] server: &ObjectServer,
    ) -> zbus::fdo::Result<OwnedObjectPath> {
        if name != "yanliu" {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "Unknown engine: {name}"
            )));
        }
        let path =
            OwnedObjectPath::try_from(format!("/org/freedesktop/IBus/Engine/{}", self.next_id))
                .expect("numbered engine paths are valid");
        self.next_id += 1;
        let session = Session::new(self.decoder.clone(), self.learning.clone());
        server
            .at(&path, Engine::new(session, self.queue.clone()))
            .await?;
        server.at(&path, Service).await?;
        Ok(path)
    }
}

/// `org.freedesktop.IBus.Service`, served beside the factory and each engine.
struct Service;
// Not spawned, so no later call on this connection reaches a destroyed engine.
#[interface(name = "org.freedesktop.IBus.Service", spawn = false)]
impl Service {
    async fn destroy(
        &self,
        #[zbus(object_server)] server: &ObjectServer,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let path = emitter.path();
        match server.interface::<_, Engine>(path).await {
            Ok(engine) => {
                // The lock waits out a publication in progress; later ones are stale.
                engine.get().await.invalidate();
                server.remove::<Engine, _>(path).await?;
            }
            Err(zbus::Error::InterfaceNotFound) => {
                server.remove::<Factory, _>(path).await?;
            }
            Err(error) => return Err(error.into()),
        }
        server.remove::<Service, _>(path).await?;
        Ok(())
    }
}

struct Engine {
    session: Session,
    queue: Arc<RankQueue>,
    /// Advanced by every change that makes queued or running ranking stale.
    generation: Arc<AtomicU64>,
    /// The highlighted candidate.
    cursor: usize,
    sensitive: bool,
    chinese: bool,
}
impl Engine {
    fn new(session: Session, queue: Arc<RankQueue>) -> Self {
        Self {
            session,
            queue,
            generation: Arc::default(),
            cursor: 0,
            sensitive: false,
            chinese: true,
        }
    }
    fn invalidate(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }
    /// Supersede older ranking and queue the current candidates, if any.
    fn rank(&self, emitter: &SignalEmitter<'_>) {
        let serial = self.generation.fetch_add(1, Ordering::Release) + 1;
        if !self.session.candidates().is_empty() {
            self.queue.submit(RankJob {
                emitter: emitter.to_owned(),
                generation: self.generation.clone(),
                serial,
                snapshot: self.session.snapshot(),
            });
        }
    }
    /// The highlight moved by `delta`, kept within the candidates.
    fn moved(&self, delta: isize) -> usize {
        let last = self.session.candidates().len().saturating_sub(1);
        self.cursor.saturating_add_signed(delta).min(last)
    }
    fn page_start(&self) -> usize {
        self.cursor / PAGE_SIZE * PAGE_SIZE
    }
    // Composition exists only in Chinese, non-sensitive mode, so the panel
    // shows whatever the session holds.
    async fn publish(&self, emitter: &SignalEmitter<'_>) -> zbus::Result<()> {
        // A nonempty inline preedit can delete the application's selection
        // before CommitText. Keep composition in the panel until explicit commit.
        Self::update_preedit_text(emitter, text_value(""), 0, false, PREEDIT_MODE_CLEAR).await?;
        let input = self.session.input();
        let candidates = self.session.candidates();
        let auxiliary = if input.is_empty() {
            String::new()
        } else {
            let (before, after) = input.split_at(self.session.cursor());
            let mut text = format!("{before}▏{after}");
            if let Some(candidate) = candidates.get(self.cursor) {
                if candidate.source == CandidateSource::Completion {
                    text.push_str("；补全：包含未输入内容");
                } else {
                    if candidate.consumed < input.len() {
                        text.push_str("；部分转换：保留剩余拼音");
                    }
                    text.push_str(match candidate.source {
                        CandidateSource::Transposition { .. } => "；纠错：相邻字母换位",
                        CandidateSource::NeighborSubstitution { .. } => "；纠错：同排邻键替换",
                        CandidateSource::MissingLetter { .. } => "；纠错：补回漏键",
                        _ => "",
                    });
                }
            }
            text
        };
        Self::update_auxiliary_text(emitter, text_value(&auxiliary), !auxiliary.is_empty()).await?;
        Self::update_lookup_table(
            emitter,
            lookup_value(candidates, self.cursor),
            !candidates.is_empty(),
        )
        .await
    }
    /// Move the highlight; a late ranking result would reorder candidates under it.
    async fn navigate(&mut self, cursor: usize, emitter: &SignalEmitter<'_>) -> zbus::Result<()> {
        self.cursor = cursor;
        self.invalidate();
        self.publish(emitter).await
    }
    async fn clear(&mut self, emitter: &SignalEmitter<'_>) -> zbus::Result<()> {
        self.session.reset();
        self.cursor = 0;
        self.invalidate();
        self.publish(emitter).await
    }
    async fn commit(emitter: &SignalEmitter<'_>, text: Option<String>) -> zbus::Result<()> {
        match text {
            Some(text) => Self::commit_text(emitter, text_value(&text)).await,
            None => Ok(()),
        }
    }
}
// Not spawned: focus and reset callbacks finish before later keys from the same connection.
#[interface(name = "org.freedesktop.IBus.Engine", spawn = false)]
impl Engine {
    async fn process_key_event(
        &mut self,
        keyval: u32,
        _keycode: u32,
        state: u32,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<bool> {
        // Modifier presses neither edit composition nor make ranking stale.
        if self.sensitive
            || state & MODIFIER_RELEASE != 0
            || matches!(keyval, 0xffe1..=0xffe5 | 0xffe7..=0xffec)
        {
            return Ok(false);
        }
        // Engine shortcuts take Ctrl with at most Caps Lock and Num Lock.
        let control_only = state & !(MODIFIER_LOCK | MODIFIER_MOD2) == MODIFIER_CONTROL;
        if control_only && keyval == KEYSYM_SPACE {
            Self::commit(&emitter, self.session.commit_raw()).await?;
            self.chinese = !self.chinese;
            self.clear(&emitter).await?;
            return Ok(true);
        }
        if !self.chinese {
            return Ok(false);
        }
        let count = self.session.candidates().len();
        if control_only && count > 0 {
            let cursor = match keyval {
                KEYSYM_LOWER_H => (self.cursor + count - 1) % count,
                KEYSYM_LOWER_L => (self.cursor + 1) % count,
                KEYSYM_LOWER_K => self.moved(-(PAGE_SIZE as isize)),
                KEYSYM_LOWER_J => self.moved(PAGE_SIZE as isize),
                _ => return Ok(false),
            };
            self.navigate(cursor, &emitter).await?;
            return Ok(true);
        }
        if state
            & (MODIFIER_CONTROL
                | MODIFIER_MOD1
                | MODIFIER_MOD4
                | MODIFIER_SUPER
                | MODIFIER_HYPER
                | MODIFIER_META)
            != 0
        {
            return Ok(false);
        }
        let active = !self.session.input().is_empty();
        let revision = self.session.revision();
        let handled = match keyval {
            KEYSYM_LOWER_A..=KEYSYM_LOWER_Z
            | KEYSYM_UPPER_A..=KEYSYM_UPPER_Z
            | KEYSYM_APOSTROPHE
                if keyval != KEYSYM_APOSTROPHE || active =>
            {
                self.session
                    .insert(&char::from(keyval as u8).to_string())
                    .map_err(|e| zbus::fdo::Error::InvalidArgs(e.to_string()))?;
                true
            }
            KEYSYM_BACKSPACE if active => {
                self.session.backspace();
                true
            }
            KEYSYM_DELETE if active => {
                self.session.delete_forward();
                true
            }
            KEYSYM_LEFT if active => {
                self.session.move_cursor(-1);
                true
            }
            KEYSYM_RIGHT if active => {
                self.session.move_cursor(1);
                true
            }
            KEYSYM_HOME if active => {
                self.session.move_cursor(isize::MIN);
                true
            }
            KEYSYM_END if active => {
                self.session.move_cursor(isize::MAX);
                true
            }
            KEYSYM_UP if active => {
                self.cursor = self.moved(-1);
                true
            }
            KEYSYM_DOWN if active => {
                self.cursor = self.moved(1);
                true
            }
            KEYSYM_PAGE_UP if active => {
                self.cursor = self.moved(-(PAGE_SIZE as isize));
                true
            }
            KEYSYM_PAGE_DOWN if active => {
                self.cursor = self.moved(PAGE_SIZE as isize);
                true
            }
            KEYSYM_ESCAPE if active => {
                self.session.reset();
                true
            }
            KEYSYM_RETURN if active => {
                Self::commit(&emitter, self.session.commit_raw()).await?;
                true
            }
            KEYSYM_SPACE | KEYSYM_DIGIT_0..=KEYSYM_DIGIT_9 if active => {
                let index = match keyval {
                    KEYSYM_SPACE => Some(self.cursor),
                    KEYSYM_DIGIT_0 => None,
                    digit => Some(self.page_start() + (digit - KEYSYM_DIGIT_1) as usize),
                };
                match index.filter(|&index| index < count) {
                    Some(index) => {
                        Self::commit(&emitter, self.session.select(index)).await?;
                        true
                    }
                    // Without that candidate the key is literal: flush the raw
                    // input before the application receives it.
                    None => {
                        Self::commit(&emitter, self.session.commit_raw()).await?;
                        false
                    }
                }
            }
            0x21..=0x7e if (keyval as u8).is_ascii_punctuation() => {
                // Punctuation accepts only the highlighted candidate; a raw
                // remainder is committed as typed, without learning.
                Self::commit(&emitter, self.session.select(self.cursor)).await?;
                Self::commit(&emitter, self.session.commit_raw()).await?;
                let punctuation = match keyval as u8 {
                    b',' => '，',
                    b'.' => '。',
                    b'!' => '！',
                    b'?' => '？',
                    b':' => '：',
                    b';' => '；',
                    b'(' => '（',
                    b')' => '）',
                    b'[' => '【',
                    b']' => '】',
                    b'<' => '《',
                    b'>' => '》',
                    b'\\' => '、',
                    other => char::from(other),
                };
                Self::commit(&emitter, Some(punctuation.to_string())).await?;
                true
            }
            _ => return Ok(false),
        };
        // New candidates start at the first one and are ranked again; anything
        // else is navigation, which keeps the shown order.
        let changed = self.session.revision() != revision;
        if changed {
            self.cursor = 0;
        }
        self.publish(&emitter).await?;
        if changed {
            self.rank(&emitter);
        } else {
            self.invalidate();
        }
        Ok(handled)
    }
    async fn focus_in(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.clear(&emitter).await?)
    }
    async fn focus_in_id(
        &mut self,
        _path: &str,
        _client: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.clear(&emitter).await?)
    }
    async fn focus_out(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.clear(&emitter).await?)
    }
    async fn focus_out_id(
        &mut self,
        _path: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.clear(&emitter).await?)
    }
    async fn reset(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.clear(&emitter).await?)
    }
    fn enable(&self) {}
    async fn disable(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.clear(&emitter).await?)
    }
    fn set_cursor_location(&self, _x: i32, _y: i32, _w: i32, _h: i32) {}
    fn set_capabilities(&self, _caps: u32) {}
    fn property_activate(&self, _name: &str, _state: u32) {}
    fn property_show(&self, _name: &str) {}
    fn property_hide(&self, _name: &str) {}
    async fn set_surrounding_text(
        &mut self,
        text: Value<'_>,
        cursor: u32,
        anchor: u32,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        if self.sensitive || !self.chinese {
            return Ok(());
        }
        // IBusText serializes as (type name, attachments, text, attributes).
        if *text.value_signature() != "(sa{sv}sv)" {
            return Err(zbus::fdo::Error::InvalidArgs(
                "expected IBusText (sa{sv}sv)".into(),
            ));
        }
        let invalid =
            |error: zbus::zvariant::Error| zbus::fdo::Error::InvalidArgs(error.to_string());
        let fields = Structure::try_from(text).map_err(invalid)?;
        let text = <&str>::try_from(&fields.fields()[2]).map_err(invalid)?;
        if cursor.max(anchor) as usize > text.chars().count() {
            return Err(zbus::fdo::Error::InvalidArgs(
                "surrounding cursor exceeds Unicode character count".into(),
            ));
        }
        // Replacement starts at the earlier end of the selection in either direction.
        let context: String = text.chars().take(cursor.min(anchor) as usize).collect();
        self.session.set_context(&context);
        self.rank(&emitter);
        Ok(())
    }
    async fn candidate_clicked(
        &mut self,
        index: u32,
        button: u32,
        _state: u32,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        if button == MOUSE_BUTTON_LEFT && (index as usize) < PAGE_SIZE {
            let index = self.page_start() + index as usize;
            Self::commit(&emitter, self.session.select(index)).await?;
            self.cursor = 0;
            self.publish(&emitter).await?;
            self.rank(&emitter);
        }
        Ok(())
    }
    async fn page_up(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self
            .navigate(self.moved(-(PAGE_SIZE as isize)), &emitter)
            .await?)
    }
    async fn page_down(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self
            .navigate(self.moved(PAGE_SIZE as isize), &emitter)
            .await?)
    }
    async fn cursor_up(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.navigate(self.moved(-1), &emitter).await?)
    }
    async fn cursor_down(
        &mut self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        Ok(self.navigate(self.moved(1), &emitter).await?)
    }
    #[zbus(property)]
    fn focus_id(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn active_surrounding_text(&self) -> bool {
        true
    }
    #[zbus(property)]
    async fn set_content_type(
        &mut self,
        value: (u32, u32),
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (purpose, hints) = value;
        self.sensitive = matches!(purpose, INPUT_PURPOSE_PASSWORD | INPUT_PURPOSE_PIN)
            || hints & SENSITIVE_INPUT_HINTS != 0;
        if self.sensitive {
            self.clear(&emitter).await?;
        }
        Ok(())
    }
    #[zbus(signal)]
    async fn update_preedit_text(
        emitter: &SignalEmitter<'_>,
        text: Value<'_>,
        cursor_pos: u32,
        visible: bool,
        mode: u32,
    ) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn update_auxiliary_text(
        emitter: &SignalEmitter<'_>,
        text: Value<'_>,
        visible: bool,
    ) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn update_lookup_table(
        emitter: &SignalEmitter<'_>,
        table: Value<'_>,
        visible: bool,
    ) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn commit_text(emitter: &SignalEmitter<'_>, text: Value<'_>) -> zbus::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dictionary;
    use std::{
        io::Cursor,
        path::{Path, PathBuf},
        pin::Pin,
        process::{Child, Command, Stdio},
        sync::mpsc,
        time::{Duration, Instant},
    };
    use zbus::{
        Message, Proxy, export::futures_core::Stream, object_server::InterfaceRef,
        proxy::SignalStream, zvariant::Array,
    };

    const FIXTURE: &str =
        "...\n你\tni\t100\n泥\tni\t90\n拟\tni\t80\n好\thao\t100\n你好\tni hao\t100\n";
    const WAIT: Duration = Duration::from_secs(5);

    fn decoder(fixture: &str) -> Arc<Decoder> {
        Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(fixture)).unwrap(),
        ))
    }
    /// `count` homophones of `ni`, the first one weighted highest.
    fn homophones(count: u32) -> Arc<Decoder> {
        let mut fixture = String::from("...\n");
        for i in 0..count {
            let text = char::from_u32(0x4e00 + i).unwrap();
            fixture.push_str(&format!("{text}\tni\t{}\n", 1000 - i));
        }
        decoder(&fixture)
    }
    fn report(cause: String) {
        eprintln!("ranking failure: {cause}");
    }
    fn take_job(queue: &RankQueue) -> RankJob {
        queue.state.lock().unwrap().pending.pop_front().unwrap()
    }

    struct TestBus {
        daemon: Child,
        directory: PathBuf,
        connection: Connection,
        client: Connection,
    }
    impl TestBus {
        async fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "yanliu-ibus-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&directory).unwrap();
            let address = format!("unix:path={}/bus", directory.display());
            let daemon = Command::new("ibus-daemon")
                .args([
                    "--single",
                    "--panel=disable",
                    "--config=disable",
                    "--emoji-extension=disable",
                    "--cache=none",
                    &format!("--address={address}"),
                ])
                .env("IBUS_ADDRESS", &address)
                .env("XDG_STATE_HOME", directory.join("state"))
                .env("XDG_CONFIG_HOME", directory.join("config"))
                .env("XDG_CACHE_HOME", directory.join("cache"))
                .env("XDG_DATA_HOME", directory.join("data"))
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("test requires ibus-daemon");
            let start = Instant::now();
            while !directory.join("bus").exists() {
                assert!(start.elapsed() < WAIT);
                std::thread::sleep(Duration::from_millis(2));
            }
            let connection = zbus::connection::Builder::address(address.as_str())
                .unwrap()
                // Serving at build time starts the object server before any call arrives.
                .serve_at("/test", Service)
                .unwrap()
                .build()
                .await
                .unwrap();
            let client = zbus::connection::Builder::address(address.as_str())
                .unwrap()
                .build()
                .await
                .unwrap();
            Self {
                daemon,
                directory,
                connection,
                client,
            }
        }
        /// Serve an engine at /test/engine/{id}, as CreateEngine would.
        async fn engine(
            &self,
            queue: &Arc<RankQueue>,
            id: u64,
            session: Session,
        ) -> (Proxy<'static>, InterfaceRef<Engine>) {
            let path = format!("/test/engine/{id}");
            let server = self.connection.object_server();
            server
                .at(path.as_str(), Engine::new(session, queue.clone()))
                .await
                .unwrap();
            server.at(path.as_str(), Service).await.unwrap();
            let proxy = Proxy::new_owned(
                self.client.clone(),
                self.connection.unique_name().unwrap().to_owned(),
                path.clone(),
                "org.freedesktop.IBus.Engine",
            )
            .await
            .unwrap();
            (proxy, server.interface(path.as_str()).await.unwrap())
        }
    }
    impl Drop for TestBus {
        fn drop(&mut self) {
            self.daemon.kill().unwrap();
            self.daemon.wait().unwrap();
            std::fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    async fn key(proxy: &Proxy<'_>, keyval: u32) -> bool {
        key_with(proxy, keyval, 0).await
    }
    async fn key_with(proxy: &Proxy<'_>, keyval: u32, state: u32) -> bool {
        tokio::time::timeout(WAIT, proxy.call("ProcessKeyEvent", &(keyval, 0_u32, state)))
            .await
            .expect("ProcessKeyEvent did not return in time")
            .unwrap()
    }
    async fn type_text(proxy: &Proxy<'_>, text: &str) {
        for byte in text.bytes() {
            assert!(key(proxy, byte.into()).await, "{text}");
        }
    }
    async fn surrounding(
        proxy: &Proxy<'_>,
        text: Value<'_>,
        cursor: u32,
        anchor: u32,
    ) -> zbus::Result<()> {
        proxy
            .call("SetSurroundingText", &(text, cursor, anchor))
            .await
    }
    async fn next_signal(signals: &mut SignalStream<'_>) -> Message {
        tokio::time::timeout(
            WAIT,
            std::future::poll_fn(|cx| Pin::new(&mut *signals).poll_next(cx)),
        )
        .await
        .expect("expected D-Bus signal did not arrive in time")
        .unwrap()
    }
    fn member(message: &Message) -> String {
        message.header().member().unwrap().to_string()
    }
    /// The string inside a serialized IBusText.
    fn text(value: &Value<'_>) -> String {
        let text: &Structure = value.downcast_ref().unwrap();
        <&str>::try_from(&text.fields()[2]).unwrap().to_owned()
    }
    /// Visibility of the next lookup table.
    async fn wait_lookup(signals: &mut SignalStream<'_>) -> bool {
        loop {
            let message = next_signal(signals).await;
            if member(&message) == "UpdateLookupTable" {
                let (_, visible): (OwnedValue, bool) = message.body().deserialize().unwrap();
                return visible;
            }
        }
    }
    /// Emit a marker preedit and return every signal that preceded it.
    async fn barrier(signals: &mut SignalStream<'_>, emitter: &SignalEmitter<'_>) -> Vec<Message> {
        Engine::update_preedit_text(emitter, text_value("barrier"), 0, false, 0)
            .await
            .unwrap();
        let mut before = Vec::new();
        loop {
            let message = next_signal(signals).await;
            if member(&message) == "UpdatePreeditText" {
                let (value, ..): (OwnedValue, u32, bool, u32) =
                    message.body().deserialize().unwrap();
                if text(&value) == "barrier" {
                    return before;
                }
            }
            before.push(message);
        }
    }
    /// Panel text and visibility of the next publication, which must clear the
    /// inline preedit and commit nothing.
    async fn auxiliary(signals: &mut SignalStream<'_>) -> (String, bool) {
        let mut auxiliary = None;
        let mut preedit_cleared = false;
        loop {
            let message = next_signal(signals).await;
            match member(&message).as_str() {
                "UpdatePreeditText" => {
                    let (value, cursor, visible, mode): (OwnedValue, u32, bool, u32) =
                        message.body().deserialize().unwrap();
                    assert_eq!(
                        (text(&value).as_str(), cursor, visible, mode),
                        ("", 0, false, PREEDIT_MODE_CLEAR)
                    );
                    preedit_cleared = true;
                }
                "UpdateAuxiliaryText" => {
                    let (value, visible): (OwnedValue, bool) =
                        message.body().deserialize().unwrap();
                    auxiliary = Some((text(&value), visible));
                }
                "UpdateLookupTable" => {
                    assert!(preedit_cleared, "publish omitted the empty preedit");
                    return auxiliary.expect("publish omitted auxiliary text");
                }
                other => panic!("unexpected {other} in a panel update"),
            }
        }
    }
    /// Text committed before the next lookup table.
    async fn committed(signals: &mut SignalStream<'_>) -> String {
        let mut committed = String::new();
        loop {
            let message = next_signal(signals).await;
            match member(&message).as_str() {
                "CommitText" => {
                    let (value,): (OwnedValue,) = message.body().deserialize().unwrap();
                    committed.push_str(&text(&value));
                }
                "UpdateLookupTable" => return committed,
                _ => {}
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn uppercase_pinyin_stays_raw_and_selection_commits_chinese() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let learning = Arc::new(Learning::new(false));
        let session = Session::new(
            decoder("...\n用\tyong\t100\n开发\tkai fa\t100\n"),
            Some(learning.clone()),
        );
        let (proxy, engine) = bus.engine(&queue, 1, session).await;
        let mut signals = proxy.receive_all_signals().await.unwrap();
        type_text(&proxy, "YongKaiFa").await;
        let index = {
            let engine = engine.get().await;
            assert_eq!(engine.session.input(), "YongKaiFa");
            engine
                .session
                .candidates()
                .iter()
                .position(|c| c.text == "用开发" && c.consumed == 9)
                .unwrap()
        };
        for _ in 0..index {
            assert!(key(&proxy, KEYSYM_DOWN).await);
        }
        barrier(&mut signals, engine.signal_emitter()).await;
        assert_eq!(learning.revision(), 0);
        assert!(key(&proxy, KEYSYM_SPACE).await);
        assert_eq!(committed(&mut signals).await, "用开发");
        assert!(engine.get().await.session.input().is_empty());
        assert_eq!(learning.revision(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn digits_space_and_punctuation_select_or_flush_before_the_reply() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        // Twelve exact homophones and their partial completions: the last
        // page holds fewer than four candidates.
        let homophones = homophones(12);
        for (id, (input, last_page, keyval, selects, expected)) in [
            ("ni", false, b'1', true, "一"),
            ("ni", false, b' ', true, "一"),
            ("ni", false, b',', true, "一，"),
            ("ni", false, b'0', false, "ni"),
            ("ni", true, b'4', false, "ni"),
            ("ni", true, b'9', false, "ni"),
            ("zzz", false, b' ', false, "zzz"),
            ("zzz", false, b'1', false, "zzz"),
        ]
        .into_iter()
        .enumerate()
        {
            let learning = Arc::new(Learning::new(false));
            let session = Session::new(homophones.clone(), Some(learning.clone()));
            let (proxy, engine) = bus.engine(&queue, id as u64, session).await;
            let mut signals = proxy.receive_all_signals().await.unwrap();
            type_text(&proxy, input).await;
            if last_page {
                assert!(key(&proxy, KEYSYM_PAGE_DOWN).await);
                assert!(key(&proxy, KEYSYM_PAGE_DOWN).await);
                let engine = engine.get().await;
                let len = engine.session.candidates().len();
                assert!(engine.page_start() + PAGE_SIZE >= len && len - engine.page_start() < 4);
            }
            barrier(&mut signals, engine.signal_emitter()).await;
            let reply = proxy
                .call_method("ProcessKeyEvent", &(u32::from(keyval), 0_u32, 0_u32))
                .await
                .unwrap();
            assert_eq!(reply.body().deserialize::<bool>().unwrap(), selects);
            // An unhandled key reaches the application after this reply, so
            // every commit and panel update must precede it on the wire.
            let mut commits = String::new();
            loop {
                let message = next_signal(&mut signals).await;
                assert!(
                    message.primary_header().serial_num() < reply.primary_header().serial_num()
                );
                match member(&message).as_str() {
                    "CommitText" => {
                        let (value,): (OwnedValue,) = message.body().deserialize().unwrap();
                        commits.push_str(&text(&value));
                    }
                    "UpdateLookupTable" => {
                        let (_, visible): (OwnedValue, bool) =
                            message.body().deserialize().unwrap();
                        assert!(!visible);
                        break;
                    }
                    _ => {}
                }
            }
            assert_eq!(commits, expected, "{input} {}", char::from(keyval));
            assert!(engine.get().await.session.input().is_empty());
            assert_eq!(learning.revision(), u64::from(selects));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panel_notes_the_highlighted_candidate_kind_and_click_commits_it() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        type Kind = fn(&CandidateSource) -> bool;
        let ordinary: Kind = |source| {
            matches!(
                source,
                CandidateSource::ExactDictionary | CandidateSource::Composed
            )
        };
        let completion: Kind = |source| *source == CandidateSource::Completion;
        let transposition: Kind = |source| matches!(source, CandidateSource::Transposition { .. });
        let neighbor: Kind =
            |source| matches!(source, CandidateSource::NeighborSubstitution { .. });
        let missing: Kind = |source| matches!(source, CandidateSource::MissingLetter { .. });
        let partial = "；部分转换：保留剩余拼音";
        for (id, (input, target, kind, note)) in [
            ("ni", "你", ordinary, String::new()),
            ("nih", "你好", completion, "；补全：包含未输入内容".into()),
            ("niha", "你好", completion, "；补全：包含未输入内容".into()),
            ("nix", "你", ordinary, partial.into()),
            ("in", "你", transposition, "；纠错：相邻字母换位".into()),
            (
                "inx",
                "你",
                transposition,
                format!("{partial}；纠错：相邻字母换位"),
            ),
            ("no", "你", neighbor, "；纠错：同排邻键替换".into()),
            (
                "nox",
                "你",
                neighbor,
                format!("{partial}；纠错：同排邻键替换"),
            ),
            ("ihao", "你好", missing, "；纠错：补回漏键".into()),
            (
                "ihaoq",
                "你好",
                missing,
                format!("{partial}；纠错：补回漏键"),
            ),
            ("niha'", "你好", missing, "；纠错：补回漏键".into()),
        ]
        .into_iter()
        .enumerate()
        {
            let (proxy, engine) = bus
                .engine(&queue, id as u64, Session::new(decoder(FIXTURE), None))
                .await;
            let mut signals = proxy.receive_all_signals().await.unwrap();
            type_text(&proxy, input).await;
            let index = engine
                .get()
                .await
                .session
                .candidates()
                .iter()
                .position(|c| c.text == target && kind(&c.source))
                .expect("fixture recalls the target candidate");
            assert!(index < PAGE_SIZE);
            barrier(&mut signals, engine.signal_emitter()).await;
            assert!(key(&proxy, KEYSYM_UP).await);
            let mut shown = auxiliary(&mut signals).await;
            for _ in 0..index {
                assert!(key(&proxy, KEYSYM_DOWN).await);
                shown = auxiliary(&mut signals).await;
            }
            assert_eq!(shown, (format!("{input}▏{note}"), true));
            proxy
                .call::<_, _, ()>(
                    "CandidateClicked",
                    &(index as u32, MOUSE_BUTTON_LEFT, 0_u32),
                )
                .await
                .unwrap();
            assert_eq!(committed(&mut signals).await, target, "{input}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panel_shows_raw_input_and_only_the_current_ranking_reorders() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let session = Session::new(decoder("...\n你\tni\t100\n泥\tni\t90\n拟\tni\t80\n"), None);
        let (proxy, engine) = bus.engine(&queue, 1, session).await;
        let mut signals = proxy.receive_all_signals().await.unwrap();
        assert!(key(&proxy, b'n'.into()).await);
        let stale = take_job(&queue);
        assert_eq!(stale.snapshot.input, "n");
        assert!(auxiliary(&mut signals).await.0.starts_with("n▏"));
        assert!(key(&proxy, b'i'.into()).await);
        assert_eq!(auxiliary(&mut signals).await, ("ni▏".into(), true));
        let mut expected = engine.get().await.session.candidates().to_vec();
        let last = expected.last().unwrap();
        assert_eq!(
            (&last.source, last.consumed),
            (&CandidateSource::Completion, 1)
        );
        // The stale "n" job publishes nothing; the "ni" job promotes a full
        // conversion and leaves the partial completion in place.
        stale.publish(1).await.unwrap();
        let current = take_job(&queue);
        assert_eq!(current.snapshot.input, "ni");
        current.publish(1).await.unwrap();
        assert_eq!(auxiliary(&mut signals).await, ("ni▏".into(), true));
        expected[..=1].rotate_right(1);
        assert_eq!(engine.get().await.session.candidates(), expected);
        for (keyval, shown) in [
            (KEYSYM_LEFT, "n▏i"),
            (KEYSYM_LEFT, "▏ni"),
            (KEYSYM_LEFT, "▏ni"),
            (KEYSYM_RIGHT, "n▏i"),
            (KEYSYM_END, "ni▏"),
            (KEYSYM_HOME, "▏ni"),
        ] {
            assert!(key(&proxy, keyval).await);
            assert_eq!(auxiliary(&mut signals).await, (shown.into(), true));
        }
        assert_eq!(engine.get().await.session.input(), "ni");
        assert!(key(&proxy, KEYSYM_ESCAPE).await);
        assert_eq!(auxiliary(&mut signals).await, (String::new(), false));
        assert!(!key(&proxy, KEYSYM_HOME).await);
        // Raw input stays visible when nothing can be recalled.
        for shown in ["z▏", "zz▏"] {
            assert!(key(&proxy, b'z'.into()).await);
            assert_eq!(auxiliary(&mut signals).await, (shown.into(), true));
        }
        assert!(engine.get().await.session.candidates().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shortcuts_and_modifier_presses_pass_through_untouched() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let learning = Arc::new(Learning::new(false));
        let session = Session::new(decoder(FIXTURE), Some(learning.clone()));
        let (proxy, engine) = bus.engine(&queue, 1, session).await;
        let mut signals = proxy.receive_all_signals().await.unwrap();
        let observe = |engine: &Engine| {
            (
                engine.session.input().to_owned(),
                engine.session.candidates().to_vec(),
                engine.session.revision(),
                engine.generation.load(Ordering::Acquire),
                engine.cursor,
            )
        };
        // Shift, Control, Alt and Super presses.
        let presses = [0xffe1, 0xffe3, 0xffe9, 0xffeb].map(|keyval| (keyval, 0));
        let states = [
            MODIFIER_RELEASE,
            MODIFIER_CONTROL,
            MODIFIER_CONTROL | 1,
            MODIFIER_CONTROL | MODIFIER_MOD1,
            MODIFIER_MOD1,
            MODIFIER_MOD4,
            MODIFIER_SUPER,
            MODIFIER_HYPER,
            MODIFIER_META,
        ];
        let keys = b"epcEx/,!019"
            .iter()
            .map(|&byte| u32::from(byte))
            .chain([KEYSYM_HOME, KEYSYM_END]);
        let shortcuts = states
            .into_iter()
            .flat_map(|state| keys.clone().map(move |keyval| (keyval, state)));
        let attempts: Vec<_> = presses.into_iter().chain(shortcuts).collect();
        for active in [false, true] {
            if active {
                type_text(&proxy, "ni").await;
                assert!(key(&proxy, KEYSYM_DOWN).await);
            }
            barrier(&mut signals, engine.signal_emitter()).await;
            let before = observe(&*engine.get().await);
            for &(keyval, state) in &attempts {
                assert!(
                    !key_with(&proxy, keyval, state).await,
                    "active={active}, key={keyval:#x}, state={state:#x}"
                );
                assert_eq!(observe(&*engine.get().await), before);
            }
            assert_eq!(learning.revision(), 0);
            let emitted = barrier(&mut signals, engine.signal_emitter()).await;
            assert!(emitted.is_empty(), "shortcuts emitted signals");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_ranking_does_not_block_keys_and_navigation_keeps_the_order() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let (proxy, engine) = bus
            .engine(&queue, 1, Session::new(decoder(FIXTURE), None))
            .await;
        let mut signals = proxy.receive_all_signals().await.unwrap();
        // The key returns and publishes before its ranking is processed.
        assert!(key(&proxy, b'n'.into()).await);
        let pending = take_job(&queue);
        assert!(wait_lookup(&mut signals).await);
        let shown = engine.get().await.session.candidates().to_vec();
        assert!(key(&proxy, KEYSYM_DOWN).await);
        assert!(wait_lookup(&mut signals).await);
        pending.publish(1).await.unwrap();
        {
            let engine = engine.get().await;
            assert_eq!(engine.cursor, 1);
            assert_eq!(engine.session.candidates(), shown);
        }
        let late = barrier(&mut signals, engine.signal_emitter()).await;
        assert!(late.is_empty(), "late ranking reordered navigation");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn later_pages_select_by_digit_or_click_after_cancelling_late_ranking() {
        let bus = TestBus::new().await;
        let homophones = homophones(40);
        let lowest = char::from_u32(0x4e00 + 39).unwrap().to_string();
        for (id, (click, last)) in [(false, false), (true, false), (false, true), (true, true)]
            .into_iter()
            .enumerate()
        {
            let queue = Arc::default();
            let learning = Arc::new(Learning::new(false));
            let id = 10 * (id as u64 + 1);
            let session = Session::new(homophones.clone(), Some(learning.clone()));
            let (proxy, engine) = bus.engine(&queue, id, session).await;
            let mut signals = proxy.receive_all_signals().await.unwrap();
            assert!(key(&proxy, b'n'.into()).await);
            let first = take_job(&queue);
            assert!(wait_lookup(&mut signals).await);
            assert!(key(&proxy, b'i'.into()).await);
            assert!(wait_lookup(&mut signals).await);
            let shown = engine.get().await.session.candidates().to_vec();
            // Every exact homophone stays reachable beyond the ordinary budget.
            for i in 0..40 {
                let text = char::from_u32(0x4e00 + i).unwrap().to_string();
                assert!(shown.iter().any(|c| c.text == text
                    && c.consumed == 2
                    && c.source == CandidateSource::ExactDictionary));
            }
            let selected = if last {
                shown.len() - 1
            } else {
                shown.iter().position(|c| c.text == lowest).unwrap()
            };
            let final_page = selected / PAGE_SIZE;
            for page in 1..=final_page {
                proxy.call::<_, _, ()>("PageDown", &()).await.unwrap();
                let message = loop {
                    let message = next_signal(&mut signals).await;
                    if member(&message) == "UpdateLookupTable" {
                        break message;
                    }
                };
                let (table, visible): (OwnedValue, bool) = message.body().deserialize().unwrap();
                assert!(visible);
                let table = Structure::try_from(table).unwrap();
                let fields = table.fields();
                assert_eq!(u32::try_from(&fields[2]).unwrap(), PAGE_SIZE as u32);
                assert_eq!(
                    u32::try_from(&fields[3]).unwrap(),
                    (page * PAGE_SIZE) as u32
                );
                let rows: Vec<_> = <&Array>::try_from(&fields[7])
                    .unwrap()
                    .inner()
                    .iter()
                    .map(text)
                    .collect();
                assert!(rows.iter().eq(shown.iter().map(|c| &c.text)));
            }
            // The in-flight "n" job and the queued "ni" job are both stale.
            first.publish(1).await.unwrap();
            take_job(&queue).publish(1).await.unwrap();
            {
                let engine = engine.get().await;
                assert_eq!(engine.cursor, final_page * PAGE_SIZE);
                assert_eq!(engine.session.candidates(), shown);
            }
            let late = barrier(&mut signals, engine.signal_emitter()).await;
            assert!(late.is_empty(), "late ranking reordered navigation");
            let slot = (selected % PAGE_SIZE) as u32;
            if click {
                proxy
                    .call::<_, _, ()>("CandidateClicked", &(slot, MOUSE_BUTTON_LEFT, 0_u32))
                    .await
                    .unwrap();
            } else {
                assert!(key(&proxy, u32::from(b'1') + slot).await);
            }
            let candidate = &shown[selected];
            assert_eq!(committed(&mut signals).await, candidate.text);
            assert_eq!(
                engine.get().await.session.input(),
                &"ni"[candidate.consumed..]
            );
            if candidate.source == CandidateSource::Completion {
                assert_eq!(learning.revision(), 0);
            } else {
                let raw = &"ni"[..candidate.consumed];
                assert_eq!(learning.counts(raw, std::slice::from_ref(candidate)), [1]);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn surrounding_text_sets_context_before_the_selection_and_rejects_bad_input() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let (proxy, engine) = bus
            .engine(&queue, 1, Session::new(decoder(FIXTURE), None))
            .await;
        // IBus offsets count Unicode scalar values, including combining marks.
        let text = "前中😀e\u{301}选😺a\u{301}后";
        let long = format!("{}中😀e\u{301}选后", "前".repeat(66));
        let long_context = format!("{}中😀e\u{301}", "前".repeat(60));
        for (text, cursor, anchor, context) in [
            (text, 9, 5, "前中😀e\u{301}"),
            (text, 5, 9, "前中😀e\u{301}"),
            (text, 9, 9, "前中😀e\u{301}选😺a\u{301}"),
            (text, 4, 4, "前中😀e"),
            (text, 9, 0, ""),
            (text, 0, 9, ""),
            (text, 0, 0, ""),
            (long.as_str(), 71, 70, long_context.as_str()),
            (long.as_str(), 70, 71, long_context.as_str()),
        ] {
            surrounding(&proxy, text_value(text), cursor, anchor)
                .await
                .unwrap();
            assert_eq!(
                engine.get().await.session.context(),
                context,
                "cursor={cursor}, anchor={anchor}"
            );
        }
        surrounding(&proxy, text_value("原上下文"), 4, 4)
            .await
            .unwrap();
        let short = "中😀e\u{301}";
        for (value, cursor, anchor) in [
            (text_value(short), 5, 4),
            (text_value(short), 4, 5),
            (text_value(short), u32::MAX, 0),
            (text_value(short), 0, u32::MAX),
            (Value::from("plain string"), 0, 0),
        ] {
            let error = surrounding(&proxy, value, cursor, anchor)
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                zbus::Error::MethodError(name, _, _)
                    if name.as_str() == "org.freedesktop.DBus.Error.InvalidArgs"
            ));
            assert_eq!(engine.get().await.session.context(), "原上下文");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn english_mode_passes_keys_through_and_cancels_late_ranking() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let learning = Arc::new(Learning::new(false));
        let session = Session::new(decoder(FIXTURE), Some(learning.clone()));
        let (proxy, engine) = bus.engine(&queue, 1, session).await;
        let mut signals = proxy.receive_all_signals().await.unwrap();
        assert!(key(&proxy, b'n'.into()).await);
        let pending = take_job(&queue);
        assert!(wait_lookup(&mut signals).await);
        assert!(key_with(&proxy, KEYSYM_SPACE, MODIFIER_CONTROL).await);
        assert_eq!(committed(&mut signals).await, "n");
        surrounding(&proxy, text_value("private"), 7, 7)
            .await
            .unwrap();
        for byte in b"English https://Example.com/a?x=10 curl -I URL\t\n" {
            assert!(!key(&proxy, u32::from(*byte)).await);
        }
        {
            let engine = engine.get().await;
            assert!(!engine.chinese);
            assert!(engine.session.input().is_empty());
            assert!(engine.session.context().is_empty());
        }
        assert_eq!(learning.revision(), 0);
        pending.publish(1).await.unwrap();
        let late = barrier(&mut signals, engine.signal_emitter()).await;
        assert!(late.is_empty(), "late ranking restored English composition");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_keeps_each_engines_latest_job_in_arrival_order() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let mut engines = Vec::new();
        for id in 1..=3 {
            let session = Session::new(decoder(FIXTURE), None);
            engines.push(bus.engine(&queue, id, session).await.0);
        }
        let started = |path: &str, input: &str| (path.to_owned(), input.to_owned());
        assert!(key(&engines[0], b'n'.into()).await);
        let first = take_job(&queue);
        assert_eq!(first.snapshot.input, "n");
        for (engine, byte) in [(1, b'n'), (2, b'n'), (1, b'i'), (0, b'i')] {
            assert!(key(&engines[engine], byte.into()).await);
        }
        let expected = [
            started("/test/engine/2", "ni"),
            started("/test/engine/3", "n"),
            started("/test/engine/1", "ni"),
        ];
        let pending: Vec<_> = queue
            .state
            .lock()
            .unwrap()
            .pending
            .iter()
            .map(|job| started(job.emitter.path().as_str(), &job.snapshot.input))
            .collect();
        assert_eq!(pending, expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ranking_queue_stays_bounded_across_engine_lifetimes() {
        let bus = TestBus::new().await;
        let queue = Arc::default();
        let (first, _) = bus
            .engine(&queue, 0, Session::new(decoder(FIXTURE), None))
            .await;
        assert!(key(&first, b'n'.into()).await);
        let _first = take_job(&queue);
        for id in 1..=(MAX_PENDING_JOBS as u64 * 2 + 11) {
            let (proxy, _) = bus
                .engine(&queue, id, Session::new(decoder(FIXTURE), None))
                .await;
            assert!(key(&proxy, b'n'.into()).await);
            if id % 2 == 0 {
                let service = Proxy::new(
                    &bus.client,
                    bus.connection.unique_name().unwrap(),
                    proxy.path(),
                    "org.freedesktop.IBus.Service",
                )
                .await
                .unwrap();
                service.call::<_, _, ()>("Destroy", &()).await.unwrap();
            }
            assert!(queue.state.lock().unwrap().pending.len() <= MAX_PENDING_JOBS);
        }
        let state = queue.state.lock().unwrap();
        assert_eq!(state.pending.len(), MAX_PENDING_JOBS);
        assert!(state.pending.iter().all(RankJob::current));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lifecycle_events_cancel_late_ranking() {
        let bus = TestBus::new().await;
        for (id, action) in [
            "Reset", "FocusIn", "FocusOut", "Disable", "Password", "Destroy",
        ]
        .into_iter()
        .enumerate()
        {
            let queue = Arc::default();
            let id = 10 * (id as u64 + 1);
            let (proxy, engine) = bus
                .engine(&queue, id, Session::new(decoder(FIXTURE), None))
                .await;
            let mut signals = proxy.receive_all_signals().await.unwrap();
            assert!(key(&proxy, b'n'.into()).await);
            let pending = take_job(&queue);
            assert!(wait_lookup(&mut signals).await);
            match action {
                "Password" => {
                    proxy
                        .set_property("ContentType", (INPUT_PURPOSE_PASSWORD, 0_u32))
                        .await
                        .unwrap();
                    assert!(!wait_lookup(&mut signals).await);
                    assert!(!key(&proxy, b'n'.into()).await);
                }
                "Destroy" => {
                    let service = Proxy::new(
                        &bus.client,
                        bus.connection.unique_name().unwrap(),
                        proxy.path(),
                        "org.freedesktop.IBus.Service",
                    )
                    .await
                    .unwrap();
                    service.call::<_, _, ()>("Destroy", &()).await.unwrap();
                    let call =
                        proxy.call::<_, _, bool>("ProcessKeyEvent", &(0x6e_u32, 0_u32, 0_u32));
                    assert!(call.await.is_err(), "destroyed engine still answers");
                }
                _ => {
                    proxy.call::<_, _, ()>(action, &()).await.unwrap();
                    assert!(!wait_lookup(&mut signals).await);
                }
            }
            pending.publish(1).await.unwrap();
            let late = barrier(&mut signals, engine.signal_emitter()).await;
            assert!(late.is_empty(), "late ranking after {action}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publisher_failure_reports_the_original_cause_and_drop_stops_the_queue() {
        let bus = TestBus::new().await;
        let weights =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors");
        let (sender, failures) = mpsc::channel();
        let worker =
            RankingWorker::start(Model::load(&weights, "cpu", None).unwrap(), move |cause| {
                sender.send(cause).unwrap()
            })
            .unwrap();
        let mut session = Session::new(
            decoder("...\n现\txian\t1000\n西安\txi an\t1\n西\txi\t100\n安\tan\t90\n"),
            None,
        );
        session.set_context("我去");
        session.insert("xian").unwrap();
        worker.queue.submit(RankJob {
            emitter: SignalEmitter::new(&bus.connection, "/missing/engine")
                .unwrap()
                .to_owned(),
            generation: Arc::default(),
            serial: 0,
            snapshot: session.snapshot(),
        });
        let cause = failures.recv_timeout(WAIT).unwrap();
        assert!(
            cause.starts_with("ranking publisher at /missing/engine: "),
            "{cause}"
        );
        assert!(
            cause.contains(&zbus::Error::InterfaceNotFound.to_string()),
            "{cause}"
        );
        let queue = worker.queue.clone();
        drop(worker);
        assert!(queue.state.lock().unwrap().stopped);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_promotion_reaches_the_panel_with_candidate_identity() {
        let bus = TestBus::new().await;
        let weights =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors");
        let learning = Arc::new(Learning::new(false));
        let worker =
            RankingWorker::start(Model::load(&weights, "cpu", None).unwrap(), report).unwrap();
        let decoder = decoder("...\n现\txian\t1000\n西安\txi an\t1\n西\txi\t100\n安\tan\t90\n");
        let mut expected = Session::new(decoder.clone(), Some(learning.clone()));
        expected.set_context("我去");
        expected.insert("xian").unwrap();
        let index = Model::load(&weights, "cpu", None)
            .unwrap()
            .promotion(&expected.snapshot())
            .unwrap()
            .expect("fixture promotes a full conversion");
        assert!(expected.apply_promotion(expected.revision(), index));
        assert_eq!(expected.candidates()[0].text, "西安");
        let mut session = Session::new(decoder, Some(learning));
        session.set_context("我去");
        session.insert("xia").unwrap();
        let (proxy, engine) = bus.engine(&worker.queue, 1, session).await;
        let mut signals = proxy.receive_all_signals().await.unwrap();
        assert!(key(&proxy, b'n'.into()).await);
        assert!(wait_lookup(&mut signals).await); // Static candidates.
        assert!(wait_lookup(&mut signals).await); // Ranked candidates.
        let engine = engine.get().await;
        assert_eq!(engine.session.input(), "xian");
        assert_eq!(engine.session.context(), "我去");
        let identity = |candidates: &[Candidate]| {
            candidates
                .iter()
                .map(|c| {
                    (
                        c.text.clone(),
                        c.consumed,
                        c.source.clone(),
                        c.score.to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            identity(engine.session.candidates()),
            identity(expected.candidates())
        );
    }
}

//! What a view read while it was drawn, and whether any of it changed since.
//!
//! While a view is drawn, a recording is open: every entity accessed, every
//! global read, every piece of versioned state read (a scroll handle, a list
//! state, a [`DrawDependency`]), the pointer and modifier keys, and any time a
//! view said it would look different at, are logged. A view drawn again from
//! the last frame is checked against what it read then.
//!
//! None of this is kept while view retention is off, beyond a branch per
//! entity access, update and notification.

use crate::{App, EntityId, EntityMap};
use collections::{FxHashMap, FxHashSet, TypeIdHashMap};
use std::{
    any::TypeId,
    cell::{Cell, RefCell},
    ops::Range,
    rc::Rc,
    time::Instant,
};

/// The app's half of recording what views read: when each global last
/// changed, and the globals read while a recording is open.
#[derive(Default)]
pub(crate) struct AppDependencies {
    /// Counts the changes to globals made while no recording is open, each
    /// stamped into `global_changed_at`.
    global_generation: u64,
    global_changed_at: TypeIdHashMap<u64>,
    /// When each global was last written while a recording was open, in the
    /// entities' write generation: written while a window draws.
    global_written_at: TypeIdHashMap<u64>,
    /// Every global read while a recording is open, with the write
    /// generation it was read at, pointer and modifier reads included (see
    /// [`ambient`]).
    global_read_log: Rc<RefCell<Vec<(TypeId, u64)>>>,
    /// Where the recordings nested in the ones open, and the dependencies
    /// replayed inside them, put their reads in the logs, in order: what a
    /// view read itself leaves them out.
    nested: Vec<Stretch>,
}

/// Where a recording, or a replay, put its reads in the logs.
#[derive(Clone)]
struct Stretch {
    entities: Range<usize>,
    globals: Range<usize>,
    states: Range<usize>,
    deadlines: Range<usize>,
}

/// The entries of `log[range]` outside the `nested` stretches of it that
/// `of` picks, which follow one another.
fn outside<T: Clone>(
    log: &[T],
    range: Range<usize>,
    nested: &[Stretch],
    of: impl Fn(&Stretch) -> Range<usize>,
) -> Vec<T> {
    let mut own = Vec::new();
    let mut from = range.start;
    for stretch in nested {
        let stretch = of(stretch);
        if stretch.start > from {
            own.extend_from_slice(&log[from..stretch.start.min(range.end)]);
        }
        from = from.max(stretch.end);
    }
    if from < range.end {
        own.extend_from_slice(&log[from..range.end]);
    }
    own
}

impl AppDependencies {
    pub(crate) fn global_changed(&mut self, global_type: TypeId) {
        // Stamped on every change, not only the first one an effect is queued
        // for: a view drawn in between has seen only the first.
        self.global_generation += 1;
        self.global_changed_at
            .insert(global_type, self.global_generation);
    }
}

/// Stands for whether a global of type `G` is set, which a view that only
/// asked [`App::has_global`] depends on, rather than on the global itself.
struct GlobalPresence<G>(std::marker::PhantomData<G>);

/// Records that whether a global of type `G` is set was read.
#[inline]
pub(crate) fn note_global_presence_read<G: 'static>(cx: &App) {
    note_global_read(cx, TypeId::of::<GlobalPresence<G>>());
}

/// Records that the global of type `global` was read.
#[inline]
pub(crate) fn note_global_read(cx: &App, global: TypeId) {
    let log = &cx.entities.access_log;
    if log.recording() {
        cx.dependencies
            .global_read_log
            .borrow_mut()
            .push((global, log.write_generation));
    }
}

/// Stamps a change to the global of type `global_type`, as it is written.
///
/// Written while a window draws, it is a write in the entities' write
/// generation: a view that read the global before the write read what it
/// held before, and a view that writes it and reads it back does not
/// depend on itself.
#[inline]
pub(crate) fn global_changed(cx: &mut App, global_type: TypeId) {
    let log = &mut cx.entities.access_log;
    if !log.enabled {
        return;
    }
    let drawing = log.recording();
    super::culprits::note_global_change(global_type, drawing);
    if drawing {
        log.write_generation += 1;
        log.pinning_writes += 1;
        cx.dependencies
            .global_written_at
            .insert(global_type, log.write_generation);
    } else {
        cx.dependencies.global_changed(global_type);
    }
}

/// Records that the global of type `global_type` is written and read, as
/// `global_mut` or `update_global` do: after the write, so that the view
/// writing it does not depend on its own write.
#[inline]
pub(crate) fn global_written_and_read(cx: &mut App, global_type: TypeId) {
    global_changed(cx, global_type);
    note_global_read(cx, global_type);
}

/// Stamps a change to whether a global of type `G` is set, when it is about
/// to be set where it was not.
pub(crate) fn note_global_inserted<G: 'static>(cx: &mut App) {
    if cx.entities.access_log.enabled && !cx.globals_by_type.contains_key(&TypeId::of::<G>()) {
        global_changed(cx, TypeId::of::<GlobalPresence<G>>());
    }
}

/// Stamps a change to whether a global of type `G` is set, as it is about to
/// be removed.
pub(crate) fn note_global_removed<G: 'static>(cx: &mut App) {
    global_changed(cx, TypeId::of::<GlobalPresence<G>>());
}

/// Parts of a window's state a view can read without reading an entity or a
/// global, each recorded as a global of its own type and marked changed when
/// input changes it.
pub(crate) mod ambient {
    /// Where the pointer is: [`crate::Window::mouse_position`].
    pub(crate) struct Pointer;
    /// The modifier keys and caps lock: [`crate::Window::modifiers`] and
    /// [`crate::Window::capslock`].
    pub(crate) struct Keys;
    /// The window's appearance: [`crate::Window::appearance`].
    pub(crate) struct Appearance;
    /// Which actions are available and bound where the window is focused:
    /// [`crate::Window::is_action_available`],
    /// [`crate::Window::available_actions`], the binding lookups,
    /// [`crate::Window::context_stack`] and focus containment, which answer
    /// from the frame last drawn and the keymap.
    pub(crate) struct Actions;
}

/// Stamps a change to one of the window's [`ambient`] states.
pub(crate) fn ambient_changed<T: 'static>(cx: &mut App) {
    if cx.entities.access_log.enabled {
        cx.dependencies.global_changed(TypeId::of::<T>());
    }
}

/// A window's handle on the app's recording, so that reading the window's own
/// state while a view is drawn is recorded as a global read would be.
#[derive(Clone)]
pub(crate) struct AmbientReads {
    globals: Rc<RefCell<Vec<(TypeId, u64)>>>,
    recordings: Rc<Cell<usize>>,
    untracked: Rc<RefCell<Vec<EntityId>>>,
}

impl AmbientReads {
    #[inline]
    pub(crate) fn note<T: 'static>(&self) {
        // Input changes these, never a view while it is drawn, so the
        // write generation they are read at does not matter.
        if self.recordings.get() > 0 {
            self.globals.borrow_mut().push((TypeId::of::<T>(), 0));
        }
    }

    /// The entities whose reads are left out of the recordings now.
    pub(crate) fn untracked(&self) -> smallvec::SmallVec<[EntityId; 2]> {
        self.untracked.borrow().iter().copied().collect()
    }
}

/// The pointer and modifier keys before a window handled an input event, to
/// tell afterwards which of them it changed.
pub(crate) struct AmbientInput {
    position: crate::Point<crate::Pixels>,
    modifiers: crate::Modifiers,
    capslock: crate::Capslock,
}

impl AmbientInput {
    pub(crate) fn of(window: &crate::Window) -> Self {
        AmbientInput {
            position: window.mouse_position,
            modifiers: window.modifiers,
            capslock: window.capslock,
        }
    }

    pub(crate) fn stamp_changes(self, window: &crate::Window, cx: &mut App) {
        if !cx.entities.access_log.enabled {
            return;
        }
        if window.mouse_position != self.position {
            cx.dependencies
                .global_changed(TypeId::of::<ambient::Pointer>());
        }
        if window.modifiers != self.modifiers || window.capslock != self.capslock {
            cx.dependencies.global_changed(TypeId::of::<ambient::Keys>());
        }
    }
}

thread_local! {
    /// Versioned state read while a recording is open, with the version it
    /// was at. Kept per thread rather than per app, since state such as a
    /// scroll handle is read without an app at hand; recordings only happen
    /// while a window draws, one window at a time.
    static STATE_READS: RefCell<Vec<(StateVersion, u64)>> = const { RefCell::new(Vec::new()) };
    static STATE_RECORDINGS: Cell<usize> = const { Cell::new(0) };

    /// The times views said they would look different at, while a recording
    /// is open. See [`crate::Window::rebuild_at`].
    static DEADLINES: RefCell<Vec<Instant>> = const { RefCell::new(Vec::new()) };
}

/// Records that the state `version` belongs to was read as it is now.
#[inline]
pub(crate) fn note_state_read(version: &StateVersion) {
    if STATE_RECORDINGS.with(Cell::get) > 0 {
        STATE_READS.with_borrow_mut(|reads| reads.push((version.clone(), version.get())));
    }
}

/// Records that what is being drawn will look different at `deadline`.
pub(crate) fn note_deadline(deadline: Instant) {
    if STATE_RECORDINGS.with(Cell::get) > 0 {
        DEADLINES.with_borrow_mut(|deadlines| deadlines.push(deadline));
    }
}

/// A counter that state shared outside of entities bumps whenever it changes,
/// so that a view that read it is drawn again, as it is for an entity that was
/// changed.
#[derive(Clone, Default, Debug)]
pub(crate) struct StateVersion(Rc<Cell<u64>>);

impl StateVersion {
    pub(crate) fn get(&self) -> u64 {
        self.0.get()
    }

    pub(crate) fn bump(&self) {
        self.0.set(self.0.get().wrapping_add(1));
    }

    /// Bumps the version if `changed`, for a change that may leave the state
    /// as it was.
    #[inline]
    pub(crate) fn bump_if(&self, changed: bool) {
        if changed {
            self.bump();
        }
    }

    fn same_state(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// Something a view's look depends on that the view cannot read from an
/// entity or a global: a registry, a cache, a clock the application keeps.
///
/// A view reads it while it is drawn with [`crate::Window::depend_on`], and
/// whoever changes what it stands for calls [`DrawDependency::changed`]. With
/// view retention on, a view drawn again from the last frame is built instead
/// once a dependency it read changed. Without retention, reading one does
/// nothing and a change only asks the windows for a frame.
#[derive(Clone, Default, Debug)]
pub struct DrawDependency(StateVersion);

impl DrawDependency {
    /// A new dependency, which nothing has read yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks what this dependency stands for as changed, so that the views
    /// that read it are drawn again, and asks every window for a frame.
    pub fn changed(&self, cx: &mut App) {
        self.0.bump();
        cx.defer(|cx| cx.request_frame_in_every_window());
    }

    pub(crate) fn version(&self) -> &StateVersion {
        &self.0
    }
}

/// The entity map's half of recording what views read.
#[derive(Default)]
pub(crate) struct EntityAccessLog {
    /// Whether view retention is on in the app, which is what these stamps
    /// and logs are for.
    pub(crate) enabled: bool,
    /// Entities whose reads are left out of every recording while the views
    /// that said so (see [`crate::Context::untrack_reads_of`]) are drawn,
    /// innermost last.
    untracked: Rc<RefCell<Vec<EntityId>>>,
    /// Every entity accessed while a recording is open, in order and with
    /// repeats, with the write generation it was accessed at.
    access_log: RefCell<Vec<(EntityId, u64)>>,
    /// Where in `access_log` the last recording or replay began or ended; an
    /// access repeating the one just before it is only left out after this.
    boundary: Cell<usize>,
    /// How many recordings are open.
    recordings: Rc<Cell<usize>>,
    /// Counts the entities updated or changed while no recording is open.
    update_generation: u64,
    /// When each entity was last updated while no recording was open.
    updated_at: FxHashMap<EntityId, u64>,
    /// Counts the entities written while a recording is open: written while
    /// the window draws.
    write_generation: u64,
    written_at: FxHashMap<EntityId, u64>,
    /// Counts the writes while the window draws that keep the view drawing
    /// them in place (see `stays_put`): all of them but writes to an entity
    /// the views being drawn said they do not depend on, which a view drawn
    /// again, where it was or elsewhere, leaves out all the same.
    pinning_writes: u64,
    /// The entity the framework is about to lease to render it, which is not
    /// a write to it.
    rendering: Option<EntityId>,
    /// The entities updated and not notified since.
    updated_unnotified: FxHashSet<EntityId>,
    /// When each entity was last changed: notified after being updated, or
    /// notified while drawing.
    changed_at: FxHashMap<EntityId, u64>,
    /// The entity asked something by the platform (a text input's selection,
    /// say), whose update does not count as a change unless it notifies.
    queried: Option<EntityId>,
}

impl EntityAccessLog {
    #[inline]
    pub(crate) fn recording(&self) -> bool {
        self.recordings.get() > 0
    }

    /// Leaves reads of `entities` out of the recordings until popped,
    /// returning how many to pop.
    pub(crate) fn push_untracked(&self, entities: &[EntityId]) -> usize {
        if !self.enabled || entities.is_empty() {
            return 0;
        }
        self.untracked.borrow_mut().extend_from_slice(entities);
        entities.len()
    }

    pub(crate) fn pop_untracked(&self, count: usize) {
        if count > 0 {
            let mut untracked = self.untracked.borrow_mut();
            let keep = untracked.len().saturating_sub(count);
            untracked.truncate(keep);
        }
    }

    fn stamp_changed(&mut self, entity_id: EntityId) {
        self.update_generation += 1;
        self.changed_at.insert(entity_id, self.update_generation);
    }

    /// Forgets every stamp, as view retention is turned off: `forget` keeps
    /// nothing up to date while it is off, so an entity released meanwhile
    /// would otherwise keep its stamps for the life of the app.
    pub(crate) fn forget_all(&mut self) {
        self.updated_at = FxHashMap::default();
        self.written_at = FxHashMap::default();
        self.updated_unnotified = FxHashSet::default();
        self.changed_at = FxHashMap::default();
    }

    /// Forgets a released entity.
    pub(crate) fn forget(&mut self, entity_id: EntityId) {
        super::culprits::forget(entity_id);
        if self.enabled {
            self.updated_at.remove(&entity_id);
            self.written_at.remove(&entity_id);
            self.updated_unnotified.remove(&entity_id);
            self.changed_at.remove(&entity_id);
        }
    }

    fn changed_since(&self, entities: &[(EntityId, u64)], generation: u64, updates: bool) -> bool {
        let after = |stamps: &FxHashMap<EntityId, u64>, entity| {
            stamps.get(entity).is_some_and(|at| *at > generation)
        };
        generation != self.update_generation
            && entities.iter().any(|(entity, _)| {
                after(&self.changed_at, entity) || (updates && after(&self.updated_at, entity))
            })
    }

    /// Whether an entity in `entities` was written after it was read.
    fn written_since(&self, entities: &[(EntityId, u64)], floor: u64) -> bool {
        self.write_generation > floor
            && entities.iter().any(|(entity, read_at)| {
                self.written_at
                    .get(entity)
                    .is_some_and(|written_at| *written_at > (*read_at).max(floor))
            })
    }
}

impl EntityMap {
    fn mark_access_boundary(&mut self) {
        let log = &mut self.access_log;
        log.boundary.set(log.access_log.get_mut().len());
    }

    pub(crate) fn write_generation(&self) -> u64 {
        self.access_log.write_generation
    }

    /// See `EntityAccessLog::pinning_writes`.
    pub(crate) fn pinning_writes(&self) -> u64 {
        self.access_log.pinning_writes
    }
}

/// Records that `entity_id` was accessed.
#[inline]
pub(crate) fn note_access(entities: &EntityMap, entity_id: EntityId) {
    let log = &entities.access_log;
    if log.recordings.get() > 0 {
        if log
            .untracked
            .try_borrow()
            .is_ok_and(|untracked| !untracked.is_empty() && untracked.contains(&entity_id))
        {
            return;
        }
        let mut accesses = log.access_log.borrow_mut();
        // A view reads the same entity many times in a row as it renders.
        if accesses.len() > log.boundary.get()
            && accesses
                .last()
                .is_some_and(|(last, read_at)| *last == entity_id && *read_at == log.write_generation)
        {
            return;
        }
        accesses.push((entity_id, log.write_generation));
    }
}

/// Records that `entity_id` is notified. A notification following an update
/// marks the entity changed, and so does one while a view is being drawn (a
/// view changing a model it read as it renders) or while the platform asks it
/// something; nothing else tells whether what it holds changed. A
/// notification alone, outside drawing, changes nothing another view could
/// have read: the view notified is drawn again, not the views that read it.
#[inline]
pub(crate) fn note_notify(entities: &mut EntityMap, entity_id: EntityId) {
    let log = &mut entities.access_log;
    if !log.enabled {
        return;
    }
    if log.updated_unnotified.remove(&entity_id)
        || log.recordings.get() > 0
        || log.queried.is_some()
    {
        super::culprits::note_entity_change(entity_id, "notified");
        log.stamp_changed(entity_id);
    }
}

/// Records that `entity_id` is being updated.
///
/// Outside drawing (a task, a listener, an action), an update may change what
/// the entity holds without a notification, as when a view changes a model it
/// renders and notifies only itself; a view inside a notified view that read
/// it is drawn again, as the whole notified view would have been. While the
/// window draws, an update is a write: a view that read the entity before it
/// (a sibling drawn earlier, the view around both, or the writer itself
/// before it wrote) read what it held before and is drawn again. The update
/// is recorded as read after the write, so that a view writing an entity and
/// reading it back does not depend on its own write. The update that renders
/// a view is neither.
#[inline]
pub(crate) fn note_update(entities: &mut EntityMap, entity_id: EntityId) {
    let log = &mut entities.access_log;
    if log.enabled {
        if log.rendering == Some(entity_id) {
            log.rendering = None;
        } else if log.queried != Some(entity_id) {
            if log.recordings.get() == 0 {
                super::culprits::note_entity_change(entity_id, "updated");
                log.update_generation += 1;
                log.updated_at.insert(entity_id, log.update_generation);
                log.updated_unnotified.insert(entity_id);
            } else {
                let untracked = log
                    .untracked
                    .try_borrow()
                    .is_ok_and(|untracked| untracked.contains(&entity_id));
                if !untracked {
                    log.pinning_writes += 1;
                    super::culprits::note_entity_change(entity_id, "written while drawing");
                }
                log.write_generation += 1;
                log.written_at.insert(entity_id, log.write_generation);
            }
        }
    }
    note_access(entities, entity_id);
}

/// Marks the next lease of `entity_id` as the framework rendering it.
#[inline]
pub(crate) fn render_next(entities: &mut EntityMap, entity_id: EntityId) {
    if entities.access_log.enabled {
        entities.access_log.rendering = Some(entity_id);
    }
}

/// Runs `ask` on `entity` as the platform asks a text input about its
/// selection or bounds every frame, which does not count as changing what it
/// holds unless it notifies meanwhile.
#[inline]
pub(crate) fn query<T: 'static, R>(
    entity: &crate::Entity<T>,
    cx: &mut App,
    ask: impl FnOnce(&mut T, &mut crate::Context<T>) -> R,
) -> R {
    if !cx.entities.access_log.enabled {
        return entity.update(cx, ask);
    }
    let outer = cx
        .entities
        .access_log
        .queried
        .replace(entity.entity_id());
    let result = entity.update(cx, ask);
    cx.entities.access_log.queried = outer;
    result
}

/// Where a recording started by [`App::begin_recording_dependencies`] begins.
pub(crate) struct DependencyRecording {
    entities: usize,
    globals: usize,
    states: usize,
    deadlines: usize,
    generation: u64,
    updates: u64,
    /// Where its nested stretches begin in [`AppDependencies::nested`].
    nested: usize,
}

/// What a recording saw: everything read while it was open, and what was
/// read outside the recordings nested in it and the dependencies replayed
/// in it, which is what the view it recorded read itself.
pub(crate) struct Recorded {
    pub(crate) all: RenderDependencies,
    pub(crate) own: RenderDependencies,
}

/// What a view read while it was drawn.
#[derive(Clone, Default, Debug)]
pub(crate) struct RenderDependencies {
    /// The entities read, by id, each with the write generation it was
    /// first read at: a write after that changes what the view read.
    pub(crate) entities: Rc<[(EntityId, u64)]>,
    /// The globals read, likewise.
    pub(crate) globals: Rc<[(TypeId, u64)]>,
    /// Versioned state, with the version each was read at.
    pub(crate) states: Rc<[(StateVersion, u64)]>,
    /// The earliest time the view said it would look different at.
    pub(crate) rebuild_at: Option<Instant>,
    /// The global generation the recording began at.
    pub(crate) generation: u64,
    /// The entity update generation the recording began at.
    pub(crate) updates: u64,
    /// A write generation every read is known to be up to date with: a view
    /// drawn again was checked against every write up to it.
    pub(crate) floor: u64,
}

impl RenderDependencies {
    /// The same dependencies, known to be up to date with every write up to
    /// `writes`: a reused view's, checked when it was reused.
    pub(crate) fn written_up_to(&self, writes: u64) -> Self {
        Self {
            floor: self.floor.max(writes),
            ..self.clone()
        }
    }

    /// Both sets at once, as of the earlier generation, each read as of the
    /// earlier of the two.
    pub(crate) fn union(&self, other: &Self) -> Self {
        let mut states = self.states.to_vec();
        for state in other.states.iter() {
            if !states.iter().any(|(version, _)| version.same_state(&state.0)) {
                states.push(state.clone());
            }
        }
        Self {
            entities: merge_reads(&self.entities, self.floor, &other.entities, other.floor),
            globals: merge_reads(&self.globals, self.floor, &other.globals, other.floor),
            states: states.into(),
            rebuild_at: match (self.rebuild_at, other.rebuild_at) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            generation: self.generation.min(other.generation),
            updates: self.updates.min(other.updates),
            floor: 0,
        }
    }
}

/// Two sets of reads, each keyed and sorted by what was read, as one: a
/// read in both is taken as of the earlier of the two, each first raised to
/// its set's floor.
fn merge_reads<T: Ord + Copy>(
    a: &Rc<[(T, u64)]>,
    a_floor: u64,
    b: &Rc<[(T, u64)]>,
    b_floor: u64,
) -> Rc<[(T, u64)]> {
    if b.is_empty() && a_floor == 0 {
        return a.clone();
    }
    if a.is_empty() && b_floor == 0 {
        return b.clone();
    }
    let mut merged: Vec<(T, u64)> = a
        .iter()
        .map(|(key, read_at)| (*key, (*read_at).max(a_floor)))
        .chain(b.iter().map(|(key, read_at)| (*key, (*read_at).max(b_floor))))
        .collect();
    earliest_reads(&mut merged);
    merged.into()
}

/// Sorts reads by what was read and keeps the earliest read of each.
fn earliest_reads<T: Ord + Copy>(reads: &mut Vec<(T, u64)>) {
    reads.sort_unstable();
    reads.dedup_by_key(|(key, _)| *key);
}

/// The first read of each state, at the earliest version read, so that a
/// change in between still counts.
fn unique_states(states: &[(StateVersion, u64)]) -> Rc<[(StateVersion, u64)]> {
    let mut unique: Vec<(StateVersion, u64)> = Vec::with_capacity(states.len());
    for state in states {
        if !unique
            .iter()
            .any(|(version, _)| version.same_state(&state.0))
        {
            unique.push(state.clone());
        }
    }
    unique.into()
}

impl App {
    /// A handle for a window to record reads of its own state with.
    pub(crate) fn ambient_reads(&self) -> AmbientReads {
        AmbientReads {
            globals: self.dependencies.global_read_log.clone(),
            recordings: self.entities.access_log.recordings.clone(),
            untracked: self.entities.access_log.untracked.clone(),
        }
    }

    /// Starts recording what is read from here on. Recordings nest; each
    /// sees everything read while it is open, including what nested ones saw.
    pub(crate) fn begin_recording_dependencies(&mut self) -> DependencyRecording {
        let entities = &mut self.entities;
        entities.mark_access_boundary();
        let log = &mut entities.access_log;
        log.recordings.set(log.recordings.get() + 1);
        STATE_RECORDINGS.with(|recordings| recordings.set(recordings.get() + 1));
        DependencyRecording {
            entities: log.access_log.get_mut().len(),
            globals: self.dependencies.global_read_log.borrow().len(),
            states: STATE_READS.with_borrow(Vec::len),
            deadlines: DEADLINES.with_borrow(Vec::len),
            generation: self.dependencies.global_generation,
            updates: log.update_generation,
            nested: self.dependencies.nested.len(),
        }
    }

    /// Ends `recording`, returning what was read while it was open.
    pub(crate) fn finish_recording_dependencies(
        &mut self,
        recording: DependencyRecording,
    ) -> Recorded {
        let entity_log = self.entities.access_log.access_log.borrow();
        let global_log = self.dependencies.global_read_log.borrow();
        let stretch = Stretch {
            entities: recording.entities..entity_log.len(),
            globals: recording.globals..global_log.len(),
            states: recording.states..STATE_READS.with_borrow(Vec::len),
            deadlines: recording.deadlines..DEADLINES.with_borrow(Vec::len),
        };
        let nested = &self.dependencies.nested[recording.nested.min(self.dependencies.nested.len())..];
        let reads = |entities: Vec<(EntityId, u64)>,
                     globals: Vec<(TypeId, u64)>,
                     states: &[(StateVersion, u64)],
                     deadlines: &[Instant]| {
            let mut entities = entities;
            earliest_reads(&mut entities);
            let mut globals = globals;
            earliest_reads(&mut globals);
            RenderDependencies {
                entities: entities.into(),
                globals: globals.into(),
                states: unique_states(states),
                rebuild_at: deadlines.iter().copied().min(),
                // As of when the recording began, so that a global written
                // while it was open, after being read, counts as changed.
                generation: recording.generation,
                updates: recording.updates,
                floor: 0,
            }
        };
        let all = STATE_READS.with_borrow(|states| {
            DEADLINES.with_borrow(|deadlines| {
                reads(
                    entity_log[stretch.entities.clone()].to_vec(),
                    global_log[stretch.globals.clone()].to_vec(),
                    &states[stretch.states.clone()],
                    &deadlines[stretch.deadlines.clone()],
                )
            })
        });
        // Nested recordings that read nothing leave the enclosing recording's
        // all and own dependencies identical, including its deadline.
        let own = if nested.is_empty()
            || (stretch.entities.is_empty()
                && stretch.globals.is_empty()
                && stretch.states.is_empty()
                && stretch.deadlines.is_empty())
        {
            all.clone()
        } else {
            STATE_READS.with_borrow(|states| {
                DEADLINES.with_borrow(|deadlines| {
                    reads(
                        outside(&entity_log, stretch.entities.clone(), nested, |s| {
                            s.entities.clone()
                        }),
                        outside(&global_log, stretch.globals.clone(), nested, |s| {
                            s.globals.clone()
                        }),
                        &outside(states, stretch.states.clone(), nested, |s| s.states.clone()),
                        &outside(deadlines, stretch.deadlines.clone(), nested, |s| {
                            s.deadlines.clone()
                        }),
                    )
                })
            })
        };
        drop(entity_log);
        drop(global_log);

        let log = &mut self.entities.access_log;
        let open = log.recordings.get() - 1;
        log.recordings.set(open);
        STATE_RECORDINGS.with(|recordings| recordings.set(recordings.get() - 1));
        let nested = &mut self.dependencies.nested;
        nested.truncate(recording.nested);
        if open == 0 {
            log.access_log.get_mut().clear();
            self.dependencies.global_read_log.borrow_mut().clear();
            STATE_READS.with_borrow_mut(Vec::clear);
            DEADLINES.with_borrow_mut(Vec::clear);
            nested.clear();
        } else {
            nested.push(stretch);
        }
        self.entities.mark_access_boundary();

        Recorded { all, own }
    }

    /// Reads `dependencies` again, as a view drawn again from them does: the
    /// window tracks the entities, and any recording open includes them all.
    pub(crate) fn replay_dependencies(&mut self, dependencies: &RenderDependencies) {
        self.entities.mark_access_boundary();
        let recording = self.entities.access_log.recording();
        {
            let accessed = self.entities.accessed_entities.get_mut();
            accessed.extend(dependencies.entities.iter().map(|(entity, _)| *entity));
        }
        if recording {
            let floor = dependencies.floor;
            let start = Stretch {
                entities: self.entities.access_log.access_log.get_mut().len()..0,
                globals: self.dependencies.global_read_log.borrow().len()..0,
                states: STATE_READS.with_borrow(Vec::len)..0,
                deadlines: DEADLINES.with_borrow(Vec::len)..0,
            };
            self.entities.access_log.access_log.get_mut().extend(
                dependencies
                    .entities
                    .iter()
                    .map(|(entity, read_at)| (*entity, (*read_at).max(floor))),
            );
            self.dependencies.global_read_log.borrow_mut().extend(
                dependencies
                    .globals
                    .iter()
                    .map(|(global, read_at)| (*global, (*read_at).max(floor))),
            );
            STATE_READS.with_borrow_mut(|reads| reads.extend(dependencies.states.iter().cloned()));
            if let Some(deadline) = dependencies.rebuild_at {
                DEADLINES.with_borrow_mut(|deadlines| deadlines.push(deadline));
            }
            // What a view drawn again read is its own, not the recording's
            // around it.
            self.dependencies.nested.push(Stretch {
                entities: start.entities.start..self.entities.access_log.access_log.get_mut().len(),
                globals: start.globals.start..self.dependencies.global_read_log.borrow().len(),
                states: start.states.start..STATE_READS.with_borrow(Vec::len),
                deadlines: start.deadlines.start..DEADLINES.with_borrow(Vec::len),
            });
        }
        self.entities.mark_access_boundary();
    }

    /// Whether anything in `dependencies` but `except` may have changed, as
    /// [`Self::dependencies_changed`] answers.
    pub(crate) fn dependencies_changed_except(
        &self,
        dependencies: &RenderDependencies,
        inside_notified: bool,
        now: Instant,
        except: &StateVersion,
    ) -> Option<DependencyChange> {
        self.dependencies_changed_inner(dependencies, inside_notified, now, Some(except))
    }

    /// Whether anything in `dependencies` may have changed since it was
    /// recorded, or `now` is past when the view said it would change.
    ///
    /// An entity notified without being updated (as a scroll wheel or an
    /// animation notifies a view to draw it again) holds what it held: the
    /// view notified is drawn again, but a view that read it is not. An
    /// entity updated without being notified (as every subscriber is updated
    /// for each event it emits) counts as changed only `inside_notified`, for
    /// a view drawn inside a view notified since the last frame.
    pub(crate) fn dependencies_changed(
        &self,
        dependencies: &RenderDependencies,
        inside_notified: bool,
        now: Instant,
    ) -> Option<DependencyChange> {
        self.dependencies_changed_inner(dependencies, inside_notified, now, None)
    }

    fn dependencies_changed_inner(
        &self,
        dependencies: &RenderDependencies,
        inside_notified: bool,
        now: Instant,
        except: Option<&StateVersion>,
    ) -> Option<DependencyChange> {
        let log = &self.entities.access_log;
        if log.changed_since(&dependencies.entities, dependencies.updates, inside_notified)
            || log.written_since(&dependencies.entities, dependencies.floor)
        {
            if super::culprits::enabled()
                && let Some(entity) = dependencies.entities.iter().find(|entity| {
                    let entity = std::slice::from_ref(*entity);
                    log.changed_since(entity, dependencies.updates, inside_notified)
                        || log.written_since(entity, dependencies.floor)
                })
            {
                super::culprits::blame_entity(entity.0);
            }
            return Some(DependencyChange::Entity);
        }
        let floor = dependencies.floor;
        if let Some((global, _)) = dependencies.globals.iter().find(|(global, read_at)| {
            self.dependencies
                .global_changed_at
                .get(global)
                .is_some_and(|changed_at| *changed_at > dependencies.generation)
                || self
                    .dependencies
                    .global_written_at
                    .get(global)
                    .is_some_and(|written_at| *written_at > (*read_at).max(floor))
        }) {
            if super::culprits::enabled() {
                super::culprits::blame_global(*global);
            }
            return Some(DependencyChange::Global);
        }
        if dependencies
            .states
            .iter()
            .any(|(version, read_at)| {
                version.get() != *read_at
                    && except.is_none_or(|except| !version.same_state(except))
            })
        {
            return Some(DependencyChange::State);
        }
        if dependencies
            .rebuild_at
            .is_some_and(|deadline| deadline <= now)
        {
            return Some(DependencyChange::Deadline);
        }
        None
    }
}

/// What changed in a view's dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DependencyChange {
    Entity,
    Global,
    State,
    Deadline,
}

#[cfg(all(test, target_os = "linux"))]
#[path = "dependency_profile.rs"]
mod dependency_profile;

#[cfg(test)]
mod empty_nested_tests {
    use super::*;

    #[test]
    fn empty_nested_recording_shares_all_and_own_lists() {
        let cx = crate::TestAppContext::single();
        cx.update(|cx| {
            cx.set_view_retention(true);
            let outer = cx.begin_recording_dependencies();
            let inner = cx.begin_recording_dependencies();
            drop(cx.finish_recording_dependencies(inner));
            let outer = cx.finish_recording_dependencies(outer);
            assert!(outer.all.entities.is_empty());
            assert!(outer.all.globals.is_empty());
            assert!(outer.all.states.is_empty());
            assert!(Rc::ptr_eq(&outer.all.entities, &outer.own.entities));
            assert!(Rc::ptr_eq(&outer.all.globals, &outer.own.globals));
            assert!(Rc::ptr_eq(&outer.all.states, &outer.own.states));
        });
    }

    #[test]
    fn nested_deadline_stays_out_of_own_dependencies() {
        let cx = crate::TestAppContext::single();
        cx.update(|cx| {
            cx.set_view_retention(true);
            let outer = cx.begin_recording_dependencies();
            let inner = cx.begin_recording_dependencies();
            let deadline = Instant::now();
            note_deadline(deadline);
            drop(cx.finish_recording_dependencies(inner));
            let outer = cx.finish_recording_dependencies(outer);
            assert_eq!(outer.all.rebuild_at, Some(deadline));
            assert_eq!(outer.own.rebuild_at, None);
        });
    }
}

#[cfg(test)]
mod state_check_tests {
    use super::*;

    #[test]
    fn ignoring_a_frame_marker_still_checks_other_state() {
        let cx = crate::TestAppContext::single();
        cx.update(|cx| {
            cx.set_view_retention(true);
            let marker = StateVersion::default();
            let other = StateVersion::default();
            let recording = cx.begin_recording_dependencies();
            note_state_read(&marker);
            note_state_read(&other);
            let dependencies = cx.finish_recording_dependencies(recording).all;
            marker.bump();
            let now = Instant::now();
            assert_eq!(cx.dependencies_changed(&dependencies, false, now), Some(DependencyChange::State));
            assert_eq!(cx.dependencies_changed_except(&dependencies, false, now, &marker), None);
            other.bump();
            assert_eq!(cx.dependencies_changed_except(&dependencies, false, now, &marker), Some(DependencyChange::State));
        });
    }

    #[test]
    fn an_absent_exception_does_not_hide_state_changes() {
        let cx = crate::TestAppContext::single();
        cx.update(|cx| {
            cx.set_view_retention(true);
            let marker = StateVersion::default();
            let other = StateVersion::default();
            let recording = cx.begin_recording_dependencies();
            note_state_read(&other);
            let dependencies = cx.finish_recording_dependencies(recording).all;
            other.bump();
            assert_eq!(cx.dependencies_changed_except(&dependencies, false, Instant::now(), &marker), Some(DependencyChange::State));
        });
    }

    #[test]
    fn ignoring_state_preserves_entity_global_and_deadline_priority() {
        let cx = crate::TestAppContext::single();
        cx.update(|cx| {
            cx.set_view_retention(true);
            let marker = StateVersion::default();
            let now = Instant::now();
            let entity = EntityId::from(1);
            let global = TypeId::of::<u32>();
            let recording = cx.begin_recording_dependencies();
            note_access(&cx.entities, entity);
            note_global_read(cx, global);
            note_state_read(&marker);
            note_deadline(now);
            let dependencies = cx.finish_recording_dependencies(recording).all;
            marker.bump();
            assert_eq!(cx.dependencies_changed_except(&dependencies, false, now, &marker), Some(DependencyChange::Deadline));
            cx.dependencies.global_changed(global);
            assert_eq!(cx.dependencies_changed_except(&dependencies, false, now, &marker), Some(DependencyChange::Global));
            cx.entities.access_log.stamp_changed(entity);
            assert_eq!(cx.dependencies_changed_except(&dependencies, false, now, &marker), Some(DependencyChange::Entity));
            assert_eq!(cx.dependencies_changed(&dependencies, false, now), Some(DependencyChange::Entity));
        });
    }
}

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
use smallvec::SmallVec;
use std::{
    any::TypeId,
    cell::{Cell, RefCell},
    rc::Rc,
    time::Instant,
};

/// The app's half of recording what views read: when each global last
/// changed, and the globals read while a recording is open.
#[derive(Default)]
pub(crate) struct AppDependencies {
    /// Counts the changes to globals, each stamped into `global_changed_at`.
    global_generation: u64,
    global_changed_at: TypeIdHashMap<u64>,
    /// Every global read while a recording is open, pointer and modifier
    /// reads included (see [`ambient`]).
    global_read_log: Rc<RefCell<Vec<TypeId>>>,
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
    if cx.entities.access_log.recording() {
        cx.dependencies.global_read_log.borrow_mut().push(global);
    }
}

/// Stamps a change to the global of type `global_type`, as it is written.
#[inline]
pub(crate) fn global_changed(cx: &mut App, global_type: TypeId) {
    if cx.entities.access_log.enabled {
        cx.dependencies.global_changed(global_type);
    }
}

/// Stamps a change to whether a global of type `G` is set, when it is about
/// to be set where it was not.
pub(crate) fn note_global_inserted<G: 'static>(cx: &mut App) {
    if cx.entities.access_log.enabled && !cx.globals_by_type.contains_key(&TypeId::of::<G>()) {
        cx.dependencies
            .global_changed(TypeId::of::<GlobalPresence<G>>());
    }
}

/// Stamps a change to whether a global of type `G` is set, as it is about to
/// be removed.
pub(crate) fn note_global_removed<G: 'static>(cx: &mut App) {
    if cx.entities.access_log.enabled {
        cx.dependencies
            .global_changed(TypeId::of::<GlobalPresence<G>>());
    }
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
}

/// A window's handle on the app's recording, so that reading the window's own
/// state while a view is drawn is recorded as a global read would be.
#[derive(Clone)]
pub(crate) struct AmbientReads {
    globals: Rc<RefCell<Vec<TypeId>>>,
    recordings: Rc<Cell<usize>>,
}

impl AmbientReads {
    #[inline]
    pub(crate) fn note<T: 'static>(&self) {
        if self.recordings.get() > 0 {
            self.globals.borrow_mut().push(TypeId::of::<T>());
        }
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
    /// Every entity accessed while a recording is open, in order and with
    /// repeats.
    access_log: RefCell<Vec<EntityId>>,
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

    fn stamp_changed(&mut self, entity_id: EntityId) {
        self.update_generation += 1;
        self.changed_at.insert(entity_id, self.update_generation);
    }

    /// Forgets a released entity.
    pub(crate) fn forget(&mut self, entity_id: EntityId) {
        if self.enabled {
            self.updated_at.remove(&entity_id);
            self.written_at.remove(&entity_id);
            self.updated_unnotified.remove(&entity_id);
            self.changed_at.remove(&entity_id);
        }
    }

    fn changed_since(&self, entities: &[EntityId], generation: u64, updates: bool) -> bool {
        let after = |stamps: &FxHashMap<EntityId, u64>, entity| {
            stamps.get(entity).is_some_and(|at| *at > generation)
        };
        generation != self.update_generation
            && entities.iter().any(|entity| {
                after(&self.changed_at, entity) || (updates && after(&self.updated_at, entity))
            })
    }

    fn written_since(&self, entities: &[EntityId], writes: &Writes) -> bool {
        self.write_generation != writes.to
            && entities.iter().any(|entity| {
                self.written_at
                    .get(entity)
                    .is_some_and(|written_at| writes.is_foreign(*written_at))
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
}

/// Records that `entity_id` was accessed.
#[inline]
pub(crate) fn note_access(entities: &EntityMap, entity_id: EntityId) {
    let log = &entities.access_log;
    if log.recordings.get() > 0 {
        let mut accesses = log.access_log.borrow_mut();
        // A view reads the same entity many times in a row as it renders.
        if accesses.len() > log.boundary.get() && accesses.last() == Some(&entity_id) {
            return;
        }
        accesses.push(entity_id);
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
        log.stamp_changed(entity_id);
    }
}

/// Records that `entity_id` is being updated.
///
/// Outside drawing (a task, a listener, an action), an update may change what
/// the entity holds without a notification, as when a view changes a model it
/// renders and notifies only itself; a view inside a notified view that read
/// it is drawn again, as the whole notified view would have been. While the
/// window draws, an update is a write, and a view that read the entity is
/// drawn again unless it wrote the entity itself while it was being built.
/// The update that renders a view is neither.
#[inline]
pub(crate) fn note_update(entities: &mut EntityMap, entity_id: EntityId) {
    note_access(entities, entity_id);
    let log = &mut entities.access_log;
    if !log.enabled {
        return;
    }
    if log.rendering == Some(entity_id) {
        log.rendering = None;
        if log.recordings.get() > 0 {
            return;
        }
    }
    if log.queried == Some(entity_id) {
        return;
    }
    if log.recordings.get() == 0 {
        log.update_generation += 1;
        log.updated_at.insert(entity_id, log.update_generation);
        log.updated_unnotified.insert(entity_id);
    } else {
        log.write_generation += 1;
        log.written_at.insert(entity_id, log.write_generation);
    }
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
    writes: u64,
}

/// Where in the write generation a view was built: writes after `from`
/// change what it read, except those it made itself, in the `own` stretches.
#[derive(Clone, Default, Debug)]
pub(crate) struct Writes {
    from: u64,
    to: u64,
    own: SmallVec<[(u64, u64); 2]>,
}

impl Writes {
    fn is_foreign(&self, written_at: u64) -> bool {
        written_at > self.from
            && !self
                .own
                .iter()
                .any(|(began, finished)| written_at > *began && written_at <= *finished)
    }

    fn union(&self, other: &Self) -> Self {
        let mut own = self.own.clone();
        own.extend_from_slice(&other.own);
        Writes {
            from: self.from.min(other.from),
            to: self.to.max(other.to),
            own,
        }
    }
}

/// What a view read while it was drawn.
#[derive(Clone, Default, Debug)]
pub(crate) struct RenderDependencies {
    pub(crate) entities: Rc<[EntityId]>,
    pub(crate) globals: Rc<[TypeId]>,
    /// Versioned state, with the version each was read at.
    pub(crate) states: Rc<[(StateVersion, u64)]>,
    /// The earliest time the view said it would look different at.
    pub(crate) rebuild_at: Option<Instant>,
    /// The global generation the recording began at.
    pub(crate) generation: u64,
    /// The entity update generation the recording began at.
    pub(crate) updates: u64,
    pub(crate) writes: Writes,
}

impl RenderDependencies {
    /// The same dependencies, known to be up to date with every write up to
    /// `writes`: a reused view's, checked when it was reused.
    pub(crate) fn written_up_to(&self, writes: u64) -> Self {
        Self {
            writes: Writes {
                from: writes,
                to: writes,
                own: SmallVec::new(),
            },
            ..self.clone()
        }
    }

    /// Both sets at once, as of the earlier generation.
    pub(crate) fn union(&self, other: &Self) -> Self {
        let mut states = self.states.to_vec();
        for state in other.states.iter() {
            if !states.iter().any(|(version, _)| version.same_state(&state.0)) {
                states.push(state.clone());
            }
        }
        Self {
            entities: merge_sorted(&self.entities, &other.entities),
            globals: merge_sorted(&self.globals, &other.globals),
            states: states.into(),
            rebuild_at: match (self.rebuild_at, other.rebuild_at) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            generation: self.generation.min(other.generation),
            updates: self.updates.min(other.updates),
            writes: self.writes.union(&other.writes),
        }
    }
}

fn merge_sorted<T: Ord + Copy>(a: &Rc<[T]>, b: &Rc<[T]>) -> Rc<[T]> {
    if b.is_empty() || Rc::ptr_eq(a, b) {
        return a.clone();
    }
    if a.is_empty() {
        return b.clone();
    }
    let mut merged: Vec<T> = a.iter().chain(b.iter()).copied().collect();
    merged.sort_unstable();
    merged.dedup();
    merged.into()
}

fn sorted_unique<T: Ord + Copy>(items: &[T]) -> Rc<[T]> {
    let mut items = items.to_vec();
    items.sort_unstable();
    items.dedup();
    items.into()
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
            writes: log.write_generation,
        }
    }

    /// Ends `recording`, returning what was read while it was open.
    pub(crate) fn finish_recording_dependencies(
        &mut self,
        recording: DependencyRecording,
    ) -> RenderDependencies {
        let entities = {
            let accesses = self.entities.access_log.access_log.borrow();
            sorted_unique(&accesses[recording.entities..])
        };
        let globals = {
            let reads = self.dependencies.global_read_log.borrow();
            sorted_unique(&reads[recording.globals..])
        };
        let states = STATE_READS.with_borrow(|reads| unique_states(&reads[recording.states..]));
        let rebuild_at = DEADLINES.with_borrow(|deadlines| {
            deadlines[recording.deadlines..].iter().copied().min()
        });

        let log = &mut self.entities.access_log;
        let open = log.recordings.get() - 1;
        log.recordings.set(open);
        STATE_RECORDINGS.with(|recordings| recordings.set(recordings.get() - 1));
        if open == 0 {
            log.access_log.get_mut().clear();
            self.dependencies.global_read_log.borrow_mut().clear();
            STATE_READS.with_borrow_mut(Vec::clear);
            DEADLINES.with_borrow_mut(Vec::clear);
        }
        self.entities.mark_access_boundary();

        let log = &self.entities.access_log;
        let mut own = SmallVec::new();
        if log.write_generation > recording.writes {
            own.push((recording.writes, log.write_generation));
        }
        RenderDependencies {
            entities,
            globals,
            states,
            rebuild_at,
            // As of when the recording began, so that a global written while
            // it was open, after being read, counts as changed.
            generation: recording.generation,
            updates: recording.updates,
            writes: Writes {
                from: recording.writes,
                to: log.write_generation,
                own,
            },
        }
    }

    /// Reads `dependencies` again, as a view drawn again from them does: the
    /// window tracks the entities, and any recording open includes them all.
    pub(crate) fn replay_dependencies(&mut self, dependencies: &RenderDependencies) {
        self.entities.mark_access_boundary();
        let recording = self.entities.access_log.recording();
        {
            let accessed = self.entities.accessed_entities.get_mut();
            accessed.extend(dependencies.entities.iter().copied());
        }
        if recording {
            self.entities
                .access_log
                .access_log
                .get_mut()
                .extend(dependencies.entities.iter().copied());
            self.dependencies
                .global_read_log
                .borrow_mut()
                .extend(dependencies.globals.iter().copied());
            STATE_READS.with_borrow_mut(|reads| reads.extend(dependencies.states.iter().cloned()));
            if let Some(deadline) = dependencies.rebuild_at {
                DEADLINES.with_borrow_mut(|deadlines| deadlines.push(deadline));
            }
        }
        self.entities.mark_access_boundary();
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
        let log = &self.entities.access_log;
        if log.changed_since(&dependencies.entities, dependencies.updates, inside_notified)
            || log.written_since(&dependencies.entities, &dependencies.writes)
        {
            return Some(DependencyChange::Entity);
        }
        if dependencies.globals.iter().any(|global| {
            self.dependencies
                .global_changed_at
                .get(global)
                .is_some_and(|changed_at| *changed_at > dependencies.generation)
        }) {
            return Some(DependencyChange::Global);
        }
        if dependencies
            .states
            .iter()
            .any(|(version, read_at)| version.get() != *read_at)
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

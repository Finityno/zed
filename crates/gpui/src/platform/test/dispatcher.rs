use crate::{PlatformDispatcher, Priority, ResourceSample, RunnableVariant};
use parking_lot::Mutex;
use scheduler::Instant;
use scheduler::{Clock, Scheduler, SessionId, TestScheduler, TestSchedulerConfig, Yield};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

/// TestDispatcher provides deterministic async execution for tests.
///
/// This implementation delegates task scheduling to the scheduler crate's `TestScheduler`.
/// Access the scheduler directly via `scheduler()` for clock, rng, and parking control.
#[doc(hidden)]
pub struct TestDispatcher {
    session_id: SessionId,
    scheduler: Arc<TestScheduler>,
    num_cpus_override: Arc<AtomicUsize>,
    draw_resources: Arc<Mutex<ScriptedDrawResources>>,
}

/// Draw resource samples a test scripted, and the requests made for them.
#[derive(Default)]
struct ScriptedDrawResources {
    samples: VecDeque<ResourceSample>,
    requests: Vec<bool>,
}

impl TestDispatcher {
    pub fn new(seed: u64) -> Self {
        let scheduler = Arc::new(TestScheduler::new(TestSchedulerConfig {
            seed,
            randomize_order: true,
            allow_parking: false,
            capture_pending_traces: std::env::var("PENDING_TRACES")
                .map_or(false, |var| var == "1" || var == "true"),
            timeout_ticks: 0..=1000,
        }));
        Self::from_scheduler(scheduler)
    }

    pub fn from_scheduler(scheduler: Arc<TestScheduler>) -> Self {
        TestDispatcher {
            session_id: scheduler.allocate_session_id(),
            scheduler,
            num_cpus_override: Arc::new(AtomicUsize::new(0)),
            draw_resources: Arc::default(),
        }
    }

    pub fn scheduler(&self) -> &Arc<TestScheduler> {
        &self.scheduler
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub fn drain_tasks(&self) {
        self.scheduler.drain_tasks();
    }

    pub fn advance_clock(&self, by: Duration) {
        self.scheduler.advance_clock(by);
    }

    pub fn advance_clock_to_next_timer(&self) -> bool {
        self.scheduler.advance_clock_to_next_timer()
    }

    pub fn simulate_random_delay(&self) -> Yield {
        self.scheduler.yield_random()
    }

    pub fn tick(&self, background_only: bool) -> bool {
        if background_only {
            self.scheduler.tick_background_only()
        } else {
            self.scheduler.tick()
        }
    }

    pub fn run_until_parked(&self) {
        while self.tick(false) {}
    }

    pub fn allow_parking(&self) {
        self.scheduler.allow_parking();
    }

    pub fn forbid_parking(&self) {
        self.scheduler.forbid_parking();
    }

    /// Override the value returned by `BackgroundExecutor::num_cpus()` in tests.
    /// A value of 0 means no override (the default of 4 is used).
    pub fn set_num_cpus(&self, count: usize) {
        self.num_cpus_override.store(count, Ordering::SeqCst);
    }

    /// Queues samples for [`PlatformDispatcher::sample_draw_resources`] to
    /// return, one per call; once they run out it returns `None`.
    pub fn script_draw_resource_samples(&self, samples: impl IntoIterator<Item = ResourceSample>) {
        self.draw_resources.lock().samples.extend(samples);
    }

    /// Takes the `process_counters` argument of every
    /// [`PlatformDispatcher::sample_draw_resources`] call made since the
    /// last time this was called.
    pub fn take_draw_resource_requests(&self) -> Vec<bool> {
        std::mem::take(&mut self.draw_resources.lock().requests)
    }

    /// Returns the overridden CPU count, or `None` if no override is set.
    pub fn num_cpus_override(&self) -> Option<usize> {
        match self.num_cpus_override.load(Ordering::SeqCst) {
            0 => None,
            n => Some(n),
        }
    }
}

impl Clone for TestDispatcher {
    fn clone(&self) -> Self {
        let session_id = self.scheduler.allocate_session_id();
        Self {
            session_id,
            scheduler: self.scheduler.clone(),
            num_cpus_override: self.num_cpus_override.clone(),
            draw_resources: self.draw_resources.clone(),
        }
    }
}

impl PlatformDispatcher for TestDispatcher {
    fn is_main_thread(&self) -> bool {
        self.scheduler.is_main_thread()
    }

    fn now(&self) -> Instant {
        self.scheduler.clock().now()
    }

    fn dispatch(&self, runnable: RunnableVariant, priority: Priority) {
        self.scheduler
            .schedule_background_with_priority(runnable, priority);
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, _priority: Priority) {
        self.scheduler.schedule_local(self.session_id, runnable);
    }

    fn dispatch_after(&self, _duration: Duration, _runnable: RunnableVariant) {
        panic!(
            "dispatch_after should not be called in tests. \
            Use BackgroundExecutor::timer() which uses the scheduler's native timer."
        );
    }

    fn as_test(&self) -> Option<&TestDispatcher> {
        Some(self)
    }

    fn sample_draw_resources(&self, process_counters: bool) -> Option<ResourceSample> {
        let mut draw_resources = self.draw_resources.lock();
        draw_resources.requests.push(process_counters);
        draw_resources.samples.pop_front()
    }

    fn spawn_realtime(&self, f: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(move || {
            f();
        });
    }
}

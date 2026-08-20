use std::{
    collections::VecDeque,
    fmt,
    iter::FusedIterator,
    sync::{Arc, atomic::AtomicUsize},
};

use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::Priority;

/// Depth of the platform main-thread runnable queue as of the most recent
/// enqueue or drain-batch exit. Exposed so application crash reporting can
/// stamp a starved-queue incident with its scale — a runaway producer reads as
/// six-figure depth in the report itself, without needing the multi-gigabyte
/// dump that queue becomes.
///
/// Only the Windows dispatcher writes it today. `threaded_dispatcher` and the
/// web dispatcher drain a [`PriorityQueueReceiver`] for main-thread runnables
/// just the same and could starve identically, so a 0 here means "this
/// platform does not report yet", not "this platform does not queue"; macOS
/// and Linux hand runnables straight to an OS run loop and have no depth to
/// report at all.
pub static MAIN_THREAD_QUEUE_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// See [`MAIN_THREAD_QUEUE_DEPTH`].
pub fn main_thread_queue_depth() -> usize {
    MAIN_THREAD_QUEUE_DEPTH.load(std::sync::atomic::Ordering::Relaxed)
}

struct PriorityQueues<T> {
    high_priority: VecDeque<T>,
    medium_priority: VecDeque<T>,
    low_priority: VecDeque<T>,
}

impl<T> PriorityQueues<T> {
    fn is_empty(&self) -> bool {
        self.high_priority.is_empty()
            && self.medium_priority.is_empty()
            && self.low_priority.is_empty()
    }

    fn len(&self) -> usize {
        self.high_priority.len() + self.medium_priority.len() + self.low_priority.len()
    }

    /// A burst can double a queue's backing buffer into the gigabytes (a
    /// field dump caught a starved main-thread queue at 4 GiB of capacity),
    /// and `VecDeque` never returns capacity on its own — after the drain the
    /// empty queue would pin the burst's memory for the life of the process.
    ///
    /// Shrinking toward twice the occupancy once it falls under a quarter of
    /// capacity reallocates at most once per halving of the queue (amortized
    /// O(1) per pop) while a drain is in progress, and releases the whole
    /// buffer down to the floor on the last pops. The 1024-entry gate with a
    /// 512-entry target is hysteresis: steady-state queues never churn.
    ///
    /// Note the cost of releasing progressively: `shrink_to` reallocates and
    /// moves every live element while holding the queue mutex, which
    /// `spin_send` and `spin_try_recv` busy-wait on with `try_lock` and no
    /// yield. The first shrink of a multi-million-entry burst moves a quarter
    /// of it in one hold and spins every waiting producer core for that long.
    /// That is the deliberate trade: returning a multi-gigabyte buffer to the
    /// allocator early, during the drain, beats holding it to the tail when
    /// the incident being recovered from is an out-of-memory — but only where
    /// that trade was actually made, which is why the drain paths run this
    /// behind [`PriorityQueueReceiver::reclaiming_capacity`].
    fn shrink_oversized(&mut self) {
        for queue in [
            &mut self.high_priority,
            &mut self.medium_priority,
            &mut self.low_priority,
        ] {
            let capacity = queue.capacity();
            if capacity >= 1024 && queue.len() <= capacity / 4 {
                queue.shrink_to((queue.len() * 2).max(512));
            }
        }
    }
}

struct PriorityQueueState<T> {
    queues: parking_lot::Mutex<PriorityQueues<T>>,
    condvar: parking_lot::Condvar,
    receiver_count: AtomicUsize,
    sender_count: AtomicUsize,
}

impl<T> PriorityQueueState<T> {
    fn send(&self, priority: Priority, item: T) -> Result<usize, SendError<T>> {
        if self
            .receiver_count
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            return Err(SendError(item));
        }

        let mut queues = self.queues.lock();
        Self::push(&mut queues, priority, item);
        let queued = queues.len();
        self.condvar.notify_one();
        Ok(queued)
    }

    fn spin_send(&self, priority: Priority, item: T) -> Result<(), SendError<T>> {
        if self
            .receiver_count
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            return Err(SendError(item));
        }

        let mut queues = loop {
            if let Some(guard) = self.queues.try_lock() {
                break guard;
            }
            std::hint::spin_loop();
        };
        Self::push(&mut queues, priority, item);
        self.condvar.notify_one();
        Ok(())
    }

    fn push(queues: &mut PriorityQueues<T>, priority: Priority, item: T) {
        match priority {
            Priority::RealtimeAudio => unreachable!(
                "Realtime audio priority runs on a dedicated thread and is never queued"
            ),
            Priority::High => queues.high_priority.push_back(item),
            Priority::Medium => queues.medium_priority.push_back(item),
            Priority::Low => queues.low_priority.push_back(item),
        };
    }

    fn recv<'a>(&'a self) -> Result<parking_lot::MutexGuard<'a, PriorityQueues<T>>, RecvError> {
        let mut queues = self.queues.lock();

        let sender_count = self.sender_count.load(std::sync::atomic::Ordering::Relaxed);
        if queues.is_empty() && sender_count == 0 {
            return Err(crate::queue::RecvError);
        }

        while queues.is_empty() {
            // The one place every GPUI-owned worker blocks: the Linux and
            // Windows dispatcher pools and `ThreadedDispatcher` all park here.
            // (macOS dispatches onto GCD, whose threads GPUI does not own and
            // cannot announce for.)
            let _parked = crate::thread_park::park();
            self.condvar.wait(&mut queues);
        }

        Ok(queues)
    }

    fn try_recv<'a>(
        &'a self,
    ) -> Result<Option<parking_lot::MutexGuard<'a, PriorityQueues<T>>>, RecvError> {
        let mut queues = self.queues.lock();

        let sender_count = self.sender_count.load(std::sync::atomic::Ordering::Relaxed);
        if queues.is_empty() && sender_count == 0 {
            return Err(crate::queue::RecvError);
        }

        if queues.is_empty() {
            Ok(None)
        } else {
            Ok(Some(queues))
        }
    }

    fn spin_try_recv<'a>(
        &'a self,
    ) -> Result<Option<parking_lot::MutexGuard<'a, PriorityQueues<T>>>, RecvError> {
        let queues = loop {
            if let Some(guard) = self.queues.try_lock() {
                break guard;
            }
            std::hint::spin_loop();
        };

        let sender_count = self.sender_count.load(std::sync::atomic::Ordering::Relaxed);
        if queues.is_empty() && sender_count == 0 {
            return Err(crate::queue::RecvError);
        }

        if queues.is_empty() {
            Ok(None)
        } else {
            Ok(Some(queues))
        }
    }
}

#[doc(hidden)]
pub struct PriorityQueueSender<T> {
    state: Arc<PriorityQueueState<T>>,
}

impl<T> PriorityQueueSender<T> {
    fn new(state: Arc<PriorityQueueState<T>>) -> Self {
        Self { state }
    }

    pub fn send(&self, priority: Priority, item: T) -> Result<(), SendError<T>> {
        self.state.send(priority, item)?;
        Ok(())
    }

    /// Like [`Self::send`], but returns the total number of queued elements
    /// after the push, measured under the same lock. Lets a dispatcher watch
    /// for a runaway producer (a queue that only ever grows because its
    /// consumer is starved) without taking the lock a second time.
    pub fn send_and_len(&self, priority: Priority, item: T) -> Result<usize, SendError<T>> {
        self.state.send(priority, item)
    }

    pub fn spin_send(&self, priority: Priority, item: T) -> Result<(), SendError<T>> {
        self.state.spin_send(priority, item)?;
        Ok(())
    }
}

impl<T> Drop for PriorityQueueSender<T> {
    fn drop(&mut self) {
        self.state
            .sender_count
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

#[doc(hidden)]
pub struct PriorityQueueReceiver<T> {
    state: Arc<PriorityQueueState<T>>,
    rand: SmallRng,
    disconnected: bool,
    reclaim_capacity: bool,
}

impl<T> Clone for PriorityQueueReceiver<T> {
    fn clone(&self) -> Self {
        self.state
            .receiver_count
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self {
            state: Arc::clone(&self.state),
            rand: SmallRng::seed_from_u64(0),
            disconnected: self.disconnected,
            reclaim_capacity: self.reclaim_capacity,
        }
    }
}

#[doc(hidden)]
pub struct SendError<T>(pub T);

impl<T: fmt::Debug> fmt::Debug for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SendError").field(&self.0).finish()
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct RecvError;

#[allow(dead_code)]
impl<T> PriorityQueueReceiver<T> {
    pub fn new() -> (PriorityQueueSender<T>, Self) {
        let state = PriorityQueueState {
            queues: parking_lot::Mutex::new(PriorityQueues {
                high_priority: VecDeque::new(),
                medium_priority: VecDeque::new(),
                low_priority: VecDeque::new(),
            }),
            condvar: parking_lot::Condvar::new(),
            receiver_count: AtomicUsize::new(1),
            sender_count: AtomicUsize::new(1),
        };
        let state = Arc::new(state);

        let sender = PriorityQueueSender::new(Arc::clone(&state));

        let receiver = PriorityQueueReceiver {
            state,
            rand: SmallRng::seed_from_u64(0),
            disconnected: false,
            reclaim_capacity: false,
        };

        (sender, receiver)
    }

    /// Opts this receiver's drain into returning backing-buffer capacity to
    /// the allocator as the queue empties (see
    /// [`PriorityQueues::shrink_oversized`]). Clones of the receiver inherit
    /// it, so call this on the receiver `new` hands back, before it is handed
    /// around.
    ///
    /// Off by default because the shrink reallocates and moves the live
    /// elements while holding the queue mutex that `send` also blocks on. On a
    /// many-producer background pool that turns one burst's cleanup into a
    /// stall for every producer, which is a bad trade for a queue whose bursts
    /// are bounded by the work that spawned them. It pays for itself on a
    /// single-consumer queue whose producer can outrun it without bound — the
    /// platform main-thread queue, where a starved drain has been seen leaving
    /// gigabytes of capacity pinned for the life of the process.
    pub fn reclaiming_capacity(mut self) -> Self {
        self.reclaim_capacity = true;
        self
    }

    /// Returns whether the queue currently contains no elements.
    pub fn is_empty(&self) -> bool {
        self.state.queues.lock().is_empty()
    }

    /// Returns the number of queued elements across all priorities.
    pub fn len(&self) -> usize {
        self.state.queues.lock().len()
    }

    #[cfg(test)]
    fn low_priority_capacity(&self) -> usize {
        self.state.queues.lock().low_priority.capacity()
    }

    /// Tries to pop one element from the priority queue without blocking.
    ///
    /// This will early return if there are no elements in the queue.
    ///
    /// This method is best suited if you only intend to pop one element, for better performance
    /// on large queues see [`Self::try_iter`]
    ///
    /// # Errors
    ///
    /// If the sender was dropped
    pub fn try_pop(&mut self) -> Result<Option<T>, RecvError> {
        self.pop_inner(false)
    }

    pub fn spin_try_pop(&mut self) -> Result<Option<T>, RecvError> {
        use Priority as P;

        let Some(mut queues) = self.state.spin_try_recv()? else {
            return Ok(None);
        };
        if self.reclaim_capacity {
            queues.shrink_oversized();
        }

        let high = P::High.weight() * !queues.high_priority.is_empty() as u32;
        let medium = P::Medium.weight() * !queues.medium_priority.is_empty() as u32;
        let low = P::Low.weight() * !queues.low_priority.is_empty() as u32;
        let mut mass = high + medium + low;

        if !queues.high_priority.is_empty() {
            let flip = self.rand.random_ratio(P::High.weight(), mass);
            if flip {
                return Ok(queues.high_priority.pop_front());
            }
            mass -= P::High.weight();
        }

        if !queues.medium_priority.is_empty() {
            let flip = self.rand.random_ratio(P::Medium.weight(), mass);
            if flip {
                return Ok(queues.medium_priority.pop_front());
            }
            mass -= P::Medium.weight();
        }

        if !queues.low_priority.is_empty() {
            let flip = self.rand.random_ratio(P::Low.weight(), mass);
            if flip {
                return Ok(queues.low_priority.pop_front());
            }
        }

        Ok(None)
    }

    /// Pops an element from the priority queue blocking if necessary.
    ///
    /// This method is best suited if you only intend to pop one element, for better performance
    /// on large queues see [`Self::iter``]
    ///
    /// # Errors
    ///
    /// If the sender was dropped
    pub fn pop(&mut self) -> Result<T, RecvError> {
        self.pop_inner(true).map(|e| e.unwrap())
    }

    /// Returns an iterator over the elements of the queue
    /// this iterator will end when all elements have been consumed and will not wait for new ones.
    pub fn try_iter(self) -> TryIter<T> {
        TryIter {
            receiver: self,
            ended: false,
        }
    }

    /// Returns an iterator over the elements of the queue
    /// this iterator will wait for new elements if the queue is empty.
    pub fn iter(self) -> Iter<T> {
        Iter(self)
    }

    #[inline(always)]
    // algorithm is the loaded die from biased coin from
    // https://www.keithschwarz.com/darts-dice-coins/
    fn pop_inner(&mut self, block: bool) -> Result<Option<T>, RecvError> {
        use Priority as P;

        let mut queues = if !block {
            let Some(queues) = self.state.try_recv()? else {
                return Ok(None);
            };
            queues
        } else {
            self.state.recv()?
        };
        if self.reclaim_capacity {
            queues.shrink_oversized();
        }

        let high = P::High.weight() * !queues.high_priority.is_empty() as u32;
        let medium = P::Medium.weight() * !queues.medium_priority.is_empty() as u32;
        let low = P::Low.weight() * !queues.low_priority.is_empty() as u32;
        let mut mass = high + medium + low; //%

        if !queues.high_priority.is_empty() {
            let flip = self.rand.random_ratio(P::High.weight(), mass);
            if flip {
                return Ok(queues.high_priority.pop_front());
            }
            mass -= P::High.weight();
        }

        if !queues.medium_priority.is_empty() {
            let flip = self.rand.random_ratio(P::Medium.weight(), mass);
            if flip {
                return Ok(queues.medium_priority.pop_front());
            }
            mass -= P::Medium.weight();
        }

        if !queues.low_priority.is_empty() {
            let flip = self.rand.random_ratio(P::Low.weight(), mass);
            if flip {
                return Ok(queues.low_priority.pop_front());
            }
        }

        Ok(None)
    }
}

impl<T> Drop for PriorityQueueReceiver<T> {
    fn drop(&mut self) {
        self.state
            .receiver_count
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

#[doc(hidden)]
pub struct Iter<T>(PriorityQueueReceiver<T>);
impl<T> Iterator for Iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop().ok()
    }
}
impl<T> FusedIterator for Iter<T> {}

#[doc(hidden)]
pub struct TryIter<T> {
    receiver: PriorityQueueReceiver<T>,
    ended: bool,
}
impl<T> Iterator for TryIter<T> {
    type Item = Result<T, RecvError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }

        let res = self.receiver.try_pop();
        self.ended = res.is_err();

        res.transpose()
    }
}
impl<T> FusedIterator for TryIter<T> {}

#[cfg(test)]
mod tests {
    use collections::HashSet;

    use super::*;

    #[test]
    fn all_tasks_get_yielded() {
        let (tx, mut rx) = PriorityQueueReceiver::new();
        tx.send(Priority::Medium, 20).unwrap();
        tx.send(Priority::High, 30).unwrap();
        tx.send(Priority::Low, 10).unwrap();
        tx.send(Priority::Medium, 21).unwrap();
        tx.send(Priority::High, 31).unwrap();

        drop(tx);

        assert_eq!(
            rx.iter().collect::<HashSet<_>>(),
            [30, 31, 20, 21, 10].into_iter().collect::<HashSet<_>>()
        )
    }

    #[test]
    fn new_high_prio_task_get_scheduled_quickly() {
        let (tx, mut rx) = PriorityQueueReceiver::new();
        for _ in 0..100 {
            tx.send(Priority::Low, 1).unwrap();
        }

        assert_eq!(rx.pop().unwrap(), 1);
        tx.send(Priority::High, 3).unwrap();
        assert_eq!(rx.pop().unwrap(), 3);
        assert_eq!(rx.pop().unwrap(), 1);
    }

    #[test]
    fn burst_capacity_is_reclaimed_only_when_opted_in() {
        const BURST: usize = 8192;

        let drain = |receiver: &mut PriorityQueueReceiver<usize>| {
            while receiver.try_pop().unwrap().is_some() {}
        };

        let (sender, mut receiver) = PriorityQueueReceiver::new();
        for item in 0..BURST {
            sender.send(Priority::Low, item).unwrap();
        }
        drain(&mut receiver);
        assert!(receiver.low_priority_capacity() >= BURST);

        let (sender, receiver) = PriorityQueueReceiver::new();
        let mut receiver = receiver.reclaiming_capacity();
        for item in 0..BURST {
            sender.send(Priority::Low, item).unwrap();
        }
        drain(&mut receiver);
        assert!(receiver.low_priority_capacity() <= 1024);
    }
}

use super::*;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

#[test]
fn queued_work_survives_dropped_handles_and_senders_in_submission_order() {
    let queue = OrderedWorkQueue::new("ruccl-work-order-test").unwrap();
    let peer = queue.clone();
    let order = Arc::new(Mutex::new(Vec::new()));
    let resource = Arc::new(());
    let weak_resource = Arc::downgrade(&resource);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first_order = order.clone();
    let first = queue
        .submit(move || {
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(TEST_TIMEOUT).unwrap();
            first_order.lock().unwrap().push(0);
            drop(resource);
            Ok::<_, WorkError>(())
        })
        .unwrap();
    started_rx.recv_timeout(TEST_TIMEOUT).unwrap();
    let second_order = order.clone();
    let second = peer
        .submit(move || {
            second_order.lock().unwrap().push(1);
            Ok::<_, WorkError>(17)
        })
        .unwrap();
    drop(first);
    drop(queue);
    drop(peer);
    assert!(weak_resource.upgrade().is_some());
    assert!(!second.is_complete());
    assert!(!second.wait_for(Duration::ZERO).unwrap());
    release_tx.send(()).unwrap();
    assert!(second.wait_for(TEST_TIMEOUT).unwrap());
    assert_eq!(second.wait().unwrap(), 17);
    assert_eq!(*order.lock().unwrap(), vec![0, 1]);
    assert!(weak_resource.upgrade().is_none());
}

#[derive(Debug)]
enum CallerError {
    Work(WorkError),
    Operation(&'static str),
}

impl From<WorkError> for CallerError {
    fn from(error: WorkError) -> Self {
        Self::Work(error)
    }
}

#[test]
fn operation_errors_and_panics_do_not_stop_later_queued_work() {
    let queue = OrderedWorkQueue::new("ruccl-work-error-test").unwrap();
    let operation = queue
        .submit(|| Err::<(), _>(CallerError::Operation("original failure")))
        .unwrap();
    let panic = queue
        .submit(|| -> Result<(), CallerError> { panic!("worker panic") })
        .unwrap();
    let success = queue.submit(|| Ok::<_, CallerError>(23)).unwrap();
    assert!(success.wait_for(TEST_TIMEOUT).unwrap());
    assert!(matches!(
        operation.wait(),
        Err(CallerError::Operation("original failure"))
    ));
    assert!(matches!(
        panic.wait(),
        Err(CallerError::Work(WorkError::WorkerPanic))
    ));
    assert_eq!(success.wait().unwrap(), 23);
}

#[test]
fn spawned_work_can_be_joined_or_detached_without_cancellation() {
    let joined = CollectiveWork::<_, WorkError>::spawn("ruccl-work-join-test", || Ok(31)).unwrap();
    assert!(joined.wait_for(TEST_TIMEOUT).unwrap());
    assert_eq!(joined.wait().unwrap(), 31);

    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let detached = CollectiveWork::<_, WorkError>::spawn("ruccl-work-detach-test", move || {
        release_rx.recv_timeout(TEST_TIMEOUT).unwrap();
        finished_tx.send(37).unwrap();
        Ok(())
    })
    .unwrap();
    drop(detached);
    release_tx.send(()).unwrap();
    assert_eq!(finished_rx.recv_timeout(TEST_TIMEOUT).unwrap(), 37);
}

#[test]
fn closed_queue_reports_failure_without_running_the_task() {
    let (sender, receiver) = mpsc::channel::<OrderedTask>();
    drop(receiver);
    let queue = OrderedWorkQueue { sender };
    let resource = Arc::new(());
    let weak_resource = Arc::downgrade(&resource);
    let result = queue.submit(move || -> Result<(), CallerError> {
        drop(resource);
        panic!("closed queue must not execute the task")
    });
    assert!(matches!(
        result,
        Err(CallerError::Work(WorkError::WorkerQueueClosed))
    ));
    assert!(weak_resource.upgrade().is_none());
}

#[test]
fn poisoned_completion_keeps_original_observation_and_wait_behavior() {
    let completion = Arc::new(WorkCompletion::<(), WorkError> {
        result: Mutex::new(None),
        changed: Condvar::new(),
    });
    let poisoned = completion.clone();
    let _ = catch_unwind(move || {
        let _guard = poisoned.result.lock().unwrap();
        panic!("poison completion state")
    });
    let work = CollectiveWork {
        completion,
        worker: None,
    };
    assert!(work.is_complete());
    assert!(matches!(
        work.wait_for(Duration::ZERO),
        Err(WorkError::Poisoned)
    ));
    assert!(matches!(work.wait(), Err(WorkError::Poisoned)));
}

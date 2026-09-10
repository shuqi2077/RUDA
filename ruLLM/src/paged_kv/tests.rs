use super::*;

fn manager() -> PagedKvCacheManager {
    PagedKvCacheManager::new(PagedKvCacheConfig {
        block_size: 2, num_pages: 8, max_sequence_length: 16,
    }).unwrap()
}

#[test]
fn counters_track_reservation_commit_and_release() {
    let mut cache = manager();
    cache.create_sequence(RequestId(1)).unwrap();
    assert_eq!(cache.sequence_count(), 1);
    let reservation = cache.begin_append(RequestId(1), 3).unwrap();
    assert_eq!(cache.allocated_page_count(), 2);
    assert_eq!(cache.free_page_count(), 6);
    assert_eq!(cache.reservation_count(), 1);
    cache.commit_append(reservation.id).unwrap();
    assert_eq!(cache.committed_tokens(RequestId(1)).unwrap(), 3);
    assert_eq!(cache.reservation_count(), 0);
    cache.remove_sequence(RequestId(1)).unwrap();
    assert_eq!(cache.sequence_count(), 0);
    assert_eq!(cache.free_page_count(), 8);
}

fn reservations(cache: &mut PagedKvCacheManager) -> Vec<KvReservationId> {
    [1, 2].into_iter().map(|id| {
        cache.create_sequence(RequestId(id)).unwrap();
        cache.begin_append(RequestId(id), 3).unwrap().id
    }).collect()
}

#[test]
fn unknown_id_does_not_partially_commit_or_cancel_a_group() {
    let mut cache = manager();
    let ids = reservations(&mut cache);
    let before = cache.snapshot();
    assert!(cache.commit_appends(&[ids[0], KvReservationId(999)]).is_err());
    assert_eq!(cache.snapshot(), before);
    assert!(cache.cancel_appends(&[ids[0], KvReservationId(999)]).is_err());
    assert_eq!(cache.snapshot(), before);
    assert_eq!(cache.reservation_count(), 2);
    assert_eq!(cache.commit_appends(&ids).unwrap(), vec![3, 3]);
}

#[test]
fn duplicate_ids_do_not_partially_commit_or_cancel() {
    let mut cache = manager();
    let ids = reservations(&mut cache);
    let before = cache.snapshot();
    assert!(cache.commit_appends(&[ids[0], ids[0]]).is_err());
    assert!(cache.cancel_appends(&[ids[0], ids[0]]).is_err());
    assert_eq!(cache.snapshot(), before);
    cache.cancel_appends(&ids).unwrap();
    assert_eq!(cache.free_page_count(), 8);
    assert_eq!(cache.committed_tokens(RequestId(1)).unwrap(), 0);
}

#[test]
fn cancellation_keeps_previously_committed_pages() {
    let mut cache = manager();
    cache.create_sequence(RequestId(1)).unwrap();
    let initial = cache.begin_append(RequestId(1), 1).unwrap();
    cache.commit_append(initial.id).unwrap();
    let before = cache.snapshot();
    let append = cache.begin_append(RequestId(1), 4).unwrap();
    cache.cancel_appends(&[append.id]).unwrap();
    assert_eq!(cache.snapshot(), before);
}

#[test]
fn ownership_error_preserves_reservation_for_repair() {
    let mut cache = manager();
    cache.create_sequence(RequestId(1)).unwrap();
    let reservation = cache.begin_append(RequestId(1), 1).unwrap();
    cache.sequences.get_mut(&RequestId(1)).unwrap().pending = None;
    let before = cache.snapshot();
    assert!(cache.commit_append(reservation.id).is_err());
    assert!(cache.cancel_append(reservation.id).is_err());
    assert_eq!(cache.snapshot(), before);
    assert_eq!(cache.reservation_count(), 1);
    cache.sequences.get_mut(&RequestId(1)).unwrap().pending = Some(reservation.id);
    cache.cancel_append(reservation.id).unwrap();
    assert_eq!(cache.free_page_count(), 8);
}

#[test]
fn id_overflow_and_out_of_pages_leave_cache_unchanged() {
    let mut cache = manager();
    cache.create_sequence(RequestId(1)).unwrap();
    cache.next_reservation = u64::MAX;
    let before = cache.snapshot();
    assert!(cache.begin_append(RequestId(1), 2).is_err());
    assert_eq!(cache.snapshot(), before);
    cache.next_reservation = 1;
    let full = cache.begin_append(RequestId(1), 16).unwrap();
    cache.commit_append(full.id).unwrap();
    cache.create_sequence(RequestId(2)).unwrap();
    let before = cache.snapshot();
    assert!(!cache.can_append(RequestId(2), 1).unwrap());
    assert!(cache.begin_append(RequestId(2), 1).is_err());
    assert_eq!(cache.snapshot(), before);
}

#[test]
fn empty_group_operations_are_noops() {
    let mut cache = manager();
    let before = cache.snapshot();
    assert!(cache.commit_appends(&[]).unwrap().is_empty());
    cache.cancel_appends(&[]).unwrap();
    assert_eq!(cache.snapshot(), before);
}

#[test]
fn repeated_cycles_never_duplicate_or_lose_pages() {
    let mut cache = manager();
    for round in 1..=128 {
        let id = RequestId(round);
        cache.create_sequence(id).unwrap();
        let first = cache.begin_append(id, 3).unwrap();
        cache.commit_append(first.id).unwrap();
        let second = cache.begin_append(id, 6).unwrap();
        cache.cancel_append(second.id).unwrap();
        assert_eq!(cache.block_table(id).unwrap(), first.block_table);
        cache.remove_sequence(id).unwrap();
        assert_eq!(cache.free_page_count(), 8);
        assert_eq!(cache.reservation_count(), 0);
    }
}

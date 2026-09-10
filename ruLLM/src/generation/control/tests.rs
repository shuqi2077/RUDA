use super::*;

#[test]
fn stop_sequences_validate_empty_and_out_of_range_patterns() {
    for stop_token_sequences in [vec![vec![]], vec![vec![-1]], vec![vec![8]]] {
        assert!(GenerationControl { stop_token_sequences, cancellation: None }.validate(8).is_err());
    }
    assert!(GenerationControl::default().validate(8).is_ok());
}

#[test]
fn cancellation_is_shared_and_one_shot() {
    let token = GenerationCancellation::new();
    let other = token.clone();
    assert!(!token.is_cancelled());
    std::thread::spawn(move || other.cancel()).join().unwrap();
    assert!(token.is_cancelled());
    token.cancel();
    assert!(token.is_cancelled());
}

#[test]
fn matcher_handles_overlap_and_simultaneous_matches() {
    let patterns = vec![vec![1, 2, 1], vec![2, 1], vec![1]];
    let mut matcher = StopSequenceMatcher::new(&patterns);
    let matches: Vec<_> = [1, 2, 1, 2, 1].into_iter().map(|id| matcher.push(id)).collect();
    assert_eq!(matches, [Some(2), None, Some(0), None, Some(0)]);
}

#[test]
fn matcher_handles_repeated_prefixes_and_mismatches() {
    let patterns = vec![vec![1, 1, 1, 2]];
    let mut matcher = StopSequenceMatcher::new(&patterns);
    for token in [1, 1, 1, 1, 1, 3, 1, 1, 1] { assert_eq!(matcher.push(token), None); }
    assert_eq!(matcher.push(2), Some(0));
}

#[test]
fn empty_matcher_never_stops() {
    let mut matcher = StopSequenceMatcher::new(&[]);
    for token in 0..100 { assert_eq!(matcher.push(token), None); }
}

#[test]
fn incremental_matcher_matches_naive_suffix_search() {
    let mut state = 7_u64;
    let mut next = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1); (state >> 32) as usize };
    for _ in 0..128 {
        let patterns: Vec<Vec<i32>> = (0..6).map(|_| {
            (0..1 + next() % 8).map(|_| (next() % 4) as i32).collect()
        }).collect();
        let mut matcher = StopSequenceMatcher::new(&patterns);
        let mut history = Vec::new();
        for _ in 0..256 {
            let token = (next() % 4) as i32;
            history.push(token);
            let expected = patterns.iter().position(|pattern| history.ends_with(pattern));
            assert_eq!(matcher.push(token), expected);
        }
    }
}

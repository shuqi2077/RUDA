//! Immutable physical-page partition for ordered attention backward.
//! Only pages reachable from requests WITH queries are active. Effective KV
//! lengths (not table capacity) define reachability. Causal pruning is left to
//! the original kernel: this is a safe superset for causal and noncausal calls.
use super::{HostPlan, PagedAttentionError};

pub(super) const ENV: &str = "RUDA_PAGED_ORDERED_COMPACT_HISTORY";

pub(super) fn parse(value: Option<&str>) -> Result<bool, PagedAttentionError> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(PagedAttentionError("RUDA_PAGED_ORDERED_COMPACT_HISTORY must be 0 or 1")),
    }
}

/// Sample once, when a workspace is constructed, BEFORE any GPU allocation.
pub(super) fn from_env() -> Result<bool, PagedAttentionError> {
    let value = std::env::var_os(ENV);
    match value.as_ref() {
        None => parse(None),
        Some(value) => parse(Some(value.to_str().ok_or(PagedAttentionError(
            "RUDA_PAGED_ORDERED_COMPACT_HISTORY must be valid UTF-8",
        ))?)),
    }
}

/// `pages[..active]` are ascending reachable physical pages; the remainder are
/// ascending inactive pages. Together they form a disjoint permutation of ALL
/// physical pages. Each final gradient element has exactly one writer.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct PagePartition {
    pub pages: Vec<u32>,
    pub active: usize,
}

pub(super) fn build(host: &HostPlan, max_words: usize) -> Result<PagePartition, PagedAttentionError> {
    // Bound the retained device map BEFORE constructing page-sized host arrays.
    if host.pages > max_words || host.pages > u32::MAX as usize {
        return Err(PagedAttentionError("ordered page compaction exceeds workspace budget"));
    }
    let mut queried = vec![false; host.sequences];
    for &sequence in &host.words[..host.queries] { queried[sequence as usize] = true; }
    let mut used = vec![false; host.pages];
    for (sequence, &has_queries) in queried.iter().enumerate() {
        if !has_queries { continue; }
        let length = host.words[2 * host.queries + sequence] as usize;
        let base = 2 * host.queries + host.sequences + sequence * host.table_width;
        for logical in 0..length.div_ceil(host.page_size) {
            used[host.words[base + logical] as usize] = true;
        }
    }
    let mut pages = Vec::with_capacity(host.pages);
    for (page, &active) in used.iter().enumerate() { if active { pages.push(page as u32); } }
    let active = pages.len();
    for (page, &active) in used.iter().enumerate() { if !active { pages.push(page as u32); } }
    Ok(PagePartition { pages, active })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(tables: &[Vec<u32>], lengths: &[u32], ids: &[u32], pos: &[u32]) -> HostPlan {
        HostPlan::new(4, 8, tables, lengths, ids, pos).unwrap()
    }
    fn check(host: &HostPlan) -> PagePartition {
        let p = build(host, host.pages).unwrap();
        let mut sorted = p.pages.clone(); sorted.sort_unstable();
        assert_eq!(sorted, (0..host.pages as u32).collect::<Vec<_>>());
        assert!(p.pages[..p.active].windows(2).all(|w| w[0] < w[1]));
        assert!(p.pages[p.active..].windows(2).all(|w| w[0] < w[1]));
        p
    }
    #[test] fn defaults_off() { assert_eq!(parse(None), Ok(false)); assert_eq!(parse(Some("0")), Ok(false)); }
    #[test] fn explicit_on() { assert_eq!(parse(Some("1")), Ok(true)); }
    #[test] fn invalid_switch_rejected() { for s in ["", "true", "01", " 1", "2"] { assert!(parse(Some(s)).is_err()); } }
    #[test] fn sparse_nonidentity() {
        let h=plan(&[vec![6,2]], &[5], &[0], &[4]); let p=check(&h);
        assert_eq!(p.active,2); assert_eq!(p.pages,vec![2,6,0,1,3,4,5,7]);
    }
    #[test] fn table_capacity_not_effective_length() {
        let h=plan(&[vec![6,2,7]], &[4], &[0], &[3]); let p=check(&h);
        assert_eq!(&p.pages[..p.active], &[6]);
    }
    #[test] fn shared_page_is_unique() {
        let h=plan(&[vec![6,2],vec![6,4]], &[5,6], &[0,1], &[4,5]); let p=check(&h);
        assert_eq!(&p.pages[..p.active], &[2,4,6]);
    }
    #[test] fn no_query_request_omitted() {
        let h=plan(&[vec![6,2],vec![1,4]], &[5,6], &[0], &[4]); let p=check(&h);
        assert_eq!(&p.pages[..p.active], &[2,6]);
    }
    #[test] fn empty_queries_all_inactive() {
        let h=plan(&[vec![6,2]], &[5], &[], &[]); assert_eq!(check(&h).active,0);
    }
    #[test] fn empty_history_nonempty_queries() {
        let h=plan(&[vec![]], &[0], &[0,0], &[0,17]); assert_eq!(check(&h).active,0);
    }
    #[test] fn future_causal_pages_retained_for_noncausal_reuse() {
        let h=plan(&[vec![6,2]], &[8], &[0], &[0]); assert_eq!(check(&h).active,2);
    }
    #[test] fn full_occupancy_identity() {
        let h=HostPlan::new(4,2,&[vec![1,0]],&[8],&[0],&[0]).unwrap(); let p=check(&h);
        assert_eq!(p.active,2); assert_eq!(p.pages,vec![0,1]);
    }
    #[test] fn budget_exact_and_one_short() {
        let h=plan(&[vec![6]], &[1], &[0], &[0]); assert!(build(&h,8).is_ok()); assert!(build(&h,7).is_err());
    }
    #[test] fn position_order_does_not_change_partition() {
        let a=plan(&[vec![6,2]], &[8], &[0,0,0], &[0,7,2]);
        let b=plan(&[vec![6,2]], &[8], &[0,0,0], &[7,0,2]); assert_eq!(check(&a),check(&b));
    }
    #[test] fn randomized_partition_matches_independent_reachability() {
        let mut seed=19u32;
        for _ in 0..200 {
            seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let len=seed%9; let ids=if seed&1==0 {vec![0]} else {vec![0,1]};
            let pos=vec![0;ids.len()];
            let h=plan(&[vec![6,2],vec![1,6]], &[len,len], &ids, &pos);
            let p=check(&h); let mut expected=std::collections::BTreeSet::new();
            for &s in &ids { for token in 0..len as usize {
                let page=h.words[2*h.queries+h.sequences+s as usize*h.table_width+token/4]; expected.insert(page);
            }}
            assert_eq!(p.pages[..p.active],expected.into_iter().collect::<Vec<_>>());
        }
    }
}

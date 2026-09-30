//! Stable inverse scheduling and visibility bounds for ordered backward.
//! Query rows remain in their original packed order. The extra integer metadata
//! lets the GPU skip provably invisible queries WITHOUT changing sum order.
use super::{HostPlan, PagedAttentionError};

/// A fallback block contains consecutive rows of ONE sequence, not global rows.
/// Keep this a compile-time kernel argument so host and device cannot disagree.
pub(super) const QUERY_BLOCK_ROWS: usize = 32;

pub(super) fn build(host: &HostPlan, max_words: usize) -> Result<Vec<u32>, PagedAttentionError> {
    // Existing prefix: header(3), page offsets(P+1), links(2*E),
    // sequence offsets(S+1), row ids(Q). New suffix, starting at query_base+Q:
    // monotonic flags(S), maximum positions(S), block offsets(S+1),
    // block maxima(sum ceil(rows_in_sequence / QUERY_BLOCK_ROWS)).
    // Check the fixed size BEFORE allocating page/sequence-sized vectors.
    let fixed = host.pages.checked_add(4)
        .and_then(|n| host.sequences.checked_mul(4).and_then(|s| n.checked_add(s)))
        .and_then(|n| n.checked_add(2))
        .and_then(|n| n.checked_add(host.queries))
        .filter(|&n| n <= max_words && n <= u32::MAX as usize)
        .ok_or(PagedAttentionError("ordered history metadata exceeds workspace budget"))?;
    let mut counts = vec![0usize; host.sequences];
    for row in 0..host.queries { counts[host.words[row] as usize] += 1; }
    let mut entries = 0usize;
    let mut blocks = 0usize;
    for (s, &count) in counts.iter().enumerate() {
        // A history with no query contributes nothing even if it owns pages.
        if count != 0 {
            let length = host.words[2 * host.queries + s] as usize;
            entries = entries.checked_add(length.div_ceil(host.page_size))
                .ok_or(PagedAttentionError("ordered history index overflow"))?;
        }
        blocks = blocks.checked_add(count.div_ceil(QUERY_BLOCK_ROWS))
            .ok_or(PagedAttentionError("ordered visibility index overflow"))?;
    }
    let total = entries.checked_mul(2).and_then(|n| fixed.checked_add(n))
        .and_then(|n| n.checked_add(blocks))
        .filter(|&n| n <= max_words && n <= u32::MAX as usize)
        .ok_or(PagedAttentionError("ordered history metadata exceeds workspace budget"))?;
    let entries_base = 4 + host.pages;
    let sequence_base = entries_base + 2 * entries;
    let query_base = sequence_base + host.sequences + 1;
    let visibility_base = query_base + host.queries;
    let block_offsets_base = visibility_base + 2 * host.sequences;
    let block_maxima_base = block_offsets_base + host.sequences + 1;
    debug_assert_eq!(block_maxima_base + blocks, total);

    let mut links = Vec::with_capacity(entries);
    for (s, &count) in counts.iter().enumerate() {
        if count == 0 { continue; }
        let length = host.words[2 * host.queries + s] as usize;
        for logical in 0..length.div_ceil(host.page_size) {
            let page = host.words[2 * host.queries + host.sequences + s * host.table_width + logical];
            links.push((page, s as u32, logical as u32));
        }
    }
    links.sort_unstable(); // same (page, sequence, logical) total order as v33
    let mut rows: Vec<_> = (0..host.queries)
        .map(|row| (host.words[row], row as u32)).collect();
    rows.sort_unstable(); // do NOT sort by position: preserve floating sum order
    let mut out = vec![0u32; total];
    out[0] = entries_base as u32; out[1] = sequence_base as u32; out[2] = query_base as u32;
    for &(page, _, _) in &links { out[3 + page as usize + 1] += 1; }
    for page in 0..host.pages { out[3 + page + 1] += out[3 + page]; }
    for (i, &(_, sequence, logical)) in links.iter().enumerate() {
        out[entries_base + 2 * i] = sequence;
        out[entries_base + 2 * i + 1] = logical;
    }
    for &(sequence, _) in &rows { out[sequence_base + sequence as usize + 1] += 1; }
    for sequence in 0..host.sequences {
        out[sequence_base + sequence + 1] += out[sequence_base + sequence];
    }
    for (i, &(_, row)) in rows.iter().enumerate() { out[query_base + i] = row; }
    let mut block_cursor = 0usize;
    for sequence in 0..host.sequences {
        let begin = out[sequence_base + sequence] as usize;
        let end = out[sequence_base + sequence + 1] as usize;
        let mut monotonic = true;
        let mut previous = 0u32;
        let mut maximum = 0u32;
        out[block_offsets_base + sequence] = block_cursor as u32;
        for i in begin..end {
            let pos = host.words[host.queries + out[query_base + i] as usize];
            if i != begin && pos < previous { monotonic = false; }
            previous = pos; maximum = maximum.max(pos);
            let block = block_cursor + (i - begin) / QUERY_BLOCK_ROWS;
            out[block_maxima_base + block] = out[block_maxima_base + block].max(pos);
        }
        out[visibility_base + sequence] = u32::from(monotonic);
        out[visibility_base + host.sequences + sequence] = maximum;
        block_cursor += (end - begin).div_ceil(QUERY_BLOCK_ROWS);
    }
    out[block_offsets_base + host.sequences] = block_cursor as u32;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    // This evaluator consumes the EXACT production metadata, independently of
    // the GPU DSL, and checks that only invisible terms are skipped.
    fn selected(host: &HostPlan, index: &[u32], seq: usize, token: u32, causal: bool) -> Vec<u32> {
        let sb=index[1] as usize; let qb=index[2] as usize;
        let vb=qb+host.queries; let ob=vb+2*host.sequences; let mb=ob+host.sequences+1;
        let begin=index[sb+seq] as usize; let end=index[sb+seq+1] as usize;
        let mono=index[vb+seq]!=0;
        if causal && token>index[vb+host.sequences+seq] { return vec![]; }
        let mut cursor=begin;
        if causal && mono && cursor<end && host.words[host.queries+index[qb+cursor] as usize]<token {
            let mut right=end;
            while cursor<right {
                let mid=cursor+(right-cursor)/2;
                if host.words[host.queries+index[qb+mid] as usize]<token {cursor=mid+1;} else {right=mid;}
            }
        }
        let mut out=vec![];
        while cursor<end {
            let mut stop=end;
            if causal && !mono {
                let b=(cursor-begin)/QUERY_BLOCK_ROWS;
                stop=end.min(begin+(b+1)*QUERY_BLOCK_ROWS);
                if index[mb+index[ob+seq] as usize+b]<token {cursor=stop;continue;}
            }
            while cursor<stop {
                let row=index[qb+cursor];
                if !causal || mono || host.words[host.queries+row as usize]>=token {out.push(row);}
                cursor+=1;
            }
        }
        out
    }
    #[test] fn shared_pages_and_unsorted_rows() {
        let h=HostPlan::new(2,4,&[vec![2,0],vec![2,3]],&[3,4],&[1,0,1],&[3,2,1]).unwrap();
        let x=build(&h,100).unwrap();
        assert_eq!(&x[3..8],&[0,1,1,3,4]);
        let e=x[0] as usize; assert_eq!(&x[e..e+8],&[0,1,0,0,1,0,1,1]);
        let q=x[2] as usize; assert_eq!(&x[q..q+h.queries],&[1,0,2]);
        assert_eq!(selected(&h,&x,1,2,true),vec![0]);
    }
    #[test] fn unused_pages_and_empty_queries() {
        let h=HostPlan::new(4,3,&[vec![]],&[0],&[],&[]).unwrap();
        let x=build(&h,100).unwrap(); assert_eq!(&x[3..7],&[0,0,0,0]);
        assert!(selected(&h,&x,0,0,true).is_empty());
    }
    #[test] fn size_guard_before_allocation() {
        let h=HostPlan::new(2,100,&[vec![0]],&[1],&[0],&[0]).unwrap();
        assert!(build(&h,10).is_err());
    }
    #[test] fn exact_budget_accepts_one_less_rejects() {
        let h=HostPlan::new(4,1,&[vec![0]],&[4],&[0,0],&[1,3]).unwrap();
        let x=build(&h,100).unwrap(); assert_eq!(build(&h,x.len()).unwrap(),x);
        assert!(build(&h,x.len()-1).is_err());
    }
    #[test] fn no_query_sequence_has_no_page_links() {
        let h=HostPlan::new(4,2,&[vec![0],vec![1]],&[4,4],&[0],&[2]).unwrap();
        let x=build(&h,100).unwrap(); assert_eq!(&x[3..6],&[0,1,1]);
    }
    #[test] fn monotonic_duplicates_include_boundary() {
        let h=HostPlan::new(8,1,&[vec![0]],&[8],&[0;5],&[0,2,2,6,7]).unwrap();
        let x=build(&h,100).unwrap();
        assert_eq!(selected(&h,&x,0,2,true),vec![1,2,3,4]);
        assert!(selected(&h,&x,0,8,true).is_empty());
        assert_eq!(selected(&h,&x,0,8,false),vec![0,1,2,3,4]);
    }
    #[test] fn unsorted_blocks_keep_original_accumulation_order() {
        let mut positions:Vec<u32>=(0..97).map(|x| ((x*7)%97) as u32).collect();
        positions[32..64].fill(0);
        let h=HostPlan::new(128,1,&[vec![0]],&[128],&vec![0;97],&positions).unwrap();
        let x=build(&h,1000).unwrap();
        for token in 0..130 {
            let expected:Vec<_>=positions.iter().enumerate().filter(|(_,p)| **p>=token).map(|(i,_)|i as u32).collect();
            assert_eq!(selected(&h,&x,0,token,true),expected);
        }
    }
    #[test] fn randomized_visibility_keeps_exact_row_sequence() {
        let mut seed=0x9e3779b9u32;
        let mut rand=|| {seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);seed};
        for _ in 0..300 {
            let n=(rand()%180) as usize;
            let ids:Vec<_>=(0..n).map(|_|rand()%3).collect();
            let positions:Vec<_>=(0..n).map(|_|rand()%99).collect();
            let h=HostPlan::new(128,3,&[vec![0],vec![0],vec![2]],&[99;3],&ids,&positions).unwrap();
            let x=build(&h,10000).unwrap();
            for seq in 0..3 { for token in [0,1,31,32,63,98,99,u32::MAX] { for causal in [true,false] {
                let expected:Vec<_>=(0..n).filter(|&r|ids[r]==seq as u32 && (!causal||positions[r]>=token)).map(|r|r as u32).collect();
                assert_eq!(selected(&h,&x,seq,token,causal),expected);
            }}}
        }
    }
}

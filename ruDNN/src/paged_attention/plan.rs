//! Validated host scheduling metadata. Token/K/V data never crosses to the host.
use super::PagedAttentionError;
use std::collections::BTreeSet;

/// Immutable metadata for packed queries and a paged device cache.
/// Cache layout is [physical_pages, page_size, KV_heads, feature].
/// Metadata is packed as [sequence_ids, query_positions, kv_lengths, block_table].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPlan {
    pub(crate) words: Vec<u32>,
    pub(crate) queries: usize,
    pub(crate) sequences: usize,
    pub(crate) table_width: usize,
    pub(crate) page_size: usize,
    pub(crate) pages: usize,
}

impl HostPlan {
    /// Plan right-aligned prefill/decode from FlashInfer-style CSR page tables.
    /// All arguments are scheduler metadata, not device KV/logit readback.
    /// `query_indptr` partitions packed queries and each query chunk ends at
    /// the corresponding committed KV length.
    pub fn from_csr(
        page_size: usize,
        pages: usize,
        page_indptr: &[u32],
        page_indices: &[u32],
        last_page_lengths: &[u32],
        query_indptr: &[u32],
    ) -> Result<Self, PagedAttentionError> {
        let sequences = last_page_lengths.len();
        if page_size == 0
            || page_size > u32::MAX as usize
            || sequences == 0
            || page_indptr.len() != sequences + 1
            || query_indptr.len() != sequences + 1
            || page_indptr[0] != 0
            || query_indptr[0] != 0
            || page_indices.len() > u32::MAX as usize
            || page_indptr.last().copied() != Some(page_indices.len() as u32)
            || page_indptr.windows(2).any(|v| v[0] > v[1])
            || query_indptr.windows(2).any(|v| v[0] > v[1])
        {
            return Err(PagedAttentionError("invalid CSR page/query metadata"));
        }
        let query_count = *query_indptr.last().unwrap() as usize;
        if query_count
            .checked_mul(2)
            .and_then(|n| n.checked_add(sequences))
            .filter(|&n| n <= u32::MAX as usize)
            .is_none()
        {
            return Err(PagedAttentionError(
                "CSR queries exceed packed metadata indexing",
            ));
        }
        let mut tables = Vec::with_capacity(sequences);
        let mut lengths = Vec::with_capacity(sequences);
        let mut ids = Vec::new();
        let mut positions = Vec::new();
        for seq in 0..sequences {
            let start = page_indptr[seq] as usize;
            let end = page_indptr[seq + 1] as usize;
            let last = last_page_lengths[seq] as usize;
            if (start == end && last != 0) || (start != end && (last == 0 || last > page_size)) {
                return Err(PagedAttentionError("invalid last-page length"));
            }
            let length = if start == end {
                0
            } else {
                (end - start - 1)
                    .checked_mul(page_size)
                    .and_then(|n| n.checked_add(last))
                    .filter(|&n| n <= u32::MAX as usize)
                    .ok_or(PagedAttentionError("CSR KV length overflow"))?
            };
            let queries = (query_indptr[seq + 1] - query_indptr[seq]) as usize;
            if queries > length {
                return Err(PagedAttentionError(
                    "query chunk exceeds committed KV history",
                ));
            }
            tables.push(page_indices[start..end].to_vec());
            lengths.push(length as u32);
            for pos in length - queries..length {
                ids.push(seq as u32);
                positions.push(pos as u32);
            }
        }
        Self::new(page_size, pages, &tables, &lengths, &ids, &positions)
    }
    /// `positions` are absolute zero-based positions in each request's history,
    /// NOT local positions inside the prefill chunk. Empty requests are legal.
    /// Shared physical pages are legal for reads; writes require unique slots.
    pub fn new(
        page_size: usize,
        pages: usize,
        tables: &[Vec<u32>],
        lengths: &[u32],
        sequence_ids: &[u32],
        positions: &[u32],
    ) -> Result<Self, PagedAttentionError> {
        let fail = |s| Err(PagedAttentionError(s));
        if page_size == 0
            || pages == 0
            || tables.is_empty()
            || tables.len() != lengths.len()
            || sequence_ids.len() != positions.len()
            || page_size > u32::MAX as usize
            || pages > u32::MAX as usize
        {
            return fail("invalid paged plan dimensions");
        }
        let width = tables.iter().map(Vec::len).max().unwrap().max(1);
        let count = sequence_ids
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(tables.len()))
            .and_then(|n| {
                width
                    .checked_mul(tables.len())
                    .and_then(|m| n.checked_add(m))
            })
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or(PagedAttentionError("paged plan exceeds U32 indexing"))?;
        for (table, &length) in tables.iter().zip(lengths) {
            let required = (length as usize).div_ceil(page_size);
            if table.len() < required || table.iter().any(|&p| p as usize >= pages) {
                return fail("missing/out-of-range physical page");
            }
            // Repeated pages in one history generally indicate a corrupt table.
            let unique: BTreeSet<_> = table[..required].iter().copied().collect();
            if unique.len() != required {
                return fail("duplicate page in one history");
            }
        }
        for (&seq, &pos) in sequence_ids.iter().zip(positions) {
            if seq as usize >= tables.len() {
                return fail("query sequence is out of range");
            }
            let length = lengths[seq as usize];
            if length != 0 && pos >= length {
                return fail("query position exceeds effective KV length");
            }
        }
        let mut words = Vec::with_capacity(count);
        words.extend_from_slice(sequence_ids);
        words.extend_from_slice(positions);
        words.extend_from_slice(lengths);
        for table in tables {
            words.extend_from_slice(table);
            words.resize(words.len() + width - table.len(), u32::MAX);
        }
        Ok(Self {
            words,
            queries: sequence_ids.len(),
            sequences: tables.len(),
            table_width: width,
            page_size,
            pages,
        })
    }

    /// Decode a bounded in-process ABI payload; validates it before any upload.
    pub fn from_packed(
        page_size: usize,
        pages: usize,
        sequences: usize,
        queries: usize,
        table_width: usize,
        words: &[u32],
    ) -> Result<Self, PagedAttentionError> {
        let expected = queries
            .checked_mul(2)
            .and_then(|n| n.checked_add(sequences))
            .and_then(|n| {
                sequences
                    .checked_mul(table_width)
                    .and_then(|m| n.checked_add(m))
            });
        if expected != Some(words.len()) || sequences == 0 || table_width == 0 || page_size == 0 {
            return Err(PagedAttentionError("invalid packed paged metadata length"));
        }
        let lengths = &words[2 * queries..2 * queries + sequences];
        let mut tables = Vec::with_capacity(sequences);
        for (seq, &length) in lengths.iter().enumerate() {
            let needed = (length as usize).div_ceil(page_size);
            if needed > table_width {
                return Err(PagedAttentionError("KV length exceeds page table"));
            }
            let start = 2 * queries + sequences + seq * table_width;
            tables.push(words[start..start + needed].to_vec());
        }
        Self::new(
            page_size,
            pages,
            &tables,
            lengths,
            &words[..queries],
            &words[queries..2 * queries],
        )
    }

    pub fn query_count(&self) -> usize {
        self.queries
    }
    pub fn page_count(&self) -> usize {
        self.pages
    }
    pub fn page_size(&self) -> usize {
        self.page_size
    }
    pub fn sequence_count(&self) -> usize {
        self.sequences
    }

    /// Required for appending token K/V: duplicate physical writes are rejected.
    /// Read-only shared-prefix pages must be copied by the scheduler before writes.
    pub fn validate_writes(&self) -> Result<(), PagedAttentionError> {
        let mut slots = BTreeSet::new();
        let mut owners = std::collections::BTreeMap::<u32, usize>::new();
        for seq in 0..self.sequences {
            let length = self.words[2 * self.queries + seq] as usize;
            for logical in 0..length.div_ceil(self.page_size) {
                let page = self.words
                    [2 * self.queries + self.sequences + seq * self.table_width + logical];
                *owners.entry(page).or_default() += 1;
            }
        }
        for row in 0..self.queries {
            let seq = self.words[row] as usize;
            let pos = self.words[self.queries + row] as usize;
            let len = self.words[2 * self.queries + seq] as usize;
            if pos >= len {
                return Err(PagedAttentionError(
                    "append requires a committed valid position",
                ));
            }
            let p = self.words
                [2 * self.queries + self.sequences + seq * self.table_width + pos / self.page_size]
                as usize;
            if owners.get(&(p as u32)).copied().unwrap_or(0) != 1 {
                return Err(PagedAttentionError(
                    "shared prefix page requires scheduler copy-on-write before append",
                ));
            }
            let slot = p
                .checked_mul(self.page_size)
                .and_then(|n| n.checked_add(pos % self.page_size))
                .ok_or(PagedAttentionError("physical slot overflow"))?;
            if !slots.insert(slot) {
                return Err(PagedAttentionError("duplicate physical cache write"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csr_prefill_and_decode_are_right_aligned() {
        let p = HostPlan::from_csr(4, 6, &[0, 2, 3], &[3, 0, 5], &[3, 2], &[0, 2, 3]).unwrap();
        let expected = HostPlan::new(
            4,
            6,
            &[vec![3, 0], vec![5]],
            &[7, 2],
            &[0, 0, 1],
            &[5, 6, 1],
        )
        .unwrap();
        assert_eq!(p, expected);
        p.validate_writes().unwrap();
    }
    #[test]
    fn csr_rejects_invalid_offsets_and_last_pages() {
        assert!(HostPlan::from_csr(4, 1, &[0, 1], &[0], &[0], &[0, 1]).is_err());
        assert!(HostPlan::from_csr(4, 1, &[0, 0], &[], &[1], &[0, 0]).is_err());
        assert!(HostPlan::from_csr(4, 1, &[0, 1], &[0], &[2], &[0, 3]).is_err());
        assert!(HostPlan::from_csr(4, 1, &[0, 2], &[0], &[2], &[0, 1]).is_err());
    }
    #[test]
    fn ragged_chunk_absolute_positions() {
        let p = HostPlan::new(
            4,
            6,
            &[vec![3, 0], vec![5]],
            &[7, 2],
            &[0, 0, 1],
            &[5, 6, 1],
        )
        .unwrap();
        assert_eq!(p.query_count(), 3);
        p.validate_writes().unwrap();
    }
    #[test]
    fn missing_page_rejected() {
        assert!(HostPlan::new(4, 2, &[vec![0]], &[5], &[0], &[4]).is_err());
    }
    #[test]
    fn shared_reads_but_duplicate_writes_rejected() {
        let p = HostPlan::new(4, 2, &[vec![0], vec![0]], &[3, 3], &[0, 1], &[2, 2]).unwrap();
        assert!(p.validate_writes().is_err());
    }
    #[test]
    fn empty_attention_is_legal_but_append_is_not() {
        let p = HostPlan::new(16, 1, &[vec![]], &[0], &[0], &[0]).unwrap();
        assert!(p.validate_writes().is_err());
    }
}

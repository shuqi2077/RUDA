use rusparse::{CooMatrix, IndexBase};

const VALUE_BITS: [u32; 8] = [
    0,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0123,
    0x7f80_0001,
    1,
    0xff7f_ffff,
];

fn compare_with_stable_sort(rows: usize, columns: usize, row_ids: &[u32], base: IndexBase) {
    let offset = match base {
        IndexBase::Zero => 0,
        IndexBase::One => 1,
    };
    let encoded_rows: Vec<_> = row_ids.iter().map(|row| row + offset).collect();
    let column_ids: Vec<_> = (0..row_ids.len())
        .map(|entry| ((entry * 7 + 3) % columns) as u32 + offset)
        .collect();
    let values: Vec<_> = (0..row_ids.len())
        .map(|entry| f32::from_bits(VALUE_BITS[entry % VALUE_BITS.len()]))
        .collect();
    let input = CooMatrix::new(rows, columns, &encoded_rows, &column_ids, &values, base).unwrap();
    let output = input.to_csr().unwrap();
    let output = output.as_ref();
    let mut order: Vec<_> = (0..row_ids.len()).collect();
    order.sort_by_key(|&entry| row_ids[entry]);
    let mut offsets = vec![offset];
    let mut position = 0;
    for row in 0..rows {
        while position < order.len() && row_ids[order[position]] as usize == row {
            position += 1;
        }
        offsets.push(position as u32 + offset);
    }
    assert_eq!(
        (output.rows(), output.columns(), output.index_base()),
        (rows, columns, base)
    );
    assert_eq!(output.row_offsets(), offsets);
    assert_eq!(
        output.column_indices(),
        order.iter().map(|&i| column_ids[i]).collect::<Vec<_>>()
    );
    assert_eq!(
        output
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        order
            .iter()
            .map(|&i| values[i].to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(input.row_indices(), encoded_rows);
    assert_eq!(input.column_indices(), column_ids);
    assert_eq!(
        input
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn empty_shapes_and_empty_rows() {
    for base in [IndexBase::Zero, IndexBase::One] {
        for (rows, columns) in [(0, 0), (0, 4), (7, 0), (7, 4)] {
            compare_with_stable_sort(rows, columns, &[], base);
        }
        compare_with_stable_sort(9, 3, &[7, 1, 7, 1, 7, 1, 1, 7], base);
    }
}

#[test]
fn duplicate_order_and_special_value_bits() {
    for base in [IndexBase::Zero, IndexBase::One] {
        let rows = [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0];
        compare_with_stable_sort(4, 1, &rows, base);
        compare_with_stable_sort(1, 1, &[0; 16], base);
    }
}

#[test]
fn exhaustive_small_row_orders() {
    for base in [IndexBase::Zero, IndexBase::One] {
        for rows in 1usize..=4 {
            for length in 0u32..=6 {
                for mut code in 0..rows.pow(length) {
                    let ids: Vec<_> = (0..length)
                        .map(|_| {
                            let row = (code % rows) as u32;
                            code /= rows;
                            row
                        })
                        .collect();
                    compare_with_stable_sort(rows, 3, &ids, base);
                }
            }
        }
    }
}

#[test]
fn large_random_sorted_reverse_and_sparse_rows() {
    let mut seed = 0x91ab_27c3u32;
    let ids: Vec<_> = (0..65_537)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed % 1024
        })
        .collect();
    for base in [IndexBase::Zero, IndexBase::One] {
        let mut ids = ids.clone();
        compare_with_stable_sort(1024, 19, &ids, base);
        compare_with_stable_sort(100_003, 19, &[100_002, 0, 50_000, 0], base);
        ids.sort();
        compare_with_stable_sort(1024, 19, &ids, base);
        ids.reverse();
        compare_with_stable_sort(1024, 19, &ids, base);
    }
}

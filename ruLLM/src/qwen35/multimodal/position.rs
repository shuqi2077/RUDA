use crate::GenerationError;

pub(super) struct ImageSpan {
    pub batch: usize,
    pub start: usize,
    pub count: usize,
    pub feature_start: usize,
}

pub(super) struct ImagePositions {
    pub coordinates: Vec<[usize; 3]>,
    pub next: Vec<usize>,
    pub spans: Vec<ImageSpan>,
}

/// Unpadded, equal-length rows. Grids follow image-placeholder order across rows.
pub(super) fn image_positions(
    tokens: &[Vec<i32>],
    image_token: i32,
    grids: &[[usize; 3]],
    merge: usize,
) -> Result<ImagePositions, GenerationError> {
    let fail = || GenerationError("image placeholders, grids or batch shape do not match".into());
    let sequence = tokens.first().map_or(0, Vec::len);
    if sequence == 0 || merge == 0 || tokens.iter().any(|row| row.len() != sequence) {
        return Err(fail());
    }
    let mut result = ImagePositions { coordinates: Vec::new(), next: Vec::new(), spans: Vec::new() };
    let mut image = 0;
    let mut feature_start = 0usize;
    for (batch, row) in tokens.iter().enumerate() {
        let mut token = 0;
        let mut position = 0usize;
        while token < sequence {
            if row[token] != image_token {
                result.coordinates.push([position; 3]);
                position = position.checked_add(1).ok_or_else(fail)?;
                token += 1;
                continue;
            }
            let &[t, h, w] = grids.get(image).ok_or_else(fail)?;
            if t != 1 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(fail());
            }
            let (h, w) = (h / merge, w / merge);
            let count = h.checked_mul(w).ok_or_else(fail)?;
            let end = token.checked_add(count).filter(|&end| end <= sequence).ok_or_else(fail)?;
            if row[token..end].iter().any(|&id| id != image_token)
                || row.get(end) == Some(&image_token)
            {
                return Err(fail());
            }
            let next = position.checked_add(h.max(w)).ok_or_else(fail)?;
            for y in 0..h {
                for x in 0..w {
                    result.coordinates.push([position, position + y, position + x]);
                }
            }
            result.spans.push(ImageSpan { batch, start: token, count, feature_start });
            feature_start = feature_start.checked_add(count).ok_or_else(fail)?;
            token = end;
            position = next;
            image += 1;
        }
        result.next.push(position);
    }
    if image != grids.len() { return Err(fail()); }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_image_positions_and_decode_offset() {
        let p = image_positions(&[vec![1, 2, 99, 99, 99, 99, 99, 99, 3]], 99, &[[1, 4, 6]], 2).unwrap();
        assert_eq!(p.coordinates, [[0,0,0],[1,1,1],[2,2,2],[2,2,3],[2,2,4],[2,3,2],[2,3,3],[2,3,4],[5,5,5]]);
        assert_eq!(p.next, [6]);
        assert_eq!((p.spans[0].start, p.spans[0].count, p.spans[0].feature_start), (2,6,0));
    }

    #[test]
    fn image_order_across_prompts_and_multiple_images() {
        let p = image_positions(&[vec![99,1,99,2], vec![0,99,3,4]], 99, &[[1,2,2];3], 2).unwrap();
        assert_eq!(p.next, [4,4]);
        assert_eq!(p.spans.iter().map(|s| (s.batch,s.start,s.feature_start)).collect::<Vec<_>>(), [(0,0,0),(0,2,1),(1,1,2)]);
        assert!(image_positions(&[vec![99]],99,&[],2).is_err());
        assert!(image_positions(&[vec![99,99]],99,&[[1,2,2]],2).is_err());
        assert!(image_positions(&[vec![99]],99,&[[2,2,2]],2).is_err());
        assert!(image_positions(&[vec![1]],99,&[[1,2,2]],2).is_err());
        assert!(image_positions(&[vec![1],vec![1,2]],99,&[],2).is_err());
        assert!(image_positions(&[],99,&[],2).is_err());
    }
}

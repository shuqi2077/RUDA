use crate::{GenerationError, HuggingFaceLoadError};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub hidden_act: String,
    pub in_channels: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Qwen35VisionConfig {
        Qwen35VisionConfig {
            depth: 12,
            hidden_size: 768,
            intermediate_size: 3072,
            num_heads: 12,
            hidden_act: "gelu_pytorch_tanh".into(),
            in_channels: 3,
            num_position_embeddings: 2304,
            out_hidden_size: 1024,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            deepstack_visual_indexes: vec![],
        }
    }

    #[test]
    fn validates_local_architecture_and_ragged_grids() {
        let c = config();
        c.validate().unwrap();
        assert_eq!(c.grid_tokens(&[[1, 16, 16], [2, 4, 6]]).unwrap(), 304);
        for grids in [
            vec![],
            vec![[0, 2, 2]],
            vec![[1, 3, 2]],
            vec![[usize::MAX, 2, 2]],
        ] {
            assert!(c.grid_tokens(&grids).is_err());
        }
    }

    #[test]
    fn rejects_incompatible_or_overflowing_configs() {
        let c = config();
        let mut bad = c.clone();
        bad.num_heads = 0;
        assert!(bad.validate().is_err());
        let mut bad = c.clone();
        bad.hidden_size = 767;
        assert!(bad.validate().is_err());
        let mut bad = c.clone();
        bad.num_position_embeddings = 2303;
        assert!(bad.validate().is_err());
        let mut bad = c.clone();
        bad.patch_size = i32::MAX as usize;
        assert!(bad.validate().is_err());
        let mut bad = c.clone();
        bad.hidden_act = "gelu".into();
        assert!(bad.validate().is_err());
        let mut bad = c;
        bad.deepstack_visual_indexes = vec![1];
        assert!(bad.validate().is_err());
    }
}

impl Qwen35VisionConfig {
    pub fn validate(&self) -> Result<(), HuggingFaceLoadError> {
        let dims = [
            self.depth,
            self.hidden_size,
            self.intermediate_size,
            self.num_heads,
            self.in_channels,
            self.num_position_embeddings,
            self.out_hidden_size,
            self.patch_size,
            self.spatial_merge_size,
            self.temporal_patch_size,
        ];
        if dims.contains(&0) || dims.iter().any(|&n| n > i32::MAX as usize) {
            return Err(HuggingFaceLoadError(
                "invalid Qwen3.5 vision dimensions".into(),
            ));
        }
        let side = self.num_position_embeddings.isqrt();
        if self.hidden_size % self.num_heads != 0
            || (self.hidden_size / self.num_heads) % 4 != 0
            || side * side != self.num_position_embeddings
            || self.hidden_act != "gelu_pytorch_tanh"
            || !self.deepstack_visual_indexes.is_empty()
        {
            return Err(HuggingFaceLoadError(
                "unsupported Qwen3.5 vision heads, position grid, activation or deepstack".into(),
            ));
        }
        for factors in [
            vec![
                self.hidden_size,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ],
            vec![self.hidden_size, self.hidden_size, 3],
            vec![self.hidden_size, self.intermediate_size],
            vec![self.num_position_embeddings, self.hidden_size],
            vec![
                self.hidden_size,
                self.spatial_merge_size,
                self.spatial_merge_size,
                self.hidden_size,
                self.spatial_merge_size,
                self.spatial_merge_size,
            ],
            vec![
                self.out_hidden_size,
                self.hidden_size,
                self.spatial_merge_size,
                self.spatial_merge_size,
            ],
        ] {
            if factors
                .into_iter()
                .try_fold(1usize, usize::checked_mul)
                .is_none_or(|n| n > u32::MAX as usize)
            {
                return Err(HuggingFaceLoadError(
                    "Qwen3.5 vision tensor exceeds supported indexing".into(),
                ));
            }
        }
        Ok(())
    }

    pub(in crate::qwen35) fn grid_tokens(&self, grids: &[[usize; 3]]) -> Result<usize, GenerationError> {
        if grids.is_empty() {
            return Err(GenerationError("vision grid must not be empty".into()));
        }
        grids.iter().try_fold(0usize, |sum, &[t, h, w]| {
            if [t, h, w].contains(&0)
                || h % self.spatial_merge_size != 0
                || w % self.spatial_merge_size != 0
            {
                return Err(GenerationError(
                    "vision grid dimensions must be positive and spatially mergeable".into(),
                ));
            }
            t.checked_mul(h)
                .and_then(|n| n.checked_mul(w))
                .and_then(|n| n.checked_add(sum))
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| GenerationError("vision grid exceeds supported indexing".into()))
        })
    }
}

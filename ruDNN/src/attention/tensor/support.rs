use ruda_core::tensor::spatial::AttentionModuleOptions;

/// Describe requests whose semantics the current FlashAttention kernels cannot
/// represent. Keep the explicit launch and automatic selection checks together.
pub(super) fn unsupported_flash_reason(
    options: &AttentionModuleOptions,
    has_attn_bias: bool,
    seq_q: usize,
    seq_k: usize,
) -> Option<&'static str> {
    if has_attn_bias {
        Some("FlashAttention does not support additive attention bias; use the fallback strategy")
    } else if options.scale.is_some() {
        Some("FlashAttention does not support an explicit scale; use the fallback strategy")
    } else if options.softcap.is_some() {
        Some("FlashAttention does not support softcap; use the fallback strategy")
    } else if options.is_causal && seq_q != seq_k {
        // The fallback aligns the causal mask at the bottom-right. The current
        // FlashAttention mask uses col > row, which agrees only for square scores.
        Some("FlashAttention requires equal query/key sequence lengths for causal attention; use the fallback strategy")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_square_causal_attention_remain_eligible() {
        let options = AttentionModuleOptions::default();
        for (seq_q, seq_k) in [(1, 1), (1, 128), (128, 1), (128, 128)] {
            assert!(unsupported_flash_reason(&options, false, seq_q, seq_k).is_none());
        }
        let options = AttentionModuleOptions { is_causal: true, ..options };
        for length in [1, 32, 128] {
            assert!(unsupported_flash_reason(&options, false, length, length).is_none());
        }
    }

    #[test]
    fn additive_bias_and_each_logit_transform_require_fallback() {
        let options = AttentionModuleOptions::default();
        assert!(unsupported_flash_reason(&options, true, 32, 32).unwrap().contains("bias"));
        // Even an explicitly supplied default scale must not be silently ignored.
        for scale in [0.0, 0.125, 0.5, -1.0, f64::INFINITY, f64::NAN] {
            let options = AttentionModuleOptions { scale: Some(scale), ..options };
            assert!(unsupported_flash_reason(&options, false, 32, 32).unwrap().contains("scale"));
        }
        for softcap in [1.0, 30.0, 0.0, -1.0, f64::INFINITY, f64::NAN] {
            let options = AttentionModuleOptions { softcap: Some(softcap), ..options };
            assert!(unsupported_flash_reason(&options, false, 32, 32).unwrap().contains("softcap"));
        }
    }

    #[test]
    fn rectangular_causal_attention_requires_bottom_right_fallback() {
        let options = AttentionModuleOptions { is_causal: true, ..Default::default() };
        for (seq_q, seq_k) in [(1, 128), (32, 128), (128, 32)] {
            assert!(unsupported_flash_reason(&options, false, seq_q, seq_k)
                .unwrap().contains("sequence lengths"));
        }
    }

    #[test]
    fn combined_options_never_make_an_unsupported_request_eligible() {
        for has_bias in [false, true] {
            for scale in [None, Some(0.5)] {
                for softcap in [None, Some(30.0)] {
                    for is_causal in [false, true] {
                        for seq_k in [1, 32] {
                            let options = AttentionModuleOptions { scale, softcap, is_causal };
                            let expected = has_bias || scale.is_some() || softcap.is_some()
                                || (is_causal && seq_k != 1);
                            assert_eq!(
                                unsupported_flash_reason(&options, has_bias, 1, seq_k).is_some(),
                                expected,
                            );
                        }
                    }
                }
            }
        }
    }
}

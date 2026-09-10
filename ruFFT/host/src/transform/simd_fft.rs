    use macerator::{Simd, vload_unaligned, vstore_unaligned};

    /// Scalar radix-4 butterfly for the SIMD tail path.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn scalar_radix4(
        re: &mut [f32],
        im: &mut [f32],
        p0: usize,
        quarter: usize,
        tw_re: &[f32],
        tw_im: &[f32],
        tw_off_inner: usize,
        tw_off_outer: usize,
        k: usize,
    ) {
        let p1 = p0 + quarter;
        let p2 = p1 + quarter;
        let p3 = p2 + quarter;

        let wi_r = tw_re[tw_off_inner + k];
        let wi_i = tw_im[tw_off_inner + k];
        let tw1_re = wi_r * re[p1] - wi_i * im[p1];
        let tw1_im = wi_r * im[p1] + wi_i * re[p1];
        let tw3_re = wi_r * re[p3] - wi_i * im[p3];
        let tw3_im = wi_r * im[p3] + wi_i * re[p3];

        let a_re = re[p0] + tw1_re;
        let a_im = im[p0] + tw1_im;
        let b_re = re[p0] - tw1_re;
        let b_im = im[p0] - tw1_im;
        let c_re = re[p2] + tw3_re;
        let c_im = im[p2] + tw3_im;
        let d_re = re[p2] - tw3_re;
        let d_im = im[p2] - tw3_im;

        let wo_r = tw_re[tw_off_outer + k];
        let wo_i = tw_im[tw_off_outer + k];
        let tc_re = wo_r * c_re - wo_i * c_im;
        let tc_im = wo_r * c_im + wo_i * c_re;
        let td_re = wo_r * d_re - wo_i * d_im;
        let td_im = wo_r * d_im + wo_i * d_re;

        re[p0] = a_re + tc_re;
        im[p0] = a_im + tc_im;
        re[p2] = a_re - tc_re;
        im[p2] = a_im - tc_im;
        re[p1] = b_re + td_im;
        im[p1] = b_im - td_re;
        re[p3] = b_re - td_im;
        im[p3] = b_im + td_re;
    }

    /// SIMD radix-4 butterfly passes (pairs of radix-2 stages).
    #[macerator::with_simd]
    #[allow(clippy::too_many_arguments)]
    pub fn radix4_simd<S: Simd>(
        re: &mut [f32],
        im: &mut [f32],
        n: usize,
        tw_re: &[f32],
        tw_im: &[f32],
        offsets: &[usize],
        start_stage: usize,
        num_stages: usize,
    ) {
        let lanes = S::lanes32();
        let mut stage = start_stage;

        while stage + 1 < num_stages {
            let quarter = 1 << stage;
            let group_size = quarter << 2;
            let tw_off_inner = offsets[stage];
            let tw_off_outer = offsets[stage + 1];

            if quarter >= lanes {
                let mut group_start = 0;
                while group_start < n {
                    let mut k = 0;
                    while k + lanes <= quarter {
                        unsafe {
                            // Inner twiddle: W_{2q}^k
                            let wi_r =
                                vload_unaligned::<S, f32>(tw_re.as_ptr().add(tw_off_inner + k));
                            let wi_i =
                                vload_unaligned::<S, f32>(tw_im.as_ptr().add(tw_off_inner + k));

                            let p0 = group_start + k;
                            let p1 = p0 + quarter;
                            let p2 = p1 + quarter;
                            let p3 = p2 + quarter;

                            let r0 = vload_unaligned::<S, f32>(re.as_ptr().add(p0));
                            let i0 = vload_unaligned::<S, f32>(im.as_ptr().add(p0));
                            let r1 = vload_unaligned::<S, f32>(re.as_ptr().add(p1));
                            let i1 = vload_unaligned::<S, f32>(im.as_ptr().add(p1));
                            let r2 = vload_unaligned::<S, f32>(re.as_ptr().add(p2));
                            let i2 = vload_unaligned::<S, f32>(im.as_ptr().add(p2));
                            let r3 = vload_unaligned::<S, f32>(re.as_ptr().add(p3));
                            let i3 = vload_unaligned::<S, f32>(im.as_ptr().add(p3));

                            // Apply inner twiddle to p1 and p3
                            let tw1_re = wi_r * r1 - wi_i * i1;
                            let tw1_im = wi_r * i1 + wi_i * r1;
                            let tw3_re = wi_r * r3 - wi_i * i3;
                            let tw3_im = wi_r * i3 + wi_i * r3;

                            // Inner butterflies
                            let a_re = r0 + tw1_re;
                            let a_im = i0 + tw1_im;
                            let b_re = r0 - tw1_re;
                            let b_im = i0 - tw1_im;
                            let c_re = r2 + tw3_re;
                            let c_im = i2 + tw3_im;
                            let d_re = r2 - tw3_re;
                            let d_im = i2 - tw3_im;

                            // Outer twiddle: W_{4q}^k
                            let wo_r =
                                vload_unaligned::<S, f32>(tw_re.as_ptr().add(tw_off_outer + k));
                            let wo_i =
                                vload_unaligned::<S, f32>(tw_im.as_ptr().add(tw_off_outer + k));
                            let tc_re = wo_r * c_re - wo_i * c_im;
                            let tc_im = wo_r * c_im + wo_i * c_re;
                            let td_re = wo_r * d_re - wo_i * d_im;
                            let td_im = wo_r * d_im + wo_i * d_re;

                            // Outer butterflies: -i*(td_re+i*td_im) = (td_im, -td_re)
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p0), a_re + tc_re);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p0), a_im + tc_im);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p2), a_re - tc_re);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p2), a_im - tc_im);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p1), b_re + td_im);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p1), b_im - td_re);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p3), b_re - td_im);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p3), b_im + td_re);
                        }
                        k += lanes;
                    }
                    while k < quarter {
                        scalar_radix4(
                            re,
                            im,
                            group_start + k,
                            quarter,
                            tw_re,
                            tw_im,
                            tw_off_inner,
                            tw_off_outer,
                            k,
                        );
                        k += 1;
                    }
                    group_start += group_size;
                }
            } else {
                let mut group_start = 0;
                while group_start < n {
                    for k in 0..quarter {
                        scalar_radix4(
                            re,
                            im,
                            group_start + k,
                            quarter,
                            tw_re,
                            tw_im,
                            tw_off_inner,
                            tw_off_outer,
                            k,
                        );
                    }
                    group_start += group_size;
                }
            }
            stage += 2;
        }
    }

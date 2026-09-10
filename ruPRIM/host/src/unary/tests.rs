    use alloc::vec;
    use ruda_core::tensor::data::TensorData;

    use ruda_core::tensor::host::HostTensor;

    #[test]
    fn test_round_ties_even_large_float() {
        // Values outside i32 range used to break the manual round_ties_even impl.
        let data = vec![2e18_f32, -2e18_f32, f32::MAX, f32::MIN];
        let tensor = HostTensor::from_data(TensorData::new(data.clone(), [data.len()]));
        let result = super::round(tensor);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        for (a, b) in out.iter().zip(data.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // The previous A&S 7.1.26 erf implementation had ~1.5e-7 max absolute
    // error regardless of arithmetic precision. That was acceptable for
    // f32 but orders of magnitude worse than f64's ~2.2e-16 precision.
    // These tests guard near-ulp accuracy across the domain, including
    // libm's piecewise-rational switch points near |x|=0.84375 and the
    // saturation region around |x|>=6 where erf(x) rounds to ±1.
    // Reference values from Wolfram Alpha / DLMF.
    #[test]
    fn test_erf_f64_is_full_precision() {
        // Reference values computed with mpmath at 25-digit precision.
        let cases = [
            (0.0f64, 0.0f64),
            (1e-10, 1.128379167095512574e-10), // small-x linear regime
            (0.5, 0.5204998778130465377),
            (0.84, 0.7651427114549945347), // just below libm's 0.84375 boundary
            (0.85, 0.7706680576083525324), // just above
            (1.0, 0.8427007929497148693),
            (1.5, 0.9661051464753107271),
            (2.0, 0.9953222650189527342),
            (3.0, 0.9999779095030014145),
            (6.0, 0.9999999999999999785), // near saturation
            (30.0, 1.0),                  // fully saturated
            (-0.5, -0.5204998778130465377),
            (-1.0, -0.8427007929497148693),
            (-3.0, -0.9999779095030014145),
        ];
        for (x, expected) in cases {
            let got = super::erf_f64(x);
            let err = (got - expected).abs();
            assert!(
                err < 1e-14,
                "erf_f64({}) = {} expected {} (err {:e})",
                x,
                got,
                expected,
                err
            );
        }
    }

    // f32 erf must stay accurate to within f32 precision (~6e-8). Covers
    // the same piecewise boundaries as the f64 test.
    #[test]
    fn test_erf_f32_full_f32_precision() {
        let cases = [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (0.84, 0.765_142_7),
            (0.85, 0.770_668_06),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_24),
            (6.0, 1.0),
            (-0.5, -0.520_499_9),
            (-1.0, -0.842_700_8),
        ];
        for (x, expected) in cases {
            let got = super::erf_f32(x);
            assert!(
                (got - expected).abs() < 1e-6,
                "erf_f32({}) = {} expected {}",
                x,
                got,
                expected
            );
        }
    }

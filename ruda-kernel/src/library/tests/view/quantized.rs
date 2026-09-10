use crate::dsl::prelude::*;
use ruda_core::{
    e2m1, e2m1x2,
    quant::scheme::{QuantScheme, QuantValue},
};

use crate::library::tensor::{
    View,
    launch::ViewArg,
    layout::{plain::PlainLayout, *},
};

#[derive(RudaType, RudaLaunch)]
struct TestPerTensorScaleLayout {
    length: usize,
}

#[ruda]
impl Layout for TestPerTensorScaleLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, _pos: Self::Coordinates) -> Self::SourceCoordinates {
        0usize.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), true.runtime())
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        self.length
    }
}

#[ruda(launch_unchecked)]
pub fn kernel_quantized_view<F: Float, N: Size>(
    lhs: View<Vector<F, N>, Coords1d>,
    output: &mut Array<Vector<F, N>>,
) {
    if (UNIT_POS as usize) < lhs.shape() {
        output[UNIT_POS as usize] = lhs[UNIT_POS as usize];
    }
}

#[allow(clippy::needless_range_loop)]
pub fn test_quantized_per_tensor_int<R: Runtime, F: Float + RudaElement>(
    client: ComputeClient<R>,
    vector_size_values: VectorSize,
) {
    let vector_size_float = 8 * vector_size_values;

    let scheme = QuantScheme::default().with_value(QuantValue::Q4F);
    let float_data = (-8..=7)
        .map(|it| F::new(it as f32 * 3.4))
        .collect::<Vec<_>>();

    let output = client.empty(16 * size_of::<F>());
    let values = client.create_from_slice(u32::as_bytes(&[0xFEDCBA98, 0x76543210]));
    let scales = client.create_from_slice(f32::as_bytes(&[3.4]));

    let float_values = client.create_from_slice(F::as_bytes(&float_data));
    let float_output = client.empty(16 * size_of::<F>());

    let scales_layout = TestPerTensorScaleLayoutLaunch::new(16);

    let values_view =
        ViewArg::new_array::<PlainLayout>(unsafe { ArrayArg::from_raw_parts(values, 2) }, ());
    let scales_view = ViewArg::new_array::<TestPerTensorScaleLayout>(
        unsafe { ArrayArg::from_raw_parts(scales, 1) },
        scales_layout,
    );
    let quantized_view = ViewArg::new_quantized(values_view, scales_view, scheme);
    let float_view = ViewArg::new_array::<PlainLayout>(
        unsafe { ArrayArg::from_raw_parts(float_values, 16) },
        (),
    );

    unsafe {
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(2),
            vector_size_float,
            quantized_view,
            ArrayArg::from_raw_parts(output.clone(), 16),
        );
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(2),
            vector_size_float,
            float_view,
            ArrayArg::from_raw_parts(float_output.clone(), 16),
        );
    }

    let actual = client.read_one_unchecked(output);
    let actual_float = client.read_one_unchecked(float_output);
    let actual = F::from_bytes(&actual);
    let actual_float = F::from_bytes(&actual_float);

    assert_eq!(&actual, &float_data);
    assert_eq!(&actual_float, &float_data);
}

#[allow(clippy::needless_range_loop)]
pub fn test_quantized_per_tensor_fp4<R: Runtime, F: Float + RudaElement>(
    client: ComputeClient<R>,
    vector_size_values: VectorSize,
) {
    if !client.properties().supports_type(e2m1x2::ruda_type()) {
        return;
    }

    let vector_size_float = 8 * vector_size_values;

    let scheme = QuantScheme::default().with_value(QuantValue::E2M1);
    let float_data = (0..16)
        .map(e2m1::from_bits)
        .map(|it| F::new(it.to_f32() * 3.4))
        .collect::<Vec<_>>();

    let output = client.empty(16 * size_of::<F>());
    let values = client.create_from_slice(u32::as_bytes(&[0x76543210, 0xFEDCBA98]));
    let scales = client.create_from_slice(f32::as_bytes(&[3.4]));

    let float_values = client.create_from_slice(F::as_bytes(&float_data));
    let float_output = client.empty(16 * size_of::<F>());

    let scales_layout = TestPerTensorScaleLayoutLaunch::new(16);

    let values_view =
        ViewArg::new_array::<PlainLayout>(unsafe { ArrayArg::from_raw_parts(values, 2) }, ());
    let scales_view = ViewArg::new_array::<TestPerTensorScaleLayout>(
        unsafe { ArrayArg::from_raw_parts(scales, 1) },
        scales_layout,
    );
    let quantized_view = ViewArg::new_quantized(values_view, scales_view, scheme);
    let float_view = ViewArg::new_array::<PlainLayout>(
        unsafe { ArrayArg::from_raw_parts(float_values, 16) },
        (),
    );

    unsafe {
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(2),
            vector_size_float,
            quantized_view,
            ArrayArg::from_raw_parts(output.clone(), 16),
        );
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(2),
            vector_size_float,
            float_view,
            ArrayArg::from_raw_parts(float_output.clone(), 16),
        );
    }

    let actual = client.read_one_unchecked(output);
    let actual_float = client.read_one_unchecked(float_output);
    let actual = F::from_bytes(&actual);
    let actual_float = F::from_bytes(&actual_float);

    assert_eq!(&actual, &float_data);
    assert_eq!(&actual_float, &float_data);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_quantized_view {
    ($ty: ty) => {
        use super::*;

        #[$crate::library::tests::test_log::test]
        fn test_quantized_view_per_tensor_int() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::library::tests::view::quantized::test_quantized_per_tensor_int::<TestRuntime, $ty>(
                client.clone(),
                1,
            );
            ruda_kernel::library::tests::view::quantized::test_quantized_per_tensor_int::<TestRuntime, $ty>(
                client, 2,
            );
        }

        #[$crate::library::tests::test_log::test]
        fn test_quantized_view_per_tensor_fp4() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::library::tests::view::quantized::test_quantized_per_tensor_fp4::<TestRuntime, $ty>(
                client.clone(),
                1,
            );
            ruda_kernel::library::tests::view::quantized::test_quantized_per_tensor_fp4::<TestRuntime, $ty>(
                client, 2,
            );
        }
    };
}

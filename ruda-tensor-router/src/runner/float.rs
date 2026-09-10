use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_float(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &FloatOperationIr,
    ) {
        match op {
            FloatOperationIr::Exp(desc) => {
                unary_float_ops!(handles, desc, B::float_exp)
            }
            FloatOperationIr::Powf(desc) => {
                binary_float_ops!(handles, desc, B::float_powf)
            }
            FloatOperationIr::Log(desc) => {
                unary_float_ops!(handles, desc, B::float_log)
            }
            FloatOperationIr::Log1p(desc) => {
                unary_float_ops!(handles, desc, B::float_log1p)
            }
            FloatOperationIr::Erf(desc) => {
                unary_float_ops!(handles, desc, B::float_erf)
            }
            FloatOperationIr::PowfScalar(desc) => {
                scalar_float_ops!(handles, desc, B::float_powf_scalar)
            }
            FloatOperationIr::Sqrt(desc) => {
                unary_float_ops!(handles, desc, B::float_sqrt)
            }
            FloatOperationIr::Rsqrt(desc) => {
                unary_float_ops!(handles, desc, B::float_rsqrt)
            }
            FloatOperationIr::Silu(desc) => {
                unary_float_ops!(handles, desc, B::silu)
            }
            FloatOperationIr::Cos(desc) => {
                unary_float_ops!(handles, desc, B::float_cos)
            }
            FloatOperationIr::Sin(desc) => {
                unary_float_ops!(handles, desc, B::float_sin)
            }
            FloatOperationIr::Tanh(desc) => {
                unary_float_ops!(handles, desc, B::float_tanh)
            }
            FloatOperationIr::Tan(desc) => unary_float_ops!(handles, desc, B::float_tan),
            FloatOperationIr::Cosh(desc) => unary_float_ops!(handles, desc, B::float_cosh),
            FloatOperationIr::Sinh(desc) => unary_float_ops!(handles, desc, B::float_sinh),
            FloatOperationIr::ArcCos(desc) => unary_float_ops!(handles, desc, B::float_acos),
            FloatOperationIr::ArcCosh(desc) => unary_float_ops!(handles, desc, B::float_acosh),
            FloatOperationIr::ArcSin(desc) => unary_float_ops!(handles, desc, B::float_asin),
            FloatOperationIr::ArcSinh(desc) => unary_float_ops!(handles, desc, B::float_asinh),
            FloatOperationIr::ArcTan(desc) => unary_float_ops!(handles, desc, B::float_atan),
            FloatOperationIr::ArcTanh(desc) => unary_float_ops!(handles, desc, B::float_atanh),
            FloatOperationIr::ArcTan2(desc) => binary_float_ops!(handles, desc, B::float_atan2),
            FloatOperationIr::Round(desc) => {
                unary_float_ops!(handles, desc, B::float_round)
            }
            FloatOperationIr::Floor(desc) => {
                unary_float_ops!(handles, desc, B::float_floor)
            }
            FloatOperationIr::Ceil(desc) => {
                unary_float_ops!(handles, desc, B::float_ceil)
            }
            FloatOperationIr::Trunc(desc) => {
                unary_float_ops!(handles, desc, B::float_trunc)
            }
            FloatOperationIr::IntoInt(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_into_int(tensor, desc.out.dtype.into());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::Matmul(desc) => {
                binary_float_ops!(handles, desc, B::float_matmul)
            }
            FloatOperationIr::Cross(desc) => {
                let lhs = handles.get_float_tensor::<B>(&desc.lhs);
                let rhs = handles.get_float_tensor::<B>(&desc.rhs);
                let output = B::float_cross(lhs, rhs, desc.dim);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::Random(desc) => {
                let shape = desc.out.shape.clone();

                let output = B::float_random(
                    shape,
                    desc.distribution,
                    &self.device,
                    desc.out.dtype.into(),
                );
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::Recip(desc) => {
                unary_float_ops!(handles, desc, B::float_recip)
            }
            FloatOperationIr::Quantize(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let scales = handles.get_float_tensor::<B>(&desc.qparams.scales);
                let output = B::quantize(tensor, &desc.scheme,
                    ruda_tensor::quantization::QuantizationParametersPrimitive { scales });
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::QuantizeDynamic(desc) => {
                let DType::QFloat(scheme) = desc.out.dtype else {
                    panic!("dynamic quantization requires a quantized output dtype");
                };
                let tensor = handles.get_float_tensor::<B>(&desc.input);
                let output = B::quantize_dynamic(tensor, &scheme);
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::Dequantize(desc) => {
                let tensor = handles.get_quantized_tensor::<B>(&desc.input);
                let output = B::dequantize(tensor, desc.out.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::IsNan(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_is_nan(tensor, desc.out.dtype.into());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::IsInf(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_is_inf(tensor, desc.out.dtype.into());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            FloatOperationIr::GridSample2d(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let grid = handles.get_float_tensor::<B>(&desc.grid);

                let output = B::float_grid_sample_2d(tensor, grid, desc.options.clone().into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
        }
    }
}

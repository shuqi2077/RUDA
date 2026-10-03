using Arange = int (*)(const Descriptor*, uint64_t, uint64_t);
using Normal = int (*)(const Descriptor*, float, float, const uint32_t*);
Arange arange_native = nullptr;
Normal normal_native = nullptr;

at::Tensor& normal_(at::Tensor& self, double mean, double std, std::optional<at::Generator> generator) {
  TORCH_CHECK(normal_native, "RUDA factory API 1 unavailable; rebuild both native libraries");
  TORCH_CHECK(self.is_floating_point() && std>=0, "normal_ requires floating storage and std >= 0");
  at::assert_no_internal_overlap(self);
  Argument out(self);
  auto gen = at::get_generator_or_default<Generator>(generator,default_generator());
  if (!self.numel()) return self;
  uint32_t seeds[4];
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    for (auto& seed:seeds) seed=gen->random();
  }
  check(normal_native(&out.desc,float(mean),float(std),seeds));
  return self;
}

at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end,
                       const at::Scalar& step, at::Tensor& out) {
  TORCH_CHECK(arange_native, "RUDA factory API 1 unavailable; rebuild both native libraries");
  validate(out.device());
  const auto dtype = dtype_code(out.scalar_type());
  TORCH_CHECK(dtype!=3, "arange does not support bool storage");
  at::native::arange_check_bounds(start,end,step);
  int64_t size;
  if (out.scalar_type()==at::kLong) {
    auto a=start.to<int64_t>(), b=end.to<int64_t>(), s=step.to<int64_t>();
    TORCH_CHECK(s!=0, "step must be nonzero");
    TORCH_CHECK(s>0 ? b>=a : b<=a, "upper bound and lower bound inconsistent with step sign");
    uint64_t distance = s>0 ? uint64_t(b)-uint64_t(a) : uint64_t(a)-uint64_t(b);
    uint64_t magnitude = s>0 ? uint64_t(s) : uint64_t(0)-uint64_t(s);
    uint64_t count = distance/magnitude + (distance%magnitude!=0);
    TORCH_CHECK(count<=uint64_t(std::numeric_limits<int64_t>::max()), "invalid size, possible overflow?");
    size=int64_t(count);
  } else {
    auto count = std::ceil((end.to<double>()-start.to<double>())/step.to<double>());
    TORCH_CHECK(count>=0 && count<double(std::numeric_limits<int64_t>::max()), "invalid size, possible overflow?");
    size=int64_t(count);
  }
  if (out.numel()!=size) {
    if (out.numel()) TORCH_WARN("arange out tensor will be resized to match the computed number of elements");
    resize_tensor(out,{size},std::nullopt);
  }
  at::assert_no_internal_overlap(out);
  Argument result(out);
  auto a = dtype<=2 ? scalar_bits<float>(start) : scalar_bits<int64_t>(start);
  auto s = dtype<=2 ? scalar_bits<float>(step) : scalar_bits<int64_t>(step);
  check(arange_native(&result.desc,a,s));
  return out;
}

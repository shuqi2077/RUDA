#include <ATen/EmptyTensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/ops/empty.h>
#include <torch/csrc/utils/pybind.h> // Tensor caster without the full C++ frontend
#include <torch/library.h>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {
struct Descriptor {
  void* allocation;
  size_t offset_bytes;
  size_t rank;
  const size_t* shape;
  const size_t* strides;
  uint32_t dtype;
};
using Alloc = int (*)(size_t, void**, uint64_t*);
using Free = int (*)(void*);
using Error = const char* (*)();
using Execute = int (*)(uint32_t, const Descriptor*, const Descriptor*, const Descriptor*, float);
using Addmm = int (*)(const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, float, float);
using LayerNorm = int (*)(const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, const Descriptor*, float);
using RMSNorm = int (*)(const Descriptor*, const Descriptor*, const Descriptor*, float);
using Spatial = int (*)(uint32_t, const Descriptor*, const Descriptor*, const Descriptor*, const int64_t*, size_t);
using Fill = int (*)(const Descriptor*, uint64_t);
using Transfer = int (*)(const Descriptor*, void*, bool);
using Sync = int (*)();
using StreamInterop = int (*)(uint32_t,uint64_t,uint64_t,uint32_t,uint64_t*);
using Paged = int (*)(uint32_t,void**,const Descriptor*,const Descriptor*,const Descriptor*,
                     const Descriptor*,const Descriptor*,const Descriptor*,const uint32_t*,size_t,const uint32_t*,float,bool);
Alloc allocate_native = nullptr;
Free free_native = nullptr;
Error error_native = nullptr;
Execute execute_native = nullptr;
Addmm addmm_native = nullptr;
LayerNorm layer_norm_native = nullptr;
RMSNorm rms_norm_native = nullptr;
Spatial spatial_native = nullptr;
Fill fill_native = nullptr;
Transfer transfer_native = nullptr;
Sync sync_native = nullptr;
StreamInterop stream_native = nullptr;
Paged paged_native = nullptr;

void check(int code) { TORCH_CHECK(code == 0, error_native ? error_native() : "RUDA bridge is not initialized"); }
void synchronize() { TORCH_CHECK(sync_native, "RUDA bridge is not initialized"); check(sync_native()); }
void validate(c10::Device device) {
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1 && device.index() <= 0,
              "RUDA currently exposes only ruda:0");
}
uint32_t dtype_code(c10::ScalarType dtype) {
  if (dtype == at::kFloat) return 0;
  if (dtype == at::kHalf) return 1;
  if (dtype == at::kBFloat16) return 2;
  if (dtype == at::kBool) return 3;
  if (dtype == at::kLong) return 4;
  if (dtype == at::kInt) return 5;
  if (dtype == at::kShort) return 6;
  if (dtype == at::kChar) return 7;
  if (dtype == at::kByte) return 8;
  TORCH_CHECK(false, "RUDA native storage supports float32, float16, bfloat16, bool, int64, int32, int16, int8 and uint8");
}
void release(void* handle) {
  if (handle && free_native(handle)) std::fprintf(stderr, "RUDA release: %s\n", error_native());
}
class Allocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    TORCH_CHECK(allocate_native, "RUDA bridge is not initialized");
    void* handle = nullptr;
    uint64_t ptr = 0;
    check(allocate_native(bytes, &handle, &ptr));
    return c10::DataPtr(reinterpret_cast<void*>(ptr), handle, release,
                       c10::Device(c10::DeviceType::PrivateUse1, 0));
  }
  void copy_data(void*, const void*, size_t) const override {
    TORCH_CHECK(false, "RUDA raw storage copy is not implemented; use Tensor.copy_");
  }
};
Allocator allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);

uint64_t stream_command(uint32_t op, uint64_t stream=0, uint64_t object=0, uint32_t flags=0) {
  TORCH_CHECK(stream_native, "RUDA ABI 8 stream bridge is not initialized");
  uint64_t result=0;check(stream_native(op,stream,object,flags,&result));return result;
}
c10::Stream make_stream(uint64_t id) {
  return c10::Stream(c10::Stream::UNSAFE,c10::Device(c10::DeviceType::PrivateUse1,0),id);
}
void check_stream(const c10::Stream& stream) { validate(stream.device()); }
class Guard final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }
  c10::Device getDevice() const override { return c10::Device(type(),0); }
  void setDevice(c10::Device device) const override {validate(device);}
  c10::Device exchangeDevice(c10::Device device) const override {validate(device);return getDevice();}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device device) const override {validate(device);return make_stream(stream_command(0));}
  c10::Stream getDefaultStream(c10::Device device) const override {validate(device);return make_stream(0);}
  c10::Stream getNewStream(c10::Device device,int priority=0) const override {
    validate(device);TORCH_CHECK(priority==0,"RUDA priority streams are not supported");
    return make_stream(stream_command(1));
  }
  c10::Stream getStreamFromGlobalPool(c10::Device device,bool high=false) const override {
    TORCH_CHECK(!high,"RUDA high-priority streams are not supported");return getNewStream(device);
  }
  c10::Stream exchangeStream(c10::Stream value) const override {check_stream(value);return make_stream(stream_command(2,value.id()));}
  c10::DeviceIndex deviceCount() const noexcept override {return 1;}
  bool queryStream(const c10::Stream& value) const override {check_stream(value);return stream_command(3,value.id())!=0;}
  void synchronizeStream(const c10::Stream& value) const override {check_stream(value);stream_command(4,value.id());}
  void synchronizeDevice(c10::DeviceIndex index) const override {validate(c10::Device(type(),index));stream_command(11);}
  void record(void** event,const c10::Stream& stream,c10::DeviceIndex index,c10::EventFlag flag) const override {
    validate(c10::Device(type(),index));check_stream(stream);
    TORCH_CHECK(flag!=c10::EventFlag::INVALID,"invalid event flag");
    *event=reinterpret_cast<void*>(stream_command(5,stream.id(),reinterpret_cast<uintptr_t>(*event),flag==c10::EventFlag::BACKEND_DEFAULT));
  }
  void block(void* event,const c10::Stream& stream) const override {check_stream(stream);stream_command(6,stream.id(),reinterpret_cast<uintptr_t>(event));}
  bool queryEvent(void* event) const override {return stream_command(7,0,reinterpret_cast<uintptr_t>(event))!=0;}
  void synchronizeEvent(void* event) const override {stream_command(8,0,reinterpret_cast<uintptr_t>(event));}
  void destroyEvent(void* event,c10::DeviceIndex) const noexcept override {
    if (!event || !stream_native) return;
    uint64_t ignored=0;
    if(stream_native(9,0,reinterpret_cast<uintptr_t>(event),0,&ignored))
      std::fprintf(stderr,"RUDA event destruction: %s\n",error_native());
  }
  double elapsedTime(void* a,void* b,c10::DeviceIndex index) const override {
    validate(c10::Device(type(),index));uint32_t bits=stream_command(10,reinterpret_cast<uintptr_t>(a),reinterpret_cast<uintptr_t>(b));
    float ms;std::memcpy(&ms,&bits,sizeof(ms));return ms;
  }
  void recordDataPtrOnStream(const c10::DataPtr& ptr,const c10::Stream& stream) const override {
    check_stream(stream);TORCH_CHECK(ptr.get_deleter()==release,"record_stream requires RUDA storage");
    if(ptr.get_context())stream_command(12,stream.id(),reinterpret_cast<uintptr_t>(ptr.get_context()));
  }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, Guard);

class Hooks final : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return allocate_native != nullptr; }
  bool hasPrimaryContext(c10::DeviceIndex device) const override { return device == 0 && isAvailable(); }
  c10::DeviceIndex deviceCount() const override { return 1; }
  void setCurrentDevice(c10::DeviceIndex device) const override { validate(c10::Device(c10::DeviceType::PrivateUse1, device)); }
  c10::DeviceIndex getCurrentDevice() const override { return 0; }
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override { setCurrentDevice(device); return 0; }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override { return exchangeDevice(device); }
};

struct Argument {
  std::vector<size_t> shape, strides;
  Descriptor desc;
  explicit Argument(const at::Tensor& tensor) {
    validate(tensor.device());
    TORCH_CHECK(!tensor.is_conj() && !tensor.is_neg(), "RUDA does not support unresolved view bits");
    TORCH_CHECK(tensor.storage().data_ptr().get_deleter() == release, "tensor is not allocated by RUDA");
    for (auto d : tensor.sizes()) { TORCH_CHECK(d >= 0); shape.push_back(d); }
    for (auto d : tensor.strides()) { TORCH_CHECK(d >= 0); strides.push_back(d); }
    desc = {tensor.storage().data_ptr().get_context(), size_t(tensor.storage_offset()) * tensor.element_size(),
            shape.size(), shape.data(), strides.data(), dtype_code(tensor.scalar_type())};
  }
};

class PagedPlanBridge {
  void* plan_=nullptr;
 public:
  PagedPlanBridge(const at::Tensor& like,const std::vector<uint32_t>& spec,const std::vector<uint32_t>& words) {
    TORCH_CHECK(paged_native && spec.size()==6,"invalid native paged plan header");
    Argument q(like);
    check(paged_native(0,&plan_,&q.desc,nullptr,nullptr,nullptr,nullptr,nullptr,
                       words.data(),words.size(),spec.data(),1.0f,true));
  }
  PagedPlanBridge(const PagedPlanBridge&)=delete;
  PagedPlanBridge& operator=(const PagedPlanBridge&)=delete;
  ~PagedPlanBridge() {
    if(plan_ && paged_native(2,&plan_,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr,1.0f,true))
      std::fprintf(stderr,"RUDA paged plan release: %s\n",error_native());
  }
  at::Tensor run(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                 const std::optional<at::Tensor>& qp,const std::optional<at::Tensor>& kp,
                 double scale,bool causal) {
    TORCH_CHECK(q.dim()==3 && k.dim()==4 && v.dim()==4,"paged Q/cache ranks must be 3/4/4");
    TORCH_CHECK(qp.has_value()==kp.has_value(),"MLA requires both positional tensors");
    for(const auto& t: {q,k,v})TORCH_CHECK(!t.requires_grad(),"paged attention is inference-only");
    if(qp)TORCH_CHECK(!qp->requires_grad() && !kp->requires_grad(),"MLA is inference-only");
    auto out=at::empty({q.size(0),q.size(1),v.size(3)},q.options());
    for(const auto& t: {q,k,v})at::assert_no_overlap(out,t);
    Argument aq(q),ak(k),av(v),ao(out);
    auto ap=qp?std::make_unique<Argument>(*qp):nullptr;
    auto akp=kp?std::make_unique<Argument>(*kp):nullptr;
    check(paged_native(1,&plan_,&aq.desc,&ak.desc,&av.desc,ap?&ap->desc:nullptr,akp?&akp->desc:nullptr,
                       &ao.desc,nullptr,0,nullptr,static_cast<float>(scale),causal));
    return out;
  }
};

at::Tensor empty(c10::IntArrayRef size, std::optional<c10::ScalarType> dtype,
  std::optional<c10::Layout> layout, std::optional<c10::Device> device,
  std::optional<bool> pin, std::optional<c10::MemoryFormat> format) {
  validate(c10::device_or_default(device));
  dtype_code(c10::dtype_or_default(dtype));
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided && !c10::pinned_memory_or_default(pin));
  return at::detail::empty_generic(size, &allocator, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), c10::dtype_or_default(dtype), format);
}
at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
  std::optional<c10::ScalarType> dtype, std::optional<c10::Layout> layout,
  std::optional<c10::Device> device, std::optional<bool> pin) {
  validate(c10::device_or_default(device));
  dtype_code(c10::dtype_or_default(dtype));
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided && !c10::pinned_memory_or_default(pin));
  return at::detail::empty_strided_generic(size, stride, &allocator,
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), c10::dtype_or_default(dtype));
}
void execute(uint32_t op, const at::Tensor& a, const at::Tensor& b, at::Tensor out, float scalar) {
  at::assert_no_internal_overlap(out);
  if (op == 7 || op == 30) {
    at::assert_no_overlap(out, a);
    at::assert_no_overlap(out, b);
  } else {
    at::assert_no_partial_overlap(out, a);
    at::assert_no_partial_overlap(out, b);
  }
  Argument av(a), bv(b), ov(out);
  check(execute_native(op, &av.desc, &bv.desc, &ov.desc, scalar));
}

void addmm(const at::Tensor& bias, const at::Tensor& a, const at::Tensor& b,
           at::Tensor out, float alpha, float beta) {
  TORCH_CHECK(addmm_native, "RUDA ABI 5 is required; rebuild Rust and C++ extensions");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, a);
  at::assert_no_overlap(out, b);
  at::assert_no_overlap(out, bias);
  Argument cv(bias), av(a), bv(b), ov(out);
  check(addmm_native(&cv.desc, &av.desc, &bv.desc, &ov.desc, alpha, beta));
}

void layer_norm(const at::Tensor& input, const std::optional<at::Tensor>& weight,
                const std::optional<at::Tensor>& bias, at::Tensor out, at::Tensor mean,
                at::Tensor rstd, double epsilon) {
  TORCH_CHECK(layer_norm_native, "RUDA ABI 6 is required; rebuild Rust and C++ extensions");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, input);
  if (weight) at::assert_no_overlap(out, *weight);
  if (bias) at::assert_no_overlap(out, *bias);
  Argument iv(input), ov(out), mv(mean), rv(rstd);
  std::optional<Argument> wv, bv;
  if (weight) wv.emplace(*weight);
  if (bias) bv.emplace(*bias);
  check(layer_norm_native(&iv.desc, wv ? &wv->desc : nullptr, bv ? &bv->desc : nullptr,
                          &ov.desc, &mv.desc, &rv.desc, static_cast<float>(epsilon)));
}

void rms_norm(const at::Tensor& input, const std::optional<at::Tensor>& weight, at::Tensor out, double epsilon) {
  TORCH_CHECK(rms_norm_native, "RUDA ABI 7 is required; rebuild Rust and C++ extensions");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, input);
  if (weight) at::assert_no_overlap(out, *weight);
  Argument iv(input), ov(out);
  std::optional<Argument> wv;
  if (weight) wv.emplace(*weight);
  check(rms_norm_native(&iv.desc, wv ? &wv->desc : nullptr, &ov.desc, static_cast<float>(epsilon)));
}

void spatial(uint32_t op, const at::Tensor& a, const at::Tensor& b, at::Tensor out,
             const std::vector<int64_t>& params) {
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, a);
  at::assert_no_overlap(out, b);
  if (op == 6) {
    at::assert_no_internal_overlap(b);
    at::assert_no_overlap(b, a);
  }
  Argument av(a), bv(b), ov(out);
  check(spatial_native(op, &av.desc, &bv.desc, &ov.desc, params.data(), params.size()));
}

template <typename T>
uint64_t scalar_bits(const at::Scalar& value) {
  const T converted = value.to<T>();
  uint64_t bits = 0;
  std::memcpy(&bits, &converted, sizeof(T));
  return bits;
}
void fill(at::Tensor out, const at::Scalar& value) {
  at::assert_no_internal_overlap(out);
  Argument ov(out);
  uint64_t bits = 0;
  switch (ov.desc.dtype) {
    case 0: bits = scalar_bits<float>(value); break;
    case 1: bits = scalar_bits<at::Half>(value); break;
    case 2: bits = scalar_bits<at::BFloat16>(value); break;
    case 3: bits = scalar_bits<bool>(value); break;
    case 4: bits = scalar_bits<int64_t>(value); break;
    case 5: bits = scalar_bits<int32_t>(value); break;
    case 6: bits = scalar_bits<int16_t>(value); break;
    case 7: bits = scalar_bits<int8_t>(value); break;
    case 8: bits = scalar_bits<uint8_t>(value); break;
  }
  check(fill_native(&ov.desc, bits));
}

at::Tensor copy_from(const at::Tensor& source, const at::Tensor& dest, bool non_blocking) {
  dtype_code(source.scalar_type());
  dtype_code(dest.scalar_type());
  at::assert_no_internal_overlap(dest);
  auto input = source.expand(dest.sizes());
  if (source.device().is_cpu()) {
    auto packed = input.contiguous();
    auto staging = source.scalar_type() == dest.scalar_type() ? dest : at::empty(dest.sizes(), dest.options().dtype(source.scalar_type()));
    Argument target(staging);
    check(transfer_native(&target.desc, packed.data_ptr(), true));
    if (!staging.is_same(dest)) execute(0, staging, staging, dest, 0);
  } else if (dest.device().is_cpu()) {
    auto staging = source.scalar_type() == dest.scalar_type() ? input : at::empty(dest.sizes(), source.options().dtype(dest.scalar_type()));
    if (!staging.is_same(input)) execute(0, input, input, staging, 0);
    auto packed = at::empty(dest.sizes(), dest.options());
    Argument origin(staging);
    check(transfer_native(&origin.desc, packed.data_ptr(), false));
    const_cast<at::Tensor&>(dest).copy_(packed, non_blocking);
  } else {
    if (source.is_same(dest)) return dest;
    execute(0, input, input, dest, 0);
  }
  return dest;
}
at::Tensor& copy_(at::Tensor& dest, const at::Tensor& source, bool non_blocking) { copy_from(source, dest, non_blocking); return dest; }
template <typename T>
T read_scalar(const at::Tensor& value) {
  T result{};
  Argument arg(value);
  check(transfer_native(&arg.desc, &result, false));
  return result;
}
at::Scalar scalar(const at::Tensor& value) {
  TORCH_CHECK(value.numel() == 1);
  switch (dtype_code(value.scalar_type())) {
    case 3: return read_scalar<uint8_t>(value) != 0;
    case 4: return read_scalar<int64_t>(value);
    case 5: return int64_t(read_scalar<int32_t>(value));
    case 6: return int64_t(read_scalar<int16_t>(value));
    case 7: return int64_t(read_scalar<int8_t>(value));
    case 8: return int64_t(read_scalar<uint8_t>(value));
  }
  auto converted = value.scalar_type() == at::kFloat ? value : value.to(at::kFloat);
  return read_scalar<float>(converted);
}
void unsupported(const c10::OperatorHandle& op, torch::jit::Stack*) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "RUDA GPU operator not implemented: ", op.schema().operator_name(), "; CPU fallback is disabled");
}
void record_tensor_stream(at::Tensor& value, c10::Stream stream) {
  check_stream(stream); Argument arg(value);
  stream_command(12,stream.id(),reinterpret_cast<uintptr_t>(arg.desc.allocation));
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("record_stream",record_tensor_stream);
  m.impl("empty.memory_format", empty);
  m.impl("empty_strided", empty_strided);
  m.impl("as_strided", at::native::as_strided_tensorimpl);
  m.impl("as_strided_", at::native::as_strided__symint);
  m.impl("view", at::native::view);
  m.impl("_reshape_alias", at::native::_reshape_alias);
  m.impl("copy_", copy_);
  m.impl("_copy_from", copy_from);
  m.impl("_local_scalar_dense", scalar);
}
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&unsupported>());
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("abi_version") = 9;
  m.def("initialize", [](std::vector<uintptr_t> addresses) {
    TORCH_CHECK(addresses.size() == 13 && !allocate_native, "invalid or repeated RUDA initialization");
    for (auto address : addresses) TORCH_CHECK(address != 0, "null RUDA ABI function");
    allocate_native = reinterpret_cast<Alloc>(addresses[0]);
    free_native = reinterpret_cast<Free>(addresses[1]);
    error_native = reinterpret_cast<Error>(addresses[2]);
    execute_native = reinterpret_cast<Execute>(addresses[3]);
    transfer_native = reinterpret_cast<Transfer>(addresses[4]);
    sync_native = reinterpret_cast<Sync>(addresses[5]);
    fill_native = reinterpret_cast<Fill>(addresses[6]);
    spatial_native = reinterpret_cast<Spatial>(addresses[7]);
    addmm_native = reinterpret_cast<Addmm>(addresses[8]);
    layer_norm_native = reinterpret_cast<LayerNorm>(addresses[9]);
    rms_norm_native = reinterpret_cast<RMSNorm>(addresses[10]);
    stream_native = reinterpret_cast<StreamInterop>(addresses[11]);
    paged_native = reinterpret_cast<Paged>(addresses[12]);
    synchronize();
    at::RegisterPrivateUse1HooksInterface(new Hooks());
  });
  m.def("stream_command",stream_command,pybind11::arg("op"),pybind11::arg("stream")=0,
        pybind11::arg("object")=0,pybind11::arg("flags")=0);
  m.def("record_stream",[](const at::Tensor& t,uint64_t id){Argument a(t);stream_command(12,id,reinterpret_cast<uintptr_t>(a.desc.allocation));});
  pybind11::class_<PagedPlanBridge,std::shared_ptr<PagedPlanBridge>>(m,"NativePagedPlan")
    .def(pybind11::init<const at::Tensor&,const std::vector<uint32_t>&,const std::vector<uint32_t>&>())
    .def("run",&PagedPlanBridge::run);
  m.def("execute", execute);
  m.def("addmm", addmm);
  m.def("layer_norm", layer_norm);
  m.def("rms_norm", rms_norm);
  m.def("spatial", spatial);
  m.def("check_inplace", [](const at::Tensor& out, const at::Tensor& input) {
    at::assert_no_internal_overlap(out);
    at::assert_no_partial_overlap(out, input);
  });
  m.def("check_index_output", [](const at::Tensor& out, const at::Tensor& source, const at::Tensor& index) {
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, source);
    at::assert_no_overlap(out, index);
  });
  m.def("fill", fill);
  m.def("synchronize", synchronize);
}

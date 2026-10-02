using TriangularSolve = int (*)(const Descriptor*, size_t, bool, bool);
TriangularSolve triangular_native = nullptr;
using DeltaForward = int (*)(const Descriptor*, size_t, float, uint32_t);
DeltaForward delta_native = nullptr;

std::tuple<at::Tensor,at::Tensor> delta_forward(const at::Tensor& q,const at::Tensor& k,
    const at::Tensor& v,const at::Tensor& beta,const at::Tensor& decay,const at::Tensor& initial,
    double scale,int64_t chunk) {
  TORCH_CHECK(delta_native, "RUDA sequence API 1 unavailable; rebuild both native libraries");
  for(const auto& t:{q,k,v,beta,decay,initial}) {
    training_tensor(t);
    TORCH_CHECK(t.scalar_type()==at::kFloat && t.device()==q.device(), "native delta operands must share FP32 storage/device");
  }
  TORCH_CHECK(q.dim()==4 && q.sizes()==k.sizes() && v.dim()==4 && beta.dim()==3 &&
              beta.sizes()==decay.sizes() && initial.dim()==4 && std::isfinite(scale) &&
              std::isfinite(float(scale)) && chunk>0 && uint64_t(chunk)<=std::numeric_limits<uint32_t>::max(),
              "invalid gated-delta inputs");
  for(int64_t i=0;i<3;++i) TORCH_CHECK(q.size(i)==v.size(i) && q.size(i)==beta.size(i), "delta batch/head/sequence mismatch");
  TORCH_CHECK(initial.size(0)==q.size(0) && initial.size(1)==q.size(1) &&
              initial.size(2)==q.size(3) && initial.size(3)==v.size(3), "delta initial state shape mismatch");
  auto output=at::empty(v.sizes(),v.options()); auto final=at::empty(initial.sizes(),initial.options());
  std::vector<Argument> args; args.reserve(8);
  for(const auto& t:{q,k,v,beta,decay,initial,output,final})args.emplace_back(t);
  std::vector<Descriptor> ds;for(const auto& arg:args)ds.push_back(arg.desc);
  check(delta_native(ds.data(),ds.size(),float(scale),uint32_t(chunk)));
  return {output,final};
}

at::Tensor triangular_solve(const at::Tensor& a, const at::Tensor& b, bool upper, bool unit) {
  TORCH_CHECK(triangular_native, "RUDA sequence API 1 unavailable; rebuild both native libraries");
  training_tensor(a); training_tensor(b);
  TORCH_CHECK(a.dim()>=2 && a.size(-1)==a.size(-2) && b.dim()==a.dim() &&
              b.size(-2)==a.size(-1) && a.device()==b.device() && a.scalar_type()==b.scalar_type(),
              "invalid triangular solve operands");
  for(int64_t d=0;d<a.dim()-2;++d) TORCH_CHECK(a.size(d)==b.size(d), "broadcast operands before native solve");
  TORCH_CHECK(uint64_t(a.numel())<=std::numeric_limits<uint32_t>::max() &&
              uint64_t(b.numel())<=std::numeric_limits<uint32_t>::max(), "triangular solve exceeds uint32 indexing");
  auto result=at::empty(b.sizes(),b.options());
  std::vector<Argument> args; args.reserve(3);
  for(const auto& t:{a,b,result}) args.emplace_back(t);
  std::vector<Descriptor> ds; for(const auto& arg:args) ds.push_back(arg.desc);
  check(triangular_native(ds.data(),ds.size(),upper,unit));
  return result;
}

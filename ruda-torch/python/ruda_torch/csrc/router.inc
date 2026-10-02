// Optional router-weight API 1, independent of training API 4 and graph API 3.
using RouterCommand = int (*)(uint32_t, const Descriptor*, size_t, uint32_t, bool, float);
RouterCommand router_native = nullptr;

void router_inputs(const at::Tensor& logits, const at::Tensor& indices,
                   uint32_t scoring, double scale) {
  training_tensor(logits);
  validate(indices.device());
  TORCH_CHECK(indices.is_contiguous() && !indices.is_conj() && !indices.is_neg(),
              "router indices must be contiguous resolved integer tensors");
  TORCH_CHECK(indices.scalar_type()==at::kInt || indices.scalar_type()==at::kLong,
              "router indices must be int32 or int64");
  TORCH_CHECK(logits.dim()==2 && indices.dim()==2 && logits.size(0)==indices.size(0)
              && logits.size(1)>0 && indices.size(1)>0 && indices.size(1)<=64
              && indices.size(1)<=logits.size(1) && logits.device()==indices.device(),
              "router expects [tokens,experts] and [tokens,top_k], 1<=top_k<=min(experts,64)");
  TORCH_CHECK(logits.numel()<=std::numeric_limits<uint32_t>::max()
              && logits.size(1)<=std::numeric_limits<uint32_t>::max()
              && logits.size(0)<=std::numeric_limits<uint32_t>::max(),
              "router dimensions exceed uint32 indexing");
  TORCH_CHECK(scoring<=1, "router scoring must be softmax(0) or sigmoid(1)");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(float(scale)) && float(scale)>0,
              "router scale must be finite positive FP32");
  // Device index VALUES are not copied to the host. Kernel guards the original
  // 64-bit value before narrowing: an invalid row returns NaN, never an OOB read.
}
void router_call(uint32_t op, const std::vector<at::Tensor>& tensors,
                 uint32_t scoring, bool renormalize, float scale) {
  TORCH_CHECK(router_native, "RUDA router API 1 unavailable; rebuild both native libraries");
  std::vector<Argument> args; args.reserve(tensors.size());
  for (const auto& t : tensors) args.emplace_back(t);
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for (const auto& a : args) ds.push_back(a.desc);
  check(router_native(op, ds.data(), ds.size(), scoring, renormalize, scale));
}
at::Tensor router_weights_forward(const at::Tensor& logits, const at::Tensor& indices,
                                  uint32_t scoring, bool renormalize, double scale) {
  router_inputs(logits, indices, scoring, scale);
  auto output = at::empty(indices.sizes(), logits.options().dtype(at::kFloat));
  router_call(0, {logits, indices, output}, scoring, renormalize, float(scale));
  return output;
}
at::Tensor router_weights_backward(const at::Tensor& logits, const at::Tensor& indices,
                                   const at::Tensor& gradient, uint32_t scoring,
                                   bool renormalize, double scale) {
  router_inputs(logits, indices, scoring, scale);
  training_tensor(gradient);
  TORCH_CHECK(gradient.scalar_type()==at::kFloat && gradient.sizes()==indices.sizes()
              && gradient.device()==logits.device(), "router weight gradient must be matching FP32");
  auto output = at::empty(logits.sizes(), logits.options());
  router_call(1, {logits, indices, gradient, output}, scoring, renormalize, float(scale));
  return output;
}

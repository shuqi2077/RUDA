// Additive training API 4. Included after Argument in backend.cpp.
using TrainingCommand = int (*)(uint32_t, const Descriptor*, size_t, const float*, size_t);
TrainingCommand training_native = nullptr;

void training_tensor(const at::Tensor& t) {
  validate(t.device());
  TORCH_CHECK(t.is_contiguous() && !t.is_conj() && !t.is_neg(), "training requires contiguous resolved tensors");
  TORCH_CHECK(dtype_code(t.scalar_type()) <= 2, "training supports FP32, FP16 and BF16");
}
void training_same(const at::Tensor& a, const at::Tensor& b) {
  training_tensor(a); training_tensor(b);
  TORCH_CHECK(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() && a.device()==b.device(),
              "training tensor shape/dtype/device mismatch");
}
void training_mutable(const at::Tensor& t) {
  training_tensor(t);
  TORCH_CHECK(!c10::GradMode::is_enabled(), "optimizer mutation requires no_grad");
  TORCH_CHECK(!t.is_inference(), "training mutation does not accept inference tensors");
  at::assert_no_internal_overlap(t);
}
void training_call(uint32_t op, const std::vector<at::Tensor>& tensors, const std::vector<float>& scalars) {
  TORCH_CHECK(training_native, "RUDA training API is not initialized; rebuild both native libraries");
  std::vector<Argument> args; args.reserve(tensors.size());
  for (const auto& t: tensors) { training_tensor(t); args.emplace_back(t); }
  std::vector<Descriptor> ds; ds.reserve(args.size());
  for (const auto& a: args) ds.push_back(a.desc);
  check(training_native(op,ds.data(),ds.size(),scalars.data(),scalars.size()));
}
void training_norm_inputs(const at::Tensor& x, const std::optional<at::Tensor>& w) {
  training_tensor(x);
  TORCH_CHECK(x.dim()>0 && x.size(-1)>0, "RMSNorm requires a positive last dimension");
  if (w) {
    training_tensor(*w);
    TORCH_CHECK(w->dim()==1 && w->numel()==x.size(-1) && w->device()==x.device(), "RMSNorm weight shape/device mismatch");
    TORCH_CHECK(w->scalar_type()==x.scalar_type() || w->scalar_type()==at::kFloat, "RMSNorm weight must match input dtype or be FP32");
  }
}
std::tuple<at::Tensor,at::Tensor> training_rms_forward(const at::Tensor& x,const std::optional<at::Tensor>& w,double eps) {
  training_norm_inputs(x,w);
  TORCH_CHECK(std::isfinite(eps) && std::isfinite(float(eps)) && float(eps)>0, "epsilon must be finite and positive in FP32");
  auto y=at::empty(x.sizes(),x.options());
  auto stats=at::empty({x.numel()/x.size(-1)},x.options().dtype(at::kFloat));
  training_call(0,{x,w.value_or(x),y,stats},{float(eps),w?1.f:0.f});
  return {y,stats};
}
std::tuple<at::Tensor,at::Tensor> training_rms_backward(const at::Tensor& x,const std::optional<at::Tensor>& w,
    const at::Tensor& dy,const at::Tensor& stats,bool need_x,bool need_w) {
  training_norm_inputs(x,w);training_same(x,dy);training_tensor(stats);
  TORCH_CHECK(stats.scalar_type()==at::kFloat && stats.dim()==1 && stats.numel()==x.numel()/x.size(-1), "RMSNorm saved statistics mismatch");
  TORCH_CHECK(!need_w || w, "weight gradient requested without a weight");
  at::Tensor dx,dw;
  if(need_x) dx=at::empty(x.sizes(),x.options());
  if(need_w) dw=at::empty(w->sizes(),w->options());
  training_call(1,{x,w.value_or(x),dy,stats,need_x?dx:x,need_w?dw:x},{w?1.f:0.f,need_x?1.f:0.f,need_w?1.f:0.f});
  return {dx,dw};
}

void training_layer_inputs(const at::Tensor& x,const std::optional<at::Tensor>& w,const std::optional<at::Tensor>& b) {
  training_tensor(x);
  TORCH_CHECK(x.dim()>0 && x.size(-1)>0,"LayerNorm requires a positive last dimension");
  auto check_affine=[&](const at::Tensor& t,const char* name) {
    training_tensor(t);
    TORCH_CHECK(t.dim()==1 && t.numel()==x.size(-1) && t.device()==x.device(),name," shape/device mismatch");
    TORCH_CHECK(t.scalar_type()==x.scalar_type() || t.scalar_type()==at::kFloat,name," must match input dtype or be FP32");
  };
  if(w) check_affine(*w,"LayerNorm weight");
  if(b) check_affine(*b,"LayerNorm bias");
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> training_layer_forward(
    const at::Tensor& x,const std::optional<at::Tensor>& w,const std::optional<at::Tensor>& b,double eps) {
  training_layer_inputs(x,w,b);
  TORCH_CHECK(std::isfinite(eps) && std::isfinite(float(eps)) && float(eps)>0,"epsilon must be finite and positive in FP32");
  auto y=at::empty(x.sizes(),x.options());
  const auto rows=x.numel()/x.size(-1);
  auto mean=at::empty({rows},x.options().dtype(at::kFloat));
  auto rstd=at::empty({rows},x.options().dtype(at::kFloat));
  training_call(9,{x,w.value_or(x),b.value_or(x),y,mean,rstd},{float(eps),w?1.f:0.f,b?1.f:0.f});
  return {y,mean,rstd};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> training_layer_backward(
    const at::Tensor& x,const std::optional<at::Tensor>& w,const std::optional<at::Tensor>& b,
    const at::Tensor& dy,const at::Tensor& mean,const at::Tensor& rstd,
    bool need_x,bool need_w,bool need_b) {
  training_layer_inputs(x,w,b);training_same(x,dy);training_tensor(mean);training_tensor(rstd);
  const auto rows=x.numel()/x.size(-1);
  TORCH_CHECK(mean.scalar_type()==at::kFloat && rstd.scalar_type()==at::kFloat && mean.dim()==1 && rstd.dim()==1
              && mean.numel()==rows && rstd.numel()==rows,"LayerNorm saved statistics mismatch");
  TORCH_CHECK(!need_w || w,"weight gradient requested without LayerNorm weight");
  TORCH_CHECK(!need_b || b,"bias gradient requested without LayerNorm bias");
  at::Tensor dx,dw,db;
  if(need_x) dx=at::empty(x.sizes(),x.options());
  if(need_w) dw=at::empty(w->sizes(),w->options());
  if(need_b) db=at::empty(b->sizes(),b->options());
  training_call(10,{x,w.value_or(x),b.value_or(x),dy,mean,rstd,
                    need_x?dx:x,need_w?dw:x,need_b?db:x},
                    {w?1.f:0.f,b?1.f:0.f,need_x?1.f:0.f,need_w?1.f:0.f,need_b?1.f:0.f});
  return {dx,dw,db};
}

at::Tensor training_silu_forward(const at::Tensor& gate,const at::Tensor& up) {
  training_same(gate,up);auto out=at::empty(gate.sizes(),gate.options());
  training_call(2,{gate,up,out},{});return out;
}
std::tuple<at::Tensor,at::Tensor> training_silu_backward(const at::Tensor& gate,const at::Tensor& up,
    const at::Tensor& dy,bool need_gate,bool need_up) {
  training_same(gate,up);training_same(gate,dy);at::Tensor dg,du;
  if(need_gate) dg=at::empty(gate.sizes(),gate.options());
  if(need_up) du=at::empty(up.sizes(),up.options());
  training_call(3,{gate,up,dy,need_gate?dg:gate,need_up?du:up},{need_gate?1.f:0.f,need_up?1.f:0.f});
  return {dg,du};
}
void training_unscale(std::vector<at::Tensor> grads,double inverse,at::Tensor workspace,at::Tensor found) {
  TORCH_CHECK(!grads.empty() && grads.size()<=4096,"unscale requires 1..4096 gradients");
  TORCH_CHECK(std::isfinite(inverse) && std::isfinite(float(inverse)) && float(inverse)>0,"inverse scale must be finite positive FP32");
  training_mutable(workspace);training_mutable(found);
  TORCH_CHECK(workspace.scalar_type()==at::kFloat && workspace.dim()==1 && found.scalar_type()==at::kFloat && found.numel()==1,"unscale workspace must be FP32");
  int64_t flags=0;
  for(size_t i=0;i<grads.size();++i) {
    training_mutable(grads[i]);
    TORCH_CHECK(!grads[i].requires_grad(),"higher-order gradients are unsupported by optimizer unscale");
    flags+=std::clamp<int64_t>((grads[i].numel()+31)/32,1,1024);
    at::assert_no_overlap(grads[i],workspace);at::assert_no_overlap(grads[i],found);
    for(size_t j=0;j<i;++j) at::assert_no_overlap(grads[i],grads[j]);
  }
  TORCH_CHECK(workspace.numel()==flags,"unscale workspace length mismatch");
  at::assert_no_overlap(workspace,found);
  auto args=grads;args.push_back(workspace);args.push_back(found);
  training_call(4,args,{float(inverse)});
  for(auto& g:grads) g.unsafeGetTensorImpl()->bump_version();
  workspace.unsafeGetTensorImpl()->bump_version();found.unsafeGetTensorImpl()->bump_version();
}
void training_adamw(at::Tensor p,const at::Tensor& g,at::Tensor master,at::Tensor m,at::Tensor v,
                    double lr,double b1,double b2,double eps,double decay,double c1,double c2) {
  training_mutable(p);training_same(p,g);training_mutable(m);training_mutable(v);training_mutable(master);
  TORCH_CHECK(!g.requires_grad(),"AdamW does not support differentiable gradients");
  for(const auto& t:{master,m,v}) TORCH_CHECK(t.scalar_type()==at::kFloat && t.sizes()==p.sizes(),"AdamW states must be matching FP32 tensors");
  const bool separate=p.scalar_type()!=at::kFloat;
  if(!separate) TORCH_CHECK(master.is_same(p),"FP32 AdamW master must be the parameter itself");
  std::vector<at::Tensor> writes={p,m,v};if(separate) writes.push_back(master);
  for(size_t i=0;i<writes.size();++i) {
    at::assert_no_overlap(writes[i],g);
    for(size_t j=0;j<i;++j)at::assert_no_overlap(writes[i],writes[j]);
  }
  std::vector<float> values={float(lr),float(b1),float(b2),float(eps),float(decay),float(c1),float(c2)};
  for(float s:values) TORCH_CHECK(std::isfinite(s),"AdamW scalar is not finite FP32");
  TORCH_CHECK(lr>=0 && b1>=0 && values[1]<1 && b2>=0 && values[2]<1 && values[3]>0 && decay>=0
              && values[5]>0 && values[5]<=1 && values[6]>0 && values[6]<=1,"invalid AdamW hyperparameters");
  training_call(5,{p,g,master,m,v},values);
  for(auto& t:writes)t.unsafeGetTensorImpl()->bump_version();
}


// All buffers are contiguous. Sort actual device-address intervals so cross-
// parameter/state aliases are checked in O(N log N), not with an N-by-N loop.
void training_disjoint(const std::vector<at::Tensor>& tensors) {
  std::vector<std::pair<uintptr_t,uintptr_t>> spans; spans.reserve(tensors.size());
  for (const auto& t : tensors) {
    if (!t.numel()) continue;
    const auto begin = reinterpret_cast<uintptr_t>(t.const_data_ptr());
    const auto bytes = size_t(t.numel()) * t.element_size();
    TORCH_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin, "training address interval overflow");
    spans.emplace_back(begin, begin + bytes);
  }
  std::sort(spans.begin(),spans.end());
  for(size_t i=1;i<spans.size();++i)
    TORCH_CHECK(spans[i].first >= spans[i-1].second, "optimizer buffers must not overlap");
}

void training_analyze(const std::vector<at::Tensor>& grads, double inverse,
                      at::Tensor workspace, at::Tensor report, bool with_norm) {
  TORCH_CHECK(!grads.empty() && grads.size()<=4096, "analysis requires 1..4096 gradients");
  TORCH_CHECK(std::isfinite(inverse) && std::isfinite(float(inverse)) && float(inverse)>0,
              "inverse scale must be finite positive FP32");
  training_mutable(workspace); training_mutable(report);
  TORCH_CHECK(workspace.scalar_type()==at::kFloat && workspace.dim()==1
      && report.scalar_type()==at::kFloat && report.dim()==1 && report.numel()==3,
      "gradient statistics workspace/report must be FP32 (report length 3)");
  int64_t rows=0;
  for(const auto& g:grads) {
    training_tensor(g);
    TORCH_CHECK(!g.requires_grad(), "gradient analysis does not support higher-order derivatives");
    rows+=std::clamp<int64_t>((g.numel()+31)/32,1,1024);
  }
  TORCH_CHECK(workspace.numel()==rows*3, "gradient statistics workspace length mismatch");
  auto args=grads; args.push_back(workspace); args.push_back(report);
  training_disjoint(args);
  // Scratch may have been partially written even if a device operation fails.
  workspace.unsafeGetTensorImpl()->bump_version(); report.unsafeGetTensorImpl()->bump_version();
  training_call(6,args,{float(inverse),with_norm?1.f:0.f});
}

// API 4. A bounded hierarchy merges at most 1024 rows per warp per stage.
void training_analyze_hierarchical(const std::vector<at::Tensor>& grads, double inverse,
    at::Tensor workspace, at::Tensor scratch, at::Tensor report, bool with_norm) {
  TORCH_CHECK(!grads.empty() && grads.size()<=4096, "analysis requires 1..4096 gradients");
  TORCH_CHECK(std::isfinite(inverse) && std::isfinite(float(inverse)) && float(inverse)>0,
              "inverse scale must be finite positive FP32");
  for(const auto& t:{workspace,scratch,report}) {
    training_mutable(t);
    TORCH_CHECK(t.scalar_type()==at::kFloat && t.dim()==1, "statistics buffers must be one-dimensional FP32");
  }
  TORCH_CHECK(report.numel()==3, "gradient report requires three elements");
  int64_t rows=0;
  for(const auto& g:grads) {
    training_tensor(g);
    TORCH_CHECK(!g.requires_grad(), "gradient analysis does not support higher-order derivatives");
    rows+=std::clamp<int64_t>((g.numel()+31)/32,1,1024);
  }
  int64_t needed=0;
  for(auto count=rows;count>1024;) { count=(count+1023)/1024; needed+=count*3; }
  TORCH_CHECK(workspace.numel()==rows*3, "gradient statistics workspace length mismatch");
  TORCH_CHECK(scratch.numel()==needed, "hierarchical statistics scratch length mismatch");
  auto args=grads;args.push_back(workspace);args.push_back(scratch);args.push_back(report);
  training_disjoint(args); // validate before version counters or native writes
  workspace.unsafeGetTensorImpl()->bump_version();
  if(needed) scratch.unsafeGetTensorImpl()->bump_version();
  report.unsafeGetTensorImpl()->bump_version();
  training_call(8,args,{float(inverse),with_norm?1.f:0.f});
}

void training_adamw_batch(const std::vector<at::Tensor>& parameters,
    const std::vector<at::Tensor>& gradients, const std::vector<at::Tensor>& masters,
    const std::vector<at::Tensor>& first, const std::vector<at::Tensor>& second,
    const std::vector<std::vector<double>>& hypers, double inverse, double clip) {
  const auto n=parameters.size();
  TORCH_CHECK(n>=1 && n<=4096 && gradients.size()==n && masters.size()==n
      && first.size()==n && second.size()==n && hypers.size()==n, "AdamW batch list length mismatch");
  TORCH_CHECK(std::isfinite(inverse) && std::isfinite(float(inverse)) && float(inverse)>0
      && std::isfinite(clip) && clip>=0 && clip<=1, "invalid fused scale or clipping factor");
  std::vector<at::Tensor> tensors, occupied, writes;
  tensors.reserve(n*5); occupied.reserve(n*5); writes.reserve(n*4);
  std::vector<float> scalars={float(inverse),float(clip)}; scalars.reserve(2+n*7);
  for(size_t i=0;i<n;++i) {
    const auto& p=parameters[i]; const auto& g=gradients[i]; const auto& master=masters[i];
    const auto& m=first[i]; const auto& v=second[i];
    training_mutable(p); training_same(p,g); training_mutable(master); training_mutable(m); training_mutable(v);
    TORCH_CHECK(!g.requires_grad(), "AdamW does not support differentiable gradients");
    for(const auto& t:{master,m,v})
      TORCH_CHECK(t.scalar_type()==at::kFloat && t.sizes()==p.sizes(), "AdamW states must be matching FP32 tensors");
    const bool separate=p.scalar_type()!=at::kFloat;
    if(!separate) TORCH_CHECK(master.is_same(p), "FP32 AdamW master must be parameter itself");
    TORCH_CHECK(hypers[i].size()==7, "AdamW requires seven hyperparameters per tensor");
    std::vector<float> h; h.reserve(7);
    for(double s:hypers[i]) { TORCH_CHECK(std::isfinite(s) && std::isfinite(float(s)), "AdamW hyperparameter not finite FP32"); h.push_back(float(s)); }
    TORCH_CHECK(hypers[i][0]>=0 && hypers[i][1]>=0 && h[1]<1 && hypers[i][2]>=0 && h[2]<1
        && h[3]>0 && hypers[i][4]>=0 && h[5]>0 && h[5]<=1 && h[6]>0 && h[6]<=1, "invalid AdamW hyperparameters");
    tensors.insert(tensors.end(),{p,g,master,m,v});
    occupied.insert(occupied.end(),{p,g,m,v}); writes.insert(writes.end(),{p,m,v});
    if(separate) {occupied.push_back(master);writes.push_back(master);}
    scalars.insert(scalars.end(),h.begin(),h.end());
  }
  training_disjoint(occupied); // reject every invalid entry before the first launch
  // Native partial failure is not transactional. Conservatively mark every
  // destination modified before entering native code, and let Python poison the
  // optimizer on error. Unchanged gradient versions are deliberately preserved.
  for(auto& t:writes)t.unsafeGetTensorImpl()->bump_version();
  training_call(7,tensors,scalars);
}

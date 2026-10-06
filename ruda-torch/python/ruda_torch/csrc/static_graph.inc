// Included inside backend.cpp's anonymous namespace, after Argument.
// Optional extension API 3; deliberately does not change base tensor ABI 10.
struct StaticNode { uint32_t op, a, b; float scalar; };
static_assert(sizeof(StaticNode)==16 && alignof(StaticNode)==4, "static graph ABI layout");
using StaticGraphCommand = int (*)(uint32_t,void**,const Descriptor*,size_t,
                                   const StaticNode*,size_t,size_t,uint32_t,uint64_t*);
StaticGraphCommand static_graph_native=nullptr;

class StaticGraphBridge {
  struct Snapshot {
    void* allocation;
    void* address;
    size_t offset;
    c10::ScalarType dtype;
    std::vector<int64_t> shape,strides;
  };
  void* plan_=nullptr;
  std::vector<at::Tensor> tensors_;
  std::vector<Snapshot> snapshots_;
  size_t inputs_=0;
  uint64_t stream_=0, edges_=0;
  bool tracked_=false;
  std::mutex mutex_;
  uint64_t command(uint32_t op) {
    uint64_t value=0;
    check(static_graph_native(op,&plan_,nullptr,0,nullptr,0,0,0,&value));
    return value;
  }
  void check_bindings() const {
    TORCH_CHECK(plan_,"static graph is closed");
    TORCH_CHECK(stream_command(0)==stream_,"static graph replay must use its creation stream");
    for(size_t i=0;i<tensors_.size();++i) {
      const auto& t=tensors_[i]; const auto& s=snapshots_[i];
      TORCH_CHECK(!t.requires_grad(),"static graph is inference-only");
      // Replay needs identity/metadata checks, not fresh Descriptor vectors.
      // Keep every safety guard from Argument, then compare borrowed array views.
      validate(t.device());
      TORCH_CHECK(!t.is_conj() && !t.is_neg(), "RUDA does not support unresolved view bits");
      const auto storage=t.storage();
      TORCH_CHECK(storage.data_ptr().get_deleter()==release,"tensor is not allocated by RUDA");
      TORCH_CHECK(storage.data_ptr().get_context()==s.allocation
        && size_t(t.storage_offset())*t.element_size()==s.offset
        && t.data_ptr()==s.address && t.scalar_type()==s.dtype
        && t.sizes()==c10::IntArrayRef(s.shape) && t.strides()==c10::IntArrayRef(s.strides),
        "static graph binding changed: keep storage, shape, strides and dtype fixed");
    }
    for(size_t i=inputs_;i<tensors_.size();++i) {
      TORCH_CHECK(!tensors_[i].is_inference() || c10::InferenceMode::is_enabled(),
                  "replay writing inference tensors requires torch.inference_mode()");
    }
  }
 public:
  StaticGraphBridge(const std::vector<at::Tensor>& tensors,size_t inputs,
                    const std::vector<uint32_t>& words,const std::vector<float>& scalars,
                    bool tracked,bool infer):tensors_(tensors),inputs_(inputs),tracked_(tracked) {
    TORCH_CHECK(static_graph_native,"native static graph API is unavailable; rebuild Rust and C++");
    TORCH_CHECK(inputs>0 && scalars.size()>0 && scalars.size()<=256
      && tensors.size()<=512 && tensors.size()==inputs+scalars.size()
      && words.size()==scalars.size()*3,"invalid static graph header");
    std::vector<std::unique_ptr<Argument>> args;
    std::vector<Descriptor> descriptors;
    for(const auto& t:tensors_) {
      validate(t.device());
      TORCH_CHECK(t.layout()==c10::Layout::Strided && t.is_contiguous()
        && t.dim()>=0 && t.dim()<=8 && t.numel()>0 && t.numel()<=std::numeric_limits<uint32_t>::max(),
        "static graph requires nonempty contiguous rank 0..8 tensors");
      TORCH_CHECK(dtype_code(t.scalar_type())<=2,"static graph supports FP32/FP16/BF16 only");
      TORCH_CHECK(!t.requires_grad(),"static graph is inference-only");
      args.emplace_back(std::make_unique<Argument>(t));
      descriptors.push_back(args.back()->desc);
      snapshots_.push_back({args.back()->desc.allocation,t.data_ptr(),args.back()->desc.offset_bytes,
                            t.scalar_type(),t.sizes().vec(),t.strides().vec()});
    }
    std::vector<StaticNode> nodes;
    for(size_t i=0;i<scalars.size();++i) {
      const uint32_t op=words[i*3], a=words[i*3+1], b=words[i*3+2];
      const size_t out=inputs+i;
      TORCH_CHECK(a<out,"static graph node refers to a future input");
      const auto& x=tensors_[a]; const auto& y=tensors_[out];
      TORCH_CHECK((op==7 || op==30 || op==104 || op==105 || (op>=112 && op<=120) || x.sizes()==y.sizes())
        && (op==114 || x.scalar_type()==y.scalar_type()),"static graph output mismatch");
      if(op==112 || op==114) {
        TORCH_CHECK(b==a && scalars[i]==0.0f && !std::signbit(scalars[i])
          && (op==112 ? x.numel()==y.numel() : x.sizes()==y.sizes()),"invalid graph reshape/cast");
      } else if(op==113) {
        TORCH_CHECK(b==a && x.dim()==y.dim() && std::isfinite(scalars[i]) && scalars[i]>=0
          && std::floor(scalars[i])==scalars[i] && scalars[i]<float(1u<<(3*x.dim())),"invalid graph permutation encoding");
        const auto code=static_cast<uint32_t>(scalars[i]); uint32_t seen=0;
        for(int d=0;d<x.dim();++d) {
          const uint32_t axis=(code>>(3*d))&7u;
          TORCH_CHECK(axis<uint32_t(x.dim()) && !(seen&(1u<<axis)) && y.size(d)==x.size(axis),"invalid graph permutation axes/shape");
          seen|=1u<<axis;
        }
      } else if(op==115) {
        TORCH_CHECK(b==a && scalars[i]==0.0f && !std::signbit(scalars[i]) && y.dim()>=x.dim(),"invalid graph expansion");
        for(int d=0;d<x.dim();++d) TORCH_CHECK(x.size(d)==1 || x.size(d)==y.size(y.dim()-x.dim()+d),"invalid graph expansion shape");
      } else if(op>=116 && op<=118) {
        TORCH_CHECK(b<out && tensors_[b].scalar_type()==x.scalar_type() && std::isfinite(scalars[i])
          && (op==116 || (scalars[i]==0.0f && !std::signbit(scalars[i]))),"invalid broadcast graph inputs");
        const auto& r=tensors_[b]; const int rank=std::max(x.dim(),r.dim());
        TORCH_CHECK(y.dim()==rank,"invalid broadcast output rank");
        for(int d=0;d<rank;++d) {
          const auto lhs=d<rank-x.dim() ? 1 : x.size(d-rank+x.dim());
          const auto rhs=d<rank-r.dim() ? 1 : r.size(d-rank+r.dim());
          TORCH_CHECK((lhs==rhs || lhs==1 || rhs==1) && y.size(d)==std::max(lhs,rhs),"invalid broadcast shape");
        }
      } else if(op==119 || op==120) {
        TORCH_CHECK(b==a && std::isfinite(scalars[i]) && scalars[i]>0 && scalars[i]<(1u<<x.dim())
          && std::floor(scalars[i])==scalars[i],"invalid squeezed reduction mask");
        const auto mask=static_cast<uint32_t>(scalars[i]); int out_dim=0;
        for(int d=0;d<x.dim();++d) if(!(mask&(1u<<d))) {
          TORCH_CHECK(out_dim<y.dim() && y.size(out_dim)==x.size(d),"invalid squeezed reduction shape"); ++out_dim;
        }
        TORCH_CHECK(out_dim==y.dim(),"invalid squeezed reduction rank");
      } else if(op==0 || op==3 || (op>=9 && op<=14) || op==17 || (op>=19 && op<=25) || op==27 || (op>=35 && op<=40) || op==43 || op==45 || op==108) {
        TORCH_CHECK(b==a && scalars[i]==0.0f && !std::signbit(scalars[i]),"invalid unary static graph node");
      } else if(op==1 || op==2 || op==8 || op==15 || op==16 || op==18 || op==101) {
        TORCH_CHECK(b<out && tensors_[b].sizes()==x.sizes() && tensors_[b].scalar_type()==x.scalar_type(),
                    "static graph broadcasting/type promotion is not supported");
        TORCH_CHECK(std::isfinite(scalars[i]) && (op==1 || (scalars[i]==0.0f && !std::signbit(scalars[i]))),
                    "invalid static graph scalar");
      } else if(op>=109 && op<=111) {
        TORCH_CHECK(b==a && std::isfinite(scalars[i]), "invalid scalar graph operation");
      } else if(op==7 || op==30) {
        TORCH_CHECK(b<out, "future matrix graph input");
        const auto& r=tensors_[b]; const int rank=op==7 ? 2 : 3;
        TORCH_CHECK(x.dim()==rank && r.dim()==rank && y.dim()==rank && x.scalar_type()==r.scalar_type()
          && x.size(-1)==r.size(-2) && y.size(-2)==x.size(-2) && y.size(-1)==r.size(-1)
          && (rank==2 || (x.size(0)==r.size(0) && y.size(0)==x.size(0)))
          && scalars[i]==0.0f && !std::signbit(scalars[i]), "invalid graph matrix multiply");
      } else if(op==102 || op==103 || op==106 || op==107) {
        TORCH_CHECK(std::isfinite(scalars[i]) && scalars[i]>=0 && scalars[i]<x.dim()
          && std::floor(scalars[i])==scalars[i] && b<out, "invalid graph softmax axis/input");
        TORCH_CHECK(tensors_[b].sizes()==x.sizes() && tensors_[b].scalar_type()==x.scalar_type()
          && (op>=106 || b==a), "graph softmax gradient/output mismatch");
      } else if(op==104 || op==105) {
        TORCH_CHECK(std::isfinite(scalars[i]) && scalars[i]>0 && scalars[i]<(1u<<x.dim())
          && std::floor(scalars[i])==scalars[i] && b==a && y.dim()==x.dim(), "invalid graph reduction mask");
        const auto mask=static_cast<uint32_t>(scalars[i]);
        for(int d=0;d<x.dim();++d) TORCH_CHECK(y.size(d)==((mask & (1u<<d)) ? 1 : x.size(d)),
          "graph reduction output mismatch");
      } else if(op==100) {
        TORCH_CHECK(x.dim()>0 && std::isfinite(scalars[i]) && scalars[i]>0.0f,"invalid RMSNorm epsilon/rank");
        TORCH_CHECK(x.numel()/x.size(-1)<=std::numeric_limits<uint32_t>::max()/32,
                    "RMSNorm row grid overflow");
        if(b!=std::numeric_limits<uint32_t>::max()) {
          TORCH_CHECK(b<out && tensors_[b].dim()==1 && tensors_[b].size(0)==x.size(-1)
            && tensors_[b].scalar_type()==x.scalar_type(),"invalid RMSNorm weight");
        }
      } else { TORCH_CHECK(false,"unsupported static graph operator; no CPU fallback"); }
      nodes.push_back({op,a,b,scalars[i]});
    }
    // Inclusive lifetimes: never overwrite an input used by the SAME node.
    // Inputs never share writable storage. Leaf outputs remain live until end.
    // Python additionally dedicates storage to every externally requested result.
    std::vector<size_t> last(tensors_.size());
    std::vector<bool> consumed(tensors_.size(),false);
    for(size_t i=0;i<last.size();++i)last[i]=i;
    for(size_t i=0;i<nodes.size();++i) {
      const auto& n=nodes[i];const size_t out=inputs+i;
      last[n.a]=std::max(last[n.a],out);consumed[n.a]=true;
      if(n.b!=std::numeric_limits<uint32_t>::max()) {
        last[n.b]=std::max(last[n.b],out);consumed[n.b]=true;
      }
    }
    for(size_t i=0;i<tensors_.size();++i)
      if(i<inputs || !consumed[i])last[i]=tensors_.size();
    for(size_t i=inputs;i<tensors_.size();++i) {
      for(size_t j=0;j<i;++j) {
        if(tensors_[i].storage().unsafeGetStorageImpl()==tensors_[j].storage().unsafeGetStorageImpl()) {
          const auto& a=snapshots_[i];const auto& b=snapshots_[j];
          TORCH_CHECK(j>=inputs && last[j]<i && a.allocation==b.allocation
              && a.address==b.address && a.offset==b.offset && a.dtype==b.dtype && a.shape==b.shape
              && a.strides==b.strides,
              "static graph writable slots need distinct storage or nonoverlapping lifetimes with exact buffer reuse");
        } else {
          at::assert_no_overlap(tensors_[i],tensors_[j]);
        }
      }
    }
    stream_=stream_command(0);
    check(static_graph_native(0,&plan_,descriptors.data(),descriptors.size(),nodes.data(),nodes.size(),
                              inputs,static_cast<uint32_t>(tracked)|(static_cast<uint32_t>(infer)<<1),&edges_));
    TORCH_CHECK(plan_,"native graph build returned a null plan");
  }
  ~StaticGraphBridge() {
    if(plan_) {
      uint64_t ignored=0;
      if(static_graph_native(4,&plan_,nullptr,0,nullptr,0,0,0,&ignored))
        std::fprintf(stderr,"RUDA static graph cleanup failed (native resources retained): %s\n",error_native());
    }
  }
  void run(bool eager=false) {
    std::lock_guard<std::mutex> lock(mutex_); check_bindings();
    // Account for raw native writes in PyTorch's tensor mutation tracking.
    for(size_t i=inputs_;i<tensors_.size();++i) tensors_[i].unsafeGetTensorImpl()->bump_version();
    command(eager?5:1);
  }
  void synchronize() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_,"static graph is closed");command(2);}
  bool query() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_,"static graph is closed");return command(3)!=0;}
  bool query_completion() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_ && tracked_,"tracked graph required");return command(6)!=0;}
  void wait_completion() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_ && tracked_,"tracked graph required");command(7);}
  void close() {std::lock_guard<std::mutex> lock(mutex_);if(plan_)command(4);tensors_.clear();snapshots_.clear();}
  uint64_t edge_count() const {return edges_;}
  uint64_t stream_id() const {return stream_;}
};

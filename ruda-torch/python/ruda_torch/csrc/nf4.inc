using NF4Decode = int (*)(const Descriptor*, size_t, uint32_t, uint32_t);
NF4Decode nf4_native = nullptr;
using NF4Matmul = int (*)(const Descriptor*, size_t, uint32_t, uint32_t, uint32_t, bool);
NF4Matmul nf4_matmul_native = nullptr;

at::Tensor nf4_matmul(const at::Tensor& input, const at::Tensor& packed,
                     const at::Tensor& scales, const at::Tensor& table,
                     int64_t columns, int64_t width, int64_t block, bool backward) {
  TORCH_CHECK(nf4_matmul_native, "RUDA NF4 matmul API 1 unavailable; rebuild both native libraries");
  training_tensor(input); training_tensor(scales); training_tensor(table); validate(packed.device());
  TORCH_CHECK(input.dim()==2 && (input.scalar_type()==at::kHalf || input.scalar_type()==at::kBFloat16),
              "NF4 Tensor Core matmul requires FP16/BF16 input");
  const auto limit=std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(columns>0 && width>0 && block>0 && block%2==0 &&
              uint64_t(columns)<=limit && uint64_t(width)<=limit && uint64_t(block)<=limit &&
              uint64_t(columns)<=limit/uint64_t(width), "invalid NF4 matmul geometry");
  TORCH_CHECK(packed.dim()==1 && packed.scalar_type()==at::kByte && packed.is_contiguous() &&
              !packed.is_conj() && !packed.is_neg() && scales.dim()==1 && table.dim()==1 &&
              scales.scalar_type()==at::kFloat && table.scalar_type()==at::kFloat && table.numel()==16,
              "invalid NF4 packed tensors");
  for(const auto& t:{packed,scales,table}) TORCH_CHECK(t.device()==input.device(), "NF4 devices must match");
  const uint64_t size=uint64_t(columns)*uint64_t(width);
  TORCH_CHECK(uint64_t(packed.numel())==(size+1)/2 && uint64_t(scales.numel())==(size+block-1)/block &&
              input.size(1)==(backward?columns:width) && uint64_t(input.numel())<=limit &&
              uint64_t(input.size(0))<=limit/uint64_t(backward?width:columns), "NF4 storage/matmul shape mismatch");
  auto output=at::empty({input.size(0),backward?width:columns},input.options().dtype(backward?at::kFloat:input.scalar_type()));
  std::vector<Argument> args;args.reserve(5);
  for(const auto& t:{input,packed,scales,table,output}) args.emplace_back(t);
  std::vector<Descriptor> ds;for(const auto& arg:args)ds.push_back(arg.desc);
  check(nf4_matmul_native(ds.data(),ds.size(),uint32_t(columns),uint32_t(width),uint32_t(block),backward));
  return output;
}

at::Tensor nf4_decode(const at::Tensor& packed, const at::Tensor& scales,
                      const at::Tensor& table, int64_t begin, int64_t rows,
                      int64_t width, int64_t block, uint32_t dtype) {
  TORCH_CHECK(nf4_native, "RUDA NF4 API 1 unavailable; rebuild both libraries");
  validate(packed.device());
  TORCH_CHECK(packed.dim()==1 && packed.scalar_type()==at::kByte,
              "NF4 weights must be packed bytes");
  TORCH_CHECK(scales.dim()==1 && scales.scalar_type()==at::kFloat &&
              table.dim()==1 && table.numel()==16 && table.scalar_type()==at::kFloat,
              "NF4 requires FP32 block scales and a 16-value codebook");
  for (const auto& t : {packed, scales, table}) {
    TORCH_CHECK(t.device()==packed.device() && t.is_contiguous() && !t.is_conj() && !t.is_neg(),
                "NF4 inputs must be contiguous on the same device");
  }
  TORCH_CHECK(begin>=0 && rows>0 && width>0 && block>0 && block%2==0 && dtype<=2,
              "invalid NF4 tile geometry/dtype");
  const uint64_t limit = std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(uint64_t(begin)<=limit && uint64_t(rows)<=limit &&
              uint64_t(width)<=limit && uint64_t(block)<=limit,
              "NF4 tile exceeds uint32 indexing");
  TORCH_CHECK(uint64_t(begin)+uint64_t(rows)<=limit/uint64_t(width),
              "NF4 tile exceeds uint32 indexing");
  const uint64_t end = (uint64_t(begin)+uint64_t(rows))*uint64_t(width);
  TORCH_CHECK(end<=limit && (end+1)/2<=uint64_t(packed.numel()) &&
              (end+uint64_t(block)-1)/uint64_t(block)<=uint64_t(scales.numel()),
              "NF4 tile exceeds packed storage");
  auto scalar_type = dtype==0 ? at::kFloat : dtype==1 ? at::kHalf : at::kBFloat16;
  auto output = at::empty({rows,width}, packed.options().dtype(scalar_type));
  std::vector<Argument> args;
  args.reserve(4);
  for (const auto& t : {packed, scales, table, output}) args.emplace_back(t);
  std::vector<Descriptor> ds;
  for (const auto& a : args) ds.push_back(a.desc);
  check(nf4_native(ds.data(), ds.size(), uint32_t(uint64_t(begin)*width), uint32_t(block)));
  return output;
}

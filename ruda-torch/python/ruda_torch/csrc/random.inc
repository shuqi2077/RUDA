using RandomFill = int (*)(const Descriptor*,uint64_t,uint64_t,uint32_t,float,float);
RandomFill random_fill_native=nullptr;
at::Tensor random_fill(const at::Tensor& output,uint64_t seed,uint64_t counter,uint32_t distribution,double first,double second) {
  TORCH_CHECK(random_fill_native,"RUDA random API 1 unavailable; rebuild both native libraries");
  training_tensor(output);
  TORCH_CHECK(distribution<=2 && std::isfinite(first) && std::isfinite(second)
    && std::isfinite(float(first)) && std::isfinite(float(second)),"invalid random distribution parameters");
  TORCH_CHECK(distribution!=0 || first<=second,"invalid uniform bounds");
  TORCH_CHECK(distribution!=1 || second>=0,"invalid normal standard deviation");
  TORCH_CHECK(distribution!=2 || (first>=0 && first<=1),"invalid Bernoulli probability");
  Argument argument(output);
  check(random_fill_native(&argument.desc,seed,counter,distribution,float(first),float(second)));
  return output;
}

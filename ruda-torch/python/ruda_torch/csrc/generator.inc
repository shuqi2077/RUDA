// Host state supplies only launch seeds; ruRAND generates every sample on device.
class Generator final : public c10::GeneratorImpl {
  at::CPUGeneratorImpl seeds_;
 public:
  explicit Generator(uint64_t seed=c10::default_rng_seed_val)
      : c10::GeneratorImpl(c10::Device(c10::DeviceType::PrivateUse1,0),
                           c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)), seeds_(seed) {}
  static c10::DeviceType device_type() { return c10::DeviceType::PrivateUse1; }
  void set_current_seed(uint64_t seed) override { seeds_.set_current_seed(seed); }
  uint64_t current_seed() const override { return seeds_.current_seed(); }
  uint64_t seed() override { return seeds_.seed(); }
  void set_offset(uint64_t offset) override { seeds_.set_offset(offset); }
  uint64_t get_offset() const override { return seeds_.get_offset(); }
  void set_state(const c10::TensorImpl& state) override { seeds_.set_state(state); }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { return seeds_.get_state(); }
  uint32_t random() { return seeds_.random(); }
 private:
  Generator* clone_impl() const override {
    auto result = std::make_unique<Generator>(current_seed());
    result->seeds_.set_state(*seeds_.get_state());
    return result.release();
  }
};

at::Generator new_generator(c10::DeviceIndex index) {
  validate(c10::Device(c10::DeviceType::PrivateUse1,index));
  return at::make_generator<Generator>();
}
REGISTER_GENERATOR_PRIVATEUSE1(new_generator);

const at::Generator& default_generator() {
  static const auto generator = at::make_generator<Generator>(c10::detail::getNonDeterministicRandom());
  return generator;
}

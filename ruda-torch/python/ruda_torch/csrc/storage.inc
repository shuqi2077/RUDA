void resize_storage(const c10::Storage& storage, size_t bytes) {
  validate(storage.device());
  TORCH_CHECK(storage.resizable() && storage.allocator()==&allocator,
              "Trying to resize storage that is not resizable by RUDA");
  TORCH_CHECK(storage.data_ptr().get_deleter()==release, "storage is not allocated by RUDA");
  if (storage.nbytes()==bytes) return;
  auto replacement = allocator.allocate(bytes);
  size_t count = std::min(storage.nbytes(), bytes), stride = 1;
  if (count) {
    Descriptor from{storage.data_ptr().get_context(),0,1,&count,&stride,8};
    Descriptor to{replacement.get_context(),0,1,&count,&stride,8};
    check(execute_native(0,&from,&from,&to,0));
  }
  storage.set_data_ptr(std::move(replacement));
  storage.set_nbytes(bytes);
}

const at::Tensor& resize_tensor(const at::Tensor& self, at::IntArrayRef size,
                               std::optional<c10::MemoryFormat> format) {
  validate(self.device());
  TORCH_CHECK(!format || *format!=c10::MemoryFormat::Preserve,
              "Unsupported memory format Preserve for resize_");
  for (auto value:size) TORCH_CHECK(value>=0, "Trying to create tensor with negative dimension ",value);
  auto impl = self.unsafeGetTensorImpl();
  if (self.sizes()!=size) {
    auto bytes = at::detail::computeStorageNbytesContiguous(size, self.element_size(), self.storage_offset());
    if (bytes>self.storage().nbytes()) resize_storage(self.storage(), bytes);
    impl->set_sizes_contiguous(size);
  }
  if (format) impl->empty_tensor_restride(*format);
  return self;
}

at::Tensor& set_storage_offset(at::Tensor& self, at::Storage source, int64_t offset,
                               at::IntArrayRef size, at::IntArrayRef stride) {
  validate(self.device()); validate(source.device());
  TORCH_CHECK(source.data_ptr().get_deleter()==release, "storage is not allocated by RUDA");
  TORCH_CHECK(offset>=0, "Tensor: invalid storage offset ",offset);
  TORCH_CHECK(stride.empty() || stride.size()==size.size(), "unequal size and stride length");
  for (auto value:size) TORCH_CHECK(value>=0, "negative tensor size");
  for (auto value:stride) TORCH_CHECK(value>=0, "negative tensor stride");
  auto bytes = stride.empty()
      ? at::detail::computeStorageNbytesContiguous(size,self.element_size(),offset)
      : at::detail::computeStorageNbytes(size,stride,self.element_size(),offset);
  at::native::checkSetStorage(self,source,offset,size,stride);
  if (bytes>source.nbytes()) resize_storage(source,bytes);
  auto impl = self.unsafeGetTensorImpl();
  impl->set_storage_offset(offset);
  if (stride.empty()) impl->set_sizes_contiguous(size);
  else impl->set_sizes_and_strides(size,stride);
  return self;
}

at::Tensor& set_storage(at::Tensor& self, at::Storage source) {
  int64_t size = source.nbytes()/self.element_size(), stride = 1;
  return set_storage_offset(self,std::move(source),0,{size},{stride});
}

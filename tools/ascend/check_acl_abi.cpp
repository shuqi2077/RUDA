// Compile against the *installed* SDK, never a test substitute header.
#include <acl/acl.h>
#include <cstdint>
#include <cstddef>
#include <type_traits>

using Load = aclError (*)(const void*,size_t,const aclrtBinaryLoadOptions*,aclrtBinHandle*);
using Lookup = aclError (*)(aclrtBinHandle,const char*,aclrtFuncHandle*);
using Launch = aclError (*)(void*,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void**);
using Unload = aclError (*)(aclrtBinHandle);
static_assert(std::is_same_v<decltype(&aclrtBinaryLoadFromData), Load>);
static_assert(std::is_same_v<decltype(&aclrtBinaryGetFunction), Lookup>);
static_assert(std::is_same_v<decltype(&aclrtLaunchKernelWithArgsArray), Launch>);
static_assert(std::is_same_v<decltype(&aclrtBinaryUnLoad), Unload>);
static_assert(sizeof(aclError)==4);
static_assert(sizeof(aclrtBinHandle)==8);


int main(){return 0;}

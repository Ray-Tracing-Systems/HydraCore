#include "HydraDLib.h"
#include <string>
#include <dlfcn.h>

std::string  ws2s(const std::wstring& s);

struct HydraDllHandle { void* handle = nullptr; };
HydraDllHandle g_handle;

HydraDllHandle* HydraLoadLibrary(const wchar_t* a_path)
{
  std::string filePath = ws2s(std::wstring(a_path));
  g_handle.handle = dlopen(filePath.c_str(), RTLD_LAZY);
  return &g_handle;
}

int HydraUnloadLibrary(HydraDllHandle* handle)
{
  if(handle != nullptr && handle->handle != nullptr)
    return dlclose(handle->handle);
  else
    return 0;
}

void*  HydraGetProcAddress(HydraDllHandle* handle, const char* a_name)
{
  if(handle == nullptr)
    return nullptr;
  if(handle->handle == nullptr)
    return nullptr;
  return dlsym(handle->handle, a_name);
}
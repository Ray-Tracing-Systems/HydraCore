#include "HydraDLib.h"
#include <string>
#include <windows.h>

std::string  ws2s(const std::wstring& s);

struct HydraDllHandle { HMODULE handle = NULL; };
HydraDllHandle g_handle;

HydraDllHandle* HydraLoadLibrary(const wchar_t* a_path)
{
  g_handle.handle = LoadLibraryW(a_path);
  return &g_handle;
}

int HydraUnloadLibrary(HydraDllHandle* handle)
{
  if (handle != nullptr)
  {
    FreeLibrary(handle->handle);
    handle->handle = NULL;
  }
  return 0;
}

void*  HydraGetProcAddress(HydraDllHandle* handle, const char* a_name)
{
  if(handle == nullptr)
    return nullptr;
  if(handle->handle == NULL)
    return nullptr;
  return GetProcAddress(handle->handle, a_name);
}
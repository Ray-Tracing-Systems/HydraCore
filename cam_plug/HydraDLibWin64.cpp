#include "HydraDLib.h"
#include <string>
#include <windows.h>
#include <iostream>
#include <strsafe.h>

std::string  ws2s(const std::wstring& s);

struct HydraDllHandle { HMODULE handle = NULL; };
HydraDllHandle g_handle;

HydraDllHandle* HydraLoadLibrary(const wchar_t* a_path)
{
  g_handle.handle = LoadLibraryW(a_path);
  if (g_handle.handle == nullptr)
  {
    DWORD dw = GetLastError();
    LPWSTR lpMsgBuf;
    LPVOID lpDisplayBuf;
  
    FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      dw,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR)&lpMsgBuf,
      0, NULL);

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, (lstrlen((LPCTSTR)lpMsgBuf) + 512) * sizeof(TCHAR));

    StringCchPrintf((LPTSTR)lpDisplayBuf, LocalSize(lpDisplayBuf) / sizeof(TCHAR), TEXT("Hydra failed loading DLL from %s with error %d: %s"), a_path, dw, lpMsgBuf);

    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
  }
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
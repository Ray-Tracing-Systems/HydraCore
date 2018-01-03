#include "IBVHBuilderAPI.h"

#include <windows.h>

typedef IBVHBuilder2* (*CreateFunc)(char* cfg);

IBVHBuilder2* CreateBuilderFromDLL(const wchar_t* a_path, char* a_cfg)
{
  HMODULE hDll = LoadLibraryW(a_path);
  if (hDll == NULL)
    return nullptr;

  CreateFunc func = (CreateFunc)GetProcAddress(hDll, "CreateBuilder2");
  if (func == nullptr)
    return nullptr;

  return func(a_cfg);
}

#pragma once

struct HydraDllHandle;

HydraDllHandle* HydraLoadLibrary(const wchar_t* a_path);
int             HydraUnloadLibrary(HydraDllHandle* handle);
void*           HydraGetProcAddress(HydraDllHandle* handle, const char* a_name);
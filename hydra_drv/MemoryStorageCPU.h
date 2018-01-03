#pragma once

#include "IMemoryStorage.h"


struct LinearStorageCPU : public IMemoryStorage
{
  LinearStorageCPU(){}
  ~LinearStorageCPU() {}

  void   Clear()                       override;
  size_t Reserve(uint64_t a_totalSize) override;
  size_t Resize(uint64_t a_totalSize)  override;

  const void*   GetBegin()    const;
  const size_t  GetSize()     const;
  const size_t  GetCapacity() const;

  void MemCopyAt(uint64_t a_offsetInInts, const void* a_data, uint64_t a_sizeInBytes) override;

  void DebugSaveToFile(const char* a_fileName);

protected:

  std::vector<uint8_t> data;

};

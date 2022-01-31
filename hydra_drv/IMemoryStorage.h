#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>

struct LChunk
{
  int begin;
  int endMax;
  int endCur;
  int offset; // not used right now, equal to 0
};

struct IMemoryStorage
{
  IMemoryStorage() : maxId(0) {}
  virtual ~IMemoryStorage() {}

  virtual void   Clear()                              = 0;
  virtual size_t Reserve(uint64_t a_totalSizeInBytes) = 0;
  virtual size_t Resize(uint64_t a_totalSize)         = 0;

  virtual const void*   GetBegin()    const = 0;
  virtual const size_t  GetSize()     const = 0;
  virtual const size_t  GetCapacity() const = 0;

  virtual int32_t Update(int32_t id, const void* a_data, uint64_t a_sizeInBytes);                                  ///< can do realloc
  virtual void    UpdatePartial(int32_t id, const void* a_data, uint64_t a_offsetInBytes, uint64_t a_sizeInBytes); ///< in place update only

  virtual std::vector<int32_t> GetTable();

  virtual int GetAlignSizeInBytes() const { return 16; }
  virtual int GetMaxObjectId()      const { return maxId; }
  virtual void DebugSaveToFile(const char* a_fileName) = 0;

  virtual void FreeHostMem() {}

protected:

  std::unordered_map<int, LChunk> objects;
  int maxId;

  virtual void   MemCopyAt(uint64_t a_offsetInInts, const void* a_data, uint64_t a_sizeInBytes) = 0;
  virtual LChunk AppendToTheEnd(const void* a_data, uint64_t a_sizeInBytes);
  virtual size_t SizeInBlocks(uint64_t a_sizeInBytes);

};


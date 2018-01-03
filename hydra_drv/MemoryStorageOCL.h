#pragma once
#include "IMemoryStorage.h"
#include "MemoryStorageCPU.h"

#include "../vsgl3/clHelper.h"

struct MemoryStorageOCL : public IMemoryStorage
{
  MemoryStorageOCL()                                              : m_dataBuffer(nullptr), m_currSize(0), m_totalSize(0), m_ctx(nullptr), m_queue(nullptr) {  }
  MemoryStorageOCL(cl_context a_ctx, cl_command_queue a_cmdQueue) : m_dataBuffer(nullptr), m_currSize(0), m_totalSize(0), m_ctx(a_ctx), m_queue(a_cmdQueue) {  }
  ~MemoryStorageOCL() { Clear(); clReleaseMemObject(m_dataBuffer); m_dataBuffer = nullptr; }

  void   Clear()                       override;
  size_t Reserve(uint64_t a_totalSize) override;
  size_t Resize(uint64_t a_totalSize)  override;

  const void*   GetBegin()    const;
  const size_t  GetSize()     const;
  const size_t  GetCapacity() const;

  void MemCopyAt(uint64_t a_offsetInInts, const void* a_data, uint64_t a_sizeInBytes) override;

  void DebugSaveToFile(const char* a_fileName);

  cl_mem GetOCLBuffer() { return m_dataBuffer; }

protected:

  std::vector<int> data;

  cl_mem   m_dataBuffer;
  uint64_t m_currSize;
  uint64_t m_totalSize;

  cl_context       m_ctx;
  cl_command_queue m_queue;

};


struct MemoryStorageBothCPUAndGPU : public IMemoryStorage
{

  MemoryStorageBothCPUAndGPU() : m_pStorageCPU(nullptr), m_pStorageGPU(nullptr) {}
  MemoryStorageBothCPUAndGPU(LinearStorageCPU* a_pStorageCPU, MemoryStorageOCL* a_pStorageGPU) : m_pStorageCPU(a_pStorageCPU), m_pStorageGPU(a_pStorageGPU) {}
  ~MemoryStorageBothCPUAndGPU() 
  {
    delete m_pStorageCPU; m_pStorageCPU = nullptr;
    delete m_pStorageGPU; m_pStorageGPU = nullptr;
  }

  void   Clear()                       override;
  size_t Reserve(uint64_t a_totalSize) override;
  size_t Resize(uint64_t a_totalSize)  override;

  const void*   GetBegin()    const;
  const size_t  GetSize()     const;
  const size_t  GetCapacity() const;

  void MemCopyAt(uint64_t a_offsetInInts, const void* a_data, uint64_t a_sizeInBytes) override;

  void DebugSaveToFile(const char* a_fileName);

  cl_mem GetOCLBuffer() { return m_pStorageGPU->GetOCLBuffer(); }

  void FreeHostMem() override { delete m_pStorageCPU; m_pStorageCPU = nullptr; }

protected:

  LinearStorageCPU* m_pStorageCPU;
  MemoryStorageOCL* m_pStorageGPU;

};

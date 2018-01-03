#include "MemoryStorageOCL.h"
#include <fstream>

void MemoryStorageOCL::Clear()
{
  m_currSize  = 0;
  m_totalSize = 0;
}

size_t MemoryStorageOCL::Reserve(uint64_t a_totalSize)
{
  if (m_totalSize == a_totalSize)
    return m_totalSize;

  cl_int ciErr1 = CL_SUCCESS;
  if(m_dataBuffer != 0)
    clReleaseMemObject(m_dataBuffer);

  m_dataBuffer = clCreateBuffer(m_ctx, CL_MEM_READ_ONLY, a_totalSize, nullptr, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    return size_t(-1);

  m_totalSize = a_totalSize;
  m_currSize  = 0;

  return m_totalSize;
}

size_t MemoryStorageOCL::Resize(uint64_t a_size)
{
  if (a_size <= m_totalSize)
  {
    m_currSize = a_size;
    return m_currSize;
  }
  else
    return size_t (-1);
}

const void* MemoryStorageOCL::GetBegin() const 
{
  return nullptr;
}

const size_t  MemoryStorageOCL::GetSize() const 
{
  return size_t(m_currSize);
}

const size_t  MemoryStorageOCL::GetCapacity() const
{
  return size_t(m_totalSize);
}

void MemoryStorageOCL::MemCopyAt(uint64_t a_offsetInBytes, const void* a_data, uint64_t a_sizeInBytes)
{
  CHECK_CL(clEnqueueWriteBuffer(m_queue, m_dataBuffer, CL_TRUE, a_offsetInBytes, a_sizeInBytes, a_data, 0, NULL, NULL));
}

void MemoryStorageOCL::DebugSaveToFile(const char* a_fileName)
{
  std::vector<uint8_t> data(m_totalSize);
 
  CHECK_CL(clEnqueueReadBuffer(m_queue, m_dataBuffer, CL_TRUE, 0, m_totalSize, &data[0], 0, NULL, NULL));

  std::ofstream fout(a_fileName);
  for (size_t i = 0; i < data.size(); i++)
    fout << data[i] << std::endl;
  fout.flush();
  fout.close();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MemoryStorageBothCPUAndGPU::Clear()  
{
  if(m_pStorageCPU != nullptr) m_pStorageCPU->Clear();
  if(m_pStorageGPU != nullptr) m_pStorageGPU->Clear();
}

size_t MemoryStorageBothCPUAndGPU::Reserve(uint64_t a_totalSize) 
{
  if (m_pStorageCPU != nullptr) 
    m_pStorageCPU->Reserve(a_totalSize);

  if (m_pStorageGPU != nullptr)
    return m_pStorageGPU->Reserve(a_totalSize);
  else
    return 0;
}

size_t MemoryStorageBothCPUAndGPU::Resize(uint64_t a_totalSize)
{
  if (m_pStorageCPU != nullptr)
    m_pStorageCPU->Resize(a_totalSize);

  if (m_pStorageGPU != nullptr)
    return m_pStorageGPU->Resize(a_totalSize);
  else
    return 0;
}

const void*   MemoryStorageBothCPUAndGPU::GetBegin()    const
{
  if (m_pStorageCPU != nullptr)
    return m_pStorageCPU->GetBegin();
  else
    return nullptr;
}

const size_t  MemoryStorageBothCPUAndGPU::GetSize()     const
{
  if (m_pStorageGPU != nullptr)
    return m_pStorageGPU->GetSize();
  else
    return 0;
}

const size_t  MemoryStorageBothCPUAndGPU::GetCapacity() const
{
  if (m_pStorageGPU != nullptr)
    return m_pStorageGPU->GetCapacity();
  else
    return 0;
}

void MemoryStorageBothCPUAndGPU::MemCopyAt(uint64_t a_offsetInBytes, const void* a_data, uint64_t a_sizeInBytes)
{
  if (m_pStorageCPU != nullptr) m_pStorageCPU->MemCopyAt(a_offsetInBytes, a_data, a_sizeInBytes);
  if (m_pStorageGPU != nullptr) m_pStorageGPU->MemCopyAt(a_offsetInBytes, a_data, a_sizeInBytes);
}

void MemoryStorageBothCPUAndGPU::DebugSaveToFile(const char* a_fileName)
{
  if (m_pStorageGPU != nullptr) 
    m_pStorageGPU->DebugSaveToFile(a_fileName);
}
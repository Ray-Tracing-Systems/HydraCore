#include "MemoryStorageCPU.h"
#include <fstream>
#include <assert.h>
#include <cstring>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t IMemoryStorage::SizeInBlocks(uint64_t a_sizeInBytes)
{
  uint64_t blockSize = GetAlignSizeInBytes();

  if ((a_sizeInBytes % blockSize) == 0)
    return a_sizeInBytes / blockSize;
  else
    return (a_sizeInBytes / blockSize) + 1;
}

LChunk IMemoryStorage::AppendToTheEnd(const void* a_data, uint64_t a_sizeInBytes)
{
  const int bytesPerBlock   = GetAlignSizeInBytes();
  const size_t sizeInBlocks = SizeInBlocks(a_sizeInBytes);
  const size_t begin        = this->GetSize();

  size_t newSize = this->Resize(begin + sizeInBlocks*bytesPerBlock);
  if (newSize == size_t(-1))
  {
    LChunk chunk;
    chunk.begin  = -1;
    chunk.endCur = -1;
    chunk.endMax = -1;
    chunk.offset = -1;
    return chunk;
  }

  const size_t end = this->GetSize();

  assert(begin % bytesPerBlock == 0);
  assert(end   % bytesPerBlock == 0);

  if (a_data != nullptr)
    MemCopyAt(begin, a_data, a_sizeInBytes);


  LChunk chunk;
  chunk.begin  = int(begin / bytesPerBlock);
  chunk.endCur = int(end   / bytesPerBlock);
  chunk.endMax = int(end   / bytesPerBlock);
  chunk.offset = 0;
  return chunk;
}

int32_t IMemoryStorage::Update(int32_t id, const void* a_data, uint64_t a_sizeInBytes)
{
  if (id > maxId)
    maxId = id;

  const int bytesPerBlock   = GetAlignSizeInBytes();
  const size_t sizeInBlocks = SizeInBlocks(a_sizeInBytes);

  auto p = objects.find(id);

  if (p != objects.end())
  {
    LChunk chunk = p->second;

    if (chunk.begin + sizeInBlocks <= chunk.endMax) // Update in the place it's already located
    {
      if (a_data != nullptr)
        MemCopyAt(size_t(chunk.begin) * size_t(bytesPerBlock), a_data, a_sizeInBytes);
      p->second.endCur = chunk.begin + int(sizeInBlocks);
      return chunk.begin;
    }
    else
    {
      auto chunk = AppendToTheEnd(a_data, a_sizeInBytes); // inser to the end of the buffer
      objects[id] = chunk;
      return chunk.begin;
    }
  }
  else
  {
    auto chunk = AppendToTheEnd(a_data, a_sizeInBytes);
    objects[id] = chunk;
    return chunk.begin;
  }
}

void IMemoryStorage::UpdatePartial(int32_t id, const void* a_data, uint64_t a_offsetInBytes, uint64_t a_sizeInBytes)
{
  auto p = objects.find(id);
  if (p == objects.end())
    return;

  const int bytesPerBlock = GetAlignSizeInBytes();
  LChunk chunk = p->second;

  if (a_offsetInBytes + a_sizeInBytes > size_t(chunk.endMax) * bytesPerBlock)
    return;

  size_t offset = size_t(chunk.begin) * size_t(bytesPerBlock) + a_offsetInBytes;

  assert(a_offsetInBytes % bytesPerBlock == 0);
  assert(offset          % bytesPerBlock == 0);
  
  if (chunk.begin == -1)
    return;
  // assert(chunk.begin != -1);

  MemCopyAt(offset, a_data, a_sizeInBytes);
}

std::vector<int32_t> IMemoryStorage::GetTable()
{
  const int bytesPerBlock = GetAlignSizeInBytes();
  const int mult =  bytesPerBlock / (sizeof(int) * 4);

  std::vector<int32_t> res(maxId + 1);

  for (auto& initialOffset : res)
    initialOffset = -1;

  for (auto p = objects.begin(); p != objects.end(); ++p)
    res[p->first] = p->second.begin * mult;

  return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LinearStorageCPU::Clear()
{
  data    = std::vector<uint8_t>();
  objects = std::unordered_map<int, LChunk>();
  maxId   = 0;
}

size_t LinearStorageCPU::Reserve(uint64_t a_totalSizeInBytes)
{
  data.reserve(a_totalSizeInBytes);
  return data.capacity();
}

size_t LinearStorageCPU::Resize(uint64_t a_totalSizeInBytes)
{
  data.resize(a_totalSizeInBytes);
  return data.size();
}

void LinearStorageCPU::MemCopyAt(uint64_t a_offsetInBytes, const void* a_data, uint64_t a_sizeInBytes)
{
  if (a_data != nullptr)
    memcpy(&data[a_offsetInBytes], a_data, a_sizeInBytes);
}

const void* LinearStorageCPU::GetBegin() const
{
  if(data.size() == 0)
    return nullptr;
  else
    return &data[0];
}

const size_t LinearStorageCPU::GetSize() const
{
  return data.size();
}

const size_t LinearStorageCPU::GetCapacity() const
{
  return data.capacity();
}

void LinearStorageCPU::DebugSaveToFile(const char* a_fileName)
{
  std::ofstream fout(a_fileName);
  for (size_t i = 0; i < data.size(); i++)
    fout << data[i] << std::endl;
  fout.flush();
  fout.close();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GPUOCLLayer.h"
#include "crandom.h"

#include "MemoryStorageCPU.h"
#include "MemoryStorageOCL.h"

void GPUOCLLayer::CreateBuffersGeom(InputGeom a_input, cl_mem_flags a_flags) { }
void GPUOCLLayer::CreateBuffersBVH(InputGeomBVH a_input, cl_mem_flags a_flags) { }

//m_memoryTaken[MEM_TAKEN_GEOMETRY] = compressedNorm.size()*sizeof(float)*5 + a_input.vertNum*sizeof(float2) + a_input.numIndices*sizeof(int);

////

void GPUOCLLayer::Clear(CLEAR_FLAGS a_flags)
{

}

void GPUOCLLayer::SetNamedBuffer(const char* a_name, void* a_data, size_t a_size)
{
  cl_int ciErr1 = CL_SUCCESS;

  if (std::string(a_name) == "ao" && a_size == size_t(-1))
  {
    m_rays.aoCompressed  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, m_rays.MEGABLOCKSIZE, nullptr, &ciErr1); // byte buffer
    m_rays.aoCompressed2 = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, m_rays.MEGABLOCKSIZE, nullptr, &ciErr1); // byte buffer
    return;
  }

  auto p = m_scene.namedBuffers.find(a_name);
  if (p != m_scene.namedBuffers.end())
  {
    clReleaseMemObject(p->second);
    p->second = 0;
  }

  m_scene.namedBuffers[a_name] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_size, a_data, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR((std::string("Error in clCreateBuffer, SetNamedBuffer, name = ") + a_name).c_str());
}


IMemoryStorage* GPUOCLLayer::CreateMemStorage(uint64_t a_maxSizeInBytes, const char* a_name)
{
  IMemoryStorage* pStorage = nullptr;

  cl_mem buff = 0;

  if (std::string(a_name) == "geom" || std::string(a_name) == "textures")
  {
    LinearStorageCPU* pCPUImpl = new LinearStorageCPU();
    MemoryStorageOCL* pGPUImpl = new MemoryStorageOCL(m_globals.ctx, m_globals.cmdQueue);
    pStorage = new MemoryStorageBothCPUAndGPU(pCPUImpl, pGPUImpl);
    pStorage->Reserve(a_maxSizeInBytes);
    buff = pGPUImpl->GetOCLBuffer();
  }
  else
  {
    MemoryStorageOCL* pGPUImpl = new MemoryStorageOCL(m_globals.ctx, m_globals.cmdQueue);
    pGPUImpl->Reserve(a_maxSizeInBytes);
    buff = pGPUImpl->GetOCLBuffer();
    pStorage = pGPUImpl;
  }

  m_allMemStorages[a_name] = pStorage;

  if (std::string(a_name) == "textures")
    m_scene.storageTex = buff;
  else if (std::string(a_name) == "geom")
    m_scene.storageGeom = buff;
  else if (std::string(a_name) == "materials")
    m_scene.storageMat  = buff;
  else if(std::string(a_name) == "pdfs")
    m_scene.storagePdfs = buff;
  else if(std::string(a_name) == "textures_aux")
    m_scene.storageTexAux = buff;

  return pStorage;
}


void GPUOCLLayer::SetAllBVH4(const ConvertionResult& a_convertedBVH, IBVHBuilder2* a_inBuilderAPI, int a_flags)
{
  for (int i = 0; i < m_scene.bvhNumber; i++)
  {
    if (m_scene.bvhBuff[i]     != nullptr) { clReleaseMemObject(m_scene.bvhBuff[i]);     m_scene.bvhBuff    [i] = nullptr; }
    if (m_scene.objListBuff[i] != nullptr) { clReleaseMemObject(m_scene.objListBuff[i]); m_scene.objListBuff[i] = nullptr; }
    if (m_scene.alphTstBuff[i] != nullptr) { clReleaseMemObject(m_scene.alphTstBuff[i]); m_scene.alphTstBuff[i] = nullptr; }
  }

  cl_int ciErr1 = CL_SUCCESS;

  m_memoryTaken[MEM_TAKEN_BVH] = 0;

  for (int i = 0; i < a_convertedBVH.treesNum; i++)
  {
    const size_t nodesSize = a_convertedBVH.nodesNum[i]*sizeof(BVHNode);
    const size_t primsSize = a_convertedBVH.trif4Num[i]*sizeof(float4);
    const size_t alphaSize = a_convertedBVH.triAfNum[i]*sizeof(uint2);

    m_memoryTaken[MEM_TAKEN_BVH] += (nodesSize + primsSize);
    if (a_convertedBVH.pTriangleAlpha[i])
      m_memoryTaken[MEM_TAKEN_BVH] += alphaSize;

    m_scene.bvhBuff    [i] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nodesSize, (void*)a_convertedBVH.pBVH[i],          &ciErr1);
    m_scene.objListBuff[i] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, primsSize, (void*)a_convertedBVH.pTriangleData[i], &ciErr1);

    if(a_convertedBVH.pTriangleAlpha[i] != nullptr)
      m_scene.alphTstBuff[i] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, alphaSize, (void*)a_convertedBVH.pTriangleAlpha[i], &ciErr1);

    m_scene.bvhHaveInst[i]      = (std::string(a_convertedBVH.bvhType[i]) != "triangle4v"); // or (bvhType == "object")
    m_bvhTrees[i].smoothOpacity = (a_flags & BVH_ENABLE_SMOOTH_OPACITY) != 0;
  }

  m_scene.bvhNumber = a_convertedBVH.treesNum;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::SetAllBVH4 Error in clCreateBuffer");

}

void GPUOCLLayer::SetAllInstMatrices(const float4x4* a_matrices, int32_t a_matrixNum)
{
  if (a_matrices == nullptr || a_matrixNum == 0)
  {
    if (m_scene.matrices != nullptr)
      clReleaseMemObject(m_scene.matrices);
    m_scene.matrices     = nullptr;
    m_scene.matricesSize = 0;
    return;
  }

  const size_t newSize = a_matrixNum * sizeof(float4x4);

  cl_int ciErr1 = CL_SUCCESS;

  if (m_scene.matrices == nullptr || m_scene.matricesSize < newSize)
  {
    if (m_scene.matrices != nullptr)
      clReleaseMemObject(m_scene.matrices);

    m_scene.matrices     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, newSize, (void*)a_matrices, &ciErr1);
    m_scene.matricesSize = newSize;
  }
  else
  {
    CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.matrices, CL_TRUE, 0, newSize, (void*)a_matrices, 0, NULL, NULL));
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::SetAllInstMatrices: Error in clCreateBuffer / clEnqueueWriteBuffer");

  m_scene.totalInstanceNum = a_matrixNum;
}

void GPUOCLLayer::SetAllInstLightInstId(const int32_t* a_lightInstIds, int32_t a_instNum)
{
  if (a_instNum == 0 || a_lightInstIds == nullptr)
  {
    std::cerr << "WARNING: GPUOCLLayer::SetAllInstLightInstId, no lights!" << std::endl;
    m_scene.instLightInst     = nullptr;
    m_scene.instLightInstSize = 0;
    return;
  }

  const size_t newSize = a_instNum * sizeof(int32_t);

  cl_int ciErr1 = CL_SUCCESS;

  if (m_scene.instLightInst == nullptr || m_scene.instLightInstSize < newSize)
  {
    if (m_scene.instLightInst != nullptr)
      clReleaseMemObject(m_scene.instLightInst);

    m_scene.instLightInst     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, newSize, (void*)a_lightInstIds, &ciErr1);
    m_scene.instLightInstSize = newSize;
  }
  else
  {
    CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.instLightInst, CL_TRUE, 0, newSize, (void*)a_lightInstIds, 0, NULL, NULL));
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::SetAllInstLightInstId: Error in clCreateBuffer / clEnqueueWriteBuffer");

}

void GPUOCLLayer::SetAllPODLights(PlainLight* a_lights2, size_t a_number)
{
  Base::SetAllPODLights(a_lights2, a_number);
}

void GPUOCLLayer::SetAllRemapLists(const int* a_allLists, const int2* a_table, int a_allSize, int a_tableSize)
{

  if (a_allSize == 0 || a_tableSize == 0 || a_allLists == nullptr || a_table == nullptr)
  {
    m_scene.remapLists     = nullptr;
    m_scene.remapTable     = nullptr;
    m_scene.remapListsSize = 0;
    m_scene.remapTableSize = 0;
    return;
  }

  cl_int ciErr1 = CL_SUCCESS;

  // (1) update all remap lists
  //
  if (m_scene.remapLists == nullptr || m_scene.remapListsSize < a_allSize)
  {
    if (m_scene.remapLists != nullptr)
      clReleaseMemObject(m_scene.remapLists);

    m_scene.remapLists     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_allSize*sizeof(int), (void*)a_allLists, &ciErr1);
    m_scene.remapListsSize = a_allSize;
  }
  else
  {
    CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.remapLists, CL_TRUE, 0, a_allSize * sizeof(int), (void*)a_allLists, 0, NULL, NULL));
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::SetAllInstLightInstId: Error in clCreateBuffer / clEnqueueWriteBuffer");

  // (2) update remap table
  //
  if (m_scene.remapTable == nullptr || m_scene.remapTableSize < a_tableSize)
  {
    if (m_scene.remapTable != nullptr)
      clReleaseMemObject(m_scene.remapTable);

    m_scene.remapTable     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_tableSize * sizeof(int2), (void*)a_table, &ciErr1);
    m_scene.remapTableSize = a_tableSize;
  }
  else
  {
    CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.remapTable, CL_TRUE, 0, a_tableSize * sizeof(int2), (void*)a_table, 0, NULL, NULL));
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::SetAllInstLightInstId: Error in clCreateBuffer / clEnqueueWriteBuffer");

}

void GPUOCLLayer::SetAllInstIdToRemapId(const int* a_allInstId, int a_instNum)
{

  if (a_instNum == 0 || a_allInstId == nullptr)
  {
    m_scene.remapInst     = nullptr;
    m_scene.remapInstSize = 0;
    return;
  }

  cl_int ciErr1 = CL_SUCCESS;

  if (m_scene.remapInst == nullptr || m_scene.remapInstSize < a_instNum)
  {
    if (m_scene.remapInst != nullptr)
      clReleaseMemObject(m_scene.remapInst);

    m_scene.remapInst     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_instNum * sizeof(int), (void*)a_allInstId, &ciErr1);
    m_scene.remapInstSize = a_instNum;
  }
  else
  {
    CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.remapInst, CL_TRUE, 0, a_instNum * sizeof(int), (void*)a_allInstId, 0, NULL, NULL));
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::SetAllInstLightInstId: Error in clCreateBuffer / clEnqueueWriteBuffer");
}

void GPUOCLLayer::UpdateVarsOnGPU()
{
  memcpy(m_globsBuffHeader.varsI, m_vars.m_varsI, sizeof(int)*GMAXVARS);
  memcpy(m_globsBuffHeader.varsF, m_vars.m_varsF, sizeof(float)*GMAXVARS);
  m_globsBuffHeader.g_flags = m_vars.m_flags;
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.allGlobsData, CL_FALSE, 0, sizeof(EngineGlobals), &m_globsBuffHeader, 0, NULL, NULL));  // put m_globsBuffHeader to the beggining of the buffer
}

void GPUOCLLayer::PrepareEngineGlobals()
{
  Base::PrepareEngineGlobals();
  size_t totalBuffSize = m_cdataPrepared.size()*sizeof(int);

  cl_int ciErr1 = 0;

  if (m_scene.allGlobsData == 0)
  {
    m_scene.allGlobsData     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, totalBuffSize, &m_cdataPrepared[0], &ciErr1); //
    m_scene.allGlobsDataSize = totalBuffSize;
  }
  else
  {
    size_t newSize = sizeof(EngineGlobals);
    CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.allGlobsData, CL_FALSE, 0, newSize, &m_cdataPrepared[0], 0, NULL, NULL));
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::PrepareEngineGlobals: Error in clCreateBuffer / clEnqueueWriteBuffer");

}

void GPUOCLLayer::PrepareEngineTables()
{
  Base::PrepareEngineTables();
  size_t totalBuffSize = m_cdataPrepared.size()*sizeof(int);

  if (m_scene.allGlobsDataSize < totalBuffSize)
  {
    cl_int ciErr1 = 0;
    clReleaseMemObject(m_scene.allGlobsData);
    m_scene.allGlobsData = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, totalBuffSize, &m_cdataPrepared[0], &ciErr1); //
    if (ciErr1 != CL_SUCCESS)
      RUN_TIME_ERROR("GPUOCLLayer::PrepareEngineTables: Error in clCreateBuffer");
  }
  else
    CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.allGlobsData, CL_FALSE, 0, totalBuffSize, &m_cdataPrepared[0], 0, NULL, NULL));
}

void GPUOCLLayer::UpdateConstants()
{
  if (m_cdataPrepared.size() == 0)
    return;

  const size_t constantsSize = sizeof(EngineGlobals);
  int* pbuff = &m_cdataPrepared[0];
  memcpy(pbuff, &m_globsBuffHeader, constantsSize);
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_scene.allGlobsData, CL_TRUE, 0, constantsSize, &m_cdataPrepared[0], 0, NULL, NULL));
}

void GPUOCLLayer::SetAllFlagsAndVars(const AllRenderVarialbes& a_vars)
{
  this->Base::SetAllFlagsAndVars(a_vars);
  UpdateConstants();

  m_storeShadowInAlphaChannel = (a_vars.m_varsI[HRT_STORE_SHADOW_COLOR_W] == 1); 
}


void GPUOCLLayer::CallNamedFunc(const char* a_name, const char* a_args)
{
  
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUOCLLayer::runTraceCPU(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_hits, size_t a_size)
{
  CHECK_CL(clFinish(m_globals.cmdQueue));

  cl_int ciErr1 = 0;

  float4*   rpos = (float4*)  clEnqueueMapBuffer(m_globals.cmdQueue, a_rpos, CL_TRUE, CL_MAP_READ, 0, a_size*sizeof(float4), 0, 0, 0, &ciErr1);
  float4*   rdir = (float4*)  clEnqueueMapBuffer(m_globals.cmdQueue, a_rdir, CL_TRUE, CL_MAP_READ, 0, a_size*sizeof(float4), 0, 0, 0, &ciErr1);
  Lite_Hit* hits = (Lite_Hit*)clEnqueueMapBuffer(m_globals.cmdQueue, a_hits, CL_TRUE, CL_MAP_WRITE, 0, a_size*sizeof(Lite_Hit), 0, 0, 0, &ciErr1);
  uint*     flgs = (uint*)    clEnqueueMapBuffer(m_globals.cmdQueue, m_rays.rayFlags, CL_TRUE, CL_MAP_READ, 0, a_size*sizeof(uint), 0, 0, 0, &ciErr1);

  const float4* nodes = (const float4*)&m_scnBVH.nodes[0];
  const float4* plist = (const float4*)&m_scnBVH.primListData[0];

  IntegratorCommon* pCore = dynamic_cast<IntegratorCommon*>(m_pIntegrator);
  if (pCore == nullptr)
  {
    std::cerr << "GPUOCLLayer::runTraceCPU: bad dynamic cast for integrator" << std::endl;
    return;
  }

  #pragma omp parallel for
  for (int i = 0; i < a_size; i++)
  {
    uint flags = flgs[i];
    if (!rayIsActiveU(flags))
      continue;

    float3 rayPos = to_float3(rpos[i]);
    float3 rayDir = to_float3(rdir[i]);

    hits[i] = pCore->rayTrace(rayPos, rayDir, flags);
  }

  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, a_rpos, rpos, 0, 0, 0));
  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, a_rdir, rdir, 0, 0, 0));
  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, a_hits, hits, 0, 0, 0));
  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, m_rays.rayFlags, flgs, 0, 0, 0));

  //CHECK_CL(clFinish(m_globals.cmdQueue));

}


void GPUOCLLayer::runTraceShadowCPU(size_t a_size)
{
  cl_int ciErr1 = 0;

  float4*  a_data1       = (float4*) clEnqueueMapBuffer(m_globals.cmdQueue, m_rays.lsam1, CL_TRUE, CL_MAP_READ, 0, a_size*sizeof(float4), 0, 0, 0, &ciErr1);
  float4*  a_data2       = (float4*) clEnqueueMapBuffer(m_globals.cmdQueue, m_rays.lsam2, CL_TRUE, CL_MAP_READ, 0, a_size*sizeof(float4), 0, 0, 0, &ciErr1);
  ushort4* a_shadow      = (ushort4*)clEnqueueMapBuffer(m_globals.cmdQueue, m_rays.lshadow, CL_TRUE, CL_MAP_WRITE, 0, a_size*sizeof(ushort4), 0, 0, 0, &ciErr1);
  float4*  in_hitPosNorm = (float4*) clEnqueueMapBuffer(m_globals.cmdQueue, m_rays.hitPosNorm, CL_TRUE, CL_MAP_READ, 0, a_size*sizeof(float4), 0, 0, 0, &ciErr1);
  uint*    flgs          = (uint*)   clEnqueueMapBuffer(m_globals.cmdQueue, m_rays.rayFlags, CL_TRUE, CL_MAP_READ, 0, a_size*sizeof(uint), 0, 0, 0, &ciErr1);

  const float4* nodes = (const float4*)&m_scnBVH.nodes[0];
  const float4* plist = (const float4*)&m_scnBVH.primListData[0];
  
  IntegratorCommon* pCore = dynamic_cast<IntegratorCommon*>(m_pIntegrator);
  if (pCore == nullptr)
  {
    std::cerr << "GPUOCLLayer::runTraceShadowCPU: bad dynamic cast for integrator" << std::endl;
    return;
  }

  #pragma omp parallel for
  for (int tid = 0; tid < a_size; tid++)
  {
    uint flags = flgs[tid];
    if (!rayIsActiveU(flags))
      continue;

    float4 data1 = a_data1[tid];
    float4 data2 = a_data2[tid];

    ShadowSample explicitSam;

    explicitSam.pos     = to_float3(data1);
    explicitSam.color   = to_float3(data2);
    explicitSam.pdf     = data1.w > 0 ? data1.w : 1.0f;
    explicitSam.maxDist = data2.w;
    explicitSam.isPoint = (data1.w <= 0);

    float4 data    = in_hitPosNorm[tid];
    float3 hitPos  = to_float3(data);
    float3 hitNorm = normalize(decodeNormal(as_int(data.w)));

    float3 shadowRayDir = normalize(explicitSam.pos - hitPos); // explicitSam.direction;
    float3 shadowRayPos = hitPos + shadowRayDir*fmax(maxcomp(hitPos), 1.0f)*GEPSILON;

    float3 shadow       = pCore->shadowTrace(shadowRayPos, shadowRayDir, explicitSam.maxDist);

    ushort4 shadowCompressed;

    shadowCompressed.x = (ushort)(65535.0f * shadow.x);
    shadowCompressed.y = (ushort)(65535.0f * shadow.y);
    shadowCompressed.z = (ushort)(65535.0f * shadow.z);
    shadowCompressed.w = 0;

    a_shadow[tid] = shadowCompressed;
  }


  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, m_rays.lsam1, a_data1, 0, 0, 0));
  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, m_rays.lsam2, a_data2, 0, 0, 0));
  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, m_rays.lshadow, a_shadow, 0, 0, 0));
  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, m_rays.hitPosNorm, in_hitPosNorm, 0, 0, 0));
  CHECK_CL(clEnqueueUnmapMemObject(m_globals.cmdQueue, m_rays.rayFlags, flgs, 0, 0, 0));
}

void GPUOCLLayer::saveBlocksInfoToFile(cl_mem a_blocks, size_t a_size)
{
  static int fileId = 0;

  std::vector<ZBlock> blocks(a_size);
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, a_blocks, CL_TRUE, 0, blocks.size() * sizeof(ZBlock), &blocks[0], 0, NULL, NULL));



}


void GPUOCLLayer::debugSaveFrameBuffer(const char* a_fileName, cl_mem targetBuff)
{
  std::vector<float4> screenColor(m_width*m_height);
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, targetBuff, CL_TRUE, 0, screenColor.size()*sizeof(float4), &screenColor[0], 0, NULL, NULL));

  std::vector<float4> screenColor2(m_width*m_height);

  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
      screenColor2[y*m_width + x] = screenColor[IndexZBlock2D(x, y, m_width, MortonTable256)];
  }

  int wh[2] = { m_width, m_height };
  std::ofstream fout(a_fileName, std::ios::binary);
  fout.write((const char*)wh, sizeof(int) * 2);
  fout.write((const char*)&screenColor2[0], sizeof(float) * 4 * screenColor2.size());
  fout.close();
}

void GPUOCLLayer::debugSaveRays(const char* a_folderName, cl_mem rpos, cl_mem rdir)
{
  std::vector<float4> rposCPU(m_rays.MEGABLOCKSIZE);
  std::vector<float4> rdirCPU(m_rays.MEGABLOCKSIZE);

  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, rpos, CL_TRUE, 0, rposCPU.size()*sizeof(float4), &rposCPU[0], 0, NULL, NULL));
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, rdir, CL_TRUE, 0, rdirCPU.size()*sizeof(float4), &rdirCPU[0], 0, NULL, NULL));

  int isize = int(m_rays.MEGABLOCKSIZE);

  std::string fileName1 = std::string(a_folderName) + "rpos.array4f";
  std::string fileName2 = std::string(a_folderName) + "rdir.array4f";

  std::ofstream fout(fileName1.c_str(), std::ios::binary);
  fout.write((const char*)&isize, sizeof(int));
  fout.write((const char*)&rposCPU[0], sizeof(float) * 4 * rposCPU.size());
  fout.close();

  std::ofstream fout2(fileName2.c_str(), std::ios::binary);
  fout2.write((const char*)&isize, sizeof(int));
  fout2.write((const char*)&rdirCPU[0], sizeof(float) * 4 * rdirCPU.size());
  fout2.close();

}


void GPUOCLLayer::debugSaveRaysText(const char* a_folderName, cl_mem rpos, cl_mem rdir)
{
  std::vector<float4> rposCPU(m_rays.MEGABLOCKSIZE);
  std::vector<float4> rdirCPU(m_rays.MEGABLOCKSIZE);

  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, rpos, CL_TRUE, 0, rposCPU.size()*sizeof(float4), &rposCPU[0], 0, NULL, NULL));
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, rdir, CL_TRUE, 0, rdirCPU.size()*sizeof(float4), &rdirCPU[0], 0, NULL, NULL));

  int isize = int(m_rays.MEGABLOCKSIZE);

  std::string fileName1 = std::string(a_folderName) + "rpos.txt";
  std::string fileName2 = std::string(a_folderName) + "rdir.txt";

  std::ofstream fout(fileName1.c_str());
  for (int i = 0; i < rposCPU.size(); i++)
    fout << rposCPU[i].x << " " << rposCPU[i].y << " " << rposCPU[i].z << " " << rposCPU[i].w << std::endl;
  fout.close();

  std::ofstream fout2(fileName2.c_str());
  for (int i = 0; i < rdirCPU.size(); i++)
    fout << rdirCPU[i].x << " " << rdirCPU[i].y << " " << rdirCPU[i].z << " " << rdirCPU[i].w << std::endl;
  fout2.close();
}

void GPUOCLLayer::debugSaveFloat4Text(const char* a_fileName, cl_mem data)
{
  std::vector<float4> rposCPU(m_rays.MEGABLOCKSIZE);

  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, data, CL_TRUE, 0, rposCPU.size()*sizeof(float4), &rposCPU[0], 0, NULL, NULL));

  int isize = int(m_rays.MEGABLOCKSIZE);

  std::ofstream fout(a_fileName);
  for (int i = 0; i < rposCPU.size(); i++)
    fout << rposCPU[i].x << " " << rposCPU[i].y << " " << rposCPU[i].z << " " << rposCPU[i].w << std::endl;
  fout.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUOCLLayer::Denoise(cl_mem textureIn, cl_mem textureOut, int w, int h, float smoothLvl)
{
  const int w2 = blocks(w, 16) * 16;
  const int h2 = blocks(h, 16) * 16;

  size_t global_item_size[2] = { size_t(w2), size_t(h2) };
  size_t local_item_size[2]  = { 16, 16 };

  int radius     = int(smoothLvl) + 2;
  float noiseLvl = clamp(0.1f*smoothLvl, 0.0f, 1.0f);

  if (radius < 3) radius = 3;
  if (radius > 9) radius = 9;

  //cl_kernel myKernel = m_progs.imagep.kernel("BilateralFilter"); // NonLocalMeansFilter
  cl_kernel myKernel = m_progs.imagep.kernel("NonLocalMeansFilter"); // NonLocalMeansFilter

  CHECK_CL(clSetKernelArg(myKernel, 0, sizeof(cl_mem), (void*)&textureOut));
  CHECK_CL(clSetKernelArg(myKernel, 1, sizeof(cl_mem), (void*)&textureIn));
  CHECK_CL(clSetKernelArg(myKernel, 2, sizeof(cl_int), (void*)&w));
  CHECK_CL(clSetKernelArg(myKernel, 3, sizeof(cl_int), (void*)&h));
  CHECK_CL(clSetKernelArg(myKernel, 4, sizeof(cl_int), (void*)&radius));
  CHECK_CL(clSetKernelArg(myKernel, 5, sizeof(cl_float), (void*)&noiseLvl));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, myKernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

std::vector<uchar4> GPUOCLLayer::NormalMapFromDisplacement(int w, int h, const uchar4* a_data, float bumpAmt, bool invHeight, float smoothLvl)
{
  //std::cout << "[cl_core]: CPU NormalMapFromDisplacement" << std::endl;
  //return Base::NormalMapFromDisplacement(w, h, a_data, bumpAmt, invHeight, smoothLvl);
  // std::cout << "[cl_core]: GPU NormalMapFromDisplacement" << std::endl;

  cl_int ciErr1 = CL_SUCCESS;
  cl_int ciErr2 = CL_SUCCESS;

  cl_image_format imgFormat;
  imgFormat.image_channel_order     = CL_RGBA;
  imgFormat.image_channel_data_type = CL_UNORM_INT8;

  size_t maxSizeX = 0;
  size_t maxSizeY = 0;
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(size_t), &maxSizeX, NULL));
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &maxSizeY, NULL));

  if (maxSizeX < w || maxSizeY < h)
  {
    std::cout << "[cl_core]: NormalMapFromDisplacement(); maxSizeX = " << maxSizeX << ", w = " << w << std::endl;
    std::cout << "[cl_core]: NormalMapFromDisplacement(); maxSizeY = " << maxSizeY << ", h = " << h << std::endl;
    std::cout << "[cl_core]: NormalMapFromDisplacement(); CPU fallback; Performance ALERT !!!" << std::endl;
    return Base::NormalMapFromDisplacement(w, h, a_data, bumpAmt, invHeight, smoothLvl);
  }

  const size_t maxWorkGroupSize = m_globals.m_maxWorkGroupSize;

  cl_mem textureIn  = clCreateImage2D(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &imgFormat, size_t(w), size_t(h), size_t(w) * sizeof(uchar4), (void*)a_data, &ciErr1);
  cl_mem textureOut = clCreateImage2D(m_globals.ctx, CL_MEM_READ_WRITE, &imgFormat, size_t(w), size_t(h), 0, NULL, &ciErr2);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::NormalMapFromDisplacement, clCreateImage2D(1) failed");

  if (ciErr2 != CL_SUCCESS)
    RUN_TIME_ERROR("GPUOCLLayer::NormalMapFromDisplacement, clCreateImage2D(2) failed");

  if (smoothLvl >= 1.0f) // denoise input heightmap
  {
    Denoise(textureIn, textureOut, w, h, smoothLvl);
    std::swap(textureIn, textureOut);
  }

  // run kernel NormalmapFromHeight
  {
    const int w2 = blocks(w, 16) * 16;
    const int h2 = blocks(h, 16) * 16;

    size_t global_item_size[2] = { size_t(w2), size_t(h2) };
    size_t local_item_size[2]  = { 16, 16 };
    int invH                   = invHeight ? 1 : 0;

    cl_kernel myKernel = m_progs.imagep.kernel("NormalmapFromHeight");

    CHECK_CL(clSetKernelArg(myKernel, 0, sizeof(cl_mem),   (void*)&textureOut));
    CHECK_CL(clSetKernelArg(myKernel, 1, sizeof(cl_mem),   (void*)&textureIn));
    CHECK_CL(clSetKernelArg(myKernel, 2, sizeof(cl_int),   (void*)&w));
    CHECK_CL(clSetKernelArg(myKernel, 3, sizeof(cl_int),   (void*)&h));
    CHECK_CL(clSetKernelArg(myKernel, 4, sizeof(cl_int),   (void*)&invHeight));
    CHECK_CL(clSetKernelArg(myKernel, 5, sizeof(cl_float), (void*)&bumpAmt));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, myKernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }

  // if (smoothLvl >= 1) // denoise output normalmap
  // {
  //   std::swap(textureIn, textureOut);
  //   Denoise(textureIn, textureOut, w, h, smoothLvl);
  // }

  const size_t origin[3] = { 0, 0, 0 };
  const size_t region[3] = { size_t(w), size_t(h), 1 };

  std::vector<uchar4> resData(w*h);
  CHECK_CL(clEnqueueReadImage(m_globals.cmdQueue, textureOut, CL_TRUE, origin, region, 0, 0, (void*)&resData[0], 0, NULL, NULL));

  clReleaseMemObject(textureIn);  textureIn  = 0;
  clReleaseMemObject(textureOut); textureOut = 0;

  return resData;
}


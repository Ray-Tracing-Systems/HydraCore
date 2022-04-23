#include "GPUOCLLayer.h"
#include "cl_scan_gpu.h"

void GPUOCLLayer::waitIfDebug(const char* file, int line) const
{
#ifdef _DEBUG
  cl_int cErr = clFinish(m_globals.cmdQueue);
  if (cErr != CL_SUCCESS)
  {
    const char* err = getOpenCLErrorString(cErr);
    RUN_TIME_ERROR_AT(err, file, line);
  }
#endif
}

void GPUOCLLayer::runKernel_MakeEyeSamplesOnly(size_t a_size, int a_passNumber,
                                               cl_mem a_zindex, cl_mem a_samples)
{
  cl_mem pssVector       = a_samples;

  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  cl_kernel makeSamples  = m_progs.screen.kernel("MakeEyeRaysSamplesOnly");

  CHECK_CL(clSetKernelArg(makeSamples, 0, sizeof(cl_mem), (void*)&m_rays.randGenState));
  CHECK_CL(clSetKernelArg(makeSamples, 1, sizeof(cl_mem), (void*)&pssVector));   
  CHECK_CL(clSetKernelArg(makeSamples, 2, sizeof(cl_mem), (void*)&a_zindex));
  CHECK_CL(clSetKernelArg(makeSamples, 3, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(makeSamples, 4, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(makeSamples, 5, sizeof(cl_mem), (void*)&m_globals.qmcTable));
  CHECK_CL(clSetKernelArg(makeSamples, 6, sizeof(cl_int), (void*)&a_passNumber));
  CHECK_CL(clSetKernelArg(makeSamples, 7, sizeof(cl_int), (void*)&m_width));
  CHECK_CL(clSetKernelArg(makeSamples, 8, sizeof(cl_int), (void*)&m_height));
  CHECK_CL(clSetKernelArg(makeSamples, 9, sizeof(cl_int), (void*)&iSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeSamples, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

  m_globals.m_passNumberQMC = a_passNumber; 
}


void GPUOCLLayer::runKernel_MakeEyeRaysQMC(size_t a_size, int a_passNumber,
                                           cl_mem a_zindex, cl_mem a_samples)
{
  cl_mem pssVector       = a_samples;

  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  cl_kernel makeSamples  = m_progs.screen.kernel("MakeEyeRaysQMC");

  CHECK_CL(clSetKernelArg(makeSamples, 0, sizeof(cl_mem), (void*)&m_rays.randGenState));
  CHECK_CL(clSetKernelArg(makeSamples, 1, sizeof(cl_mem), (void*)&pssVector));   
  CHECK_CL(clSetKernelArg(makeSamples, 2, sizeof(cl_mem), (void*)&a_zindex));
  CHECK_CL(clSetKernelArg(makeSamples, 3, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(makeSamples, 4, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(makeSamples, 5, sizeof(cl_mem), (void*)&m_globals.qmcTable));
  CHECK_CL(clSetKernelArg(makeSamples, 6, sizeof(cl_int), (void*)&a_passNumber));
  CHECK_CL(clSetKernelArg(makeSamples, 7, sizeof(cl_int), (void*)&m_width));
  CHECK_CL(clSetKernelArg(makeSamples, 8, sizeof(cl_int), (void*)&m_height));
  CHECK_CL(clSetKernelArg(makeSamples, 9, sizeof(cl_int), (void*)&iSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeSamples, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

  m_globals.m_passNumberQMC = a_passNumber; 
}

void GPUOCLLayer::runKernel_MakeRaysFromEyeSam(cl_mem a_zindex, cl_mem a_samples, size_t a_size, int a_passNumber,
                                               cl_mem a_rpos, cl_mem a_rdir, cl_mem a_color)
{
  cl_mem pssVector       = a_samples;

  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  {
    BitonicCLArgs sortArgs;
    sortArgs.bitonicPassK = m_progs.sort.kernel("bitonic_pass_kernel");
    sortArgs.bitonic512   = m_progs.sort.kernel("bitonic_512");
    sortArgs.bitonic1024  = m_progs.sort.kernel("bitonic_1024");
    sortArgs.bitonic2048  = m_progs.sort.kernel("bitonic_2048");
    sortArgs.cmdQueue     = m_globals.cmdQueue;
    sortArgs.dev          = m_globals.device;
    
    bitonic_sort_gpu(a_zindex, int(a_size), sortArgs);
  }

  waitIfDebug(__FILE__, __LINE__);

  cl_kernel makeRaysKern = m_progs.screen.kernel("MakeEyeRaysUnifiedSampling");

  int packIndexForCPU    = m_screen.m_cpuFrameBuffer ? 1 : 0;

  CHECK_CL(clSetKernelArg(makeRaysKern, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(makeRaysKern, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(makeRaysKern, 2, sizeof(cl_mem), (void*)&a_color));
  CHECK_CL(clSetKernelArg(makeRaysKern, 3, sizeof(cl_mem), (void*)&m_rays.packedXY));

  CHECK_CL(clSetKernelArg(makeRaysKern, 4, sizeof(cl_int), (void*)&m_width));
  CHECK_CL(clSetKernelArg(makeRaysKern, 5, sizeof(cl_int), (void*)&m_height));
  CHECK_CL(clSetKernelArg(makeRaysKern, 6, sizeof(cl_int), (void*)&iSize));

  CHECK_CL(clSetKernelArg(makeRaysKern, 7, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(makeRaysKern, 8, sizeof(cl_mem), (void*)&a_zindex));
  CHECK_CL(clSetKernelArg(makeRaysKern, 9, sizeof(cl_mem), (void*)&pssVector));  
  CHECK_CL(clSetKernelArg(makeRaysKern,10, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(makeRaysKern,11, sizeof(cl_mem), (void*)&m_globals.qmcTable));
  CHECK_CL(clSetKernelArg(makeRaysKern,12, sizeof(cl_int), (void*)&a_passNumber));
  CHECK_CL(clSetKernelArg(makeRaysKern,13, sizeof(cl_int), (void*)&packIndexForCPU));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeRaysKern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

  m_globals.m_passNumberQMC = a_passNumber; 
}

void GPUOCLLayer::runKernel_MakeEyeRays(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_zindex, size_t a_size, int a_passNumber)
{ 
  runKernel_MakeEyeSamplesOnly(a_size, a_passNumber,
                               a_zindex, m_rays.pathShadeColor);

  runKernel_MakeRaysFromEyeSam(a_zindex, m_rays.pathShadeColor, a_size, a_passNumber,
                               a_rpos, a_rdir, m_rays.pathAccColor);
}

void GPUOCLLayer::runKernel_TakeHostRays(cl_mem in_rays, cl_mem out_rpos, cl_mem out_rdir, cl_mem out_clearColor, size_t a_size)
{
  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));
  cl_kernel makeRaysKern = m_progs.screen.kernel("TakeHostRays");

  CHECK_CL(clSetKernelArg(makeRaysKern, 0, sizeof(cl_mem), (void*)&out_rpos));
  CHECK_CL(clSetKernelArg(makeRaysKern, 1, sizeof(cl_mem), (void*)&out_rdir));   
  CHECK_CL(clSetKernelArg(makeRaysKern, 2, sizeof(cl_mem), (void*)&out_clearColor));
  CHECK_CL(clSetKernelArg(makeRaysKern, 3, sizeof(cl_mem), (void*)&m_rays.packedXY));
  CHECK_CL(clSetKernelArg(makeRaysKern, 4, sizeof(cl_mem), (void*)&m_rays.randGenState));

  CHECK_CL(clSetKernelArg(makeRaysKern, 5, sizeof(cl_int), (void*)&m_width));
  CHECK_CL(clSetKernelArg(makeRaysKern, 6, sizeof(cl_int), (void*)&m_height));
  CHECK_CL(clSetKernelArg(makeRaysKern, 7, sizeof(cl_int), (void*)&iSize));
  CHECK_CL(clSetKernelArg(makeRaysKern, 8, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(makeRaysKern, 9, sizeof(cl_mem), (void*)&in_rays));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeRaysKern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_ClearAllInternalTempBuffers(size_t a_size)
{
  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  cl_kernel makeRaysKern = m_progs.screen.kernel("ClearAllInternalTempBuffers");
  
  CHECK_CL(clSetKernelArg(makeRaysKern, 0, sizeof(cl_mem), (void*)&m_rays.rayFlags));        // pass this data to clear them only!
  CHECK_CL(clSetKernelArg(makeRaysKern, 1, sizeof(cl_mem), (void*)&m_rays.pathAccColor));    // pass this data to clear them only!
  CHECK_CL(clSetKernelArg(makeRaysKern, 2, sizeof(cl_mem), (void*)&m_rays.pathThoroughput)); // pass this data to clear them only!
  CHECK_CL(clSetKernelArg(makeRaysKern, 3, sizeof(cl_mem), (void*)&m_rays.fogAtten));        // pass this data to clear them only!
  CHECK_CL(clSetKernelArg(makeRaysKern, 4, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));   // pass this data to clear them only!
  CHECK_CL(clSetKernelArg(makeRaysKern, 5, sizeof(cl_mem), (void*)&m_rays.accPdf));
  CHECK_CL(clSetKernelArg(makeRaysKern, 6, sizeof(cl_int), (void*)&iSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeRaysKern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_AccumColor(cl_mem a_inColor, cl_mem in_surfId, cl_mem a_outColor, size_t a_size, float a_mult)
{
  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  cl_int insertSurfId   = m_vars.m_varsI[HRT_ENABLE_SURFACE_PACK];
  cl_kernel accumColorK = m_progs.screen.kernel("AccumColor3f");

  //std::cout << "insertSurfId = " << insertSurfId << std::endl;

  CHECK_CL(clSetKernelArg(accumColorK, 0, sizeof(cl_mem),   (void*)&a_inColor));
  CHECK_CL(clSetKernelArg(accumColorK, 1, sizeof(cl_mem),   (void*)&in_surfId));     
  CHECK_CL(clSetKernelArg(accumColorK, 2, sizeof(cl_mem),   (void*)&a_outColor));    
  CHECK_CL(clSetKernelArg(accumColorK, 3, sizeof(cl_int),   (void*)&iSize));
  CHECK_CL(clSetKernelArg(accumColorK, 4, sizeof(cl_float), (void*)&a_mult));
  CHECK_CL(clSetKernelArg(accumColorK, 5, sizeof(cl_int),   (void*)&insertSurfId));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, accumColorK, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MakeEyeRaysSpp(int32_t a_blocksSize, int32_t yBegin, size_t a_size, cl_mem in_pixels,
                                           cl_mem a_rpos, cl_mem a_rdir)
{
  int isize            = int(a_size);
  size_t localWorkSize = CMP_RESULTS_BLOCK_SIZE;
  a_size               = roundBlocks(a_size, int(localWorkSize));

  cl_kernel kernX = nullptr;
  if(a_blocksSize == GBUFFER_SAMPLES) 
  {
    kernX = m_progs.screen.kernel("MakeEyeRaysSPP");
  
    CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_rpos));
    CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rdir));

    CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&m_width));
    CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&m_height));
    CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_int), (void*)&a_blocksSize));
    CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_int), (void*)&yBegin));

    CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_globals.hammersley2DGBuff));
    CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  }
  else if(a_blocksSize == PMPIX_SAMPLES && in_pixels!= nullptr) // && input != nullptr
  {
    kernX = m_progs.screen.kernel("MakeEyeRaysSPPPixels");

    CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_rpos));
    CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rdir));
    CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&m_rays.packedXY));

    CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&m_width));
    CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_int), (void*)&m_height));
    CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&in_pixels));
    
    CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_rays.randGenState));
    CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_globals.qmcTable));
    CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_int), (void*)&m_globals.m_passNumberQMC));

    CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_globals.hammersley2D256));  
    CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
    CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_int), (void*)&isize));
  }
  else
  {
    std::cerr << "bad SPP size for runKernel_MakeEyeRaysSpp = " << a_blocksSize << std::endl;
    std::cerr << "                 in_pixels                = " << in_pixels << std::endl;
    return;
  }

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_CopyLightSampleToSurfaceHit(cl_mem a_rayPos, size_t a_size,
                                                        cl_mem out_hitSurfaceAll)
{
  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  cl_kernel makeSamKern  = m_progs.lightp.kernel("CopyLightSampleToSurfaceHit");
  
  CHECK_CL(clSetKernelArg(makeSamKern, 0, sizeof(cl_mem), (void*)&a_rayPos));
  CHECK_CL(clSetKernelArg(makeSamKern, 1, sizeof(cl_mem), (void*)&out_hitSurfaceAll));
  CHECK_CL(clSetKernelArg(makeSamKern, 2, sizeof(cl_int), (void*)&iSize));
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeSamKern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}                                                        

void GPUOCLLayer::runKernel_MakeLightRays(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size)
{
  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  // (1) generate samples in primary space
  //
  cl_kernel makeSamKern = m_progs.lightp.kernel("LightSampleForwardCreate");

  CHECK_CL(clSetKernelArg(makeSamKern, 0, sizeof(cl_mem), (void*)&m_rays.shadowRayPos));
  CHECK_CL(clSetKernelArg(makeSamKern, 1, sizeof(cl_mem), (void*)&m_rays.shadowRayDir));
  CHECK_CL(clSetKernelArg(makeSamKern, 2, sizeof(cl_mem), (void*)&m_rays.samZindex));
  CHECK_CL(clSetKernelArg(makeSamKern, 3, sizeof(cl_mem), (void*)&m_rays.randGenState));
  CHECK_CL(clSetKernelArg(makeSamKern, 4, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(makeSamKern, 5, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(makeSamKern, 6, sizeof(cl_int), (void*)&iSize));
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeSamKern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

  // (2) sort them
  //
  BitonicCLArgs sortArgs;
  sortArgs.bitonicPassK = m_progs.sort.kernel("bitonic_pass_kernel");
  sortArgs.bitonic512   = m_progs.sort.kernel("bitonic_512");
  sortArgs.bitonic1024  = m_progs.sort.kernel("bitonic_1024");
  sortArgs.bitonic2048  = m_progs.sort.kernel("bitonic_2048");
  sortArgs.cmdQueue     = m_globals.cmdQueue;
  sortArgs.dev          = m_globals.device;

  bitonic_sort_gpu(m_rays.samZindex, int(a_size), sortArgs);

  runKernel_ClearAllInternalTempBuffers(a_size); // #TODO: opt this!

  // (3) generate light sample
  //
  cl_kernel makeRaysKern = m_progs.lightp.kernel("LightSampleForwardKernel");

  CHECK_CL(clSetKernelArg(makeRaysKern, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(makeRaysKern, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(makeRaysKern, 2, sizeof(cl_mem), (void*)&m_rays.shadowRayPos));
  CHECK_CL(clSetKernelArg(makeRaysKern, 3, sizeof(cl_mem), (void*)&m_rays.shadowRayDir));
  CHECK_CL(clSetKernelArg(makeRaysKern, 4, sizeof(cl_mem), (void*)&m_rays.samZindex));

  CHECK_CL(clSetKernelArg(makeRaysKern, 5, sizeof(cl_mem), (void*)&m_rays.accPdf)); 
  CHECK_CL(clSetKernelArg(makeRaysKern, 6, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  CHECK_CL(clSetKernelArg(makeRaysKern, 7, sizeof(cl_mem), (void*)&m_rays.lightOffsetBuff));
  CHECK_CL(clSetKernelArg(makeRaysKern, 8, sizeof(cl_mem), (void*)&m_rays.lsamRev));

  CHECK_CL(clSetKernelArg(makeRaysKern, 9, sizeof(cl_mem), (void*)&m_rays.rayFlags));
  CHECK_CL(clSetKernelArg(makeRaysKern,10, sizeof(cl_mem), (void*)&a_outColor));
  CHECK_CL(clSetKernelArg(makeRaysKern,11, sizeof(cl_mem), (void*)&m_rays.pathThoroughput)); // pass this data to clear them only!
  CHECK_CL(clSetKernelArg(makeRaysKern,12, sizeof(cl_mem), (void*)&m_rays.fogAtten));        // pass this data to clear them only!

  CHECK_CL(clSetKernelArg(makeRaysKern,13, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(makeRaysKern,14, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(makeRaysKern,15, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(makeRaysKern,16, sizeof(cl_int), (void*)&iSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeRaysKern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void RoundBlocks2D(size_t global_item_size[2], size_t local_item_size[2])
{
  global_item_size[0] = roundBlocks(global_item_size[0], int(local_item_size[0]));
  global_item_size[1] = roundBlocks(global_item_size[1], int(local_item_size[1]));
}


void GPUOCLLayer::AddContributionToScreenGPU(cl_mem in_color,     cl_mem in_indices, int a_size, int a_width, int a_height, int a_spp, bool a_copyToLDRNow,
                                             cl_mem out_colorHDR, cl_mem out_colorLDR)
{
  //// (3) sort references
  ////
  //{
  //  BitonicCLArgs sortArgs;
  //  sortArgs.bitonicPassK = m_progs.sort.kernel("bitonic_pass_kernel");
  //  sortArgs.bitonic512   = m_progs.sort.kernel("bitonic_512");
  //  sortArgs.bitonic1024  = m_progs.sort.kernel("bitonic_1024");
  //  sortArgs.bitonic2048  = m_progs.sort.kernel("bitonic_2048");
  //  sortArgs.cmdQueue     = m_globals.cmdQueue;
  //  sortArgs.dev          = m_globals.device;
  //
  //  bitonic_sort_gpu(in_indices, int(m_rays.MEGABLOCKSIZE), sortArgs);
  //}

  int alreadySorted = 1;

  // (4) run contrib kernel
  //
  cl_kernel contribKern      = m_progs.screen.kernel("ContribSampleToScreen");
  size_t global_item_size[2] = { size_t(m_width), size_t(m_height) };
  size_t local_item_size[2]  = { 16, 16 };

  RoundBlocks2D(global_item_size, local_item_size);

  const float invGamma = 1.0f/m_globsBuffHeader.varsF[HRT_IMAGE_GAMMA];

  CHECK_CL(clSetKernelArg(contribKern, 0, sizeof(cl_mem), (void*)&in_color));
  CHECK_CL(clSetKernelArg(contribKern, 1, sizeof(cl_mem), (void*)&in_indices));
  CHECK_CL(clSetKernelArg(contribKern, 2, sizeof(cl_mem), (void*)&m_globals.cMortonTable));

  CHECK_CL(clSetKernelArg(contribKern, 3, sizeof(cl_int),   (void*)&a_size));
  CHECK_CL(clSetKernelArg(contribKern, 4, sizeof(cl_int),   (void*)&a_width));
  CHECK_CL(clSetKernelArg(contribKern, 5, sizeof(cl_int),   (void*)&a_height));
  CHECK_CL(clSetKernelArg(contribKern, 6, sizeof(cl_float), (void*)&m_spp));
  CHECK_CL(clSetKernelArg(contribKern, 7, sizeof(cl_float), (void*)&invGamma));

  CHECK_CL(clSetKernelArg(contribKern, 8, sizeof(cl_mem), (void*)&out_colorHDR));
  CHECK_CL(clSetKernelArg(contribKern, 9, sizeof(cl_mem), (void*)&out_colorLDR));
  CHECK_CL(clSetKernelArg(contribKern,10, sizeof(cl_int), (void*)&alreadySorted));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, contribKern, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
  
  // recalculate LDR image rely on normalisation constants
  // 
  if((m_vars.m_flags & HRT_ENABLE_MMLT) != 0 && (m_vars.m_flags & HRT_ENABLE_SBPT) == 0 && a_copyToLDRNow)
  {
    ReduceCLArgs args;
    args.cmdQueue   = m_globals.cmdQueue;
    args.reductionK = m_progs.screen.kernel("ReductionFloat4Avg256");
    
    double avg[4] = {0,0,0,0};
    reduce_average4f_gpu(out_colorHDR, m_width*m_height, avg, args);  
    const float avgBrightness = contribFunc(float3(avg[0], avg[1], avg[2]));
    const float kScale        = m_avgBrightness / fmax(avgBrightness, DEPSILON2);

    runKernel_HDRToLDRWithScale(out_colorHDR, kScale, m_width, m_height, 
                                out_colorLDR);
  }
}

void GPUOCLLayer::runKernel_HDRToLDRWithScale(cl_mem in_colorHDR, float a_kScale, int a_width, int a_height, 
                                              cl_mem out_colorLDR)
{
  cl_kernel contribKern      = m_progs.screen.kernel("HDRToLDRWithScale");
  size_t global_item_size[2] = { size_t(m_width), size_t(m_height) };
  size_t local_item_size[2]  = { 16, 16 };
  RoundBlocks2D(global_item_size, local_item_size);

  const float invGamma = 1.0f/m_globsBuffHeader.varsF[HRT_IMAGE_GAMMA];

  CHECK_CL(clSetKernelArg(contribKern, 0, sizeof(cl_mem), (void*)&in_colorHDR));
  CHECK_CL(clSetKernelArg(contribKern, 1, sizeof(cl_mem), (void*)&out_colorLDR));
  
  CHECK_CL(clSetKernelArg(contribKern, 2, sizeof(cl_float), (void*)&invGamma));
  CHECK_CL(clSetKernelArg(contribKern, 3, sizeof(cl_float), (void*)&a_kScale));
  CHECK_CL(clSetKernelArg(contribKern, 4, sizeof(cl_int),   (void*)&a_width));
  CHECK_CL(clSetKernelArg(contribKern, 5, sizeof(cl_int),   (void*)&a_height));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, contribKern, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


void GPUOCLLayer::runKernel_Trace(cl_mem a_rpos, cl_mem a_rdir, size_t a_size,
                                  cl_mem a_hits)
{
  if (m_globals.cpuTrace)
  {
    runTraceCPU(a_rpos, a_rdir, m_rays.hits, a_size);
  }
  else
  {
    cl_kernel kernTrace1 = m_progs.trace.kernel("BVH4TraversalKernel");
    cl_kernel kernTrace2 = m_progs.trace.kernel("BVH4TraversalInstKernel");
    cl_kernel kernTrace3 = m_progs.trace.kernel("BVH4TraversalInstKernelA");
    cl_kernel kernTrace4 = m_progs.trace.kernel("BVH4TraversalInstKernelAS");

    size_t localWorkSize = 256;
    int    isize         = int(a_size);
    a_size               = roundBlocks(a_size, int(localWorkSize));

    for(int runId = 0; runId < m_scene.bvhNumber; runId++)
    {
      bool smoothOpacity  = m_bvhTrees[runId].smoothOpacity && ((m_vars.m_flags & HRT_ENABLE_MMLT) == 0);

      cl_mem    bvhBuff   = m_scene.bvhBuff    [runId];
      cl_mem    triBuff   = m_scene.objListBuff[runId];
      cl_mem    triAlpha  = m_scene.alphTstBuff[runId];
      cl_kernel kernTrace = m_scene.bvhHaveInst[runId] ? kernTrace2 : kernTrace1;

      if (triAlpha != nullptr)
      {
        if (smoothOpacity)
        {
          kernTrace = kernTrace4;

          CHECK_CL(clSetKernelArg(kernTrace, 0, sizeof(cl_mem), (void*)&a_rpos));
          CHECK_CL(clSetKernelArg(kernTrace, 1, sizeof(cl_mem), (void*)&a_rdir));
          CHECK_CL(clSetKernelArg(kernTrace, 2, sizeof(cl_mem), (void*)&bvhBuff));
          CHECK_CL(clSetKernelArg(kernTrace, 3, sizeof(cl_mem), (void*)&triBuff));

          CHECK_CL(clSetKernelArg(kernTrace, 4, sizeof(cl_mem), (void*)&triAlpha));
          CHECK_CL(clSetKernelArg(kernTrace, 5, sizeof(cl_mem), (void*)&m_scene.storageTex));
          CHECK_CL(clSetKernelArg(kernTrace, 6, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

          CHECK_CL(clSetKernelArg(kernTrace, 7, sizeof(cl_mem), (void*)&m_rays.rayFlags));
          CHECK_CL(clSetKernelArg(kernTrace, 8, sizeof(cl_mem), (void*)&a_hits));
          CHECK_CL(clSetKernelArg(kernTrace, 9, sizeof(cl_mem), (void*)&m_rays.randGenState));

          CHECK_CL(clSetKernelArg(kernTrace, 10, sizeof(cl_int), (void*)&runId));
          CHECK_CL(clSetKernelArg(kernTrace, 11, sizeof(cl_int), (void*)&isize));
        }
        else
        {
          kernTrace = kernTrace3;

          CHECK_CL(clSetKernelArg(kernTrace, 0, sizeof(cl_mem), (void*)&a_rpos));
          CHECK_CL(clSetKernelArg(kernTrace, 1, sizeof(cl_mem), (void*)&a_rdir));
          
          CHECK_CL(clSetKernelArg(kernTrace, 2, sizeof(cl_mem), (void*)&bvhBuff));
          CHECK_CL(clSetKernelArg(kernTrace, 3, sizeof(cl_mem), (void*)&triBuff));
          CHECK_CL(clSetKernelArg(kernTrace, 4, sizeof(cl_mem), (void*)&triAlpha));

          CHECK_CL(clSetKernelArg(kernTrace, 5, sizeof(cl_mem), (void*)&m_scene.storageTex));
          CHECK_CL(clSetKernelArg(kernTrace, 6, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

          CHECK_CL(clSetKernelArg(kernTrace, 7, sizeof(cl_mem), (void*)&m_rays.rayFlags));
          CHECK_CL(clSetKernelArg(kernTrace, 8, sizeof(cl_mem), (void*)&a_hits));
          CHECK_CL(clSetKernelArg(kernTrace, 9, sizeof(cl_int), (void*)&runId));
          CHECK_CL(clSetKernelArg(kernTrace, 10, sizeof(cl_int), (void*)&isize));
        }
      }
      else
      {
        CHECK_CL(clSetKernelArg(kernTrace, 0, sizeof(cl_mem), (void*)&a_rpos));
        CHECK_CL(clSetKernelArg(kernTrace, 1, sizeof(cl_mem), (void*)&a_rdir));
        CHECK_CL(clSetKernelArg(kernTrace, 2, sizeof(cl_mem), (void*)&bvhBuff));
        CHECK_CL(clSetKernelArg(kernTrace, 3, sizeof(cl_mem), (void*)&triBuff));
        CHECK_CL(clSetKernelArg(kernTrace, 4, sizeof(cl_mem), (void*)&m_rays.rayFlags));
        CHECK_CL(clSetKernelArg(kernTrace, 5, sizeof(cl_mem), (void*)&a_hits));
        CHECK_CL(clSetKernelArg(kernTrace, 6, sizeof(cl_int), (void*)&runId));
        CHECK_CL(clSetKernelArg(kernTrace, 7, sizeof(cl_int), (void*)&isize));
      }

      CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernTrace, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
      waitIfDebug(__FILE__, __LINE__);
      
    }

  }

}

void GPUOCLLayer::runKernel_ComputeAO(cl_mem outCompressedAO, size_t a_size)
{
  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  // (1) read AO params and put (rpos, tmax) to m_rays.shadowRayPos
  // (2) create random vector in hemisphere and put it to m_rays.shadowRayDir
  //
  int numIters = 4;

  for (int iter = 0; iter < numIters; iter++)
  {
    cl_kernel kernAO = m_progs.lightp.kernel("MakeAORays");

    CHECK_CL(clSetKernelArg(kernAO, 0, sizeof(cl_mem), (void*)&m_rays.rayFlags));
    CHECK_CL(clSetKernelArg(kernAO, 1, sizeof(cl_mem), (void*)&m_rays.randGenState));

    CHECK_CL(clSetKernelArg(kernAO, 2, sizeof(cl_mem), (void*)&m_rays.hits));
    CHECK_CL(clSetKernelArg(kernAO, 3, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));

    CHECK_CL(clSetKernelArg(kernAO, 4, sizeof(cl_mem), (void*)&m_rays.shadowRayPos));
    CHECK_CL(clSetKernelArg(kernAO, 5, sizeof(cl_mem), (void*)&m_rays.shadowRayDir));

    CHECK_CL(clSetKernelArg(kernAO, 6, sizeof(cl_mem), (void*)&m_scene.storageTex));
    CHECK_CL(clSetKernelArg(kernAO, 7, sizeof(cl_mem), (void*)&m_scene.storageMat));
    CHECK_CL(clSetKernelArg(kernAO, 8, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
    CHECK_CL(clSetKernelArg(kernAO, 9, sizeof(cl_int), (void*)&iter));
    CHECK_CL(clSetKernelArg(kernAO,10, sizeof(cl_int), (void*)&isize));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernAO, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);

    // (3) calc shadows
    //
    runKernel_ShadowTrace(m_rays.rayFlags, m_rays.shadowRayPos, m_rays.shadowRayDir, a_size,
                          m_rays.lshadow);

    // (4) pack them in outCompressedAO
    //
    cl_kernel kernAOPack = m_progs.lightp.kernel("PackAO");

    CHECK_CL(clSetKernelArg(kernAOPack, 0, sizeof(cl_mem), (void*)&m_rays.rayFlags));
    CHECK_CL(clSetKernelArg(kernAOPack, 1, sizeof(cl_mem), (void*)&m_rays.lshadow));
    CHECK_CL(clSetKernelArg(kernAOPack, 2, sizeof(cl_mem), (void*)&outCompressedAO));
    CHECK_CL(clSetKernelArg(kernAOPack, 3, sizeof(cl_int), (void*)&iter));
    CHECK_CL(clSetKernelArg(kernAOPack, 4, sizeof(cl_int), (void*)&numIters));
    CHECK_CL(clSetKernelArg(kernAOPack, 5, sizeof(cl_int), (void*)&isize));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernAOPack, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
}

void GPUOCLLayer::runKernel_ComputeAO2(cl_mem outCompressedAO, size_t a_size, int aoId) // #TODO: implement this! Both Up and down !!!
{
  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  // (1) read AO params and put (rpos, tmax) to m_rays.shadowRayPos
  // (2) create random vector in hemisphere and put it to m_rays.shadowRayDir
  //
  cl_kernel kernAO = m_progs.lightp.kernel("MakeAORaysPacked4"); 

  CHECK_CL(clSetKernelArg(kernAO, 0, sizeof(cl_mem), (void*)&m_rays.rayFlags));
  CHECK_CL(clSetKernelArg(kernAO, 1, sizeof(cl_mem), (void*)&m_rays.randGenState));

  CHECK_CL(clSetKernelArg(kernAO, 2, sizeof(cl_mem), (void*)&m_rays.hits));
  CHECK_CL(clSetKernelArg(kernAO, 3, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));

  CHECK_CL(clSetKernelArg(kernAO, 4, sizeof(cl_mem), (void*)&m_rays.shadowRayPos));
  CHECK_CL(clSetKernelArg(kernAO, 5, sizeof(cl_mem), (void*)&m_rays.shadowRayDir));
  CHECK_CL(clSetKernelArg(kernAO, 6, sizeof(cl_mem), (void*)&m_rays.shadowTemp1i)); // put target inst id to 'shadowTemp1i'

  CHECK_CL(clSetKernelArg(kernAO, 7,  sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernAO, 8,  sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernAO, 9,  sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernAO, 10, sizeof(cl_int), (void*)&aoId));
  CHECK_CL(clSetKernelArg(kernAO, 11, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernAO, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

  // (3) calc shadows
  //
  runKernel_ShadowTraceAO(m_rays.rayFlags, m_rays.shadowRayPos, m_rays.shadowRayDir, m_rays.shadowTemp1i, // read target inst id from 'shadowTemp1i'
                          m_rays.lshadow, AO_RAYS_PACKED*a_size);

  // (4) pack them in outCompressedAO
  //
  cl_kernel kernAOPack = m_progs.lightp.kernel("PackAO4");

  CHECK_CL(clSetKernelArg(kernAOPack, 0, sizeof(cl_mem), (void*)&m_rays.rayFlags));
  CHECK_CL(clSetKernelArg(kernAOPack, 1, sizeof(cl_mem), (void*)&m_rays.lshadow));
  CHECK_CL(clSetKernelArg(kernAOPack, 2, sizeof(cl_mem), (void*)&outCompressedAO));
  CHECK_CL(clSetKernelArg(kernAOPack, 3, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernAOPack, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_ComputeHit(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_hits, size_t a_size, size_t a_sizeRun,
                                       cl_mem out_hitSurface, cl_mem a_outProcTexData)
{
  // eval common surface parameters
  //
  cl_kernel kernHit    = m_progs.trace.kernel("ComputeHit");

  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size,    int(localWorkSize));
  a_sizeRun            = roundBlocks(a_sizeRun, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernHit, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernHit, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernHit, 2, sizeof(cl_mem), (void*)&a_hits));
  
  CHECK_CL(clSetKernelArg(kernHit, 3, sizeof(cl_mem), (void*)&m_scene.matrices));            
  CHECK_CL(clSetKernelArg(kernHit, 4, sizeof(cl_mem), (void*)&m_scene.storageGeom));
  CHECK_CL(clSetKernelArg(kernHit, 5, sizeof(cl_mem), (void*)&m_scene.storageMat));

  CHECK_CL(clSetKernelArg(kernHit, 6, sizeof(cl_mem), (void*)&m_scene.remapLists));
  CHECK_CL(clSetKernelArg(kernHit, 7, sizeof(cl_mem), (void*)&m_scene.remapTable));
  CHECK_CL(clSetKernelArg(kernHit, 8, sizeof(cl_mem), (void*)&m_scene.remapInst));
  
  CHECK_CL(clSetKernelArg(kernHit, 9,  sizeof(cl_mem), (void*)&m_rays.rayFlags));
  CHECK_CL(clSetKernelArg(kernHit, 10, sizeof(cl_mem), (void*)&out_hitSurface));
  
  CHECK_CL(clSetKernelArg(kernHit, 11, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernHit, 12, sizeof(cl_int), (void*)&m_scene.remapTableSize));
  CHECK_CL(clSetKernelArg(kernHit, 13, sizeof(cl_int), (void*)&m_scene.totalInstanceNum));
  CHECK_CL(clSetKernelArg(kernHit, 14, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernHit, 1, NULL, &a_sizeRun, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

  //if (a_doNotEvaluateProcTex)
    //return;

  // eval AO
  //
  if (m_rays.aoCompressed != nullptr)
    runKernel_ComputeAO2(m_rays.aoCompressed , a_sizeRun, 0);

  if(m_rays.aoCompressed2 != nullptr)
    runKernel_ComputeAO2(m_rays.aoCompressed2, a_sizeRun, 1);

  // eval procedure textures
  //
  if (a_outProcTexData != nullptr)
  {
    cl_kernel kernProcT = m_progs.texproc.kernel("ProcTexExec");

    CHECK_CL(clSetKernelArg(kernProcT, 0, sizeof(cl_mem), (void*)&m_rays.rayFlags));
    CHECK_CL(clSetKernelArg(kernProcT, 1, sizeof(cl_mem), (void*)&m_rays.rayDir));

    CHECK_CL(clSetKernelArg(kernProcT, 2, sizeof(cl_mem), (void*)&out_hitSurface));

    CHECK_CL(clSetKernelArg(kernProcT, 3, sizeof(cl_mem), (void*)&m_rays.aoCompressed));
    CHECK_CL(clSetKernelArg(kernProcT, 4, sizeof(cl_mem), (void*)&m_rays.aoCompressed2));
    CHECK_CL(clSetKernelArg(kernProcT, 5, sizeof(cl_mem), (void*)&m_rays.hits));

    CHECK_CL(clSetKernelArg(kernProcT, 6, sizeof(cl_mem), (void*)&m_scene.matrices));
    
    CHECK_CL(clSetKernelArg(kernProcT, 7, sizeof(cl_mem), (void*)&a_outProcTexData));
    CHECK_CL(clSetKernelArg(kernProcT, 8, sizeof(cl_mem), (void*)&m_scene.storageTex));
    CHECK_CL(clSetKernelArg(kernProcT, 9, sizeof(cl_mem), (void*)&m_scene.storageMat));
    CHECK_CL(clSetKernelArg(kernProcT,10, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
    CHECK_CL(clSetKernelArg(kernProcT,11, sizeof(cl_int), (void*)&isize));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernProcT, 1, NULL, &a_sizeRun, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
}

void GPUOCLLayer::runKernel_GetGBufferSamples(cl_mem a_rdir, cl_mem a_gbuff1, cl_mem a_gbuff2, int a_blockSize, size_t a_size)
{
  cl_kernel kernHit    = m_progs.material.kernel("GetGBufferSample");
   
  size_t localWorkSize = a_blockSize;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));
   
  CHECK_CL(clSetKernelArg(kernHit, 0, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernHit, 1, sizeof(cl_mem), (void*)&m_rays.hits));
  CHECK_CL(clSetKernelArg(kernHit, 2, sizeof(cl_mem), (void*)&m_rays.rayFlags));
  CHECK_CL(clSetKernelArg(kernHit, 3, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
  CHECK_CL(clSetKernelArg(kernHit, 4, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

  CHECK_CL(clSetKernelArg(kernHit, 5, sizeof(cl_mem), (void*)&a_gbuff1));
  CHECK_CL(clSetKernelArg(kernHit, 6, sizeof(cl_mem), (void*)&a_gbuff2));

  CHECK_CL(clSetKernelArg(kernHit, 7, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernHit, 8, sizeof(cl_mem), (void*)&m_scene.storageTex)); 
  CHECK_CL(clSetKernelArg(kernHit, 9, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernHit,10, sizeof(cl_int), (void*)&isize));
  
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernHit, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_PutAlphaToGBuffer(cl_mem a_inThoroughput, cl_mem a_gbuff1, size_t a_size)
{
  cl_kernel kernHit    = m_progs.material.kernel("PutAlphaToGBuffer");
   
  size_t localWorkSize = GBUFFER_SAMPLES;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));
  
  CHECK_CL(clSetKernelArg(kernHit, 0, sizeof(cl_mem), (void*)&a_inThoroughput));
  CHECK_CL(clSetKernelArg(kernHit, 1, sizeof(cl_mem), (void*)&a_gbuff1));
  CHECK_CL(clSetKernelArg(kernHit, 2, sizeof(cl_int), (void*)&isize));
  
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernHit, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_GetShadowToAlpha(cl_mem a_color, cl_mem a_shadow, size_t a_size)
{
  cl_kernel kernHit    = m_progs.screen.kernel("GetShadowToAlpha");
   
  size_t localWorkSize = GBUFFER_SAMPLES;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));
  
  CHECK_CL(clSetKernelArg(kernHit, 0, sizeof(cl_mem), (void*)&a_color));
  CHECK_CL(clSetKernelArg(kernHit, 1, sizeof(cl_mem), (void*)&a_shadow));
  CHECK_CL(clSetKernelArg(kernHit, 2, sizeof(cl_int), (void*)&isize));
  
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernHit, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_HitEnvOrLight(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, int a_currBounce, int a_minBounce, size_t a_size)
{
  cl_kernel kernX = m_progs.material.kernel("HitEnvOrLightKernel");

  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  cl_float mLightSubPathCount = cl_float(m_width*m_height); // cl_float(m_rays.MEGABLOCKSIZE);
  cl_int currBounce           = a_currBounce;

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rayFlags));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&m_rays.packedXY));

  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

  CHECK_CL(clSetKernelArg(kernX, 6,  sizeof(cl_mem), (void*)&a_outColor));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_rays.pathThoroughput));     // a_thoroughput
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));     // a_misDataPrev

  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_rays.oldRayDir));           // when PT: use oldRayDir to store emission color
  CHECK_CL(clSetKernelArg(kernX, 10, sizeof(cl_mem), (void*)&m_rays.pathShadow8B));       // 

  CHECK_CL(clSetKernelArg(kernX, 11, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  CHECK_CL(clSetKernelArg(kernX, 12, sizeof(cl_mem), (void*)&m_rays.accPdf));
  CHECK_CL(clSetKernelArg(kernX, 13, sizeof(cl_mem), (void*)&m_rays.oldColor));            // when 3-Way PT pass run, it use unused 'oldColor' to store prevData that is a copy of m_rays.accPdf
  CHECK_CL(clSetKernelArg(kernX, 14, sizeof(cl_mem), (void*)&m_rays.oldFlags));            // when 3-Way PT pass run, it use unused 'oldFlags' to store pdfCamA as single float (sizeof(int) == sizeof(float)) 
  
  CHECK_CL(clSetKernelArg(kernX, 15, sizeof(cl_mem), (void*)&m_scene.storageTex));  
  CHECK_CL(clSetKernelArg(kernX, 16, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kernX, 17, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX, 18, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX, 19, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX, 20, sizeof(cl_mem), (void*)&m_scene.instLightInst));
  CHECK_CL(clSetKernelArg(kernX, 21, sizeof(cl_mem), (void*)&m_rays.hits));

  CHECK_CL(clSetKernelArg(kernX, 22, sizeof(cl_float), (void*)&mLightSubPathCount)); // a_mLightSubPathCount
  CHECK_CL(clSetKernelArg(kernX, 23, sizeof(cl_int),   (void*)&currBounce));         // a_currDepth
  CHECK_CL(clSetKernelArg(kernX, 24, sizeof(cl_int),   (void*)&a_minBounce));         // a_currDepth
  CHECK_CL(clSetKernelArg(kernX, 25, sizeof(cl_int),   (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_NextBounce(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size)
{
  cl_kernel kernX = m_progs.material.kernel("NextBounce");

  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  if (true)
  {
    CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&kmlt.currZind));
    CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&kmlt.currVec));

    CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rpos));
    CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&a_rdir));
    CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&a_rayFlags));
    CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_rays.randGenState));

    CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
    CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

    CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_rays.pathThoroughput));      // a_thoroughput
    CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));      // a_misDataPrev
    CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_rays.lshadow));              // a_shadow
    CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&m_rays.fogAtten));            // a_fog
    CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&m_rays.pathShadeColor));      // in_shadeColor

    if (m_vars.m_flags & HRT_FORWARD_TRACING)
    {
      CHECK_CL(clSetKernelArg(kernX, 14, sizeof(cl_mem), nullptr));
    }
    else
    {
      CHECK_CL(clSetKernelArg(kernX, 14, sizeof(cl_mem), (void*)&m_rays.oldRayDir));          // PT can use unused oldRays as input emission color from kernel HitEnvOrLight
    }

    CHECK_CL(clSetKernelArg(kernX, 15, sizeof(cl_mem), (void*)&m_rays.accPdf));               // a_pdfAcc

    if (m_vars.m_flags & HRT_FORWARD_TRACING)
    {
      CHECK_CL(clSetKernelArg(kernX, 16, sizeof(cl_mem), nullptr));
    }
    else
    {
      CHECK_CL(clSetKernelArg(kernX, 16, sizeof(cl_mem), (void*)&m_rays.oldFlags));           // PT can use unused oldFlags to store camPdfA; require sizeof(int) == sizeof(float);
    }

    CHECK_CL(clSetKernelArg(kernX, 17, sizeof(cl_mem), (void*)&m_scene.storageTex));  
    CHECK_CL(clSetKernelArg(kernX, 18, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
    CHECK_CL(clSetKernelArg(kernX, 19, sizeof(cl_mem), (void*)&m_scene.storageMat));
    CHECK_CL(clSetKernelArg(kernX, 20, sizeof(cl_mem), (void*)&m_scene.storagePdfs));

    CHECK_CL(clSetKernelArg(kernX, 21, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
    CHECK_CL(clSetKernelArg(kernX, 22, sizeof(cl_int), (void*)&isize));
  }

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

}

void GPUOCLLayer::runKernel_NextTransparentBounce(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_thoroughput, size_t a_size)
{
  cl_kernel kernX = m_progs.material.kernel("NextTransparentBounce");

  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&m_rays.rayFlags));

  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&a_thoroughput)); // a_thoroughput
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_ShadowTrace(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, size_t a_size,
                                        cl_mem a_outShadow)
{
  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  cl_kernel kernTrace1 = m_progs.trace.kernel("BVH4TraversalShadowKenrel");
  cl_kernel kernTrace2 = m_progs.trace.kernel("BVH4TraversalInstShadowKenrel");
  cl_kernel kernTrace4 = m_progs.trace.kernel("BVH4TraversalInstShadowKenrelAS");

  for (int runId = 0; runId < m_scene.bvhNumber; runId++)
  {
    cl_mem bvhBuff  = m_scene.bvhBuff[runId];
    cl_mem triBuff  = m_scene.objListBuff[runId];
    cl_mem triAlpha = m_scene.alphTstBuff[runId];

    cl_kernel kernY = m_scene.bvhHaveInst[runId] ? kernTrace2 : kernTrace1;

    if (triAlpha != nullptr)
    {
      kernY = kernTrace4;

      CHECK_CL(clSetKernelArg(kernY, 0, sizeof(cl_mem), (void*)&a_rayFlags));
      CHECK_CL(clSetKernelArg(kernY, 1, sizeof(cl_mem), (void*)&a_rpos));
      CHECK_CL(clSetKernelArg(kernY, 2, sizeof(cl_mem), (void*)&a_rdir));
      CHECK_CL(clSetKernelArg(kernY, 3, sizeof(cl_mem), (void*)&a_outShadow));
     
      CHECK_CL(clSetKernelArg(kernY, 4, sizeof(cl_mem), (void*)&bvhBuff));
      CHECK_CL(clSetKernelArg(kernY, 5, sizeof(cl_mem), (void*)&triBuff));
      CHECK_CL(clSetKernelArg(kernY, 6, sizeof(cl_mem), (void*)&triAlpha));
      CHECK_CL(clSetKernelArg(kernY, 7, sizeof(cl_mem), (void*)&m_scene.storageTex));
      CHECK_CL(clSetKernelArg(kernY, 8, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

      CHECK_CL(clSetKernelArg(kernY, 9, sizeof(cl_int), (void*)&runId));
      CHECK_CL(clSetKernelArg(kernY,10, sizeof(cl_int), (void*)&isize));
    }
    else
    {
      CHECK_CL(clSetKernelArg(kernY, 0, sizeof(cl_mem), (void*)&a_rayFlags));
      CHECK_CL(clSetKernelArg(kernY, 1, sizeof(cl_mem), (void*)&a_rpos));
      CHECK_CL(clSetKernelArg(kernY, 2, sizeof(cl_mem), (void*)&a_rdir));
      CHECK_CL(clSetKernelArg(kernY, 3, sizeof(cl_mem), (void*)&a_outShadow));

      CHECK_CL(clSetKernelArg(kernY, 4, sizeof(cl_mem), (void*)&bvhBuff));
      CHECK_CL(clSetKernelArg(kernY, 5, sizeof(cl_mem), (void*)&triBuff));
      CHECK_CL(clSetKernelArg(kernY, 6, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

      CHECK_CL(clSetKernelArg(kernY, 7, sizeof(cl_int), (void*)&runId));
      CHECK_CL(clSetKernelArg(kernY, 8, sizeof(cl_int), (void*)&isize));
    }

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernY, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
}

void GPUOCLLayer::runKernel_ShadowTraceAO(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_instId,
                                          cl_mem a_outShadow, size_t a_size)
{
  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  cl_kernel kernTrace1 = m_progs.trace.kernel("BVH4TraversalShadowKenrel_Packed");
  cl_kernel kernTrace2 = m_progs.trace.kernel("BVH4TraversalInstShadowKenrel_Packed");
  cl_kernel kernTrace4 = m_progs.trace.kernel("BVH4TraversalInstShadowKenrelAS_Packed");

  for (int runId = 0; runId < m_scene.bvhNumber; runId++)
  {
    cl_mem bvhBuff  = m_scene.bvhBuff    [runId];
    cl_mem triBuff  = m_scene.objListBuff[runId];
    cl_mem triAlpha = m_scene.alphTstBuff[runId];

    cl_kernel kernY = m_scene.bvhHaveInst[runId] ? kernTrace2 : kernTrace1;

    if (triAlpha != nullptr)
    {
      kernY = kernTrace4;

      CHECK_CL(clSetKernelArg(kernY, 0, sizeof(cl_mem), (void*)&a_rayFlags));
      CHECK_CL(clSetKernelArg(kernY, 1, sizeof(cl_mem), (void*)&a_rpos));
      CHECK_CL(clSetKernelArg(kernY, 2, sizeof(cl_mem), (void*)&a_rdir));
      CHECK_CL(clSetKernelArg(kernY, 3, sizeof(cl_mem), (void*)&a_instId));
      CHECK_CL(clSetKernelArg(kernY, 4, sizeof(cl_mem), (void*)&a_outShadow));
     
      CHECK_CL(clSetKernelArg(kernY, 5, sizeof(cl_mem), (void*)&bvhBuff));
      CHECK_CL(clSetKernelArg(kernY, 6, sizeof(cl_mem), (void*)&triBuff));
      CHECK_CL(clSetKernelArg(kernY, 7, sizeof(cl_mem), (void*)&triAlpha));
      CHECK_CL(clSetKernelArg(kernY, 8, sizeof(cl_mem), (void*)&m_scene.storageTex));
      CHECK_CL(clSetKernelArg(kernY, 9, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

      CHECK_CL(clSetKernelArg(kernY,10, sizeof(cl_int), (void*)&runId));
      CHECK_CL(clSetKernelArg(kernY,11, sizeof(cl_int), (void*)&isize));
    }
    else
    {
      CHECK_CL(clSetKernelArg(kernY, 0, sizeof(cl_mem), (void*)&a_rayFlags));
      CHECK_CL(clSetKernelArg(kernY, 1, sizeof(cl_mem), (void*)&a_rpos));
      CHECK_CL(clSetKernelArg(kernY, 2, sizeof(cl_mem), (void*)&a_rdir));
      CHECK_CL(clSetKernelArg(kernY, 3, sizeof(cl_mem), (void*)&a_instId));
      CHECK_CL(clSetKernelArg(kernY, 4, sizeof(cl_mem), (void*)&a_outShadow));

      CHECK_CL(clSetKernelArg(kernY, 5, sizeof(cl_mem), (void*)&bvhBuff));
      CHECK_CL(clSetKernelArg(kernY, 6, sizeof(cl_mem), (void*)&triBuff));
      CHECK_CL(clSetKernelArg(kernY, 7, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

      CHECK_CL(clSetKernelArg(kernY, 8, sizeof(cl_int), (void*)&runId));
      CHECK_CL(clSetKernelArg(kernY, 9, sizeof(cl_int), (void*)&isize));
    }

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernY, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
}

void GPUOCLLayer::ShadePass(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size, bool a_measureTime)
{
  bool transparensyShadowEnabled = false; // !(m_vars.m_flags & HRT_ENABLE_PT_CAUSTICS);

  cl_kernel kernX = m_progs.lightp.kernel("LightSample");
  //cl_kernel kernT = m_progs.material.kernel("TransparentShadowKenrel");
  cl_kernel kernZ = m_progs.material.kernel("Shade");
  cl_kernel kernN = m_progs.trace.kernel("NoShadow");

  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  const bool traceShadows = (m_vars.m_flags & HRT_COMPUTE_SHADOWS);
  
  if (true)
  {
    CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&kmlt.currZind));
    CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&kmlt.currVec));

    CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rpos));
    CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&a_rdir));
    
    CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_rays.rayFlags));
    CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
    CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_rays.randGenState));
    CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_rays.lsamRev));
    
    CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_rays.shadowRayPos));    // float4
    CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_rays.shadowRayDir));    // float4
    
    CHECK_CL(clSetKernelArg(kernX, 10, sizeof(cl_mem), (void*)&m_scene.storageTex));
    CHECK_CL(clSetKernelArg(kernX, 11, sizeof(cl_mem), (void*)&m_scene.storageTexAux)); 
    CHECK_CL(clSetKernelArg(kernX, 12, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
    
    CHECK_CL(clSetKernelArg(kernX, 13, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
    CHECK_CL(clSetKernelArg(kernX, 14, sizeof(cl_int), (void*)&isize));
  }
  
  if (m_globals.cpuTrace)
  {
    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));  
    runTraceShadowCPU(a_size);
    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernZ, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));  
  }
  else
  {
    float timeBeginLightSample = 0.0f;
    float timeLightSample      = 0.0f;
    float timeShadow           = 0.0f;
    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && a_measureTime)
    {
      clFinish(m_globals.cmdQueue);
      timeBeginLightSample = m_timer.getElapsed();
    }
  
    waitIfDebug(__FILE__, __LINE__);
    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));  
    waitIfDebug(__FILE__, __LINE__);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && a_measureTime)
    {
      clFinish(m_globals.cmdQueue);
      timeLightSample       = m_timer.getElapsed();
      m_stat.samLightTimeMs = 1000.0f*(timeLightSample - timeBeginLightSample);
    }

    if (traceShadows)
    {
      runKernel_ShadowTrace(m_rays.rayFlags, m_rays.shadowRayPos, m_rays.shadowRayDir, a_size,
                            m_rays.lshadow);
    }
    else
    {
      CHECK_CL(clSetKernelArg(kernN, 0, sizeof(cl_mem), (void*)&m_rays.lshadow));
      CHECK_CL(clSetKernelArg(kernN, 1, sizeof(cl_int), (void*)&isize));
      CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernN, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    }

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && a_measureTime)
    {
      clFinish(m_globals.cmdQueue);
      timeShadow          = m_timer.getElapsed();
      m_stat.shadowTimeMs = 1000.0f*(timeShadow - timeLightSample);
    }

    CHECK_CL(clSetKernelArg(kernZ, 0, sizeof(cl_mem), (void*)&a_rpos));
    CHECK_CL(clSetKernelArg(kernZ, 1, sizeof(cl_mem), (void*)&a_rdir));
    CHECK_CL(clSetKernelArg(kernZ, 2, sizeof(cl_mem), (void*)&m_rays.rayFlags));
    CHECK_CL(clSetKernelArg(kernZ, 3, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
   
    CHECK_CL(clSetKernelArg(kernZ, 4, sizeof(cl_mem), (void*)&m_rays.lshadow));
    CHECK_CL(clSetKernelArg(kernZ, 5, sizeof(cl_mem), (void*)&m_rays.lsamRev));
    CHECK_CL(clSetKernelArg(kernZ, 6, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

    CHECK_CL(clSetKernelArg(kernZ, 7, sizeof(cl_mem), (void*)&m_rays.oldColor));        // pdfAccCopy
    CHECK_CL(clSetKernelArg(kernZ, 8, sizeof(cl_mem), (void*)&m_rays.oldFlags));        // camPdfA

    CHECK_CL(clSetKernelArg(kernZ, 9, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernZ, 10, sizeof(cl_mem), (void*)&m_rays.pathShadow8B));

    CHECK_CL(clSetKernelArg(kernZ, 11, sizeof(cl_mem), (void*)&m_scene.storageTex));
    CHECK_CL(clSetKernelArg(kernZ, 12, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
    CHECK_CL(clSetKernelArg(kernZ, 13, sizeof(cl_mem), (void*)&m_scene.storageMat));
    CHECK_CL(clSetKernelArg(kernZ, 14, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
    CHECK_CL(clSetKernelArg(kernZ, 15, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
    CHECK_CL(clSetKernelArg(kernZ, 16, sizeof(cl_int), (void*)&isize));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernZ, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));  
    waitIfDebug(__FILE__, __LINE__);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && a_measureTime)
    {
      clFinish(m_globals.cmdQueue);
      m_stat.shadeTimeMs = 1000.0f*(m_timer.getElapsed() - timeShadow);
    }
  }
  
  waitIfDebug(__FILE__, __LINE__);

}

void GPUOCLLayer::runKernel_EyeShadowRays(cl_mem a_rayFlags, cl_mem a_rdir2,
                                          cl_mem a_rpos,     cl_mem a_rdir, size_t a_size)
{
  const int usematerials = (a_rdir2 == nullptr) ? 0 : 1;

  cl_kernel kernMakeRays = m_progs.material.kernel("MakeEyeShadowRays");

  size_t localWorkSize   = 256;
  int    isize           = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernMakeRays, 0, sizeof(cl_mem), (void*)&a_rayFlags));
  CHECK_CL(clSetKernelArg(kernMakeRays, 1, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));

  CHECK_CL(clSetKernelArg(kernMakeRays, 2, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernMakeRays, 3, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

  CHECK_CL(clSetKernelArg(kernMakeRays, 4, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernMakeRays, 5, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernMakeRays, 6, sizeof(cl_int), (void*)&isize));
  CHECK_CL(clSetKernelArg(kernMakeRays, 7, sizeof(cl_int), (void*)&usematerials));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernMakeRays, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_ProjectSamplesToScreen(cl_mem a_rayFlags, cl_mem a_rdir, cl_mem a_rdir2, cl_mem a_colorsIn,
                                                   cl_mem a_colorsOut, cl_mem a_zindex, size_t a_size, int a_currBounce)
{
  cl_kernel kern       = m_progs.material.kernel("ConnectToEyeKernel");

  size_t localWorkSize = 256;
  cl_int isize         = cl_int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  cl_float mLightSubPathCount = cl_float(m_width*m_height); //
  cl_int currBounce           = a_currBounce+1;  

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&a_rayFlags));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&a_rdir2));
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kern, 3, sizeof(cl_mem), (void*)&m_rays.lshadow));
  CHECK_CL(clSetKernelArg(kern, 4, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));

  CHECK_CL(clSetKernelArg(kern, 5, sizeof(cl_mem), (void*)&m_rays.accPdf));
  CHECK_CL(clSetKernelArg(kern, 6, sizeof(cl_mem), (void*)&m_rays.lightOffsetBuff)); //#TODO: remove this buffer, store it inside 'lsamRev'
  CHECK_CL(clSetKernelArg(kern, 7, sizeof(cl_mem), (void*)&m_rays.lsamRev));
  CHECK_CL(clSetKernelArg(kern, 8, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));
  CHECK_CL(clSetKernelArg(kern, 9, sizeof(cl_mem), (void*)&m_rays.randGenState));

  CHECK_CL(clSetKernelArg(kern, 10, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kern, 11, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kern, 12, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kern, 13, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kern, 14, sizeof(cl_mem), (void*)&m_globals.cMortonTable));

  CHECK_CL(clSetKernelArg(kern, 15, sizeof(cl_mem), (void*)&a_colorsIn));
  CHECK_CL(clSetKernelArg(kern, 16, sizeof(cl_mem), (void*)&a_colorsOut));
  CHECK_CL(clSetKernelArg(kern, 17, sizeof(cl_mem), (void*)&a_zindex));
 
  CHECK_CL(clSetKernelArg(kern, 18, sizeof(cl_float), (void*)&mLightSubPathCount));
  CHECK_CL(clSetKernelArg(kern, 19, sizeof(cl_int),   (void*)&currBounce));
  CHECK_CL(clSetKernelArg(kern, 20, sizeof(cl_int),   (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


void GPUOCLLayer::runKernel_UpdateForwardPdfFor3Way(cl_mem a_flags, cl_mem old_rayDir, cl_mem next_rayDir, cl_mem acc_pdf, size_t a_size)
{
  cl_kernel kern       = m_progs.material.kernel("UpdateForwardPdfFor3Way");

  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&a_flags));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&old_rayDir));
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&next_rayDir));

  CHECK_CL(clSetKernelArg(kern, 3, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
  CHECK_CL(clSetKernelArg(kern, 4, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

  CHECK_CL(clSetKernelArg(kern, 5, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  CHECK_CL(clSetKernelArg(kern, 6, sizeof(cl_mem), (void*)&acc_pdf)); // m_rays.accPdf

  CHECK_CL(clSetKernelArg(kern, 7, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kern, 8, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kern, 9, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kern,10, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kern,11, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


void GPUOCLLayer::runKernel_InitRandomGen(cl_mem a_buffer, size_t a_size, int a_seed)
{
  cl_kernel kernInitR  = m_progs.trace.kernel("InitRandomGen");

  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  //std::cerr << "a_size = " << a_size << std::endl

  CHECK_CL(clSetKernelArg(kernInitR, 0, sizeof(cl_mem), (void*)&a_buffer));
  CHECK_CL(clSetKernelArg(kernInitR, 1, sizeof(cl_int), (void*)&a_seed));
  CHECK_CL(clSetKernelArg(kernInitR, 2, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernInitR, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


//// ML Render
//

void GPUOCLLayer::runKernel_GenerateSPPRays(cl_mem a_pixels, cl_mem a_sppPos, cl_mem a_rpos, cl_mem a_rdir, size_t a_size, int a_blockSize)
{
  cl_kernel kernX = m_progs.screen.kernel("MakeEyeRaysSPP");

  int isize = int(a_size);
  size_t localWorkSize = size_t(a_blockSize);
  a_size = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_pixels));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rdir));

  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&m_width));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_int), (void*)&m_height));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_int), (void*)&a_blockSize));

  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&a_sppPos));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


void GPUOCLLayer::runKernel_ReductionFloat4Average(cl_mem a_src, cl_mem a_dst, size_t a_size, int a_bsize)
{
  if (a_bsize != 256 && a_bsize != 64 && a_bsize != 16)
    RUN_TIME_ERROR("ReductionFloat4Avg256 not implemented for bsize != 256/64/16");

  //ReductionFloat4Avg16
  cl_kernel kernX = m_progs.screen.kernel("ReductionFloat4Avg256");
  if (a_bsize == 64)
    kernX = m_progs.screen.kernel("ReductionFloat4Avg64");
  else if (a_bsize == 16)
    kernX = m_progs.screen.kernel("ReductionFloat4Avg16");

  size_t localWorkSize = size_t(a_bsize);
  int    isize         = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_src));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_dst));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

bool GPUOCLLayer::testSimpleReduction()
{
  int testData[256];

  for (int i = 0; i < 256; i++)
    testData[i] = rand() % 100;

  int goldSumm = 0;
  for (int i = 0; i < 256; i++)
    goldSumm += testData[i];

  cl_int ciErr1;
  cl_mem testBuffer = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 256*sizeof(int), (void*)(testData), &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in clCreateBuffer");

  cl_kernel kernX = m_progs.screen.kernel("SimpleReductionTest");

  size_t a_size        = 256;
  int    isize         = 256;
  size_t localWorkSize = 256;
  a_size = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&testBuffer));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));

  int resSumm = 0;
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, testBuffer, CL_TRUE, 0, sizeof(int), &resSumm, 0, NULL, NULL));

  clReleaseMemObject(testBuffer);

  return (resSumm == goldSumm);
}




void GPUOCLLayer::memsetu32(cl_mem buff, uint a_val, size_t a_size)
{
  cl_kernel kern = m_progs.screen.kernel("MemSetu32");

  size_t szLocalWorkSize = 256;
  cl_int iNumElements = cl_int(a_size);
  a_size = roundBlocks(a_size, int(szLocalWorkSize));

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&buff));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_uint), (void*)&a_val));
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&iNumElements));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &szLocalWorkSize, 0, NULL, NULL));
}

void GPUOCLLayer::memsetf4(cl_mem buff, float4 a_val, size_t a_size, size_t a_offset)
{
  cl_kernel kern = m_progs.screen.kernel("MemSetf4");

  size_t szLocalWorkSize = 256;
  cl_int iNumElements    = cl_int(a_size);
  cl_int iOffset         = cl_int(a_offset);
  a_size                 = roundBlocks(a_size, int(szLocalWorkSize));

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&buff));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_float4), (void*)&a_val));
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&iNumElements));
  CHECK_CL(clSetKernelArg(kern, 3, sizeof(cl_int), (void*)&iOffset));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &szLocalWorkSize, 0, NULL, NULL));
}

void GPUOCLLayer::memcpyu32(cl_mem buff1, uint a_offset1, cl_mem buff2, uint a_offset2, size_t a_size)
{
  if (buff2 == 0 || buff1 == 0)
    return;

  cl_kernel kern = m_progs.screen.kernel("MemCopyu32");

  size_t szLocalWorkSize = 256;
  cl_int iNumElements    = cl_int(a_size);
  a_size                 = roundBlocks(a_size, int(szLocalWorkSize));

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&buff2));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_uint), (void*)&a_offset2));
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&buff1));
  CHECK_CL(clSetKernelArg(kern, 3, sizeof(cl_uint), (void*)&a_offset1));
  CHECK_CL(clSetKernelArg(kern, 4, sizeof(cl_int), (void*)&iNumElements));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &szLocalWorkSize, 0, NULL, NULL));
}


void GPUOCLLayer::float2half(cl_mem tempFloatBuff, cl_mem tempHalfBuff, size_t a_size)
{
  cl_kernel kernX = m_progs.screen.kernel("FloatToHalf");
  int isize = int(a_size);
  int iOffset = 0;
  size_t localWorkSize = 256;
  a_size = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&tempFloatBuff));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&tempHalfBuff));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&iOffset));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
}

void GPUOCLLayer::float2half(const float* a_inData, size_t a_size, std::vector<cl_half>& a_out)
{
  if (a_size == 0)
    return;

  a_out.resize(a_size);

  cl_int ciErr1;

  cl_mem tempFloatBuff = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_size * sizeof(float), (float*)a_inData, &ciErr1); if (ciErr1 != CL_SUCCESS) RUN_TIME_ERROR("Error in clCreateBuffer");
  cl_mem tempHalfBuff  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, a_size*sizeof(cl_half), NULL, &ciErr1);                                   if (ciErr1 != CL_SUCCESS) RUN_TIME_ERROR("Error in clCreateBuffer");

  float2half(tempFloatBuff, tempHalfBuff, a_size);

  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, tempHalfBuff, CL_TRUE, 0, a_size*sizeof(cl_half), &a_out[0], 0, NULL, NULL));

  clReleaseMemObject(tempFloatBuff);
  clReleaseMemObject(tempHalfBuff);
}

void GPUOCLLayer::float2half(const std::vector<float>& a_in, std::vector<cl_half>& a_out)
{
  if (a_in.size() == 0 || a_in.size() != a_out.size())
    return;

  float2half(&a_in[0], a_in.size(), a_out);
}

void GPUOCLLayer::runKernel_ClampFloat4(cl_mem buff1, cl_float a_min, cl_float a_max, size_t a_size)
{
  if(buff1 == 0)
    return;

  cl_kernel kern = m_progs.screen.kernel("ClampFloat4");

  size_t szLocalWorkSize = 256;
  cl_int iNumElements    = cl_int(a_size);
  a_size                 = roundBlocks(a_size, int(szLocalWorkSize));

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem),   (void*)&buff1));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_float), (void*)&a_min));
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_float), (void*)&a_max));
  CHECK_CL(clSetKernelArg(kern, 3, sizeof(cl_int),   (void*)&iNumElements));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &szLocalWorkSize, 0, NULL, NULL));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float2 GPUOCLLayer::runKernel_TestAtomicsPerf(size_t a_size)
{
  // return make_float2(0, 0);

  Timer myTimer;

  cl_mem buff = m_screen.color0;

  cl_kernel kernX = m_progs.screen.kernel("TestAtomicsInt");
  cl_kernel kernY = m_progs.screen.kernel("TestAtomicsFloat");
  

  int isize            = int(a_size);
  size_t localWorkSize = 256;
  a_size               = roundBlocks(a_size, int(localWorkSize));

  memsetf4(buff, float4(0, 0, 0, 0), a_size);

  clFinish(m_globals.cmdQueue);
  myTimer.start();

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&buff));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));

  clFinish(m_globals.cmdQueue);
  float time1 = myTimer.getElapsed()*1000.0f;
  
  memsetf4(buff, float4(0, 0, 0, 0), a_size);
  clFinish(m_globals.cmdQueue);
  myTimer.start();

  CHECK_CL(clSetKernelArg(kernY, 0, sizeof(cl_mem), (void*)&buff));
  CHECK_CL(clSetKernelArg(kernY, 1, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernY, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));

  clFinish(m_globals.cmdQueue);
  float time2 = myTimer.getElapsed()*1000.0f;

  if (0)
  {
    std::vector<float> testData(1024);
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, buff, CL_TRUE, 0, testData.size()*sizeof(float), &testData[0], 0, NULL, NULL));

    std::ofstream out("testAtomicsf2.txt");
    for (int i = 0; i < testData.size(); i++)
      out << testData[i] << std::endl;
    out.close();
  }

  return make_float2(time1, time2);
}


#include "GPUOCLLayer.h"
#include "crandom.h"
#include "cl_scan_gpu.h"

#include <algorithm>
#undef min
#undef max

void GPUOCLLayer::trace1D_Rev(int a_minBounce, int a_maxBounce, cl_mem a_rpos, cl_mem a_rdir, size_t a_size,
                              cl_mem a_outColor)
{
  runKernel_ClearAllInternalTempBuffers(a_size);

  // trace rays
  //
  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    m_timer.start();
  }

  float timeForSample    = 0.0f;
  float timeForBounce    = 0.0f;
  float timeForTrace     = 0.0f;
  float timeBeforeShadow = 0.0f;
  float timeForShadow    = 0.0f;
  float timeStart        = 0.0f;

  float timeNextBounceStart = 0.0f;
  float timeForNextBounce   = 0.0f;

  float timeForHitStart = 0.0f;
  float timeForHit      = 0.0f;

  int measureBounce = m_vars.m_varsI[HRT_MEASURE_RAYS_TYPE];

  for (int bounce = 0; bounce < a_maxBounce; bounce++)
  {
    const bool measureThisBounce = (bounce == measureBounce);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeStart = m_timer.getElapsed();
    }

    runKernel_Trace(a_rpos, a_rdir, a_size,
                    m_rays.hits);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeForHitStart = m_timer.getElapsed();
      timeForTrace    = timeForHitStart - timeStart;
    }

    runKernel_ComputeHit(a_rpos, a_rdir, m_rays.hits, a_size, a_size,
                         m_rays.hitSurfaceAll, m_rays.hitProcTexData);

    runKernel_HitEnvOrLight(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, bounce, a_minBounce, a_size);

    if (FORCE_DRAW_SHADOW && bounce == 1)
    {
      CopyShadowTo(a_outColor, m_rays.MEGABLOCKSIZE);
      break;
    }

    if((m_vars.m_flags & HRT_PRODUCTION_IMAGE_SAMPLING) != 0 && (bounce%2 == 0)) // opt for empty environment rendering.
    {
      if(CountNumActiveThreads(m_rays.rayFlags, a_size) == 0)
        break;
    }

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeBeforeShadow = m_timer.getElapsed();
      timeForHit       = timeBeforeShadow - timeForHitStart;
    }

    if (m_vars.shadePassEnable(bounce, a_minBounce, a_maxBounce))
    {
      ShadePass(a_rpos, a_rdir, m_rays.pathShadeColor, a_size, measureThisBounce);
    }
    else
    {
      memsetf4(m_rays.pathShadeColor, float4(0,0,0,0), a_size);
    }
    
    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeNextBounceStart = m_timer.getElapsed();
      timeForShadow       = timeNextBounceStart - timeBeforeShadow;
    }

    runKernel_NextBounce(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, a_size);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeForBounce     = (m_timer.getElapsed() - timeStart);
      timeForNextBounce = (m_timer.getElapsed() - timeNextBounceStart);
    }
  }

  if(m_vars.m_varsF[HRT_PATH_TRACE_CLAMPING] < 1e+6f)
    runKernel_ClampFloat4(a_outColor, 0.0f, m_vars.m_varsF[HRT_PATH_TRACE_CLAMPING], a_size);  // perform clamping if it is enabled


  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    timeForSample = m_timer.getElapsed();
  }

  m_stat.raysPerSec      = float(a_size) / timeForTrace;
  m_stat.traversalTimeMs = timeForTrace*1000.0f;
  m_stat.sampleTimeMS    = timeForSample*1000.0f;
  m_stat.bounceTimeMS    = timeForBounce*1000.0f;
  //m_stat.shadowTimeMs    = timeForShadow*1000.0f;
  m_stat.evalHitMs       = timeForHit*1000.0f;
  m_stat.nextBounceMs    = timeForNextBounce*1000.0f;

  m_stat.samplesPerSec    = float(a_size) / timeForSample;
  m_stat.traceTimePerCent = int( ((timeForTrace + timeForShadow) / timeForBounce)*100.0f );
  //std::cout << "measureBounce = " << measureBounce << std::endl;
}


void GPUOCLLayer::trace1D_Fwd(int a_minBounce, int a_maxBounce, cl_mem a_rpos, cl_mem a_rdir, size_t a_size,
                              cl_mem a_outColor)
{
  
  // runKernel_ClearAllInternalTempBuffers(a_size); // called when light is sampled

  // trace rays
  //
  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    m_timer.start();
  }

  float timeForSample    = 0.0f;
  float timeForBounce    = 0.0f;
  float timeForTrace     = 0.0f;
  float timeBeforeShadow = 0.0f;
  float timeForShadow    = 0.0f;
  float timeStart        = 0.0f;

  float timeNextBounceStart = 0.0f;
  float timeForNextBounce   = 0.0f;

  float timeForHitStart = 0.0f;
  float timeForHit      = 0.0f;

  //int measureBounce = m_vars.m_varsI[HRT_MEASURE_RAYS_TYPE];

  for (int bounce = 0; bounce < a_maxBounce - 1; bounce++)
  {
    runKernel_Trace(a_rpos, a_rdir, a_size,
                    m_rays.hits);

    runKernel_ComputeHit(a_rpos, a_rdir, m_rays.hits, a_size, a_size,
                         m_rays.hitSurfaceAll, m_rays.hitProcTexData);
                         
    // postpone 'ConnectEyePass' call to the end of bounce;
    // ConnectEyePass(m_rays.rayFlags, m_rays.hitPosNorm, m_rays.hitNormUncompressed, a_rdir, a_outColor, bounce, a_size);
    //
    if(bounce >= a_minBounce - 2)
    {
      CopyForConnectEye(m_rays.rayFlags, a_rdir,             a_outColor,
                        m_rays.oldFlags, m_rays.oldRayDir,   m_rays.oldColor, a_size);
    }

    runKernel_NextBounce(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, a_size);
   
    if(bounce >= a_minBounce - 2)
    {
      ConnectEyePass(m_rays.oldFlags, m_rays.oldRayDir, m_rays.oldColor, bounce, a_size);
      if (m_vars.m_flags & HRT_3WAY_MIS_WEIGHTS)
        runKernel_UpdateForwardPdfFor3Way(m_rays.oldFlags, m_rays.oldRayDir, m_rays.rayDir, m_rays.accPdf, a_size);
    }
  }


  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    timeForSample = m_timer.getElapsed();
  }

  m_stat.raysPerSec      = float(a_size) / timeForTrace;
  m_stat.traversalTimeMs = timeForTrace*1000.0f;
  m_stat.sampleTimeMS    = timeForSample*1000.0f;
  m_stat.bounceTimeMS    = timeForBounce*1000.0f;
  //m_stat.shadowTimeMs    = timeForShadow*1000.0f;
  m_stat.evalHitMs       = timeForHit*1000.0f;
  m_stat.nextBounceMs    = timeForNextBounce*1000.0f;

  m_stat.samplesPerSec    = float(a_size) / timeForSample;
  m_stat.traceTimePerCent = int( ((timeForTrace + timeForShadow) / timeForBounce)*100.0f );
  //std::cout << "measureBounce = " << measureBounce << std::endl;
}

void GPUOCLLayer::EvalPT(cl_mem in_xVector, cl_mem in_zind, int minBounce, int maxBounce, size_t a_size,
                         cl_mem a_outColor)
{ 
  // This is important fix due to dead threads on last state (black env) could contrib old values (!!!)
  // Thus, you have to clear 'kmlt.xMultOneMinusAlpha' as trace1D_Rev result because trace1D_Rev _always_ do "+=" operation on result buffer.
  // 
  memsetf4(a_outColor, float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); 
  
  // create rays from 'in_xVector' samples
  //
  runKernel_MakeRaysFromEyeSam(in_zind, in_xVector, m_rays.MEGABLOCKSIZE, m_passNumberForQMC,
                               m_rays.rayPos, m_rays.rayDir);
   
  auto temp     = kmlt.currVec;  // save
  auto temp2    = kmlt.currZind; // save 

  kmlt.currVec  = in_xVector;
  kmlt.currZind = in_zind;

  trace1D_Rev(minBounce, maxBounce, m_rays.rayPos, m_rays.rayDir, m_rays.MEGABLOCKSIZE,
              a_outColor);

  kmlt.currVec  = temp;  // restore
  kmlt.currZind = temp2; // restore
}

void GPUOCLLayer::EvalLT(cl_mem in_xVector, int minBounce, int maxBounce, size_t a_size,
                         cl_mem a_outColor)
{
  assert(in_xVector == nullptr);  // #NOTE: This is not implemeneted yet!!! (and actually will not be implemnented, no reason why we should do that)

  auto temp      = kmlt.currVec;  // save
  auto temp2     = kmlt.currZind; // save 

  kmlt.currVec  = nullptr;
  kmlt.currZind = nullptr;
  
  /////////////////////////////////////////////////////////////////////////////////////////////////

  runKernel_MakeLightRays(m_rays.rayPos, m_rays.rayDir, a_outColor, m_rays.MEGABLOCKSIZE);

  if ((m_vars.m_flags & HRT_DRAW_LIGHT_LT) && minBounce <= 1 ) 
  {
    runKernel_CopyLightSampleToSurfaceHit(m_rays.rayPos, m_rays.MEGABLOCKSIZE,
                                          m_rays.hitSurfaceAll);

    ConnectEyePass(m_rays.rayFlags, nullptr, a_outColor, -1, m_rays.MEGABLOCKSIZE);
  }

  trace1D_Fwd(minBounce, maxBounce, m_rays.rayPos, m_rays.rayDir, m_rays.MEGABLOCKSIZE,
              a_outColor);

  /////////////////////////////////////////////////////////////////////////////////////////////////

  kmlt.currVec  = temp;  // restore
  kmlt.currZind = temp2; // restore
}

int GPUOCLLayer::CountNumActiveThreads(cl_mem a_rayFlags, size_t a_size)
{
  int zero = 0;
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_rays.atomicCounterMem, CL_TRUE, 0,
                                sizeof(int), &zero, 0, NULL, NULL));

  {
    cl_kernel kern = m_progs.screen.kernel("CountNumLiveThreads");

    size_t szLocalWorkSize = 256;
    cl_int iNumElements    = cl_int(a_size);
    a_size                 = roundBlocks(a_size, int(szLocalWorkSize));

    CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem),  (void*)&m_rays.atomicCounterMem));
    CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem),  (void*)&a_rayFlags));
    CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int),  (void*)&iNumElements));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &szLocalWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }

  int counter = 0;
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.atomicCounterMem, CL_TRUE, 0,
                               sizeof(int), &counter, 0, NULL, NULL));

  return counter;
}

void GPUOCLLayer::trace1DPrimaryOnly(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size, size_t a_offset)
{
  cl_kernel kernShowN = m_progs.trace.kernel("ShowNormals");
  cl_kernel kernShowT = m_progs.trace.kernel("ShowTexCoord");
  cl_kernel kernFill  = m_progs.trace.kernel("ColorIndexTriangles");

  //cl_kernel kern = m_progs.screen.kernel("FillColorTest");

  size_t localWorkSize = 256;
  int    isize   = int(a_size);
  int    ioffset = int(a_offset);

  // trace rays
  //
  memsetu32(m_rays.rayFlags, 0, a_size);                                                           // fill flags with zero data
  memsetf4(m_rays.hitSurfaceAll, float4(0,0,0,0), SURFACE_HIT_SIZE_IN_F4*m_rays.MEGABLOCKSIZE, 0); // clear surface hit

  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    m_timer.start();
  }

  runKernel_Trace(a_rpos, a_rdir, a_size,
                  m_rays.hits);

  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    m_stat.raysPerSec = float(a_size) / m_timer.getElapsed();
  }

  runKernel_ComputeHit(a_rpos, a_rdir, m_rays.hits, a_size, a_size,
                       m_rays.hitSurfaceAll, m_rays.hitProcTexData);

  //
  //
  if (true)
  {
    CHECK_CL(clSetKernelArg(kernShowN, 0, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
    CHECK_CL(clSetKernelArg(kernShowN, 1, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernShowN, 2, sizeof(cl_int), (void*)&isize));
    CHECK_CL(clSetKernelArg(kernShowN, 3, sizeof(cl_int), (void*)&ioffset));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernShowN, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
  else if (false)
  {
    CHECK_CL(clSetKernelArg(kernShowT, 0, sizeof(cl_mem), (void*)&m_rays.hitSurfaceAll));
    CHECK_CL(clSetKernelArg(kernShowT, 1, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernShowT, 2, sizeof(cl_int), (void*)&isize));
    CHECK_CL(clSetKernelArg(kernShowT, 3, sizeof(cl_int), (void*)&ioffset));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernShowT, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
  else
  {
    CHECK_CL(clSetKernelArg(kernFill, 0, sizeof(cl_mem), (void*)&m_rays.hits));
    CHECK_CL(clSetKernelArg(kernFill, 1, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernFill, 2, sizeof(cl_int), (void*)&isize));
    CHECK_CL(clSetKernelArg(kernFill, 3, sizeof(cl_int), (void*)&ioffset));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernFill, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }

}

void GPUOCLLayer::CopyShadowTo(cl_mem a_color, size_t a_size)
{
  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  cl_kernel kern         = m_progs.screen.kernel("CopyShadowTo");

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&m_rays.pathShadow8B));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&a_color)); 
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&iSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::DrawNormals()
{
  //memsetf4(m_screen.color0, float4(0,0,0,0), m_width*m_height, 0);

  cl_kernel makeRaysKern = m_progs.screen.kernel("MakeEyeRays");

  int iter = 0;
  for (size_t offset = 0; offset < m_width*m_height; offset += m_rays.MEGABLOCKSIZE)
  {
    size_t localWorkSize = 256;
    size_t globalWorkSize = m_rays.MEGABLOCKSIZE;
    cl_int iOffset = cl_int(offset);

    if (offset + globalWorkSize > (m_width*m_height))
      globalWorkSize = (m_width*m_height) - offset;

    CHECK_CL(clSetKernelArg(makeRaysKern, 0, sizeof(cl_int), (void*)&iOffset));
    CHECK_CL(clSetKernelArg(makeRaysKern, 1, sizeof(cl_mem), (void*)&m_rays.rayPos));
    CHECK_CL(clSetKernelArg(makeRaysKern, 2, sizeof(cl_mem), (void*)&m_rays.rayDir));
    CHECK_CL(clSetKernelArg(makeRaysKern, 3, sizeof(cl_int), (void*)&m_width));
    CHECK_CL(clSetKernelArg(makeRaysKern, 4, sizeof(cl_int), (void*)&m_height));
    CHECK_CL(clSetKernelArg(makeRaysKern, 5, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

    if (globalWorkSize % localWorkSize != 0)
      globalWorkSize = (globalWorkSize / localWorkSize)*localWorkSize;

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeRaysKern, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);

    trace1DPrimaryOnly(m_rays.rayPos, m_rays.rayDir, m_screen.color0, globalWorkSize, offset); //  m_screen.colorSubBuffers[iter]
    iter++;
  }

  cl_mem tempLDRBuff = m_screen.pbo;

  //if (!(m_initFlags & GPU_RT_NOWINDOW))
  //  CHECK_CL(clEnqueueAcquireGLObjects(m_globals.cmdQueue, 1, &tempLDRBuff, 0, 0, 0));

  cl_kernel colorKern = m_progs.screen.kernel("RealColorToRGB256");

  size_t global_item_size[2] = { size_t(m_width), size_t(m_height) };
  size_t local_item_size[2]  = { 16, 16 };

  RoundBlocks2D(global_item_size, local_item_size);

  CHECK_CL(clSetKernelArg(colorKern, 0, sizeof(cl_mem), (void*)&m_screen.color0));
  CHECK_CL(clSetKernelArg(colorKern, 1, sizeof(cl_mem), (void*)&tempLDRBuff));
  CHECK_CL(clSetKernelArg(colorKern, 2, sizeof(cl_int), (void*)&m_width));
  CHECK_CL(clSetKernelArg(colorKern, 3, sizeof(cl_int), (void*)&m_height));
  CHECK_CL(clSetKernelArg(colorKern, 4, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(colorKern, 5, sizeof(cl_float), (void*)&m_globsBuffHeader.varsF[HRT_IMAGE_GAMMA]));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, colorKern, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);

  //if (!(m_initFlags & GPU_RT_NOWINDOW))
  //  CHECK_CL(clEnqueueReleaseGLObjects(m_globals.cmdQueue, 1, &tempLDRBuff, 0, 0, 0));
}

/**
\file
\brief Metropolis algorithm implementation in OpenCL

*/

#include "GPUOCLLayer.h"
#include "crandom.h"

#include "../../HydraAPI/hydra_api/xxhash.h"
#include "cl_scan_gpu.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cl_mem GPUOCLLayer::FrameBuff_Get(int a_id) const
{
  if (a_id == 0)
    return m_screen.color0;
  else
    return nullptr;
}


bool GPUOCLLayer::FrameBuff_IsAllocated(int a_id) const
{
  return (FrameBuff_Get(a_id) != 0);
}

void GPUOCLLayer::FrameBuff_Alloc(int a_id)
{
  if (a_id == 0)
    return;

  FrameBuff_Free(a_id);

  cl_int ciErr1 = CL_SUCCESS;

  if (a_id == 1)
  {
    
  }
  else if (a_id == 2)
  {
   
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create cl full screen additional buffer");
}

void GPUOCLLayer::FrameBuff_Free(int a_id)
{
  if (a_id == 0)
    return;

  if (a_id == 1)
  {
   
  }
  else if (a_id == 2)
  {
   
  }

}


void GPUOCLLayer::FrameBuff_Clear(int a_id)
{
  cl_mem targetBuff = FrameBuff_Get(a_id);
  memsetf4(targetBuff, float4(0, 0, 0, 0), m_width*m_height, 0);
}

float4 GPUOCLLayer::FrameBuff_AverageColor(int a_id)
{
  cl_mem targetBuff = FrameBuff_Get(a_id);

  ReduceCLArgs args;
  args.cmdQueue   = m_globals.cmdQueue;
  args.reductionK = m_progs.screen.kernel("ReductionFloat4Avg256");

  float4 avg(0,0,0,0);
  reduce_average4f_gpu(targetBuff, m_width*m_height, &avg.x, args);
  
  //if(0)
  //{
  //  std::vector<float4> screenColor(m_width*m_height);
  //  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, targetBuff, CL_TRUE, 0, screenColor.size()*sizeof(float4), &screenColor[0], 0, NULL, NULL));
  //  
  //  float4 summ2(0, 0, 0, 0);
  //  for (size_t i = 0; i < screenColor.size(); i++)
  //  summ2 += screenColor[i];
  //  
  //  float4 avg2 = summ2 * (1.0f / float(m_width*m_height));
  //  
  //  std::cout << "avg  = " << avg  << std::endl;
  //  std::cout << "avg2 = " << avg2 << std::endl;
  //}

  return avg;
}

float4 GPUOCLLayer::FrameBuff_AverageSqrtColor(int a_id)
{

  return float4(0, 0, 0, 0);
}


void GPUOCLLayer::FrameBuff_SetRenderTarget(int a_id)
{
  if (a_id == 0)
    m_screen.targetFrameBuffPointer = m_screen.color0;
  else
    m_screen.targetFrameBuffPointer = m_screen.color0;
}

cl_mem GPUOCLLayer::getFrameBuffById(int a_id)
{
  if (a_id == 0)
    return m_screen.color0;
  else
    return 0;
}

void GPUOCLLayer::FrameBuff_Blend(int a_buffRes, int a_buff1, int a_buff2, float4 k1, float4 k2)
{
  cl_mem dst0 = getFrameBuffById(a_buffRes);
  cl_mem src1 = getFrameBuffById(a_buff1);
  cl_mem src2 = getFrameBuffById(a_buff2);


  cl_kernel opKern = m_progs.screen.kernel("BlendFrameBuffers");
  size_t    a_size = m_width*m_height;

  size_t localWorkSize = CMP_RESULTS_BLOCK_SIZE;
  int iSize = int(a_size);
  a_size    = roundBlocks(a_size, int(localWorkSize));

  cl_float4 k1_4 = { k1.x, k1.y, k1.z, k1.w };
  cl_float4 k2_4 = { k2.x, k2.y, k2.z, k2.w };

  CHECK_CL(clSetKernelArg(opKern, 0, sizeof(cl_mem),    (void*)&dst0));
  CHECK_CL(clSetKernelArg(opKern, 1, sizeof(cl_mem),    (void*)&src1));
  CHECK_CL(clSetKernelArg(opKern, 2, sizeof(cl_mem),    (void*)&src2));
  CHECK_CL(clSetKernelArg(opKern, 3, sizeof(cl_float4), (void*)&k1_4));
  CHECK_CL(clSetKernelArg(opKern, 4, sizeof(cl_float4), (void*)&k2_4));
  CHECK_CL(clSetKernelArg(opKern, 5, sizeof(cl_int),    (void*)&iSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, opKern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUOCLLayer::CL_MLT_DATA::free()
{
  if(rstateForAcceptReject) clReleaseMemObject(rstateForAcceptReject);
  //if(rstateCurr) clReleaseMemObject(rstateCurr); 
  if(rstateOld) clReleaseMemObject(rstateOld);
  if(rstateNew) clReleaseMemObject(rstateNew);

  if(xVector) clReleaseMemObject(xVector);
  if(yVector) clReleaseMemObject(yVector);

  if(xColor) clReleaseMemObject(xColor);
  if(yColor) clReleaseMemObject(yColor);
  if(zColor) clReleaseMemObject(zColor);

  if(oldXY) clReleaseMemObject(oldXY);
  if(xOldNewId) clReleaseMemObject(xOldNewId);
  if(yOldNewId) clReleaseMemObject(yOldNewId);
  //if(zOldNewId) clReleaseMemObject(zOldNewId);
  if(samplesLum) clReleaseMemObject(samplesLum);

  if(pssVector) clReleaseMemObject(pssVector);

  //if(qmcPositions) clReleaseMemObject(qmcPositions);
  //if(qmcPositions2) clReleaseMemObject(qmcPositions2);
  //if(qmcPosCounter) clReleaseMemObject(qmcPosCounter);

  rstateForAcceptReject = 0;
  rstateCurr = 0;
  rstateOld  = 0;
  rstateNew  = 0;

  xVector = 0;
  yVector = 0;
  zColor  = 0;

  xColor = 0;
  yColor = 0;

  xOldNewId = 0;
  yOldNewId = 0;

  samplesLum  = 0;
  memTaken = 0;

  pssVector = 0;

  //qmcPositions  = 0;
  //qmcPositions2 = 0;
  //qmcPosCounter = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool GPUOCLLayer::MLT_IsAllocated() const
{
  return (m_mlt.rstateForAcceptReject != 0);
}

int GPUOCLLayer::MLT_Alloc(int a_maxBounce)
{
  m_mlt.free();

  cl_int ciErr1 = CL_SUCCESS;

  // alloac additional random number
  //
  m_mlt.rstateForAcceptReject = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.rstateCurr            = m_rays.randGenState; // clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1); // m_rays.randGenState;
  m_mlt.rstateOld             = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.rstateNew             = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  m_mlt.memTaken = 3 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE; // rstateForAcceptReject, rstateOld, rstateNew 

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create rstateForAcceptReject ");

  const int vecSize1 = 8*blocks(a_maxBounce * 8, 8) + 8;

  m_vars.m_varsI[HRT_MLT_MAX_NUMBERS] = vecSize1;
  if (m_vars.m_varsI[HRT_MLT_MAX_NUMBERS] > 256)
    m_vars.m_varsI[HRT_MLT_MAX_NUMBERS] = 256;

  const int QMC_NUMBERS = m_vars.m_varsI[HRT_MLT_MAX_NUMBERS];

  // init big buffers for path space state // (QMC_NUMBERS / MLT_PROPOSALS)
  //
#ifdef MLT_MULTY_PROPOSAL
  m_mlt.xVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, (QMC_NUMBERS / MLT_PROPOSALS)*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
#else
  m_mlt.xVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, QMC_NUMBERS*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
#endif

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create xVector ");

#ifdef MCMC_LAZY
  m_mlt.yVector = 0;
#else
  m_mlt.yVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, QMC_NUMBERS*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create yVector ");
#endif

#ifdef MLT_MULTY_PROPOSAL
  m_mlt.memTaken += (QMC_NUMBERS / MLT_PROPOSALS)*sizeof(float)*m_rays.MEGABLOCKSIZE + 2 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE;
#else
  m_mlt.memTaken += QMC_NUMBERS*sizeof(float)*m_rays.MEGABLOCKSIZE + 2 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE;
#endif

  m_mlt.xColor = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.yColor = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.memTaken += 2*sizeof(float4)*m_rays.MEGABLOCKSIZE;

#ifdef MLT_SORT_RAYS
  m_mlt.zColor = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.memTaken += sizeof(float4)*m_rays.MEGABLOCKSIZE;
#endif

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create yColor ");

#ifndef MCMC_LAZY
  m_mlt.oldXY     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float2)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
#endif
  m_mlt.xOldNewId = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(int2)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.yOldNewId = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(int2)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create xOldNewId/yOldNewId ");

  m_mlt.samplesLum = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create samplesLum, xyVScreen ");

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create qmc data ");

  m_mlt.memTaken += 2 * sizeof(int2)*m_rays.MEGABLOCKSIZE + sizeof(float)*m_rays.MEGABLOCKSIZE;
  m_mlt.memTaken += m_width*m_height*sizeof(float4);

  m_memoryTaken[MEM_TAKEN_MLT] = m_mlt.memTaken;
  size_t MBNumbs = (QMC_NUMBERS*sizeof(float)*m_rays.MEGABLOCKSIZE) / (1024 * 1024);
  size_t MBTaken = m_mlt.memTaken   / (1024 * 1024);
  size_t MBTotal = GetMemoryTaken() / (1024 * 1024);

  std::cout << "[cl_core]: MLT QMC_NUMBERS    = " << QMC_NUMBERS << std::endl;
  std::cout << "[cl_core]: MEGABLOCKSIZE      = " << m_rays.MEGABLOCKSIZE << " threads/rays " << std::endl;
  std::cout << "[cl_core]: MLT(rnd) mem taken = " << MBNumbs << " MB" << std::endl;
  std::cout << "[cl_core]: MLT(all) mem taken = " << MBTaken << " MB" << std::endl;
  std::cout << "[cl_core]: Total    mem taken = " << MBTotal << " MB" << std::endl;

  UpdateVarsOnGPU(); // must do this because we change m_vars.m_varsI[HRT_MLT_MAX_NUMBERS]
  return m_vars.m_varsI[HRT_MLT_MAX_NUMBERS];
}

void GPUOCLLayer::MLT_Free()
{
  m_mlt.free();
}

void GPUOCLLayer::MLT_Init(int a_seed)
{
  int a_seed2 = (a_seed | rand());

  runKernel_InitRandomGen(m_mlt.rstateForAcceptReject, m_rays.MEGABLOCKSIZE, a_seed);
  runKernel_InitRandomGen(m_mlt.rstateCurr, m_rays.MEGABLOCKSIZE, a_seed2);


}

float4 GPUOCLLayer::MLT_Burn(int a_iters)
{

#ifdef MLT_SORT_RAYS
  runKernel_MLTInitIdPairForSorting(m_mlt.xOldNewId, m_mlt.yOldNewId, m_rays.MEGABLOCKSIZE);
#endif

  double avgB[4] = {0.0, 0.0, 0.0, 0.0};

  for (int pass = 0; pass < a_iters; pass++)
  {
    memcpyu32(m_mlt.rstateOld, 0, m_mlt.rstateCurr, 0, m_rays.MEGABLOCKSIZE*sizeof(RandomGen) / sizeof(int)); // oldGen := currGen
    runKernel_MLTMakeProposal(0, m_mlt.xVector, m_mlt.rstateCurr, true, m_rays.MEGABLOCKSIZE);                // latge step

    // eval all F(x)
    //
    m_mlt.pssVector = m_mlt.xVector;
    {
      runKernel_MLTMakeEyeRaysFromPrimeSpaceSample(MUTATE_LAZY_NO, pass*int(m_rays.MEGABLOCKSIZE), m_rays.rayPos, m_rays.rayDir, m_mlt.rstateCurr, m_mlt.pssVector, m_mlt.xColor, 0, m_rays.MEGABLOCKSIZE);
      trace1D(m_rays.rayPos, m_rays.rayDir, m_mlt.xColor, m_rays.MEGABLOCKSIZE);
    }
    m_mlt.pssVector = 0;

    // avegare luminance/contrib (to estimate average brightness of xColor)
    //
    ReduceCLArgs args;
    args.cmdQueue   = m_globals.cmdQueue;
    args.reductionK = m_progs.screen.kernel("ReductionFloat4Avg256");

    float4 brightness(0, 0, 0, 0);
    reduce_average4f_gpu(m_mlt.xColor, m_rays.MEGABLOCKSIZE, &brightness.x, args);

    avgB[0] += double(brightness.x);
    avgB[1] += double(brightness.y);
    avgB[2] += double(brightness.z);
    avgB[3] += double(brightness.w);

    // eval all contribFunc
    //
    runKernel_MLTEvalContribFunc(m_mlt.xColor, m_mlt.samplesLum, m_rays.MEGABLOCKSIZE); // m_mlt.xColor --> m_mlt.samplesLum

    inPlaceScanAnySize1f(m_mlt.samplesLum, m_rays.MEGABLOCKSIZE);                       // calculate parallel pefix summ

    // select K = (N/a_iters) samples proportional to their luminance, append them to xVector
    //
    const int N = int(m_rays.MEGABLOCKSIZE);

#ifdef MLT_MULTY_PROPOSAL
    const int K      = N / (a_iters*MLT_PROPOSALS);
    const int offset = pass*K;
#else
    const int K      = N / a_iters;
    const int offset = pass*K;
#endif

    // newGen[offset .. offset + K] = selectProportionalToContrib(xVector);
    //
    runKernel_MLTSelectSampleProportionalToContrib(offset, pass*N, m_mlt.rstateOld, m_mlt.rstateNew, m_mlt.samplesLum, K);

    if (m_progressBar != nullptr && (pass%4 == 0))
      m_progressBar(L"MCMC Burning-In", float(pass) / float(a_iters));
  }
  
  memcpyu32(m_mlt.rstateOld, 0,  m_mlt.rstateNew, 0, m_rays.MEGABLOCKSIZE*sizeof(RandomGen) / sizeof(int)); // oldGen := initialSelectedGen
  memcpyu32(m_mlt.rstateCurr, 0, m_mlt.rstateNew, 0, m_rays.MEGABLOCKSIZE*sizeof(RandomGen) / sizeof(int)); // curGen := initialSelectedGen
  runKernel_MLTMakeProposal(0, m_mlt.xVector, m_mlt.rstateNew, true, m_rays.MEGABLOCKSIZE);                 // large step

  // eval xColor
  //
  runKernel_MLTMakeEyeRaysFromPrimeSpaceSample(MUTATE_LAZY_NO, 0, m_rays.rayPos, m_rays.rayDir, m_mlt.rstateCurr, m_mlt.xVector, m_mlt.xColor, 0, m_rays.MEGABLOCKSIZE);
  trace1D(m_rays.rayPos, m_rays.rayDir, m_mlt.xColor, m_rays.MEGABLOCKSIZE);

  m_mlt.timer.start();
  m_mlt.mppDone = 0;

  avgB[0] = avgB[0] * (1.0 / double(a_iters));
  avgB[1] = avgB[1] * (1.0 / double(a_iters));
  avgB[2] = avgB[2] * (1.0 / double(a_iters));
  avgB[3] = avgB[3] * (1.0 / double(a_iters));

  return float4(float(avgB[0]), float(avgB[1]), float(avgB[2]), float(avgB[3]));
}


void GPUOCLLayer::MLT_DoPass()
{
  const bool measure = false;

  Timer myTimer(false);
  float timeSort = 0.0f;
  float timeGather = 0.0f;
  float timeTrace = 0.0f;

  BitonicCLArgs sortArgs;
  sortArgs.bitonicPassK = m_progs.sort.kernel("bitonic_pass_kernel");
  sortArgs.bitonic512   = m_progs.sort.kernel("bitonic_512");
  sortArgs.cmdQueue     = m_globals.cmdQueue;

#ifdef MLT_MULTY_PROPOSAL
  const int CHAINS         = int(m_rays.MEGABLOCKSIZE/MLT_PROPOSALS);
  const bool multiProposal = true;
#else
  const int CHAINS         = int(m_rays.MEGABLOCKSIZE);
  const bool multiProposal = false;
#endif

#ifdef MCMC_LAZY
  const int mutateLazy = MUTATE_LAZY_YES;
  memcpyu32(m_mlt.rstateOld, 0, m_mlt.rstateCurr, 0, m_rays.MEGABLOCKSIZE*sizeof(RandomGen) / sizeof(int)); // oldGen := curGen
  m_mlt.pssVector = m_mlt.xVector;
#else
  const int mutateLazy = MUTATE_LAZY_NO;
  runKernel_MLTStoreOldXY(m_mlt.oldXY, m_mlt.xVector, m_rays.MEGABLOCKSIZE);
  runKernel_MLTMakeProposal(m_mlt.xVector, m_mlt.yVector, m_mlt.rstateCurr, false, m_rays.MEGABLOCKSIZE);
  m_mlt.pssVector = m_mlt.yVector;
#endif 

  //runKernel_MLTEvalQMCLargeStepIndex(m_mlt.rstateCurr, m_mlt.qmcPositions, m_mlt.qmcPosCounter, m_rays.MEGABLOCKSIZE);
  runKernel_MLTMakeIdPairForSorting(m_mlt.rstateCurr, m_mlt.xOldNewId, m_mlt.yOldNewId, m_mlt.pssVector, m_mlt.oldXY, m_rays.MEGABLOCKSIZE);

  if (measure)
  {
    clFinish(m_globals.cmdQueue);
    myTimer.start();
  }

  // sort both old and new results (sort ids only)
  //
  bitonic_sort_gpu(m_mlt.xOldNewId, int(CHAINS), sortArgs);
  bitonic_sort_gpu(m_mlt.yOldNewId, int(m_rays.MEGABLOCKSIZE), sortArgs);

  if (measure)
  {
    clFinish(m_globals.cmdQueue);
    timeSort = myTimer.getElapsed()*1000.0f;
    myTimer.start();
  }

  cl_mem qmcPositions = 0;

#ifdef MLT_SORT_RAYS
  runKernel_MLTMoveRandStateByIndex(m_mlt.rstateCurr, m_mlt.rstateOld, 0, 0, m_mlt.yOldNewId,  m_rays.MEGABLOCKSIZE);
#endif

  runKernel_MLTMakeEyeRaysFromPrimeSpaceSample(mutateLazy, 0, m_rays.rayPos, m_rays.rayDir, m_mlt.rstateCurr, m_mlt.pssVector, m_mlt.yColor, 0, m_rays.MEGABLOCKSIZE);

  trace1D(m_rays.rayPos, m_rays.rayDir, m_mlt.yColor, m_rays.MEGABLOCKSIZE);

  m_mlt.pssVector = 0;

#ifdef MLT_SORT_RAYS
  runKernel_MLTMoveColorByIndexBack(m_mlt.zColor, m_mlt.yColor, m_mlt.yOldNewId, m_rays.MEGABLOCKSIZE);
  {
    cl_mem temp  = m_mlt.yColor;
    m_mlt.yColor = m_mlt.zColor;
    m_mlt.zColor = temp;
  }
#endif

  // results are in pair<m_mlt.yVector, m_mlt.yColor>

  if (measure)
  {
    clFinish(m_globals.cmdQueue);
    timeTrace = myTimer.getElapsed()*1000.0f;
    myTimer.start();
  }

  runKernel_MLTContribToScreen(m_mlt.xOldNewId, m_mlt.yOldNewId, m_mlt.xColor, m_mlt.yColor, m_rays.MEGABLOCKSIZE);
  //runKernel_MLTContribToScreenAtomics(m_mlt.xVector, m_mlt.xColor, m_mlt.yColor, m_rays.MEGABLOCKSIZE);

  if (measure)
  {
    clFinish(m_globals.cmdQueue);
    timeGather = myTimer.getElapsed()*1000.0f;
    myTimer.start();
  }

  runKernel_MLTAcceptReject(m_mlt.rstateForAcceptReject, m_mlt.xVector, m_mlt.yVector, m_mlt.xColor, m_mlt.yColor, m_rays.MEGABLOCKSIZE);
  
  if (measure)
  {
    clFinish(m_globals.cmdQueue);
    float timeAcceptReject = myTimer.getElapsed()*1000.0f;

    std::cout << "t_trace  = " << timeTrace  << " ms " << std::endl;
    std::cout << "t_sort   = " << timeSort   << " ms " << std::endl;
    std::cout << "t_gather = " << timeGather << " ms " << std::endl;
    std::cout << "t_accept = " << timeAcceptReject << " ms " << std::endl;
  }

  m_mlt.mppDone += ( double(m_rays.MEGABLOCKSIZE) / double(m_width*m_height) );

  if ((m_mlt.mppDone > 16) && (int(m_mlt.mppDone) % 256 == 0) && (round(m_mlt.mppDone) == m_mlt.mppDone))
    std::cout << "[mlt_timer]: " << m_mlt.mppDone << " spp, time = " << m_mlt.timer.getElapsed() << std::endl;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// auxilary algorithms

void GPUOCLLayer::inPlaceScanAnySize1f(cl_mem a_inBuff, size_t a_size)
{
  if (a_size > m_width*m_height)
    RUN_TIME_ERROR("inPlaceScanAnySize1f : too big input size");

  ScanCLArgs args;
  args.cmdQueue   = m_globals.cmdQueue;
  args.scanBlockK = m_progs.sort.kernel("scan_block_scan1f");
  args.propagateK = m_progs.sort.kernel("scan_propagate1f");

  scan1f_gpu(a_inBuff, a_size, args);
}



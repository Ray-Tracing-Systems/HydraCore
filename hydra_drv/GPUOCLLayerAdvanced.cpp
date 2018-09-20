#include "GPUOCLLayer.h"
#include "crandom.h"

#include "../../HydraAPI/hydra_api/xxhash.h"
#include "../../HydraAPI/hydra_api/ssemath.h"

#include "cl_scan_gpu.h"

extern "C" void initQuasirandomGenerator(unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]);

#include <algorithm>
#undef min
#undef max

void GPUOCLLayer::CopyForConnectEye(cl_mem in_flags,  cl_mem in_raydir,  cl_mem in_color,
                                    cl_mem out_flags, cl_mem out_raydir, cl_mem out_color, size_t a_size)
{
  cl_kernel kernX      = m_progs.lightp.kernel("CopyAndPackForConnectEye");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_flags));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_raydir));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&in_color));

  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&out_flags));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&out_raydir));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&out_color));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::ConnectEyePass(cl_mem in_rayFlags, cl_mem in_rayDirOld, cl_mem in_color, int a_bounce, size_t a_size)
{
  runKernel_EyeShadowRays(in_rayFlags, in_rayDirOld,
                          m_rays.shadowRayPos, m_rays.shadowRayDir, a_size);

  runKernel_ShadowTrace(in_rayFlags, m_rays.shadowRayPos, m_rays.shadowRayDir, a_size,
                        m_rays.lshadow);

  runKernel_ProjectSamplesToScreen(in_rayFlags, m_rays.shadowRayDir, in_rayDirOld, in_color,
                                   m_rays.pathShadeColor, m_rays.samZindex, a_size, a_bounce);

  AddContributionToScreen(m_rays.pathShadeColor, m_rays.samZindex); // because GPU contributio for LT could be very expensieve (imagine point light)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t GPUOCLLayer::MMLTInitSplitDataUniform(int bounceBeg, int a_maxDepth, size_t a_size,
                                             cl_mem a_splitData, cl_mem a_scaleTable, std::vector<int>& activeThreads)
{
  std::vector<int2> splitDataCPU(a_size);
  const size_t blocksNum = splitDataCPU.size() / 256;

  const int bounceEnd = a_maxDepth;
  assert(bounceEnd >= bounceBeg);

  const int bouncesIntoAccount      = bounceEnd - bounceBeg + 1;
  const size_t blocksPerTargetDepth = blocksNum / bouncesIntoAccount;
  const size_t finalThreadsNum      = (blocksPerTargetDepth*bouncesIntoAccount)*256;

  activeThreads.resize(a_maxDepth+1);
  for(int i=0;i<=bounceBeg;i++)
    activeThreads[i] = int(finalThreadsNum);

  size_t currPos = 0;
  for(int bounce = a_maxDepth; bounce >= bounceBeg; bounce--)
  {
    for(int b=0;b<blocksPerTargetDepth;b++)
    {
      for(int i=0;i<256;i++)
        splitDataCPU[(currPos + b)*256 + i] = make_int2(bounce, bounce);
    }
    currPos += blocksPerTargetDepth;

    const int index = activeThreads.size() - bounce + bounceBeg;
    if(index >=0 && index < activeThreads.size())
      activeThreads[index] = int(finalThreadsNum - currPos*256);
  }

  std::vector<float> scale(a_maxDepth+1);
  for(size_t i=bounceBeg;i<scale.size();i++)
  {
    float& selectorInvPdf = scale[i];
    const int d    = i;
    selectorInvPdf = float((d+1)*bouncesIntoAccount);
  }

  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, a_splitData,  CL_TRUE, 0, splitDataCPU.size()*sizeof(int2), (void*)splitDataCPU.data(), 0, NULL, NULL));
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, a_scaleTable, CL_TRUE, 0, scale.size()*sizeof(float),       (void*)scale.data(), 0, NULL, NULL));

  std::cout << std::endl << "[MMLT]: dd info (initial): " << std::endl;
  for(int i=0;i<=a_maxDepth;i++)
    std::cout << "[d = " << i << ",\tN = " << activeThreads[i] << ", coeff = " << scale[i] << "]" << std::endl;
  std::cout << "finalThreadsNum = " << finalThreadsNum << std::endl;
  std::cout << std::endl;

  return finalThreadsNum;
}

std::vector<float> PrefixSumm(const std::vector<float>& a_vec);

float GPUOCLLayer::MMLT_BurningIn(int minBounce, int maxBounce,
                                  cl_mem out_rstate, cl_mem out_dsplit, cl_mem out_split2, cl_mem out_normC, std::vector<int>& out_activeThreads)
{
  //testScanFloatsAnySize();
 
  if(m_mlt.rstateOld == out_rstate || out_dsplit == m_mlt.dOld)
  {
    std::cerr << "MMLT_BurningIn, wrong input buffers! Select (m_mlt.rstateNew, dNew) instead!" << std::endl;
    std::cout << "MMLT_BurningIn, wrong input buffers! Select (m_mlt.rstateNew, dNew) instead!" << std::endl;
    return 1.0f;
  }

  // zero out_normC table because we are going to increment it via simulated floating points atomics ... 
  cl_int ciErr1 = CL_SUCCESS;

  //std::vector<float> scale(maxBounce+1);
  //for(auto& coeff : scale)
  //  coeff = 0.0f;
  //cl_mem avgBTableGPU = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256*sizeof(float), (void*)scale.data(), &ciErr1);
  //
  
  MMLTInitSplitDataUniform(minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
                           m_mlt.splitData, m_mlt.scaleTable, out_activeThreads);

  const int BURN_ITERS   = 64;
  const int BURN_PORTION = m_rays.MEGABLOCKSIZE/BURN_ITERS;

  cl_mem temp_f1 = out_dsplit; // #NOTE: well, that's not ok in general, but due to sizeof(int) == sizeof(float) we can use this buffer temporary
                               // #NOTE: do you allocate enough memory for this buffer? --> seems yes (see inPlaceScanAnySize1f impl).
                               // #NOTE: current search will not work !!! It need size+1 array !!!  
                               // #NOTE: you can just set last element to size-2, not size-1. So it will work in this way.
  float avgBrightness = 0.0f;

  std::cout << std::endl;
  for(int iter=0; iter<BURN_ITERS;iter++)
  {
    runKernel_MMLTMakeProposal(m_mlt.rstateCurr, m_mlt.xVector, 1, maxBounce, m_rays.MEGABLOCKSIZE,
                               m_mlt.rstateCurr, m_mlt.xVector);

    EvalSBDPT(m_mlt.xVector, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
              m_rays.pathAccColor, m_rays.samZindex);

    runKernel_MLTEvalContribFunc(m_rays.pathAccColor, m_mlt.splitData, m_rays.MEGABLOCKSIZE,
                                 temp_f1, nullptr);
    
    {
      ReduceCLArgs args;
      args.cmdQueue   = m_globals.cmdQueue;
      args.reductionK = m_progs.screen.kernel("ReductionFloat4Avg256");
      
      float4 avg(0,0,0,0);
      reduce_average4f_gpu(temp_f1, m_rays.MEGABLOCKSIZE/4, &avg.x, args);

      avgBrightness += (0.25f/float(BURN_ITERS))*(avg.x + avg.y + avg.z + avg.w);
    }

    inPlaceScanAnySize1f(temp_f1, m_rays.MEGABLOCKSIZE);

    // select BURN_PORTION (state,d) from (m_mlt.rstateCurr, m_mlt.splitData) => (m_mlt.rstateOld, dOld)
    // 
    runKernel_MLTSelectSampleProportionalToContrib(m_mlt.rstateCurr, m_mlt.splitData, temp_f1, m_rays.MEGABLOCKSIZE, m_mlt.rstateForAcceptReject, BURN_PORTION,
                                                   BURN_PORTION*iter, m_mlt.rstateOld, m_mlt.dOld);

    // accum normalisation constant per bounce (#TBD!!!)
    //

    clFinish(m_globals.cmdQueue);
    
    if(iter%4 == 0)
    {
      std::cout << "MMLT Burning in, progress = " << 100.0f*float(iter)/float(BURN_ITERS) << "% \r";
      std::cout.flush();
    }
  }
  std::cout << std::endl;

  // sort all selected pairs of (m_mlt.rstateOld, dOld) by d and screen (x,y) => (out_rstate, out_dsplit)
  //
  {
    runKernel_MMLTMakeStatesIndexToSort(m_mlt.rstateOld, m_mlt.dOld, m_rays.MEGABLOCKSIZE,
                                        m_rays.samZindex);

    BitonicCLArgs sortArgs;
    sortArgs.bitonicPassK = m_progs.sort.kernel("bitonic_pass_kernel");
    sortArgs.bitonic512   = m_progs.sort.kernel("bitonic_512");
    sortArgs.bitonic1024  = m_progs.sort.kernel("bitonic_1024");
    sortArgs.bitonic2048  = m_progs.sort.kernel("bitonic_2048");
    sortArgs.cmdQueue     = m_globals.cmdQueue;
    sortArgs.dev          = m_globals.device;
    
    bitonic_sort_gpu(m_rays.samZindex, int(m_rays.MEGABLOCKSIZE), sortArgs);

    runKernel_MMLTMoveStatesByIndex(m_rays.samZindex, m_mlt.rstateOld, m_mlt.dOld, m_rays.MEGABLOCKSIZE,
                                    out_rstate, out_dsplit, out_split2);
  }

  // build threads table 
  //
  std::vector<int> depthCPU(m_rays.MEGABLOCKSIZE);
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, out_dsplit, CL_TRUE, 0, m_rays.MEGABLOCKSIZE * sizeof(int), depthCPU.data(), 0, NULL, NULL));

  std::vector<int> threadsNumCopy = out_activeThreads;
  for(auto& N : threadsNumCopy)
    N = 0;
  
  std::cout << std::endl;
  int old_d = -1;
  for(size_t i=0;i<depthCPU.size();i++)
  {
    const int d = abs(depthCPU[i]);
    threadsNumCopy[d]++;
    if(d!=old_d)
    {
      //std::cout << "[chnge]: d = " << i << ", N = " << d << std::endl;
      old_d = d;
    }
  }

  std::cout << std::endl;
  std::cout << "[BurningIn] dd per_depth:" << std::endl;
  for(int i=0;i<threadsNumCopy.size();i++)
     std::cout << "[d = " << i << ", N = " << threadsNumCopy[i] << ", coeff = 0]" << std::endl;

  for(int i=0;i<=minBounce;i++)
    out_activeThreads[i] = int(m_rays.MEGABLOCKSIZE);

  // now get get active bounce threads number from 'threads per depth'
  //
  size_t summ = 0;
  for(int i=threadsNumCopy.size()-1;i>minBounce;i--)
  {
    summ += threadsNumCopy[i];
    out_activeThreads[i] = summ;
  }

  std::cout << std::endl;
  std::cout << "[BurningIn] dd final:" << std::endl;
  for(int i=0;i<out_activeThreads.size();i++)
     std::cout << "[d = " << i << ", N = " << out_activeThreads[i] << ", coeff = 0]" << std::endl;

  // get average brightness
  //
  //float avgBrightness = 0.0f;
  //{
  //  std::vector<float> avgB(maxBounce+1);
  //  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, avgBTableGPU, CL_TRUE, 0, avgB.size()*sizeof(float), (void*)avgB.data(), 0, NULL, NULL));
  //
  //  const float scaleInv = 1.0f/float(BURN_ITERS*m_rays.MEGABLOCKSIZE);
  //
  //  for(int i=0;i<avgB.size();i++)
  //  {
  //    const float avgBPerBounce = avgB[i]*scaleInv;
  //    std::cout << "[d = " << i << ", avgB = " << avgBPerBounce << ", coeff = " << float(i + 1) << "]" << std::endl;
  //    avgBrightness += avgBPerBounce;
  //  }
  //  std::cout << "[d = a, avgB = " << avgBrightness << "]" << std::endl;
  //}

  std::cout << "[d = a, avgB = " << avgBrightness << "]" << std::endl;
  //clReleaseMemObject(avgBTableGPU); avgBTableGPU = nullptr;

  std::vector<float> scale(maxBounce+1);
  for(int i=0;i<scale.size();i++)
    scale[i] = float(i+1);
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, out_normC, CL_TRUE, 0, scale.size()*sizeof(float), (void*)scale.data(), 0, NULL, NULL));

  return avgBrightness;
}


void GPUOCLLayer::MMLTDebugDrawSelectedSamples(int minBounce, int maxBounce, cl_mem in_rstate, cl_mem in_dsplit, size_t a_size)
{
  runKernel_MMLTCopySelectedDepthToSplit(in_dsplit, a_size,
                                         m_mlt.splitData);
 
  runKernel_MMLTMakeProposal(in_rstate, m_mlt.xVector, 1, maxBounce, m_rays.MEGABLOCKSIZE,
                             in_rstate, m_mlt.xVector);
  
  m_raysWasSorted = false;
  EvalSBDPT(m_mlt.xVector, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
            m_rays.pathAccColor, m_rays.samZindex);

  AddContributionToScreen(m_rays.pathAccColor, m_rays.samZindex);
}

void GPUOCLLayer::EvalSBDPT(cl_mem in_xVector, int minBounce, int maxBounce, size_t a_size,
                            cl_mem a_outColor, cl_mem a_outZIndex)
{
  m_mlt.currVec = in_xVector;
  cl_mem a_rpos = m_rays.rayPos;
  cl_mem a_rdir = m_rays.rayDir;
  
  // (1) init and camera pass 
  //
  runKernel_MMLTMakeEyeRays(a_size,
                            m_rays.rayPos, m_rays.rayDir, m_rays.samZindex);
  
  runKernel_MMLTInitSplitAndCamV(m_rays.rayFlags, a_outColor, m_mlt.splitData, m_mlt.cameraVertexSup, a_size);

  m_mlt.currBounceThreadsNum = a_size;
  for (int bounce = 1; bounce <= maxBounce; bounce++)
  {
    if(bounce >= minBounce)
      m_mlt.currBounceThreadsNum = m_mlt.perBounceActiveThreads[bounce]; 

    runKernel_Trace(a_rpos, a_rdir, m_mlt.currBounceThreadsNum,
                    m_rays.hits);
  
    runKernel_ComputeHit(a_rpos, a_rdir, m_rays.hits, a_size, m_mlt.currBounceThreadsNum, 
                         m_mlt.cameraVertexHit, m_rays.hitProcTexData);
  
    runKernel_MMLTCameraPathBounce(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, m_mlt.splitData, a_size,  //#NOTE: m_mlt.rstateCurr used inside
                                   m_mlt.cameraVertexHit, m_mlt.cameraVertexSup);
  }

  // (2) light pass
  //
  cl_mem lightVertexHit = m_rays.hitSurfaceAll;
  cl_mem lightVertexSup = m_mlt.lightVertexSup;

  runKernel_MMLTLightSampleForward(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, lightVertexSup, a_size);
  
  m_mlt.currBounceThreadsNum = a_size;
  for (int bounce = 1; bounce <= (maxBounce-1); bounce++) // last bounce is always a connect stage
  {
    if(bounce >= minBounce)
      m_mlt.currBounceThreadsNum = m_mlt.perBounceActiveThreads[bounce]; 

    runKernel_Trace(a_rpos, a_rdir, m_mlt.currBounceThreadsNum,
                    m_rays.hits);

    runKernel_ComputeHit(a_rpos, a_rdir, m_rays.hits, a_size, m_mlt.currBounceThreadsNum,
                         lightVertexHit, m_rays.hitProcTexData);

    runKernel_MMLTLightPathBounce(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, m_mlt.splitData, a_size,  //#NOTE: m_mlt.rstateCurr used inside
                                  lightVertexHit, lightVertexSup);
  }
 
  // (3) ConnectEye, ConnectShadow and ConnectEndPoinst
  //
  runkernel_MMLTMakeShadowRay(m_mlt.splitData, m_mlt.cameraVertexHit, m_mlt.cameraVertexSup, lightVertexHit, lightVertexSup, a_size, 
                              m_rays.shadowRayPos, m_rays.shadowRayDir, m_rays.rayFlags);

  runKernel_ShadowTrace(m_rays.rayFlags, m_rays.shadowRayPos, m_rays.shadowRayDir, a_size,
                        m_rays.lshadow);
  
  runKernel_MMLTConnect(m_mlt.splitData, m_mlt.cameraVertexHit, m_mlt.cameraVertexSup, lightVertexHit, lightVertexSup, m_rays.lshadow, a_size, m_rays.MEGABLOCKSIZE, 
                        a_outColor, a_outZIndex);


  m_mlt.currVec = nullptr;
}



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

void GPUOCLLayer::DL_Pass(int a_maxBounce, int a_itersNum)
{
  if(m_pExternalImage == nullptr && m_mlt.colorDLCPU.size() == 0)
    m_mlt.colorDLCPU.resize(m_width*m_height);

  auto oldFlags = m_vars.m_flags;

  m_vars.m_flags &= (~HRT_ENABLE_MMLT);
  m_vars.m_flags |= (HRT_UNIFIED_IMAGE_SAMPLING | HRT_DIRECT_LIGHT_MODE);  // #TODO: add QMC mode if enabled (experimental, only for 1 GPU !!!)
  UpdateVarsOnGPU(m_vars);

  for(int iter = 0; iter < a_itersNum; iter++)
  {
    assert(kmlt.xVectorQMC != nullptr);
    
    m_vars.m_varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] = kmlt.maxBounceQMC;
    m_vars.m_varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES] = kmlt.maxBounceQMC;
    UpdateVarsOnGPU(m_vars);

    if(kmlt.maxBounceQMC != 0) 
    {
      runKernel_MakeEyeRaysQMC(m_rays.MEGABLOCKSIZE, m_passNumberForQMC,
                              m_rays.samZindex, kmlt.xVectorQMC);
    }
    else
    {
      runKernel_MakeEyeSamplesOnly(m_rays.MEGABLOCKSIZE, m_passNumberForQMC,
                                   m_rays.samZindex, kmlt.xVectorQMC);          
    } 
    
    EvalPT(kmlt.xVectorQMC, m_rays.samZindex, 1, a_maxBounce, m_rays.MEGABLOCKSIZE,
           m_rays.pathAccColor);
    
    AddContributionToScreen(m_rays.pathAccColor, m_rays.samZindex, false, 1); 

    if(iter != a_itersNum-1)
      m_passNumberForQMC++;
  }  

  if(m_screen.m_cpuFrameBuffer) ////////////////////////////////////////// #NOTE: strange bug with pointer swap. direct light contributes to indirect!
  {                                
    memsetf4(m_rays.pathAccColor, float4(0, 0, 0, as_float(0xFFFFFFFF)), m_rays.MEGABLOCKSIZE, 0);
    AddContributionToScreen(m_rays.pathAccColor, m_rays.samZindex, false, 1); 
  }

  m_vars.m_varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] = 0;
  m_vars.m_varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES] = 0;
  m_vars.m_flags                              = oldFlags;
  UpdateVarsOnGPU(m_vars);

  m_sppDL += float(a_itersNum*m_rays.MEGABLOCKSIZE)/float(m_width*m_height);
}


void _DebugPrintContribAndIndices(cl_command_queue cmdQueue, const std::string& a_fileName, cl_mem in_contrib1f, size_t a_size1, cl_mem in_indices, size_t a_size2)
{
  std::vector<float> data1(a_size1);

  if(in_contrib1f != nullptr)
  {
    clEnqueueReadBuffer(cmdQueue, in_contrib1f, CL_TRUE, 0, data1.size()*sizeof(float), data1.data(), 0, NULL, NULL);
  
    std::ofstream fout(a_fileName.c_str());
    for(size_t i=0;i<data1.size();i++)
      fout << i << "\t: " << data1[i] << std::endl;
  }

  if(in_indices != nullptr)
  {
    std::vector<int> data2(a_size2);
    clEnqueueReadBuffer(cmdQueue, in_indices, CL_TRUE, 0, data2.size()*sizeof(float), data2.data(), 0, NULL, NULL);
  
    std::ofstream fout2("zz_selected.txt");
    for(size_t i=0;i<data2.size();i++)
    {
      const int index = data2[i];
      fout2 << index << "\t: " << data1[index] << std::endl;
    }
  }
}


float GPUOCLLayer::KMLT_BurningIn(int minBounce, int maxBounce, int BURN_ITERS,
                                  cl_mem out_rstate)
{
  
  if(kmlt.rndState1 == out_rstate || kmlt.rndState3 == out_rstate)
  {
    std::cerr << "KMLT_BurningIn, wrong output buffer! Select (kmlt.rndState2) instead!" << std::endl;
    std::cout << "KMLT_BurningIn, wrong output buffer! Select (kmlt.rndState2) instead!" << std::endl;
    return 1.0f;
  }
  
  // zero out_normC table because we are going to increment it via simulated floating points atomics ... 
  cl_int ciErr1 = CL_SUCCESS;

  const int BURN_PORTION = m_rays.MEGABLOCKSIZE/BURN_ITERS;

  cl_mem temp_f1 = kmlt.yMultAlpha;  // #NOTE: well, that's not ok in general, but due to sizeof(int) < sizeof(float4) we can use this buffer temporary
                                     // #NOTE: do you allocate enough memory for this buffer? --> seems yes (see inPlaceScanAnySize1f impl).
                                     // #NOTE: current search will not work !!! It need size+1 array !!!  
                                     // #NOTE: you can just set last element to size-2, not size-1. So it will work in this way.

  cl_mem temp_i1 = kmlt.xMultOneMinusAlpha;

  double avgBrightness = 0.0;

  double avgTimeMk     = 0.0;
  double avgTimeEv     = 0.0;
  double avgTimeCt     = 0.0;
  double avgTimeSel    = 0.0;
  Timer timer(false);
  
  const bool measureTime = false; 

  cl_mem debugBuffer2i = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(int2)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  cl_mem debugBuffer1f = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  std::cout << std::endl;
  for(int iter=0; iter<BURN_ITERS;iter++)
  { 
    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      timer.start();
    }
    
    runKernel_KMLTMakeProposal(kmlt.rndState1, nullptr, 1, m_rays.MEGABLOCKSIZE,
                               kmlt.rndState3, kmlt.xVector, kmlt.xZindex);        // kmlt.rndState1 => kmlt.xVector; kmlt.rndState1 should not change.
                                                                                   // new random generator state is written to kmlt.rndState3
      
    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      avgTimeMk += timer.getElapsed();
      timer.start();
    }
    
    EvalPT(kmlt.xVector, kmlt.xZindex, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
           kmlt.xColor);

    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      avgTimeEv += timer.getElapsed();
      timer.start();
    }
    
    runKernel_MLTEvalContribIndexedFunc(kmlt.xColor, kmlt.xZindex, 0, m_rays.MEGABLOCKSIZE,
                                        temp_f1);

    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      avgTimeCt += timer.getElapsed();
      timer.start();
    }                                 

    memcpyu32(debugBuffer1f, 0, temp_f1, 0, m_rays.MEGABLOCKSIZE);                                                     // debugBuffer1f[i] := temp_f1[i]
    runKernel_DebugClearInt2WithTID(debugBuffer2i, m_rays.MEGABLOCKSIZE);                                              // debugBuffer2i[i] := int2(i,i);
    //_DebugPrintContribAndIndices(m_globals.cmdQueue, "zz_contrib_old.txt", temp_f1, m_rays.MEGABLOCKSIZE, nullptr, 0); // 
    
    avgBrightness += reduce_avg1f(temp_f1, m_rays.MEGABLOCKSIZE)*(1.0/double(BURN_ITERS));

    inPlaceScanAnySize1f(temp_f1, m_rays.MEGABLOCKSIZE);

    // select BURN_PORTION (rnd state) from (kmlt.rndState1) => out_rstate; use kmlt.rndState3 as aux generator
    // 
    runKernel_MLTSelectSampleProportionalToContrib(kmlt.rndState1, debugBuffer2i, temp_f1, m_rays.MEGABLOCKSIZE, kmlt.rndState3, BURN_PORTION,
                                                   BURN_PORTION*iter, out_rstate, temp_i1); 

    _DebugPrintContribAndIndices(m_globals.cmdQueue, "zz_contrib_new.txt", debugBuffer1f, m_rays.MEGABLOCKSIZE, temp_i1, BURN_PORTION);

    if(measureTime)
    {
      avgTimeSel += timer.getElapsed();
      timer.start();
    }     
    
    if(iter%16 == 0)
    {
      std::cout << "KMLT Burning in, progress = " << 100.0f*float(iter)/float(BURN_ITERS) << "% \r";
      std::cout.flush();
    }

    //AddContributionToScreen(kmlt.xColor, kmlt.xZindex);

    //swap curr and new random generator states
    {
      cl_mem temp    = kmlt.rndState1;
      kmlt.rndState1 = kmlt.rndState3;
      kmlt.rndState3 = temp;
    }
  
    //if(iter >= 1)
    //break; 
  }

  std::cout << std::endl;
  std::cout.flush();

  clReleaseMemObject(debugBuffer2i); debugBuffer2i = nullptr; // #TODO: remove this, it is only for debug purpose
  clReleaseMemObject(debugBuffer1f); debugBuffer1f = nullptr;

  return avgBrightness;
}

void GPUOCLLayer::KMLT_Pass(int a_passNumber, int minBounce, int maxBounce, int BURN_ITERS)
{
  auto oldFlagAndVars = m_vars;  // save all variables and flags

  if(kmlt.xVector == nullptr)
  {
    size_t mltMem     = KMLT_Alloc(maxBounce);
    kmlt.maxBonceKMLT = maxBounce;

    std::cout << "[AllocAll]: MEM(KMLT)   = " << mltMem / size_t(1024*1024) << "\tMB" << std::endl;  
    runKernel_ClearAllInternalTempBuffers(m_rays.MEGABLOCKSIZE);                  waitIfDebug(__FILE__, __LINE__);

    runKernel_InitRandomGen(kmlt.rndState1, m_rays.MEGABLOCKSIZE, 12345*GetTickCount()*rand());
    runKernel_InitRandomGen(kmlt.rndState2, m_rays.MEGABLOCKSIZE, 56789*GetTickCount()*rand());
    runKernel_InitRandomGen(kmlt.rndState3, m_rays.MEGABLOCKSIZE, 37561*GetTickCount()*rand());
  }
  
  // this is essential for KMLT/PT pass to properly work
  //
  m_vars.m_flags |= HRT_INDIRECT_LIGHT_MODE;                       // evaluate indirect light only
  m_vars.m_varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] = kmlt.maxBonceKMLT; // get random numbers from input vector until this bounce; other are pseudo random;
  m_vars.m_varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES] = kmlt.maxBonceKMLT; // get random numbers from input vector until this bounce; other are pseudo random;
  UpdateVarsOnGPU(m_vars);
  
  // select random generator states due to burning in and generate initial state in 'kmlt.xVector'
  //
  //if(m_spp < 1e-5f) // run init stage and burning in
  //{ 
    m_avgBrightness = KMLT_BurningIn(minBounce, maxBounce, BURN_ITERS,
                                     kmlt.rndState2);
    
    // MakeProposal(kmlt.rndState2) => (kmlt.xVector, kmlt.xZindex) 
    //
    runKernel_KMLTMakeProposal(kmlt.rndState2, nullptr, 1, m_rays.MEGABLOCKSIZE,
                               kmlt.rndState2, kmlt.xVector, kmlt.xZindex);
    
    // EvalPT(kmlt.xVector, kmlt.xZindex) => kmlt.xColor
    //
    EvalPT(kmlt.xVector, kmlt.xZindex, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
           kmlt.xColor);

    memsetf4(kmlt.yMultAlpha,         float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); waitIfDebug(__FILE__, __LINE__); // because we can use tham inside KMLT_BurningIn
    memsetf4(kmlt.xMultOneMinusAlpha, float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); waitIfDebug(__FILE__, __LINE__); // because we can use tham inside KMLT_BurningIn
  //}

  AddContributionToScreen(kmlt.xColor, kmlt.xZindex);

  for(int pass = 0; pass < a_passNumber; pass++)
  {
    // MakeProposal(kmlt.rndState1, kmlt.xVector, kmlt.xZindex) => (kmlt.yVector, kmlt.yZindex)  

    // EvalPT(kmlt.yVector, kmlt.yZindex) => kmlt.yColor

    // KMLT_AcceptReject(kmlt.xColor, kmlt.xZindex, kmlt.yColor, kmlt.yZindex, kmlt.rndState2, ... ) => (kmlt.xMultOneMinusAlpha, kmlt.yMultAlpha)

    // if(m_screen.m_cpuFrameBuffer)
    // {
    //   int width, height;
    //   float4* resultPtr = const_cast<float4*>(GetCPUScreenBuffer(0, width, height));
    //  
    //   AddContributionToScreenCPU2(kmlt.yMultAlpha, kmlt.xMultOneMinusAlpha, int(m_rays.MEGABLOCKSIZE), width, height,
    //                               resultPtr);
    // }
  }


  m_vars = oldFlagAndVars;
}

void GPUOCLLayer::MMLT_Pass(int a_passNumber, int minBounce, int maxBounce, int BURN_ITERS)
{

  if(m_rays.pathAuxColor == nullptr || !m_screen.m_cpuFrameBuffer)
  {
    std::cerr << "GPUOCLLayer::MMLT_Pass: Error! Please use CPU frame buffer for MLT" << std::endl;
    exit(0);
  }

  if(!MLT_IsAllocated())
  {
    size_t mltMem = MLT_Alloc(maxBounce + 1); // #TODO: maxBounce works too !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    std::cout << "[AllocAll]: MEM(MLT)    = " << mltMem / size_t(1024*1024) << "\tMB" << std::endl;  
    runKernel_ClearAllInternalTempBuffers(m_rays.MEGABLOCKSIZE);                  waitIfDebug(__FILE__, __LINE__);

    memsetf4(m_rays.pathAuxColor,      float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); waitIfDebug(__FILE__, __LINE__);
    memsetf4(m_mlt.pathAuxColor,       float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); waitIfDebug(__FILE__, __LINE__);
    memsetf4(m_mlt.pathAuxColor2,      float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); waitIfDebug(__FILE__, __LINE__);
    memsetf4(m_mlt.yMultAlpha,         float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); waitIfDebug(__FILE__, __LINE__);
    memsetf4(m_mlt.xMultOneMinusAlpha, float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0); waitIfDebug(__FILE__, __LINE__);
  } 
  
  if(m_spp < 1e-5f) // run init stage and burning in
  {
    m_avgBrightness = MMLT_BurningIn(minBounce, maxBounce, BURN_ITERS,
                                     m_mlt.rstateNew, m_mlt.dNew, m_mlt.splitData, m_mlt.scaleTable, m_mlt.perBounceActiveThreads);

    if(m_pExternalImage != nullptr)
      m_pExternalImage->Header()->avgImageB = m_avgBrightness;

    //#NOTE: force large step = 1 to generate numbers from current state
    runKernel_MMLTMakeProposal(m_mlt.rstateNew, nullptr, 1, maxBounce, m_rays.MEGABLOCKSIZE, 
                               m_mlt.rstateNew, m_mlt.xVector);

    // swap (m_mlt.rstateNew, m_mlt.dNew) and (m_mlt.rstateOld, m_mlt.dOld)
    {
      cl_mem sTmp     = m_mlt.rstateNew; cl_mem dTmp = m_mlt.dNew;
      m_mlt.rstateNew = m_mlt.rstateOld;  m_mlt.dNew = m_mlt.dOld;
      m_mlt.rstateOld = sTmp;             m_mlt.dOld = dTmp;
    }

    EvalSBDPT(m_mlt.xVector, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
              m_mlt.xColor);

    m_mlt.lastBurnIters = BURN_ITERS;
  }

  for(int pass = 0; pass < a_passNumber; pass++)
  {
    // (1) make poposal / gen rands
    //
    const int largeStep = (pass + 1) % 3 == 0 ? 1 : 0;
    runKernel_MMLTMakeProposal(m_mlt.rstateOld, m_mlt.xVector, largeStep, maxBounce, m_rays.MEGABLOCKSIZE,
                               m_mlt.rstateOld, m_mlt.yVector);
    
    // (2) trace; 
    //
    EvalSBDPT(m_mlt.yVector, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
              m_mlt.yColor);
    
    // if(largeStep)
    //   MMLTUpdateAverageBrightnessConstants(minBounce, m_mlt.yColor, m_rays.MEGABLOCKSIZE);

    // (3) Accept/Reject => (xColor, yColor)
    //
    runKernel_AcceptReject(m_mlt.xVector, m_mlt.yVector, m_mlt.xColor, m_mlt.yColor, m_mlt.scaleTable2, m_mlt.splitData,
                           m_mlt.rstateForAcceptReject, maxBounce, m_rays.MEGABLOCKSIZE,
                           m_mlt.xMultOneMinusAlpha, m_mlt.yMultAlpha);
    
    // (4) (xColor, yColor) => ContribToScreen
    //
    if(m_screen.m_cpuFrameBuffer)
    {
      int width, height;
      float4* resultPtr = const_cast<float4*>(GetCPUScreenBuffer(0, width, height));
     
      AddContributionToScreenCPU2(m_mlt.yMultAlpha, m_mlt.xMultOneMinusAlpha, int(m_rays.MEGABLOCKSIZE), width, height,
                                  resultPtr);
    }
    else
    {
      AddContributionToScreen(m_mlt.xMultOneMinusAlpha, nullptr, false);                         
      AddContributionToScreen(m_mlt.yMultAlpha        , nullptr, (pass == a_passNumber-1));
    } 
    
  }

}

std::vector<float> CalcSBPTScaleTable(int bounceBeg, int bounceEnd)
{
  const int bouncesIntoAccount = bounceEnd - bounceBeg + 1;

  std::vector<float> scale(bounceEnd+1);
  for(size_t i=bounceBeg;i<scale.size();i++)
  {
    float& selectorInvPdf = scale[i]; 
    const int d    = i;
    selectorInvPdf = float((d+1)*bouncesIntoAccount);  
    #ifdef SBDPT_DEBUG_SPLIT
    selectorInvPdf = 1.0f;
    #endif
  }
  
  return scale;
}

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

      #ifdef SBDPT_DEBUG_SPLIT
      for(int i=0;i<256;i++)
        splitDataCPU[(currPos + b)*256 + i] = make_int2(SBDPT_DEBUG_DEPTH, SBDPT_DEBUG_DEPTH);
      #endif
    }
    currPos += blocksPerTargetDepth;

    const int index = activeThreads.size() - bounce + bounceBeg;
    if(index >=0 && index < activeThreads.size())
      activeThreads[index] = int(finalThreadsNum - currPos*256);

    #ifdef SBDPT_DEBUG_SPLIT
    activeThreads[index] = int(finalThreadsNum);
    #endif
  }

  //std::vector<int> testThreadsPerBounce(a_maxDepth+1);
  //for(int& x : testThreadsPerBounce) x = 0;
  //if(true)
  //{
  //  for(size_t i=0;i<splitDataCPU.size();i++)
  //    testThreadsPerBounce[splitDataCPU[i].x]++;
  //
  //  for(int i=0;i<=a_maxDepth;i++)
  //    std::cout << "[d = " << i << ",\tN = " << testThreadsPerBounce[i] << "]" << std::endl;
  //}

  std::vector<float> scale = CalcSBPTScaleTable(bounceBeg, a_maxDepth);

  // test some concrete bounce
  //
  #ifdef SBDPT_CHECK_BOUNCE
  {
    int concreteBounce = SBDPT_CHECK_BOUNCE;
    for(size_t i=0;i<splitDataCPU.size();i++)
      splitDataCPU[i] = make_int2(concreteBounce, concreteBounce);

    for(int i=0;i<=bounceBeg;i++)
      activeThreads[i] = int(finalThreadsNum);
  }
  #endif

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

void GPUOCLLayer::SBDPT_Pass(int minBounce, int maxBounce, int ITERS)
{
  if(!MLT_IsAllocated())
  {
    size_t mltMem = MLT_Alloc(maxBounce + 1); // #TODO: maxBounce works too !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    std::cout << "[AllocAll]: MEM(MLT)    = " << mltMem / size_t(1024*1024) << "\tMB" << std::endl;  
    runKernel_ClearAllInternalTempBuffers(m_rays.MEGABLOCKSIZE);
    memsetf4(m_rays.pathAuxColor,      float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
    memsetf4(m_mlt.pathAuxColor,       float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
    memsetf4(m_mlt.pathAuxColor2,      float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
    memsetf4(m_mlt.yMultAlpha,         float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
    memsetf4(m_mlt.xMultOneMinusAlpha, float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
  
    MMLTInitSplitDataUniform(minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
                             m_mlt.splitData, m_mlt.scaleTable, m_mlt.perBounceActiveThreads);
  } 

  for(int iter = 0; iter < ITERS; iter++)
  {
    runKernel_MMLTMakeProposal(m_mlt.rstateCurr, m_mlt.xVector, 1, maxBounce, m_rays.MEGABLOCKSIZE,
                               m_mlt.rstateCurr, m_mlt.xVector);
    
    EvalSBDPT(m_mlt.xVector, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
              m_rays.pathAccColor);
    
    AddContributionToScreen(m_rays.pathAccColor, nullptr, (iter == ITERS-1));
  }
}

double GPUOCLLayer::reduce_avg1f(cl_mem a_buff, size_t a_size)
{
   ReduceCLArgs args;
   args.cmdQueue   = m_globals.cmdQueue;
   args.reductionK = m_progs.screen.kernel("ReductionFloat4Avg256");
   
   double avg[4] = {0,0,0,0};
   reduce_average4f_gpu(a_buff, a_size/4, avg, args); 
   return 0.25*(avg[0] + avg[1] + avg[2] + avg[3]);
}


float GPUOCLLayer::MMLT_BurningIn(int minBounce, int maxBounce, int BURN_ITERS,
                                  cl_mem out_rstate, cl_mem out_dsplit, cl_mem out_split2, cl_mem out_normC, std::vector<int>& out_activeThreads)
{

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
                           out_split2, out_normC, out_activeThreads);

  const int BURN_PORTION = m_rays.MEGABLOCKSIZE/BURN_ITERS;

  cl_mem temp_f1 = out_dsplit; // #NOTE: well, that's not ok in general, but due to sizeof(int) == sizeof(float) we can use this buffer temporary
                               // #NOTE: do you allocate enough memory for this buffer? --> seems yes (see inPlaceScanAnySize1f impl).
                               // #NOTE: current search will not work !!! It need size+1 array !!!  
                               // #NOTE: you can just set last element to size-2, not size-1. So it will work in this way.
  double avgBrightness = 0.0;

  double avgTimeMk     = 0.0;
  double avgTimeEv     = 0.0;
  double avgTimeCt     = 0.0;
  double avgTimeSel    = 0.0;
  Timer timer(false);

  const bool measureTime = false;

  std::cout << std::endl;
  for(int iter=0; iter<BURN_ITERS;iter++)
  {
    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      timer.start();
    }

    runKernel_MMLTMakeProposal(m_mlt.rstateCurr, nullptr, 1, maxBounce, m_rays.MEGABLOCKSIZE,
                               m_mlt.rstateNew,  m_mlt.xVector);

    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      avgTimeMk += timer.getElapsed();
      timer.start();
    }

    EvalSBDPT(m_mlt.xVector, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
              m_rays.pathAccColor);

    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      avgTimeEv += timer.getElapsed();
      timer.start();
    }

    runKernel_MLTEvalContribFunc(m_rays.pathAccColor, 0, m_rays.MEGABLOCKSIZE,
                                 temp_f1);

    if(measureTime)
    {
      clFinish(m_globals.cmdQueue);
      avgTimeCt += timer.getElapsed();
      timer.start();
    }                                 
    
    avgBrightness += reduce_avg1f(temp_f1, m_rays.MEGABLOCKSIZE)*(1.0/double(BURN_ITERS));

    MMLTCheatThirdBounceContrib(m_mlt.splitData, 0.5f, m_rays.MEGABLOCKSIZE, temp_f1); // THIS IS IN UNKNOWN (bounce==3) ISSUE/BUG !!!!

    inPlaceScanAnySize1f(temp_f1, m_rays.MEGABLOCKSIZE);

    // select BURN_PORTION (state,d) from (m_mlt.rstateCurr, m_mlt.splitData) => (m_mlt.rstateOld, dOld)
    // 
    runKernel_MLTSelectSampleProportionalToContrib(m_mlt.rstateCurr, m_mlt.splitData, temp_f1, m_rays.MEGABLOCKSIZE, m_mlt.rstateForAcceptReject, BURN_PORTION,
                                                   BURN_PORTION*iter, m_mlt.rstateOld, m_mlt.dOld);
 
    {
      cl_mem temp      = m_mlt.rstateCurr;
      m_mlt.rstateCurr = m_mlt.rstateNew;
      m_mlt.rstateNew  = temp;
    }

    clFinish(m_globals.cmdQueue);

    if(measureTime)
    {
      avgTimeSel += timer.getElapsed();
      timer.start();
    }     
    
    if(iter%16 == 0)
    {
      std::cout << "MMLT Burning in, progress = " << 100.0f*float(iter)/float(BURN_ITERS) << "% \r";
      std::cout.flush();
    }

    //AddContributionToScreen(m_rays.pathAccColor, nullptr);
  }
  std::cout << std::endl;
  
  if(measureTime)
  {
    std::cout.flush();
    std::cout << "[MakeProposal]: avgTime = " <<  avgTimeMk *1000.0/double(BURN_ITERS) << "\t ms" << std::endl;
    std::cout << "[EvalSBDPT   ]: avgTime = " <<  avgTimeEv *1000.0/double(BURN_ITERS) << "\t ms" << std::endl;
    std::cout << "[ContribFunc ]: avgTime = " <<  avgTimeCt *1000.0/double(BURN_ITERS) << "\t ms" << std::endl;
    std::cout << "[SelectSampls]: avgTime = " <<  avgTimeSel*1000.0/double(BURN_ITERS) << "\t ms" << std::endl;
    std::cout.flush();
  }

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
     std::cout << "[d = " << i << ", N = " << threadsNumCopy[i] << ", coeff = 1]" << std::endl;
  
  // init per bounce arrays for future normalisation constants rectification
  //
  {
    m_mlt.perBounceThreads = threadsNumCopy;
    m_mlt.avgBrightnessCPU.resize(threadsNumCopy.size());
    for(auto& val : m_mlt.avgBrightnessCPU)
      val = 0;
    m_mlt.avgBrightnessSamples = 0;
  }
  // \\

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
     std::cout << "[d = " << i << ", N = " << out_activeThreads[i] << ", coeff = 1]" << std::endl;

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
  {
    //scale[i] = 1.0f;
    scale[i] = float(i+1)*( float(m_rays.MEGABLOCKSIZE) / fmax(float(threadsNumCopy[i]), 2.0f) );
  }
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, out_normC, CL_TRUE, 0, scale.size()*sizeof(float), (void*)scale.data(), 0, NULL, NULL));
  
  // init m_mlt.scaleTable2
  // 
  scale.resize(256);
  for(auto& x : scale)
    x = 1.0f;
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_mlt.scaleTable2, CL_TRUE, 0, scale.size()*sizeof(float), (void*)scale.data(), 0, NULL, NULL));

  return float(avgBrightness);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUOCLLayer::MMLTUpdateAverageBrightnessConstants(int minBounce, cl_mem in_color, size_t a_size)
{
  std::vector<int>  & perBounceThreads = m_mlt.perBounceThreads;
  cl_mem temp_f1                       = m_mlt.dNew;

  // std::cout << std::endl;
  // for(size_t i = 0; i < m_mlt.perBounceThreads.size(); i++)
  //   std::cout << "N(threads) = " << perBounceThreads[i] << ",\t" << "coeff = " << perBounceCoeff[i] << std::endl;

  const double alpha = 1.0/double(m_mlt.avgBrightnessSamples + 1);

  int currOffset = 0;
  for(int bounce = minBounce; bounce < perBounceThreads.size(); bounce++)
  {
    int currSize = perBounceThreads[bounce];

    // (1) Evaluate contrib func from in_color to temp_buffer1f
    //
    runKernel_MLTEvalContribFunc(in_color, currOffset, currSize,
                                 temp_f1);

    // (2) Reduce contrib func
    //
    const double avgBrightness     = reduce_avg1f(temp_f1, currSize);
    m_mlt.avgBrightnessCPU[bounce] = avgBrightness*alpha + (1.0 - alpha)*m_mlt.avgBrightnessCPU[bounce];
    currOffset += currSize;
  }
  
  if(m_mlt.avgBrightnessSamples % 128 == 0 && m_mlt.avgBrightnessSamples > 0)
  {
    double summ = 0.0;
    int iters = 0;
    //std::cout << std::endl;
    for(int bounce = minBounce; bounce < m_mlt.avgBrightnessCPU.size(); bounce++)
    {
      summ += double(m_mlt.avgBrightnessCPU[bounce]);
      iters++;
      //std::cout << "avgB(" << bounce << ") = " << m_mlt.avgBrightnessCPU[bounce] << "\tthreads = " << perBounceThreads[bounce] << std::endl;
    }
    //std::cout << std::endl;

    if(1.8f*m_mlt.avgBrightnessCPU[3] < m_mlt.avgBrightnessCPU[4]) // strange 3 bounce bug.
    {
      std::cout << std::endl;
      std::cout << "3 bounce bug happened!" << std::endl;
      m_mlt.avgBrightnessCPU[3] *= 2.0f;
    }

    std::vector<float> coeffs(m_mlt.avgBrightnessCPU.size());

    if(m_mlt.avgBrightnessSamples > m_mlt.lastBurnIters)
    {
      std::cout << std::endl;
      for(int bounce = minBounce; bounce < m_mlt.avgBrightnessCPU.size(); bounce++)
      {
        coeffs[bounce] = float(iters)*m_mlt.avgBrightnessCPU[bounce] / float(summ);
        std::cout << "coef(" << bounce << ") = " << coeffs[bounce] << "\tthreads = " << perBounceThreads[bounce] << std::endl;
      }
     
      CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_mlt.scaleTable2, CL_TRUE, 0, coeffs.size()*sizeof(float), (void*)coeffs.data(), 0, NULL, NULL));
      std::cout << "coeff has updated" << std::endl;
    }
  }

  m_mlt.avgBrightnessSamples++; 
}

void GPUOCLLayer::EvalSBDPT(cl_mem in_xVector, int minBounce, int maxBounce, size_t a_size,
                            cl_mem a_outColor)
{
  m_mlt.currVec = in_xVector;
  cl_mem a_rpos = m_rays.rayPos;
  cl_mem a_rdir = m_rays.rayDir;
  cl_mem a_zind = m_rays.samZindex;
  
  // (1) init and camera pass 
  //
  runKernel_MMLTMakeEyeRays(a_size,
                            m_rays.rayPos, m_rays.rayDir, a_zind);
  
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
                        a_outColor, a_zind);


  m_mlt.currVec = nullptr;
}



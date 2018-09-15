/**
\file
\brief Metropolis algorithm implementation in OpenCL

*/

#include "GPUOCLLayer.h"
#include "crandom.h"

#include "../../HydraAPI/hydra_api/xxhash.h"
#include "cl_scan_gpu.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUOCLLayer::CL_MLT_DATA::free()
{
  if (rstateForAcceptReject) { clReleaseMemObject(rstateForAcceptReject); rstateForAcceptReject = 0; }
  if (rstateOld)             { clReleaseMemObject(rstateOld);             rstateOld             = 0; }
  if (rstateNew)             { clReleaseMemObject(rstateNew);             rstateNew             = 0; }
                                                                         
  if (xVector)               { clReleaseMemObject(xVector); xVector = 0; }
  if (yVector)               { clReleaseMemObject(yVector); yVector = 0; }
                             
  if (xColor)                { clReleaseMemObject(xColor); xColor = 0; }
  if (yColor)                { clReleaseMemObject(yColor); yColor = 0; }
 
  if (lightVertexSup)        { clReleaseMemObject(lightVertexSup);    lightVertexSup = 0; }
  if (cameraVertexSup)       { clReleaseMemObject(cameraVertexSup);   cameraVertexSup = 0; }
  if (cameraVertexHit)       { clReleaseMemObject(cameraVertexHit);   cameraVertexHit = 0; }
  if (pdfArray)              { clReleaseMemObject(pdfArray);          pdfArray        = 0; }
  if (splitData)             { clReleaseMemObject(splitData);         splitData       = 0; }

  rstateCurr = 0;
  memTaken   = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool GPUOCLLayer::MLT_IsAllocated() const
{
  return (m_mlt.rstateForAcceptReject != 0);
}

size_t GPUOCLLayer::MLT_Alloc(int a_maxBounce)
{
  m_mlt.free();

  cl_int ciErr1 = CL_SUCCESS;

  // alloc additional random number
  //
  m_mlt.rstateForAcceptReject = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.rstateCurr            = m_rays.randGenState; // clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1); // m_rays.randGenState;
  m_mlt.rstateOld             = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.rstateNew             = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  m_mlt.memTaken = 3 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE; // rstateForAcceptReject, rstateOld, rstateNew 

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create rstateForAcceptReject ");

  const int MLT_RAND_NUMBERS_PER_BOUNCE = MMLT_HEAD_TOTAL_SIZE + 6*a_maxBounce; //randArraySizeOfDepthMMLT(a_maxBounce);
  m_vars.m_varsI[HRT_MLT_MAX_NUMBERS]   = MLT_RAND_NUMBERS_PER_BOUNCE;

  // init big buffers for path space state // (MLT_RAND_NUMBERS_PER_BOUNCE / MLT_PROPOSALS)
  //
  m_mlt.xVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, MLT_RAND_NUMBERS_PER_BOUNCE*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create xVector ");

#ifdef MCMC_LAZY
  m_mlt.yVector = 0;
#else
  m_mlt.yVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, MLT_RAND_NUMBERS_PER_BOUNCE*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core.MLT_Alloc]: Failed to alloc yVector ");
#endif

  m_mlt.memTaken += MLT_RAND_NUMBERS_PER_BOUNCE*sizeof(float)*m_rays.MEGABLOCKSIZE + 2 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE;

  m_mlt.xColor = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.yColor = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.memTaken += 2*sizeof(float4)*m_rays.MEGABLOCKSIZE;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core.MLT_Alloc]: Failed to alloc yColor ");

  const size_t pathVertexSizeHit = SURFACE_HIT_SIZE_IN_F4*sizeof(float4)*m_rays.MEGABLOCKSIZE;
  const size_t pathVertexSizeSup = PATH_VERTEX_SUPPLEMENT_SIZE_IN_F4*sizeof(float4)*m_rays.MEGABLOCKSIZE;

  m_mlt.lightVertexSup    = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, pathVertexSizeSup, NULL, &ciErr1);
  m_mlt.cameraVertexSup   = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, pathVertexSizeSup, NULL, &ciErr1);
  m_mlt.cameraVertexHit   = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, pathVertexSizeHit, NULL, &ciErr1);
  m_mlt.pdfArray          = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 2*sizeof(float)*m_rays.MEGABLOCKSIZE*(a_maxBounce+1), NULL, &ciErr1);

  m_mlt.memTaken      += pathVertexSizeHit;
  m_mlt.memTaken      += pathVertexSizeSup;
  m_mlt.memTaken      += 2*sizeof(float) *m_rays.MEGABLOCKSIZE*a_maxBounce;
  
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core.MLT_Alloc]: Failed to alloc vertex storage and pdf array ");

  m_mlt.splitData = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 2*sizeof(int)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core.MLT_Alloc]: Failed to alloc splitData ");

  return m_mlt.memTaken;
}

void GPUOCLLayer::MLT_Free()
{
  m_mlt.free();
}

void GPUOCLLayer::MLT_Init(int a_seed)
{
  int a_seed2 = (a_seed | rand());

  runKernel_InitRandomGen(m_mlt.rstateForAcceptReject, m_rays.MEGABLOCKSIZE, a_seed);
  runKernel_InitRandomGen(m_mlt.rstateCurr,            m_rays.MEGABLOCKSIZE, a_seed2);

}

float4 GPUOCLLayer::MLT_Burn(int a_iters)
{
  double avgB[4] = {0.0, 0.0, 0.0, 0.0};

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
  sortArgs.bitonic1024  = m_progs.sort.kernel("bitonic_1024");
  sortArgs.bitonic2048  = m_progs.sort.kernel("bitonic_2048");
  sortArgs.cmdQueue     = m_globals.cmdQueue;
  sortArgs.dev          = m_globals.device;

  m_mlt.mppDone += ( double(m_rays.MEGABLOCKSIZE) / double(m_width*m_height) );

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

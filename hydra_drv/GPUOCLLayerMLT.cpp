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
  if(rstateNew)              { clReleaseMemObject(rstateNew);             rstateNew             = 0; }
                                                                         
  if(xVector)                { clReleaseMemObject(xVector); xVector = 0; }
  if(yVector)                { clReleaseMemObject(yVector); yVector = 0; }
                             
  if (xColor)                { clReleaseMemObject(xColor); xColor = 0; }
  if (yColor)                { clReleaseMemObject(yColor); yColor = 0; }

  if (cameraVertex)          { clReleaseMemObject(cameraVertex);   cameraVertex = 0; }
  if (pdfArray)              { clReleaseMemObject(pdfArray);       pdfArray     = 0; }

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

  const int vecSize1 = randArraySizeOfDepthMMLT(a_maxBounce);

  m_vars.m_varsI[HRT_MLT_MAX_NUMBERS] = vecSize1;
  if (m_vars.m_varsI[HRT_MLT_MAX_NUMBERS] > 256)
    m_vars.m_varsI[HRT_MLT_MAX_NUMBERS] = 256;

  const int QMC_NUMBERS = m_vars.m_varsI[HRT_MLT_MAX_NUMBERS];

  // init big buffers for path space state // (QMC_NUMBERS / MLT_PROPOSALS)
  //
  m_mlt.xVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, QMC_NUMBERS*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create xVector ");

#ifdef MCMC_LAZY
  m_mlt.yVector = 0;
#else
  m_mlt.yVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, QMC_NUMBERS*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create yVector ");
#endif

  m_mlt.memTaken += QMC_NUMBERS*sizeof(float)*m_rays.MEGABLOCKSIZE + 2 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE;

  m_mlt.xColor = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.yColor = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.memTaken += 2*sizeof(float4)*m_rays.MEGABLOCKSIZE;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create yColor ");

  const size_t pathVertexSize = PATH_VERTEX_SIZE_IN_F4*sizeof(float4)*m_rays.MEGABLOCKSIZE;

  m_mlt.cameraVertex   = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, pathVertexSize, NULL, &ciErr1);
  m_mlt.pdfArray       = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 2*sizeof(float)*m_rays.MEGABLOCKSIZE*a_maxBounce, NULL, &ciErr1);
  m_mlt.memTaken      += pathVertexSize;
  m_mlt.memTaken      += 2*sizeof(float) *m_rays.MEGABLOCKSIZE*a_maxBounce;
  
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create yColor ");

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


void TestPathVertexReadWrite()
{
  PathVertex origin, copy;
  float4 data[PATH_VERTEX_SIZE_IN_F4];

  origin.hit.pos        = float3(1,2,3);
  origin.hit.normal     = float3(4,5,6);
  origin.hit.texCoord   = float2(7,8);
  origin.hit.flatNormal = float3(0,1,0);
  origin.hit.tangent    = float3(1,0,0);
  origin.hit.biTangent  = float3(0,0,1);
  origin.hit.matId      = 777;
  origin.hit.hfi        = false;
  origin.hit.t          = 500.0f;
  origin.hit.sRayOff    = 0.001f;

  origin.ray_dir        = float3(9,10,11);
  origin.accColor       = float3(12,13,14);
  origin.lastGTerm      = 15.0f;
  origin.valid          = true;
  origin.wasSpecOnly    = false;
  
  WritePathVertex(&origin, 0, 1, 
                  data);

  ReadPathVertex(data, 0, 1, 
                 &copy);

  int a = 2;

}

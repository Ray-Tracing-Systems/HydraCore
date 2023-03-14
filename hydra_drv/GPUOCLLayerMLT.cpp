/**
\file
\brief Metropolis algorithm implementation in OpenCL

*/

#include "GPUOCLLayer.h"
#include "crandom.h"

#include "hydra_api/xxhash.h"
#include "cl_scan_gpu.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUOCLLayer::CL_MLT_DATA::free()
{
  if (rstateForAcceptReject) { clReleaseMemObject(rstateForAcceptReject); rstateForAcceptReject = 0; }
  //if (rstateCurr)            { clReleaseMemObject(rstateCurr);            rstateCurr            = 0; }
  if (rstateOld)             { clReleaseMemObject(rstateOld);             rstateOld             = 0; }
  if (rstateNew)             { clReleaseMemObject(rstateNew);             rstateNew             = 0; }
  if (dNew)                  { clReleaseMemObject(dNew);                  dNew                  = 0; }
  if (dOld)                  { clReleaseMemObject(dOld);                  dOld                  = 0; }

  if (xVector)               { clReleaseMemObject(xVector);    xVector    = 0; }
  if (yVector)               { clReleaseMemObject(yVector);    yVector    = 0; }
                             
  if (xColor)                { clReleaseMemObject(xColor); xColor = 0; }
  if (yColor)                { clReleaseMemObject(yColor); yColor = 0; }
 
  if (lightVertexSup)        { clReleaseMemObject(lightVertexSup);    lightVertexSup  = 0; }
  if (cameraVertexSup)       { clReleaseMemObject(cameraVertexSup);   cameraVertexSup = 0; }
  if (cameraVertexHit)       { clReleaseMemObject(cameraVertexHit);   cameraVertexHit = 0; }
  if (pdfArray)              { clReleaseMemObject(pdfArray);          pdfArray        = 0; }

  if (pathAuxColor)          { clReleaseMemObject(pathAuxColor);      pathAuxColor     = 0;}
  if (pathAuxColorCPU)       { clReleaseMemObject(pathAuxColorCPU);   pathAuxColorCPU  = 0;}
  if (pathAuxColor2)         { clReleaseMemObject(pathAuxColor2);     pathAuxColor2    = 0;}
  if (pathAuxColorCPU2)      { clReleaseMemObject(pathAuxColorCPU2);  pathAuxColorCPU2 = 0;}

  if (yMultAlpha)            { clReleaseMemObject(yMultAlpha);        yMultAlpha = 0;}
  if (xMultOneMinusAlpha)    { clReleaseMemObject(xMultOneMinusAlpha);xMultOneMinusAlpha = 0;}

  if (splitData)             { clReleaseMemObject(splitData);         splitData       = 0; }
  if (scaleTable)            { clReleaseMemObject(scaleTable);        scaleTable      = 0; }
  if (scaleTable2)           { clReleaseMemObject(scaleTable2);       scaleTable2     = 0; }

  rstateCurr = 0;
  memTaken   = 0;
  currVec    = 0;
}

void GPUOCLLayer::CL_KMLT_DATA::free()
{
  if (xVectorQMC) { clReleaseMemObject(xVectorQMC); xVectorQMC = 0; }

  if(xVector) { clReleaseMemObject(xVector); xVector = nullptr; }
  if(yVector) { clReleaseMemObject(yVector); yVector = nullptr; }
  if(xZindex) { clReleaseMemObject(xZindex); xZindex = nullptr; }
  if(yZindex) { clReleaseMemObject(yZindex); yZindex = nullptr; }
  if(xColor ) { clReleaseMemObject(xColor ); xColor  = nullptr; }
  if(yColor ) { clReleaseMemObject(yColor ); yColor  = nullptr; }

  if(rndState1) { clReleaseMemObject(rndState1 ); rndState1 = nullptr; }
  if(rndState2) { clReleaseMemObject(rndState2 ); rndState2 = nullptr; }
  //if(rndState3) { clReleaseMemObject(rndState3 ); rndState3 = nullptr; } // DO NOT FREE IT, this is same buffer as m_rays.randGenState!

  if(xMultOneMinusAlpha) { clReleaseMemObject(xMultOneMinusAlpha); xMultOneMinusAlpha = nullptr; } 
  if(yMultAlpha        ) { clReleaseMemObject(yMultAlpha);         yMultAlpha         = nullptr; }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool GPUOCLLayer::MLT_IsAllocated() const
{
  return (m_mlt.rstateForAcceptReject != 0);
}

size_t GPUOCLLayer::MLT_Alloc_For_PT_QMC(int a_maxBounce, cl_mem& a_vecQmc)
{
  if(a_vecQmc != nullptr)
  {
    clReleaseMemObject(a_vecQmc); 
    a_vecQmc = nullptr;
  }
  
  cl_int ciErr1 = CL_SUCCESS;

  kmlt.maxBounceQMC = a_maxBounce;

  const int MLT_RAND_NUMBERS_PER_BOUNCE = KMLT_HEAD_SIZE + \
                                          KMLT_PER_LIGHT*a_maxBounce + \
                                          KMLT_PER_MATERIAL*a_maxBounce;

  // init big buffers for path space state // (MLT_RAND_NUMBERS_PER_BOUNCE / MLT_PROPOSALS)
  //
  a_vecQmc = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, MLT_RAND_NUMBERS_PER_BOUNCE*sizeof(float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create MLT_Alloc_For_PT_QMC(a_vecQmc) vector");

  return (MLT_RAND_NUMBERS_PER_BOUNCE*sizeof(float))*m_rays.MEGABLOCKSIZE;
}

size_t GPUOCLLayer::KMLT_Alloc(int a_maxBounce)
{
  const int MLT_RAND_NUMBERS_PER_BOUNCE = KMLT_HEAD_SIZE + \
                                          KMLT_PER_LIGHT*a_maxBounce + \
                                          KMLT_PER_MATERIAL*a_maxBounce;
  
  const size_t randsBuffSize = (MLT_RAND_NUMBERS_PER_BOUNCE*sizeof(float))*m_rays.MEGABLOCKSIZE;

  cl_int ciErr1 = CL_SUCCESS;

  kmlt.xVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, randsBuffSize, NULL, &ciErr1);
  kmlt.yVector = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, randsBuffSize, NULL, &ciErr1);
  
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create kmlt.xVector/kmlt.yVector ");

  kmlt.xZindex = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(int2)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  kmlt.yZindex = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(int2)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create kmlt.xZindex/kmlt.yZindex ");

  kmlt.xColor  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  kmlt.yColor  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create kmlt.xColor/kmlt.yColor ");

  kmlt.rndState1 = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  kmlt.rndState2 = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  kmlt.rndState3 = m_rays.randGenState;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create kmlt.rndState1/kmlt.rndState2 ");

  kmlt.xMultOneMinusAlpha = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  kmlt.yMultAlpha         = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create kmlt.xMultOneMinusAlpha/kmlt.yMultAlpha ");
  
  if(!scan_alloc_internal(m_rays.MEGABLOCKSIZE, m_globals.ctx))
    RUN_TIME_ERROR("Error in scan_alloc_internal");

  return 2*(randsBuffSize + m_rays.MEGABLOCKSIZE*( sizeof(int2) + 2*sizeof(float4) + sizeof(RandomGen) ) );
}

size_t GPUOCLLayer::MLT_Alloc(int a_maxBounce)
{
  m_mlt.free();

  cl_int ciErr1 = CL_SUCCESS;

  // alloc additional random number
  //
  m_mlt.rstateForAcceptReject = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.rstateCurr            = m_rays.randGenState; //clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1); 
  m_mlt.rstateOld             = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.rstateNew             = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.dOld                  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(int)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.dNew                  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 1 * sizeof(int)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.memTaken              = (3 * sizeof(RandomGen) + sizeof(int)*2)*m_rays.MEGABLOCKSIZE; // rstateForAcceptReject, rstateOld, rstateNew 

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create rstateForAcceptReject ");

  srand(GetTickCount()); // #TODO: use some more precise system timer !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  runKernel_InitRandomGen(m_mlt.rstateForAcceptReject, m_rays.MEGABLOCKSIZE, rand()*GetTickCount());
  runKernel_InitRandomGen(m_mlt.rstateCurr,            m_rays.MEGABLOCKSIZE, rand()*GetTickCount());
  runKernel_InitRandomGen(m_mlt.rstateOld,             m_rays.MEGABLOCKSIZE, rand()*GetTickCount());
  runKernel_InitRandomGen(m_mlt.rstateNew,             m_rays.MEGABLOCKSIZE, rand()*GetTickCount());

  const int MLT_RAND_NUMBERS_PER_BOUNCE = MMLT_HEAD_TOTAL_SIZE + MMLT_COMPRESSED_F_PERB*a_maxBounce; //randArraySizeOfDepthMMLT(a_maxBounce);
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
  m_mlt.memTaken += 2*sizeof(int)*m_rays.MEGABLOCKSIZE; 
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core.MLT_Alloc]: Failed to alloc splitData ");
  
  // init coeffs table
  //
  std::vector<float> scale(256);
  for(auto& coeff : scale)
    coeff = 1.0f;

  m_mlt.scaleTable  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256*sizeof(float), (void*)scale.data(), &ciErr1);
  m_mlt.scaleTable2 = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256*sizeof(float), (void*)scale.data(), &ciErr1);    
  if (ciErr1 != CL_SUCCESS) 
    RUN_TIME_ERROR("Error in clCreateBuffer");

  if(!scan_alloc_internal(m_rays.MEGABLOCKSIZE, m_globals.ctx))
    RUN_TIME_ERROR("Error in scan_alloc_internal");

  m_mlt.pathAuxColor     = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE,                         4 * sizeof(cl_float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.pathAuxColorCPU  = clCreateBuffer(m_globals.ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 4 * sizeof(cl_float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);  
  m_mlt.pathAuxColor2    = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE,                         4 * sizeof(cl_float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.pathAuxColorCPU2 = clCreateBuffer(m_globals.ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 4 * sizeof(cl_float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);  
  if (ciErr1 != CL_SUCCESS) 
    RUN_TIME_ERROR("Error in clCreateBuffer");
  m_mlt.memTaken += 8*sizeof(float)*m_rays.MEGABLOCKSIZE; 

  m_mlt.yMultAlpha         = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  m_mlt.xMultOneMinusAlpha = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  if (ciErr1 != CL_SUCCESS) 
    RUN_TIME_ERROR("Error in clCreateBuffer");
  m_mlt.memTaken += 4*sizeof(float)*m_rays.MEGABLOCKSIZE; 

  return m_mlt.memTaken;
}

void GPUOCLLayer::MLT_Free()
{
  m_mlt.colorDLCPU = std::vector<float4, aligned16<float4> >();
  scan_free_internal();
  m_mlt.free();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// auxilary algorithms

void GPUOCLLayer::inPlaceScanAnySize1f(cl_mem a_inBuff, size_t a_size)
{
  if (a_size > scan_get_size())
    RUN_TIME_ERROR("inPlaceScanAnySize1f : too big input size");

  ScanCLArgs args;
  args.cmdQueue   = m_globals.cmdQueue;
  args.scanBlockK = m_progs.sort.kernel("scan_block_scan1f");
  args.propagateK = m_progs.sort.kernel("scan_propagate1f");

  scan1f_gpu(a_inBuff, a_size, args);
}

void GPUOCLLayer::runKernel_MLTEvalContribFunc(cl_mem in_buff, size_t a_offset, size_t a_size,
                                               cl_mem out_buff)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTEvalContribFunc");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  int            ioffs = int(a_offset);
  a_size               = roundBlocks(a_size, int(localWorkSize));
  
  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_buff));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&out_buff));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&ioffs));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MLTEvalContribIndexedFunc(cl_mem in_buff, cl_mem in_indexBuff, size_t a_offset, size_t a_size,
                                                      cl_mem out_buff)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTEvalContribIndexedFunc");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  int            ioffs = int(a_offset);
  a_size               = roundBlocks(a_size, int(localWorkSize));
  
  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_buff));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_indexBuff));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&out_buff));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&ioffs));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::MMLTCheatThirdBounceContrib(cl_mem in_split, float a_multValue, size_t a_size,
                                              cl_mem a_contrib1f)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTCheatThirdBounceContrib");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));
  
  cl_float multVal     = cl_float(a_multValue);

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem),   (void*)&in_split));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem),   (void*)&a_contrib1f));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_float), (void*)&multVal));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int),   (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_UpdateZIndexFromColorW(cl_mem in_color, size_t a_size,
                                                   cl_mem out_zind)
{
  cl_kernel kernX      = m_progs.mlt.kernel("UpdateZIndexFromColorW");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_color));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&out_zind));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTCopySelectedDepthToSplit(cl_mem in_buff, size_t a_size,
                                                         cl_mem out_buff)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTCopySelectedDepthToSplit");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_buff));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&out_buff));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTMakeStatesIndexToSort(cl_mem in_gens, cl_mem in_depth, size_t a_size,
                                                      cl_mem out_index)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTMakeStatesIndexToSort");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_gens));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_depth));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&out_index));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTMoveStatesByIndex(cl_mem in_index, cl_mem in_gens, cl_mem in_depth, size_t a_size,
                                                  cl_mem out_gen, cl_mem out_depth, cl_mem out_split)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTMoveStatesByIndex");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_index));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_gens));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&in_depth));

  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&out_gen));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&out_depth));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&out_split));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MLTSelectSampleProportionalToContrib(cl_mem in_rndState, cl_mem in_split, cl_mem in_array, int a_arraySize, cl_mem gen_select, size_t a_size,
                                                                 cl_int offset, cl_mem out_rndState, cl_mem out_split)

{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTSelectSampleProportionalToContrib");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&out_rndState));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&out_split));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&offset));

  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&in_rndState));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&in_split));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&gen_select));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&in_array));

  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_int), (void*)&a_arraySize));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_AcceptReject(cl_mem a_xVector, cl_mem a_yVector, cl_mem a_xColor,  cl_mem a_yColor, 
                                         cl_mem a_scaleTable, cl_mem a_split,
                                         cl_mem a_rstateForAcceptReject, int a_maxBounce, size_t a_size,
                                         cl_mem xMultOneMinusAlpha, cl_mem yMultAlpha)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTAcceptReject");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_xVector));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_yVector));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_xColor));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&a_yColor));

  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&a_split));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&a_scaleTable));

  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&a_rstateForAcceptReject));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&xMultOneMinusAlpha));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&yMultAlpha));
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_int), (void*)&a_maxBounce));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_int), (void*)&isize));
  
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


void GPUOCLLayer::runKernel_MMLTInitSplitAndCamV(cl_mem a_flags, cl_mem a_color, cl_mem a_split, cl_mem a_hitSup,
                                                 size_t a_size)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTInitCameraPath");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_flags));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_color));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_split));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&a_hitSup));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTMakeProposal(cl_mem in_rgen, cl_mem in_vec, cl_int a_mutationType, cl_int a_maxBounce, size_t a_size,
                                             cl_mem out_rgen, cl_mem out_vec)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTMakeProposal");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&m_mlt.splitData));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_rgen));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&out_rgen));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&in_vec));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&out_vec));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_int), (void*)&a_mutationType));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_int), (void*)&a_maxBounce));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_KMLTMakeProposal(cl_mem in_rgen, cl_mem in_vec, int a_largeStep, size_t a_size,
                                             cl_mem out_rgen, cl_mem out_vec, cl_mem out_zindex)
{
  cl_kernel kernX      = m_progs.mlt.kernel("KMLTMakeProposal");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_rgen));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&out_rgen));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&in_vec));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&out_vec));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&out_zindex));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_int), (void*)&a_largeStep));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
  
}     

void GPUOCLLayer::runKernel_UnsortColors(cl_mem in_color, cl_mem in_zindex, size_t a_size,
                                         cl_mem out_color)
{
  cl_kernel kernX      = m_progs.mlt.kernel("UnsortColors");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));
 
  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_zindex));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_color));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&out_color));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_DebugClearInt2WithTID(cl_mem index, size_t a_size)
{

  cl_kernel kernX      = m_progs.mlt.kernel("DebugClearInt2WithTID");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));
 
  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&index));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTMakeEyeRays(size_t a_size,
                                            cl_mem a_rpos, cl_mem a_rdir, cl_mem a_zindex)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTMakeEyeRays");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&m_mlt.currVec));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&a_zindex));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTCameraPathBounce(cl_mem rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_color, cl_mem a_split, size_t a_size,
                                                 cl_mem a_outHitCom, cl_mem a_outHitSup)
{
  const cl_float mLightSubPathCount = cl_float(m_width*m_height);

  cl_kernel kernX      = m_progs.mlt.kernel("MMLTCameraPathBounce");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&rayFlags));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&m_mlt.rstateCurr)); 

  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_mlt.currVec)); 
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&a_split));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_rays.hits));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_scene.instLightInst));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&a_outHitCom));
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));
 
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&a_color));
  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&m_rays.fogAtten));
  CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX,14, sizeof(cl_mem), (void*)&a_outHitSup));

  CHECK_CL(clSetKernelArg(kernX,15, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,16, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kernX,17, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX,18, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
 
  CHECK_CL(clSetKernelArg(kernX,19, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,20, sizeof(cl_int), (void*)&isize));
  CHECK_CL(clSetKernelArg(kernX,21, sizeof(cl_float), (void*)&mLightSubPathCount));

  size_t size2 = m_mlt.currBounceThreadsNum;
  size2        = roundBlocks(size2, int(localWorkSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &size2, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_CopyAccColorTo(cl_mem cameraVertexSup, size_t a_size, cl_mem a_outColor)
{
  cl_kernel kernX      = m_progs.mlt.kernel("CopyAccColorTo");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&cameraVertexSup));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_outColor));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTLightSampleForward(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, cl_mem lightVertexSup, size_t a_size)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTLightSampleForward");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rayFlags));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&m_mlt.rstateCurr));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_mlt.currVec));

  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&a_outColor));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&lightVertexSup));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,12, sizeof(int),    (void*)&isize));
  
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTLightPathBounce(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_color, cl_mem a_split, size_t a_size,
                                                cl_mem a_outHitCom, cl_mem a_outHitSup)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTLightPathBounce");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rayFlags));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&m_mlt.rstateCurr));

  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&m_mlt.currVec));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&a_split));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&a_outHitCom));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&a_color));
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_rays.fogAtten));
  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&a_outHitSup));

  CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,14, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kernX,15, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX,16, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX,17, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,18, sizeof(cl_int), (void*)&isize));

  size_t size2 = m_mlt.currBounceThreadsNum;
  size2        = roundBlocks(size2, int(localWorkSize));
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &size2, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}


void GPUOCLLayer::runkernel_MMLTMakeShadowRay(cl_mem in_splitInfo, cl_mem  in_cameraVertexHit, cl_mem in_cameraVertexSup, cl_mem  in_lightVertexHit, cl_mem  in_lightVertexSup, size_t a_size,
                                              cl_mem sray_pos, cl_mem sray_dir, cl_mem sray_flags)
                                              
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTMakeShadowRay");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_splitInfo));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_lightVertexHit));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&in_lightVertexSup));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&in_cameraVertexHit));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&in_cameraVertexSup));

  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&sray_pos));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&sray_dir));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&sray_flags));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_rays.lsamRev));

  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_mlt.rstateCurr));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_mlt.currVec));

  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,14, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,15, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTConnect(cl_mem in_splitInfo, cl_mem  in_cameraVertexHit, cl_mem in_cameraVertexSup, cl_mem  in_lightVertexHit, cl_mem  in_lightVertexSup, cl_mem in_shadow, size_t a_size, size_t a_sizeWholeBuff, 
                                        cl_mem a_outColor, cl_mem a_outZIndex)
{
  const cl_float mLightSubPathCount = cl_float(m_width*m_height);

  cl_kernel kernX      = m_progs.mlt.kernel("MMLTConnect");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  int isize2 = int(a_sizeWholeBuff);

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_splitInfo));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_lightVertexHit));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&in_lightVertexSup));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&in_cameraVertexHit));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&in_cameraVertexSup));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&in_shadow));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_rays.lsamRev));
  
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&a_outColor));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&a_outZIndex));

  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX,14, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX,15, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,16, sizeof(cl_mem), (void*)&m_mlt.scaleTable));
  CHECK_CL(clSetKernelArg(kernX,17, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(kernX,18, sizeof(cl_int), (void*)&isize));
  CHECK_CL(clSetKernelArg(kernX,19, sizeof(cl_float), (void*)&mLightSubPathCount));
  CHECK_CL(clSetKernelArg(kernX,20, sizeof(cl_int), (void*)&isize2));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

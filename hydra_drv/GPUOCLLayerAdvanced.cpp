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

void GPUOCLLayer::runKernal_MMLTMakeEyeRays(size_t a_size,
                                            cl_mem a_rpos, cl_mem a_rdir)
{
  cl_kernel kernX      = m_progs.mlt.kernel("MMLTMakeEyeRays");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&m_mlt.currVec));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&a_rpos));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&a_rdir));
  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_int), (void*)&isize));

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

  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&a_split));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_rays.hits));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_scene.instLightInst));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&a_outHitCom));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));
 
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&a_color));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_rays.fogAtten));
  CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&a_outHitSup));

  CHECK_CL(clSetKernelArg(kernX,14, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,15, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kernX,16, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX,17, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
 
  CHECK_CL(clSetKernelArg(kernX,18, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,19, sizeof(cl_int), (void*)&isize));
  CHECK_CL(clSetKernelArg(kernX,20, sizeof(cl_float), (void*)&mLightSubPathCount));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
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

  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&a_outColor));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&lightVertexSup));
  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,11, sizeof(int),    (void*)&isize));
  
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

  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&a_split));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&a_outHitCom));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_mem), (void*)&m_rays.hitProcTexData));

  CHECK_CL(clSetKernelArg(kernX, 7, sizeof(cl_mem), (void*)&a_color));
  CHECK_CL(clSetKernelArg(kernX, 8, sizeof(cl_mem), (void*)&m_rays.pathMisDataPrev));
  CHECK_CL(clSetKernelArg(kernX, 9, sizeof(cl_mem), (void*)&m_rays.fogAtten));
  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_mlt.pdfArray));
  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&a_outHitSup));

  CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&m_scene.storageTexAux));
  CHECK_CL(clSetKernelArg(kernX,14, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX,15, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX,16, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,17, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
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

  CHECK_CL(clSetKernelArg(kernX,10, sizeof(cl_mem), (void*)&m_scene.storageMat));
  CHECK_CL(clSetKernelArg(kernX,11, sizeof(cl_mem), (void*)&m_scene.storagePdfs));
  CHECK_CL(clSetKernelArg(kernX,12, sizeof(cl_mem), (void*)&m_scene.storageTex));
  CHECK_CL(clSetKernelArg(kernX,13, sizeof(cl_mem), (void*)&m_scene.allGlobsData));
  CHECK_CL(clSetKernelArg(kernX,14, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::runKernel_MMLTConnect(cl_mem in_splitInfo, cl_mem  in_cameraVertexHit, cl_mem in_cameraVertexSup, cl_mem  in_lightVertexHit, cl_mem  in_lightVertexSup, cl_mem in_shadow, size_t a_size, 
                                        cl_mem a_outColor, cl_mem a_outZIndex)
{
  const cl_float mLightSubPathCount = cl_float(m_width*m_height);

  cl_kernel kernX      = m_progs.mlt.kernel("MMLTConnect");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));

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
  CHECK_CL(clSetKernelArg(kernX,16, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(kernX,17, sizeof(cl_int), (void*)&isize));
  CHECK_CL(clSetKernelArg(kernX,18, sizeof(cl_float), (void*)&mLightSubPathCount));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::EvalSBDPT(cl_mem in_xVector, size_t a_size,
                            cl_mem a_outColor, cl_mem a_outZIndex)
{
  m_mlt.currVec = in_xVector;
  cl_mem a_rpos = m_rays.rayPos;
  cl_mem a_rdir = m_rays.rayDir;
  
  
  int maxBounce = MMLT_GPU_TEST_DEPTH;

  // (1) init and camera pass 
  //
  runKernel_MakeEyeRays(m_rays.rayPos, m_rays.rayDir, m_rays.samZindex, m_rays.MEGABLOCKSIZE, m_passNumberForQMC, false);
  runKernel_MMLTInitSplitAndCamV(m_rays.rayFlags, a_outColor, m_mlt.splitData, m_mlt.cameraVertexSup, a_size);

  for (int bounce = 0; bounce < maxBounce; bounce++)
  {
    runKernel_Trace(a_rpos, a_rdir, a_size,
                    m_rays.hits);
  
    runKernel_ComputeHit(a_rpos, a_rdir, m_rays.hits, a_size, 
                         m_mlt.cameraVertexHit, m_rays.hitProcTexData);
  
    runKernel_MMLTCameraPathBounce(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, m_mlt.splitData, a_size,  //#NOTE: m_mlt.rstateCurr used inside
                                   m_mlt.cameraVertexHit, m_mlt.cameraVertexSup);
  }

  //runKernel_CopyAccColorTo(m_mlt.cameraVertexSup, a_size, 
  //                         a_outColor);
  //return;

  // (2) light pass
  //
  cl_mem lightVertexHit = m_rays.hitSurfaceAll;
  cl_mem lightVertexSup = m_mlt.lightVertexSup;

  runKernel_MMLTLightSampleForward(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, lightVertexSup, a_size);
  
  for (int bounce = 0; bounce < (maxBounce-1); bounce++) // last bounce is always a connect stage
  {
    runKernel_Trace(a_rpos, a_rdir, a_size,
                    m_rays.hits);

    runKernel_ComputeHit(a_rpos, a_rdir, m_rays.hits, a_size, 
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
  
  runKernel_MMLTConnect(m_mlt.splitData, m_mlt.cameraVertexHit, m_mlt.cameraVertexSup, lightVertexHit, lightVertexSup, m_rays.lshadow, a_size, 
                        a_outColor, a_outZIndex);


  m_mlt.currVec = nullptr;
}



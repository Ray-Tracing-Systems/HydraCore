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

  const int bouncesIntoAccount       = bounceEnd - bounceBeg + 1;
  const size_t blocksPerTargetDepth  = blocksNum / bouncesIntoAccount;
  const size_t finalThreadsNum       = (blocksPerTargetDepth*bouncesIntoAccount)*256;

  activeThreads.resize(a_maxDepth+1);
  for(int i=0;i<bounceBeg;i++)
    activeThreads[i] = 0;

  size_t currPos = 0;
  for(int bounce = a_maxDepth; bounce >= bounceBeg; bounce--)
  {
    for(int b=0;b<blocksPerTargetDepth;b++)
    {
      for(int i=0;i<256;i++)
        splitDataCPU[(currPos + b)*256 + i] = make_int2(bounce, bounce);
    }
    currPos += blocksPerTargetDepth;
    activeThreads[activeThreads.size() - bounce + 1] = int(currPos*256);
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
  for(int i=a_maxDepth;i>=bounceBeg;i--)
    std::cout << "[d = " << i << ",\tN = " << activeThreads[i] << ", coeff = " << scale[i] << "]" << std::endl;
  std::cout << "finalThreadsNum = " << finalThreadsNum << std::endl;
  std::cout << std::endl;

  return finalThreadsNum;
}

void GPUOCLLayer::EvalSBDPT(cl_mem in_xVector, int maxBounce, size_t a_size,
                            cl_mem a_outColor, cl_mem a_outZIndex)
{
  m_mlt.currVec = in_xVector;
  cl_mem a_rpos = m_rays.rayPos;
  cl_mem a_rdir = m_rays.rayDir;

  // (1) init and camera pass 
  //
  runKernal_MMLTMakeEyeRays(a_size,
                            m_rays.rayPos, m_rays.rayDir, m_rays.samZindex);
  
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



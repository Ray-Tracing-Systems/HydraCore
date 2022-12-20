#include "GPUOCLLayer.h"
#include "crandom.h"
#include "cl_scan_gpu.h"

#include <algorithm>
#undef min
#undef max

#include <iomanip>
#include <chrono>

#include "hydra_api/ssemath.h"

#ifndef WIN32
#include <csignal>
#endif

void GPUOCLLayer::AddContributionToScreen(cl_mem& in_color, cl_mem in_indices, bool a_copyToLDRNow, int a_layerId, bool a_repackIndex)
{
  if (m_screen.m_cpuFrameBuffer)
  { 
    int width, height, channels;
    auto resultPtr = const_cast<float*>( GetCPUScreenBuffer(a_layerId, width, height, channels) );

    assert(resultPtr != nullptr);

    AddContributionToScreenCPU(in_color, int(m_rays.MEGABLOCKSIZE), width, height, channels,
                               resultPtr, a_repackIndex);
  }
  else
  {
    if(in_indices == nullptr)
    {
      in_indices = m_rays.samZindex;
      runKernel_UpdateZIndexFromColorW(in_color, m_rays.MEGABLOCKSIZE, 
                                       in_indices);
    }

    AddContributionToScreenGPU(in_color, in_indices, int(m_rays.MEGABLOCKSIZE), m_width, m_height, m_passNumber, a_copyToLDRNow,
                               m_screen.color0, m_screen.pbo);
  }

  m_passNumber++;
}

/**
\brief Add contribution
\param out_color  - out float4 image of size a_width*a_height
\param colors     - in float4 array of size a_size
\param a_size     - array size
\param a_width    - image width
\param a_height   - image height

*/
void AddSamplesContribution(float* out_color, const float4* colors, int a_size, int a_width, int a_height, int a_channels)
{
  const int ch = a_channels == 4 ? 3 : a_channels;

  for (int i = 0; i < a_size; i++)
  {
    const float4 color    = colors[i];
    const int packedIndex = as_int(color.w);
    const int x           = (packedIndex & 0x0000FFFF);
    const int y           = (packedIndex & 0xFFFF0000) >> 16;
    const int offset      = y * a_width + x;

    if (x >= 0 && y >= 0 && x < a_width && y < a_height)
    {
      for(int j = 0; j < ch; ++j)
      {
        out_color[offset * a_channels + j] += color[j];
      }
//      out_color[offset].x += color.x;
//      out_color[offset].y += color.y;
//      out_color[offset].z += color.z;
    }
  }
}

/**
\brief Add contribution with storing shadows in the fourth channel
\param out_color  - out float4 image of size a_width*a_height
\param colors     - in float4 array of size a_size
\param shadows    - in cl_uint8 array of compressed shadow value
\param a_size     - array size
\param a_width    - image width
\param a_height   - image height

*/
void AddSamplesContributionS(float* out_color, const float4* colors, const unsigned char* shadows, int a_size,
                             int a_width, int a_height, int a_channels)
{
  const float multInv = 1.0f / 255.0f;
  const int ch = a_channels == 4 ? 3 : a_channels;

  for (int i = 0; i < a_size; i++)
  {
    const float4 color    = colors[i];
    const auto   shad     = shadows[i];

    const int packedIndex = as_int(color.w);
    const int x           = (packedIndex & 0x0000FFFF);
    const int y           = (packedIndex & 0xFFFF0000) >> 16;
    const int offset      = y * a_width + x;

    if (x >= 0 && y >= 0 && x < a_width && y < a_height)
    {
//      out_color[offset].x += color.x;
//      out_color[offset].y += color.y;
//      out_color[offset].z += color.z;
      for(int j = 0; j < ch; ++j)
      {
        out_color[offset * a_channels + j] += color[j];
      }
      if(a_channels == 4)
        out_color[offset * a_channels + 3] += multInv * float(shad);
    }
  }
}


void GPUOCLLayer::AddContributionToScreenCPU2(cl_mem& in_color, cl_mem& in_color2, int a_size, int a_width, int a_height,
                                              int a_channels, float* out_color)
{
  clFlush(m_globals.cmdQueue);

  // (2) sync copy of data (sync asyncronious call in future, pin pong) and eval contribution
  //
  int workingPass = (m_camPlugin.pCamPlugin == nullptr) ? 1 : 2; // for CPU

  if (m_passNumber >= workingPass)
  {
    clEnqueueCopyBuffer(m_globals.cmdQueueDevToHost, m_mlt.pathAuxColor,  m_mlt.pathAuxColorCPU,  0, 0, a_size * sizeof(float4), 0, nullptr, nullptr);
    clEnqueueCopyBuffer(m_globals.cmdQueueDevToHost, m_mlt.pathAuxColor2, m_mlt.pathAuxColorCPU2, 0, 0, a_size * sizeof(float4), 0, nullptr, nullptr);

    cl_int ciErr1  = 0;
    float4* colors1 = (float4*)clEnqueueMapBuffer(m_globals.cmdQueueDevToHost, m_mlt.pathAuxColorCPU,  CL_TRUE, CL_MAP_READ, 0, a_size * sizeof(float4), 0, 0, 0, &ciErr1);
    float4* colors2 = (float4*)clEnqueueMapBuffer(m_globals.cmdQueueDevToHost, m_mlt.pathAuxColorCPU2, CL_TRUE, CL_MAP_READ, 0, a_size * sizeof(float4), 0, 0, 0, &ciErr1);

    const float contribSPP = float(double(a_size) / double(a_width*a_height));

    bool lockSuccess = (m_pExternalImage == nullptr);
    if (m_pExternalImage != nullptr)
      lockSuccess = m_pExternalImage->Lock(500); // can wait 500 ms for success lock

    if (lockSuccess)
    {
      AddSamplesContribution(out_color, colors1, int(a_size), a_width, a_height, a_channels);
      AddSamplesContribution(out_color, colors2, int(a_size), a_width, a_height, a_channels);

      if (m_pExternalImage != nullptr)
      {
        m_pExternalImage->Header()->counterRcv++;
        m_pExternalImage->Unlock();
      }
    }
    else
    {
      std::cerr << "AddContributionToScreenCPU2, failed to lock image!" << std::endl;
      std::cerr.flush();
    }
    m_sppDone += contribSPP;

    clEnqueueUnmapMemObject(m_globals.cmdQueueDevToHost, m_mlt.pathAuxColorCPU,  colors1, 0, 0, 0);
    clEnqueueUnmapMemObject(m_globals.cmdQueueDevToHost, m_mlt.pathAuxColorCPU2, colors2, 0, 0, 0);
  }

  clFinish(m_globals.cmdQueueDevToHost);
  clFinish(m_globals.cmdQueue);

  //memsetf4(m_mlt.pathAuxColor,  float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
  //memsetf4(m_mlt.pathAuxColor2, float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
  //clFinish(m_globals.cmdQueue);

  // (3) swap color buffers
  //
  {
    cl_mem temp         = m_mlt.pathAuxColor;
    m_mlt.pathAuxColor  = in_color;
    in_color            = temp;

    temp                = m_mlt.pathAuxColor2;
    m_mlt.pathAuxColor2 = in_color2;
    in_color2           = temp;
  }

  m_passNumber++;
}

void GPUOCLLayer::AddContributionToScreenCPU(cl_mem& in_color, int a_size, int a_width, int a_height, int a_channels, float* out_color, bool repackIndex)
{
  // (1) compute compressed index in color.w; use runKernel_MakeEyeRaysAndClearUnified for that task if CPU FB is enabled!!!
  //
  size_t szLocalWorkSize = 256;
  cl_int iNumElements    = cl_int(a_size);
  size_t size            = roundBlocks(size_t(a_size), int(szLocalWorkSize));

  if (repackIndex && (m_vars.m_flags & HRT_FORWARD_TRACING)   == 0 && 
                     (m_vars.m_flags & HRT_ENABLE_MMLT)       == 0 && 
                      m_vars.m_varsI[HRT_ENABLE_SURFACE_PACK] == 0) // lt/mmlt already pack index to color, so, don't do that again. HRT_ENABLE_SURFACE_PACK from the other side needs surface id at this place
  {
    cl_kernel kern = m_progs.screen.kernel("PackIndexToColorW");

    CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&m_rays.packedXY));
    CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&in_color));
    CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&iNumElements));
    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &size, &szLocalWorkSize, 0, NULL, NULL));
  }
  
  clFlush(m_globals.cmdQueue);
  clFlush(m_globals.cmdQueueHostToDev);

  auto startExec = std::chrono::high_resolution_clock::now();

  const bool ltPassOfIBPT = (m_vars.m_flags & HRT_3WAY_MIS_WEIGHTS) && (m_vars.m_flags & HRT_FORWARD_TRACING);

  Timer copyTimer(true);
  const bool measureTime = false;

  float timeCopy    = 0.0f;
  float timeContrib = 0.0f;

  // (2) sync copy of data (sync asyncronious call in future, pin pong) and eval contribution
  //
  const int startPass = (m_camPlugin.pCamPlugin == nullptr) ? 0 : 1;
  if (m_passNumber > startPass)
  {
    clEnqueueCopyBuffer(m_globals.cmdQueueDevToHost, m_rays.pathAuxColor, m_rays.pathAuxColorCPU, 0, 0, a_size * sizeof(float4), 0, nullptr, nullptr);
    if (m_storeShadowInAlphaChannel)
      clEnqueueCopyBuffer(m_globals.cmdQueueDevToHost, m_rays.pathShadow8BAux, m_rays.pathShadow8BAuxCPU, 0, 0, a_size * sizeof(float4), 0, nullptr, nullptr);

    cl_int ciErr1  = 0;
    float4* colors = (float4*)clEnqueueMapBuffer(m_globals.cmdQueueDevToHost, m_rays.pathAuxColorCPU, CL_TRUE, CL_MAP_READ, 0, a_size * sizeof(float4), 0, 0, 0, &ciErr1);

    cl_uint8* shadows = nullptr;
    if (m_storeShadowInAlphaChannel)
      shadows = (cl_uint8*)( clEnqueueMapBuffer(m_globals.cmdQueueDevToHost, m_rays.pathShadow8BAuxCPU, CL_TRUE, CL_MAP_READ, 0, a_size * sizeof(cl_uint8), 0, 0, 0, &ciErr1) );


    if (measureTime)
    {
      //clFinish(m_globals.cmdQueueDevToHost);
      timeCopy = copyTimer.getElapsed();
    }

    const float contribSPP = float(double(a_size) / double(a_width*a_height));

    bool lockSuccess = (m_pExternalImage == nullptr);
    if (m_pExternalImage != nullptr)
    {
//#ifndef WIN32
//      { // block termination signal from HydraAPI
//        sigset_t nset;
//        sigaddset(&nset, SIGUSR1);
//        //sigfillset(&nset);
//        sigprocmask(SIG_BLOCK, &nset, NULL);
//      }
//#endif
      lockSuccess = m_pExternalImage->Lock(400);
    }

    if (lockSuccess)
    {
      if(m_camPlugin.pCamPlugin != nullptr) 
      {
        auto startContrib = std::chrono::high_resolution_clock::now();
        m_camPlugin.pCamPlugin->AddSamplesContribution((float*)out_color, (const float*)colors, size, a_width, a_height, m_passNumber);
        m_camPlugin.pipeTime[2] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startContrib).count()/1000.f;
      }
      else if (m_storeShadowInAlphaChannel)
        AddSamplesContributionS(out_color, colors, (const unsigned char*)shadows, int(size), a_width, a_height, a_channels);
      else
        AddSamplesContribution(out_color, colors, int(size), a_width, a_height, a_channels);

      if (m_pExternalImage != nullptr) //#TODO: if ((m_vars.m_flags & HRT_FORWARD_TRACING) == 0) IT IS DIFFERENT FOR LT !!!!!!!!!!
      {
        if (!ltPassOfIBPT) // don't update counters if this is only first pass of two-pass IBPT
        {
          m_pExternalImage->Header()->counterRcv++;
          m_pExternalImage->Header()->spp += contribSPP;
          m_sppContrib += contribSPP;
        }
        m_pExternalImage->Unlock();

        if(m_vars.m_varsI[HRT_BOX_MODE_ON] == 1 && m_sppContrib >= m_vars.m_varsI[HRT_CONTRIB_SAMPLES])  // quit immediately
          exit(0);
//#ifndef WIN32
//        { // unblock termination signal from HydraAPI
//          sigset_t nset;
//          sigaddset(&nset, SIGUSR1);
//          //sigfillset(&nset);
//          sigprocmask(SIG_UNBLOCK, &nset, NULL);
//        }
//#endif
      }
    }
    else
    {
      std::cerr << "AddContributionToScreenCPU, failed to lock image!" << std::endl;
      std::cerr.flush();
    }

    m_sppDone += contribSPP;

    if (m_camPlugin.pCamPlugin != nullptr) 
      std::cout << "progress = " << 100.0f*m_sppDone / float(m_vars.m_varsI[HRT_MAX_SAMPLES_PER_PIXEL]) << " %" << std::endl;

    if (measureTime && lockSuccess)
    {
      timeContrib = copyTimer.getElapsed();
      std::cout << "time copy    = " << timeCopy*1000.0f << std::endl;
      std::cout << "time contrib = " << (timeContrib - timeCopy)*100.0f << std::endl;
    }

    clEnqueueUnmapMemObject(m_globals.cmdQueueDevToHost, m_rays.pathAuxColorCPU, colors, 0, 0, 0);
    if (m_storeShadowInAlphaChannel)
      clEnqueueUnmapMemObject(m_globals.cmdQueueDevToHost, m_rays.pathShadow8BAuxCPU, shadows, 0, 0, 0);

    clFlush(m_globals.cmdQueueDevToHost);
  }

  clFinish(m_globals.cmdQueue);
  clFinish(m_globals.cmdQueueHostToDev);
  clFinish(m_globals.cmdQueueDevToHost);

  m_camPlugin.pipeTime[1] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startExec).count()/1000.f;

  if (m_camPlugin.pCamPlugin != nullptr && m_settingsNode.child(L"pipeTime").text().as_int() == 1) 
  {
    std::cout << "time[0] = " << m_camPlugin.pipeTime[0] << " ms" << std::endl;
    std::cout << "time[1] = " << m_camPlugin.pipeTime[1] << " ms" << std::endl;
    std::cout << "time[2] = " << m_camPlugin.pipeTime[2] << " ms" << std::endl;
  }
  //memsetf4(m_rays.pathAuxColor, float4(0,0,0,0), m_rays.MEGABLOCKSIZE, 0);
  //clFinish(m_globals.cmdQueue);

  if (measureTime)
  {
    std::cout << "time total   = " << copyTimer.getElapsed()*1000.0f << std::endl;
    std::cout << std::endl;
  }

  // (3) swap color and shadow8 buffers
  //
  {
    cl_mem temp         = m_rays.pathAuxColor;
    m_rays.pathAuxColor = in_color;
    in_color            = temp;

    temp                   = m_rays.pathShadow8BAux;
    m_rays.pathShadow8BAux = m_rays.pathShadow8B;
    m_rays.pathShadow8B    = temp;
  }
  
}

void GPUOCLLayer::CPUPluginFinish()
{
  if(m_camPlugin.pCamPlugin != nullptr)
    m_camPlugin.pCamPlugin->FinishRendering();
}

extern int g_maxCPUThreads;

void GPUOCLLayer::ContribToExternalImageAccumulator(IHRSharedAccumImage* a_pImage)
{
  if (!m_screen.m_cpuFrameBuffer)
  {
    if(m_screen.color0CPU.size() != m_width * m_height)
      m_screen.color0CPU.resize(m_width*m_height);
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_screen.color0, CL_TRUE, 0, m_width*m_height * sizeof(cl_float4), m_screen.color0CPU.data(), 0, NULL, NULL));
  }

  float* input = m_screen.color0CPU.data();
  if (input == nullptr)
  {
    std::cerr << "GPUOCLLayer::ContribToExternalImageAccumulator: nullptr internal image" << std::endl;
    return;
  }

  if (a_pImage == nullptr)
  {
    std::cerr << "GPUOCLLayer::ContribToExternalImageAccumulator: nullptr external image" << std::endl;
    return;
  }

  const bool lockSuccess = a_pImage->Lock(100); // can wait 100 ms for success lock

  if (lockSuccess)
  {
    float* output  = a_pImage->ImageData(0);
    const int size = m_width * m_height;

    if(m_screen.m_cpuFrameBuffer)
    {
      assert(a_pImage->Header()->channels == m_screen.m_cpuFbufChannels);
#pragma omp parallel for num_threads(g_maxCPUThreads)
      for (int i = 0; i < size; i++)
      {
//      const __m128 color1 = _mm_load_ps(input  + i * 4);
//      const __m128 color2 = _mm_load_ps(output + i * 4);
//      _mm_store_ps(output + i * 4, _mm_add_ps(color1, color2));
        for(int j = 0; j < m_screen.m_cpuFbufChannels; ++j)
          output[i * m_screen.m_cpuFbufChannels + j] += input[i * m_screen.m_cpuFbufChannels + j];
      }
    }

    else
    {
      #pragma omp parallel for num_threads(g_maxCPUThreads)
      for (int i = 0; i < size; i++)
      {
      const __m128 color1 = _mm_load_ps(input  + i * 4);
      const __m128 color2 = _mm_load_ps(output + i * 4);
      _mm_store_ps(output + i * 4, _mm_add_ps(color1, color2));
      }
    }

    a_pImage->Header()->counterRcv++;
    a_pImage->Header()->spp += m_spp;
    a_pImage->Unlock();

    ClearAccumulatedColor();

    m_sppContrib += m_spp;
    m_sppDone    += m_spp;
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<int> GPUOCLLayer::MakeAllPixelsList()
{
  std::vector<int> allPixels(m_width*m_height);

  //#pragma omp parallel for
  //for(int y=0;y<m_height;y++)
  //{
  //  for(int x=0;x<m_width;x++)
  //    allPixels[y*m_width+x] = packXY1616(x,y);
  //}
  //return allPixels;

  const int TILE_SIZE = 32;

  const int maxXRounded = (int(m_width) / int(TILE_SIZE)) * TILE_SIZE;
  const int maxYRounded = (int(m_height) / int(TILE_SIZE)) * TILE_SIZE;

  int top = 0;

  for(int ty=0; ty<maxYRounded; ty+=TILE_SIZE)
  {
    for(int tx=0; tx<maxXRounded; tx+=TILE_SIZE)
    {
      for(int y1=0;y1<TILE_SIZE;y1++)
      {
        const int y = ty + y1;
        for(int x1=0;x1<TILE_SIZE;x1++)
        {
          const int x = tx + x1;
          allPixels[top + y1*TILE_SIZE + x1] = packXY1616(x,y);  // ZIndexHost(x1,y1)
        }
      }

      top += (TILE_SIZE*TILE_SIZE);
    }
  }

  // push borders
  //
  const int remX = m_width  - maxXRounded;
  const int remY = m_height - maxYRounded;

  for(int y = 0; y < m_height; y++)
  {
    for(int x = maxXRounded; x < m_width; x++)
    {
      allPixels[top] = packXY1616(x,y);
      top++;
    }
  }

  for(int x=0;x<maxXRounded;x++)
  {
    for(int y = maxYRounded; y < m_height; y++)
    {
      allPixels[top] = packXY1616(x,y);
      top++;
    }
  }

  assert(top == m_width*m_height);

  return allPixels;
}


void GPUOCLLayer::RunProductionSamplingMode()
{
  std::cout << "ProductionSamplingMode begin" << std::endl; std::cout.flush();

  Timer timer(true);

  if(m_screen.color0CPU.size() != m_width*m_height)
  {
    m_screen.color0CPU.resize(m_width*m_height);
    memset(m_screen.color0CPU.data(), 0, m_screen.color0CPU.size()*sizeof(float4));
  }

  // (1) create pixels list
  //
  std::vector<int> allPixels = MakeAllPixelsList();

  const int pixelsPerPass = GetRayBuffSize()  / PMPIX_SAMPLES;
  const int numPasses     =  (m_width*m_height)%pixelsPerPass == 0 ? (m_width*m_height)/pixelsPerPass : (m_width*m_height)/pixelsPerPass + 1;

  cl_int ciErr1 = CL_SUCCESS;

  cl_mem pixCoordGPU = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY,  pixelsPerPass*sizeof(int),    nullptr, &ciErr1);
  cl_mem pixColorGPU = clCreateBuffer(m_globals.ctx, CL_MEM_WRITE_ONLY, pixelsPerPass*sizeof(float4), nullptr, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in clCreateBuffer, RunProductionSamplingMode");

  cvex::vector<float4> pixColors(pixelsPerPass);

  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, pixCoordGPU, CL_TRUE, 0, // CL_FALSECL_TRUE
                                pixelsPerPass*sizeof(int), (void*)(allPixels.data() + 0), 0, NULL, NULL));

  int currPos = 0;
  bool earlyExit = false;
  for(int pass = 0; pass < numPasses; pass++)
  {
    if(m_pExternalImage != nullptr)
    {
      bool q1 = false, q2 = false;
      int maxSamplesPerPixel = 0;

      if(m_pExternalImage != nullptr)
      {
        maxSamplesPerPixel = m_vars.m_varsI[HRT_MAX_SAMPLES_PER_PIXEL];
        auto pHeader = m_pExternalImage->Header();
        std::string msg(m_pExternalImage->MessageSendData());
        q1 = (pHeader->spp >= maxSamplesPerPixel);
        q2 = (msg.find("exitnow") != std::string::npos);
      }

      if(q1 || q2) // to quit immediately
      {
        m_sppDone    = maxSamplesPerPixel;
        m_sppContrib = maxSamplesPerPixel;
        earlyExit    = true;
        break;
      }
    }

    // (3) generate PMPIX_SAMPLES rays per each pixel
    //

    const int pixelsDone       = pass * pixelsPerPass;
    const int pixelsInThisPass = (pixelsDone + pixelsPerPass <= allPixels.size()) ? pixelsPerPass : int(allPixels.size() - pixelsDone);
    const int finalSize        = PMPIX_SAMPLES*pixelsInThisPass;

	  if (pass >= numPasses-5)
	  {
      int a = 2;
	  }

    runKernel_MakeEyeRaysSpp(PMPIX_SAMPLES, 0, finalSize, pixCoordGPU,
                             m_rays.rayPos, m_rays.rayDir);

    // (4) trace rays/paths
    //
    trace1D_Rev(1, m_vars.m_varsI[HRT_TRACE_DEPTH], m_rays.rayPos, m_rays.rayDir, finalSize,
                m_rays.pathAccColor);

    runKernel_GetShadowToAlpha(m_rays.pathAccColor, m_rays.pathShadow8B, finalSize);

    // (5) average colors
    //
    runKernel_ReductionFloat4Average(m_rays.pathAccColor, pixColorGPU, finalSize, PMPIX_SAMPLES);

    // (6) copy resulting colors to the CPU and add them to the image
    //
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, pixColorGPU, CL_TRUE, 0,
                                 pixelsInThisPass*sizeof(float4), pixColors.data(), 0, NULL, NULL));

    if(pass < numPasses-2) // copy next pixels portion asynchronious
    {
      const int pixelsInNextPass = (pixelsDone + pixelsInThisPass <= allPixels.size()) ? pixelsPerPass : int(allPixels.size()) - pixelsDone - pixelsInThisPass;

      CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, pixCoordGPU, CL_FALSE, 0,
                                    pixelsInNextPass*sizeof(int), (void*)(allPixels.data() + currPos + pixelsInThisPass), 0, NULL, NULL));
      clFlush(m_globals.cmdQueue);
    }

    const float multf = float(PMPIX_SAMPLES);
    for(int pixId = 0; pixId < pixelsInThisPass; pixId++) // contribute to image here
    {
      const int pixelPacked = allPixels[currPos + pixId];
      const int x           = (pixelPacked & 0x0000FFFF);
      const int y           = (pixelPacked & 0xFFFF0000) >> 16;
      const int offset      = y * m_width + x;

      int ch = std::min(m_screen.m_cpuFbufChannels, 4); //TODO: spectral?
      for(int i = 0; i < ch; ++i)
        m_screen.color0CPU[offset * ch + i] += (pixColors[pixId]*multf)[i];
    }

    currPos += pixelsPerPass;
    if(pass % 16 == 0)
    {
      std::cout << "production rendering: " << 100.0f*float(pass)/float(numPasses) << "% \r";
      std::cout.flush();
    }
  } // for

  m_globals.m_passNumberQMC += PMPIX_SAMPLES;

  std::cout << std::endl;

  clReleaseMemObject(pixCoordGPU); pixCoordGPU = nullptr;
  clReleaseMemObject(pixColorGPU); pixColorGPU = nullptr;

  m_spp        += PMPIX_SAMPLES;
  m_passNumber += 2; // just for GetLDRImage works correctly it have to be not 0, see pipelined copy for common pt ... ;

  const float renderingTime = timer.getElapsed();
  const int maxSamplesPerPixel = m_vars.m_varsI[HRT_MAX_SAMPLES_PER_PIXEL];

  std::cout << "ProductionSamplingMode end, time = " << renderingTime << "s" << std::endl; std::cout.flush();

  if(m_pExternalImage != nullptr && !earlyExit)
  {
    float* resultPtr  = m_pExternalImage->ImageData(0);
    const int width   = m_pExternalImage->Header()->width;
    const int height  = m_pExternalImage->Header()->height;
    assert(m_pExternalImage->Header()->channels == m_screen.m_cpuFbufChannels);

    const bool lockSuccess = m_pExternalImage->Lock(1000); // can wait 1s for success lock

    if (lockSuccess)
    {
      const int size = m_width*m_height;

      #pragma omp parallel for
      for(int i = 0; i < size; i++)
      {
        for(int j = 0; j < m_screen.m_cpuFbufChannels; ++j)
        {
          const float c = m_screen.color0CPU[i * m_screen.m_cpuFbufChannels + j];
          resultPtr[i * m_screen.m_cpuFbufChannels + j] += c;
          m_screen.color0CPU[i * m_screen.m_cpuFbufChannels + j] = 0.0f;
        }
//        const float4 color = m_screen.color0CPU[i];
//        resultPtr[i] += color;
//        m_screen.color0CPU[i] = float4(0,0,0,0);
      }

      m_pExternalImage->Header()->counterRcv++;
      m_pExternalImage->Header()->spp += PMPIX_SAMPLES;
      m_sppContrib                    += PMPIX_SAMPLES;

      m_pExternalImage->Unlock();

      //std::cerr << "m_sppContrib        = " << m_sppContrib << std::endl;
      //std::cerr << "HRT_CONTRIB_SAMPLES = " << m_vars.m_varsI[HRT_CONTRIB_SAMPLES] << std::endl;
      //std::cerr << "flags.prod.mode     = " << (m_vars.m_flags & HRT_PRODUCTION_IMAGE_SAMPLING) << std::endl;

      if(m_vars.m_varsI[HRT_BOX_MODE_ON] == 1 && m_sppContrib >= m_vars.m_varsI[HRT_CONTRIB_SAMPLES])  // to quit immediately
        exit(0);
    }

    m_sppDone += PMPIX_SAMPLES;

    if(m_pExternalImage->Header()->spp >= maxSamplesPerPixel) // to quit immediately
    {
      m_sppDone    = maxSamplesPerPixel;
      m_sppContrib = maxSamplesPerPixel;
    }
  }

}

void DebugSaveFuckingGBufferAsManyImages(int a_width, int a_height, const std::vector<GBufferAll>& gbuffer, const wchar_t* a_path);

void GPUOCLLayer::EvalGBuffer(IHRSharedAccumImage* a_pAccumImage, const std::vector<int32_t>& a_instIdByInstId)
{
  if (a_pAccumImage == nullptr)
    return;

  if (a_pAccumImage->Header()->gbufferIsEmpty != 1)
    return;

  bool locked = false;
  for (int i = 0; i < 20; i++)
  {
    locked = a_pAccumImage->Lock(100);
    if (locked)
      break;
    else
      std::cout << "GPUOCLLayer::EvalGBuffer: trying to lock shared image " << std::endl;
  }

  if (!locked)
    return;

  if (a_pAccumImage->Header()->gbufferIsEmpty != 1) // some other process already have computed gbuffer
  {
    a_pAccumImage->Unlock();
    return;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #TODO: refactor this
  float4* data1 = nullptr;
  float4* data2 = nullptr;
  if (a_pAccumImage->Header()->depth == 4)     //
  {
    data1 = (float4*)a_pAccumImage->ImageData(2);
    data2 = (float4*)a_pAccumImage->ImageData(3);
  }
  else if (a_pAccumImage->Header()->depth == 3) //
  {
    data1 = (float4*)a_pAccumImage->ImageData(1);
    data2 = (float4*)a_pAccumImage->ImageData(2);
  }
  else
  {
    std::cerr << "GPUOCLLayer::EvalGBuffer: wrong shared image layers num; num = " << a_pAccumImage->Header()->depth << std::endl;
    a_pAccumImage->Unlock();
    return;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #TODO: refactor this


  size_t  bufferSize = m_rays.MEGABLOCKSIZE;
  int32_t lineSize   = m_width * GBUFFER_SAMPLES;

  if (bufferSize % lineSize != 0)
    bufferSize -= (bufferSize % lineSize);

  assert(bufferSize % lineSize == 0);

  int32_t linesPerBlock = int32_t(bufferSize / lineSize);

  for (int32_t line = 0; line < m_height; line += linesPerBlock)
  {
    int32_t yBegin = line;
    int32_t yEnd   = line + linesPerBlock;
    if (yEnd > m_height)
      yEnd = m_height;

    int32_t finalSize = (yEnd - yBegin)*lineSize;

    // (1) generate eye rays
    //
    runKernel_MakeEyeRaysSpp(GBUFFER_SAMPLES, yBegin, finalSize, nullptr,
                             m_rays.rayPos, m_rays.rayDir);

    // (2) trace1D with single bounce
    //
    memsetu32(m_rays.rayFlags, 0, finalSize);                                                                // fill flags with zero
    //memsetf4 (m_rays.hitMatId, make_float4(0, 0, 0, 0), (finalSize * sizeof(HitMatRef)) / sizeof(float4)); // #TODO: fill accumulated rays dist with zero

    runKernel_Trace(m_rays.rayPos, m_rays.rayDir, finalSize,
                    m_rays.hits);

    runKernel_ComputeHit(m_rays.rayPos, m_rays.rayDir, m_rays.hits, finalSize, finalSize,
                         m_rays.hitSurfaceAll, m_rays.hitProcTexData);

    // (3) get compressed samples
    //
    runKernel_GetGBufferSamples(m_rays.rayDir, m_rays.pathAccColor, m_rays.pathShadeColor, GBUFFER_SAMPLES, finalSize);

    // (4) trace some more bounces to get alpha.
    //
    memsetf4(m_rays.pathThoroughput, make_float4(1, 1, 1, 1), finalSize);

    int maxBounce = m_vars.m_varsI[HRT_TRACE_DEPTH];
    if (maxBounce < 2)
      maxBounce = 2;
    for (int bounce = 1; bounce < maxBounce; bounce++)
    {
      runKernel_NextTransparentBounce(m_rays.rayPos, m_rays.rayDir, m_rays.pathThoroughput, finalSize);
      if (bounce == maxBounce - 1)
        break;

      runKernel_Trace(m_rays.rayPos, m_rays.rayDir, finalSize,
                      m_rays.hits);

      runKernel_ComputeHit(m_rays.rayPos, m_rays.rayDir, m_rays.hits, finalSize, finalSize,
                           m_rays.hitSurfaceAll, m_rays.hitProcTexData);
    }

    runKernel_PutAlphaToGBuffer(m_rays.pathThoroughput, m_rays.pathAccColor, finalSize);

    // (5) pass them to the host mem
    //
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.pathAccColor,    CL_FALSE, 0, (finalSize/GBUFFER_SAMPLES)*sizeof(float4), &data1[line*m_width], 0, NULL, NULL));
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.pathShadeColor,  CL_FALSE, 0, (finalSize/GBUFFER_SAMPLES)*sizeof(float4), &data2[line*m_width], 0, NULL, NULL));

    for (int x = 0; x < m_width; x++)
    {
      int* pInstId  = (int*)(&data2[line*m_width + x].w);
      int oldInstId = (*pInstId);
      if (oldInstId >= 0 && oldInstId <= a_instIdByInstId.size())
        (*pInstId) = a_instIdByInstId[oldInstId];
    }
  }

  clFinish(m_globals.cmdQueue);

  #pragma omp parallel for
  for (int32_t line = 0; line < m_height; line++)
  {
    for (int x = 0; x < m_width; x++)
    {
      int oldInstId = as_int(data2[line*m_width + x].w);
      if (oldInstId >= 0 && oldInstId < a_instIdByInstId.size())
        data2[line*m_width + x].w = as_float(a_instIdByInstId[oldInstId]);
    }
  }

  // //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // std::vector<GBufferAll> gbuffer(m_width*m_height);
  // #pragma omp parallel for
  // for (int i = 0; i < int(gbuffer.size()); i++)
  // {
  //   GBufferAll all;
  //   all.data1 = unpackGBuffer1(data1[i]);
  //   all.data2 = unpackGBuffer2(data2[i]);
  //   gbuffer[i] = all;
  // }
  // DebugSaveFuckingGBufferAsManyImages(m_width, m_height, gbuffer, L"gbufferout");
  // //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (a_pAccumImage != nullptr)
  {
    a_pAccumImage->Header()->gbufferIsEmpty = 0;
    a_pAccumImage->Unlock();
  }
}

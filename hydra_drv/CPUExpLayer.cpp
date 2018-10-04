#include "IHWLayer.h"

#include <iostream>
#include <fstream>
#include <vector>

#include "MemoryStorageCPU.h"

class CPUExpLayer : public CPUSharedData
{
  typedef CPUSharedData Base;

public:

  CPUExpLayer(int w, int h, int a_flags);
  ~CPUExpLayer();

  void Clear(CLEAR_FLAGS a_flags);
  void SetAllBVH4(const ConvertionResult& a_convertedBVH, IBVHBuilder2* a_inBuilderAPI, int a_flags) override;

  IMemoryStorage* CreateMemStorage(uint64_t a_maxSizeInBytes, const char* a_name);

  unsigned int AddLightMesh(LightMeshData a_lmesh);

  void PrepareEngineGlobals() override;

  //
  //
  void GetLDRImage(uint* data, int width, int height) const;
  void GetHDRImage(float4* data, int width, int height) const;

  void ResetPerfCounters();

  void BeginTracingPass();
  void EndTracingPass();
  void InitPathTracing(int seed);
  void ClearAccumulatedColor();

  void ResizeScreen(int w, int h, int a_flags);

  size_t GetAvaliableMemoryAmount(bool allMem);
  MRaysStat GetRaysStat();

  bool StoreCPUData()     const { return true; }

protected:

  void renderSubPixelData(const char* a_dataName, const std::vector<ushort2>& a_pixels, int spp, float4* a_pixValues, float4* a_subPixValues);

  std::vector<uint>       m_tempImage;
  std::vector<ZBlock>     m_tempBlocks;
  std::vector<float4>     m_cachedTx;
};


CPUExpLayer::CPUExpLayer(int w, int h, int a_flags) : Base(w, h, a_flags)
{
  ResizeScreen(w, h, a_flags);
}


CPUExpLayer::~CPUExpLayer()
{

}


void CPUExpLayer::ResizeScreen(int width, int height, int a_flags)
{
  IHWLayer::ResizeScreen(width, height, a_flags);
  m_tempImage.resize(width*height);
  m_width  = width;
  m_height = height;

  const float scaleX = sqrtf( float(width)  / 1024.0f );
  const float scaleY = sqrtf( float(height) / 1024.0f );

  m_vars.m_varsF[HRT_MLT_SCREEN_SCALE_X] = clamp(scaleX, 1.0f, 4.0f);
  m_vars.m_varsF[HRT_MLT_SCREEN_SCALE_Y] = clamp(scaleY, 1.0f, 4.0f);

  //m_pIntegrator->resizeScreen( ... )

  // DON'T !!!

  //if (m_pIntegrator != nullptr)
  //{
  //  delete m_pIntegrator;
  //  m_pIntegrator = new IntegratorCommon(m_width, m_height, &m_scnBVH, &m_scnGeom, (EngineGlobals*)&m_cdataPrepared[0]);
  //}

}

size_t CPUExpLayer::GetAvaliableMemoryAmount(bool allMem)
{
  return size_t(8) * size_t(1024 * 1024 * 1024); // 8 Gb 
}


void CPUExpLayer::SetAllBVH4(const ConvertionResult& a_convertedBVH, IBVHBuilder2* a_inBuilderAPI, int a_flags)
{
  Base::SetAllBVH4(a_convertedBVH, a_inBuilderAPI, a_flags);
}

void CPUExpLayer::Clear(CLEAR_FLAGS a_flags)
{

}

IMemoryStorage* CPUExpLayer::CreateMemStorage(uint64_t a_maxSizeInBytes, const char* a_name)
{
  IMemoryStorage* pStorage = new LinearStorageCPU;
  m_allMemStorages[a_name] = pStorage;
  pStorage->Reserve(a_maxSizeInBytes);
  return pStorage;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////
void CPUExpLayer::PrepareEngineGlobals()
{
  Base::PrepareEngineGlobals();
  m_pIntegrator->SetConstants((EngineGlobals*)&m_cdataPrepared[0]);
  m_pIntegrator->SetMaxDepth(m_vars.m_varsI[HRT_TRACE_DEPTH]);
  //m_pIntegrator->SetMaxDepth(2);
}

void CPUExpLayer::GetLDRImage(uint* data, int width, int height) const
{
  if (width != m_width || height != m_height)
    return;

  memcpy(data, &m_tempImage[0], m_width*m_height*sizeof(int));
}

void CPUExpLayer::GetHDRImage(float4* data, int width, int height) const
{
  if (data == nullptr)
    return;

  m_pIntegrator->GetImageHDR(data, width, height);
}

void CPUExpLayer::InitPathTracing(int seed)
{
  m_pIntegrator->Reset();
}

void CPUExpLayer::ClearAccumulatedColor()                                                                      
{
  m_pIntegrator->ClearAccumulatedColor();
}

void CPUExpLayer::renderSubPixelData(const char* a_dataName, const std::vector<ushort2>& a_pixels, int a_spp, float4* a_pixValues, float4* a_subPixValues)
{
  float4* a_norms = a_pixValues;
  float4* a_txcol = nullptr;

  if (std::string(a_dataName) == "normals" || std::string(a_dataName) == "gbuffer")
  {
    // do render, cache tx color
    m_cachedTx.resize(m_width*m_height);
    a_txcol = &m_cachedTx[0];
  }
  else if (std::string(a_dataName) == "texcolor")
  {
    // restore cached tx color, clear results
    memcpy(a_pixValues, &m_cachedTx[0], m_cachedTx.size()*sizeof(float4));
    m_cachedTx = std::vector<float4>();
    return;
  }

  bool isGBuffer = (std::string(a_dataName) == "gbuffer");

  // data for pixel clusters
  //

  std::vector<float4> clustNormals;
  std::vector<float4> clustColors;
  std::vector<int4>   clustBegEnd;

  clustBegEnd.reserve(a_pixels.size() / 100);       clustBegEnd.resize(0);
  clustNormals.reserve(clustBegEnd.capacity() * 4); clustNormals.resize(0);
  clustColors.reserve(clustBegEnd.capacity() * 4);  clustColors.resize(0);

  std::cout << "[cpu_core]: begin gbuffer render " << std::endl;

  //
  //
  const int spp = a_spp;// 16;

  std::vector<float2> samplePos(spp);
  PlaneHammersley((float*)&samplePos[0], spp);

  IntegratorCommon* pTracer = dynamic_cast<IntegratorCommon*>(m_pIntegrator);
  if (pTracer == nullptr)
    return;

  for (int y = 0; y < m_height; y++)
  {
    std::vector<float4>  normals(m_width*spp);
    std::vector<float4>  txcolor(m_width*spp);
    std::vector<ushort2> pixels(m_width);

    #pragma omp parallel for
    for (int x = 0; x < m_width; x++)
    {
      float4 avgNorm(0, 0, 0, 0);
      float4 avgColor(0, 0, 0, 0);
      float  avgAlpha = 0.0f;
      float  avgDepth = 0.0f;

      for (int s = 0; s < spp; s++)
      {
        float3 ray_pos, ray_dir;
        std::tie(ray_pos, ray_dir) = pTracer->makeEyeRaySubpixel(x, y, samplePos[s]);

        auto hit = pTracer->rayTrace(ray_pos, ray_dir);

        float alphaCurr = 0.0f;
        float depthCurr = 0.0f;

        if (HitSome(hit))
        {
          normals[x*spp + s] = make_float4(0, 0, 0, 1000000000.0f);
          txcolor[x*spp + s] = make_float4(0, 0, 0, 0);
          alphaCurr          = 0.0f;
          depthCurr          = 0.0f;
        }
        else
        {
          auto   surf   = pTracer->surfaceEval(ray_pos, ray_dir, hit);
          float3 transp = pTracer->evalAlphaTransparency(ray_pos, ray_dir, surf, 1);

          normals[x*spp + s] = to_float4(surf.normal, hit.t);
          txcolor[x*spp + s] = to_float4(pTracer->evalDiffuseColor(ray_dir, surf), 0.0f);
          alphaCurr          = 1.0f - dot(transp, make_float3(0.35f, 0.51f, 0.14f));
          depthCurr          = hit.t;
        }

        avgNorm  += normals[x*spp + s];
        avgColor += txcolor[x*spp + s];
        avgAlpha += alphaCurr;
        avgDepth += depthCurr;
      }

      float invSPP = 1.0f / float(spp);

      pixels[x]              = ushort2(x, y);
      a_norms[y*m_width + x] = avgNorm*invSPP;
      a_txcol[y*m_width + x] = avgColor*invSPP;

      if (isGBuffer)
      {
        GBuffer1 buffPx;
        buffPx.depth = avgDepth*invSPP;
        buffPx.norm  = to_float3(avgNorm)*invSPP;
        buffPx.matId = 0;
        buffPx.rgba  = to_float4(to_float3(avgColor*invSPP), avgAlpha*invSPP);
        a_norms[y*m_width + x] = packGBuffer1(buffPx);
      }

    }

    // float4 ffwh = make_float4(m_vars.m_varsF[HRT_FOV_X], m_vars.m_varsF[HRT_FOV_Y], m_vars.m_varsF[HRT_WIDTH_F], m_vars.m_varsF[HRT_HEIGHT_F]);
   
  }

  std::cout << "[cpu_core]: end gbuffer render " << std::endl;

}


MRaysStat CPUExpLayer::GetRaysStat()
{
  MRaysStat res;
  return res;
}


unsigned int CPUExpLayer::AddLightMesh(LightMeshData a_lmesh)
{
  return 0; // return light mesh index that was just added
}

void CPUExpLayer::ResetPerfCounters()
{

}

void CPUExpLayer::BeginTracingPass()
{
  m_pIntegrator->DoPass(m_tempImage);
  //m_pIntegrator->TracePrimary(m_tempImage);
  //m_pIntegrator->TraceForTest(m_tempImage);
}

void CPUExpLayer::EndTracingPass()
{

}


IHWLayer* CreateCPUExpImpl(int w, int h, int a_flags) { return new CPUExpLayer(w, h, a_flags); }


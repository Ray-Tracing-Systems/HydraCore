#include "RenderDriverRTE.h"
#pragma warning(disable:4996) // for wcsncpy to be ok

#include <iostream>
#include <queue>
#include <string>
#include <regex>
#include <chrono>

#include "../../HydraAPI/hydra_api/HydraXMLHelpers.h"
#include "../../HydraAPI/hydra_api/HydraInternal.h"

#include "../../HydraAPI/hydra_api/vfloat4_x64.h"

#ifdef WIN32
#undef min
#undef max
#endif

constexpr bool MEASURE_RAYS   = false;
constexpr int  MEASURE_BOUNCE = 0;

extern bool g_exitDueToSamplesLimit;

void UpdateProgress(const wchar_t* a_message, float a_progress)
{
  fwprintf(stdout, L"%s: %.0f%%            \r", a_message, a_progress*100.0f);
  fflush(stdout);
}

#ifndef WIN32
extern "C" IBVHBuilder2* CreateBuilder2(const char* cfg);
#endif

RenderDriverRTE::RenderDriverRTE(const wchar_t* a_options, int w, int h, int a_devId, int a_flags, IHRSharedAccumImage* a_sharedImage) : m_pBVH(nullptr), m_pHWLayer(nullptr), m_pSysMutex(nullptr),
                                                                                                                                         m_pTexStorage(nullptr), m_pTexStorageAux(nullptr), 
                                                                                                                                         m_pGeomStorage(nullptr), m_pMaterialStorage(nullptr), 
                                                                                                                                         m_pPdfStorage(nullptr), m_pAccumImage(nullptr)
{
  m_alreadyDeleted = false;
  m_msg = L"";

  m_camera.fov       = 45.0f;
  m_camera.nearPlane = 0.1f;
  m_camera.farPlane  = 1000.0f;
  m_camera.pos.x     = 0.0f; m_camera.pos.y    = 0.0f; m_camera.pos.z    = 0.0f;
  m_camera.lookAt.x  = 0.0f; m_camera.lookAt.y = 0.0f; m_camera.lookAt.z = -1.0f;

  m_width  = w;
  m_height = h;
  m_devId  = a_devId;

  if (a_devId < 0)
    m_initFlags = 0;
  else
	  m_initFlags = GPU_RT_HW_LAYER_OCL;    

  m_useConvertedLayout      = false || (m_initFlags & GPU_RT_HW_LAYER_OCL);
  m_useBvhInstInsert        = false;
  m_texShadersWasRecompiled = false;

  if (MEASURE_RAYS)
    m_initFlags |= GPU_RT_MEMORY_FULL_SIZE_MODE;

  m_initFlags |= a_flags;

  m_gpuFB        = ((m_initFlags & GPU_RT_CPU_FRAMEBUFFER) == 0);
  m_renderMethod = RENDER_METHOD_RT;
  m_ptInitDone   = false;
  m_legacy.m_lastSeed         = GetTickCount();
  std::cout << "[main]::RenderDriverRTE(): SEED = " << m_legacy.m_lastSeed << std::endl;
  m_legacy.updateProgressCall = &UpdateProgress;

  m_auxImageNumber  = 0;
  m_avgStatsId      = 0;
  m_haveAtLeastOneAOMat  = false;
  m_haveAtLeastOneAOMat2 = false;
  m_texResizeEnabled     = false;

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  if (m_initFlags & GPU_RT_HW_LAYER_OCL)
    m_pHWLayer = CreateOclImpl(m_width, m_height, m_initFlags, m_devId);
  else
    m_pHWLayer = CreateCPUExpImpl(m_width, m_height, 0);
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  m_pHWLayer->SetProgressBarCallback(&UpdateProgress);
 
  m_firstResizeOfScreen = true;
#ifdef WIN32
  m_pBVH = CreateBuilderFromDLL(L"bvh_builder.dll", "");
#else
  m_pBVH = CreateBuilder2("");
#endif
  
  if (m_pBVH != nullptr)
  {
    if (m_useBvhInstInsert)
      m_pBVH->Init("-allow_insert_copy 1");
    else
      m_pBVH->Init("-allow_insert_copy 0");
  }
  else
  {
    std::cerr << "can't load 'bvh_builder.dll' " << std::endl;
  }

  m_needToFreeCPUMem = (m_devId >= 0); // not a CPU device!!!
 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (a_sharedImage != nullptr)
  {
    auto pHeader = a_sharedImage->Header();

    if (pHeader->width != w || pHeader->height != h)
    {
      std::cout << "[sharedimage]: input    (w,h) = " << "(" << w << "," << h << ")" << std::endl;
      std::cout << "[sharedimage]: external (w,h) = " << "(" << pHeader->width << "," << pHeader->height << ")" << std::endl;
    }
  }
  
  m_pAccumImage = a_sharedImage;
  if ((a_flags & GPU_RT_ALLOC_INTERNAL_IMAGEB) || (a_flags & GPU_RT_CPU_FRAMEBUFFER) == 0) // light tracing or ibpt is enabled, or we need to store framebuffer on GPU for some reason. 
  {
    m_pHWLayer->SetExternalImageAccumulator(nullptr);       // we will manage image contribution on the level of render driver;
  }
  else
  {
    m_pHWLayer->SetExternalImageAccumulator(a_sharedImage); // we will manage image contribution on the level of HWLayer;
  }
  
  m_drawPassNumber       = 0;
  m_maxRaysPerPixel      = 1000000;
  m_shadowMatteBackTexId = INVALID_TEXTURE;
  m_shadowMatteBackGamma = 2.2f;
  m_pSysMutex            = hr_create_system_mutex("hydrabvh");
}

void RenderDriverRTE::ExecuteCommand(const wchar_t* a_cmd, wchar_t* a_out)
{
#ifndef WIN32
  if(std::wstring(a_cmd) == L"exitnow" && m_pHWLayer != nullptr) 
  {
    std::cerr << "[RTE], exitnow" << std::endl;
    //m_pHWLayer->FinishAll();
    //std::cerr << "[RTE], exitnow, after  FinishAll" << std::endl;
    exit(0);
  }
#endif
}

bool RenderDriverRTE::UpdateSettings(pugi::xml_node a_settingsNode)
{
  const int oldWidth  = m_width;
  const int oldHeight = m_height;

  if (a_settingsNode.child(L"width") != nullptr)
    m_width = a_settingsNode.child(L"width").text().as_int();

  if (a_settingsNode.child(L"height") != nullptr)
    m_height = a_settingsNode.child(L"height").text().as_int();

  if (m_width < 0 || m_height < 0)
  {
    m_msg = L"RenderDriverRTE::UpdateSettings, bad input resolution";
    return false;
  }

  if (m_pHWLayer == nullptr)
    return false;
  
  m_maxRaysPerPixel = a_settingsNode.child(L"maxRaysPerPixel").text().as_int();

  if (oldWidth != m_width || oldHeight != m_height || m_firstResizeOfScreen)
  {
    int flags = MEASURE_RAYS ? GPU_RT_MEMORY_FULL_SIZE_MODE : 0;
    m_pHWLayer->ResizeScreen(m_width, m_height, (flags | m_initFlags));
    m_firstResizeOfScreen = false;
  }

  //
  //
  auto vars = m_pHWLayer->GetAllFlagsAndVars();

  vars.m_flags |= (HRT_USE_MIS | HRT_COMPUTE_SHADOWS);

  if ((m_initFlags & GPU_ALLOC_FOR_COMPACT_MLT) || (m_initFlags & GPU_MLT_ENABLED_AT_START))
  {
    vars.m_flags |= HRT_ENABLE_MMLT;
    m_renderMethod = RENDER_METHOD_MMLT;
  }
  else if(std::wstring(a_settingsNode.child(L"method_secondary").text().as_string()) == L"mmlt" || 
          std::wstring(a_settingsNode.child(L"method_secondary").text().as_string()) == L"MMLT" ||
          std::wstring(a_settingsNode.child(L"method_secondary").text().as_string()) == L"mlt")
  {
    vars.m_flags |= HRT_ENABLE_MMLT;
    m_renderMethod = RENDER_METHOD_MMLT;
  }
  else if (std::wstring(a_settingsNode.child(L"method_caustic").text().as_string()) == L"none" ||
           std::wstring(a_settingsNode.child(L"method_caustic").text().as_string()) == L"disabled")
  {
    m_renderMethod = RENDER_METHOD_PT;
    vars.m_flags   = vars.m_flags & (~HRT_ENABLE_PT_CAUSTICS);
  }
  else
  {
    vars.m_flags |= HRT_ENABLE_PT_CAUSTICS;
  }

  //vars.m_flags |= HRT_ENABLE_PT_CAUSTICS;
  //vars.m_flags |= HRT_STUPID_PT_MODE;

  vars.m_varsI[HRT_TRACE_DEPTH]            = 6;
  vars.m_varsI[HRT_DIFFUSE_TRACE_DEPTH]    = 3;

  if (MEASURE_RAYS)
  {
    vars.m_varsI[HRT_TRACE_DEPTH]         = 5;
    vars.m_varsI[HRT_DIFFUSE_TRACE_DEPTH] = 3;
  }

  vars.m_varsI[HRT_ENABLE_PATH_REGENERATE] = 1;
  vars.m_varsI[HRT_RENDER_LAYER]           = LAYER_COLOR;

  if (MEASURE_RAYS)
  {
    vars.m_varsI[HRT_MEASURE_RAYS_TYPE]      = MEASURE_BOUNCE;
    vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS]  = 1;
    vars.m_varsI[HRT_ENABLE_PATH_REGENERATE] = 0;
  }

  vars.m_varsF[HRT_IMAGE_GAMMA]      = 2.2f;
  vars.m_varsF[HRT_TEXINPUT_GAMMA]   = 2.2f;
  vars.m_varsF[HRT_BSDF_CLAMPING]    = 1e6f;
  vars.m_varsF[HRT_ENV_CLAMPING]     = 1e6f;
  vars.m_varsF[HRT_PATH_TRACE_ERROR] = 0.025f;
  vars.m_varsF[HRT_TRACE_PROCEEDINGS_TRESHOLD] = 1e-8f;

  if(a_settingsNode.child(L"mmlt_burn_iters") != nullptr)
    vars.m_varsI[HRT_MMLT_BURN_ITERS] = a_settingsNode.child(L"mmlt_burn_iters").text().as_int();
  else
    vars.m_varsI[HRT_MMLT_BURN_ITERS] = 1024;

  if(a_settingsNode.child(L"mmlt_sds_fixed_prob") != nullptr)
    vars.m_varsF[HRT_MMLT_IMPLICIT_FIXED_PROB] = clamp(a_settingsNode.child(L"mmlt_sds_fixed_prob").text().as_float(), 0.0f, 0.95f);
  else
    vars.m_varsF[HRT_MMLT_IMPLICIT_FIXED_PROB] = 0.0f;

  vars.m_varsF[HRT_MMLT_STEP_SIZE_POWER] = 1024.0f; // (512, 1024, 2048)  -- 512 is large step, 2048 is small
  vars.m_varsF[HRT_MMLT_STEP_SIZE_COEFF] = 1.0f;    // (1.0f, 1.5f, 2.0f) -- 1.0f is normal step, 2.0f is small
  vars.m_varsI[HRT_MMLT_FIRST_BOUNCE]    = 3; 

  // override default settings from scene settings
  //
  if (a_settingsNode.child(L"method_primary") != nullptr)
  {
    const std::wstring method = std::wstring(a_settingsNode.child(L"method_primary").text().as_string());
    if (method == L"IBPT" || method == L"ibpt")
    {
      m_renderMethod = RENDER_METHOD_IBPT;
    }
    else if (method == L"SBPT" || method == L"sbpt")
    {
      m_renderMethod = RENDER_METHOD_SBPT;
    }
    else if (method == L"pathtracing" || method == L"pt" || method == L"PT")
    {
      if(m_initFlags & GPU_MLT_ENABLED_AT_START)
        m_renderMethod = RENDER_METHOD_MMLT;
      else
        m_renderMethod = RENDER_METHOD_PT;
    }
    else if (method == L"lighttracing" || method == L"lt" || method == L"LT")
    {
      m_renderMethod = RENDER_METHOD_LT;
    }
    else if (method == L"mlt" || method == L"mmlt" || method == L"MMLT")
    {
      m_renderMethod = RENDER_METHOD_MMLT;
    }
    else
    {
      m_renderMethod = RENDER_METHOD_RT;
    }
  }
  else
  {
    m_renderMethod = RENDER_METHOD_PT;
  }

  if (a_settingsNode.child(L"trace_depth") != nullptr)
    vars.m_varsI[HRT_TRACE_DEPTH] = a_settingsNode.child(L"trace_depth").text().as_int();

  if (a_settingsNode.child(L"diff_trace_depth") != nullptr)
    vars.m_varsI[HRT_DIFFUSE_TRACE_DEPTH] = a_settingsNode.child(L"diff_trace_depth").text().as_int();

  if (a_settingsNode.child(L"pt_error") != nullptr)
    vars.m_varsF[HRT_PATH_TRACE_ERROR] = 0.01f*a_settingsNode.child(L"pt_error").text().as_float();

  if (a_settingsNode.child(L"minRaysPerPixel") != nullptr)
    m_legacy.minRaysPerPixel = a_settingsNode.child(L"minRaysPerPixel").text().as_int();

  if (a_settingsNode.child(L"maxRaysPerPixel") != nullptr)
  {
    m_legacy.maxRaysPerPixel = a_settingsNode.child(L"maxRaysPerPixel").text().as_int();
    vars.m_varsI[HRT_MAX_SAMPLES_PER_PIXEL] = m_legacy.maxRaysPerPixel;
  }
  if (a_settingsNode.child(L"seed") != nullptr)
    m_legacy.m_lastSeed = a_settingsNode.child(L"seed").text().as_int();

  if(m_initFlags & GPU_RT_DO_NOT_PRINT_PASS_NUMBER)
    vars.m_varsI[HRT_SILENT_MODE] = 1;

  if (vars.m_flags & HRT_STUPID_PT_MODE)
    vars.m_varsI[HRT_TRACE_DEPTH]++;

  if (a_settingsNode.child(L"evalgbuffer") != nullptr)
    vars.m_varsI[HRT_STORE_SHADOW_COLOR_W] = a_settingsNode.child(L"evalgbuffer").text().as_int();
  else
    vars.m_varsI[HRT_STORE_SHADOW_COLOR_W] = 0;

  // production pt settings
  //
  if(a_settingsNode.child(L"boxmode") != nullptr)
    vars.m_varsI[HRT_BOX_MODE_ON] = a_settingsNode.child(L"boxmode").text().as_int();
  else
    vars.m_varsI[HRT_BOX_MODE_ON] = 0;

  if(a_settingsNode.child(L"offline_pt") != nullptr)
  {
    int mode = a_settingsNode.child(L"offline_pt").text().as_int();
    if(mode == 1)
      vars.m_flags |= HRT_PRODUCTION_IMAGE_SAMPLING;
    else
      vars.m_flags = vars.m_flags & ~HRT_PRODUCTION_IMAGE_SAMPLING;

    vars.m_varsI[HRT_CONTRIB_SAMPLES] = a_settingsNode.child(L"contribsamples").text().as_int();
  }
  else
  {
    vars.m_flags = vars.m_flags & ~HRT_PRODUCTION_IMAGE_SAMPLING;
    vars.m_varsI[HRT_CONTRIB_SAMPLES] = 1000000;
  }

  if(a_settingsNode.child(L"qmc_variant") != nullptr)
    vars.m_varsI[HRT_QMC_VARIANT] = a_settingsNode.child(L"qmc_variant").text().as_int();
  else  
    vars.m_varsI[HRT_QMC_VARIANT] = 0;

  m_pHWLayer->SetAllFlagsAndVars(vars);

  return true;
}


RenderDriverRTE::~RenderDriverRTE()
{
  if (m_alreadyDeleted)
    return;

  ClearAll();

  if (m_pHWLayer != nullptr)
  {
    delete m_pHWLayer;
    m_pHWLayer = nullptr;
  }

  if (m_pBVH != nullptr)
  {
    m_pBVH->Destroy();
    m_pBVH = nullptr;
  }

  if(m_pSysMutex != nullptr)
  {
    hr_free_system_mutex(m_pSysMutex);
    m_pSysMutex = nullptr;
  }
  
  m_alreadyDeleted = true;
}

void RenderDriverRTE::Error(const wchar_t* a_msg)
{
  m_msg = std::wstring(a_msg);
}

void RenderDriverRTE::ClearAll()
{
  if (m_pHWLayer == nullptr)
    return;

  m_pHWLayer->FinishAll();

  m_pHWLayer->MLT_Free();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (m_pTexStorage != nullptr)
  {
    delete m_pTexStorage;
    m_pTexStorage = nullptr;
  }

  if (m_pTexStorageAux != nullptr)
  {
    delete m_pTexStorageAux;
    m_pTexStorageAux = nullptr;
  }

  if (m_pGeomStorage != nullptr)
  {
    delete m_pGeomStorage;
    m_pGeomStorage = nullptr;
  }

  if (m_pMaterialStorage != nullptr)
  {
    delete m_pMaterialStorage;
    m_pMaterialStorage = nullptr;
  }

  if (m_pPdfStorage != nullptr)
  {
    delete m_pPdfStorage;
    m_pPdfStorage = nullptr;
  }

  m_geomTable.clear();
  m_texTable.clear();
  m_texTableAux.clear();
  m_materialTable.clear();

  m_lights.clear();
  m_lightsInstanced.clear();
  m_lightHavePdfTable.clear();
  m_iesCache.clear();
  m_materialUpdated.clear();
  m_materialNodes.clear();
  m_blendsToUpdate.clear();
  m_texturesProcessedNM.clear();
  m_procTextures.clear();

  m_instMatricesInv.clear();
  m_instLightInstId.clear();
  m_meshIdByInstId.clear();
  m_instIdByInstId.clear();
  m_meshRemapListId.clear();

  m_auxImageNumber = 0;
  m_auxTexNormalsPerMat.clear();

  m_haveAtLeastOneAOMat  = false;
  m_haveAtLeastOneAOMat2 = false;
}

std::shared_ptr<RAYTR::IMaterial> CreateDiffuseWhiteMaterial();

std::tuple<size_t, size_t> EstimateMemNeeded(const std::vector<HRTexResInfo>& a_texuresInfo)
{
  size_t memCommon = 0;
  size_t memBump   = 0;

  for (auto texInfo : a_texuresInfo)
  {
    size_t txSize = size_t(texInfo.aw*texInfo.ah)*size_t(texInfo.bpp);

    memCommon += txSize;
    if (texInfo.usedAsBump)
      memBump += txSize;
  }

  return std::tuple<size_t, size_t>(memCommon, memBump);
}

int ResizeMostHeavyTexture(std::vector<HRTexResInfo>& a_texuresInfo, bool bump_only)
{
  size_t currSize = 0;
  int currId = 0;
  bool foundAtLeastOne = false;

  // find most heavy one
  //
  for (int i = 0; i<int(a_texuresInfo.size()); i++)
  {
    const auto& texInfo = a_texuresInfo[i];
    size_t txSize = size_t(texInfo.aw*texInfo.ah)*size_t(texInfo.bpp);

    const bool accountIt = !bump_only || texInfo.usedAsBump;
    const bool canResize = (texInfo.aw > texInfo.rw) && (texInfo.ah > texInfo.rh);

    if (txSize > currSize && accountIt && canResize)
    {
      currSize = txSize;
      currId   = i;
      foundAtLeastOne = true;
    }
  }

  if (!foundAtLeastOne)
    return -1;

  a_texuresInfo[currId].aw /= 2;
  a_texuresInfo[currId].ah /= 2;

  return currId;

}

/**
\brief allow to resize less textures
\param a_texuresInfo - in out texture res info
\param in_memToFit - memory amount we have to fit all our textures
\param in_memToFitBump - memory amount we have to fit all our textures that used for bump

Change recomended tex resolution in the way that render shout not resize more that it really needed.
For example if we have enough memory both for geom and full res textures, use full res textures.

*/
void FitTextureRes(std::vector<HRTexResInfo>& a_texuresInfo, size_t in_memToFit, size_t in_memToFitBump)
{
  if (a_texuresInfo.size() == 0)
    return;

  size_t memCommon = 0;
  size_t memBump   = 0;

  std::tie(memCommon, memBump) = EstimateMemNeeded(a_texuresInfo);

  const int maxItrer = a_texuresInfo.size()*3; // max mip level = 4, so 3 times resize for each texture
  int iterNum = 0;

  if (memCommon <= in_memToFit && memBump <= in_memToFitBump)
    return;

  while(true)
  {
    int resizedId2 = -1;
    int resizedId = ResizeMostHeavyTexture(a_texuresInfo, false);
    if (resizedId >= 0)
    {
      if (!a_texuresInfo[resizedId].usedAsBump && memBump > in_memToFitBump)
        resizedId2 = ResizeMostHeavyTexture(a_texuresInfo, true);
    }
    else if(memBump > in_memToFitBump)
      resizedId2 = ResizeMostHeavyTexture(a_texuresInfo, true);

    const bool commonOk = (memCommon <= in_memToFit)     || resizedId == -1;
    const bool auxOk    = (memBump   <= in_memToFitBump) || resizedId2 == -1;

    if ((commonOk && auxOk) || iterNum > maxItrer)
      break;

    std::tie(memCommon, memBump) = EstimateMemNeeded(a_texuresInfo);
    iterNum++;
  }
}

HRDriverAllocInfo RenderDriverRTE::AllocAll(HRDriverAllocInfo a_info)
{
  m_libPath = std::wstring(a_info.libraryPath);

  const size_t maxBufferSize = m_pHWLayer->GetMaxBufferSizeInBytes();
  const size_t totalMem      = m_pHWLayer->GetAvaliableMemoryAmount(true);
  const size_t freeMem       = m_pHWLayer->GetAvaliableMemoryAmount(false);
  const size_t memUsedByR    = totalMem - freeMem;
  const size_t MB            = size_t(1024 * 1024);

  // create memory storages and tables
  //
  const size_t approxSizeOfMatBlock = sizeof(PlainMaterial) * 8;
  const size_t approxSizeOfLight    = sizeof(PlainLight)    * 4;

  m_allTexInfo.clear(); 

  if (a_info.imgResInfoArray != nullptr)
  {
    std::vector<HRTexResInfo> allTexInfoVec;
    allTexInfoVec.reserve(a_info.imgNum);

    for (int i = 0; i < a_info.imgNum; i++)
    {
      const auto& info = a_info.imgResInfoArray[i];
      if (info.w != 0 && info.h != 0)
        allTexInfoVec.push_back(info);
    }

    const int64_t geomMem      = int64_t( std::min(size_t(a_info.geomMem), maxBufferSize));
    const double perCentCommon = double(a_info.imgMem)    / double(a_info.imgMemAux + a_info.imgMem);
    const double perCentAux    = double(a_info.imgMemAux) / double(a_info.imgMemAux + a_info.imgMem);

    int64_t memRestTotal = std::max(int64_t(freeMem) - geomMem, int64_t(0));
    int64_t memRest      = int64_t(perCentCommon*double(memRestTotal)) - 1*MB;
    int64_t memRest2     = int64_t(perCentAux*double(memRestTotal)) - 1*MB;
 
    if (memRest <= int64_t(1*MB))
      memRest = 16 * MB;

    if (memRest2 <= int64_t(1*MB))
      memRest2 = 16 * MB;

    const int64_t memForTex    = std::min(std::min(memRest,  int64_t(maxBufferSize)), a_info.imgMem    + int64_t(16*MB));
    const int64_t memForTex2   = std::min(std::min(memRest2, int64_t(maxBufferSize)), a_info.imgMemAux + int64_t(16*MB));

    FitTextureRes(allTexInfoVec, size_t(memForTex), size_t(memForTex2));
    for (auto info : allTexInfoVec)  
    {
      m_allTexInfo[info.id] = info;
      std::cout << info.id << ": " << info.aw << " " << info.ah << std::endl;  
    }
  }

  size_t auxMemGeom   = 0, auxMemTex = 64 * MB;
  size_t newMemForGeo = a_info.geomMem; // size_t(0.85*double(a_info.geomMem)); // we can save ~ 15% due to tangent compression but thhis is hard to estimate precisly.
  size_t newMemForMat = a_info.matNum*approxSizeOfMatBlock;
  size_t newMemForTab = (MAX_ENV_LIGHT_PDF_SIZE*MAX_ENV_LIGHT_PDF_SIZE)*sizeof(float) + 4*MB;

  newMemForTab += a_info.lightsWithIESNum * 1 * MB;
  if (newMemForTab > 64 * MB)
    newMemForTab = 64 * MB;

  size_t newMemForTex1 = auxMemTex + a_info.imgMem;
  size_t newMemForTex2 = auxMemTex + a_info.imgMemAux;
  size_t newMemForTex3 = newMemForTex1 + newMemForTex2;
  size_t newTotalMem   = newMemForTex3 + newMemForGeo + newMemForMat + newMemForTab;

  if (newTotalMem >= freeMem)
  {
    newMemForTex1 = auxMemTex + a_info.imgMem;
    newMemForTex2 = auxMemTex + a_info.imgMem/2;
    newMemForTex3 = newMemForTex1 + newMemForTex2;
    newTotalMem   = newMemForTex3 + newMemForGeo + newMemForMat + newMemForTab;
  }

  if (newTotalMem >= freeMem)
  {
    newMemForTex1 = auxMemTex + a_info.imgMem;
    newMemForTex2 = auxMemTex + 2*a_info.imgMem/3;
    newMemForTex3 = newMemForTex1 + newMemForTex2;
    newTotalMem   = newMemForTex3 + newMemForGeo + newMemForMat + newMemForTab;
  }

  while (newTotalMem >= freeMem)
  {
    newMemForTex1 -= 10*MB;
    newMemForTex2 -= 5*MB;
    newMemForTex3 = newMemForTex1 + newMemForTex2;
    newTotalMem   = newMemForTex3 + newMemForGeo + newMemForMat + newMemForTab;
  }

  if (true) // newMemForTex1 > maxBufferSize || newMemForTex2 > maxBufferSize
    m_texResizeEnabled = true;

  newMemForTex1      = std::min<size_t>(newMemForTex1, maxBufferSize);
  newMemForTex2      = std::min<size_t>(newMemForTex2, maxBufferSize);
  newMemForGeo       = std::min<size_t>(newMemForGeo,  maxBufferSize);
  newMemForMat       = std::min<size_t>(newMemForMat,  maxBufferSize);
  newMemForTab       = std::min<size_t>(newMemForTab,  maxBufferSize);

  m_pTexStorage      = m_pHWLayer->CreateMemStorage(newMemForTex1, "textures");     // #TODO:  estimate this more carefully pls.
  m_pTexStorageAux   = m_pHWLayer->CreateMemStorage(newMemForTex2, "textures_aux"); // #TODO:  estimate this more carefully pls.
  m_pGeomStorage     = m_pHWLayer->CreateMemStorage(newMemForGeo,  "geom");         // #TODO:  estimate this more carefully pls.
  m_pMaterialStorage = m_pHWLayer->CreateMemStorage(newMemForMat,  "materials");
  m_pPdfStorage      = m_pHWLayer->CreateMemStorage(newMemForTab,  "pdfs");         // #TODO:  estimate this more carefully pls.

  m_pHWLayer->ResizeTablesForEngineGlobals(a_info.geomNum, a_info.imgNum, a_info.matNum, a_info.lightNum);

  m_instMatricesInv.reserve(a_info.geomNum); // note this is only reserve !!!, not resize !!!
  m_lights.resize(a_info.lightNum);

  // add some gummy data to the compute core
  //
  auto pMaterial = CreateDiffuseWhiteMaterial();
  auto plainData = pMaterial->ConvertToPlainMaterial();

  const int32_t maxMaterialIndex   = a_info.matNum-1;
  const int32_t whiteDiffuseOffset = m_pMaterialStorage->Update(maxMaterialIndex, &plainData[0], plainData.size()*sizeof(PlainMaterial));


  auto vars = m_pHWLayer->GetAllFlagsAndVars();
  vars.m_varsI[HRT_WHITE_DIFFUSE_OFFSET] = whiteDiffuseOffset;
  m_pHWLayer->SetAllFlagsAndVars(vars);

  m_memAllocated = 0;
  std::cout << std::endl;
  std::cout << "[AllocAll]: MEM(TEXURE) = " << newMemForTex3 / MB << "\tMB" << std::endl; m_memAllocated += newMemForTex3;
  std::cout << "[AllocAll]: MEM(GEOM)   = " << newMemForGeo / MB << "\tMB" << std::endl;  m_memAllocated += newMemForGeo;
  std::cout << "[AllocAll]: MEM(PDFTAB) = " << (newMemForMat + newMemForTab) / MB << "\tMB" << std::endl; m_memAllocated += (newMemForMat + newMemForTab);
  std::cout << "[AllocAll]: MEM(RAYBUF) = " << memUsedByR / MB << "\tMB" << std::endl;    m_memAllocated += memUsedByR;

  if (newTotalMem >= freeMem)
  {
    std::cerr << "[AllocAll]: NOT ENOUGHT MEMORY! --- " << std::endl;
  }

  m_lastAllocInfo         = a_info;
  m_lastAllocInfo.geomMem = newMemForGeo;
  m_lastAllocInfo.imgMem  = newMemForTex3;

  return m_lastAllocInfo;
}

void RenderDriverRTE::GetLastErrorW(wchar_t a_msg[256])
{
  wcsncpy(a_msg, m_msg.c_str(), 256);
  m_msg = L"";
}

bool RenderDriverRTE::UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode)
{
  std::wstring type = a_texNode.attribute(L"type").as_string();

  if (type == L"proc")
    return UpdateImageProc(a_texId, w, h, bpp, a_data, a_texNode);

  if (a_data == nullptr)
    return false;

  std::vector<uint32_t> dataResizedI;
  HDRImage4f dst;

  if (m_texResizeEnabled && a_texNode.attribute(L"rwidth") != nullptr && a_texNode.attribute(L"rheight") != nullptr)
  {
    int rwidth  = a_texNode.attribute(L"rwidth").as_int();
    int rheight = a_texNode.attribute(L"rheight").as_int();

    auto p = m_allTexInfo.find(a_texId);
    if (p != m_allTexInfo.end())
    {
      rwidth  = p->second.aw;
      rheight = p->second.ah;
    }

    if (rwidth < w || rheight < h)
    {
      std::cout << "resize tex id = " << a_texId << " from (" << w << "," << h << ") to (" << rwidth << "," << rheight << ")" << std::endl;

      HDRImage4f src;

      if (bpp == 4)
      {
        src.resize(w, h);
        src.convertFromLDR(2.2f, (const unsigned int*)a_data, w*h);
      }
      else
        src = HDRImage4f(w, h, (const float*)a_data);

      dst.resize(rwidth, rheight);
      src.resampleTo(dst);

      if (bpp == 4)
      {
        dataResizedI.resize(rwidth*rheight);
        dst.convertToLDR(2.2f, dataResizedI);
        a_data = dataResizedI.data();
      }
      else
        a_data = dst.data();

      w = rwidth;
      h = rheight;
    }

  }

  SWTextureHeader texheader;

  texheader.width  = w;
  texheader.height = h;
  texheader.depth  = 1;
  texheader.bpp    = bpp;

  const size_t inDataBSz  = size_t(w)*size_t(h)*size_t(bpp);
  const int    align      = int(m_pTexStorage->GetAlignSizeInBytes());
  const size_t headerSize = roundBlocks(sizeof(SWTextureHeader), align);
  const size_t totalSize  = roundBlocks(inDataBSz, align) + headerSize;

  auto offset = m_pTexStorage->Update(a_texId, nullptr, totalSize);

  if (offset == -1)
  {
    std::cerr << "RenderDriverRTE::UpdateImage: can't append texture to tex storage; id = " << a_texId << std::endl;
    return false;
  }

  m_pTexStorage->UpdatePartial(a_texId, &texheader, 0, sizeof(SWTextureHeader));
  m_pTexStorage->UpdatePartial(a_texId, a_data, headerSize, inDataBSz);

  return true;
}

std::shared_ptr<RAYTR::IMaterial> CreateMaterialFromXmlNode(pugi::xml_node a_node, RenderDriverRTE* a_pRTE);

using ProcTexInfo = RenderDriverRTE::ProcTexInfo;

bool MaterialNodeHaveProceduralTextures(pugi::xml_node a_node, const std::unordered_map<int, ProcTexInfo>& a_ids, const std::unordered_map<int, pugi::xml_node >& a_matNodes);
void FindAllProcTextures(pugi::xml_node a_node, const std::unordered_map<int, ProcTexInfo>& a_ids, const std::unordered_map<int, pugi::xml_node >& a_matNodes,
                         std::vector< std::tuple<int, ProcTexInfo> >& a_outVector);
ProcTextureList MakePTListFromTupleArray(const std::vector<std::tuple<int, ProcTexInfo> >& procTextureIds);

void ReadAllProcTexArgsFromMaterialNode(pugi::xml_node a_node, const std::unordered_map<int, pugi::xml_node >& a_matNodes, 
                                        std::vector<ProcTexParams>& a_procTexParams);

void PutTexParamsToMaterialWithDamnTable(std::vector<ProcTexParams>& a_procTexParams, const std::unordered_map<int, ProcTexInfo>& a_allProcTextures,
                                         std::shared_ptr<RAYTR::IMaterial> a_pMaterial);

void PutAOToMaterialHead(const std::vector< std::tuple<int, ProcTexInfo> >& a_procTextureIds, std::shared_ptr<RAYTR::IMaterial> a_pMaterial);
void OverrideAOInMaterialHead(pugi::xml_node a_materialNode, const std::unordered_map<int, pugi::xml_node >& a_matNodes, 
                              std::shared_ptr<RAYTR::IMaterial> a_pMaterial);

bool RenderDriverRTE::UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode)
{
  //std::cerr << "RenderDriverRTE::UpdateMaterial(" << a_matId << ") " << std::endl;

  const std::wstring mtype = a_materialNode.attribute(L"type").as_string();

  if (a_matId >= m_lastAllocInfo.matNum - 1) // because we reserve one material for white diffuse dummy
  {
    Error(L"RenderDriverRTE::UpdateMaterial: can't update reserved material number %d", a_matId);
    return false;
  }

  std::shared_ptr<RAYTR::IMaterial> pMaterial = CreateMaterialFromXmlNode(a_materialNode, this); 

  if (pMaterial == nullptr)
  {
    Error(L"RenderDriverRTE::UpdateMaterial, nullptr pMaterial");
    return false;
  }

  if (MaterialNodeHaveProceduralTextures(a_materialNode, m_procTextures, m_materialNodes))
  { 
    pMaterial->AddFlags(PLAIN_MATERIAL_HAVE_PROC_TEXTURES);

    // list all proc textures bound to this material
    //
    std::vector< std::tuple<int, ProcTexInfo> > procTextureIds;
    FindAllProcTextures(a_materialNode, m_procTextures, m_materialNodes,
                        procTextureIds);
    
    ProcTextureList ptl = MakePTListFromTupleArray(procTextureIds);
    PutProcTexturesIdListToMaterialHead(&ptl, &pMaterial->m_plain);

    // prepare argumens for reading them inside procedural textures kernel
    //
    std::vector<ProcTexParams> procTexParams;
    ReadAllProcTexArgsFromMaterialNode(a_materialNode, m_materialNodes, 
                                       procTexParams);

    PutTexParamsToMaterialWithDamnTable(procTexParams, m_procTextures, 
                                        pMaterial);

    // put AO params to material head
    //
    PutAOToMaterialHead(procTextureIds, pMaterial);            // read initial ao params from proc texture
    OverrideAOInMaterialHead(a_materialNode, m_materialNodes,  // override ao params; take new values from first ao node inside 'a_materialNode'
                             pMaterial); 

    if (MaterialHaveAO(&pMaterial->m_plain))
      m_haveAtLeastOneAOMat = true;

    if (MaterialHaveAO2(&pMaterial->m_plain))
      m_haveAtLeastOneAOMat2 = true;
  }

  m_materialUpdated[a_matId] = pMaterial; // remember that we have updates this material in current update phase (between BeginMaterialUpdate and EndMaterialUpdate)
  m_materialNodes  [a_matId] = a_materialNode;

  if (mtype == L"hydra_blend")  // put blend material to storage later
  {
    m_blendsToUpdate[a_matId] = DefferedMaterialDataTuple(pMaterial, a_materialNode);
    return true;
  }
  else                          // else, put material to storage right now
    return PutAbstractMaterialToStorage(a_matId, pMaterial, a_materialNode);
}


std::shared_ptr<RAYTR::ILight> CreateLightFromXmlNode(pugi::xml_node a_node, IMemoryStorage* a_storage, std::unordered_map<std::wstring, int2>& a_iesCache, const std::wstring& a_libPath, const PlainMesh* pLightMeshHeader);

bool RenderDriverRTE::UpdateLight(int32_t a_lightId, pugi::xml_node a_lightNode)
{
  const std::wstring ltype  = a_lightNode.attribute(L"type").as_string();
  const std::wstring lshape = a_lightNode.attribute(L"shape").as_string();

  const PlainMesh* pLightMeshHeader = nullptr;
  if (lshape == L"mesh")
  {
    const int32_t meshId     = a_lightNode.attribute(L"mesh_id").as_int();
    const int4* ldata        = (const int4*)m_pGeomStorage->GetBegin();
    pLightMeshHeader         = (const PlainMesh*)(ldata + m_pGeomStorage->GetTable()[meshId]);
  }
  
  m_lights[a_lightId] = CreateLightFromXmlNode(a_lightNode, m_pPdfStorage, m_iesCache, m_libPath, pLightMeshHeader);

  if (ltype == L"sky" || (ltype == L"area" && lshape == L"cylinder"))
    UpdatePdfTablesForLight(a_lightId);

  if (ltype == L"sky")
    m_skyLightsId.insert(a_lightId);
  else
  {
    auto p = m_skyLightsId.find(a_lightId);
    if (p != m_skyLightsId.end())
      m_skyLightsId.erase(a_lightId); 
  }

  return true;
}


float NormalDiff(float3 n1, float3 n2)
{
  if (dot(n1, n2) < 0)
    n2 *= (-1.0f);

  return length(n1 - n2);
}


std::vector<float> CalcAuxShadowRaysOffsets(const HRMeshDriverInput& a_input)
{
  // (1) find box size
  //
  cvex::vfloat4 bBoxMin = {+1e30f,+1e30f,+1e30f,0.0f};
  cvex::vfloat4 bBoxMax = {-1e30f,-1e30f,-1e30f,0.0f};

  for (int triId = 0; triId < a_input.triNum; triId++) // #TODO: can opt if vectorize by vertex ... 
  {
    const int iA = a_input.indices[triId * 3 + 0];
    const int iB = a_input.indices[triId * 3 + 1];
    const int iC = a_input.indices[triId * 3 + 2];

    const cvex::vfloat4 A = cvex::load_u(a_input.pos4f + iA * 4);
    const cvex::vfloat4 B = cvex::load_u(a_input.pos4f + iB * 4);
    const cvex::vfloat4 C = cvex::load_u(a_input.pos4f + iC * 4);

    bBoxMin = cvex::min( cvex::min(bBoxMin, A), cvex::min(B, C));
    bBoxMax = cvex::max( cvex::max(bBoxMax, A), cvex::max(B, C));
  }
  
  float boxMax[4] = {0,0,0,0};
  cvex::store_u(boxMax, (bBoxMax - bBoxMin));

  const float boxMaxAtSomeAxis    = fmax(boxMax[0], fmax(boxMax[1], boxMax[2]));
  const float meshMaxShadowOffset = 1.0e-4f*boxMaxAtSomeAxis; //  0.0045f

  // (2) calc shadow offsets
  //
  std::vector<float> shadowOffsets(a_input.triNum);

  for (int triId = 0; triId < a_input.triNum; triId++)
  {
    const int iA = a_input.indices[triId * 3 + 0];
    const int iB = a_input.indices[triId * 3 + 1];
    const int iC = a_input.indices[triId * 3 + 2];

    const float3 A = float3(a_input.pos4f[iA * 4 + 0], a_input.pos4f[iA * 4 + 1], a_input.pos4f[iA * 4 + 2]);
    const float3 B = float3(a_input.pos4f[iB * 4 + 0], a_input.pos4f[iB * 4 + 1], a_input.pos4f[iB * 4 + 2]);
    const float3 C = float3(a_input.pos4f[iC * 4 + 0], a_input.pos4f[iC * 4 + 1], a_input.pos4f[iC * 4 + 2]);

    const float3 nA = float3(a_input.norm4f[iA * 4 + 0], a_input.norm4f[iA * 4 + 1], a_input.norm4f[iA * 4 + 2]);
    const float3 nB = float3(a_input.norm4f[iB * 4 + 0], a_input.norm4f[iB * 4 + 1], a_input.norm4f[iB * 4 + 2]);
    const float3 nC = float3(a_input.norm4f[iC * 4 + 0], a_input.norm4f[iC * 4 + 1], a_input.norm4f[iC * 4 + 2]);


    const float3 crpd = cross(A-B, A-C);
    const float3 fN   = normalize(crpd);

    const float normDiff = NormalDiff(fN, nA) + NormalDiff(fN, nB) + NormalDiff(fN, nC);
	
    if (normDiff > 0.001f)
      shadowOffsets[triId] = fmin(0.05f*sqrtf(length(crpd*0.5f)), meshMaxShadowOffset);
    else
      shadowOffsets[triId] = 0.0f;
    
    //const float polySize = fmax(length(A - B), fmax(length(A - C), length(B - C) ));

    //if (normDiff > 0.001f)
    //  shadowOffsets[triId] = 0.15f*fmin(normDiff, 0.15f)*polySize;
    //else
    //  shadowOffsets[triId] = 0.0f;
  }

  return shadowOffsets;
}



bool RenderDriverRTE::UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t listSize)
{
  const int align     = int(m_pGeomStorage->GetAlignSizeInBytes());
  const int alignOffs = sizeof(int) * 4;

  // (1) create and fill header with offsets
  //
  const size_t headerOffset   = 0;
  const size_t headerSize     = roundBlocks(sizeof(PlainMesh), align);

  const size_t vertPosOffset  = headerSize;
  const size_t vertPosSize    = roundBlocks(sizeof(float4)*a_input.vertNum, align);

  const size_t vertNormOffset = vertPosOffset + vertPosSize;
  const size_t vertNormSize   = roundBlocks(sizeof(float4)*a_input.vertNum, align);

  const size_t vertTexcOffset = vertNormOffset + vertNormSize;                       
  const size_t vertTexcSize   = 0; // roundBlocks(sizeof(float2)*a_input.vertNum, align);   // #TODO: add aux text coord channel if have such.
    
  const size_t vertTangOffset = vertTexcOffset + vertTexcSize;
  const size_t vertTangSize   = roundBlocks(sizeof(float4)*a_input.vertNum, align); // compressed tangent
   
  const size_t triIndOffset   = vertTangOffset + vertTangSize;
  const size_t triIndSize     = roundBlocks(a_input.triNum * 3 * sizeof(int), align);

  const size_t triMIndOffset  = triIndOffset + triIndSize;
  const size_t triMIndSize    = roundBlocks(a_input.triNum * sizeof(int), align);

  const size_t triSOffOffset  = triMIndOffset + triMIndSize;
  const size_t triSOffSize    = roundBlocks(a_input.triNum * sizeof(float), align);

  const size_t totalByteSize  = triSOffOffset + triSOffSize;

  // (1) compress tangents
  //
  const float4* tan4f  = (const float4*)a_input.tan4f;

  // (2) pack first texture coordinates to pos.w and norm.w
  //
  const float4* pos4f  = (const float4*)a_input.pos4f;
  const float4* norm4f = (const float4*)a_input.norm4f;

  std::vector<float4> posAndTx(pos4f,   pos4f  + a_input.vertNum);
  std::vector<float4> normAndTy(norm4f, norm4f + a_input.vertNum);

  for (size_t i = 0; i < a_input.vertNum; i++)
  {
    posAndTx [i].w = a_input.texcoord2f[2 * i + 0];
    normAndTy[i].w = a_input.texcoord2f[2 * i + 1];
  }

  // (3) calc per-poly shadow rays aux offset and put them to separate array.
  //
  std::vector<float> shadowOffsets = CalcAuxShadowRaysOffsets(a_input);

  // (4) put mesh to the storage
  //
  auto offset = m_pGeomStorage->Update(a_meshId, nullptr, totalByteSize); // alloc new chunk for our mesh

  if (offset == -1)
  {
    std::cerr << "RenderDriverRTE::UpdateMesh(id = " << a_meshId << ") failed. too big? :) " << std::endl;
    return false;
  }

  PlainMesh header;

  header.vPosOffset       = int(vertPosOffset  / alignOffs);
  header.vNormOffset      = int(vertNormOffset / alignOffs);
  header.vTexCoordOffset  = int(vertTexcOffset / alignOffs);                                                      //#TODO: put auxilarry tex coord channel if has such
  header.vTangentOffset   = int(vertTangOffset / alignOffs);
  header.vIndicesOffset   = int(triIndOffset   / alignOffs);
  header.mIndicesOffset   = int(triMIndOffset  / alignOffs);
  header.polyShadowOffset = int(triSOffOffset  / alignOffs);

  header.vPosNum          = a_input.vertNum;
  header.vNormNum         = a_input.vertNum;
  header.vTexCoordNum     = a_input.vertNum;                                                                       //#TODO: put auxilarry tex coord channel if has such
  header.vTangentNum      = a_input.vertNum;
  header.tIndicesNum      = a_input.triNum * 3;
  header.mIndicesNum      = a_input.triNum;
  header.totalBytesNum    = int(totalByteSize);

  if (totalByteSize > 4294967296)
  {
    std::cerr << "WARNING: RenderDriverRTE::UpdateMesh(id = " << a_meshId << ") integer overflow for mesh byte size = " << totalByteSize << std::endl;
    return false;
  }

  m_pGeomStorage->UpdatePartial(a_meshId, &header, 0, sizeof(header));
  m_pGeomStorage->UpdatePartial(a_meshId, &posAndTx[0],          vertPosOffset,  a_input.vertNum * sizeof(float4));
  m_pGeomStorage->UpdatePartial(a_meshId, &normAndTy[0],         vertNormOffset, a_input.vertNum * sizeof(float4));
  //m_pGeomStorage->UpdatePartial(a_meshId, a_input.texcoord2f,    vertTexcOffset, a_input.vertNum * sizeof(float2)); //#TODO: put auxilarry tex coord channel if has such
  m_pGeomStorage->UpdatePartial(a_meshId, tan4f, vertTangOffset, a_input.vertNum * sizeof(float4)); // compressed tangent

  m_pGeomStorage->UpdatePartial(a_meshId, a_input.indices,       triIndOffset,   a_input.triNum  * 3 * sizeof(int));
  m_pGeomStorage->UpdatePartial(a_meshId, a_input.triMatIndices, triMIndOffset,  a_input.triNum  * sizeof(int));
  m_pGeomStorage->UpdatePartial(a_meshId, &shadowOffsets[0],     triSOffOffset,  a_input.triNum  * sizeof(float));
  
  return true;
}

bool RenderDriverRTE::UpdateImageFromFile(int32_t a_texId, const wchar_t* a_fileName, pugi::xml_node a_texNode)
{ 
  return false; 
}

bool RenderDriverRTE::UpdateMeshFromFile(int32_t a_meshId, pugi::xml_node a_meshNode, const wchar_t* a_fileName)
{ 
  return false; 
}

bool RenderDriverRTE::UpdateCamera(pugi::xml_node a_camNode)
{
  if (a_camNode == nullptr)
    return true;

  m_camera.mUseMatrices = false;

  if (std::wstring(a_camNode.attribute(L"type").as_string()) == L"two_matrices")
  //if(false)
  {
    const wchar_t* m1 = a_camNode.child(L"mWorldView").text().as_string();
    const wchar_t* m2 = a_camNode.child(L"mProj").text().as_string();

    float mWorldView[16];
    float mProj[16];

    std::wstringstream str1(m1), str2(m2);
    for (int i = 0; i < 16; i++)
    {
      str1 >> mWorldView[i];
      str2 >> mProj[i];
    }

    m_camera.mProj        = float4x4(mProj);
    m_camera.mWorldView   = float4x4(mWorldView);
    m_camera.mUseMatrices = true;

    // restore FOV from projection matrix
    //
    m_camera.fov = 2.0f*atan(1.0f/m_camera.mProj.M(1, 1))*(180.f/M_PI);
  }
  else
  {
    const wchar_t* camPosStr = a_camNode.child(L"position").text().as_string();
    const wchar_t* camLAtStr = a_camNode.child(L"look_at").text().as_string();
    const wchar_t* camUpStr  = a_camNode.child(L"up").text().as_string();
    const wchar_t* testStr   = a_camNode.child(L"test").text().as_string();

    if (!a_camNode.child(L"fov").text().empty())
      m_camera.fov = a_camNode.child(L"fov").text().as_float();

    if (!a_camNode.child(L"nearClipPlane").text().empty())
      m_camera.nearPlane = a_camNode.child(L"nearClipPlane").text().as_float();

    if (!a_camNode.child(L"farClipPlane").text().empty())
      m_camera.farPlane = a_camNode.child(L"farClipPlane").text().as_float();

    if (std::wstring(camPosStr) != L"")
    {
      std::wstringstream input(camPosStr);
      input >> m_camera.pos.x >> m_camera.pos.y >> m_camera.pos.z;
    }

    if (std::wstring(camLAtStr) != L"")
    {
      std::wstringstream input(camLAtStr);
      input >> m_camera.lookAt.x >> m_camera.lookAt.y >> m_camera.lookAt.z;
    }

    if (std::wstring(camUpStr) != L"")
    {
      std::wstringstream input(camUpStr);
      input >> m_camera.up.x >> m_camera.up.y >> m_camera.up.z;
    }
  }

  auto vars = m_pHWLayer->GetAllFlagsAndVars();
  vars.m_varsF[HRT_CAM_FOV] = DEG_TO_RAD * m_camera.fov;

  if (!a_camNode.child(L"dof_lens_radius").text().empty())
    vars.m_varsF[HRT_DOF_LENS_RADIUS] = a_camNode.child(L"dof_lens_radius").text().as_float();

  vars.m_varsF[HRT_DOF_FOCAL_PLANE_DIST] = length(m_camera.pos - m_camera.lookAt);

  if (!a_camNode.child(L"enable_dof").text().empty())
    vars.m_varsI[HRT_ENABLE_DOF] = a_camNode.child(L"enable_dof").text().as_int();


  if (!a_camNode.child(L"tiltRotX").text().empty() || !a_camNode.child(L"tiltRotY").text().empty())
  {
    vars.m_varsF[HRT_TILT_ROT_Y] = a_camNode.child(L"tiltRotX").text().as_float();
    vars.m_varsF[HRT_TILT_ROT_X] = a_camNode.child(L"tiltRotY").text().as_float();
  }
  else
  {
    vars.m_varsF[HRT_TILT_ROT_Y] = 0.0f;
    vars.m_varsF[HRT_TILT_ROT_X] = 0.0f;
  }

  m_pHWLayer->SetAllFlagsAndVars(vars);

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////

HRDriverDependencyInfo RenderDriverRTE::DependencyInfo()
{ 
  HRDriverDependencyInfo res;
  res.needRedrawWhenCameraChanges = false;
  res.allowInstanceReorder        = true;
  return res; 
}

void RenderDriverRTE::CalcCameraMatrices(float4x4* a_pModelViewMatrixInv, float4x4* a_projMatrixInv, float4x4* a_pModelViewMatrix, float4x4* a_projMatrix)
{
  const float aspect = float(m_width) / float(m_height);

  float4x4 projTransposed, worldViewTransposed;

  if (m_camera.mUseMatrices)
  {
    projTransposed      = transpose(m_camera.mProj);
    worldViewTransposed = transpose(m_camera.mWorldView);
  }
  else
  {
    projTransposed      = projectionMatrixTransposed(m_camera.fov, aspect, m_camera.nearPlane, m_camera.farPlane);
    worldViewTransposed = lookAtTransposed(m_camera.pos, m_camera.lookAt, m_camera.up);
  }

  (*a_pModelViewMatrixInv)     = transpose(inverse4x4(worldViewTransposed));
  (*a_projMatrixInv)           = transpose(inverse4x4(projTransposed));
  (*a_pModelViewMatrix)        = transpose(worldViewTransposed);
  (*a_projMatrix)              = transpose(projTransposed);
}

void RenderDriverRTE::BeginScene(pugi::xml_node a_sceneNode)
{
  if (m_pBVH != nullptr)
    m_pBVH->ClearScene();
 
  m_geomTable = m_pGeomStorage->GetTable();

  m_instMatricesInv.resize(0);
  m_lightsInstanced.resize(0);
  m_instLightInstId.resize(0);
  m_meshIdByInstId.resize(0);
  m_instIdByInstId.resize(0);
  m_lightIdByLightInstId.resize(0);
  m_meshRemapListId.resize(0);

  m_sceneHaveSkyPortals = false;

  pugi::xml_node remapLists = a_sceneNode.child(L"remap_lists");
  if (remapLists != nullptr)
  {
    // create remap list table
    //
    std::vector<int>  allRemapLists;
    std::vector<int2> tableOffsetsAndSize;
    allRemapLists.reserve(10000);
    tableOffsetsAndSize.reserve(1000);

    for (pugi::xml_node listNode : remapLists.children())
    {
      const wchar_t* inputStr = listNode.attribute(L"val").as_string();
      const int listSize      = listNode.attribute(L"size").as_int();
      std::wstringstream inStrStream(inputStr);

      tableOffsetsAndSize.push_back(int2(int(allRemapLists.size()), listSize));

      for (int i = 0; i < listSize; i++)
      {
        if (inStrStream.eof())
          break;

        int data;
        inStrStream >> data;
        allRemapLists.push_back(data);
      }
    }

    if (tableOffsetsAndSize.size() > 0 && allRemapLists.size() > 0)
      m_pHWLayer->SetAllRemapLists(&allRemapLists[0], &tableOffsetsAndSize[0], int(allRemapLists.size()), int(tableOffsetsAndSize.size()));
    else
      m_pHWLayer->SetAllRemapLists(nullptr, nullptr, 0, 0);
  }
  else // set empty remap table
    m_pHWLayer->SetAllRemapLists(nullptr, nullptr, 0, 0);
}

std::vector<float> PrefixSumm(const std::vector<float>& a_vec);


size_t EstimateBVHSize(const ConvertionResult& a_bvh)
{
  size_t size = 0;
  for (int i = 0; i < a_bvh.treesNum; i++)
  {
    size += a_bvh.nodesNum[i] * sizeof(BVHNode);
    size += a_bvh.trif4Num[i] * sizeof(float4);
    if(a_bvh.pTriangleAlpha[i] != nullptr)
      size += a_bvh.triAfNum[i] * sizeof(uint2);
  }
  return size;
}


void RenderDriverRTE::EndScene() // #TODO: add dirty flags (?) to update only those things that were changed
{
  if (m_pBVH == nullptr)
    return;
  
  auto timeBeg  = std::chrono::system_clock::now();
  
  std::cout << "[EndScene]: BVH wait ... " << std::endl;
  if(m_pSysMutex != nullptr)
    hr_lock_system_mutex(m_pSysMutex, 5000); // don't allow simultanoius bvh building in several processes
  
  std::cout << "[EndScene]: BVH build ... " << std::endl;

  m_pBVH->CommitScene();
  
  if (m_useConvertedLayout)
  {
    auto convertedData = m_pBVH->ConvertMap();

    if (convertedData.treesNum == 0)
    {
      std::cout << "[RenderDriverRTE::EndScene]: critical error, no BVH trees!" << std::endl;
      exit(0);
    }

    bool smoothOpacity = false;
    CreateAlphaTestTable(convertedData, m_alphaAuxBuffers, smoothOpacity);
    //DebugTestAlphaTestTable(m_alphaAuxBuffers.buf[0], convertedData.trif4Num[0]);

    const int bvhFlags = smoothOpacity ? BVH_ENABLE_SMOOTH_OPACITY : 0;

    m_pHWLayer->SetAllBVH4(convertedData, nullptr, bvhFlags); // set converted layout with matrices inside bvh tree itself
 
    const size_t bvhSize = EstimateBVHSize(convertedData);
    std::cout << "[EndScene]: MEM(BVH)    = " << bvhSize / size_t(1024*1024) << "\tMB" << std::endl; m_memAllocated += bvhSize;

    //PrintBVHStat(convertedData, true);
    //DebugSaveBVH("D:/temp/bvh_layers2", convertedData);
    //DebugPrintBVHInfo(convertedData, "z_bvhinfo.txt");

    m_pBVH->ConvertUnmap();
    

    for(int i=0;i<MAXBVHTREES;i++)
      m_alphaAuxBuffers.buf[i] = std::vector<uint2>();

    const size_t totalMem = m_pHWLayer->GetAvaliableMemoryAmount(true);
    std::cout << "[EndScene]: MEM(TAKEN)  = " << m_memAllocated / size_t(1024 * 1024) << "\tMB" << std::endl;
    std::cout << "[EndScene]: MEM(TOTAL)  = " << totalMem / size_t(1024 * 1024) << "\tMB" << std::endl;
 
    std::cout << "[EndScene]: TexStorageS = " << m_pTexStorage->GetSize() / size_t(1024 * 1024) << "\tMB" << std::endl;
    std::cout << "[EndScene]: TexStorageC = " << m_pTexStorage->GetCapacity() / size_t(1024 * 1024) << "\tMB" << std::endl;

    std::cout << std::endl;

    int oclVer = 0;
    std::cout << "[EndScene]: using device: " << m_pHWLayer->GetDeviceName(&oclVer) << std::endl;
    std::cout << "[EndScene]: using cl_ver: " << oclVer << std::endl;
    std::cout << std::endl;
  }
  else
  {
    m_pHWLayer->SetAllBVH4(ConvertionResult(), m_pBVH, 0); // set pointer to bvh builder to trace rays on the cpu
  }
 

  m_pBVH->GetBounds(&m_sceneBoundingBoxMin.x, &m_sceneBoundingBoxMax.x);

  const float3 halfSize = 0.5f*(m_sceneBoundingBoxMax - m_sceneBoundingBoxMin);
  const float3 center   = 0.5f*(m_sceneBoundingBoxMax + m_sceneBoundingBoxMin);
  const float radius    = length(halfSize);
  m_sceneBoundingSphere = to_float4(center, radius);

  assert(m_instMatricesInv.size() != 0);
  assert(m_instLightInstId.size() != 0);

  BuildSkyPortalsDependencyDummyInstances();

  if (m_instMatricesInv.size() == 0 || m_instLightInstId.size() == 0)
    RUN_TIME_ERROR("RenderDriverRTE::EndScene, no instances in the scene!");

  m_pHWLayer->SetAllInstMatrices   (&m_instMatricesInv[0], int32_t(m_instMatricesInv.size()));
  m_pHWLayer->SetAllInstIdToRemapId(&m_meshRemapListId[0], int32_t(m_meshRemapListId.size()));

  // put bounding sphere to engine globals
  //
  auto vars = m_pHWLayer->GetAllFlagsAndVars();
  vars.m_varsF[HRT_BSPHERE_CENTER_X]    = m_sceneBoundingSphere.x;
  vars.m_varsF[HRT_BSPHERE_CENTER_Y]    = m_sceneBoundingSphere.y;
  vars.m_varsF[HRT_BSPHERE_CENTER_Z]    = m_sceneBoundingSphere.z;
  vars.m_varsF[HRT_BSPHERE_RADIUS  ]    = m_sceneBoundingSphere.w;
  vars.m_varsI[HRT_SHADOW_MATTE_BACK]   = this->m_shadowMatteBackTexId;
  vars.m_varsF[HRT_BACK_TEXINPUT_GAMMA] = this->m_shadowMatteBackGamma;
  m_pHWLayer->SetAllFlagsAndVars(vars);

  // calculate light selector pdf tables
  //
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// //#TODO: refactor. put in separate function
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if(m_lightsInstanced.size() > 0)
  {
    m_pHWLayer->SetAllInstLightInstId(&m_instLightInstId[0], int32_t(m_instLightInstId.size()));

    const std::vector<float> pickProbRev = CalcLightPickProbTable(m_lightsInstanced, false);
    const std::vector<float> pickProbFwd = CalcLightPickProbTable(m_lightsInstanced, true);

    const std::vector<float> tableRev    = PrefixSumm(pickProbRev);
    const std::vector<float> tableFwd    = PrefixSumm(pickProbFwd);

    const float normRev = 1.0f/tableRev[tableRev.size() - 1];
    const float normFwd = 1.0f/tableFwd[tableFwd.size() - 1];

    for (size_t i = 0; i < m_lightsInstanced.size(); i++)
    {
      m_lightsInstanced[i].data[PLIGHT_PICK_PROB_FWD] *= normFwd;
      m_lightsInstanced[i].data[PLIGHT_PICK_PROB_REV] *= normRev;
    }

    m_pHWLayer->SetAllLightsSelectTable(&tableRev[0], int32_t(tableRev.size()), false);
    m_pHWLayer->SetAllLightsSelectTable(&tableFwd[0], int32_t(tableFwd.size()), true);
    m_pHWLayer->SetAllPODLights(&m_lightsInstanced[0], m_lightsInstanced.size());
  }
  else
  {
    std::cerr << "WARNING: RenderDriverRTE::EndScene(), no lights!" << std::endl;
    m_pHWLayer->SetAllInstLightInstId  (nullptr, 0);
    m_pHWLayer->SetAllLightsSelectTable(nullptr, 0);
    m_pHWLayer->SetAllPODLights        (nullptr, 0);
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (m_haveAtLeastOneAOMat && m_procTextures.size() != 0)
    m_pHWLayer->SetNamedBuffer("ao", nullptr, size_t(-1));

  if(m_haveAtLeastOneAOMat2 && m_procTextures.size() != 0)
    m_pHWLayer->SetNamedBuffer("ao2", nullptr, size_t(-1));

  m_pHWLayer->PrepareEngineTables();

  if (m_needToFreeCPUMem)
    FreeCPUMem();

  if (m_pSysMutex != nullptr)
    hr_unlock_system_mutex(m_pSysMutex);
  
  auto timeEnd  = std::chrono::system_clock::now();
  auto msPassed = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeBeg).count();
  
  std::cout << "[EndScene]: BVH finished; bvh build time = " << float(msPassed)/1000.0f << " s" << std::endl;
}

void RenderDriverRTE::FreeCPUMem()
{
  m_pTexStorage->FreeHostMem();
  m_pTexStorageAux->FreeHostMem();
  m_pGeomStorage->FreeHostMem();
  m_pMaterialStorage->FreeHostMem();
  m_pPdfStorage->FreeHostMem();

  m_lights              = std::vector< std::shared_ptr<RAYTR::ILight> >();
  m_lightsInstanced     = std::vector<PlainLight>();
  m_lightHavePdfTable   = std::unordered_set<int>();
  m_iesCache            = std::unordered_map<std::wstring, int2>();
  m_materialUpdated     = std::unordered_map<int, std::shared_ptr<RAYTR::IMaterial> >();
  m_materialNodes       = std::unordered_map<int, pugi::xml_node>();
  m_texturesProcessedNM = std::unordered_map<std::wstring, int32_t>();
  m_blendsToUpdate      = std::unordered_map<int, DefferedMaterialDataTuple >();

  m_geomTable     = std::vector<int>();
  m_texTable      = std::vector<int>();
  m_texTableAux   = std::vector<int>();
  m_materialTable = std::vector<int>();

  m_instMatricesInv      = std::vector<float4x4>();
  m_instLightInstId      = std::vector<int32_t>();
  m_lightIdByLightInstId = std::vector<int32_t>();
  m_meshIdByInstId       = std::vector<int32_t>();
  //m_instIdByInstId       = std::vector<int32_t>();
  m_meshRemapListId      = std::vector<int32_t>();

  if (m_pBVH != nullptr)
  {
    m_pBVH->Destroy();
    m_pBVH = nullptr;
  }

  for (int i = 0; i < MAXBVHTREES; i++)
    m_alphaAuxBuffers.buf[i] = std::vector<uint2>();

  m_auxTexNormalsPerMat = std::unordered_map<int64_t, int32_t>();
}

void RenderDriverRTE::BuildSkyPortalsDependencyDummyInstances()
{
  // (1) if we have sky portals in scene, set global flag that identify we must not sample any sky light
  //
  auto vars = m_pHWLayer->GetAllFlagsAndVars();
  vars.m_varsI[HRT_HRT_SCENE_HAVE_PORTALS] = (m_sceneHaveSkyPortals == true) ? 1 : 0;
  m_pHWLayer->SetAllFlagsAndVars(vars);

  // (2) iterale all instance lights and collect all instance sky lights id's
  //
  std::unordered_map<int32_t, int32_t> alreadyInstanced;
  for (size_t i = 0; i < m_lightsInstanced.size(); i++)
  {
    if (lightType(&m_lightsInstanced[i]) == PLAIN_LIGHT_TYPE_SKY_DOME)
    {
      int32_t lightId = m_lightIdByLightInstId[i];                               
      alreadyInstanced[lightId] = int32_t(i);
      const int sunId = as_int(m_lightsInstanced[i].data[SKY_DOME_SUN_DIR_ID]);

      // OVERRIDE SUN PARAMETERS for sky light is we have valid sunId
      //
      if(sunId != -1)
      {
        // now we need to find first instance light with id == sunId
        //
        int32_t instId = -1; 
        for (size_t j = 0; i < m_lightIdByLightInstId.size(); j++)
        {
          if (m_lightIdByLightInstId[j] == sunId)
          {
            instId = int32_t(j);
            break;
          }
        }

        if (instId >= 0)
        {
          auto sunData = m_lightsInstanced[instId];
          float3 sunDirection, sunColor;

          sunDirection.x = sunData.data[PLIGHT_NORM_X];
          sunDirection.y = sunData.data[PLIGHT_NORM_Y];
          sunDirection.z = sunData.data[PLIGHT_NORM_Z];

          sunColor.x = sunData.data[PLIGHT_COLOR_X];
          sunColor.y = sunData.data[PLIGHT_COLOR_Y];
          sunColor.z = sunData.data[PLIGHT_COLOR_Z];

          m_lightsInstanced[i].data[SKY_DOME_SUN_DIR_X] = sunDirection.x;
          m_lightsInstanced[i].data[SKY_DOME_SUN_DIR_Y] = sunDirection.y;
          m_lightsInstanced[i].data[SKY_DOME_SUN_DIR_Z] = sunDirection.z;

          m_lightsInstanced[i].data[SKY_SUN_COLOR_X] = sunColor.x;
          m_lightsInstanced[i].data[SKY_SUN_COLOR_Y] = sunColor.y;
          m_lightsInstanced[i].data[SKY_SUN_COLOR_Z] = sunColor.z;
        }
      }
    }
  }

  if (!m_sceneHaveSkyPortals)
    return;

  // (3) instance the rest of sky lights
  //
  for (auto p = m_skyLightsId.begin(); p != m_skyLightsId.end(); ++p)
  {
    int32_t lightId = (*p);
    if (alreadyInstanced.find(lightId) == alreadyInstanced.end())         // instance sky light just to put them in m_lightsInstanced array at some address
    {
      float4x4 identity;
      m_lightsInstanced.push_back(m_lights[lightId]->Transform(identity));
      m_lightIdByLightInstId.push_back(lightId);
      alreadyInstanced[lightId] = int32_t(m_lightsInstanced.size() - 1);
    } 
  }

  // (4) fix offets to sky lights in sky portals
  //
  for (size_t i = 0; i < m_lightsInstanced.size(); i++)
  {
    if (lightFlags(&m_lightsInstanced[i]) & AREA_LIGHT_SKY_PORTAL)
    {
      int* idata = (int*)m_lightsInstanced[i].data;

      const int oldId = idata[AREA_LIGHT_SKY_SOURCE];
      const int newId = alreadyInstanced[oldId];
      idata[AREA_LIGHT_SKY_OFFSET] = newId - int32_t(i);  // relative offset to we could get sky light offset from sky portal offset
    }
  }

  m_sceneHaveSkyPortals = false;
}

void RenderDriverRTE::Draw()
{
  // (1) set camera
  //
  float4x4 mWorldView, mProj;
  CalcCameraMatrices(&m_modelViewInv, &m_projInv, &mWorldView, &mProj);

  const float aspect = float(m_width) / float(m_height);
  m_pHWLayer->SetCamMatrices(m_projInv.L(), m_modelViewInv.L(), mProj.L(), mWorldView.L(), aspect, DEG_TO_RAD*m_camera.fov);
  m_pHWLayer->PrepareEngineGlobals();

  const int NUM_PASS = 1;

  // (2) run rendering pass (depends on enabled algorithm)
  //
  if (m_renderMethod == RENDER_METHOD_MMLT)
  {

    if (!m_ptInitDone)
    {
      // (1) init random gens
      //
      m_pHWLayer->InitPathTracing(m_legacy.m_lastSeed);
      m_ptInitDone = true;

      auto flagsAndVars = m_pHWLayer->GetAllFlagsAndVars();
      flagsAndVars.m_flags |= HRT_UNIFIED_IMAGE_SAMPLING;
      flagsAndVars.m_flags |= HRT_ENABLE_MMLT;

      flagsAndVars.m_flags &= (~HRT_3WAY_MIS_WEIGHTS);
      flagsAndVars.m_flags &= (~HRT_FORWARD_TRACING);
      flagsAndVars.m_flags &= (~HRT_DRAW_LIGHT_LT);
      flagsAndVars.m_flags &= (~HRT_ENABLE_SBPT);

      m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);
      m_drawPassNumber = 0;
    }

    // (3) Do MMLT pass
    //
    auto flagsAndVars = m_pHWLayer->GetAllFlagsAndVars();
    flagsAndVars.m_flags |= HRT_ENABLE_MMLT;
    m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);

    for (int i = 0; i < NUM_PASS; i++)
    {
      m_pHWLayer->BeginTracingPass();
      m_pHWLayer->EndTracingPass();
    }

    // std::cout << "MMLT pass" << std::endl;
  }
  else if (m_renderMethod != RENDER_METHOD_RT)
  {
    if (!m_ptInitDone)
    {
      m_pHWLayer->InitPathTracing(m_legacy.m_lastSeed);
      m_ptInitDone = true;
 
      auto flagsAndVars = m_pHWLayer->GetAllFlagsAndVars();
      flagsAndVars.m_flags |= HRT_UNIFIED_IMAGE_SAMPLING;
      flagsAndVars.m_flags &= (~HRT_3WAY_MIS_WEIGHTS);

      if(m_renderMethod == RENDER_METHOD_SBPT)
      {
        flagsAndVars.m_flags |= HRT_ENABLE_MMLT;
        flagsAndVars.m_flags |= HRT_ENABLE_SBPT;
        flagsAndVars.m_flags &= (~HRT_FORWARD_TRACING);
        flagsAndVars.m_flags &= (~HRT_DRAW_LIGHT_LT);
      }
      else if (m_renderMethod == RENDER_METHOD_LT)
      {
        flagsAndVars.m_flags |= HRT_FORWARD_TRACING;
        flagsAndVars.m_flags |= HRT_DRAW_LIGHT_LT;

        flagsAndVars.m_flags &= (~HRT_ENABLE_MMLT);
        flagsAndVars.m_flags &= (~HRT_ENABLE_SBPT);
      }
      else
      {
        flagsAndVars.m_flags &= (~HRT_FORWARD_TRACING);
        flagsAndVars.m_flags &= (~HRT_DRAW_LIGHT_LT);

        if(m_renderMethod == RENDER_METHOD_MMLT)
        {
          flagsAndVars.m_flags &= (~HRT_ENABLE_SBPT);
          flagsAndVars.m_flags |= HRT_ENABLE_MMLT;
        }
        else
        {
          flagsAndVars.m_flags &= (~HRT_ENABLE_MMLT);
          flagsAndVars.m_flags &= (~HRT_ENABLE_SBPT);
        }
      }

      m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);
      m_drawPassNumber = 0;
    }

    if (m_renderMethod == RENDER_METHOD_IBPT)
    {
      for (int i = 0; i < NUM_PASS; i++)
      {
        // LT PASS
        //
        auto flagsAndVars = m_pHWLayer->GetAllFlagsAndVars();
        flagsAndVars.m_flags &= (~HRT_FORWARD_TRACING);
        flagsAndVars.m_flags &= (~HRT_DRAW_LIGHT_LT);
        flagsAndVars.m_flags |= HRT_UNIFIED_IMAGE_SAMPLING;
        flagsAndVars.m_flags |= HRT_FORWARD_TRACING;
        flagsAndVars.m_flags |= HRT_3WAY_MIS_WEIGHTS;

        m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);
        m_pHWLayer->BeginTracingPass();
        // m_pHWLayer->EndTracingPass(); //#NOTE: dont call EndTracingPass, because it will increase m_spp

        // PT PASS
        //
        flagsAndVars.m_flags &= (~HRT_FORWARD_TRACING);
        flagsAndVars.m_flags &= (~HRT_DRAW_LIGHT_LT);
        flagsAndVars.m_flags |= HRT_UNIFIED_IMAGE_SAMPLING;
        flagsAndVars.m_flags |= HRT_3WAY_MIS_WEIGHTS;

        m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);
        m_pHWLayer->BeginTracingPass();
        m_pHWLayer->EndTracingPass();
      }
    }
    else // run PT or LT depends on previous set 'HRT_FORWARD_TRACING' flag
    {
      for (int i = 0; i < NUM_PASS; i++)
      {
        m_pHWLayer->BeginTracingPass();
        m_pHWLayer->EndTracingPass();
      }
    }

    //imageB to imageA contribution
    //
    if ((m_renderMethod == RENDER_METHOD_LT || m_gpuFB || m_renderMethod == RENDER_METHOD_IBPT) && m_pAccumImage != nullptr)
    {
      auto size = m_pHWLayer->GetRayBuffSize();
      const double freq = 8.0*double(m_width*m_height)/double(size);

      int freqInt = int(freq);   
      if (freqInt < 2) 
        freqInt = 2;
      if (m_gpuFB)
        freqInt *= 4;

      if (m_drawPassNumber % freqInt == 0 && m_drawPassNumber > 0)
        m_pHWLayer->ContribToExternalImageAccumulator(m_pAccumImage);
    }

    m_drawPassNumber += NUM_PASS;
  }
  else // ray tracing and show normals
  {
    auto flagsAndVars    = m_pHWLayer->GetAllFlagsAndVars();
    flagsAndVars.m_flags &= (~HRT_UNIFIED_IMAGE_SAMPLING);
    flagsAndVars.m_flags &= (~HRT_FORWARD_TRACING);
    flagsAndVars.m_flags &= (~HRT_DRAW_LIGHT_LT);
    flagsAndVars.m_flags &= (~HRT_ENABLE_MMLT);
    flagsAndVars.m_flags &= (~HRT_ENABLE_SBPT);
    m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);

    m_pHWLayer->BeginTracingPass();
    m_pHWLayer->EndTracingPass();
    m_ptInitDone = false;
  }


  if (MEASURE_RAYS && m_renderMethod != RENDER_METHOD_RT)
  {
    auto stats = m_pHWLayer->GetRaysStat();

    AverageStats(stats, m_avgStats, m_avgStatsId);

    int   mrays = (int)(m_avgStats.raysPerSec*1e-6f + 0.5f);
    float msamp = m_avgStats.samplesPerSec*1e-6f;

    auto oldPrec = std::cout.precision(4);
    std::cout << std::endl << std::fixed;
    std::cout << "[stat]: MRays/sec  = " << mrays << std::endl;
    std::cout << "[stat]: traversal  = " << m_avgStats.traversalTimeMs << "\t ms" << std::endl;
    std::cout << "[stat]: sam_light  = " << m_avgStats.samLightTimeMs  << "\t ms" << std::endl;
    std::cout << "[stat]: shadow     = " << m_avgStats.shadowTimeMs    << "\t ms" << std::endl;
    std::cout << "[stat]: shade      = " << m_avgStats.shadeTimeMs     << "\t ms" << std::endl;
    std::cout << "[stat]: computehit = " << m_avgStats.evalHitMs       << "\t ms" << std::endl;
    std::cout << "[stat]: nextbounce = " << m_avgStats.nextBounceMs    << "\t ms" << std::endl;
    std::cout << "[stat]: fullbounce = " << m_avgStats.bounceTimeMS    << "\t ms" << std::endl;
    std::cout << "[stat]: sampletime = " << m_avgStats.sampleTimeMS    << "\t ms" << std::endl;
    std::cout << "[stat]: MSampl/sec = " << msamp << std::endl;

    const float traceTimePerCent = 100.0f*(m_avgStats.traversalTimeMs + m_avgStats.shadowTimeMs) / m_avgStats.bounceTimeMS;

    std::cout << "[stat]: trace(%)   = " << traceTimePerCent << "%" << std::endl;
    std::cout.precision(oldPrec);
  }
  
  // for some unknown reason command "exitnow" passed via shared memory does not works in some Linux systems
  // So, we have to have this aux. exit condition
  //
  if(m_pAccumImage != nullptr)
  {
    const float currSPP = m_pAccumImage->Header()->spp;
    if(currSPP >= m_maxRaysPerPixel)
    {
      std::cout << std::endl;
      std::cout << "[core]: exit from render loop due to reach max samples limit = " << m_maxRaysPerPixel << std::endl;
      std::cout << std::endl;
      g_exitDueToSamplesLimit = true;
    }
  }

}


void RenderDriverRTE::EvalGBuffer()
{
  if(m_pAccumImage != nullptr)
    m_pHWLayer->EvalGBuffer(m_pAccumImage, m_instIdByInstId);
  else
  {
    std::cerr << "EvalGBuffer: nullptr external AccumImage, save gbuffer to internal image" << std::endl;
    char errMsg[256];
    m_gbufferImage.Create(m_width, m_height, 3, "gbuffer-only", errMsg);
    m_pHWLayer->EvalGBuffer(&m_gbufferImage, m_instIdByInstId);
  }
}

void RenderDriverRTE::InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, 
                                     const int* a_lightInstId, const int* a_remapId, const int* a_realInstId)
{
  if (m_pBVH == nullptr)
    return;

  const int32_t meshId = a_mesh_id;

  if (meshId >= m_geomTable.size())
  {
    std::cerr << " RenderDriverRTE::InstanceMeshes, bad mesh id = " << meshId << std::endl;
    return;
  }

  const auto offset = m_geomTable[meshId];
  if (offset < 0)
  {
    std::cerr << " RenderDriverRTE::InstanceMeshes, invalid mesh, offset = " << offset << std::endl;
    return;
  }

  IBVHBuilder2::InstanceInputData input;

  const int4* ldata        = (const int4*)m_pGeomStorage->GetBegin();
  const PlainMesh* pHeader = (const PlainMesh*)(ldata + offset);
  
  input.vert4f     = (const float*)meshVerts(pHeader);
  input.indices    = meshTriIndices(pHeader);
  input.numVert    = pHeader->vPosNum;
  input.numIndices = pHeader->tIndicesNum;

  input.meshId     = meshId;
  input.matrices   = a_matrices;
  input.numInst    = a_instNum;

  const bool useEmbreeCPU = !(m_initFlags & GPU_RT_HW_LAYER_OCL);
  const int treeId        = (!useEmbreeCPU && MeshHaveOpacity(pHeader)) ? 1 : 0;

  m_pBVH->InstanceTriangleMeshes(input, treeId, int(m_meshIdByInstId.size()));

  // (1) Remember matrices id for further usage if external CPU impl of BVH is used
  // (2) Also create light-inst id from inst id table
  // (3) Remember mesh id for each instance for debug needs
  //
  for (int32_t instId = 0; instId < a_instNum; instId++)
  {
    const float4x4 mTransform(a_matrices + 16 * instId);
    const float4x4 invMatrix  = inverse4x4(mTransform);
    const int      remapId    = a_remapId[instId];
    const int      instIdReal = a_realInstId[instId];
    m_instMatricesInv.push_back(invMatrix);
    m_instLightInstId.push_back(a_lightInstId[instId]);
    m_meshIdByInstId.push_back(meshId);
    m_instIdByInstId.push_back(instIdReal);
    m_meshRemapListId.push_back(remapId);
  }
 
}

void RenderDriverRTE::InstanceLights(int32_t a_lightId, const float* a_matrix, pugi::xml_node* a_lightNodes, int32_t a_instNum, int32_t a_lightGroupId)
{
  if (a_lightId >= m_lights.size())
  {
    Error(L"bad a_lightId = %d", a_lightId);
    return;
  }

  const std::shared_ptr<RAYTR::ILight> pLight = m_lights[a_lightId];
  if (pLight == nullptr)
  {
    Error(L"nullptr pLight, for a_lightId = %d", a_lightId);
    return;
  }

  for (int32_t i = 0; i < a_instNum; i++)
  {
    float4x4 mTransform = float4x4(a_matrix + 16 * i);
    PlainLight ldata    = pLight->Transform(mTransform);

    bool doNotSampleMe = false;

    if (a_lightNodes != nullptr)
    {
      pugi::xml_node node = a_lightNodes[i];
      if (node != nullptr)
      {
        pugi::xml_attribute cmult = node.attribute(L"color_mult");
        if (cmult != nullptr)
        {
          float3 colorMult = HydraXMLHelpers::ReadFloat3(cmult);
          ldata.data[PLIGHT_COLOR_X] *= colorMult.x;
          ldata.data[PLIGHT_COLOR_Y] *= colorMult.y;
          ldata.data[PLIGHT_COLOR_Z] *= colorMult.z;
        }

        pugi::xml_attribute skip = node.attribute(L"do_not_sample_me");
        if (skip != nullptr)
          doNotSampleMe = skip.as_bool();

        pugi::xml_attribute probMult = node.attribute(L"prob_mult");
        if (probMult != nullptr)
          ldata.data[PLIGHT_PROB_MULT] = probMult.as_float();
      }
    }

    ((int*)ldata.data)[PLIGHT_GROUP_ID] = a_lightGroupId;
    ldata.data[PLIGHT_PICK_PROB_REV]    = 1.0f; // will be overwritten further when we will total lights number and groups number and lights in each group.
    ldata.data[PLIGHT_PICK_PROB_FWD]    = 1.0f; // will be overwritten further when we will total lights number and groups number and lights in each group.

    if(doNotSampleMe)
      ((int*)ldata.data)[PLIGHT_FLAGS] |= LIGHT_DO_NOT_SAMPLE_ME;
    
    m_lightsInstanced.push_back(ldata);
    m_lightIdByLightInstId.push_back(a_lightId);
  }

  if (pLight->GetFlags() & AREA_LIGHT_SKY_PORTAL) // this will prevent sky lights from explicit sampling
    m_sceneHaveSkyPortals = true;

}

HRRenderUpdateInfo RenderDriverRTE::HaveUpdateNow(int a_maxRaysperPixel)
{
  HRRenderUpdateInfo res;

  res.haveUpdateFB  = true;
  res.haveUpdateMSG = (m_msg != L"");
  res.msg           = m_msg.c_str();

  const float spp = m_pHWLayer->GetSPP();
  
  res.progress    = spp/float(a_maxRaysperPixel);
  res.finalUpdate = (res.progress >= 1.0f); 
 
  return res;
}

void RenderDriverRTE::GetFrameBufferHDR(int32_t w, int32_t h, float*   a_out, const wchar_t* a_layerName)
{
  m_pHWLayer->GetHDRImage((float4*)a_out, w, h);

  if(m_gbufferImage.Header()->width == w && m_gbufferImage.Header()->height == h) // save shadow values in separate buffer
  {
    if(m_gbufferImage.shadowCopy.size() != w*h)
      m_gbufferImage.shadowCopy.resize(w*h);
    
    for(int i=0;i<(w*h);i++)
      m_gbufferImage.shadowCopy[i] = a_out[4*i+3];
  }
}

void RenderDriverRTE::GetFrameBufferLDR(int32_t w, int32_t h, int32_t* a_out)
{
  m_pHWLayer->GetLDRImage((uint32_t*)a_out, w, h);
}

static inline void decodeNormal2(unsigned int a_data, float a_norm[3])
{
  const float divInv = 1.0f / 32767.0f;

  short a_enc_x, a_enc_y;

  a_enc_x = (short)(a_data & 0x0000FFFF);
  a_enc_y = (short)((int)(a_data & 0xFFFF0000) >> 16);

  float sign = (a_enc_x & 0x0001) ? -1.0f : 1.0f;

  a_norm[0] = (short)(a_enc_x & 0xfffe)*divInv;
  a_norm[1] = (short)(a_enc_y & 0xfffe)*divInv;
  a_norm[2] = sign*sqrt(fmax(1.0f - a_norm[0] * a_norm[0] - a_norm[1] * a_norm[1], 0.0f));
}

static inline HRGBufferPixel UnpackGBuffer(const float a_input[4], const float a_input2[4])
{
  HRGBufferPixel res;

  res.depth = a_input[0];
  res.matId = as_int(a_input[2]) & 0x00FFFFFF;
  decodeNormal2(as_int(a_input[1]), res.norm);

  unsigned int rgba = as_int(a_input[3]);
  res.rgba[0] = (rgba & 0x000000FF)*(1.0f / 255.0f);
  res.rgba[1] = ((rgba & 0x0000FF00) >> 8)*(1.0f / 255.0f);
  res.rgba[2] = ((rgba & 0x00FF0000) >> 16)*(1.0f / 255.0f);
  res.rgba[3] = ((rgba & 0xFF000000) >> 24)*(1.0f / 255.0f);

  res.texc[0] = a_input2[0];
  res.texc[1] = a_input2[1];
  res.objId   = as_int(a_input2[2]);
  res.instId  = as_int(a_input2[3]);

  const int compressedCoverage = (as_int(a_input[2]) & 0xFF000000) >> 24;
  res.coverage = ((float)compressedCoverage)*(1.0f / 255.0f);
  res.shadow   = 0.0f;

  return res;
}

//
//
void RenderDriverRTE::GetGBufferLine(int32_t a_lineNumber, HRGBufferPixel* a_lineData, int32_t a_startX, int32_t a_endX, const std::unordered_set<int32_t>& a_shadowCatchers)
{
  const float* data1 = m_gbufferImage.ImageData(1);
  const float* data2 = m_gbufferImage.ImageData(2);

  if(data1 == nullptr || data2 == nullptr || m_gbufferImage.shadowCopy.size() == 0)
    return;

  if (a_endX > m_width)
    a_endX = m_width;

  const int32_t lineOffset = (a_lineNumber*m_width + a_startX);
  const int32_t lineSize   = (a_endX - a_startX);

  const float normC = 1.0f / m_gbufferImage.Header()->spp;

  for (int32_t x = 0; x < lineSize; x++)
  {
    const float* data11  = &data1[(lineOffset + x) * 4];
    const float* data22  = &data2[(lineOffset + x) * 4];
    a_lineData[x]        = UnpackGBuffer(data11, data22);                          // store main gbuffer data
    a_lineData[x].shadow = 1.0f - m_gbufferImage.shadowCopy[lineOffset + x]*normC; // get shadow from the preserved to 'shadowCopy' fourth channel
  }

}

HRDriverInfo RenderDriverRTE::Info()
{
  HRDriverInfo info; // very simple render driver implementation, does not support any other/advanced stuff

  info.supportHDRFrameBuffer              = false;
  info.supportHDRTextures                 = false;
  info.supportMultiMaterialInstance       = false;

  info.supportImageLoadFromInternalFormat = false;
  info.supportImageLoadFromExternalFormat = false;
  info.supportMeshLoadFromInternalFormat  = false;
  info.supportLighting                    = false;
  
  info.memTotal                           = int64_t(8) * int64_t(1024 * 1024 * 1024);

  return info;
}

const HRRenderDeviceInfoListElem* RenderDriverRTE::DeviceList() const
{
  return m_pHWLayer->ListDevices();
}

bool RenderDriverRTE::EnableDevice(int32_t id, bool a_enable)
{
  if(a_enable)
    m_devId = id;
  return true;
}

IHRRenderDriver* CreateDriverRTE(const wchar_t* a_cfg, int w, int h, int a_devId, int a_flags, IHRSharedAccumImage* a_sharedImage)
{
  return new RenderDriverRTE(a_cfg, w, h, a_devId, a_flags, a_sharedImage);
}

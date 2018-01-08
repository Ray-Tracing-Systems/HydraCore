#include "RenderDriverRTE.h"
#pragma warning(disable:4996) // for wcsncpy to be ok

#include <iostream>
#include <queue>
#include <string>

#include "../../HydraAPI/hydra_api/HydraXMLHelpers.h"
#include "../../HydraAPI/hydra_api/HydraInternal.h"

constexpr bool MEASURE_RAYS   = false;
constexpr int  MEASURE_BOUNCE = 1;

void UpdateProgress(const wchar_t* a_message, float a_progress)
{
  fwprintf(stdout, L"%s: %.0f%%            \r", a_message, a_progress*100.0f);
}

RenderDriverRTE::RenderDriverRTE(const wchar_t* a_options, int w, int h, int a_devId, int a_flags, IHRSharedAccumImage* a_sharedImage) : m_pBVH(nullptr), m_pHWLayer(nullptr),
                                                                                                                                         m_pTexStorage(nullptr), m_pTexStorageAux(nullptr), 
                                                                                                                                         m_pGeomStorage(nullptr), m_pMaterialStorage(nullptr), 
                                                                                                                                         m_pPdfStorage(nullptr)
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

  m_useConvertedLayout = false || (m_initFlags & GPU_RT_HW_LAYER_OCL);
  m_useBvhInstInsert   = false;

  if (MEASURE_RAYS)
    m_initFlags |= GPU_RT_MEMORY_FULL_SIZE_MODE;

  m_initFlags |= a_flags;

  m_usePT      = false;
  m_useLT      = false;
  m_ptInitDone = false;
  m_legacy.m_lastSeed = GetTickCount();
  m_legacy.updateProgressCall = &UpdateProgress;

  m_auxImageNumber  = 0;
  m_avgStatsId      = 0;

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  if (m_initFlags & GPU_RT_HW_LAYER_OCL)
    m_pHWLayer = CreateOclImpl(m_width, m_height, m_initFlags, m_devId);
  else
    m_pHWLayer = CreateCPUExpImpl(m_width, m_height, 0);
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  m_pHWLayer->SetProgressBarCallback(&UpdateProgress);
 
  m_firstResizeOfScreen = true;

  m_pBVH = CreateBuilderFromDLL(L"bvh_builder.dll", "");

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

  m_pExternalImage = a_sharedImage;

  if (m_pExternalImage != nullptr)
  {
    auto pHeader = m_pExternalImage->Header();

    if (pHeader->width != w || pHeader->height != h)
    {
      std::cout << "input    (w,h) = " << "(" << w << "," << h << ")" << std::endl;
      std::cout << "external (w,h) = " << "(" << pHeader->width << "," << pHeader->height << ")" << std::endl;
    }
  }

  m_pHWLayer->SetExternalImageAccumulator(m_pExternalImage);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

  const bool enableMLTFromSettings = (std::wstring(L"mlt") == a_settingsNode.child(L"method_secondary").text().as_string() ||
                                      std::wstring(L"mlt") == a_settingsNode.child(L"method_tertiary").text().as_string() ||
                                      std::wstring(L"mlt") == a_settingsNode.child(L"method_caustic").text().as_string() ||
                                      std::wstring(L"1")   == a_settingsNode.child(L"enable_mlt").text().as_string());

  if (oldWidth != m_width || oldHeight != m_height || m_firstResizeOfScreen)
  {
    int flags = MEASURE_RAYS ? GPU_RT_MEMORY_FULL_SIZE_MODE : 0;
    if (enableMLTFromSettings)
      flags |= GPU_MLT_ENABLED_AT_START;
    m_pHWLayer->ResizeScreen(m_width, m_height, flags);
    m_firstResizeOfScreen = false;
  }

  //
  //
  auto vars = m_pHWLayer->GetAllFlagsAndVars();

  vars.m_flags |= (HRT_DIFFUSE_REFLECTION | HRT_USE_MIS | HRT_COMPUTE_SHADOWS);

  if((m_initFlags & GPU_ALLOC_FOR_COMPACT_MLT) || (m_initFlags & GPU_MLT_ENABLED_AT_START) || enableMLTFromSettings)
    vars.m_flags |= HRT_ENABLE_MLT;

  if (std::wstring(a_settingsNode.child(L"method_caustic").text().as_string()) == L"none" ||
      std::wstring(a_settingsNode.child(L"method_caustic").text().as_string()) == L"disabled")
    ;
  else if (std::wstring(a_settingsNode.child(L"method_caustic").text().as_string()) == L"mlt" ||
           std::wstring(a_settingsNode.child(L"method_caustic").text().as_string()) == L"pathtracing" ||
           std::wstring(a_settingsNode.child(L"method_caustic").text().as_string()) == L"enabled")
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

  vars.m_varsF[HRT_MLT_PLARGE]        = 0.25f;
  vars.m_varsI[HRT_MLT_BURN_ITERS]    = 128;
  vars.m_varsI[HRT_MMLT_FIRST_BOUNCE] = 3;      // 2,3 or >3; if >3 separate by specular bounce.

  // override default settings from scene settings
  //
  if (a_settingsNode.child(L"method_primary") != nullptr)
  {
    if (std::wstring(a_settingsNode.child(L"method_primary").text().as_string()) == L"pathtracing")
    {
      m_usePT = true;
      m_useLT = false;
    }
    else if (std::wstring(a_settingsNode.child(L"method_primary").text().as_string()) == L"lighttracing")
    {
      m_usePT = false;
      m_useLT = true;
    }
    else
    {
      m_useLT = false;
      m_usePT = false;
    }
  }
  else
  {
    m_usePT = true;
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
    m_legacy.maxRaysPerPixel = a_settingsNode.child(L"maxRaysPerPixel").text().as_int();

  if (a_settingsNode.child(L"seed") != nullptr)
    m_legacy.m_lastSeed = a_settingsNode.child(L"seed").text().as_int();

  if(m_initFlags & GPU_RT_DO_NOT_PRINT_PASS_NUMBER)
    vars.m_varsI[HRT_SILENT_MODE] = 1;

  if (vars.m_flags & HRT_STUPID_PT_MODE)
    vars.m_varsI[HRT_TRACE_DEPTH]++;

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

  //delete m_pExternalImage;  // !We do not own this pointer, remember that!!!
  m_pExternalImage = nullptr;

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
  m_blendsToUpdate.clear();
  m_texturesProcessedNM.clear();

  m_instMatricesInv.clear();
  m_instLightInstId.clear();
  m_meshIdByInstId.clear();

  m_auxImageNumber = 0;
  m_auxTexNormalsPerMat.clear();
}

std::shared_ptr<RAYTR::IMaterial> CreateDiffuseWhiteMaterial();

HRDriverAllocInfo RenderDriverRTE::AllocAll(HRDriverAllocInfo a_info)
{
  m_libPath = std::wstring(a_info.libraryPath);

  // create memory storages and tables
  //
  const size_t approxSizeOfMatBlock = sizeof(PlainMaterial) * 4;
  const size_t approxSizeOfLight    = sizeof(PlainLight)    * 4;

  m_pTexStorage      = m_pHWLayer->CreateMemStorage((a_info.imgMem*3)/4,                "textures");     // #TODO:  estimate this more carefully pls.
  m_pTexStorageAux   = m_pHWLayer->CreateMemStorage((a_info.imgMem*1)/4,                "textures_aux"); // #TODO:  estimate this more carefully pls.
  m_pGeomStorage     = m_pHWLayer->CreateMemStorage(a_info.geomMem,                     "geom");         // #TODO:  estimate this more carefully pls.
  m_pMaterialStorage = m_pHWLayer->CreateMemStorage(a_info.matNum*approxSizeOfMatBlock, "materials");
  m_pPdfStorage      = m_pHWLayer->CreateMemStorage(a_info.imgMem/10,                   "pdfs");         // #TODO:  estimate this more carefully pls.

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

  std::cout << "[AllocAll]: image mem size = " << a_info.imgMem/(1024*1024) << " MB" << std::endl;
  std::cout << "[AllocAll]: geom  mem size = " << a_info.geomMem / (1024 * 1024) << " MB" << std::endl;
  std::cout << "[AllocAll]: total mem size = " << (a_info.imgMem + a_info.geomMem) / (1024 * 1024) << " MB" << std::endl;

  m_lastAllocInfo = a_info;
  return m_lastAllocInfo;
}

void RenderDriverRTE::GetLastErrorW(wchar_t a_msg[256])
{
  wcsncpy(a_msg, m_msg.c_str(), 256);
  m_msg = L"";
}

bool RenderDriverRTE::UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode)
{
  SWTextureHeader texheader;

  texheader.width  = w;
  texheader.height = h;
  texheader.depth  = 1;
  texheader.bpp    = bpp;

  const size_t inDataBSz  = size_t(w)*size_t(h)*size_t(bpp);
  const int    align      = int(m_pTexStorage->GetAlignSizeInBytes());
  const size_t headerSize = roundBlocks(sizeof(SWTextureHeader), align);
  const size_t totalSize  = roundBlocks(inDataBSz, align) + headerSize;

  m_pTexStorage->Update(a_texId, nullptr, totalSize);

  m_pTexStorage->UpdatePartial(a_texId, &texheader, 0, sizeof(SWTextureHeader));
  m_pTexStorage->UpdatePartial(a_texId, a_data, headerSize, inDataBSz);

  return true;
}

std::shared_ptr<RAYTR::IMaterial> CreateMaterialFromXmlNode(pugi::xml_node a_node, RenderDriverRTE* a_pRTE);

bool RenderDriverRTE::UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode)
{
  //std::cerr << "RenderDriverRTE::UpdateMaterial(" << a_matId << ") " << std::endl;

  std::wstring mtype = a_materialNode.attribute(L"type").as_string();

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

  m_materialUpdated[a_matId] = pMaterial; // remember that we have updates this material in current update phase (between BeginMaterialUpdate and EndMaterialUpdate)

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
  const size_t vertTexcSize   = roundBlocks(sizeof(float2)*a_input.vertNum, align);
    
  const size_t vertTangOffset = vertTexcOffset + vertTexcSize;
  const size_t vertTangSize   = roundBlocks(sizeof(int)*a_input.vertNum, align); // compressed tangent
   
  const size_t triIndOffset   = vertTangOffset + vertTangSize;
  const size_t triIndSize     = roundBlocks(triIndOffset + a_input.triNum * 3 * sizeof(int), align);

  const size_t triMIndOffset  = triIndOffset + triIndSize;
  const size_t triMIndSize    = roundBlocks(a_input.triNum * sizeof(int), align);

  const size_t totalByteSize  = triMIndOffset + triMIndSize;

  const float4* a_tan = (const float4*)a_input.tan4f;
  std::vector<int> compressedTangent(a_input.vertNum);

  for (int i = 0; i < a_input.vertNum; i++)
    compressedTangent[i] = encodeNormal(to_float3(a_tan[i]));

  auto offset = m_pGeomStorage->Update(a_meshId, nullptr, totalByteSize); // alloc new chunk for our mesh

  if (offset == -1)
  {
    std::cerr << "RenderDriverRTE::UpdateMesh(id = " << a_meshId << ") failed. too big? :) " << std::endl;
    return false;
  }

  PlainMesh header;

  header.vPosOffset      = int(vertPosOffset  / alignOffs);
  header.vNormOffset     = int(vertNormOffset / alignOffs);
  header.vTexCoordOffset = int(vertTexcOffset / alignOffs);
  header.vTangentOffset  = int(vertTangOffset / alignOffs);
  header.vIndicesOffset  = int(triIndOffset   / alignOffs);
  header.mIndicesOffset  = int(triMIndOffset  / alignOffs);

  header.vPosNum         = a_input.vertNum;
  header.vNormNum        = a_input.vertNum;
  header.vTexCoordNum    = a_input.vertNum;
  header.vTangentNum     = a_input.vertNum;
  header.tIndicesNum     = a_input.triNum * 3;
  header.mIndicesNum     = a_input.triNum;
  header.totalBytesNum   = int(totalByteSize);

  if (totalByteSize > 4294967296)
  {
    std::cerr << "RenderDriverRTE::UpdateMesh(id = " << a_meshId << ") integer overflow for mesh byte size = " << totalByteSize << std::endl;
    return false;
  }

  m_pGeomStorage->UpdatePartial(a_meshId, &header, 0, sizeof(header));
  m_pGeomStorage->UpdatePartial(a_meshId, a_input.pos4f,         vertPosOffset,  a_input.vertNum * sizeof(float4));
  m_pGeomStorage->UpdatePartial(a_meshId, a_input.norm4f,        vertNormOffset, a_input.vertNum * sizeof(float4));
  m_pGeomStorage->UpdatePartial(a_meshId, a_input.texcoord2f,    vertTexcOffset, a_input.vertNum * sizeof(float2));
  m_pGeomStorage->UpdatePartial(a_meshId, &compressedTangent[0], vertTangOffset, a_input.vertNum * sizeof(int)); // compressed tangent
  m_pGeomStorage->UpdatePartial(a_meshId, a_input.indices,       triIndOffset,   a_input.triNum  * 3 * sizeof(int));
  m_pGeomStorage->UpdatePartial(a_meshId, a_input.triMatIndices, triMIndOffset,  a_input.triNum  * sizeof(int));

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

  auto vars = m_pHWLayer->GetAllFlagsAndVars();
  vars.m_varsF[HRT_CAM_FOV] = DEG_TO_RAD*m_camera.fov;

  if (!a_camNode.child(L"dof_lens_radius").text().empty())
    vars.m_varsF[HRT_DOF_LENS_RADIUS] = a_camNode.child(L"dof_lens_radius").text().as_float();

  vars.m_varsF[HRT_DOF_FOCAL_PLANE_DIST] = length(m_camera.pos - m_camera.lookAt);
  
  if (!a_camNode.child(L"enable_dof").text().empty())
    vars.m_varsI[HRT_ENABLE_DOF] = a_camNode.child(L"enable_dof").text().as_int();

  m_pHWLayer->SetAllFlagsAndVars(vars);

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////

HRDriverDependencyInfo RenderDriverRTE::DependencyInfo()
{ 
  HRDriverDependencyInfo res;
  res.needRedrawWhenCameraChanges = false;
  return res; 
}

void RenderDriverRTE::CalcCameraMatrices(float4x4* a_pModelViewMatrixInv, float4x4* a_projMatrixInv, float4x4* a_pModelViewMatrix, float4x4* a_projMatrix)
{
  const float aspect = float(m_width) / float(m_height);

  float4x4 projTransposed      = projectionMatrixTransposed(m_camera.fov, aspect, m_camera.nearPlane, m_camera.farPlane);
  float4x4 worldViewTransposed = lookAtTransposed(m_camera.pos, m_camera.lookAt, m_camera.up);

  (*a_pModelViewMatrixInv)     = transpose(inverse4x4(worldViewTransposed));
  (*a_projMatrixInv)           = transpose(inverse4x4(projTransposed));
  (*a_pModelViewMatrix)        = transpose(worldViewTransposed);
  (*a_projMatrix)              = transpose(projTransposed);
}

void RenderDriverRTE::BeginScene()
{
  if (m_pBVH != nullptr)
    m_pBVH->ClearScene();
 
  m_geomTable = m_pGeomStorage->GetTable();

  m_instMatricesInv.resize(0);
  m_lightsInstanced.resize(0);
  m_instLightInstId.resize(0);
  m_meshIdByInstId.resize(0);
  m_lightIdByLightInstId.resize(0);

  m_sceneHaveSkyPortals = false;

}

std::vector<float> PrefixSumm(const std::vector<float>& a_vec);

void RenderDriverRTE::EndScene() // #TODO: add dirty flags (???) to update only those things that were changed
{
  if (m_pBVH == nullptr)
    return;

  m_pBVH->CommitScene();
  
  if (m_useConvertedLayout)
  {
    std::cout << std::endl;
    std::cout << "[RenderDriverRTE::EndScene]: begin bvh convert " << std::endl;
    auto convertedData = m_pBVH->ConvertMap();

    if (convertedData.treesNum == 0)
    {
      std::cout << "[RenderDriverRTE::EndScene]: critical error, no BVH trees!" << std::endl;
      exit(0);
    }

    bool smoothOpacity = false;
    CreateAlphaTestTable(convertedData, m_alphaAuxBuffers, smoothOpacity);

    const int bvhFlags = smoothOpacity ? BVH_ENABLE_SMOOTH_OPACITY : 0;

    m_pHWLayer->SetAllBVH4(convertedData, nullptr, bvhFlags); // set converted layout with matrices inside bvh tree itself
  
    //PrintBVHStat(convertedData, true);
    //DebugSaveBVH("D:/temp/bvh_layers2", convertedData);
    //DebugPrintBVHInfo(convertedData, "z_bvhinfo.txt");

    m_pBVH->ConvertUnmap();

    for(int i=0;i<MAXBVHTREES;i++)
      m_alphaAuxBuffers.buf[i] = std::vector<uint2>();

    std::cout << "[RenderDriverRTE::EndScene]: end bvh convert  " << std::endl << std::endl;
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

  m_pHWLayer->SetAllInstMatrices(&m_instMatricesInv[0], int32_t(m_instMatricesInv.size()));

  // put bounding sphere to engine globals
  //
  auto vars = m_pHWLayer->GetAllFlagsAndVars();
  vars.m_varsF[HRT_BSPHERE_CENTER_X] = m_sceneBoundingSphere.x;
  vars.m_varsF[HRT_BSPHERE_CENTER_Z] = m_sceneBoundingSphere.y;
  vars.m_varsF[HRT_BSPHERE_CENTER_Z] = m_sceneBoundingSphere.z;
  vars.m_varsF[HRT_BSPHERE_RADIUS  ] = m_sceneBoundingSphere.w;
  m_pHWLayer->SetAllFlagsAndVars(vars);

  // calculate light selector pdf tables
  //
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// //#TODO: refactor. put in separate function
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if(m_lightsInstanced.size() > 0)
  {
    m_pHWLayer->SetAllInstLightInstId(&m_instLightInstId[0], int32_t(m_instLightInstId.size()));

    const std::vector<float> pickProb = CalcLightPickProbTable(m_lightsInstanced);
    const std::vector<float> table    = PrefixSumm(pickProb);
    m_pHWLayer->SetAllLightsSelectTable(&table[0], int32_t(table.size()));
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

  m_pHWLayer->PrepareEngineTables();

  if (m_needToFreeCPUMem)
    FreeCPUMem();
}

void RenderDriverRTE::FreeCPUMem()
{
  m_pTexStorage->FreeHostMem();
  m_pTexStorageAux->FreeHostMem();
  m_pGeomStorage->FreeHostMem();
  m_pMaterialStorage->FreeHostMem();
  m_pPdfStorage->FreeHostMem();

  m_lights = std::vector< std::shared_ptr<RAYTR::ILight> >();
  m_lightsInstanced = std::vector<PlainLight>();
  m_lightHavePdfTable = std::unordered_set<int>();
  m_iesCache = std::unordered_map<std::wstring, int2>();
  m_materialUpdated = std::unordered_map<int, std::shared_ptr<RAYTR::IMaterial> >();
  m_texturesProcessedNM = std::unordered_map<std::wstring, int32_t>();
  m_blendsToUpdate = std::unordered_map<int, DefferedMaterialDataTuple >();

  m_geomTable = std::vector<int>();
  m_texTable = std::vector<int>();
  m_texTableAux = std::vector<int>();
  m_materialTable = std::vector<int>();

  m_instMatricesInv = std::vector<float4x4>();
  m_instLightInstId = std::vector<int32_t>();
  m_lightIdByLightInstId = std::vector<int32_t>();
  m_meshIdByInstId = std::vector<int32_t>();

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
  // set camera
  //
  float4x4 mWorldView, mProj;
  CalcCameraMatrices(&m_modelViewInv, &m_projInv, &mWorldView, &mProj);

  const float aspect = float(m_width) / float(m_height);
  m_pHWLayer->SetCamMatrices(m_projInv.L(), m_modelViewInv.L(), mProj.L(), mWorldView.L(), aspect, DEG_TO_RAD*m_camera.fov);

  m_pHWLayer->PrepareEngineGlobals();

  if (m_usePT || m_useLT)
  {
    if (!m_ptInitDone)
    {
      m_pHWLayer->InitPathTracing(m_legacy.m_lastSeed);
      m_ptInitDone = true;
 
      auto flagsAndVars = m_pHWLayer->GetAllFlagsAndVars();
      flagsAndVars.m_flags |= HRT_UNIFIED_IMAGE_SAMPLING;

      if (m_useLT)
      {
        flagsAndVars.m_flags |= HRT_FORWARD_TRACING;
        flagsAndVars.m_flags |= HRT_DRAW_LIGHT_LT;
      }
      else
      {
        flagsAndVars.m_flags &= (~HRT_FORWARD_TRACING);
        flagsAndVars.m_flags &= (~HRT_DRAW_LIGHT_LT);
      }

      m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);
    }

    m_pHWLayer->BeginTracingPass();
    m_pHWLayer->EndTracingPass();
  }
  else
  {
    auto flagsAndVars    = m_pHWLayer->GetAllFlagsAndVars();
    flagsAndVars.m_flags = flagsAndVars.m_flags & ~HRT_UNIFIED_IMAGE_SAMPLING;
    m_pHWLayer->SetAllFlagsAndVars(flagsAndVars);

    m_pHWLayer->BeginTracingPass();
    m_pHWLayer->EndTracingPass();
    m_ptInitDone = false;
  }


  if (MEASURE_RAYS && m_usePT)
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

}

void RenderDriverRTE::InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId)
{
  if (m_pBVH == nullptr)
    return;

  int32_t meshId = a_mesh_id;
  IBVHBuilder2::InstanceInputData input;

  const int4* ldata        = (const int4*)m_pGeomStorage->GetBegin();
  const PlainMesh* pHeader = (const PlainMesh*)(ldata + m_geomTable[meshId]);
  
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
    const float4x4 invMatrix = inverse4x4(mTransform);
    m_instMatricesInv.push_back(invMatrix);
    m_instLightInstId.push_back(a_lightInstId[instId]);
    m_meshIdByInstId.push_back(meshId);
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

  res.progress    = 0.0f; //#TODO: estimate progress here !!!!
  res.finalUpdate = false;

  return res;
}

void RenderDriverRTE::GetFrameBufferHDR(int32_t w, int32_t h, float*   a_out, const wchar_t* a_layerName)
{
  m_pHWLayer->GetHDRImage((float4*)a_out, w, h);
}

void RenderDriverRTE::GetFrameBufferLDR(int32_t w, int32_t h, int32_t* a_out)
{
  m_pHWLayer->GetLDRImage((uint32_t*)a_out, w, h);
}

void RenderDriverRTE::GetGBufferLine(int32_t a_lineNumber, HRGBufferPixel* a_lineData, int32_t a_startX, int32_t a_endX) 
{

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

void RenderDriverRTE::EnableDevice(int32_t id, bool a_enable)
{
  if(a_enable)
    m_devId = id;
}

IHRRenderDriver* CreateDriverRTE(const wchar_t* a_cfg, int w, int h, int a_devId, int a_flags, IHRSharedAccumImage* a_sharedImage)
{
  return new RenderDriverRTE(a_cfg, w, h, a_devId, a_flags, a_sharedImage);
}

#pragma once

#include "../../HydraAPI/hydra_api/HydraRenderDriverAPI.h"
#include "../../HydraAPI/hydra_api/HydraInternal.h"

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <memory>

#include "IBVHBuilderAPI.h"
#include "IHWLayer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int MAX_ENV_LIGHT_PDF_SIZE = 2048;

struct MeshGeometry
{
  MeshGeometry() {}

  MeshGeometry(const MeshGeometry& a_rhs) : vertices(a_rhs.vertices), indices(a_rhs.indices) {}
  MeshGeometry(MeshGeometry&& a_rhs) : vertices(std::move(a_rhs.vertices)), indices(std::move(a_rhs.indices)) {}

  MeshGeometry& operator=(const MeshGeometry& a_rhs)
  {
    vertices = a_rhs.vertices;
    indices  = a_rhs.indices;
    return *this;
  }

  MeshGeometry& operator=(MeshGeometry&& a_rhs)
  {
    vertices = std::move(a_rhs.vertices);
    indices  = std::move(a_rhs.indices);
    return *this;
  }

  std::vector<float4> vertices;
  std::vector<int>    indices;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

struct RenderDriverRTE : public IHRRenderDriver
{
  /////////////////////////////////////////////////////////////////////////////////////////////
  RenderDriverRTE(const wchar_t* a_options, int w, int h, int a_devId, int a_flags, IHRSharedAccumImage* a_sharedImage);
  ~RenderDriverRTE();

  void              ClearAll();
  HRDriverAllocInfo AllocAll(HRDriverAllocInfo a_info);

  void GetLastErrorW(wchar_t a_msg[256]);
  /////////////////////////////////////////////////////////////////////////////////////////////

  bool UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode);
  bool UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode);
  bool UpdateLight(int32_t a_lightIdId, pugi::xml_node a_lightNode);
  bool UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t listSize);

  bool UpdateImageFromFile(int32_t a_texId, const wchar_t* a_fileName, pugi::xml_node a_texNode);
  bool UpdateMeshFromFile(int32_t a_meshId, pugi::xml_node a_meshNode, const wchar_t* a_fileName);

  bool UpdateCamera(pugi::xml_node a_camNode);
  bool UpdateSettings(pugi::xml_node a_settingsNode);

  /////////////////////////////////////////////////////////////////////////////////////////////

  void BeginScene(pugi::xml_node a_sceneNode) override;
  void EndScene();
  void InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId);
  void InstanceLights(int32_t a_light_id, const float* a_matrix, pugi::xml_node* a_lightNodes, int32_t a_instNum, int32_t a_lightGroupId);

  void Draw();

  HRRenderUpdateInfo HaveUpdateNow(int a_maxRaysPerPixel);

  void GetFrameBufferHDR(int32_t w, int32_t h, float*   a_out, const wchar_t* a_layerName) override;
  void GetFrameBufferLDR(int32_t w, int32_t h, int32_t* a_out) override;

  void EvalGBuffer() override;
  void GetGBufferLine(int32_t a_lineNumber, HRGBufferPixel* a_lineData, int32_t a_startX, int32_t a_endX, const std::unordered_set<int32_t>& a_shadowCatchers) override;

  HRDriverInfo Info();
  const HRRenderDeviceInfoListElem* DeviceList() const override;
  bool EnableDevice(int32_t id, bool a_enable) override;

  HRDriverDependencyInfo DependencyInfo();

  void BeginMaterialUpdate();
  void EndMaterialUpdate();

  void BeginTexturesUpdate();
  void EndTexturesUpdate();

  float3 GetMLTAvgBrightness() { return m_legacy.m_averageBrightness; }

  struct AOProcTexInfo
  {
    AOProcTexInfo() { rayLenSam = DummySampler(); rayLen = 0.0f; upDownType = AO_TYPE_NONE; hitOnlySameInstance = false; }

    SWTexSampler rayLenSam;
    float        rayLen;
    int          upDownType;
    bool         hitOnlySameInstance;
  };

  struct ProcTexInfo //#TODO: add aoRayLength texture parameters
  {
    ProcTexInfo() { }
    int           retT;
    std::string   call;
    AOProcTexInfo ao;
  };

protected:

  void CalcCameraMatrices(float4x4* a_pModelViewMatrixInv, float4x4* a_projMatrixInv, float4x4* a_pModelViewMatrix, float4x4* a_projMatrix);

  bool UpdateImageProc(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode);

  std::wstring m_msg;
  std::wstring m_libPath;
  size_t       m_memAllocated;

  // camera parameters
  //
  struct Camera
  {
    float3 pos;
    float3 lookAt;
    float3 up;

    float fov;
    float nearPlane;
    float farPlane;

  } m_camera;

  int   m_width;
  int   m_height;
  int   m_maxRaysPerPixel;

  IBVHBuilder2* m_pBVH;
  IHWLayer*     m_pHWLayer;

  IMemoryStorage* m_pTexStorage;
  IMemoryStorage* m_pTexStorageAux;
  IMemoryStorage* m_pGeomStorage;
  IMemoryStorage* m_pMaterialStorage;
  IMemoryStorage* m_pPdfStorage;

  std::vector<int> m_geomTable;
  std::vector<int> m_texTable;
  std::vector<int> m_texTableAux;
  std::vector<int> m_materialTable;

  std::vector< std::shared_ptr<RAYTR::ILight> >               m_lights;
  std::vector<PlainLight>                                     m_lightsInstanced;
  std::unordered_set<int>                                     m_lightHavePdfTable;
  std::unordered_map<std::wstring, int2>                      m_iesCache;
  std::unordered_map<int, std::shared_ptr<RAYTR::IMaterial> > m_materialUpdated;
  std::unordered_map<int, pugi::xml_node >                    m_materialNodes;
  std::unordered_map<std::wstring, int32_t>                   m_texturesProcessedNM;


  std::unordered_map<int, ProcTexInfo>  m_procTextures;

  using DefferedMaterialDataTuple = std::tuple<std::shared_ptr<RAYTR::IMaterial>, pugi::xml_node>;
  std::unordered_map<int, DefferedMaterialDataTuple > m_blendsToUpdate;

  IHRSharedAccumImage* m_pAccumImage;
  IHRSharedAccumImage* m_pAccumImageForGBuff;
  int m_drawPassNumber;

  float4x4 m_modelViewInv;
  float4x4 m_projInv;
 
  std::vector<float4x4> m_instMatricesInv;      ///< 
  std::vector<int32_t>  m_instLightInstId;      ///<
  std::vector<int32_t>  m_lightIdByLightInstId; ///< store light id for each light instance 
  std::vector<int32_t>  m_meshIdByInstId;
  std::vector<int32_t>  m_instIdByInstId;
  std::vector<int32_t>  m_meshRemapListId;
  std::unordered_set<int32_t>  m_skyLightsId;

  struct AlphaBuffers
  {
    std::vector<uint2> buf[MAXBVHTREES];
  } m_alphaAuxBuffers;

  float4 m_sceneBoundingSphere;
  float3 m_sceneBoundingBoxMin;
  float3 m_sceneBoundingBoxMax;

  void Error(const wchar_t* a_msg);

  template<class T>
  void Error(const wchar_t* a_msg, const T& a_val)
  {
    wchar_t temp[256];
    swprintf(temp, 256, a_msg, a_val);
    m_msg = temp;
  }

  HRDriverAllocInfo m_lastAllocInfo;

  int m_initFlags;
  int m_devId;
  int m_auxImageNumber;
  std::unordered_map<int64_t, int32_t> m_auxTexNormalsPerMat;

  bool m_useConvertedLayout;
  bool m_useBvhInstInsert;
  bool m_usePT;
  bool m_useLT;
  bool m_useIBPT;
  bool m_useMMLT;
  bool m_gpuFB;
  bool m_ptInitDone;
  bool m_firstResizeOfScreen;
  bool m_sceneHaveSkyPortals;
  bool m_needToFreeCPUMem;

  std::ifstream m_inProcTexFile;
  std::ofstream m_outProcTexFile;
  std::string   m_outProcTexFileName;
  bool m_texShadersWasRecompiled;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int32_t AuxNormalTexPerMaterial(const int32_t matId, const int32_t texId);

  void          UpdatePdfTablesForLight(int32_t a_lightId);
  const uchar4* GetAuxNormalMapFromDisaplacement(std::vector<uchar4>& normal, const PlainMaterial& mat, int textureIdNM, pugi::xml_node a_node, int* pW, int* pH);
  std::wstring  GetNormalMapParameterStringForCache(int textureIdNM, pugi::xml_node a_materialNode);
  int32_t       GetCachedAuxNormalMatId(int32_t a_matId, const PlainMaterial& a_mat, int textureIdNM, pugi::xml_node a_materialNode);
  bool          UpdateImageAux(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data);

  bool PutAbstractMaterialToStorage(const int32_t matId, std::shared_ptr<RAYTR::IMaterial> a_pMaterial, pugi::xml_node a_materialNode, bool processingBlend = false);
  bool MaterialDependsOfMaterial(pugi::xml_node a, pugi::xml_node b);

  void CreateAlphaTestTable(ConvertionResult& a_cnvRes, AlphaBuffers& a_otrData, bool& a_smoothOpacity);
  int  CountMaterialsWithAlphaTest();

  bool MeshHaveOpacity(const PlainMesh* pHeader) const;

  void BuildSkyPortalsDependencyDummyInstances(); ///< fix m_instLightInstId (instance light copies) to make sky lights and sky portals working, piece of shit 

  std::vector<float> CalcLightPickProbTable(std::vector<PlainLight>& a_inOutLights, const bool a_fwd = false);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct LegacyTiledPTVars
  {
    int minRaysPerPixel;
    int maxRaysPerPixel;
    int m_pathTracingState;
    int m_lastSeed;
    int m_currPTPassNumber;
    int m_currRaysPerPixel;

    RTE_PROGRESSBAR_CALLBACK updateProgressCall;
    float3 m_averageBrightness;

  } m_legacy;

  MRaysStat m_avgStats;
  int       m_avgStatsId;
  void AverageStats(const MRaysStat& a_stats, MRaysStat& a_statsRes, int& counter);

  void DebugSaveBVH(const std::string& a_folderName, const ConvertionResult& a_inBVH);
  void PrintBVHStat(const ConvertionResult& a_inBVH, bool traverseThem);
  void DebugPrintBVHInfo(const ConvertionResult& a_inBVH, const char* a_fileName);
  void DebugTestAlphaTestTable(const std::vector<uint2>& a_alphaTable, int a_trif4Num);

  bool m_alreadyDeleted;
  bool m_haveAtLeastOneAOMat;
  bool m_haveAtLeastOneAOMat2;

  friend void ReadBumpAndOpacity(std::shared_ptr<RAYTR::IMaterial> pResult, pugi::xml_node a_node, RenderDriverRTE* a_pRTE);

  void FreeCPUMem();

};

struct ProcTexParams
{
  std::vector<float> data;
  int32_t            texId;
};

IHRRenderDriver* CreateDriverRTE(const wchar_t* a_cfg, int w, int h, int a_devId, int a_flags, IHRSharedAccumImage* pExternalImage);


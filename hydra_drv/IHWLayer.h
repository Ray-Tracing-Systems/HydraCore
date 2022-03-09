#pragma once

#include <cstdint>

#ifndef ABSTRACT_MATERIAL_GUARDIAN
  #include "AbstractMaterial.h"
#endif

#include <vector>
#include <unordered_map>
#include <stack>

#include "FastList.h"
#include "IBVHBuilderAPI.h"
#include "IMemoryStorage.h"

#include "hydra_api/HydraAPI.h"
#include "hydra_api/HydraInternal.h"

typedef void(*RTE_PROGRESSBAR_CALLBACK)(const wchar_t* message, float a_progress);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct AllRenderVarialbes
{
  AllRenderVarialbes()
  {
    memset(m_varsI, 0, sizeof(int)*GMAXVARS);
    memset(m_varsF, 0, sizeof(int)*GMAXVARS);
    m_flags = 0;

    m_varsF[HRT_ABLOW_SCALE_X]     = 1.0f;
    m_varsF[HRT_ABLOW_SCALE_Y]     = 1.0f;
    m_varsI[HRT_SHADOW_MATTE_BACK] = INVALID_TEXTURE;
  }

  int          m_varsI[GMAXVARS];
  float        m_varsF[GMAXVARS];
  unsigned int m_flags;

  void SetVariableI(int a_name, int a_val);
  void SetVariableF(int a_name, float a_val);
  void SetFlags(unsigned int bits, unsigned int a_value);

  bool shadePassEnable(int a_bounce, int a_minBounce, int a_maxBounce);
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct InputGeom
{
  const float4* vertPos;
  const float4* vertNorm;
  const float2* vertTexCoord;
  const float4* vertTangent;
  size_t vertNum;

  const unsigned int* indices;         // of size == numIndices
  const unsigned int* materialIndices; // of size == numIndices/3
  size_t numIndices;
};


struct InputGeomBVH
{
  const BVHNode* nodes;
  size_t numNodes;

  const char* primListData;
  size_t primListSizeInBytes;

  size_t triNumInList;

};


struct MegaTexData
{
  int w, h;
  const char* data;
  bool    outOfCore;          // must be explicitly set if GPU have insuffitient memory fro current texture
  float4* lookUpTable;
  int     lookUpTableSize;
  int2*   resTable;
};


struct LightMeshData
{
  float4* positions;
  float2* texCoords;
  float2* intervals;
  size_t  size;
};

const ushort* getGgxTable();
const ushort* getTranspTable();

class IHWLayer
{
public:

  IHWLayer() : m_progressBar(nullptr), m_width(0), m_height(0), m_pExternalImage(nullptr) { InitEngineGlobals(&m_globsBuffHeader, getGgxTable(), getTranspTable()); }
  virtual ~IHWLayer();

  virtual void Clear(CLEAR_FLAGS a_flags)   = 0;

  virtual IMemoryStorage* CreateMemStorage(uint64_t a_maxSizeInBytes, const char* a_name) = 0;
  virtual void ResizeTablesForEngineGlobals(int32_t a_geomNum, int32_t a_imgNum, int32_t a_matNum, int32_t a_lightNum);

  virtual void PrepareEngineGlobals(); ///< copy constant to linear device memory for further usage them by the compute core
  virtual void PrepareEngineTables();  ///< copy tables   to linear device memory for further usage them by the compute core

  virtual void SetCamMatrices(float mProjInverse[16], float mWorldViewInverse[16], float mProj[16], float mWorldView[16], float a_aspect, float a_fovX, float3 a_lookAt);
  virtual void SetCamNode(pugi::xml_node a_camNode) { m_camNode = a_camNode; }

  virtual void SetAllBVH4(const ConvertionResult& a_convertedBVH, IBVHBuilder2* a_inBuilderAPI, int a_flags) = 0;
  virtual void SetAllInstMatrices(const float4x4* a_matrices, int32_t a_matrixNum) = 0;
  virtual void SetAllInstLightInstId(const int32_t* a_lightInstIds, int32_t a_instNum) = 0;
  virtual void SetAllPODLights(PlainLight* a_lights2, size_t a_number);

  virtual void SetAllLightsSelectTable(const float* a_table, int32_t a_tableSize, bool a_fwd = false);
  virtual void SetAllRemapLists       (const int* a_allLists, const int2* a_table, int a_allSize, int a_tableSize) {}
  virtual void SetAllInstIdToRemapId  (const int* a_allInstId, int a_instNum) {}

  // render state
  //

  virtual void SetAllFlagsAndVars(const AllRenderVarialbes& a_vars);
  virtual AllRenderVarialbes GetAllFlagsAndVars() const;

  // rendering and other
  //

  virtual void BeginTracingPass()  = 0;
  virtual void EndTracingPass()    = 0;
  virtual void EvalGBuffer(IHRSharedAccumImage* a_pAccumImage, const std::vector<int32_t>& a_instIdByInstId) {}
  virtual void FinishAll() {}

  virtual void InitPathTracing(int seed) = 0;
  virtual void ClearAccumulatedColor() = 0;
  virtual void CPUPluginFinish() {}

  // Other
  //
  virtual void ResetPerfCounters() = 0;

  virtual void ResizeScreen(int w, int h, int a_flags) { m_width = w; m_height = h; }

  virtual void GetLDRImage(uint32_t* data, int width, int height) const = 0;
  virtual void GetHDRImage(float4* data, int width, int height) const = 0;

  virtual size_t    GetAvaliableMemoryAmount(bool allMem = false) = 0;
  virtual size_t    GetMaxBufferSizeInBytes() { return GetAvaliableMemoryAmount(); }

  virtual MRaysStat GetRaysStat() = 0;
  virtual int32_t   GetRayBuffSize() const { return 0; }

  virtual const char* GetDeviceName(int* pOCLVer = nullptr) const { return "CPU (Pure C/C++)"; }

  virtual const HRRenderDeviceInfoListElem* ListDevices() const { return nullptr; }

  // crappy shit
  //
  virtual void SetRaysPerPixel(int a_num) { }
  virtual int  GetRaysPerPixel() const { return 1; }

  // programible and custom pipeline
  //
  virtual void SetNamedBuffer(const char* a_name, void* a_data, size_t a_size) {}
  virtual void CallNamedFunc(const char* a_name, const char* a_args) {}

  virtual void RenderFullScreenBuffer(const char* a_dataName, float4* a_data, int width, int height, int a_spp) 
  {
    std::vector<ushort2> pixels(m_width*m_height);
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < m_width; x++)
        pixels[y*m_width + x] = ushort2(x, y);
    }

    renderSubPixelData(a_dataName, pixels, a_spp, a_data, nullptr);
  }

  virtual bool ImplementPhotonMapping() const { return false; }
  virtual bool StoreCPUData()           const { return false; }

  // color accumulators, MLT
  //             
  virtual bool   MLT_IsAllocated() const { return true; }
  virtual size_t MLT_Alloc(int a_width, int a_height, int a_maxBounce) { return 0; }
  virtual void   MLT_Free() {}

  virtual void   SetProgressBarCallback(RTE_PROGRESSBAR_CALLBACK a_pFunc) { m_progressBar = a_pFunc; }

  // normalmap and aux computations
  //
  virtual std::vector<uchar4> NormalMapFromDisplacement(int w, int h, const uchar4* a_data, float bumpAmt, bool invHeight, float smoothLvl) { return std::vector<uchar4>(); }

  virtual void SetExternalImageAccumulator(IHRSharedAccumImage* a_pImage) { m_pExternalImage = a_pImage; } ///< pass accumulator to the HWLayer and contribute to implicit during each pass. for PT and MMLT.

  virtual void ContribToExternalImageAccumulator(IHRSharedAccumImage* a_pImage) { }                        ///< explicit 'rare' contribution (used by LT and BPT).

  virtual EngineGlobals* GetEngineGlobals(); //#NOTE: this function used for debug needs only!!!

  virtual void RecompileProcTexShaders(const std::string& a_shaderPath) {}

  virtual float GetSPP       () const { return 0.0f;}
  virtual float GetSPPDone   () const { return GetSPP(); }
  virtual float GetSPPContrib() const { return GetSPP(); }
  
protected:

  virtual void renderSubPixelData(const char* a_dataName, const std::vector<ushort2>& a_pixels, int spp, float4* a_pixValues, float4* a_subPixValues) {}

  int m_width;
  int m_height;
  pugi::xml_node m_camNode;

  AllRenderVarialbes m_vars;

  std::stack<AllRenderVarialbes> m_varsStack;
  std::stack<unsigned int>       m_flagsStack;

  EngineGlobals m_globsBuffHeader;
  RTE_PROGRESSBAR_CALLBACK m_progressBar;
  IHRSharedAccumImage*     m_pExternalImage;

  struct SpherePdfTableCPU
  {
    SpherePdfTableCPU(){}
    SpherePdfTableCPU(int a_x, int a_y, const std::vector<float2>& a_intervals) : pdfSizeX(a_x), pdfSizeY(a_y), intervals(a_intervals) {}

    std::vector<float2> intervals;

    int    pdfSizeX;
    int    pdfSizeY;
  };

  std::vector<int> m_cdataPrepared;

  std::unordered_map<std::string, IMemoryStorage*> m_allMemStorages;
  std::vector<int>                                 m_geomTable;
  std::vector<float>                               m_lightSelectTableRev;
  std::vector<float>                               m_lightSelectTableFwd;
};


size_t CalcConstGlobDataOffsets(EngineGlobals* pGlobals);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IHWLayer* CreateOclImpl(int w, int h, int a_flags, int a_deviceId);
IHWLayer* CreateCPUExpImpl(int w, int h, int a_flags);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CPUExp_Integrators.h"

class CPUSharedData : public IHWLayer
{

public:

  typedef IHWLayer Base;

  CPUSharedData(int w, int h, int a_flags);
  ~CPUSharedData();

  void SetAllBVH4(const ConvertionResult& a_convertedBVH, IBVHBuilder2* a_inBuilderAPI, int a_flags) override;
  void SetAllInstMatrices(const float4x4* a_matrices, int32_t a_matrixNum);
  void SetAllInstLightInstId(const int32_t* a_lightInstIds, int32_t a_instNum);
  void SetAllRemapLists(const int* a_allLists, const int2* a_table, int a_allSize, int a_tableSize) override;
  void SetAllInstIdToRemapId(const int* a_allInstId, int a_instNum) override;

  void PrepareEngineGlobals();
  void PrepareEngineTables();

  std::vector<uchar4> NormalMapFromDisplacement(int w, int h, const uchar4* a_data, float bumpAmt, bool invHeight, float smoothLvl);

protected:

  SceneBVHData  m_scnBVH;
  SceneMTexData m_scnTexts;
  Integrator*   m_pIntegrator;
  IBVHBuilder2* m_pBVHBuilder;

  const int32_t*     m_instLightInstId;
  const float4x4*    m_instMatrices;
  int32_t            m_instMatrixNum;

  std::vector<int>   m_remapLists;
  std::vector<int2>  m_remapTable;
  std::vector<int>   m_remapInst;

  // temp and test members
  //
  struct TempBvhData
  {
    TempBvhData() : haveInst(true), smoothOpacity(false) {}
    cvex::vector<BVHNode> m_bvh;
    cvex::vector<float4>  m_tris;
    std::vector<uint2>    m_atbl;
    bool haveInst;
    bool smoothOpacity;
  };

  TempBvhData m_bvhTrees[MAXBVHTREES];
  int m_bvhTreesNum;

  SceneGeomPointers CollectPointersForCPUIntegrator();

};


enum {GPU_RT_MEMORY_FULL_SIZE_MODE     = 2,
      GPU_RT_NOWINDOW                  = 4,
      GPU_RT_HW_LAYER_OCL              = 8,
      GPU_RT_HW_LIST_OCL_DEVICES       = 16,
      GPU_RT_LITE_CORE                 = 64,
      GPU_RT_CLEAR_SHADER_CACHE        = 256,
      GPU_RT_IN_DEVELOPMENT            = 512,
      GPU_ALLOC_FOR_COMPACT_MLT        = 1024,
      GPU_MLT_ENABLED_AT_START         = 2048,
      GPU_RT_DO_NOT_PRINT_PASS_NUMBER  = 4096,
      GPU_RT_ALLOC_INTERNAL_IMAGEB     = 8192,
      GPU_RT_CPU_FRAMEBUFFER           = 32768,
      GPU_MMLT_THREADS_262K            = 65536,
      GPU_MMLT_THREADS_131K            = 65536*2,
      GPU_MMLT_THREADS_65K             = 65536*4,
      GPU_MMLT_THREADS_16K             = 65536*8,
      };

#define RECOMPILE_PROCTEX_FROM_STRING 

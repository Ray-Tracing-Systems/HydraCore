#pragma once

#ifndef RAND_MLT_CPU
  #define RAND_MLT_CPU
#endif

#include "globals.h"
#include "crandom.h"
#include "cfetch.h"
#include "cbidir.h"
#include "ctrace.h"

//#include "cmaterial.h"
//#include "clight.h"

#ifndef ABSTRACT_MATERIAL_GUARDIAN
  #include "AbstractMaterial.h"
#endif

#include <vector>
#include <tuple>
#include <omp.h>

#include "IBVHBuilderAPI.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// old
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// old

struct SceneBVHData
{
  std::vector<BVHNode> nodes;
  std::vector<char>    primListData;
  size_t triNumInList;
};


struct SceneGeomData
{
  std::vector<float4> vertPos;
  std::vector<float2> vertTexCoord;
  std::vector<float4> vertNormUncompressed;
  std::vector<float4> vertTangentUncompressed;
  size_t              vertNum;

  std::vector<unsigned int> indices;         // of size == numIndices
  std::vector<unsigned int> materialIndices; // of size == numIndices/3
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// new

struct SceneGeomPointers
{
  SceneGeomPointers() : meshes(nullptr), matrices(nullptr), instLightInstId(nullptr), pExternalImpl(nullptr), bvhTreesNumber(0), matrixNum(0)
  {
    for (int i = 0; i < MAXBVHTREES; i++)
    {
      nodesPtr[i] = nullptr;
      primsPtr[i] = nullptr;
      alphaTbl[i] = nullptr;
      haveInst[i] = false;
    }
  }

  const BVHNode*  nodesPtr[MAXBVHTREES];
  const float4*   primsPtr[MAXBVHTREES];
  const uint2*    alphaTbl[MAXBVHTREES];
  bool            haveInst[MAXBVHTREES];

  const float4*   meshes;
  const float4x4* matrices;
  const int32_t*  instLightInstId;
  IBVHBuilder2*   pExternalImpl;

  int32_t bvhTreesNumber;
  int32_t matrixNum;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

struct SceneMTexData
{
  int width[5];
  int height[5];
  std::vector<uint8_t> ldrData[5];
  std::vector<float4>  hdrData[5];
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SurfaceInfo
{
  SurfaceInfo() { traceDepth = -1; }

  int traceDepth;
};

class Integrator
{
public:

  Integrator() : m_maxDepth(6), m_computeIndirectMLT(false), m_spp(0) {}
  virtual ~Integrator(){}

  virtual float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags) = 0;

  virtual void Reset()  = 0;
  virtual void DoPass(std::vector<uint>& a_imageLDR) = 0;

  virtual void TracePrimary(std::vector<uint>& a_imageLDR) = 0;
  virtual void TraceForTest(std::vector<uint>& a_imageLDR) { }

  virtual void GetImageHDR(float4* data, int width, int height) const = 0;
  virtual void GetImageToLDR(std::vector<uint>& a_imageLDR)     const = 0;

  // full core implemenation
  //

  virtual bool   IsFullCore() const { return false; } // is this integrator support full featured core ?
  virtual ZBlock DoTile(ZBlock a_block, bool a_cond) { return a_block; }

  virtual void ClearAccumulatedColor() {}

  virtual void SetSceneGlobals(int w, int h, EngineGlobals* a_pGlobals) = 0;
  virtual void AddSpp(int a_spp) { m_spp += a_spp; } // don't use this function!!!! Fucking shit ?!
  virtual int  GetSpp() const { return m_spp;   }

  virtual void SetConstants(EngineGlobals* a_pGlobals)     = 0;
  virtual void SetSceneGeomPtrs(SceneGeomPointers a_data)  = 0;
  virtual void SetMaterialStoragePtr(const float4* a_data) = 0;
  virtual void SetTexturesStoragePtr(const float4* a_data) = 0;
  virtual void SetPdfStoragePtr     (const float4* a_data) = 0;
  virtual void SetTexturesStorageAuxPtr(const float4* a_data) = 0;

  virtual void SetMaxDepth(int a_depth) { m_maxDepth = a_depth; }

protected:

  Integrator(const Integrator& a_rhs) {}
  Integrator& operator=(const Integrator& rhs) { return *this; }

  int  m_maxDepth;
  bool m_computeIndirectMLT;
  int  m_spp;
};


class IntegratorCommon : public Integrator
{
public:

  IntegratorCommon(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags);
  ~IntegratorCommon();

  float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags); //!!!1
	virtual float3 PathTraceVol(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags, float sigmaS = 0.0, float3 sigmaA = float3(0.0, 0.0, 0.0)) { return float3(0,0,0); };

  void Reset();
  void DoPass(std::vector<uint>& a_imageLDR);

  void TracePrimary(std::vector<uint>& a_imageLDR);
  void TraceForTest(std::vector<uint>& a_imageLDR);


  // expose them for hybrid engine usage
  //
  Lite_Hit       rayTrace(float3 a_rpos, float3 a_rdir, uint flags = 0);
  virtual float3 shadowTrace(float3 a_rpos, float3 a_rdir, float t_far, uint flags = 0);
  SurfaceHit     surfaceEval(float3 a_rpos, float3 a_rdir, Lite_Hit hit);

  GBufferAll     gbufferEval(int x, int y);

  const EngineGlobals* getEngineGlobals() const { return m_pGlobals; }
  void GetImageToLDR(std::vector<uint>& a_imageLDR) const;
  void GetImageHDR(float4* a_imageHDR, int w, int h) const;

  float3 evalDiffuseColor(float3 ray_dir, const SurfaceHit& a_hit);
  float3 EnviromnentColor(float3 a_rdir, MisData misPrev, uint flags);

  std::tuple<float3, float3> makeEyeRaySubpixel(int x, int y, float2 a_offsets);
  float3 evalAlphaTransparency(float3 ray_pos, float3 ray_dir, const SurfaceHit& currSurfaceHit, int a_currDepth);

  void SetConstants(EngineGlobals* a_pGlobals);
  void SetSceneGlobals(int w, int h, EngineGlobals* a_pGlobals);
  
  void SetSceneGeomPtrs(SceneGeomPointers a_data)  { m_geom       = a_data; }
  void SetMaterialStoragePtr(const float4* a_data) { m_matStorage = a_data; }
  void SetTexturesStoragePtr(const float4* a_data) { m_texStorage = (const int4*)a_data; }
  void SetPdfStoragePtr     (const float4* a_data) { m_pdfStorage = a_data; }
  void SetTexturesStorageAuxPtr(const float4* a_data) { m_texStorageAux = (const int4*)a_data; }

  virtual void RandomizeAllGenerators();
  void UpdateLightPickProbTableFwd(const std::vector<float>& a_table, int a_spp);

protected:

  IntegratorCommon(const IntegratorCommon& a_rhs) {}
  IntegratorCommon& operator=(const IntegratorCommon& rhs) { return *this; }

  void DebugSaveNoiseImage(const wchar_t* a_path, const float4* a_data, const float a_userCoeff);
  void DebugSaveGbufferImage(const wchar_t* a_path);
  void CalcGBufferUncompressed(std::vector<GBufferAll>& a_gbuff);
  void ExtractNoise(const float4* a_data, const float a_userCoeff,
                    std::vector<float>& errArray, float& normConst);

  void SpreadNoise(const std::vector<GBufferAll>& a_gbuff, std::vector<float>& a_noise);
  void SpreadNoise2(const std::vector<GBufferAll>& a_gbuff, std::vector<float>& a_noise);

  SceneGeomPointers m_geom;
  EngineGlobals*    m_pGlobals;
  const float4*     m_matStorage;
  const int4*       m_texStorage;
  const int4*       m_texStorageAux;
  const float4*     m_pdfStorage;

  int  m_width;
  int  m_height;
  bool m_initDoneOnce;
  bool m_splitDLByGrammar;


  virtual std::tuple<float3, float3> makeEyeRay(int x, int y);
  virtual std::tuple<float3, float3> makeEyeRay2(float x, float y);
  virtual std::tuple<float3, float3> makeEyeRay3(float4 lensOffs);

  float3 emissionEval(float3 ray_pos, float3 ray_dir, const SurfaceHit& surfElem, uint flags, const MisData misPrev, const int a_instId);
  virtual std::tuple<MatSample, int, float3> sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags = 0, float3 shadow = float3(0,0,0), bool mmltMode = false);

  GBufferAll     gbufferSample(float3 ray_pos, float3 ray_dir);

  const PlainLight* getLightFromInstId(const int a_instId);

  RandomGen& randomGen();

  constexpr static int INTEGRATOR_MAX_THREADS_NUM = 32;


  struct PerThreadData
  {
    RandomGen          gen;
    RandomGen          gen2;

    std::vector<PdfVertex> pdfArray;
    float                  pdfLightA0;

    int selectedLightIdFwd;
    int mBounceDone;

    std::string grammarCam;
    std::string grammarLit;
    std::vector<float3> vert;

    void clearPathGrammar(int a_vertNum) 
    { 
      grammarCam.clear(); grammarLit.clear(); vert.resize(a_vertNum);
      for (size_t i = 0; i < vert.size(); i++)
        vert[i] = float3(0, 0, 0);
    }
  };

  std::vector<PerThreadData> m_perThread;     
  inline PerThreadData& PerThread() { return m_perThread[omp_get_thread_num()]; }
  inline const int ThreadId() const { return omp_get_thread_num(); }

  std::vector<float4>    m_summColors;  // experimental integrators use very simple not adaptive sampling, no tiles
  float4*                m_hdrData;     // @always equal to &m_summColors[0];

  float3 Test_RayTrace(float3 ray_pos, float3 ray_dir);
  float4x4 fetchMatrix(const Lite_Hit& a_liteHit);
  int      fetchInstId(const Lite_Hit& a_liteHit);

  std::vector<float> m_lightContribFwd;
  std::vector<float> m_lightContribRev;

  unsigned int m_tableQMC[QRNG_DIMENSIONS][QRNG_RESOLUTION];
};


class IntegratorStupidPT : public IntegratorCommon
{
public:

  IntegratorStupidPT(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0) {}
  float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags);

  void SetMaxDepth(int a_depth) override { m_maxDepth = a_depth + 1; } // to have the same bounce number than MISPT

};

class IntegratorStupidPTSSS : public IntegratorCommon
{
public:

	IntegratorStupidPTSSS(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0) {}


	float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags);
	float3 PathTraceVol(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags, float sigmaS = 0.0, float3 sigmaA = float3(0.0, 0.0 ,0.0)) override;
	std::tuple<MatSample, int, float3> sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float &sigmaS, float3 &sigmaA, float3 &nonAbsorbed);
};

class IntegratorShadowPTSSS : public IntegratorCommon
{
public:

	IntegratorShadowPTSSS(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0) {}

	float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags);
	std::tuple<MatSample, int, float3> sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float3 &new_ray_pos);

	float3 PathTraceVol(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags, float sigmaS = 0.0, float3 sigmaA = float3(0.0, 0.0, 0.0)) override;
	float3 directLightSample(const SurfaceHit surfElem, const PlainMaterial* pHitMaterial, float3 ray_dir, uint flags);
	float3 directLightSampleScatter(const SurfaceHit surfElem, const float3 pos, const PlainMaterial* pHitMaterial, float3 ray_dir, uint flags, float3 sigmaA, float sigmaS);
	std::tuple<MatSample, int, float3> sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float &sigmaS, float3 &sigmaA, float3 &nonAbsorbed);
	std::tuple<MatSample, int, float3> sampleAndEvalGGXBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float &sigmaS, float3 &sigmaA, float3 &nonAbsorbed);

};

class IntegratorShadowPT : public IntegratorCommon
{
public:

  IntegratorShadowPT(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0) {}

  float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags);
};


class IntegratorMISPT : public IntegratorCommon
{
public:

  IntegratorMISPT(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags) : IntegratorCommon(w, h, a_pGlobals, a_createFlags) {}

  float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags);
};

class IntegratorMISPT_trofimm : public IntegratorCommon
{
public:

  IntegratorMISPT_trofimm(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags) : IntegratorCommon(w, h, a_pGlobals, a_createFlags) {}

  float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags);
  void DoPass(std::vector<uint>& a_imageLDR);

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct PTSamle
{
  float2 xyPixelOffset;
  float2 xyLensOffset;

  float3 color;
  float  depth;
  float3 norm;
  float3 texcolor;
  float3 shadow;

  float3 shadowDir;
  float3 reflDir;
};

struct PTSampleCompressed
{
  uint16_t   xyPxOffs[2];
  uint16_t   xyLnOffs[2];

  float3     color;
  float      depth;

  uint32_t   norm;
  uint32_t   texcolor;
  uint32_t   shadow;

  uint32_t   shadowDir;
  uint32_t   reflDir;
};

static inline void initSampleInfo(PTSamle& info)
{
  memset(&info, 0, sizeof(PTSamle));
  //info.depth  = INFINITY;
  //info.shadow = float3(1, 1, 1);
}

static inline PTSampleCompressed CompressSampleInfo(const PTSamle& a_sampleInfo)
{
  PTSampleCompressed infoC;

  float2 xyOffsPx = a_sampleInfo.xyPixelOffset*2.0f + float2(1, 1);
  float2 xyOffsLs = a_sampleInfo.xyLensOffset*2.0f  + float2(1, 1);

  infoC.xyPxOffs[0] = (uint16_t)(clamp(xyOffsPx.x*65535.0f, 0.0f, 65535.0f));
  infoC.xyPxOffs[1] = (uint16_t)(clamp(xyOffsPx.y*65535.0f, 0.0f, 65535.0f));

  infoC.xyLnOffs[0] = (uint16_t)(clamp(xyOffsLs.x*65535.0f, 0.0f, 65535.0f));
  infoC.xyLnOffs[1] = (uint16_t)(clamp(xyOffsLs.y*65535.0f, 0.0f, 65535.0f));

  infoC.color = a_sampleInfo.color;
  infoC.depth = a_sampleInfo.depth;

  infoC.norm      = encodeNormal(a_sampleInfo.norm);
  infoC.shadowDir = encodeNormal(a_sampleInfo.shadowDir);
  infoC.reflDir   = encodeNormal(a_sampleInfo.reflDir);

  infoC.texcolor  = RealColorToUint32_f3(a_sampleInfo.texcolor);
  infoC.shadow    = RealColorToUint32_f3(a_sampleInfo.shadow);

  return infoC;
}

static inline float3 unpackColor(unsigned int rgba)
{
  float3 res;
  res.x = (rgba & 0x000000FF)*(1.0f / 255.0f);
  res.y = ((rgba & 0x0000FF00) >> 8)*(1.0f / 255.0f);
  res.z = ((rgba & 0x00FF0000) >> 16)*(1.0f / 255.0f);
  return res;
}

static inline PTSamle DecompressSampleInfo(const PTSampleCompressed& a_sampleInfo)
{
  PTSamle info;

  const float invMult = 1.0f / 65535.0f;

  info.xyPixelOffset.x = float(a_sampleInfo.xyPxOffs[0])*invMult*0.5f - 0.5f;
  info.xyPixelOffset.y = float(a_sampleInfo.xyPxOffs[1])*invMult*0.5f - 0.5f;

  info.xyLensOffset.x  = float(a_sampleInfo.xyLnOffs[0])*invMult*0.5f - 0.5f;
  info.xyLensOffset.y  = float(a_sampleInfo.xyLnOffs[1])*invMult*0.5f - 0.5f;

  info.color = a_sampleInfo.color;
  info.depth = a_sampleInfo.depth;

  info.norm      = decodeNormal(a_sampleInfo.norm);
  info.shadowDir = decodeNormal(a_sampleInfo.shadowDir);
  info.reflDir   = decodeNormal(a_sampleInfo.reflDir);

  info.texcolor  = unpackColor(a_sampleInfo.texcolor);
  info.shadow    = unpackColor(a_sampleInfo.shadow);

  return info;
}


class IntegratorPSSMLT : public IntegratorMISPT // unidirectional PSSMLT
{
public:

  typedef IntegratorMISPT Base;

  IntegratorPSSMLT(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags) : Base(w, h, a_pGlobals, a_createFlags)
  {
    initRands();

    m_hystColored.resize(w,h);
    m_directLight.resize(w,h);

    memset(m_hystColored.data(), 0, sizeof(float4)*w*h);
    memset(m_directLight.data(), 0, sizeof(float4)*w*h);

    m_firstPass         = true;
    m_averageBrightness = 0.0f;
    m_ordinarySpp       = 0;
  }

  bool IsFullCore() const { return false; } // don't used tiled rendering
  void DoPass(std::vector<uint>& a_imageLDR);

protected:

  /////////////////////////////////////////////////////////////////////////////////////////////////

  inline RandomGen& randomGen2() { return m_perThread[omp_get_thread_num()].gen2; }

  /////////////////////////////////////////////////////////////////////////////////////////////////

  void DoPassDL(HDRImage4f& a_summHDR);
  void DoPassIndirectMLT(int mutationsNum);
  void DoPassIndirectOrdinaryMC(int a_spp, HDRImage4f& a_color);

  typedef std::vector<float> PSSampleV;

  HDRImage4f m_hystColored;
  HDRImage4f m_directLight;
  
  float m_averageBrightness;
  int   m_ordinarySpp;
  bool  m_firstPass;

  void initRands();

  PSSampleV initialSample_PS(int a_mutationsNum);
  PSSampleV mutatePrimarySpace(const PSSampleV& a_vec, bool* pIsLargeStep = nullptr);

  float3 F(const PSSampleV& x_ps);
};

class IntegratorLT : public IntegratorCommon
{
public:

  IntegratorLT(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0)
  {

  }

  void DoPass(std::vector<uint>& a_imageLDR) override;

protected:

  // light path
  //
  void DoLightPath(int iterId);
  void TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepthm, float3 a_accColor);
  void ConnectEye(SurfaceHit a_hit, float3 ray_dir, float3 a_accColor, int a_currBounce);

  float   mLightSubPathCount; ///< piece of shit from smallVCM

};

/**
\brief Simplified 2-way BDPT. Use light and implicit strategies.

*/
class IntegratorTwoWay : public IntegratorCommon
{
public:

  IntegratorTwoWay(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0)
  {

  }

  void DoPass(std::vector<uint>& a_imageLDR) override;

  void SetMaxDepth(int a_depth) override;

protected:

  // light path
  //
  void DoLightPath();
  void TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepthm, float a_prevLightCos, float a_prevPdfW, PerRayAcc* a_pAccData, float3 a_color);
  void ConnectEye(SurfaceHit a_hit, float3 ray_pos, float3 ray_dir, int a_currBounce, PerRayAcc* a_pAccData, float3 a_color);

  // camera path
  //
  float3 PathTrace(float3 a_rpos, float3 a_rdir);
  float3 PathTraceAcc(float3 a_rpos, float3 a_rdir, const SurfaceHit& a_prevHit, MisData misPrev, int a_currDepth, uint flags,
                      SurfaceHit* pFirstHit, PerRayAcc* a_accData);

  float   mLightSubPathCount; ///< piece of shit from smallVCM
 
};

/**
\brief Bogolepov 3-way BDPT (called "IBPT" or "truncated BDPT" by him). Use light, implicit and explicit strategies.

*/
class IntegratorThreeWay : public IntegratorCommon 
{
public:

  IntegratorThreeWay(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0)
  {

  }

  void DoPass(std::vector<uint>& a_imageLDR) override;

  void SetMaxDepth(int a_depth) override;

  void GetImageHDR(float4* a_imageHDR, int w, int h) const;

protected:

  // light path
  //
  void DoLightPath();
  void TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepthm, float a_prevLightCos, 
                      PerRayAcc* a_pAccData, float3 a_color);

  void ConnectEye(SurfaceHit a_hit, float3 ray_pos, float3 ray_dir, int a_currBounce, 
                  PerRayAcc* a_pAccData, float3 a_color);

  // camera path
  //
  float3 PathTrace(float3 a_rpos, float3 a_rdir);
  float3 PathTraceAcc(float3 a_rpos, float3 a_rdir, const float a_cosPrev, MisData misPrev, int a_currDepth, uint flags,
                      float* pPdfCamA, PerRayAcc* a_accData);

  float   mLightSubPathCount; ///< piece of shit from smallVCM

  //bool m_debugFirstBounceDiffuse;

};


/**
\brief Simplified BDPT with connecting only end points.

*/
class IntegratorSBDPT : public IntegratorCommon
{
public:

  IntegratorSBDPT(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0)
  {
   
  }

  void DoPass(std::vector<uint>& a_imageLDR) override;

  void SetMaxDepth(int a_depth) override;

protected:

  PathVertex LightPath(PerThreadData* a_perThread, int a_lightTraceDepth);

  PathVertex CameraPath(float3 a_rpos, float3 a_rdir, float3 a_prevNormal, MisData misPrev, int a_currDepth, uint flags,
                        PerThreadData* a_perThread, int a_targetDepth, bool a_haveToHitLightSource, int a_fullPathDepth);

  float3 ConnectEye(const PathVertex& a_lv, int a_lightTraceDepth,
                    PerThreadData* a_perThread, int* pX, int* pY);

  float3 ConnectShadow(const PathVertex& a_cv, PerThreadData* a_perThread, const int a_camDepth);

  float3 ConnectEndPoints(const PathVertex& a_lv, const PathVertex& a_cv, const int a_spit, const int a_depth,
                          PerThreadData* a_perThread);


  void TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float a_prevLightCos, float a_prevPdfW, 
                      float3 a_color, PerThreadData* a_perThread, int a_lightTraceDepth, bool a_wasSpecular, 
                      PathVertex* a_pOutLightVertex);

  float  mLightSubPathCount; ///< piece of shit from smallVCM

  void DebugOutCurrPath(int d);
};


/**
\brief MMLT - Metropolis sampling around SBDPT

*/

#include <map>

class IntegratorMMLT : public IntegratorCommon
{
public:

  IntegratorMMLT(int w, int h, EngineGlobals* a_pGlobals) : IntegratorCommon(w, h, a_pGlobals, 0), m_mask(nullptr)
  {
    m_firstPass = true;
    m_direct.resize(w,h);
    memset(m_direct.data(), 0, w*h * sizeof(float) * 4);
    memset(m_hdrData,       0, w*h * sizeof(float) * 4);
  }

  IntegratorMMLT(int w, int h, EngineGlobals* a_pGlobals, float4* pIndirectImage) : IntegratorCommon(w, h, a_pGlobals, 0), m_mask(nullptr)
  {
    m_firstPass = true;
    m_direct.resize(1, 1);
    m_summColors.resize(1);
    m_hdrData = pIndirectImage;
  }

  void DoPass(std::vector<uint>& a_imageLDR) override;
  void SetMaxDepth(int a_depth) override;

  void SetMaskPtr(const float* a_ptr) { m_mask = a_ptr; }

  float DoPassEstimateAvgBrightness();
  void  DoPassDirectLight(float4* a_outImage);
  void  DoPassIndirectMLT(float4* a_outImage);
  float EstimateScaleCoeff() const;

  void GetImageHDR(float4* a_imageHDR, int w, int h) const;

protected:

  void DoPassIndirectMLT(int d, float a_bkScale, float4* a_outImage);

  PathVertex LightPath(PerThreadData* a_perThread, int a_lightTraceDepth);

  PathVertex CameraPath(float3 ray_pos, float3 ray_dir, float3 a_prevNormal, MisData a_misPrev, int a_currDepth, uint flags,
                        PerThreadData* a_perThread, int a_targetDepth, bool a_haveToHitLightSource, int a_fullPathDepth);

  float3 ConnectEye(const PathVertex& a_lv, int a_lightTraceDepth,
                    PerThreadData* a_perThread, int* pX, int* pY);

  float3 ConnectShadow(const PathVertex& a_cv, PerThreadData* a_perThread, const int a_camDepth);

  float3 ConnectEndPoints(const PathVertex& a_lv, const PathVertex& a_cv, const int a_spit, const int a_depth,
                          PerThreadData* a_perThread);


  void TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float a_prevLightCos, float a_prevPdfW, 
                      float3 a_color, PerThreadData* a_perThread, int a_lightTraceDepth, bool a_wasSpecular, 
                      PathVertex* a_pOutLightVertex);

  float3 PathTraceDirectLight(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags);
 
  float  mLightSubPathCount; ///< piece of shit from smallVCM

  typedef std::vector<float> PSSampleV;

  PSSampleV  m_pss       [INTEGRATOR_MAX_THREADS_NUM]; // primary space samples
  PathVertex m_oldLightV [INTEGRATOR_MAX_THREADS_NUM];
  PathVertex m_oldCameraV[INTEGRATOR_MAX_THREADS_NUM];
  bool  m_firstPass;
  float m_avgBrightness;
  std::vector<float> m_avgBPerBounce;

  enum MUTATION_TYPE { MUTATE_LIGHT = 1, MUTATE_CAMERA = 2 };

  float3    F(const PSSampleV& a_xVec, const int d, int m_type, int* pX, int* pY);
  PSSampleV InitialSamplePS(const int d, const int a_burnIters = 0);
  PSSampleV MutatePrimarySpace(const PSSampleV& a_vec, int d, int* pMutationType);

  void MutateLightPart(PSSampleV& a_vec, int s, RandomGen* pGen);
  void MutateCameraPart(PSSampleV& a_vec, int s, RandomGen* pGen);

  HDRImage4f   m_direct;
  const float* m_mask;

  // to find problem paths
  //
  void DebugSaveBadPaths();
  void DebugLoadPaths();

  struct PathShot
  {
    std::vector<float>  randNumbers;
    std::vector<float4> vpos;
  };
  std::map<float, PathShot> m_debugRaysHeap[INTEGRATOR_MAX_THREADS_NUM];
  std::vector<float4>       m_debugRaysPos [INTEGRATOR_MAX_THREADS_NUM];

};


float3 EstimateAverageBrightnessRGB(const HDRImage4f& a_color);
float  EstimateAverageBrightness   (const HDRImage4f& a_color);
float3 EstimateAverageBrightnessRGB(const std::vector<float4>& a_color);
float  EstimateAverageBrightness   (const std::vector<float4>& a_color);

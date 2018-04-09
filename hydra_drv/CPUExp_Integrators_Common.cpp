#include <omp.h>

#include "CPUExp_Integrators.h"
#include "ctrace.h"

#include <math.h>
#include <algorithm>
#include <assert.h>

void IntegratorCommon::SetConstants(EngineGlobals* a_pGlobals)
{
  m_pGlobals = a_pGlobals;

  // init contrib tables
  //
  if (m_pGlobals->lightsNum != m_lightContribFwd.size())
  {
    m_lightContribFwd.resize(m_pGlobals->lightsNum);
    m_lightContribRev.resize(m_pGlobals->lightsNum);

    for (size_t i = 0; i < m_lightContribFwd.size(); i++)
    {
      m_lightContribFwd[i] = 0.0f;
      m_lightContribRev[i] = 0.0f;
    }
  }

  m_splitDLByGrammar = (a_pGlobals->varsI[HRT_MMLT_FIRST_BOUNCE] > 3);
}


void IntegratorCommon::SetSceneGlobals(int w, int h, EngineGlobals* a_pGlobals)
{
  SetConstants(a_pGlobals);

  m_width  = w;
  m_height = h;
  
  if (!m_initDoneOnce)
  {
    for (int i = 0; i < int(m_perThread.size()); i++)
    {
      m_perThread[i].gen  = RandomGenInit(i*GetTickCount());
      m_perThread[i].gen2 = RandomGenInit(i*i*GetTickCount() + i + 7);
      m_perThread[i].pdfArray.resize(a_pGlobals->varsI[HRT_TRACE_DEPTH] + 1);
    }
  }
  m_initDoneOnce = true;

  m_summColors.resize(m_width*m_height);
  m_spp = 0;

}

void IntegratorCommon::RandomizeAllGenerators()
{
  if (m_spp % 17 == 0)
  {
    for (int i = 0; i < m_perThread.size(); i++)
    {
      m_perThread[i].gen  = RandomGenInit(GetTickCount() + rand()*i);
      m_perThread[i].gen2 = RandomGenInit((GetTickCount()*rand() + rand() + i)*m_spp + i*i*i * 19 + i*13 + 7);
    }
  }
}

extern "C" void initQuasirandomGenerator(unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]);

IntegratorCommon::IntegratorCommon(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags) : m_initDoneOnce(false), m_matStorage(nullptr)
{
  m_summColors.resize(w*h);
  m_hdrData = &m_summColors[0];
  memset(m_hdrData, 0, w*h * sizeof(float4));

  SetSceneGlobals(w, h, a_pGlobals);

  m_perThread.resize(INTEGRATOR_MAX_THREADS_NUM);
  for (int i = 0; i < int(m_perThread.size()); i++)
  {
    m_perThread[i].gen  = RandomGenInit(i*GetTickCount());
    m_perThread[i].gen2 = RandomGenInit(i*GetTickCount() + i*i + 1);
  }

  m_splitDLByGrammar = false;
  initQuasirandomGenerator(m_tableQMC);

  m_remapAllLists = nullptr; m_remapAllSize  = 0;
  m_remapTable    = nullptr; m_remapTabSize  = 0;
  m_remapInstTab  = nullptr; m_remapInstSize = 0;
}


IntegratorCommon::~IntegratorCommon()
{
  
}

void IntegratorCommon::SetMaterialRemapListPtrs(const int* a_allLists, const int2* a_table, const int* a_instTab,
                                                const int a_size1, const int a_size2, const int a_size3)
{
  m_remapAllLists = a_allLists;
  m_remapTable    = a_table;
  m_remapInstTab  = a_instTab;
  m_remapAllSize  = a_size1;
  m_remapTabSize  = a_size2;
  m_remapInstSize = a_size3;
}


void IntegratorCommon::Reset()
{
  m_spp = 0;
  for (size_t i = 0; i < m_summColors.size(); i++)
    m_summColors[i] = float4(0, 0, 0, 0);
}

Lite_Hit IntegratorCommon::rayTrace(float3 a_rpos, float3 a_rdir, uint flags)
{
  if (m_geom.pExternalImpl != nullptr)
  {
    return m_geom.pExternalImpl->RayTrace(a_rpos, a_rdir);
  }
  else if (m_geom.bvhTreesNumber > 0 && m_geom.nodesPtr[0] != nullptr)
  {
    float  t_rayMin  = 0.0f;
    Lite_Hit liteHit = Make_Lite_Hit(MAXFLOAT, -1);

    for (int i = 0; i < m_geom.bvhTreesNumber; i++)
    {
      const float4* bvhdata = (const float4*)m_geom.nodesPtr[i];
      const float4* tridata = (const float4*)m_geom.primsPtr[i];
      const uint2*  alfdata = m_geom.alphaTbl[i];

      if (m_geom.haveInst[i])
      {
        if (m_geom.alphaTbl[i] != nullptr)
          liteHit = BVH4InstTraverseAlpha(a_rpos, a_rdir, t_rayMin, liteHit, bvhdata, tridata, alfdata, m_texStorage, m_pGlobals);
        else
          liteHit = BVH4InstTraverse(a_rpos, a_rdir, t_rayMin, liteHit, bvhdata, tridata);
      }
      else
        liteHit = BVH4Traverse(a_rpos, a_rdir, t_rayMin, liteHit, bvhdata, tridata);
    }

    return liteHit;
  }
  else
    return Lite_Hit(); // BVHTraversalA_SSE(a_rpos, a_rdir, 0.0f, flags, scnOld.inputBVH, scnOld.inputObjList, scnOld.vertIndices, scnOld.vertTexCoord, MEGATEX_OPACITY, m_pGlobals);
}

float3 IntegratorCommon::shadowTrace(float3 a_rpos, float3 a_rdir, float t_far, uint flags)
{
  if (m_geom.pExternalImpl != nullptr)
  {
    return m_geom.pExternalImpl->ShadowTrace(a_rpos, a_rdir, t_far);
  }
  else if (m_geom.bvhTreesNumber > 0 && m_geom.nodesPtr[0] != nullptr)
  {
    float  t_rayMin = 0.0f;
    Lite_Hit liteHit = Make_Lite_Hit(MAXFLOAT, -1);

    // for(int i=0;i<m_geom.bvhTreesNumber;i++)
    float4* bvhdata = (float4*)m_geom.nodesPtr[0];
    float4* tridata = (float4*)m_geom.primsPtr[0];

    //liteHit = BVH4Traverse(ray_pos, ray_dir, t_rayMin, liteHit, bvhdata, tridata);
    liteHit = BVH4InstTraverse(a_rpos, a_rdir, t_rayMin, liteHit, bvhdata, tridata);
    return (HitSome(liteHit) && liteHit.t > 0.0f && liteHit.t < t_far) ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(1.0f, 1.0f, 1.0f);
  }
  else
  {
    Lite_Hit hit = Lite_Hit(); // BVHTraversalA_SSE(a_rpos, a_rdir, 0.0f, flags, scnOld.inputBVH, scnOld.inputObjList, scnOld.vertIndices, scnOld.vertTexCoord, MEGATEX_OPACITY, m_pGlobals);
    return (HitSome(hit) && hit.t > 0.0f && hit.t < t_far*0.9995f) ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(1.0f, 1.0f, 1.0f);
  }
}

float4x4 IntegratorCommon::fetchMatrix(const Lite_Hit& hit)
{
  return m_geom.matrices[hit.instId];
}

int IntegratorCommon::fetchInstId(const Lite_Hit& a_liteHit)
{
  return a_liteHit.instId;
}


SurfaceHit IntegratorCommon::surfaceEval(float3 a_rpos, float3 a_rdir, Lite_Hit hit)
{
  // (1) mul ray with instanceMatrixInv
  //
  const float4x4 instanceMatrixInv = fetchMatrix(hit); 
  
  const float3 rayPosLS = mul4x3(instanceMatrixInv, a_rpos);
  const float3 rayDirLS = mul3x3(instanceMatrixInv, a_rdir);

  // (2) get pointer to PlainMesh via hit.geomId
  //
  const PlainMesh* mesh = fetchMeshHeader(hit, m_geom.meshes, m_pGlobals);

  // (3) intersect transformed ray with triangle and get SurfaceHit in local space
  //
  const SurfaceHit surfHit = surfaceEvalLS(rayPosLS, rayDirLS, hit, mesh);
 
  // (4) get transformation to wold space
  //
  const float4x4 instanceMatrix = inverse4x4(instanceMatrixInv);

  // (5) transform SurfaceHit to world space with instanceMatrix
  //
  SurfaceHit surfHitWS = surfHit;

  const float3 transformedNormal = mul3x3(instanceMatrix, surfHit.normal);
  const float  lengthInv         = 1.0f / length(transformedNormal);

  const float multInv            = 1.0f / sqrt(3.0f);
  const float3 shadowStartPos    = mul3x3(instanceMatrix, make_float3(multInv*surfHitWS.sRayOff, multInv*surfHitWS.sRayOff, multInv*surfHitWS.sRayOff));

  surfHitWS.pos        = mul4x3(instanceMatrix, surfHit.pos);
  surfHitWS.normal     = lengthInv*transformedNormal;
  surfHitWS.flatNormal = lengthInv*mul3x3(instanceMatrix, surfHit.flatNormal);
  surfHitWS.tangent    = lengthInv*mul3x3(instanceMatrix, surfHit.tangent);
  surfHitWS.biTangent  = lengthInv*mul3x3(instanceMatrix, surfHit.biTangent);
  surfHitWS.t          = length(surfHitWS.pos - a_rpos); // seems this is more precise. VERY strange !!!
  surfHitWS.sRayOff    = length(shadowStartPos);

  if (m_remapAllLists != nullptr && m_remapTable != nullptr && m_remapInstTab != nullptr)
  {
    surfHitWS.matId = remapMaterialId(surfHitWS.matId, hit.instId,
                                      m_remapInstTab, m_remapInstSize, m_remapAllLists, 
                                      m_remapTable, m_remapTabSize);
  }
 
  return surfHitWS;
}

float3 IntegratorCommon::evalDiffuseColor(float3 ray_dir, const SurfaceHit& a_hit)
{
  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, a_hit.matId);
  return materialEvalDiffuse(pHitMaterial, ray_dir, a_hit.normal, a_hit.texCoord, m_pGlobals, m_texStorage);
}

float3 IntegratorCommon::EnviromnentColor(float3 a_rdir, MisData misPrev, uint flags)
{
  return environmentColor(a_rdir, misPrev, flags, m_pGlobals, m_matStorage, m_pdfStorage, m_texStorage);
}

float3 IntegratorCommon::PathTrace(float3 a_rpos, float3 a_rdir, MisData a_prevSample, int a_currDepth, uint flags)
{
  Lite_Hit hit = rayTrace(a_rpos, a_rdir);

  if (HitSome(hit))
  {
    SurfaceHit surfElem = surfaceEval(a_rpos, a_rdir, hit);

    float3 hit_norm = surfElem.normal;

    hit_norm.x = fabs(hit_norm.x);
    hit_norm.y = fabs(hit_norm.y);
    hit_norm.z = fabs(hit_norm.z);

    return hit_norm;
  }
  else
    return float3(0, 0, 1);
}

void IntegratorCommon::DoPass(std::vector<uint>& a_imageLDR)
{
  if (m_width*m_height != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  // Update HDR image
  //
  const float alpha = 1.0f / float(m_spp + 1);

  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {
      float3 ray_pos, ray_dir;
      std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);

			const float3 color = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0); 
      const float maxCol = maxcomp(color);

      m_summColors[y*m_width + x] = m_summColors[y*m_width + x] * (1.0f - alpha) + to_float4(color, maxCol)*alpha;
    }
  }

  RandomizeAllGenerators();
  
  m_spp++;
  GetImageToLDR(a_imageLDR);
  
  //if (m_spp == 1)
    //DebugSaveGbufferImage(L"C:/[Hydra]/rendered_images/torus_gbuff");

  std::cout << "IntegratorCommon: spp = " << m_spp << std::endl;
}


void IntegratorCommon::GetImageToLDR(std::vector<uint>& a_imageLDR) const
{
  // get HDR to LDR
  //
  float gammaPow = 1.0f / m_pGlobals->varsF[HRT_IMAGE_GAMMA];  // gamma correction

  for (size_t i = 0; i < a_imageLDR.size(); i++)
  {
    float4 color = ToneMapping4(m_summColors[i]);

    color.x = powf(color.x, gammaPow);
    color.y = powf(color.y, gammaPow);
    color.z = powf(color.z, gammaPow);

    a_imageLDR[i] = RealColorToUint32(color);
  }
}

void IntegratorCommon::GetImageHDR(float4* a_imageHDR, int w, int h) const
{
  if (w != m_width || h != m_height)
  {
    std::cout << "IntegratorCommon::GetImageHDR, bad input resolution" << std::endl;
    return;
  }

  memcpy(a_imageHDR, &m_summColors[0], m_width*m_height * sizeof(float4));
}


std::tuple<float3,float3> IntegratorCommon::makeEyeRay(int x, int y)
{
  EngineGlobals* a_globals = m_pGlobals;

  RandomGen& gen = randomGen();
  float4 offsets = rndUniform(&gen, -1.0f, 1.0f);

  float3 ray_pos, ray_dir;
  MakeRandEyeRay(x, y, m_width, m_height, offsets, m_pGlobals, 
                 &ray_pos, &ray_dir);

  return std::make_tuple(ray_pos, ray_dir);
}


std::tuple<float3, float3> IntegratorCommon::makeEyeRay2(float x, float y)
{
  EngineGlobals* a_globals = m_pGlobals;
  float4x4 a_mViewProjInv  = make_float4x4(a_globals->mProjInverse);
  float4x4 a_mWorldViewInv = make_float4x4(a_globals->mWorldViewInverse);

  float3 ray_pos = make_float3(0.0f, 0.0f, 0.0f);
  float3 ray_dir = EyeRayDir(x, y, float(m_width), float(m_height), a_mViewProjInv);

  ray_dir = tiltCorrection(ray_pos, ray_dir, a_globals);

  matrix4x4f_mult_ray3(a_mWorldViewInv, &ray_pos, &ray_dir);

  return std::make_tuple(ray_pos, ray_dir);
}

std::tuple<float3, float3> IntegratorCommon::makeEyeRay3(float4 lensOffs)
{
  const float xPosPs = lensOffs.x;
  const float yPosPs = lensOffs.y;
  const float x      = m_width*xPosPs;
  const float y      = m_height*yPosPs;
  return makeEyeRay2(x, y);
}


std::tuple<float3, float3> IntegratorCommon::makeEyeRaySubpixel(int x, int y, float2 a_offsets)
{
  EngineGlobals* a_globals = m_pGlobals;

  float4 offsets(2.0f*a_offsets.x - 1.0f, 2.0f*a_offsets.y - 1.0f, 0.0f, 0.0f);

  float3 ray_pos, ray_dir;
  MakeRandEyeRay(x, y, m_width, m_height, offsets, m_pGlobals, 
                 &ray_pos, &ray_dir);

  return std::make_tuple(ray_pos, ray_dir);
}


void IntegratorCommon::TracePrimary(std::vector<uint>& a_imageLDR)
{
  if (m_width*m_height != a_imageLDR.size())
    RUN_TIME_ERROR("TracePrimary: bad output bufffer size");

  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {
      float3 ray_pos, ray_dir;
      std::tie(ray_pos,ray_dir) = makeEyeRay(x, y);

      if (x == 512 && y == 125)
        int a = 2;
      if (x == 667 && y == 487)
        int a = 2;

      float3 color = IntegratorCommon::PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0); // IntegratorCommon::PathTrace

      a_imageLDR[y*m_width + x] = RealColorToUint32(to_float4(ToneMapping(color), 0.0f));
    }

  }

}


float3 IntegratorCommon::Test_RayTrace(float3 ray_pos, float3 ray_dir)
{
  Lite_Hit liteHit = rayTrace(ray_pos, ray_dir);

  //const float3 colorTable[8] = {float3(1,0,0),           float3(0,1,0),          float3(0,0,1),            float3(1,0.5f,0),
  //                              float3(0.25f,1.0f,0.25), float3(0.5f,0.5f,1.0f), float3(0.5f, 0.5f ,0.5f), float3(1.0f,0.25f,0.5f) };
  //
  //int id           = 0;
  //
  //if (m_geom.pExternalImpl != nullptr)
  //{
  //  id = liteHit.primId;
  //}
  //else
  //{
  //  float4* tridata = (float4*)m_geom.primsPtr[0];
  //  const float4* pTri = tridata + liteHit.primId;
  //  id = as_int(pTri->w);
  //}
  //
  //if (HitNone(liteHit))
  //  return float3(0, 0, 0);
  //else
  //  return colorTable[(id % 8)];

  if (HitNone(liteHit))
    return float3(0, 0, 0);
  else
  {
    auto surfHit = surfaceEval(ray_pos, ray_dir, liteHit);
    return surfHit.normal;
  }

}

void IntegratorCommon::TraceForTest(std::vector<uint>& a_imageLDR)
{
  if (m_width*m_height != a_imageLDR.size())
    RUN_TIME_ERROR("TracePrimary: bad output bufffer size");
  
  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {
      float3 ray_pos, ray_dir;
      std::tie(ray_pos,ray_dir) = makeEyeRay(x, y);
  
      if (x == 418 && y == 119)
      {
        int a = 2;
      }
      
      float3 color = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

      //float3 color = Test_RayTrace(ray_pos, ray_dir);
      //color.x = fabs(color.x);
      //color.y = fabs(color.y);
      //color.z = fabs(color.z);

      color.x = fminf(color.x, 1.0f);
      color.y = fminf(color.y, 1.0f);
      color.z = fminf(color.z, 1.0f);

      a_imageLDR[y*m_width + x] = RealColorToUint32(to_float4(color, 0.0f));
    }
  
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////

const PlainLight* IntegratorCommon::getLightFromInstId(const int a_instId)
{
  if (m_pGlobals->lightsNum == 0)
    return nullptr;

  const int lightOffset = m_geom.instLightInstId[a_instId];

  if (lightOffset >= 0)
    return lightAt(m_pGlobals, lightOffset);
  else
    return nullptr;
}


float3 IntegratorCommon::emissionEval(const float3 ray_pos, const float3 ray_dir, const SurfaceHit& surfElem, uint flags, const MisData misPrev, const int a_instId)
{
  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

  float3 normal;
  if (surfElem.hfi)
    normal = (-1.0f)*surfElem.normal;
  else
    normal = surfElem.normal;

  if (dot(ray_dir, normal) < 0.0f)
  {
    float3 outPathColor = materialEvalEmission(pHitMaterial, ray_dir, normal, surfElem.texCoord, m_pGlobals, m_texStorage, m_texStorage); // a_shadingTexture, a_shadingTextureHDR
   
		if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_FORBID_EMISSIVE_GI) && unpackBounceNumDiff(flags) > 0)
			outPathColor = float3(0, 0, 0);

    if (m_pGlobals->lightsNum > 0)
    {
      const int lightOffset = m_geom.instLightInstId[a_instId];
      if (lightOffset >= 0)
      {
        __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
        outPathColor = lightGetIntensity(pLight, ray_pos, ray_dir, normal, surfElem.texCoord, flags, misPrev, m_pGlobals, m_texStorage, m_pdfStorage); // a_shadingTexture, a_shadingTextureHDR
      }
    }

    return outPathColor;
  }
  else
    return float3(0, 0, 0);
}



RandomGen& IntegratorCommon::randomGen()
{
  return m_perThread[omp_get_thread_num()].gen;
}

std::tuple<MatSample, int, float3> IntegratorCommon::sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, bool a_mmltMode)
{
  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  auto& gen = randomGen();

  int matOffset = materialOffset(m_pGlobals, surfElem.matId);
  const int rayBounceNum   = unpackBounceNum(flags);
  const uint otherRayFlags = unpackRayFlags(flags);
   
  const bool canSampleReflOnly    = (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY) != 0;
  const bool sampleReflectionOnly = ((otherRayFlags & RAY_GRAMMAR_DIRECT_LIGHT) != 0) && canSampleReflOnly; 

  BRDFSelector mixSelector = materialRandomWalkBRDF(pHitMaterial, &gen, gen.rptr, ray_dir, surfElem.normal, surfElem.texCoord, m_pGlobals, m_texStorage, rayBounceNum, a_mmltMode, sampleReflectionOnly); // a_shadingTexture

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// --- >

  if ((m_pGlobals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_RADIANCE || m_pGlobals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_PRIMARY) && (m_pGlobals->varsI[HRT_RENDER_LAYER_DEPTH] == rayBounceNum)) // piss execution way (render incoming radiance)
  {
    matOffset     = m_pGlobals->varsI[HRT_WHITE_DIFFUSE_OFFSET];
    pHitMaterial  = materialAtOffset(m_matStorage, m_pGlobals->varsI[HRT_WHITE_DIFFUSE_OFFSET]);
    mixSelector.w = 1.0f;
  }
  else // normal execution way
  {
		// this is needed component for mis further (misNext.prevMaterialOffset = matOffset;)
		//
    matOffset    = matOffset    + mixSelector.localOffs*(sizeof(PlainMaterial) / sizeof(float4));   
    pHitMaterial = pHitMaterial + mixSelector.localOffs;
  }

  // const bool hitFromBack        = (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE) != 0;
  // const bool hitGlassFromInside = (materialGetType(pHitMaterial) == PLAIN_MAT_CLASS_GLASS) && hitFromBack;

  ShadeContext sc;

  sc.wp  = surfElem.pos;
  sc.l   = ray_dir; 
  sc.v   = ray_dir;
  sc.n   = surfElem.normal;
  sc.fn  = surfElem.flatNormal;
  sc.tg  = surfElem.tangent;
  sc.bn  = surfElem.biTangent;
  sc.tc  = surfElem.texCoord;
  sc.hfi = surfElem.hfi;

  const float3 rands   = a_mmltMode ? rndMatMMLT(&gen, gen.rptr, rayBounceNum) : rndMat(&gen, gen.rptr, rayBounceNum);
  MatSample brdfSample;
  MaterialLeafSampleAndEvalBRDF(pHitMaterial, rands, &sc, shadow, m_pGlobals, m_texStorage, m_texStorageAux,
                                &brdfSample);
  brdfSample.pdf *= mixSelector.w;

  return std::make_tuple(brdfSample, matOffset, make_float3(1,1,1));
}


float3 IntegratorCommon::evalAlphaTransparency(float3 ray_pos, float3 ray_dir, const SurfaceHit& currSurfaceHit, int a_currDepth)
{
  const PlainMaterial* pHitMaterial  = materialAt(m_pGlobals, m_matStorage, currSurfaceHit.matId);
  TransparencyAndFog matFogAndTransp = materialEvalTransparencyAndFog(pHitMaterial, ray_dir, currSurfaceHit.normal, currSurfaceHit.texCoord, m_pGlobals, nullptr);

  if (length(matFogAndTransp.transparency) < 1e-4f || a_currDepth > 16)
    return make_float3(0.0f, 0.0f, 0.0f);
  else
  {
    float3 hitPos      = currSurfaceHit.pos;
    float offsetLength = 10.0f*fmax(fmax(fabs(hitPos.x), fmax(fabs(hitPos.y), fabs(hitPos.z))), GEPSILON)*GEPSILON;

    float3 nextPos = hitPos + ray_dir*offsetLength;
    float3 nextDir = ray_dir;

    auto hit = this->rayTrace(nextPos, nextDir);
    if (HitNone(hit))
      return make_float3(1, 1, 1);
    else
    {
      auto surf = this->surfaceEval(ray_pos, ray_dir, hit);
      return matFogAndTransp.transparency*evalAlphaTransparency(nextPos, nextDir, surf, a_currDepth+1);
    }
  }

}


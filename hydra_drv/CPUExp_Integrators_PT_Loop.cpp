#include <omp.h>
#include "CPUExp_Integrators.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int tid = 0;

void IntegratorMISPTLoop2::kernel_InitAccumData(float3* accumColor, float3* accumuThoroughput, float3* currColor)
{
  accumColor       [tid] = make_float3(0,0,0);
  accumuThoroughput[tid] = make_float3(1,1,1);
  currColor        [tid] = make_float3(0,0,0);
}

void IntegratorMISPTLoop2::kernel_RayTrace(const float3* ray_pos, const float3* ray_dir, 
                                           Lite_Hit* hit)
{
  //#NOTE: must be replaced by some internal implementation
  hit[tid] = rayTrace(ray_pos[tid], ray_dir[tid]); 
}

bool IntegratorMISPTLoop2::kernel_HitEnvironment(const float3* ray_dir, const Lite_Hit* hit, const MisData* misPrev, const uint* flags,
                                                 float3* currColor)
{
  if (HitNone(hit[tid]))
  {
    currColor[tid] = environmentColor(ray_dir[tid], misPrev[tid], flags[tid], m_pGlobals, m_matStorage, m_pdfStorage, m_texStorage);
    return true;
  }

  return false;
}

void IntegratorMISPTLoop2::kernel_EvalSurface(const float3* ray_pos, const float3* ray_dir, const Lite_Hit* hit,
                                              SurfaceHit* surfElem)
{
  //surfElem = surfaceEval(ray_pos, ray_dir, hit); // TODO: probably we can cal some class functions like 'surfaceEval'

  // (1) mul ray with instanceMatrixInv
  //
  const Lite_Hit lHit = hit[tid];

  const float4x4 instanceMatrixInv = fetchMatrix(lHit); 
  
  const float3 rayPosLS = mul4x3(instanceMatrixInv, ray_pos[tid]);
  const float3 rayDirLS = mul3x3(instanceMatrixInv, ray_dir[tid]);

  // (2) get pointer to PlainMesh via hit.geomId
  //
  const PlainMesh* mesh = fetchMeshHeader(lHit, m_geom.meshes, m_pGlobals);

  // (3) intersect transformed ray with triangle and get SurfaceHit in local space
  //
  const SurfaceHit surfHit = surfaceEvalLS(rayPosLS, rayDirLS, lHit, mesh);
 
  // (4) get transformation to wold space
  //
  const float4x4 instanceMatrix = inverse4x4(instanceMatrixInv);

  // (5) transform SurfaceHit to world space with instanceMatrix
  //
  SurfaceHit surfHitWS = surfHit;

  const float multInv            = 1.0f / sqrt(3.0f);
  const float3 shadowStartPos    = mul3x3(instanceMatrix, make_float3(multInv*surfHitWS.sRayOff, multInv*surfHitWS.sRayOff, multInv*surfHitWS.sRayOff));

  const float4x4 normalMatrix = transpose(instanceMatrixInv);

  surfHitWS.pos        = mul4x3(instanceMatrix, surfHit.pos);
  surfHitWS.normal     = normalize( mul3x3(normalMatrix, surfHit.normal) );
  surfHitWS.flatNormal = normalize( mul3x3(normalMatrix, surfHit.flatNormal));
  surfHitWS.tangent    = normalize( mul3x3(normalMatrix, surfHit.tangent));
  surfHitWS.biTangent  = normalize( mul3x3(normalMatrix, surfHit.biTangent));
  surfHitWS.t          = length(surfHitWS.pos - ray_pos[tid]); // seems this is more precise. VERY strange !!!
  surfHitWS.sRayOff    = length(shadowStartPos);

  if (m_remapAllLists != nullptr && m_remapTable != nullptr && m_remapInstTab != nullptr)
  {
    surfHitWS.matId = remapMaterialId(surfHitWS.matId, lHit.instId,
                                      m_remapInstTab, m_remapInstSize, m_remapAllLists, 
                                      m_remapTable, m_remapTabSize);
  }
 
  surfElem[tid] = surfHitWS;
}

bool IntegratorMISPTLoop2::kernel_EvalEmission(const float3* ray_pos, const float3* ray_dir, 
                                               const SurfaceHit* surfElem, const uint* flags, 
                                               const MisData* misPrev, const Lite_Hit* hit,
                                               const int depth,
                                               float3* currColor)
{

  float3 emission = emissionEval(ray_pos[tid], ray_dir[tid], surfElem[tid], flags[tid], misPrev[tid], fetchInstId(hit[tid]));
  
  if (dot(emission, emission) > 1e-3f)
  {
    //const PlainLight* pLight = getLightFromInstId(fetchInstId(hit)); // TODO: replace with C99 function
    //                                                                 // TODO: probably we can cal some class functions like 'getLightFromInstId'
    
    // (1) get light pointer
    //
    const PlainLight* pLight = nullptr;
    if (m_pGlobals->lightsNum != 0)
    {
      const int lightOffset = m_geom.instLightInstId[fetchInstId(hit[tid])];
      if (lightOffset >= 0)
        pLight = lightAt(m_pGlobals, lightOffset);
      else
        pLight = nullptr;
    }

    // (2) evaluate light color
    //
    if (pLight != nullptr)
    {
      float lgtPdf    = lightPdfSelectRev(pLight)*lightEvalPDF(pLight, ray_pos[tid], ray_dir[tid], surfElem[tid].pos, surfElem[tid].normal, surfElem[tid].texCoord, m_pdfStorage, m_pGlobals);
      float bsdfPdf   = misPrev[tid].matSamplePdf;
      float misWeight = misWeightHeuristic(bsdfPdf, lgtPdf);  // (bsdfPdf*bsdfPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);

      if (misPrev[tid].isSpecular)
        misWeight = 1.0f;

      currColor[tid] = emission*misWeight;
      return true;
    }
    else
    {
      currColor[tid] = emission;
      return true;
    }
  }
  else if (depth >= m_maxDepth-1)
  {
    currColor[tid] = make_float3(0, 0, 0);
    return true;
  }

  return false;
}

void IntegratorMISPTLoop2::kernel_LightSelect(const SurfaceHit* surfElem, const int depth,
                                              float* lightPickProb, int* lightOffset, float4* rndLightData)
{  
  RandomGen& gen    = randomGen();
  rndLightData[tid] = rndLight(&gen, depth, m_pGlobals->rmQMC, PerThread().qmcPos, nullptr);
  
  lightPickProb[tid] = 1.0f;
  lightOffset  [tid] = SelectRandomLightRev(rndLightData[tid].z, surfElem[tid].pos, m_pGlobals,
                                            lightPickProb+tid);
}


void IntegratorMISPTLoop2::kernel_LightSample(const SurfaceHit* surfElem, const int* lightOffset, const float4* rndLightData,
                                              float3* shadowRayPos, float3* shadowRayDir, ShadowSample* explicitSam)
{
  if (lightOffset[tid] >= 0) // if need to sample direct light ?
  { 
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem[tid].matId);
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset[tid]);
    
    LightSampleRev(pLight, to_float3(rndLightData[tid]), surfElem[tid].pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   explicitSam+tid);
    
    shadowRayDir[tid] = normalize(explicitSam[tid].pos - surfElem[tid].pos);
    shadowRayPos[tid] = OffsShadowRayPos(surfElem[tid].pos, surfElem[tid].normal, shadowRayDir[tid], surfElem[tid].sRayOff);
  }
}

void IntegratorMISPTLoop2::kernel_ShadowTrace(const float3* shadowRayPos, const float3* shadowRayDir, const int* lightOffset, const float3* explicitSamPos,
                                              float3* shadow)
{
  //#NOTE: must be replaced by some internal implementation
  //
  if (lightOffset[tid] >= 0)
    shadow[tid] = shadowTrace(shadowRayPos[tid], shadowRayDir[tid], length(shadowRayPos[tid] - explicitSamPos[tid])*0.995f); 
  else 
    shadow[tid] = make_float3(0,0,0);
}

void IntegratorMISPTLoop2::kernel_Shade(const SurfaceHit* a_surfElem, const ShadowSample* a_explicitSam, const float3* shadowRayDir, const float3* ray_dir,
                                        const float3* shadow, const float* lightPickProb, const int* lightOffset,
                                        float3* explicitColor)
{
  if (lightOffset[tid] >= 0)
  {
    const SurfaceHit    surfElem    = a_surfElem[tid];
    const ShadowSample  explicitSam = a_explicitSam[tid];

    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = shadowRayDir[tid];
    sc.v  = (-1.0f)*ray_dir[tid];
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);    
    auto ptlCopy = m_ptlDummy;
    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptlCopy);
    
    const auto evalData      = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), /* global data --> */ m_pGlobals, m_texStorage, m_texStorageAux, &ptlCopy);
    
    const float cosThetaOut1 = fmax(+dot(shadowRayDir[tid], surfElem.normal), 0.0f);
    const float cosThetaOut2 = fmax(-dot(shadowRayDir[tid], surfElem.normal), 0.0f);
    const float3 bxdfVal     = (evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2);
   
    const float lgtPdf       = explicitSam.pdf*lightPickProb[tid];
    
    float misWeight = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0f;
    
    explicitColor[tid] = (1.0f / lightPickProb[tid])*(explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON2)))*bxdfVal*misWeight*shadow[tid]; // clamp brdfVal? test it !!!
  }
  else
    explicitColor[tid] = make_float3(0,0,0);
}

void IntegratorMISPTLoop2::kernel_NextBounce(const SurfaceHit* a_surfElem, const float3* a_explicitColor,
                                             MisData* misPrev, float3* ray_pos, float3* ray_dir, uint* flags, float3* accumColor, float3* accumuThoroughput)
{
  const SurfaceHit surfElem      = a_surfElem[tid];
  const float3     explicitColor = a_explicitColor[tid];

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  RandomGen& gen = randomGen();

  const int rayBounceNum   = unpackBounceNum(flags[tid]);
  const uint otherRayFlags = unpackRayFlags(flags[tid]);
   
  const bool canSampleReflOnly    = (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY) != 0;
  const bool sampleReflectionOnly = ((otherRayFlags & RAY_GRAMMAR_DIRECT_LIGHT) != 0) && canSampleReflOnly; 
  
  const float* matRandsArray      = (gen.rptr == 0) ? 0 : gen.rptr + rndMatOffsetMMLT(rayBounceNum);
  const unsigned int* qmcTablePtr = GetQMCTableIfEnabled();

  float allRands[MMLT_FLOATS_PER_BOUNCE];
  RndMatAll(&gen, matRandsArray, rayBounceNum, 
            m_pGlobals->rmQMC, PerThread().qmcPos, qmcTablePtr,
            allRands);

  MatSample matSam; int matOffset;
  MaterialSampleAndEvalBxDF(pHitMaterial, allRands, &surfElem, ray_dir[tid], make_float3(0,0,0), flags[tid], false,
                            m_pGlobals, m_texStorage, m_texStorageAux, &m_ptlDummy, 
                            &matSam, &matOffset);

  const float3 bxdfVal = matSam.color * (1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosTheta = fabs(dot(matSam.direction, surfElem.normal));
  
  ray_dir[tid]              = matSam.direction;
  ray_pos[tid]              = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);
  misPrev[tid]              = makeInitialMisData();
  misPrev[tid].isSpecular   = isPureSpecular(matSam);
  misPrev[tid].matSamplePdf = matSam.pdf;
  flags[tid]                = flagsNextBounceLite(flags[tid], matSam, m_pGlobals);
  
  accumColor[tid]         += accumuThoroughput[tid]*explicitColor;
  accumuThoroughput[tid]  *= cosTheta*bxdfVal;
}                         

void IntegratorMISPTLoop2::kernel_AddLastBouceContrib(const float3* currColor, const float3* accumuThoroughput,
                                                      float3* accumColor)
{
   accumColor[tid] += accumuThoroughput[tid]*currColor[tid];
}

float3 IntegratorMISPTLoop2::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
  //#pragma hycc exclude(depth, a_currDepth, m_maxDepth)

  float3 accumColor, accumuThoroughput, currColor;
  kernel_InitAccumData(&accumColor, &accumuThoroughput, &currColor);

  for(int depth = a_currDepth; depth < m_maxDepth; depth++) 
  {
    if (depth >= m_maxDepth)
      break; 

    Lite_Hit hit;
    kernel_RayTrace(&ray_pos, &ray_dir, 
                    &hit);
    
    if(kernel_HitEnvironment(&ray_dir, &hit, &misPrev, &flags, 
                             &currColor))
      break;
   
    SurfaceHit surfElem;
    kernel_EvalSurface(&ray_pos, &ray_dir, &hit, 
                       &surfElem);

    if(kernel_EvalEmission(&ray_pos, &ray_dir, &surfElem, &flags, &misPrev, &hit, depth, 
                           &currColor))
      break;
    
    float  lightPickProb;
    int    lightOffset;
    float4 rndLightData;
    kernel_LightSelect(&surfElem, depth, &lightPickProb, &lightOffset, 
                       &rndLightData);

    float3 shadowRayPos, shadowRayDir; 
    ShadowSample explicitSam;
    kernel_LightSample(&surfElem, &lightOffset, &rndLightData, &shadowRayPos, &shadowRayDir, 
                       &explicitSam);
    
    //#pragma hycc compress(half3)
    float3 shadow; 
    kernel_ShadowTrace(&shadowRayPos, &shadowRayDir, &lightOffset, &explicitSam.pos, // please note '&explicitSam.pos' ... 
                       &shadow);

    float3 explicitColor;
    kernel_Shade(&surfElem, &explicitSam, &shadowRayDir, &ray_dir, &shadow, &lightPickProb, &lightOffset, 
                 &explicitColor);
    
    kernel_NextBounce(&surfElem, &explicitColor, 
                      &misPrev, &ray_pos, &ray_dir, &flags, &accumColor, &accumuThoroughput);
  } // for

  kernel_AddLastBouceContrib(&currColor, &accumuThoroughput, &accumColor);

  return clamp(accumColor, 0.0F, 1e6F);
}

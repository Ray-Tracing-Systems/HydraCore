#include <omp.h>
#include "CPUExp_Integrators.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void IntegratorMISPTLoop2::kernel_InitAccumData(float3& accumColor, float3& accumuThoroughput, float3& currColor)
{
  accumColor        = make_float3(0,0,0);
  accumuThoroughput = make_float3(1,1,1);
  currColor         = make_float3(0,0,0);
}

void IntegratorMISPTLoop2::kernel_RayTrace(const float3& ray_pos, const float3& ray_dir, 
                                           Lite_Hit& hit)
{
  hit = rayTrace(ray_pos, ray_dir);
}

bool IntegratorMISPTLoop2::kernel_HitEnvironment(const float3& ray_dir, const Lite_Hit& hit, const MisData& misPrev, const int& flags,
                                                 float3& currColor)
{
  if (HitNone(hit))
  {
    currColor = environmentColor(ray_dir, misPrev, flags, m_pGlobals, m_matStorage, m_pdfStorage, m_texStorage);
    return true;
  }

  return false;
}

void IntegratorMISPTLoop2::kernel_EvalSurface(const float3& ray_pos, const float3& ray_dir, const Lite_Hit& hit,
                                              SurfaceHit& surfElem)
{
  surfElem = surfaceEval(ray_pos, ray_dir, hit);
}

bool IntegratorMISPTLoop2::kernel_EvalEmission(const float3& ray_pos, const float3& ray_dir, 
                                               const SurfaceHit& surfElem, const int& flags, 
                                               const MisData& misPrev, const Lite_Hit& hit,
                                               const int depth,
                                               float3& currColor)
{

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
  
  if (dot(emission, emission) > 1e-3f)
  {
    const PlainLight* pLight = getLightFromInstId(fetchInstId(hit));

    if (pLight != nullptr)
    {
      float lgtPdf    = lightPdfSelectRev(pLight)*lightEvalPDF(pLight, ray_pos, ray_dir, surfElem.pos, surfElem.normal, surfElem.texCoord, m_pdfStorage, m_pGlobals);
      float bsdfPdf   = misPrev.matSamplePdf;
      float misWeight = misWeightHeuristic(bsdfPdf, lgtPdf);  // (bsdfPdf*bsdfPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);

      if (misPrev.isSpecular)
        misWeight = 1.0f;

      currColor = emission*misWeight;
      return true;
    }
    else
    {
      currColor = emission;
      return true;
    }
  }
  else if (depth >= m_maxDepth-1)
  {
    currColor = make_float3(0, 0, 0);
    return true;
  }

  return false;
}

void IntegratorMISPTLoop2::kernel_LightSelect(const SurfaceHit& surfElem, const int depth,
                                              float& lightPickProb, int& lightOffset, float4& rndLightData)
{
  const unsigned int* qmcTablePtr = GetQMCTableIfEnabled();
  
  RandomGen& gen = randomGen();
  rndLightData   = rndLight(&gen, depth,
                            m_pGlobals->rmQMC, PerThread().qmcPos, qmcTablePtr);
  
  lightPickProb = 1.0f;
  lightOffset   = SelectRandomLightRev(rndLightData.z, surfElem.pos, m_pGlobals,
                                       &lightPickProb);
}


void IntegratorMISPTLoop2::kernel_LightSample(const SurfaceHit& surfElem, const int& lightOffset, const float4& rndLightData,
                                              float3& shadowRayPos, float3& shadowRayDir, ShadowSample& explicitSam)
{
  if (lightOffset >= 0) // if need to sample direct light ?
  { 
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
    
    LightSampleRev(pLight, to_float3(rndLightData), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   &explicitSam);
    
    shadowRayDir = normalize(explicitSam.pos - surfElem.pos);
    shadowRayPos = OffsShadowRayPos(surfElem.pos, surfElem.normal, shadowRayDir, surfElem.sRayOff);
  }
}

void IntegratorMISPTLoop2::kernel_ShadowTrace(const float3&  shadowRayPos, const float3&  shadowRayDir, const int& lightOffset, const float3& explicitSamPos,
                                              float3& shadow)
{
  if (lightOffset >= 0)
    shadow = shadowTrace(shadowRayPos, shadowRayDir, length(shadowRayPos - explicitSamPos)*0.995f);
  else 
    shadow = make_float3(0,0,0);
}

void IntegratorMISPTLoop2::kernel_Shade(const SurfaceHit& surfElem, const ShadowSample& explicitSam, const float3& shadowRayDir, const float3& ray_dir,
                                        const float3& shadow, const float& lightPickProb, const int& lightOffset,
                                        float3& explicitColor)
{
  if (lightOffset >= 0)
  {
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = shadowRayDir;
    sc.v  = (-1.0f)*ray_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);    
    auto ptlCopy = m_ptlDummy;
    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptlCopy);
    
    const auto evalData      = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), /* global data --> */ m_pGlobals, m_texStorage, m_texStorageAux, &ptlCopy);
    
    const float cosThetaOut1 = fmax(+dot(shadowRayDir, surfElem.normal), 0.0f);
    const float cosThetaOut2 = fmax(-dot(shadowRayDir, surfElem.normal), 0.0f);
    const float3 bxdfVal     = (evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2);
   
    const float lgtPdf       = explicitSam.pdf*lightPickProb;
    
    float misWeight = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0f;
    
    explicitColor = (1.0f / lightPickProb)*(explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON2)))*bxdfVal*misWeight*shadow; // clamp brdfVal? test it !!!
  }
  else
    explicitColor = make_float3(0,0,0);
}

void IntegratorMISPTLoop2::kernel_NextBounce(const SurfaceHit& surfElem, const float3& explicitColor,
                                             MisData& misPrev, float3& ray_pos, float3& ray_dir, uint& flags, float3& accumColor, float3& accumuThoroughput)
{
  const MatSample matSam = std::get<0>( sampleAndEvalBxDF(ray_dir, surfElem) );
  const float3 bxdfVal   = matSam.color * (1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosTheta   = fabs(dot(matSam.direction, surfElem.normal));
  ray_dir              = matSam.direction;
  ray_pos              = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);
  misPrev              = makeInitialMisData();
  misPrev.isSpecular   = isPureSpecular(matSam);
  misPrev.matSamplePdf = matSam.pdf;
  flags                = flagsNextBounceLite(flags, matSam, m_pGlobals);
  
  accumColor         += accumuThoroughput*explicitColor;
  accumuThoroughput  *= cosTheta*bxdfVal;
}                         

void IntegratorMISPTLoop2::kernel_AddLastBouceContrib(const float3& currColor, const float3& accumuThoroughput,
                                                      float3& accumColor)
{
   accumColor += accumuThoroughput*currColor;
}

float3 IntegratorMISPTLoop2::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
  //#pragma hycc exclude(depth, a_currDepth, m_maxDepth)

  float3 accumColor, accumuThoroughput, currColor;
  kernel_InitAccumData(accumColor, accumuThoroughput, currColor);

  for(int depth = a_currDepth; depth < m_maxDepth; depth++) 
  {
    if (depth >= m_maxDepth)
      break; 

    Lite_Hit hit;
    kernel_RayTrace(ray_pos, ray_dir,
                    hit);
    
    if(kernel_HitEnvironment(ray_dir, hit, misPrev, flags,
                             currColor))
      break;
   
    SurfaceHit surfElem;
    kernel_EvalSurface(ray_pos, ray_dir, hit,
                       surfElem);

    if(kernel_EvalEmission(ray_pos, ray_dir, surfElem, flags, misPrev, hit, depth,
                           currColor))
      break;
    
    float  lightPickProb;
    int    lightOffset;
    float4 rndLightData;
    kernel_LightSelect(surfElem, depth, 
                       lightPickProb, lightOffset, rndLightData);

    float3 shadowRayPos, shadowRayDir; 
    ShadowSample explicitSam;
    kernel_LightSample(surfElem, lightOffset, rndLightData,
                       shadowRayPos, shadowRayDir, explicitSam);
    
    //#pragma hycc compress(half3)
    float3 shadow; 
    kernel_ShadowTrace(shadowRayPos, shadowRayDir, lightOffset, explicitSam.pos,
                       shadow);

    float3 explicitColor;
    kernel_Shade(surfElem, explicitSam, shadowRayDir, ray_dir, shadow, lightPickProb, lightOffset,
                 explicitColor);
   
    
    kernel_NextBounce(surfElem, explicitColor,
                      misPrev, ray_pos, ray_dir, flags, accumColor, accumuThoroughput);
  } // for

  kernel_AddLastBouceContrib(currColor, accumuThoroughput,
                             accumColor);

  return accumColor;
}

#include <omp.h>
#include "CPUExp_Integrators.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


float3 IntegratorStupidPT::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
  if (a_currDepth >= m_maxDepth)
    return float3(0, 0, 0);

  const Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
    return EnviromnentColor(ray_dir, misPrev, flags);

  SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-6f)
    return emission;

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));

  const float3 bxdfVal   = matSam.color * (1.0f / fmaxf(matSam.pdf, DEPSILON2));
  const float  cosTheta  = dot(matSam.direction, surfElem.normal);

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);
  
  flags = flagsNextBounceLite(flags, matSam, m_pGlobals);

  return fabs(cosTheta)*bxdfVal*PathTrace(nextRay_pos, nextRay_dir, MisData(), a_currDepth + 1, flags);  // --*(1.0 / (1.0 - pabsorb));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float3  IntegratorShadowPT::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
  if (a_currDepth >= m_maxDepth)
    return float3(0, 0, 0);

  Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
    return float3(0, 0, 0);

  SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-3f)
    return float3(0, 0, 0);

  float3 explicitColor(0, 0, 0);

  auto& gen = randomGen();
  float lightPickProb = 1.0f;
  int lightOffset = SelectRandomLightRev(rndFloat1_Pseudo(&gen), surfElem.pos, m_pGlobals,
                                         &lightPickProb);

  if (lightOffset >= 0)
  {
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    ShadowSample explicitSam;
    LightSampleRev(pLight, rndFloat3(&gen), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage, 
                   &explicitSam); 

    const float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos); // explicitSam.direction;
    const float3 shadowRayPos = surfElem.pos + shadowRayDir*fmax(maxcomp(surfElem.pos), 1.0f)*GEPSILON;
  
    const float3 shadow       = shadowTrace(shadowRayPos, shadowRayDir, explicitSam.maxDist*0.9995f);

    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
   
    auto ptlCopy = m_ptlDummy;
    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptlCopy);

    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = shadowRayDir;
    sc.v  = (-1.0f)*ray_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const float3 brdfVal    = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage, &ptlCopy).brdf; // a_shadingTexture
    const float cosThetaOut = fmax(dot(shadowRayDir, surfElem.normal), 0.0f);

    explicitColor = (1.0f / lightPickProb)*(explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON)))*cosThetaOut*brdfVal*shadow; // clamp brdfVal ? test it !!!
  }

  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));
  const float3   bxdfVal = matSam.color * (1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float   cosTheta = dot(matSam.direction, surfElem.normal);

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  flags = flagsNextBounceLite(flags, matSam, m_pGlobals);

  return explicitColor + cosTheta*bxdfVal*PathTrace(nextRay_pos, nextRay_dir, MisData(), a_currDepth + 1, flags);  // --*(1.0 / (1.0 - pabsorb));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float3 IntegratorMISPT::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
  if (a_currDepth >= m_maxDepth)
    return float3(0, 0, 0);

  Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
    return environmentColor(ray_dir, misPrev, flags, m_pGlobals, m_matStorage, m_pdfStorage, m_texStorage);
  
  SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);
  
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

      return emission*misWeight;
    }
    else
      return emission;
  }
  else if (a_currDepth >= m_maxDepth-1)
    return float3(0, 0, 0);

  float3 explicitColor(0, 0, 0);

  // static inline float4 rndLight(RandomGen* gen, __global const float* rptr, const int bounceId,
  //                              __global const int* a_tab, const unsigned int qmcPos, __constant unsigned int* a_qmcTable)
  
  const unsigned int* qmcTablePtr = GetQMCTableIfEnabled();
  
  auto& gen = randomGen();
  const float4 rndLightData = rndLight(&gen, a_currDepth,
                                       m_pGlobals->rmQMC, PerThread().qmcPos, qmcTablePtr);
  
  float lightPickProb = 1.0f;
  int lightOffset     = SelectRandomLightRev(rndLightData.z, surfElem.pos, m_pGlobals,
                                             &lightPickProb);
  
  if (lightOffset >= 0) // if need to sample direct light ?
  { 
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
    
    ShadowSample explicitSam;
    LightSampleRev(pLight, to_float3(rndLightData), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   &explicitSam);
    
    const float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos);
    const float3 shadowRayPos = OffsShadowRayPos(surfElem.pos, surfElem.normal, shadowRayDir, surfElem.sRayOff);

    const float3 shadow = shadowTrace(shadowRayPos, shadowRayDir, length(shadowRayPos - explicitSam.pos)*0.995f);
        
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = shadowRayDir;
    sc.v  = (-1.0f)*ray_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    auto ptlCopy = m_ptlDummy;
    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptlCopy);
    
    const auto evalData      = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage, &ptlCopy);
    
    const float cosThetaOut1 = fmax(+dot(shadowRayDir, surfElem.normal), 0.0f);
    const float cosThetaOut2 = fmax(-dot(shadowRayDir, surfElem.normal), 0.0f);
    const float3 bxdfVal     = (evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2);
   
    const float lgtPdf       = explicitSam.pdf*lightPickProb;
    
    float misWeight = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0f;
    
    explicitColor = (1.0f / lightPickProb)*(explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON2)))*bxdfVal*misWeight*shadow; // clamp brdfVal? test it !!!
  }
  
  const MatSample matSam = std::get<0>( sampleAndEvalBxDF(ray_dir, surfElem) );
  const float3 bxdfVal   = matSam.color * (1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosTheta   = fabs(dot(matSam.direction, surfElem.normal));

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  MisData currMis            = makeInitialMisData();
  currMis.isSpecular         = isPureSpecular(matSam);
  currMis.matSamplePdf       = matSam.pdf;

  flags = flagsNextBounceLite(flags, matSam, m_pGlobals);

  return explicitColor + cosTheta*bxdfVal*PathTrace(nextRay_pos, nextRay_dir, currMis, a_currDepth + 1, flags);  // --*(1.0 / (1.0 - pabsorb));
}

void IntegratorMISPT_QMC::DoPass(std::vector<uint>& a_imageLDR)
{
  if (m_width*m_height != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");
  
  const float alpha   = 1.0f / float(m_spp + 1);  // Update HDR image coeff
  const auto loopSize = m_summColors.size();
  const int qmcOffset = int(loopSize)*m_spp;
  
  #pragma omp parallel for
  for (int i = 0; i < loopSize; ++i)
  {
    PerThread().qmcPos = qmcOffset + i;
    
    RandomGen& gen  = randomGen();
    float4 lensOffs = rndLens(&gen, nullptr, float2(1,1), 
                              m_pGlobals->rmQMC, PerThread().qmcPos, (const unsigned int*)m_tableQMC);
                              
    float  fx, fy;
    float3 ray_pos, ray_dir;
    MakeEyeRayFromF4Rnd(lensOffs, m_pGlobals,
                        &ray_pos, &ray_dir, &fx, &fy);
  
    int x = (int)(fx);
    int y = (int)(fy);

    if (x >= m_width)  x = m_width - 1;
    if (y >= m_height) y = m_height - 1;
    if (x < 0)  x = 0;
    if (y < 0)  y = 0;
    
    const float3 color = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
    const float maxCol = maxcomp(color);
    
    m_summColors[y*m_width + x] = m_summColors[y*m_width + x] * (1.0f - alpha) + to_float4(color, maxCol)*alpha;
  }
  
  m_spp++;
  GetImageToLDR(a_imageLDR);
  
  //RandomizeAllGenerators();
  
  std::cout << "IntegratorMISPT_QMC: spp = " << m_spp << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


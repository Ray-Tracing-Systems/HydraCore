#include <omp.h>
#include "CPUExp_Integrators.h"

//////////////////////////////////////////////////////////////////////////////


class IntegratorMISPT_trofimm : public IntegratorCommon
{
public:

  IntegratorMISPT_trofimm(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags) : IntegratorCommon(w, h, a_pGlobals, a_createFlags) {}

  float3 PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags);
  void DoPass(std::vector<uint>& a_imageLDR);
  std::tuple<MatSample, int, float3> sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, int a_currDepth, uint flags = 0, float3 shadow = float3(0, 0, 0), bool mmltMode = false);
};

//////////////////////////////////////////////////////////////////////////////

static inline BRDFSelector materialRandomWalkBRDF_trofimm(__global const PlainMaterial* a_pMat, __private RandomGen* a_pGen, __global const float* qmcVec,
  const float3 rayDir, const float3 hitNorm, const float2 hitTexCoord,
  __global const EngineGlobals* a_globals, texture2d_t a_tex, const int a_rayBounce, const bool a_mmltMode, const bool a_reflOnly)
{
  BRDFSelector res, sel;

  res.localOffs = 0;
  res.w = 1.0f;

  sel.localOffs = 0;
  sel.w = 1.0f;

  __global const PlainMaterial* node = a_pMat;
  int i = 0;
  while (!materialIsLeafBRDF(node) && i < MLT_FLOATS_PER_MLAYER)
  {
    float rndVal;

    if (a_rayBounce == 0 && i == 0)
      rndVal = a_pGen->rptr2[6];
    else
      rndVal = rndMatLayer(a_pGen, qmcVec, a_rayBounce, i);

    //////////////////////////////////////////////////////////////////////////
    const int type = materialGetType(node);

    if (type == PLAIN_MAT_CLASS_BLEND_MASK)
      sel = blendSelectBRDF(node, rndVal, rayDir, hitNorm, hitTexCoord, (a_reflOnly && (i == 0)), a_globals, a_tex);

    //////////////////////////////////////////////////////////////////////////

    res.w = res.w*sel.w;
    res.localOffs = res.localOffs + sel.localOffs;

    node = node + sel.localOffs;
    i++;
  }

  for (; i < MLT_FLOATS_PER_MLAYER; i++)  // we must generate these numbers to get predefined state of seed for each bounce
  {
    if (a_mmltMode)
      rndMatLayerMMLT(a_pGen, qmcVec, a_rayBounce, i);
    else
      rndMatLayer(a_pGen, qmcVec, a_rayBounce, i);
  }

  return res;
}

//////////////////////////////////////////////////////////////////////////////

std::tuple<MatSample, int, float3> IntegratorMISPT_trofimm::sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, int a_currDepth, uint flags, float3 shadow, bool a_mmltMode)
{
  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  auto& gen = randomGen();

  int matOffset = materialOffset(m_pGlobals, surfElem.matId);
  const int rayBounceNum = a_currDepth;// unpackBounceNum(flags);
  const uint otherRayFlags = unpackRayFlags(flags);

  const bool canSampleReflOnly = (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY) != 0;
  const bool sampleReflectionOnly = ((otherRayFlags & RAY_GRAMMAR_DIRECT_LIGHT) != 0) && canSampleReflOnly;

  BRDFSelector mixSelector = materialRandomWalkBRDF_trofimm(pHitMaterial, &gen, gen.rptr, ray_dir, surfElem.normal, surfElem.texCoord, m_pGlobals, m_texStorage, rayBounceNum, a_mmltMode, sampleReflectionOnly); // a_shadingTexture

                                                                                                                                                                                                          /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// --- >

  if ((m_pGlobals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_RADIANCE || m_pGlobals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_PRIMARY) && (m_pGlobals->varsI[HRT_RENDER_LAYER_DEPTH] == rayBounceNum)) // piss execution way (render incoming radiance)
  {
    matOffset = m_pGlobals->varsI[HRT_WHITE_DIFFUSE_OFFSET];
    pHitMaterial = materialAtOffset(m_matStorage, m_pGlobals->varsI[HRT_WHITE_DIFFUSE_OFFSET]);
    mixSelector.w = 1.0f;
  }
  else // normal execution way
  {
    // this is needed component for mis further (misNext.prevMaterialOffset = matOffset;)
    //
    matOffset = matOffset + mixSelector.localOffs*(sizeof(PlainMaterial) / sizeof(float4));
    pHitMaterial = pHitMaterial + mixSelector.localOffs;
  }

  // const bool hitFromBack        = (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE) != 0;
  // const bool hitGlassFromInside = (materialGetType(pHitMaterial) == PLAIN_MAT_CLASS_GLASS) && hitFromBack;

  ShadeContext sc;

  sc.wp = surfElem.pos;
  sc.l = ray_dir;
  sc.v = ray_dir;
  sc.n = surfElem.normal;
  sc.fn = surfElem.flatNormal;
  sc.tg = surfElem.tangent;
  sc.bn = surfElem.biTangent;
  sc.tc = surfElem.texCoord;
  sc.hfi = surfElem.hfi;

  float3 rands;
  if (a_currDepth > 0) rands = (a_mmltMode ? rndMatMMLT(&gen, gen.rptr, rayBounceNum) : rndMat(&gen, gen.rptr, rayBounceNum));
  else                 rands = { gen.rptr2[2], gen.rptr2[3],  rndFloat1_Pseudo(&gen) };


  MatSample brdfSample;
  MaterialLeafSampleAndEvalBRDF(pHitMaterial, rands, &sc, shadow, m_pGlobals, m_texStorage, m_texStorageAux,
    &brdfSample);
  brdfSample.pdf *= mixSelector.w;

  return std::make_tuple(brdfSample, matOffset, make_float3(1, 1, 1));
}

//////////////////////////////////////////////////////////////////////////////

float3 IntegratorMISPT_trofimm::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
  if (a_currDepth >= m_maxDepth)
    return float3(0, 0, 0);

  Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
    return float3(0, 0, 0);

  SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-3f)
  {
    const PlainLight* pLight = getLightFromInstId(fetchInstId(hit));

    if (pLight != nullptr)
    {
      float lgtPdf = lightPdfSelectRev(pLight)*lightEvalPDF(pLight, ray_pos, ray_dir, surfElem.pos, surfElem.normal, surfElem.texCoord, m_pdfStorage, m_pGlobals);
      float bsdfPdf = misPrev.matSamplePdf;
      float misWeight = misWeightHeuristic(bsdfPdf, lgtPdf);  // (bsdfPdf*bsdfPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);

      if (misPrev.isSpecular)
        misWeight = 1.0f;

      return emission * misWeight;
    }
    else
      return emission;
  }
  else if (a_currDepth >= m_maxDepth - 1)
    return float3(0, 0, 0);

  float3 explicitColor(0, 0, 0);

  auto& gen = randomGen();
  float lightPickProb = 1.0f;
  int lightOffset = SelectRandomLightRev(rndFloat2_Pseudo(&gen), surfElem.pos, m_pGlobals,
    &lightPickProb);

  if (lightOffset >= 0) // if need to sample direct light ?
  {
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    ShadowSample explicitSam;

    const float3 rndXyz = { gen.rptr2[4], gen.rptr2[5], rndFloat1_Pseudo(&gen) };

    LightSampleRev(pLight, rndXyz /*rndFloat3(&gen)*/, surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
      &explicitSam);

    const float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos);
    const float3 shadowRayPos = OffsShadowRayPos(surfElem.pos, surfElem.normal, shadowRayDir, surfElem.sRayOff);

    const float3 shadow = shadowTrace(shadowRayPos, shadowRayDir, length(shadowRayPos - explicitSam.pos)*0.995f);

    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l = shadowRayDir;
    sc.v = (-1.0f)*ray_dir;
    sc.n = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const auto evalData = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage);

    const float cosThetaOut1 = fmax(+dot(shadowRayDir, surfElem.normal), 0.0f);
    const float cosThetaOut2 = fmax(-dot(shadowRayDir, surfElem.normal), 0.0f);
    const float3 bxdfVal = (evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2);

    const float lgtPdf = explicitSam.pdf*lightPickProb;

    float misWeight = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0f;

    explicitColor = (1.0f / lightPickProb)*(explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON2)))*bxdfVal*misWeight*shadow; // clamp brdfVal? test it !!!
  }

  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem, a_currDepth));
  const float3 bxdfVal = matSam.color * (1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosTheta = fabs(dot(matSam.direction, surfElem.normal));

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  MisData currMis = makeInitialMisData();
  currMis.isSpecular = isPureSpecular(matSam);
  currMis.matSamplePdf = matSam.pdf;

  flags = flagsNextBounceLite(flags, matSam, m_pGlobals);

  return explicitColor + cosTheta * bxdfVal*PathTrace(nextRay_pos, nextRay_dir, currMis, a_currDepth + 1, flags);  // --*(1.0 / (1.0 - pabsorb));
}

//////////////////////////////////////////////////////////////////////////////

void IntegratorMISPT_trofimm::DoPass(std::vector<uint>& a_imageLDR)
{
  if (m_width*m_height != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output buffer size");

  // Update HDR image
  //
  const float alpha = 1.0f / float(m_spp + 1);


  const int sizeSummColors = m_summColors.size();

  unsigned int tableQMC[QRNG_DIMENSIONS][QRNG_RESOLUTION];
  initQuasirandomGenerator(tableQMC);
  

  #pragma omp parallel for
  for (int i = 0; i < sizeSummColors; i++)
  {
    auto& gen = randomGen();

    //float2 rnd2 = rndFloat2_Pseudo(&gen);
    const int offset = i + m_spp * sizeSummColors;
    float sobol1 = rndQmcSobolN(offset, 0, (const unsigned int*)tableQMC);
    float sobol2 = rndQmcSobolN(offset, 1, (const unsigned int*)tableQMC);
    float sobol3 = rndQmcSobolN(offset, 2, (const unsigned int*)tableQMC);
    float sobol4 = rndQmcSobolN(offset, 3, (const unsigned int*)tableQMC);
    float sobol5 = rndQmcSobolN(offset, 4, (const unsigned int*)tableQMC);
    float sobol6 = rndQmcSobolN(offset, 5, (const unsigned int*)tableQMC);
    float sobol7 = rndQmcSobolN(offset, 6, (const unsigned int*)tableQMC);


    float arrayRnd[7] = { sobol1, sobol2, sobol3, sobol4, sobol5, sobol6, sobol7 };

    gen.rptr2 = &arrayRnd[0];

    float fx, fy;
    float3 ray_pos, ray_dir;

    const float4 lensOffs = { gen.rptr2[0], gen.rptr2[1], 0.0f, 0.0f };

    MakeEyeRayFromF4Rnd(lensOffs, m_pGlobals, &ray_pos, &ray_dir, &fx, &fy); // x and y can be overwriten by ConnectEye later
    
    const float3 color = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
    const float maxCol = maxcomp(color);

    int x = int(fx);
    int y = int(fy);

    if (x >= m_width)  x = m_width - 1;
    if (y >= m_height) y = m_height - 1;

    m_summColors[y*m_width + x] = m_summColors[y*m_width + x] * (1.0f - alpha) + to_float4(color, maxCol)*alpha;
  }

  RandomizeAllGenerators();

  m_spp++;
  GetImageToLDR(a_imageLDR);

  //if (m_spp == 1)
  //DebugSaveGbufferImage(L"C:/[Hydra]/rendered_images/torus_gbuff");

  std::cout << "Integrator_Trofimm: spp = " << m_spp << std::endl;
}

//////////////////////////////////////////////////////////////////////////////

Integrator* CreateIntegratorTrofimm(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags)
{
  return new IntegratorMISPT_trofimm(w, h, a_pGlobals, a_createFlags);
}
#include "CPUExp_Integrators.h"
//#include <time.h>
#include <algorithm>  // sort
#include <conio.h> // getch();
#include <numeric> // accumulate
#include "CPUExp_Integrators_MISPT.h"

//////////////////////////////////////////////////////////////////////////////////////////

float Distance(const int a, const int b)
{
  return (float)sqrt(a * a + b * b);
}

float Distance(const int2 a_pos1, const int2 a_pos2)
{
  const int a = a_pos1.x - a_pos2.x;
  const int b = a_pos1.y - a_pos2.y;

  return (float)sqrt(a * a + b * b);
}

float Mean(const float3 color)
{
  return (color.x + color.y + color.z) / 3.0F;
}

float Mean(const float4 color)
{
  return (color.x + color.y + color.z) / 3.0F;
}

float Sign(const float val) { return (val > 0) ? 1.0F : -1.0F; }

float AcceptProb(const float a_currProb, const float a_nextProb)
{
  return (a_currProb == 0.0F) ? 0 : clamp(a_nextProb / a_currProb, 0.0F, 1.0F);
}

void PrintFloat3(const float3& val, const std::string& name)
{
  std::cout << name << val.x << " " << val.y << " " << val.z << std::endl;  
}


void IntegratorMISPTLoop2Adapt::DoPass(std::vector<uint>& a_imageLDR)
{  
  // Update HDR image
  if (m_imgSize != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  //const float imgRadius       = Distance(m_width, m_height) * 0.5F;  

  const bool  multiThreads    = true;
  //const int   minSpp          = 0;
  const bool  drawPath        = false;//m_spp >= minSpp && (m_spp % 10 == 0);
  //const bool  drawPdf         = false;//m_spp % 10 == 0;

  const int   maxThreads      = omp_get_max_threads();
  const int   numThreads      = multiThreads ? maxThreads : 1;  
  const int   maxSamples      = multiThreads ? m_imgSize / numThreads / 2 : m_imgSize / 2;
  //const int   maxSamples      = m_spp < minSpp ? m_imgSize : 10;
      
#ifdef NDEBUG
#pragma omp parallel num_threads(numThreads)
#endif
  {
    auto& gen           = randomGen();
    int2 pos            = GetRndFullScreenPos(gen);
    
    //float        step = imgRadius * 0.25F;

    //float a_currResLum  = 0.0F;
    //std::vector<float3> arrDir(m_maxDepth+1); // +1 for lens
    
    //GenerateRndPath(gen, arrDir);

    //Sleep(2000);

    for (int sample = 0; sample < maxSamples; ++sample)
    {
      // Screen adapt. Bull...
      //SimpleScreenRandom(gen);
      //R2Samples(pos, gen, drawPath, imgRadius, sample);
      //VariableStep(pos, gen, imgRadius, drawPath);
      AdaptFromLuminance(pos, gen);
      //AdaptDispers(pos, gen);
      //MarkovChain(pos, step, gen, imgRadius, drawPath);
      //MarkovChain2(pos, gen, imgRadius, sample);
      //WalkWithDispers(pos, gen, imgRadius, drawPath);   

      // Try multidimensional adapt. MLT primary space (mutate random number).
      //MultiDim(gen, arrDir);
    }
  }

  RandomizeAllGenerators();

    
  // get HDR to LDR and scale

  GetImageToLDR(a_imageLDR, true);
  
  const float gammaPow = 1.0F / m_pGlobals->varsF[HRT_IMAGE_GAMMA];  // gamma correction
  
//  float summ = 0.0F;
//  
//  for (int i = 0; i < m_imgSize; ++i)
//  {
//    summ += Mean(m_summColors[i]);
//  }
//
//  const float meanColor = fmax(summ / (float)a_imageLDR.size(), 1e-6F);
//
//  for (int i = 0; i < a_imageLDR.size(); i++)
//{
//  float4 color    = m_summColors[i];    
//  color.x /= meanColor;
//  color.y /= meanColor;
//  color.z /= meanColor;
//
//  color          = ToneMapping4Compress(color);
//  color.x        = powf(color.x, gammaPow);
//  color.y        = powf(color.y, gammaPow);
//  color.z        = powf(color.z, gammaPow);
//
//  a_imageLDR[i] = RealColorToUint32(color);
//}

  
  const int meanSpp = int(std::accumulate(m_samplePerPix.begin(), m_samplePerPix.end(), 0) / (float)m_imgSize);

//#ifdef NDEBUG
//#pragma omp parallel num_threads(numThreads)
//#endif
//for (int i = 0; i < m_imgSize; i++)
//{  
  // for MarkovChain2()

  //const float spp = fmax((float)(m_samplePerPix[i]), 1.0F);

  //float4 color    = m_summColors[i];    
  //color.x /= spp;
  //color.y /= spp;
  //color.z /= spp;

  //color          = ToneMapping4Compress(color);
  //color.x        = powf(color.x, gammaPow);
  //color.y        = powf(color.y, gammaPow);
  //color.z        = powf(color.z, gammaPow);
//}

  // Draw path
  if (drawPath)
  {
#pragma omp parallel num_threads(numThreads)
    for (int i = 0; i < m_imgSize; ++i)
    {  
      const float4 colorPath = float4(m_stepPass[i].x, m_stepPass[i].y, m_stepPass[i].z, 0);

      if (dot(colorPath, colorPath) > 0.001F)
        a_imageLDR[i] = RealColorToUint32(clamp(colorPath, 0.0F, 1.0F));

      m_stepPass[i] = clamp(m_stepPass[i] * 0.5F, 0.0F, 1.0F); // falloff line color
    }
  }


  //m_progress = 0;
  //for (int i = 0; i < m_imgSize; i++)
  //  if (m_pixFinish[i])
  //    m_progress += 1;

  //m_progress = m_progress / (float)maxSample * 100.0F;
  m_spp = meanSpp;

  std::cout << "[" << this->Name() << "]: mean spp     = " << meanSpp << std::endl;

  //std::cout << "[" << this->Name() << "]: progress = " << m_progress << std::endl;

  //getch();
}

float3 IntegratorMISPTLoop2Adapt::PathTrace2(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint a_flags, int& a_resultRayDepth)
{
  //#pragma hycc exclude(depth, a_currDepth, m_maxDepth)

  float3 accumColor, accumuThoroughput, currColor;
  kernel_InitAccumData(accumColor, accumuThoroughput, currColor);

  for (int depth = a_currDepth; depth < m_maxDepth; ++depth)
  {
    a_resultRayDepth = depth;

    if (depth >= m_maxDepth)
      break;

    Lite_Hit hit;
    kernel_RayTrace(a_rpos, a_rdir, hit);

    if (kernel_HitEnvironment(a_rdir, hit, misPrev, a_flags, currColor))
      break;

    SurfaceHit surfElem;
    kernel_EvalSurface(a_rpos, a_rdir, hit, surfElem);

    if (kernel_EvalEmission(a_rpos, a_rdir, surfElem, a_flags, misPrev, hit, depth, currColor))
      break;

    float  lightPickProb;
    int    lightOffset;
    float4 rndLightData;
    kernel_LightSelect(surfElem, depth, lightPickProb, lightOffset, rndLightData);

    float3 shadowRayPos, shadowRayDir;
    ShadowSample explicitSam;
    kernel_LightSample(surfElem, lightOffset, rndLightData, shadowRayPos, shadowRayDir, explicitSam);

    //#pragma hycc compress(half3)
    float3 shadow;
    kernel_ShadowTrace(shadowRayPos, shadowRayDir, lightOffset, explicitSam.pos, // please note '&explicitSam.pos' ... 
      shadow);

    float3 explicitColor;
    kernel_Shade(surfElem, explicitSam, shadowRayDir, a_rdir, shadow, lightPickProb, lightOffset, explicitColor);
    kernel_NextBounce(surfElem, explicitColor, misPrev, a_rpos, a_rdir, a_flags, accumColor, accumuThoroughput);
  } // for

  kernel_AddLastBouceContrib(currColor, accumuThoroughput, accumColor);

  return clamp(accumColor, 0.0F, 1e6F);
}



float3 IntegratorMISPTLoop2Adapt::PathTrace3(RandomGen& a_gen, float3 a_rpos, float3 a_rdir, MisData misPrev,
                                             int a_currDepth, uint flags, std::vector<float3>& a_arrDir, 
                                             const bool a_writeDirr)
{
  //#pragma hycc exclude(depth, a_currDepth, m_maxDepth)

  float3 accumColor, accumuThoroughput, currColor;
  kernel_InitAccumData(accumColor, accumuThoroughput, currColor);

  for (int depth = a_currDepth; depth < m_maxDepth; ++depth)
  {
    if (depth >= m_maxDepth)
      break;                 
    
    if (a_writeDirr)
      a_arrDir[depth + 1] = a_rdir;
      
    //PrintFloat3(a_rdir, "a_rdir:                        ");

    a_rdir = a_arrDir[depth + 1]; // [0] for pos screen
    //PrintFloat3(a_rdir, "a_rdir = a_arrDir[depth + 1]:  ");
    
    Lite_Hit hit;
    kernel_RayTrace(a_rpos, a_rdir, hit);

    //PrintFloat3(a_rdir, "kernel_RayTrace(a_rdir):       ");

    if (kernel_HitEnvironment(a_rdir, hit, misPrev, flags, currColor))
      break;

    //PrintFloat3(a_rdir, "kernel_HitEnvironment(a_rdir): ");

    SurfaceHit surfElem;
    kernel_EvalSurface(a_rpos, a_rdir, hit, surfElem);

    //PrintFloat3(a_rdir, "kernel_EvalSurface(a_rdir):    ");

    if (kernel_EvalEmission(a_rpos, a_rdir, surfElem, flags, misPrev, hit, depth, currColor))
      break;

    //PrintFloat3(a_rdir, "kernel_EvalEmission(a_rdir):   ");

    float  lightPickProb;
    int    lightOffset;
    float4 rndLightData;
    kernel_LightSelect(surfElem, depth, lightPickProb, lightOffset, rndLightData);

    float3 shadowRayPos, shadowRayDir;
    ShadowSample explicitSam;
    kernel_LightSample(surfElem, lightOffset, rndLightData, shadowRayPos, shadowRayDir, explicitSam);

    //#pragma hycc compress(half3)
    float3 shadow;
    kernel_ShadowTrace(shadowRayPos, shadowRayDir, lightOffset, explicitSam.pos, // please note '&explicitSam.pos' ... 
      shadow);

    float3 explicitColor;
    kernel_Shade(surfElem, explicitSam, shadowRayDir, a_rdir, shadow, lightPickProb, lightOffset, explicitColor);

    //PrintFloat3(a_rdir, "kernel_Shade(a_rdir):          ");

    kernel_NextBounce(surfElem, explicitColor, misPrev, a_rpos, a_rdir, flags, accumColor, accumuThoroughput); 

    //PrintFloat3(a_rdir, "kernel_NextBounce(a_rdir):     ");

  } // for

  kernel_AddLastBouceContrib(currColor, accumuThoroughput, accumColor);  

  return clamp(accumColor, 0.0F, 1e6F);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void IntegratorMISPTLoop2Adapt::kernel_InitAccumData(float3& accumColor, float3& accumuThoroughput, float3& currColor) const
{
  accumColor        = make_float3(0.0F, 0.0F, 0.0F);
  accumuThoroughput = make_float3(1.0F, 1.0F, 1.0F);
  currColor         = make_float3(0.0F, 0.0F, 0.0F);
}

void IntegratorMISPTLoop2Adapt::kernel_RayTrace(const float3& ray_pos, const float3& ray_dir, Lite_Hit& hit)
{
  //#NOTE: must be replaced by some internal implementation
  hit = rayTrace(ray_pos, ray_dir);
}

bool IntegratorMISPTLoop2Adapt::kernel_HitEnvironment(const float3& ray_dir, const Lite_Hit& hit, const MisData& misPrev, 
  const int& flags, float3& currColor) const
{
  if (HitNone(hit))
  {
    currColor = environmentColor(ray_dir, misPrev, flags, m_pGlobals, m_matStorage, m_pdfStorage, m_texStorage);
    return true;
  }

  return false;
}

void IntegratorMISPTLoop2Adapt::kernel_EvalSurface(const float3& ray_pos, const float3& ray_dir, const Lite_Hit& hit, SurfaceHit& surfElem)
{
  //surfElem = surfaceEval(ray_pos, ray_dir, hit); // TODO: probably we can cal some class functions like 'surfaceEval'

  // (1) mul ray with instanceMatrixInv
  //
  const float4x4 instanceMatrixInv = fetchMatrix(hit);

  const float3 rayPosLS            = mul4x3(instanceMatrixInv, ray_pos);
  const float3 rayDirLS            = mul3x3(instanceMatrixInv, ray_dir);

  // (2) get pointer to PlainMesh via hit.geomId
  //
  const PlainMesh* mesh            = fetchMeshHeader(hit, m_geom.meshes, m_pGlobals);

  // (3) intersect transformed ray with triangle and get SurfaceHit in local space
  //
  const SurfaceHit surfHit         = surfaceEvalLS(rayPosLS, rayDirLS, hit, mesh);

  // (4) get transformation to wold space
  //
  const float4x4 instanceMatrix    = inverse4x4(instanceMatrixInv);

  // (5) transform SurfaceHit to world space with instanceMatrix
  //
  SurfaceHit surfHitWS             = surfHit;

  const float multInv              = 1.0F / sqrt(3.0F);
  const float3 shadowStartPos      = mul3x3(instanceMatrix, make_float3(multInv * surfHitWS.sRayOff, multInv * surfHitWS.sRayOff, multInv * surfHitWS.sRayOff));

  const float4x4 normalMatrix      = transpose(instanceMatrixInv);

  surfHitWS.pos                    = mul4x3(instanceMatrix, surfHit.pos);
  surfHitWS.normal                 = normalize(mul3x3(normalMatrix, surfHit.normal));
  surfHitWS.flatNormal             = normalize(mul3x3(normalMatrix, surfHit.flatNormal));
  surfHitWS.tangent                = normalize(mul3x3(normalMatrix, surfHit.tangent));
  surfHitWS.biTangent              = normalize(mul3x3(normalMatrix, surfHit.biTangent));
  surfHitWS.t                      = length(surfHitWS.pos - ray_pos); // seems this is more precise. VERY strange !!!
  surfHitWS.sRayOff                = length(shadowStartPos);

  if (m_remapAllLists != nullptr && m_remapTable != nullptr && m_remapInstTab != nullptr)
  {
    surfHitWS.matId                = remapMaterialId(surfHitWS.matId, hit.instId,
      m_remapInstTab, m_remapInstSize, m_remapAllLists,
      m_remapTable, m_remapTabSize);
  }

  surfElem                         = surfHitWS;
}

bool IntegratorMISPTLoop2Adapt::kernel_EvalEmission(const float3& ray_pos, const float3& ray_dir, const SurfaceHit& surfElem,
  const int& flags, const MisData& misPrev, const Lite_Hit& hit, const int depth, float3& currColor)
{
  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));

  if (dot(emission, emission) > 1e-3f)
  {
    //const PlainLight* pLight = getLightFromInstId(fetchInstId(hit)); // TODO: replace with C99 function
    //                                                                 // TODO: probably we can cal some class functions like 'getLightFromInstId'

    // (1) get light pointer
    //
    const PlainLight* pLight = nullptr;
    if (m_pGlobals->lightsNum != 0)
    {
      const int lightOffset = m_geom.instLightInstId[fetchInstId(hit)];
      if (lightOffset >= 0)
        pLight = lightAt(m_pGlobals, lightOffset);
      else
        pLight = nullptr;
    }

    // (2) evaluate light color
    //
    if (pLight != nullptr)
    {
      float lgtPdf    = lightPdfSelectRev(pLight) * lightEvalPDF(pLight, ray_pos, ray_dir, surfElem.pos, surfElem.normal, surfElem.texCoord, m_pdfStorage, m_pGlobals);
      float bsdfPdf   = misPrev.matSamplePdf;
      float misWeight = misWeightHeuristic(bsdfPdf, lgtPdf);  // (bsdfPdf*bsdfPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);

      if (misPrev.isSpecular)
        misWeight = 1.0F;

      currColor = emission * misWeight;
      return true;
    }
    else
    {
      currColor = emission;
      return true;
    }
  }
  else if (depth >= m_maxDepth - 1)
  {
    currColor = make_float3(0.0F, 0.0F, 0.0F);
    return true;
  }

  return false;
}

void IntegratorMISPTLoop2Adapt::kernel_LightSelect(const SurfaceHit& surfElem, const int depth, float& lightPickProb,
  int& lightOffset, float4& rndLightData)
{
  RandomGen& gen = randomGen();
  rndLightData   = rndLight(&gen, depth, m_pGlobals->rmQMC, PerThread().qmcPos, nullptr);
  lightPickProb  = 1.0F;
  lightOffset    = SelectRandomLightRev(rndLightData.z, surfElem.pos, m_pGlobals, &lightPickProb);
}


void IntegratorMISPTLoop2Adapt::kernel_LightSample(const SurfaceHit& surfElem, const int& lightOffset, const float4& rndLightData,
  float3& shadowRayPos, float3& shadowRayDir, ShadowSample& explicitSam)
{
  if (lightOffset >= 0) // if need to sample direct light ?
  {
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    LightSampleRev(pLight, to_float3(rndLightData), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage, &explicitSam);

    shadowRayDir                      = normalize(explicitSam.pos - surfElem.pos);
    shadowRayPos                      = OffsShadowRayPos(surfElem.pos, surfElem.normal, shadowRayDir, surfElem.sRayOff);
  }
}

void IntegratorMISPTLoop2Adapt::kernel_ShadowTrace(const float3& shadowRayPos, const float3& shadowRayDir, const int& lightOffset, const float3& explicitSamPos,
  float3& shadow)
{
  //#NOTE: must be replaced by some internal implementation
  //
  if (lightOffset >= 0)
    shadow = shadowTrace(shadowRayPos, shadowRayDir, length(shadowRayPos - explicitSamPos) * 0.995F);
  else
    shadow = make_float3(0.0F, 0.0F, 0.0F);
}

void IntegratorMISPTLoop2Adapt::kernel_Shade(const SurfaceHit& surfElem, const ShadowSample& explicitSam, const float3& shadowRayDir,
  const float3& ray_dir, const float3& shadow, const float& lightPickProb, const int& lightOffset, float3& explicitColor)
{
  if (lightOffset >= 0)
  {
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l = shadowRayDir;
    sc.v = (-1.0f) * ray_dir;
    sc.n = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
    auto ptlCopy                      = m_ptlDummy;
    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptlCopy);

    const auto evalData               = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), /* global data --> */ m_pGlobals, m_texStorage, m_texStorageAux, &ptlCopy);

    const float cosThetaOut1          = fmax(+dot(shadowRayDir, surfElem.normal), 0.0F);
    const float cosThetaOut2          = fmax(-dot(shadowRayDir, surfElem.normal), 0.0F);
    const float3 bxdfVal              = (evalData.brdf * cosThetaOut1 + evalData.btdf * cosThetaOut2);

    const float lgtPdf                = explicitSam.pdf * lightPickProb;

    float misWeight                   = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0F;

    explicitColor = (1.0F / lightPickProb) * (explicitSam.color * (1.0F / fmax(explicitSam.pdf, DEPSILON2))) * bxdfVal * misWeight * shadow; // clamp brdfVal? test it !!!
  }
  else
    explicitColor = make_float3(0, 0, 0);
}

void IntegratorMISPTLoop2Adapt::kernel_NextBounce(const SurfaceHit& surfElem, const float3& explicitColor, MisData& misPrev,
  float3& ray_pos, float3& ray_dir, uint& flags, float3& accumColor, float3& accumuThoroughput)
{
  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  RandomGen& gen                    = randomGen();

  const int rayBounceNum            = unpackBounceNum(flags);
  const uint otherRayFlags          = unpackRayFlags(flags);

  const bool canSampleReflOnly      = (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY) != 0;
  const bool sampleReflectionOnly   = ((otherRayFlags & RAY_GRAMMAR_DIRECT_LIGHT) != 0) && canSampleReflOnly;

  const float* matRandsArray        = (gen.rptr == 0) ? 0 : gen.rptr + rndMatOffsetMMLT(rayBounceNum);
  const unsigned int* qmcTablePtr   = GetQMCTableIfEnabled();

  float allRands[MMLT_FLOATS_PER_BOUNCE];
  RndMatAll(&gen, matRandsArray, rayBounceNum, m_pGlobals->rmQMC, PerThread().qmcPos, qmcTablePtr, allRands);

  MatSample matSam; int matOffset;
  MaterialSampleAndEvalBxDF(pHitMaterial, allRands, &surfElem, ray_dir, make_float3(0, 0, 0), flags, false, m_pGlobals, m_texStorage,
    m_texStorageAux, &m_ptlDummy, &matSam, &matOffset);

  const float3 bxdfVal = matSam.color * (1.0F / fmaxf(matSam.pdf, 1e-20F));
  const float cosTheta = fabs(dot(matSam.direction, surfElem.normal));

  ray_dir              = matSam.direction;
  ray_pos              = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);
  misPrev              = makeInitialMisData();
  misPrev.isSpecular   = isPureSpecular(matSam);
  misPrev.matSamplePdf = matSam.pdf;
  flags                = flagsNextBounceLite(flags, matSam, m_pGlobals);

  accumColor          += accumuThoroughput * explicitColor;
  accumuThoroughput   *= cosTheta * bxdfVal;
}

void IntegratorMISPTLoop2Adapt::kernel_AddLastBouceContrib(const float3& currColor, const float3& accumuThoroughput, float3& accumColor)
{
  accumColor += accumuThoroughput * currColor;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////


void IntegratorMISPTLoop2Adapt::SimpleScreenRandom(RandomGen& a_gen)
{  
  int2 pos                   = GetRndFullScreenPos(a_gen);

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir) = makeEyeRay(pos.x, pos.y);
  const float3 color         = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
    
  AddColorInSumm(pos, color);  
}



void IntegratorMISPTLoop2Adapt::R2Samples(int2& a_pos, RandomGen& a_gen, const float a_imgRadius, const bool a_drawPath, int& a_sample)
{  
  const int maxSpp              = 8;

  std::vector<float> currSamples;
  std::vector<float> nextSamples;

  const float  step             = a_imgRadius * 0.25F;
  const float2 dir              = GetRndDir(a_gen, false);
  const int2   nextPos          = GetPosWithDirAndStep(a_pos, dir, step);
  //const int2   nextPos          = GetRndFullScreenPos(a_gen);

  for (int i = 0; i < maxSpp; ++i)
  {
    // Get samples with small random offset for R2 error (non-correlating samples) statistics.   

    const float  smallStep     = a_imgRadius * 0.01F + 1.5F;
    const float2 offsetDir     = GetRndDir(a_gen, false);
    const int2   currOffsetPos = GetPosWithDirAndStep(a_pos,   offsetDir, smallStep);    
    const int2   offset        = int2(currOffsetPos.x - a_pos.x, currOffsetPos.y - a_pos.y);
          int2   nextOffsetPos = int2(nextPos.x + offset.x, nextPos.y + offset.y);

    if (nextOffsetPos.x < 0)            nextOffsetPos.x -= offset.x;
    if (nextOffsetPos.y < 0)            nextOffsetPos.y -= offset.y;
    if (nextOffsetPos.x > m_width  - 1) nextOffsetPos.x -= offset.x;
    if (nextOffsetPos.y > m_height - 1) nextOffsetPos.y -= offset.y;

    // get samples

    float3 ray_pos;
    float3 ray_dir;
    std::tie(ray_pos, ray_dir) = makeEyeRay(currOffsetPos.x, currOffsetPos.y);
    const float3 currColor     = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
    
    std::tie(ray_pos, ray_dir) = makeEyeRay(nextOffsetPos.x, nextOffsetPos.y);
    const float3 nextColor     = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

#pragma omp atomic
    a_sample += 2;
        
    AddColorInSumm(currOffsetPos, currColor);
    AddColorInSumm(nextOffsetPos, nextColor);

    currSamples.push_back(float(currOffsetPos.x));
    currSamples.push_back(float(currOffsetPos.y));
    currSamples.push_back(Luminance(currColor));

    nextSamples.push_back(float(nextOffsetPos.x));
    nextSamples.push_back(float(nextOffsetPos.y));
    nextSamples.push_back(Luminance(nextColor));
  }                            
                               
  const float r2error          = R2Error(currSamples, nextSamples, false);
                               
  const float acceptProb       = clamp(r2error, 0.0F, 1.0F) * 0.9F + 0.1F;

  if (rndFloat1_Pseudo(&a_gen) < acceptProb)
  {
    if (a_drawPath)
      DrawDot(m_stepPass, a_pos, float3(0, 0.2F, 0));

    a_pos = nextPos;
  }
  else if (a_drawPath)
    DrawDot(m_stepPass, a_pos, float3(1.0F, 0, 0));
}



void IntegratorMISPTLoop2Adapt::AdaptFromLuminance(int2& a_pos, RandomGen& a_gen)
{
  // Calculate sample

  float3 ray_pos;
  float3 ray_dir;
  int currDepth = 0;
  int nextDepth = 0;

  // Sample 1

  std::tie(ray_pos, ray_dir) = makeEyeRay(a_pos.x, a_pos.y);
  const float3 currColor     = PathTrace2(ray_pos, ray_dir, makeInitialMisData(), 0, 0, currDepth);

  // Sample 2

  const int2 nextPos         = GetRndFullScreenPos(a_gen);
  std::tie(ray_pos, ray_dir) = makeEyeRay(nextPos.x, nextPos.y);
  const float3 nextColor     = PathTrace2(ray_pos, ray_dir, makeInitialMisData(), 0, 0, nextDepth);

  AddColorInSumm(a_pos, currColor);
  AddColorInSumm(nextPos, nextColor);

  // Accept probability  

  const float currDepthWeight = (float)(currDepth) / (float)(m_maxDepth) * 0.9F + 0.1F;
  const float nextDepthWeight = (float)(nextDepth) / (float)(m_maxDepth) * 0.9F + 0.1F;

  //float currLum = Luminance(currColor) * currDepthWeight * currDepthWeight;
  //float nextLum = Luminance(nextColor) * nextDepthWeight * nextDepthWeight;

  //const float gamma          = 1.0F / 2.2F;
  //currLum                    = pow(currLum, gamma);
  //nextLum                    = pow(nextLum, gamma);

  const float acceptProb = AcceptProb(currDepthWeight, nextDepthWeight);

  if (rndFloat1_Pseudo(&a_gen) < acceptProb)
  {
    //DrawDot(m_stepPass, nextPos, float3(0, 0.2F, 0));
    a_pos = nextPos;
  }
  //else
  //  DrawDot(m_stepPass, nextPos, float3(0.2F, 0, 0));
}




void IntegratorMISPTLoop2Adapt::AdaptDispers(int2& a_pos, RandomGen& a_gen)
{
  // Calculate samples

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir) = makeEyeRay(a_pos.x, a_pos.y);

  const int maxSamplesForDisp = 16;
  std::vector<float> arrLum(maxSamplesForDisp);

  int depth = 0;

  for (int i = 0; i < maxSamplesForDisp; ++i)
  {
    const float3 color = PathTrace2(ray_pos, ray_dir, makeInitialMisData(), 0, 0, depth);
    AddColorInSumm(a_pos, color);

    const float weightFromDepth = (float)(depth) / (float)(m_maxDepth);
    arrLum[i]               = Luminance(color) * weightFromDepth;
  }

  // Calculate MSE

  const float mse      = GetMSEFromVector(arrLum, false);
  const float mean      = MathExp(arrLum, false, false);
  const float error     = mse / fmax(mean, 1e-6F);
  const float compError = error / (1.0F + error);

  // Calculate probability

  const float acceptProb = 1.0F - compError;

  //AddColorInSumm(a_pos, float3(acceptProb, acceptProb, acceptProb));

  const int2  nextPos = GetRndFullScreenPos(a_gen);

  if (rndFloat1_Pseudo(&a_gen) < acceptProb || compError == 0)
  {
    a_pos = nextPos;
    //DrawDot(m_stepPass, nextPos, float3(0.5F, 0, 0));
  }
  //else
  //  DrawDot(m_stepPass, nextPos, float3(0, 0.5F, 0));
}



void IntegratorMISPTLoop2Adapt::VariableStep(int2& a_pos, RandomGen& a_gen, const float a_imgRadius, const bool a_drawPath)
{  
  // Get sample

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir)          = makeEyeRay(a_pos.x, a_pos.y);
  const float3 currColor              = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

  AddColorInSumm(a_pos, currColor);

  // offset x or y for get near pixel

  float2      dirOffset;
  const int2  nearPos                 = GetNewLocalPos(a_pos, 1.5F, dirOffset);
  const int   currIndex               = GetIndexFromPos(a_pos);
  const int   nearIndex               = GetIndexFromPos(nearPos);

  const float lum                     = Luminance(float3(m_summColors[currIndex].x, m_summColors[currIndex].y, m_summColors[currIndex].z));
  const float lumOffsetPix            = Luminance(float3(m_summColors[nearIndex].x, m_summColors[nearIndex].y, m_summColors[nearIndex].z));
          
  const float mean                    = (lum + lumOffsetPix) * 0.5F;
  const float mathExpSquare           = (lum * lum + lumOffsetPix * lumOffsetPix) * 0.5F;  
  const float dispersion              = mathExpSquare - (mean * mean);
  const float mse                     = sqrt(fmax(dispersion, 0.0F));

  // R2 error
  //const int   sizeLocalWin            = 1;
  //const float errorR2                 = R2Error(a_pos, nearPos, sizeLocalWin);

  // offset pixels with direction and step   

  const float maxStep                 = a_imgRadius * 0.5F;
  const float step                    = (1.0F - clamp(mse, 0.0F, 1.0F)) * maxStep + 1.5F;
  int2       nextPos                  = GetPosWithDirAndStep(a_pos, dirOffset, step); 


  if (a_drawPath)
    DrawDot(m_stepPass, a_pos, float3(0, 0.05F, 0));

  a_pos = nextPos;
}


void IntegratorMISPTLoop2Adapt::WalkWithDispers(int2& a_pos, RandomGen& a_gen, const float a_imgRadius, const bool a_drawPath)
{
  // Calculate samples

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir) = makeEyeRay(a_pos.x, a_pos.y);

  const int maxSamplesForDisp = 8;
  std::vector<float> arrayCurrColorLum(maxSamplesForDisp);

  for (size_t i = 0; i < maxSamplesForDisp; ++i)
  {
    const float3 currColor = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
    AddColorInSumm(a_pos, currColor);
    
    arrayCurrColorLum[i]   = Luminance(currColor);
  }

  // Calculate dispersion

  const float disp         = GetMSEFromVector(arrayCurrColorLum, false);
  const float mean         = MathExp(arrayCurrColorLum, false, false);
  const float error        = disp / fmax(mean, 1e-6F);
  const float compError    = error / (1.0F + error);
      
  if (a_drawPath)
    DrawDot(m_stepPass, a_pos, float3(0, 1.0F, 0));

  // step

  const float minStep      = 1.5F;
  const float maxStep      = a_imgRadius * 0.5F;
  const float step         = maxStep + compError * (minStep - maxStep);

  const float2 dirOffset   = GetRndDir(a_gen, false);            
  a_pos = GetNewLocalPos(a_pos, step, dirOffset);
}



void IntegratorMISPTLoop2Adapt::GenerateRndPath(RandomGen& a_gen, std::vector<float3>& a_arrDir) const
{
  for (auto& i : a_arrDir)
    i = normalize(to_float3(rndUniform(&a_gen, -1.0F, 1.0F)));

  a_arrDir[0]   = rndFloat3(&a_gen); // screen pos

  a_arrDir[1].z = -fabs(a_arrDir[1].z); // cam dir
  a_arrDir[1]   = normalize(a_arrDir[1]);
}



void IntegratorMISPTLoop2Adapt::MutatePath(RandomGen& a_gen, std::vector<float3>& a_nextDir, const float a_step) const
{
  float min = -1.0F;

  for (int i = 0; i < a_nextDir.size(); ++i)
  {
    const float3 offset = (rndFloat3(&a_gen) * 2.0F - 1.0F) * a_step;
    a_nextDir[i] += offset;
    
    if (i == 0) min = 0.0F; // a_nextDir[0] - screen coord [0-1].

    if (a_nextDir[i].x < min) a_nextDir[i].x -= offset.x;
    if (a_nextDir[i].y < min) a_nextDir[i].y -= offset.y;
    if (a_nextDir[i].z < min) a_nextDir[i].z -= offset.z;
                 
    if (a_nextDir[i].x >  1.0F) a_nextDir[i].x -= offset.x;
    if (a_nextDir[i].y >  1.0F) a_nextDir[i].y -= offset.y;
    if (a_nextDir[i].y >  1.0F) a_nextDir[i].z -= offset.z;

    if (i > 0)
      a_nextDir[i] = normalize(a_nextDir[i]);
  }
}



void IntegratorMISPTLoop2Adapt::MultiDim(RandomGen& a_gen, std::vector<float3>& a_arrCurrDir)
{
  // so far, everything is not working correctly.
  
  
  // First sample

  const int2 currPos         = make_int2((int)(a_arrCurrDir[0].x * (float)(m_width  - 1)),
                                         (int)(a_arrCurrDir[0].y * (float)(m_height - 1)));
  float3 ray_pos;
  float3 ray_dir;

  std::tie(ray_pos, ray_dir) = makeEyeRay(currPos.x, currPos.y);
  const float3 currColor     = PathTrace3(a_gen, ray_pos, ray_dir, makeInitialMisData(), 0, 0, a_arrCurrDir, true);
  
  std::vector<float3> arrNextDir(a_arrCurrDir);
  MutatePath(a_gen, arrNextDir, 0.001F);

  // Next sample

  const int2 nextPos         = make_int2((int)(a_arrCurrDir[0].x * (float)(m_width - 1)),
                                         (int)(a_arrCurrDir[0].y * (float)(m_height - 1)));

  std::tie(ray_pos, ray_dir) = makeEyeRay(nextPos.x, nextPos.y);
  const float3 nextColor     = PathTrace3(a_gen, ray_pos, ray_dir, makeInitialMisData(), 0, 0, arrNextDir, false);

  // Add in summ

  AddColorInSumm(currPos, currColor);
  AddColorInSumm(nextPos, nextColor);

  // Accept probability

  const float currResLum     = Luminance(currColor);
  const float nextResLum     = Luminance(nextColor);
                             
  const float acceptProp     = AcceptProb(currResLum, nextResLum);

  if (rndFloat1_Pseudo(&a_gen) < acceptProp || acceptProp == 0)
  {
    a_arrCurrDir = arrNextDir;

    //DrawDot(m_stepPass, currPos, float3(0, 1.0F, 0));
  }
  //else
    //DrawDot(m_stepPass, currPos, float3(1.0F, 0, 0));

  //for (int i = 0; i < a_arrCurrDir.size(); ++i)
  //{
  //  std::cout << "a_arrCurrDir[" << i << "]:" << a_arrCurrDir[i].x << " " << a_arrCurrDir[i].y << " " << a_arrCurrDir[i].z << std::endl;
  //  std::cout << "arrNextDir["   << i << "]:" <<   arrNextDir[i].x << " " <<   arrNextDir[i].y << " " <<   arrNextDir[i].z << std::endl;
  //}
}






void IntegratorMISPTLoop2Adapt::MarkovChain(int2& a_pos, RandomGen& a_gen, float& a_step, const float a_imgRadius, const bool a_drawPath)
{
  // Calculate sample

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir) = makeEyeRay(a_pos.x, a_pos.y);
  const float3 currColor = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

  AddColorInSumm(a_pos, currColor);

  // luminance

  //const float currLum        = Luminance(colorCurr);
  //const int   nextIndex      = GetIndexFromPos(nextPos);
  //const float nextLum        = Luminance(m_summColors[nextIndex]);

  // local statistics

  const int sizeLocalWin = 1;
  //float currMean(0);  
  //float currMediana(0);
  //float currMax(0);
  //float nextMean(0);
  //float nextMediana(0);
  //float nextMax(0);

  //GetStatisticsLocalWin(a_pos,   sizeLocalWin, currMean, currMediana, currMax);
  //GetStatisticsLocalWin(nextPos, sizeLocalWin, nextMean, nextMediana, nextMax);

  // MSE

  //const float currMseLum    = GetLocalMSE(  a_pos, sizeLocalWin);
  //const float nextMseLum    = GetLocalMSE(nextPos, sizeLocalWin);

  // step

  //const float minStep        = a_imgRadius * 0.01F + 1.5F;
  //const float maxStep        = a_imgRadius * 0.5F  + 1.5F;
  //a_step                     = maxStep + acceptProb * (minStep - maxStep);
  a_step = a_imgRadius * 0.01F + 1.5F;

  // first uniform samples

  if (m_spp < 1)
  {
    a_pos = GetRndFullScreenPos(a_gen);
    return;
  }

  const float2 dirOffset = GetRndDir(a_gen, false);
  const int2   nextPos   = GetNewLocalPos(a_pos, a_step, dirOffset);

  const float errorR2    = R2Error(a_pos, nextPos, sizeLocalWin, false);
  const float acceptProb = 0.01F + errorR2 * 0.99F;

  if (rndFloat1_Pseudo(&a_gen) < acceptProb)
  {
    if (a_drawPath)
      DrawDot(m_stepPass, a_pos, float3(0, 0.1F, 0));

    a_pos = nextPos;
  }
  else if (a_drawPath)
    DrawDot(m_stepPass, a_pos, float3(0.05F, 0, 0));

}



void IntegratorMISPTLoop2Adapt::MarkovChain2(int2& a_pos, RandomGen& a_gen, const float a_imgRadius, int& a_sample)
{
  // Get first sample

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir)  = makeEyeRay(a_pos.x, a_pos.y);
  const float3 colorCurr      = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

  // Get next local screen pos 

  float2 dirOffset;
  const float2 rnd            = rndFloat2_Pseudo(&a_gen);  
  const float  step           = a_imgRadius * 0.01F + 1.5F;
  const int2   newPos         = GetNewLocalPos(a_pos, step, dirOffset);


  // Get next sample

  std::tie(ray_pos, ray_dir)  = makeEyeRay(newPos.x, newPos.y);
  const float3 colorNext      = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

#pragma omp atomic
  a_sample++;

  // acceptance probability 

  float lumCurrPix            = Luminance(colorCurr);
  float lumNextPix            = Luminance(colorNext);
  float acceptProb            = AcceptProb(lumCurrPix, lumNextPix);
  acceptProb                  = pow(acceptProb, 1.0F / 2.2F);

  // add color

  const float3 summColorCurr = colorCurr;// *(1.0F - acceptProb);// / fmax(probCurr, 1e-6F);
  const float3 summColorNext = colorNext;// *acceptProb;// / fmax(probNext, 1e-6F);

  const int indexCurr    = a_pos.y * m_width + a_pos.x;
#pragma omp atomic
  m_summColors[indexCurr].x += summColorCurr.x;
#pragma omp atomic
  m_summColors[indexCurr].y += summColorCurr.y;
#pragma omp atomic
  m_summColors[indexCurr].z += summColorCurr.z;
#pragma omp atomic
  m_summColors[indexCurr].w += (1.0F - acceptProb);
#pragma omp atomic
  m_samplePerPix[indexCurr]++;

  const int indexNext    = newPos.y * m_width + newPos.x;
#pragma omp atomic
  m_summColors[indexNext].x += summColorNext.x;
#pragma omp atomic
  m_summColors[indexNext].y += summColorNext.y;
#pragma omp atomic
  m_summColors[indexNext].z += summColorNext.z;
#pragma omp atomic
  m_summColors[indexNext].w += acceptProb;
#pragma omp atomic
  m_samplePerPix[indexNext]++;

  // acceptance probability

  if (rndFloat1_Pseudo(&a_gen) < acceptProb || acceptProb < 1e-20F)
    a_pos = newPos;      
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IntegratorMISPTLoop2Adapt::AddColorInSumm(const int2 a_pos, const float3 a_color)
{
  const int   index   = GetIndexFromPos(a_pos);
  const float alpha   = 1.0F / float(m_samplePerPix[index] + 1);
  m_summColors[index] = m_summColors[index] * (1.0F - alpha) + to_float4(a_color, maxcomp(a_color)) * alpha;

#pragma omp atomic
  m_samplePerPix[index]++;
}

void IntegratorMISPTLoop2Adapt::AddColorAndPdfPpInSumm(const int2 a_pos, const float3 a_color, const float a_pdfPp)
{
  const int   index   = GetIndexFromPos(a_pos);
  const float alpha   = 1.0F / float(m_samplePerPix[index] + 1);
  m_summColors[index] = m_summColors[index] * (1.0F - alpha) + to_float4(a_color, maxcomp(a_color)) * alpha;

#pragma omp atomic
  m_samplePerPix[index]++;
}


float IntegratorMISPTLoop2Adapt::Correlation(const int2 a_pos, const int2 a_nextPos, const int a_sizeLocalWindow, const bool a_generalAggregate)
{  
  const float currMse     = GetLocalMSE(a_pos,     a_sizeLocalWindow, a_generalAggregate);
  const float nextMse     = GetLocalMSE(a_nextPos, a_sizeLocalWindow, a_generalAggregate);

  const float covariance  = Covariance(a_pos, a_nextPos, a_sizeLocalWindow, a_generalAggregate);
  const float dotMse      = currMse * nextMse;
  const float correlation = covariance / fmax(dotMse, 1e-6F);

  //if (m_spp >= 2)
  //{
  //  std::cout << "currMse     = " << currMse     << std::endl;
  //  std::cout << "nextMse     = " << nextMse     << std::endl;
  //  std::cout << "covariance  = " << covariance  << std::endl;
  //  std::cout << "dotMse      = " << dotMse      << std::endl;
  //  std::cout << "correlation = " << correlation << std::endl;
  //  std::cout << std::endl;
  //}

  return correlation;
}

float IntegratorMISPTLoop2Adapt::Correlation(const std::vector<float> a_array1, const std::vector<float> a_array2, const bool a_generalAggregate)
{
  const float currMse     = GetMSEFromVector(a_array1, a_generalAggregate);
  const float nextMse     = GetMSEFromVector(a_array2, a_generalAggregate);

  const float covariance  = Covariance(a_array1, a_array2, a_generalAggregate);
  const float dotMse      = currMse * nextMse;
  const float correlation = covariance / fmax(dotMse, 1e-6F);

  //std::cout << "currMse     = " << currMse     << std::endl;
  //std::cout << "nextMse     = " << nextMse     << std::endl;
  //std::cout << "covariance  = " << covariance  << std::endl;
  //std::cout << "dotMse      = " << dotMse      << std::endl;
  //std::cout << "correlation = " << correlation << std::endl;
  //std::cout << std::endl;

  return correlation;
}

float IntegratorMISPTLoop2Adapt::Covariance(const int2 a_pos, const int2 a_nextPos, const int a_sizeLocalWindow, const bool a_generalAggregate)
{
  const float summMathExp = MathExp2(a_pos, a_nextPos, a_sizeLocalWindow);
  const float currMathExp = MathExp(a_pos,     a_sizeLocalWindow, false, a_generalAggregate);
  const float nextMathExp = MathExp(a_nextPos, a_sizeLocalWindow, false, a_generalAggregate);

  const float covariance = summMathExp - currMathExp * nextMathExp;

  //if (m_spp >= 2)
  //{
  //  std::cout << "summMathExp = " << summMathExp << std::endl;
  //  std::cout << "currMathExp = " << currMathExp << std::endl;
  //  std::cout << "nextMathExp = " << nextMathExp << std::endl;
  //}

  return covariance;
}


float IntegratorMISPTLoop2Adapt::Covariance(const std::vector<float> a_array1, const std::vector<float> a_array2, const bool a_generalAggregate) const
{
  const float summMathExp = MathExp2(a_array1, a_array2, (int)a_array1.size());
  const float currMathExp = MathExp(a_array1, false, a_generalAggregate);
  const float nextMathExp = MathExp(a_array2, false, a_generalAggregate);

  const float covariance = summMathExp - currMathExp * nextMathExp;

  //if (m_spp >= 2)
  //{
  //  std::cout << "summMathExp = " << summMathExp << std::endl;
  //  std::cout << "currMathExp = " << currMathExp << std::endl;
  //  std::cout << "nextMathExp = " << nextMathExp << std::endl;
  //}

  return covariance;
}

float IntegratorMISPTLoop2Adapt::R2Error(const int2 a_pos, const int2 a_nextPos, const int a_sizeLocalWindow, const bool a_generalAggregate)
{
  const float correlation = Correlation(a_pos, a_nextPos, a_sizeLocalWindow, a_generalAggregate);
  const float R2          = 1.0F - correlation * correlation;  
  return R2;
}

float IntegratorMISPTLoop2Adapt::R2Error(const std::vector<float> a_array1, const std::vector<float> a_array2, const bool a_generalAggregate)
{
  const float correlation = Correlation(a_array1, a_array2, a_generalAggregate);
  const float R2          = 1.0F - correlation * correlation;  
  return R2;
}

void IntegratorMISPTLoop2Adapt::DrawLine(std::vector<float3>& a_img, const int2 a_pos, const int2 a_nextPos, const float3 a_color) const
{
  int2 currPos;
  const float dist = Distance(a_pos, a_nextPos);

  for (int i = 0; i < (int)dist; ++i)
  {
    const float step = (float)i / dist;
    currPos.x = (int)(lerp((float)a_pos.x, (float)a_nextPos.x, step) + 0.5F);
    currPos.y = (int)(lerp((float)a_pos.y, (float)a_nextPos.y, step) + 0.5F);

    const int index = GetIndexFromPos(currPos);
    a_img[index] = a_color * (0.5F + step * 0.5F);
  }
}

void IntegratorMISPTLoop2Adapt::DrawDot(std::vector<float3>& a_img, const int2 a_pos, const float3 a_color) const
{
  const int currIndex = GetIndexFromPos(a_pos);
  a_img[currIndex]   += a_color;
}

int IntegratorMISPTLoop2Adapt::GetSummSppLocalWin(const int2 a_pos, const int a_sizeLocalWindow)
{
  int summ = 0;

  for (int offsetY = -a_sizeLocalWindow; offsetY <= a_sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -a_sizeLocalWindow; offsetX <= a_sizeLocalWindow; offsetX++)
    {
      int NewX        = a_pos.x + offsetX;
      int NewY        = a_pos.y + offsetY;

      NewX            = clamp(NewX, 0, m_width  - 1);
      NewY            = clamp(NewY, 0, m_height - 1);

      const int index = GetIndexFromPos(NewX, NewY);

      summ           += m_samplePerPix[index];
    }
  }

  return summ;
}

int IntegratorMISPTLoop2Adapt::GetIndexFromPos(const int x, const int y) const
{
  return y * m_width + x;
}

int IntegratorMISPTLoop2Adapt::GetIndexFromPos(const int2 pos) const
{
  return pos.y * m_width + pos.x;
}

int GetSizeLocalWin(const int a_sizeLocalWindow)
{
  const int sizeLine  = a_sizeLocalWindow * 2 + 1;
  const int sizeArray = sizeLine * sizeLine;
  return sizeArray;
}

void IntegratorMISPTLoop2Adapt::GetStatisticsLocalWin(const int2 a_pos, const int a_sizeLocalWindow, float& a_mean, float& a_mediana,
  float& a_max)
{
  a_mean    = 0.0F;
  a_mediana = 0.0F;
  a_max     = 0.0F;

  const int sizeArray = GetSizeLocalWin(a_sizeLocalWindow);
  float     weight    = 1.0F / (float)sizeArray;
  std::vector<float> pixels;

  for (int offsetY = -a_sizeLocalWindow; offsetY <= a_sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -a_sizeLocalWindow; offsetX <= a_sizeLocalWindow; offsetX++)
    {
      int2 newPos     = int2(a_pos.x + offsetX, a_pos.y + offsetY);

      newPos.x        = clamp(newPos.x, 0, m_width  - 1);
      newPos.y        = clamp(newPos.y, 0, m_height - 1);

      const int index = GetIndexFromPos(newPos);

      const float val = LuminanceFloat4(m_summColors[index]);

      pixels.push_back(val);
      a_mean += val * weight;
    }
  }

  std::sort(pixels.begin(), pixels.end());

  const int lastIndex   = sizeArray - 1;
  a_max                 = pixels[lastIndex];
  a_mediana             = pixels[int(float(sizeArray) / 2.0F)];
}


float IntegratorMISPTLoop2Adapt::MathExp(const int2 a_pos, const int a_sizeLocalWindow, const bool a_square, const bool a_generalAggregate)
{
  float mathExp = 0.0F;

  const int sizeArray = a_generalAggregate ? GetSizeLocalWin(a_sizeLocalWindow) : GetSizeLocalWin(a_sizeLocalWindow) - 1;
  float     weight    = 1.0F / (float)sizeArray;

  for (int offsetY = -a_sizeLocalWindow; offsetY <= a_sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -a_sizeLocalWindow; offsetX <= a_sizeLocalWindow; offsetX++)
    {
      int NewX = a_pos.x + offsetX;
      int NewY = a_pos.y + offsetY;

      NewX = clamp(NewX, 0, m_width - 1);
      NewY = clamp(NewY, 0, m_height - 1);

      const int index = GetIndexFromPos(NewX, NewY);

      const float val = LuminanceFloat4(m_summColors[index]);

      if (a_square)
        mathExp += (val * val * weight);
      else
        mathExp += (val * weight);
    }
  }

  return mathExp;
}

float IntegratorMISPTLoop2Adapt::MathExp(const std::vector<float> a_array, const bool a_square, const bool a_generalAggregate) const
{
  const int sizeArray = a_generalAggregate ? (int)a_array.size() : (int)a_array.size() - 1;

  if (a_square)
  {
    float summ = 0.0F;

    for (auto& i : a_array)    
      summ += (i * i);
    
    return summ / (float)sizeArray ;
  }
  else
    return std::accumulate(a_array.begin(), a_array.end(), 0.0F) / (float)sizeArray;
}


float IntegratorMISPTLoop2Adapt::GetLocalDispers(const int2 a_pos, const int a_sizeLocalWindow, const bool a_generalAggregate)
{
  const float mathExp       = MathExp(a_pos, a_sizeLocalWindow, false, a_generalAggregate);
  const float mathExpSquare = MathExp(a_pos, a_sizeLocalWindow, true,  a_generalAggregate);
  const float dispersion    = mathExpSquare - (mathExp * mathExp);

  return fmax(dispersion, 0.0F);
}


float IntegratorMISPTLoop2Adapt::GetDispersFromVector(const std::vector<float> a_array, const bool a_generalAggregate) const
{
  const float mathExp       = MathExp(a_array, false, a_generalAggregate);
  const float mathExpSquare = MathExp(a_array, true,  a_generalAggregate);
  const float dispersion    = mathExpSquare - (mathExp * mathExp);

  return fmax(dispersion, 0.0F);
}


float IntegratorMISPTLoop2Adapt::GetLocalMSE(const int2 a_pos, const int a_sizeLocalWindow, const bool a_generalAggregate)
{
  return sqrt(GetLocalDispers(a_pos, a_sizeLocalWindow, a_generalAggregate));
}


float IntegratorMISPTLoop2Adapt::GetMSEFromVector(const std::vector<float> a_array, const bool a_generalAggregate) const
{
  return sqrt(GetDispersFromVector(a_array, a_generalAggregate));
}



float IntegratorMISPTLoop2Adapt::MathExp2(const int2 a_pos, const int2 a_nextPos, const int a_sizeLocalWindow)
{
  float       mathExp       = 0.0F;
  const int   sizeArray     = GetSizeLocalWin(a_sizeLocalWindow);
  const float weight        = 1.0F / (float)sizeArray;


  for (int offsetY = -a_sizeLocalWindow; offsetY <= a_sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -a_sizeLocalWindow; offsetX <= a_sizeLocalWindow; offsetX++)
    {
      int2 currNew           = int2(    a_pos.x + offsetX,     a_pos.y + offsetY);
      int2 nextNew           = int2(a_nextPos.x + offsetX, a_nextPos.y + offsetY);

      currNew.x              = clamp(currNew.x, 0, m_width  - 1);
      currNew.y              = clamp(currNew.y, 0, m_height - 1);

      nextNew.x              = clamp(nextNew.x, 0, m_width  - 1);
      nextNew.y              = clamp(nextNew.y, 0, m_height - 1);

      const int   currIndex  = GetIndexFromPos(currNew);
      const int   nextIndex  = GetIndexFromPos(nextNew);

      const float currVal    = LuminanceFloat4(m_summColors[currIndex]);
      const float nextVal    = LuminanceFloat4(m_summColors[nextIndex]);

      mathExp += (currVal * nextVal * weight);
    }
  }

  return mathExp;
}

float IntegratorMISPTLoop2Adapt::MathExp2(const std::vector<float> a_array1, const std::vector<float> a_array2, const int a_sizeArray) const
{
  float summ = 0.0F;

  for (int i = 0; i < a_sizeArray; i++)
    summ += (a_array1[i] * a_array2[i]);

  return summ / (float)a_sizeArray;
}



int2 IntegratorMISPTLoop2Adapt::GetNewLocalPos(const LiteMath::int2 a_pos, const float a_step, const float2 a_dirOffset) const 
{
  const int2 offset = int2(int(a_dirOffset.x * a_step), int(a_dirOffset.y * a_step));
  int2       newPos = int2(a_pos.x + offset.x, a_pos.y + offset.y);

  if (newPos.x < 0)            newPos.x -= offset.x;
  if (newPos.y < 0)            newPos.y -= offset.y;
  if (newPos.x > m_width  - 1) newPos.x -= offset.x;
  if (newPos.y > m_height - 1) newPos.y -= offset.y;

  return newPos;
}

float2 IntegratorMISPTLoop2Adapt::GetRndDir(RandomGen& a_gen, const bool a_constantStep) const 
{
  float2 dir(0.0F, 0.0F);

  if (a_constantStep)
  {
    const float rnd   = rndFloat1_Pseudo(&a_gen);
    const float angle = rnd * M_TWOPI;
    dir.x             = sin(angle);
    dir.y             = cos(angle);
  }
  else
  {
    const float2 rnd  = rndFloat2_Pseudo(&a_gen);
    dir.x             = rnd.x * 2.0F - 1.0F;
    dir.y             = rnd.y * 2.0F - 1.0F;
  }

  return dir;
}


int2 IntegratorMISPTLoop2Adapt::GetRndFullScreenPos(RandomGen& a_gen) const 
{
  const float2 rnd = rndFloat2_Pseudo(&a_gen);

  return int2(int(rnd.x * ((float)m_width - 0.5F)), int(rnd.y * ((float)m_height - 0.5F)));
}


int2 IntegratorMISPTLoop2Adapt::GetPosWithDirAndStep(const int2 a_pos, const float2 a_dirOffset, const float a_step) const
{
  const int2 offset = int2(int(a_dirOffset.x * a_step), int(a_dirOffset.y * a_step));
  int2       newPos = int2(a_pos.x + offset.x, a_pos.y + offset.y);

  if (newPos.x < 0)            newPos.x -= offset.x;
  if (newPos.y < 0)            newPos.y -= offset.y;
  if (newPos.x > m_width  - 1) newPos.x -= offset.x;
  if (newPos.y > m_height - 1) newPos.y -= offset.y;

  return newPos;
}



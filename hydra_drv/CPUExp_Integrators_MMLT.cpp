#include <omp.h>
#include "CPUExp_Integrators.h"
#include "time.h"

#include <algorithm> 

bool HR_SaveHDRImageToFileLDR(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scaleInv, const float a_gamma = 2.2f);
bool HR_SaveHDRImageToFileHDR(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scale = 1.0f);


static float3 EstimateAverageBrightnessRGB2(const float4* a_color, int a_size)
{
  const float4* in_color = a_color;
  int imageSize = a_size;

  float4 summ(0, 0, 0, 0);
  for (int i = 0; i < imageSize; i++)
  {
    float4 color = a_color[i];
    summ += color;
  }
  return to_float3(summ) / float(imageSize);
}

static float EstimateAverageBrightness2(const float4* a_color, int a_size)
{
  const float3 color = EstimateAverageBrightnessRGB2(a_color, a_size);
  const float val = contribFunc(color);
  if (val < 1e-20f)
    return 1e-20f;
  else
    return val;
}

void IntegratorMMLT::SetMaxDepth(int a_depth)
{
  m_maxDepth = a_depth;
  const int randArraySize = randArraySizeOfDepthMMLT(m_maxDepth); // let say d = 2 => (we have 3 vertices) => one material bounce for camera strategy;

  for (size_t i = 0; i < m_perThread.size(); i++)
  {
    m_perThread[i].pdfArray.resize(m_maxDepth + 1);
    m_pss[i].resize(randArraySize);

    m_perThread[i].gen.rptr  = nullptr;
    m_perThread[i].gen2.rptr = nullptr;
  }

}

IntegratorMMLT::PSSampleV IntegratorMMLT::InitialSamplePS(const int d, const int a_burnIters)
{
  PSSampleV result(randArraySizeOfDepthMMLT(d));
  for (size_t j = 0; j<result.size(); j++)
    result[j] = rndFloat1_Pseudo(&PerThread().gen);
  return result;
}

void IntegratorMMLT::MutateLightPart(PSSampleV& v2, int s, RandomGen* pGen)
{
  const int lightBegin = MMLT_HEAD_TOTAL_SIZE;
  const int camBegin   = camOffsetInRandArrayMMLT(s);

  // mutate light sample
  //
  for (size_t j = 4; j < 10; j++)
    v2[j] = MutateKelemen(v2[j], rndFloat2_Pseudo(pGen), MUTATE_COEFF_BSDF, 1024.0f);

  // mutate light path
  //
  for (size_t j = lightBegin; j < camBegin; j++)
    v2[j] = MutateKelemen(v2[j], rndFloat2_Pseudo(pGen), MUTATE_COEFF_BSDF, 1024.0f);
}

void IntegratorMMLT::MutateCameraPart(PSSampleV& v2, int s, RandomGen* pGen)
{
  const int camBegin = camOffsetInRandArrayMMLT(s);

  // mutate lens
  //
  v2[0] = MutateKelemen(v2[0], rndFloat2_Pseudo(pGen), MUTATE_COEFF_SCREEN*1.0f, 1024.0f);
  v2[1] = MutateKelemen(v2[1], rndFloat2_Pseudo(pGen), MUTATE_COEFF_SCREEN*1.0f, 1024.0f);
  v2[2] = MutateKelemen(v2[2], rndFloat2_Pseudo(pGen), MUTATE_COEFF_BSDF, 1024.0f);
  v2[3] = MutateKelemen(v2[3], rndFloat2_Pseudo(pGen), MUTATE_COEFF_BSDF, 1024.0f);

  // mutate cam path
  //
  for (size_t j = camBegin; j < v2.size(); j++)
    v2[j] = MutateKelemen(v2[j], rndFloat2_Pseudo(pGen), MUTATE_COEFF_BSDF, 1024.0f);
}


IntegratorMMLT::PSSampleV IntegratorMMLT::MutatePrimarySpace(const PSSampleV& a_vec, int d, int* pMutationType)
{
  auto& gen = randomGen();

  //////////////////////////////////////////////////////////////////////////////////// randomize generator
  if (clock() % 4 == 0)
  {
    const int NRandomisation = (clock() % 16) + (clock() % 3);
    for (int i = 0; i < NRandomisation; i++)
      NextState(&gen);
  }
  //////////////////////////////////////////////////////////////////////////////////// 

  PSSampleV v2(a_vec);

  const float plarge   = 0.33f;                     // 33% for large step;
  const float selector = rndFloat1_Pseudo(&gen);

  const float plight      = 0.20f;
  const float pmultiChain = 0.16f;
  //const float pcamera     = 1.0f - plarge - plight - pmultiChain;

  if (selector < plarge)                            // large step
  {
    for (size_t j = 0; j<v2.size(); j++)
      v2[j] = rndFloat1_Pseudo(&gen);
    (*pMutationType) = MUTATE_LIGHT | MUTATE_CAMERA;
  }
  else                                              // small step
  {
    const int currSplit = rndSplitMMLT(&gen, &v2[0], d); // mapRndFloatToInt(v2[9], 0, d);
  
    if (plarge < selector && selector <= plarge + plight)      // small step for light path
    {
      MutateLightPart(v2, currSplit, &gen);
      (*pMutationType) = MUTATE_LIGHT;
    }
    else if(plarge + plight < selector && selector <= plarge + plight + pmultiChain)
    {
      MutateLightPart(v2, currSplit, &gen);
      MutateCameraPart(v2, currSplit, &gen);
      (*pMutationType) = MUTATE_LIGHT | MUTATE_CAMERA;
    }
    else                                            // small step for camera path
    {
      MutateCameraPart(v2, currSplit, &gen);
      (*pMutationType) = MUTATE_CAMERA;
    }
  }
  
  return v2;
}

float3 IntegratorMMLT::F(const PSSampleV& a_xVec, const int d, int m_type,
                         int* pX, int* pY)
{
  auto* pGen = &PerThread().gen;
  m_pss[ThreadId()] = a_xVec;

  // select pair of (s,t) where 's' (split) is a light source and 't' (tail) is the camera 
  //
  pGen->rptr  = &m_pss[ThreadId()][0];
  const int s = rndSplitMMLT(pGen, pGen->rptr, d);
  const int t = d - s;                                       // note that s=2 means 1 light bounce and one connection!!!

  const int lightTraceDepth = s - 1;  // because the last light path is a connection anyway - to camera or to camera path
  const int camTraceDepth   = t;      //  

  const float screenScaleX = m_pGlobals->varsF[HRT_MLT_SCREEN_SCALE_X];
  const float screenScaleY = m_pGlobals->varsF[HRT_MLT_SCREEN_SCALE_Y];
  const float4 lensOffs    = rndLens(pGen, pGen->rptr, make_float2(screenScaleX, screenScaleY), 0, 0, 0);

  int x = (int)(lensOffs.x*float(m_width)  + 0.5f); // rndInt(&PerThread().gen, 0, m_width);  // light tracing can overwtite this variables
  int y = (int)(lensOffs.y*float(m_height) + 0.5f); // rndInt(&PerThread().gen, 0, m_height); // light tracing can overwtite this variables

  const bool doNotChangeLightVertex  = !(m_type & MUTATE_LIGHT) && (m_type & MUTATE_CAMERA); // don't recalculate subpath if you didn't change it actually
  const bool doNotChangeCameraVertex = (m_type & MUTATE_LIGHT) && !(m_type & MUTATE_CAMERA); // don't recalculate subpath if you didn't change it actually

  //////////////////////////////////////////////////////////
  m_debugRaysPos[ThreadId()].resize(d);
  for (int i = 0; i < d; i++)
    m_debugRaysPos[ThreadId()][i] = float4(0, 0, 0, 0);
  //////////////////////////////////////////////////////////

  // (1) trace path from camera with depth == camTraceDepth and (x,y)
  //
  PathVertex cv;
  InitPathVertex(&cv);
                                     
  if (doNotChangeCameraVertex)     // need to restore pdfArrays also !!!
    cv = m_oldCameraV[ThreadId()]; // need to restore pdfArrays also !!!
  if (camTraceDepth > 0)
  {
    const int camRandNumbersOffset = camOffsetInRandArrayMMLT(s);
    pGen->rptr = &m_pss[ThreadId()][camRandNumbersOffset]; // (lens, light), light path, and finally camera path

    float fx, fy;
    float3 ray_pos, ray_dir;
    MakeEyeRayFromF4Rnd(lensOffs, m_pGlobals,
                        &ray_pos, &ray_dir, &fx, &fy); // x and y can be overwriten by ConnectEye later

    x = int(fx + 0.5f);
    y = int(fy + 0.5f);

    if (x >= m_width)  x = m_width - 1;
    if (y >= m_height) y = m_height - 1;

    const bool haveToHitLight = (lightTraceDepth == -1);  // when lightTraceDepth == -1, use only camera strategy, so have to hit light at some depth level   

    cv = CameraPath(ray_pos, ray_dir, makeInitialMisData(), 1, 0,
                    &PerThread(), camTraceDepth, haveToHitLight, d);
  }

  // (2) trace path from light with depth = lightTraceDepth;
  //
  PathVertex lv;
  InitPathVertex(&lv);

  if (doNotChangeLightVertex)      //  need to restore pdfArrays also !!!
    lv = m_oldLightV[ThreadId()];  //  need to restore pdfArrays also !!!
  if (lightTraceDepth > 0)
    lv = LightPath(&PerThread(), lightTraceDepth);
  
  // (3) connect; this operation should also compute missing pdfA for camera and light 
  //
  float3 sampleColor(0, 0, 0);

  if (lightTraceDepth == -1)        // (3.1) -1 means we have full camera path, no conection is needed
  {
    sampleColor = cv.accColor;
  }
  else
  {
    if (camTraceDepth == 0)         // (3.2) connect light vertex to camera (light tracing connection)
    {
      if (lv.valid)
      {
        sampleColor = ConnectEye(lv, lightTraceDepth,
                                 &PerThread(), &x, &y);
      }
    }
    else if (lightTraceDepth == 0)  // (3.3) connect camera vertex to light (shadow ray)
    {
      if (cv.valid && !cv.wasSpecOnly) // cv.wasSpecOnly exclude direct light actually
      {
        float3 explicitColor = ConnectShadow(cv, &PerThread(), t);
        sampleColor = cv.accColor*explicitColor;
      }
    }
    else                            // (3.4) connect light and camera vertices (bidir connection)
    {
      if (cv.valid)
      {
        float3 explicitColor = ConnectEndPoints(lv, cv, s, d, &PerThread());
        sampleColor = cv.accColor*explicitColor*lv.accColor;
      }
    }
  }

  m_oldLightV [ThreadId()] = lv;
  m_oldCameraV[ThreadId()] = cv;

  // (4) calc MIS weights
  //
  float misWeight = 1.0f;
  if (dot(sampleColor, sampleColor) > 1e-12f)
  {
    float pdfThisWay = 1.0f;
    float pdfSumm = 0.0f;
    int   validStrat = 0;

    for (int split = 0; split <= d; split++)
    {
      const int s1 = split;
      const int t1 = d - split;

      const bool specularMet = (split > 0) && (split < d) && (PerThread().pdfArray[split].pdfRev < 0.0f || PerThread().pdfArray[split].pdfFwd < 0.0f);
      float pdfOtherWay = specularMet ? 0.0f : 1.0f;
      if (split == d)
        pdfOtherWay = misHeuristicPower1(PerThread().pdfArray[d].pdfFwd);

      for (int i = 0; i < s1; i++)
        pdfOtherWay *= misHeuristicPower1(PerThread().pdfArray[i].pdfFwd);
      for (int i = s1 + 1; i <= d; i++)
        pdfOtherWay *= misHeuristicPower1(PerThread().pdfArray[i].pdfRev);

      if (pdfOtherWay != 0.0f)
        validStrat++;

      if (s1 == s && t1 == t)
        pdfThisWay = pdfOtherWay;

      pdfSumm += pdfOtherWay;
    }

    misWeight = pdfThisWay / fmax(pdfSumm, DEPSILON2);
  }

  sampleColor *= misWeight;

  if (x >= 0 && x < m_width && y >= 0 && y < m_height)
  {
    (*pX) = x;
    (*pY) = y;
    if (m_mask != nullptr)                   // for mixing mmlt with 3-way
      sampleColor *= m_mask[y*m_width + x];  // 
  }
  else
  {
    (*pX)       = 0;
    (*pY)       = 0;
    sampleColor = make_float3(0, 0, 0);
  }

  //if (g_useKelemen)
  //{
  //  const float f1 = contribFunc(sampleColor) / g_bKelemen;
  //  const float w  = f1 / (f1 + 1.0f);
  //  sampleColor *= w;
  //}

  return sampleColor;
}

static std::vector<float> PrefixSumm(const std::vector<float>& a_vec)
{
  float accum = 0.0f;
  std::vector<float> avgBAccum(a_vec.size() + 1);
  for (size_t i = 0; i < a_vec.size(); i++)
  {
    avgBAccum[i] = accum;
    accum += a_vec[i];
  }
  avgBAccum[avgBAccum.size() - 1] = accum;
  return avgBAccum;
}

//int SelectIndexPropTo(const float a_r, const std::vector<float>& a_vec, float* pPDF)
//{
//  int  d = 0;
//  auto avgBAccum = a_vec;
//  const float selector = a_r*avgBAccum[avgBAccum.size() - 1];
//
//  for (int i = 0; i < avgBAccum.size() - 1; i++)
//  {
//    if (avgBAccum[i] < selector && selector <= avgBAccum[i + 1])
//    {
//      d = i;
//      (*pPDF) = (avgBAccum[i + 1] - avgBAccum[i]) / avgBAccum[avgBAccum.size() - 1];
//      break;
//    }
//  }
//  return d;
//}

void IntegratorMMLT::DoPassIndirectMLT(float4* a_outImage)
{
  float pdfSelector = 1.0f;
  auto avgBAccum  = PrefixSumm(m_avgBPerBounce);
  const float r   = rndFloat1_Pseudo(&PerThread().gen);
  const int   d   = SelectIndexPropToOpt(r, &avgBAccum[0], int(avgBAccum.size()), &pdfSelector);  
  const float wk  = 1.0f; // (m_avgBPerBounce[d] / m_avgBrightness) / pdfSelector; // because it will be wk / wk ...
  DoPassIndirectMLT(d, wk, a_outImage);
}

void IntegratorMMLT::DoPassIndirectMLT(int d, float a_bkScale, float4* a_outImage)
{
  auto& gen2 = m_perThread[ThreadId()].gen2;

  //////////////////////////////////////////////////////////////////////////////////// randomize generator
  if (clock() % 3 == 0)
  {
    const int NRandomisation = (clock() % 9) + (clock() % 4);
    for (int i = 0; i < NRandomisation; i++)
      NextState(&gen2);
  }
  //////////////////////////////////////////////////////////////////////////////////// 

  const int samplesPerPass = m_width*m_height;
  mLightSubPathCount = float(samplesPerPass);

  // select seed
  //
  int xScr = 0, yScr = 0;
  auto   xVec   = InitialSamplePS(d);
  float3 yColor = F(xVec, d, (MUTATE_CAMERA | MUTATE_LIGHT), &xScr, &yScr);
  float  y      = contribFunc(yColor);

  // run MCMC
  //
  int accept = 0;
  
  for (int sampleId = 0; sampleId < samplesPerPass; sampleId++)
  {
    int mtype = 0;
    auto xOld  = xVec;
    auto xNew  = MutatePrimarySpace(xOld, d, &mtype);

    float  yOld      = y;
    float3 yOldColor = yColor;

    int xScrOld = xScr, yScrOld = yScr;
    int xScrNew = 0,    yScrNew = 0;

    float3 yNewColor = F(xNew, d, mtype, &xScrNew, &yScrNew);
    float  yNew      = contribFunc(yNewColor);

    float a = (yOld == 0.0f) ? 1.0f : fminf(1.0f, yNew / yOld);

    float p = rndFloat1_Pseudo(&gen2);

    if (p <= a) // accept //
    {
      xVec   = xNew;
      y      = yNew;
      yColor = yNewColor;
      xScr   = xScrNew;
      yScr   = yScrNew;
      accept++;
    }
    else        // reject
    {
      //x      = x;
      //y      = y;
      //yColor = yColor;
    }

    // (5) contrib to image
    //
    float3 contribAtX = a_bkScale*yOldColor*(1.0f / fmaxf(yOld, 1e-6f))*(1.0f - a);
    float3 contribAtY = a_bkScale*yNewColor*(1.0f / fmaxf(yNew, 1e-6f))*a;

    if (dot(contribAtX, contribAtX) > 1e-12f)
    { 
      const int offset = yScrOld*m_width + xScrOld;
      #pragma omp atomic
      a_outImage[offset].x += contribAtX.x;
      #pragma omp atomic
      a_outImage[offset].y += contribAtX.y;
      #pragma omp atomic
      a_outImage[offset].z += contribAtX.z;
      #pragma omp atomic
      a_outImage[offset].w += (1.0f-a);
    }

    if (dot(contribAtY, contribAtY) > 1e-12f)
    { 
      const int offset = yScrNew*m_width + xScrNew;
      #pragma omp atomic
      a_outImage[offset].x += contribAtY.x;
      #pragma omp atomic
      a_outImage[offset].y += contribAtY.y;
      #pragma omp atomic
      a_outImage[offset].z += contribAtY.z;
      #pragma omp atomic
      a_outImage[offset].w += a;
    }
    
  }

  if (omp_get_thread_num() == 0)
  {
    float acceptanceRate = float(accept) / float(samplesPerPass);
    auto oldPrecition = std::cout.precision(3);
    std::cout << "[MMLT]: acceptanceRate = " << 100.0f*acceptanceRate << "%" << std::endl;
    std::cout.precision(oldPrecition);
  }
}


float IntegratorMMLT::DoPassEstimateAvgBrightness()
{
  std::cout << "MMLT: estimating avg brightness ... " << std::endl;
  
  m_avgBPerBounce.resize(m_maxDepth+1);
  for (size_t i = 0; i < m_avgBPerBounce.size(); i++)
    m_avgBPerBounce[i] = 0.0f;

  const int samplesPerPass = m_width*m_height;
  mLightSubPathCount = float(samplesPerPass);

  const int numPass = (m_mask == nullptr) ? 4 : 8;

  for (int pass = 0; pass < numPass; pass++)
  {
    std::cout << "pass " << pass << " begin" << std::endl;

    int mmltFirstBounce = m_pGlobals->varsI[HRT_MMLT_FIRST_BOUNCE];
    if (mmltFirstBounce > 3) mmltFirstBounce = 3;
    if (mmltFirstBounce < 2) mmltFirstBounce = 2;

    for (int d = mmltFirstBounce; d <= m_maxDepth; d++)
    {
      const float selectorInvPdf = float(d + 1);
      double brightness = 0.0f;

      std::cout << "pass " << pass << ", d = " << d << std::endl;

      #pragma omp parallel for reduction(+:brightness)
      for (int sampleId = 0; sampleId < samplesPerPass; sampleId++)
      {
        int xScrNew = 0, yScrNew = 0;

        auto xNew        = InitialSamplePS(d);
        float3 yNewColor = F(xNew, d, (MUTATE_CAMERA | MUTATE_LIGHT), &xScrNew, &yScrNew)*selectorInvPdf;
        const float c    = contribFunc(yNewColor);

        brightness += double(c);
      }

      m_avgBPerBounce[d] += float(brightness);
    }

    std::cout << "pass " << pass << " end" << std::endl;
  }

  m_avgBrightness = 0.0f;
  for (size_t i = 0; i < m_avgBPerBounce.size(); i++)
  {
    m_avgBPerBounce[i] *= (1.0f / float(numPass*samplesPerPass));
    m_avgBrightness += m_avgBPerBounce[i];
    std::cout << "[d = " << i << ", avgB = " << m_avgBPerBounce[i] << ", coeff = " << float(i + 1) << "]" << std::endl;
  }

  std::cout << "[d = a, avgB = " << m_avgBrightness << "]" << std::endl;
  std::cout << "MMLT: finish estimating avg brightness." << std::endl;
  return m_avgBrightness;
}

void IntegratorMMLT::DoPassDirectLight(float4* a_outImage)
{
  const float alpha = 1.0f / float(m_spp + 1);
 
  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {
      randomGen().rptr = nullptr; // force disable taking random numbers from array.

      float3 colors[4];
      for (int i = 0; i < 4; i++) 
      {
        float3 ray_pos, ray_dir;
        std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);
        colors[i] = PathTraceDirectLight(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
      }
      float3 color = 0.25f*(colors[0] + colors[1] + colors[2] + colors[3]);

      a_outImage[y*m_width + x] = a_outImage[y*m_width + x] * (1.0f - alpha) + to_float4(color, 1.0f)*alpha;
    }
  }

}

float IntegratorMMLT::EstimateScaleCoeff() const
{
  const float avgB = EstimateAverageBrightness2(m_hdrData, m_width*m_height);
  return m_avgBrightness / avgB;
}

void IntegratorMMLT::DoPass(std::vector<uint>& a_imageLDR)
{
  //DebugLoadPaths();

  // (0) estimate average brightness
  //
  if (m_firstPass)
  {
    DoPassEstimateAvgBrightness();
    m_firstPass    = false;
  }

  // m_avgBPerBounce.resize(m_maxDepth + 1);
  // for (size_t i = 0; i < m_avgBPerBounce.size(); i++)
  //   m_avgBPerBounce[i] = 1.0f;

  float4* direct   = (float4*)m_direct.data();
  float4* indirect = m_hdrData;

  // (1) compute direct light
  //
  DoPassDirectLight(direct);

  // (2) Run MMLT. 
  //
  constexpr int samplesPerPass = 8;
  #pragma omp parallel num_threads(samplesPerPass)
  DoPassIndirectMLT(indirect);

  // (3) estimate scale coeff
  //
  const float kScaleIndirect = EstimateScaleCoeff();

  //if (m_spp%4 == 0 && m_spp > 0)
  //  DebugSaveBadPaths();

  // (4) get final image
  //
  constexpr float gammaPow = 1.0f / 2.2f;

  const float scaleInv = 1.0f / float(m_spp + 1);

  #pragma omp parallel for
  for (int i = 0; i < int(a_imageLDR.size()); i++)
  {
    float4 color = direct[i] + kScaleIndirect*indirect[i]; 

    color.x = powf(clamp(color.x, 0.0f, 1.0f), gammaPow);
    color.y = powf(clamp(color.y, 0.0f, 1.0f), gammaPow);
    color.z = powf(clamp(color.z, 0.0f, 1.0f), gammaPow);
    color.w = 1.0f;

    a_imageLDR[i] = RealColorToUint32(color);
  }

  RandomizeAllGenerators();

  std::cout << "IntegratorMMLT: mpp  = " << m_spp*samplesPerPass << std::endl;
  m_spp++;

  //float averageBrightness = (kScaleIndirect / m_spp)*EstimateAverageBrightness(m_summColors);
  //std::cout << "avgB = " << averageBrightness << std::endl;
}

void IntegratorMMLT::GetImageHDR(float4* a_imageHDR, int w, int h) const
{
  if (w != m_width || h != m_height)
  {
    std::cout << "IntegratorMMLT::GetImageHDR, bad resolution" << std::endl;
    return;
  }

  const float kScaleIndirect = EstimateScaleCoeff();

  float4* direct = (float4*)m_direct.data();
  float4* indirect = m_hdrData;

  #pragma omp parallel for
  for (int i = 0; i < m_width*m_height; i++)
  {
    float4 color  = direct[i] + kScaleIndirect*indirect[i]; // *scaleInv;
    a_imageHDR[i] = color;
  }
}

PathVertex IntegratorMMLT::LightPath(PerThreadData* a_perThread, int a_lightTraceDepth)
{
  auto& rgen = randomGen();
  rgen.rptr  = &m_pss[ThreadId()][0];                 // lens, light, light path, and finally camera path

  LightGroup2 lightSelector;
  RndLightMMLT(&rgen, rgen.rptr, 
               &lightSelector);

  float lightPickProb = 1.0f;
  const int lightId = SelectRandomLightFwd(lightSelector.group2.z, m_pGlobals,
                                           &lightPickProb);

  const PlainLight* pLight = lightAt(m_pGlobals, lightId);

  LightSampleFwd sample;
  LightSampleForward(pLight, lightSelector.group1, make_float2(lightSelector.group2.x, lightSelector.group2.y), m_pGlobals, m_texStorage, m_pdfStorage,
                     &sample);

  a_perThread->pdfArray[0].pdfFwd = sample.pdfA*lightPickProb;
  a_perThread->pdfArray[0].pdfRev = 1.0f;

  float3 color = (1.0f/lightPickProb)*sample.color/(sample.pdfA*sample.pdfW);

  PathVertex lv;
  InitPathVertex(&lv);

  rgen.rptr = &m_pss[ThreadId()][MMLT_HEAD_TOTAL_SIZE]; // lens, light, light path, and finally camera path

  TraceLightPath(sample.pos, sample.dir, 1, sample.cosTheta, sample.pdfW, color, a_perThread, a_lightTraceDepth, false, 
                 &lv);
  return lv;
}

void IntegratorMMLT::TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float a_prevLightCos, float a_prevPdf,
                                    float3 a_color, PerThreadData* a_perThread, int a_lightTraceDepth, bool a_wasSpecular,
                                    PathVertex* a_pOutLightVertex)
{
  if (a_currDepth > a_lightTraceDepth)
    return;

  auto hit = rayTrace(ray_pos, ray_dir);
  if (!HitSome(hit))
    return;

  const SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);
  /////////////////////////////////////////////////////////////////////
  if(a_currDepth == 1)
    m_debugRaysPos[ThreadId()][0]         = to_float4(ray_pos, 1.0f);
  m_debugRaysPos[ThreadId()][a_currDepth] = to_float4(surfElem.pos, 1.0f);
  /////////////////////////////////////////////////////////////////////

  const float cosPrev = fabs(a_prevLightCos);
  const float cosCurr = fabs(-dot(ray_dir, surfElem.normal));
  const float dist    = length(surfElem.pos - ray_pos);

  // eval forward pdf
  //
  const float GTermPrev = (a_prevLightCos*cosCurr / fmax(dist*dist, DEPSILON2));
  const float prevPdfWP = a_prevPdf / fmax(a_prevLightCos, DEPSILON);
  
  if (!a_wasSpecular)
    a_perThread->pdfArray[a_currDepth].pdfFwd = prevPdfWP*GTermPrev;
  else
    a_perThread->pdfArray[a_currDepth].pdfFwd = -1.0f*GTermPrev;

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample      matSam       = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem, packBounceNum(0, a_currDepth - 1), true, float3(0, 0, 0), true));

  // calc new ray
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  const float cosNext   = fabs(+dot(nextRay_dir, surfElem.normal));

  if (a_currDepth == a_lightTraceDepth)
  {
    a_pOutLightVertex->hit       = surfElem;
    a_pOutLightVertex->ray_dir   = ray_dir;
    a_pOutLightVertex->accColor  = a_color;
    a_pOutLightVertex->valid     = true;
    a_pOutLightVertex->lastGTerm = GTermPrev;
    return;
  }

  // If we sampled specular event, then the reverse probability
  // cannot be evaluated, but we know it is exactly the same as
  // forward probability, so just set it. If non-specular event happened,
  // we evaluate the pdf
  //
  if (!isPureSpecular(matSam))
  {
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = (-1.0f)*ray_dir;
    sc.v  = (-1.0f)*nextRay_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const float pdfW         = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), m_pGlobals, m_texStorage, m_texStorageAux, &m_ptlDummy).pdfFwd;
    const float prevPdfRevWP = pdfW / fmax(cosCurr, DEPSILON);
    a_perThread->pdfArray[a_currDepth].pdfRev = prevPdfRevWP*GTermPrev;
  }
  else
  {
    a_perThread->pdfArray[a_currDepth].pdfRev = -1.0f*GTermPrev;
  }

  a_color *= matSam.color*cosNext*(1.0f / fmax(matSam.pdf, DEPSILON2));

  TraceLightPath(nextRay_pos, nextRay_dir, a_currDepth + 1, cosNext, matSam.pdf, 
                 a_color, a_perThread, a_lightTraceDepth, isPureSpecular(matSam),
                 a_pOutLightVertex);
}

PathVertex IntegratorMMLT::CameraPath(float3 ray_pos, float3 ray_dir, MisData a_misPrev, int a_currDepth, uint flags,
                                      PerThreadData* a_perThread, int a_targetDepth, bool a_haveToHitLightSource, int a_fullPathDepth)

{
  const int prevVertexId = a_fullPathDepth - a_currDepth + 1; 

  if (a_currDepth > a_targetDepth)
  {
    PathVertex resVertex;
    resVertex.valid    = false;
    resVertex.accColor = float3(0, 0, 0);
    return resVertex;
  }

  const Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
  {
    PathVertex resVertex;
    resVertex.valid    = false;
    resVertex.accColor = float3(0, 0, 0);
    return resVertex;
  }

  const SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  /////////////////////////////////////////////////////////////////////
  m_debugRaysPos[ThreadId()][prevVertexId-1] = to_float4(surfElem.pos, -1.0f);
  /////////////////////////////////////////////////////////////////////

  // (1)
  //
  const float cosHere = fabs(dot(ray_dir, surfElem.normal));
  const float cosPrev = fabs(a_misPrev.cosThetaPrev); // fabs(dot(ray_dir, a_prevNormal));
  float GTerm = 1.0f;

  if (a_currDepth == 1)
  {
    float3 camDirDummy; float zDepthDummy;
    const float imageToSurfaceFactor = CameraImageToSurfaceFactor(surfElem.pos, surfElem.normal, m_pGlobals,
                                                                  &camDirDummy, &zDepthDummy);
    const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
    a_perThread->pdfArray[a_fullPathDepth].pdfRev = cameraPdfA;
    a_perThread->pdfArray[a_fullPathDepth].pdfFwd = 1.0f;
  }
  else
  {
    const float dist = length(ray_pos - surfElem.pos);
    
    GTerm = cosHere*cosPrev / fmax(dist*dist, DEPSILON2);
  }

  // (2)
  //
  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, a_misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-6f)
  {
    PathVertex resVertex;
    if (a_currDepth == a_targetDepth && a_haveToHitLightSource)
    {
      const int instId         = fetchInstId(hit);
      const int lightOffset    = m_geom.instLightInstId[instId];
      const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
      
      const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, m_pGlobals, m_texStorage, m_pdfStorage);
      const float pdfLightWP    = lPdfFwd.pdfW / fmax(cosHere, DEPSILON);
      const float pdfMatRevWP   = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);

      a_perThread->pdfArray[0].pdfFwd = lPdfFwd.pdfA / float(m_pGlobals->lightsNum);
      a_perThread->pdfArray[0].pdfRev = 1.0f;

      a_perThread->pdfArray[1].pdfFwd = pdfLightWP*GTerm;
      a_perThread->pdfArray[1].pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfMatRevWP*GTerm;

      resVertex.hit      = surfElem;
      resVertex.ray_dir  = ray_dir;
      resVertex.accColor = emission;
      resVertex.valid    = true;
      return resVertex;
    }
    else // this branch could brobably change in future, for simple emissive materials
    {
      resVertex.accColor = float3(0, 0, 0);
      resVertex.valid    = false;
      return resVertex;
    }
   
  }
  else if (a_currDepth == a_targetDepth && !a_haveToHitLightSource) // #NOTE: what if a_targetDepth == 1 ?
  {
    PathVertex resVertex;
    resVertex.hit          = surfElem;
    resVertex.ray_dir      = ray_dir;
    resVertex.valid        = true;
    resVertex.accColor     = float3(1, 1, 1);
    resVertex.wasSpecOnly  = m_splitDLByGrammar ? flagsHaveOnlySpecular(flags) : false;

    if (a_targetDepth != 1)
    {
      const float lastPdfWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON); // we store them to calculate fwd and rev pdf later when we connect end points
      resVertex.lastGTerm   = GTerm;                                            // because right now we can not do this until we don't know the light vertex

      //a_perThread->pdfArray[prevVertexId].pdfFwd = ... // do this later, inside ConnectShadow or ConnectEndPoints
      a_perThread->pdfArray[prevVertexId].pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : GTerm*lastPdfWP;
    }
    else
      resVertex.lastGTerm = 1.0f;
    
    return resVertex;
  }
  
  // (3) eval reverse and forward pdfs
  //
  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem, packBounceNum(0, a_currDepth - 1), false, float3(0, 0, 0), true));
  const float3 bxdfVal   = matSam.color; // *(1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosNext    = fabs(dot(matSam.direction, surfElem.normal));

  if (a_currDepth == 1)
  {
    if (isPureSpecular(matSam))  //  ow ... but if we met specular reflection when tracing from camera, we must put 0 because this path cannot be sample by light strategy at all.
      a_perThread->pdfArray[a_fullPathDepth].pdfFwd = 0.0f;
  }
  else
  {
    if (!isPureSpecular(matSam))
    {
      ShadeContext sc;
      sc.wp = surfElem.pos;
      sc.l  = (-1.0f)*ray_dir;  // fliped; if compare to normal PT
      sc.v  = matSam.direction; // fliped; if compare to normal PT
      sc.n  = surfElem.normal;
      sc.fn = surfElem.flatNormal;
      sc.tg = surfElem.tangent;
      sc.bn = surfElem.biTangent;
      sc.tc = surfElem.texCoord;

      const float pdfFwdW  = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), // global data -->
                                          m_pGlobals, m_texStorage, m_texStorageAux, &m_ptlDummy).pdfFwd;
      const float pdfFwdWP = pdfFwdW / fmax(cosHere, DEPSILON);

      a_perThread->pdfArray[prevVertexId].pdfFwd = pdfFwdWP*GTerm;
    }
    else
      a_perThread->pdfArray[prevVertexId].pdfFwd = -1.0f*GTerm;

    const float pdfCamPrevWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);
    a_perThread->pdfArray[prevVertexId].pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfCamPrevWP*GTerm;
  }
  
 
  // (4) proceed to next bounce
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  MisData thisBounce       = makeInitialMisData();
  thisBounce.isSpecular    = isPureSpecular(matSam);
  thisBounce.matSamplePdf  = matSam.pdf;
  thisBounce.cosThetaPrev  = dot(nextRay_dir, surfElem.normal);

  const bool stopDL        = m_splitDLByGrammar ? flagsHaveOnlySpecular(flags) : false;

  PathVertex nextVertex  = CameraPath(nextRay_pos, nextRay_dir, thisBounce, a_currDepth + 1, flagsNextBounceLite(flags, matSam, m_pGlobals),
                                      a_perThread, a_targetDepth, a_haveToHitLightSource, a_fullPathDepth);

  nextVertex.accColor *= (bxdfVal*cosNext / fmax(matSam.pdf, DEPSILON2));

  if (stopDL && a_haveToHitLightSource && a_currDepth + 1 == a_targetDepth) // exclude direct light
    nextVertex.accColor = float3(0, 0, 0);

  return nextVertex;
}


float3 IntegratorMMLT::ConnectEye(const PathVertex& a_lv, int a_ltDepth,
                                   PerThreadData* a_perThread, int* pX, int* pY)
{

  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(a_lv.hit.pos, a_lv.hit.normal, m_pGlobals,
                                                                &camDir, &zDepth);

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, a_lv.hit.matId);
  float signOfNormal = 1.0f;
  if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, a_lv.hit.normal) < -0.01f)
    signOfNormal = -1.0f;

  auto hit = rayTrace(a_lv.hit.pos + epsilonOfPos(a_lv.hit.pos)*signOfNormal*a_lv.hit.normal, camDir);

  auto* v0 = a_perThread->pdfArray.data() + a_ltDepth + 0;
  auto* v1 = a_perThread->pdfArray.data() + a_ltDepth + 1;

  if (imageToSurfaceFactor <= 0.0f || (HitSome(hit) && hit.t <= zDepth))
  {
    (*pX)         = -1;
    (*pY)         = -1;
    return make_float3(0, 0, 0);
  }

  return ConnectEyeP(&a_lv, mLightSubPathCount, camDir, imageToSurfaceFactor,
                     m_pGlobals, m_matStorage, m_texStorage, m_texStorageAux, &m_ptlDummy,
                     v0, v1, pX, pY);
}


float3 IntegratorMMLT::ConnectShadow(const PathVertex& a_cv, PerThreadData* a_perThread, const int a_camDepth)
{
  float3 explicitColor(0, 0, 0);

   const SurfaceHit& surfElem = a_cv.hit;

   auto& gen = randomGen();
   gen.rptr  = &m_pss[ThreadId()][0]; // lens, light, light path, and finally camera path
   
   LightGroup2 lightSelector;
   RndLightMMLT(&gen, gen.rptr, 
                &lightSelector);
   
   float lightPickProb = 1.0f;
   int lightOffset = SelectRandomLightRev(lightSelector.group2.z, surfElem.pos, m_pGlobals,
                                          &lightPickProb);

   if (lightOffset >= 0)
   {
     __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
   
     ShadowSample explicitSam;
     LightSampleRev(pLight, to_float3(lightSelector.group1), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
                    &explicitSam);
   
     /////////////////////////////////////////////////////////////////////
     m_debugRaysPos[ThreadId()][0] = to_float4(explicitSam.pos, 1.0f);
     /////////////////////////////////////////////////////////////////////
   
     const float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos); // explicitSam.direction;
     const float3 shadowRayPos = OffsRayPos(surfElem.pos, surfElem.normal, shadowRayDir); 
     const float3 shadow       = shadowTrace(shadowRayPos, shadowRayDir, explicitSam.maxDist*0.9995f);
   
     if (dot(shadow, shadow) > 1e-12f)
     {
       auto* v0 = &a_perThread->pdfArray[0];
       auto* v1 = &a_perThread->pdfArray[1];
       auto* v2 = &a_perThread->pdfArray[2];

       explicitColor = shadow*ConnectShadowP(&a_cv, a_camDepth, pLight, explicitSam, lightPickProb,
                                             m_pGlobals, m_matStorage, m_texStorage, m_texStorageAux, m_pdfStorage, &m_ptlDummy,
                                             v0, v1, v2);
     }
   }
  

  return explicitColor;
}

float3 IntegratorMMLT::ConnectEndPoints(const PathVertex& a_lv, const PathVertex& a_cv, const int a_spit, const int a_depth,
                                        PerThreadData* a_perThread)
{
  if (!a_lv.valid || !a_cv.valid)
    return float3(0, 0, 0);

  const float3 diff = a_cv.hit.pos - a_lv.hit.pos;
  const float dist2 = fmax(dot(diff, diff), DEPSILON2);
  const float  dist = sqrtf(dist2);
  const float3 lToC = diff / dist; // normalize(a_cv.hit.pos - a_lv.hit.pos)

  const float cosAtLightVertex  = +dot(a_lv.hit.normal, lToC);
  const float cosAtCameraVertex = -dot(a_cv.hit.normal, lToC);

  const float GTerm = cosAtLightVertex*cosAtCameraVertex / dist2;

  if (GTerm < 0.0f)
    return float3(0, 0, 0);

  const float3 shadowRayDir = lToC; // explicitSam.direction;
  const float3 shadowRayPos = OffsRayPos(a_lv.hit.pos, a_lv.hit.normal, shadowRayDir);
  const float3 shadow       = shadowTrace(shadowRayPos, shadowRayDir, dist*0.9995f);

  auto* vSplitBefore = &a_perThread->pdfArray[a_spit-1];
  auto* vSplit       = &a_perThread->pdfArray[a_spit+0];
  auto* vSplitAfter  = &a_perThread->pdfArray[a_spit+1];

  if (dot(shadow, shadow) < 1e-12f)
    return float3(0, 0, 0);
  else
    return shadow*ConnectEndPointsP(&a_lv, &a_cv, a_depth,
                                    m_pGlobals, m_matStorage, m_texStorage, m_texStorageAux, &m_ptlDummy,
                                    vSplitBefore, vSplit, vSplitAfter);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// PT for direct light
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// PT for direct light

float3 IntegratorMMLT::PathTraceDirectLight(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
  if (a_currDepth >= m_maxDepth)
    return float3(0, 0, 0);

  Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
    return float3(0, 0, 0);

  SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  const int mmltFirstBounce = m_pGlobals->varsI[HRT_MMLT_FIRST_BOUNCE];

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-3f)
  {
    if (m_computeIndirectMLT && a_currDepth <= 1)
      return float3(0, 0, 0);

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
  else if ((!m_splitDLByGrammar && a_currDepth >= 1) || (mmltFirstBounce == 2))
    return float3(0, 0, 0);

  float3 explicitColor(0, 0, 0);

  auto& gen = randomGen();
  float lightPickProb = 1.0f;
  int lightOffset = SelectRandomLightRev(rndFloat1_Pseudo(&gen), surfElem.pos, m_pGlobals,
                                         &lightPickProb);

  if ((!m_computeIndirectMLT || a_currDepth > 0) && lightOffset >= 0) // if need to sample direct light ?
  {
    const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    ShadowSample explicitSam;
    LightSampleRev(pLight, rndFloat3(&gen), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   &explicitSam);

    float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos);
    float3 shadowRayPos = OffsRayPos(surfElem.pos, surfElem.normal, shadowRayDir);

    const float3 shadow = shadowTrace(shadowRayPos, shadowRayDir, explicitSam.maxDist*0.9995f);

    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = shadowRayDir;
    sc.v  = (-1.0f)*ray_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const auto evalData      = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), /* global data --> */ m_pGlobals, m_texStorage, m_texStorageAux, &m_ptlDummy);

    const float cosThetaOut1 = fmax(+dot(shadowRayDir, surfElem.normal), 0.0f);
    const float cosThetaOut2 = fmax(-dot(shadowRayDir, surfElem.normal), 0.0f);
    const float3 bxdfVal     = (evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2);

    const float lgtPdf       = explicitSam.pdf*lightPickProb;

    float misWeight = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0f;

    explicitColor = (1.0f / lightPickProb)*(explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON)))*bxdfVal*misWeight*shadow; // #TODO: clamp brdfVal? test it !
  }

  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));
  const float3 bxdfVal = matSam.color * (1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosTheta = fabs(dot(matSam.direction, surfElem.normal));

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  MisData currMis            = makeInitialMisData();
  currMis.isSpecular         = isPureSpecular(matSam);
  currMis.matSamplePdf       = matSam.pdf;

  const bool proceedTrace = m_splitDLByGrammar ? flagsHaveOnlySpecular(flags) : true;
  if (!proceedTrace)
    return float3(0, 0, 0);
  else
    return explicitColor + cosTheta*bxdfVal*PathTraceDirectLight(nextRay_pos, nextRay_dir, currMis, a_currDepth + 1, flags);  // --*(1.0 / (1.0 - pabsorb));
}
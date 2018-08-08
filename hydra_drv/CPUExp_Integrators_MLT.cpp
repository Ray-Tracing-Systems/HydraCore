#include <omp.h>
#include "CPUExp_Integrators.h"

namespace mltsandbox
{
  // float4 rndLensGroup(MLTRandomGen* a_gen);
  // float4 rndLightGroup(MLTRandomGen* a_gen, int bounceId);
  // float3 rndMatGroup(MLTRandomGen* a_gen, int bounceId);
  // float  rndMatLayer(MLTRandomGen* a_gen, int bounceId, int layerDepth);

  //rndLensGroup --> rndLightGroup(a_gen,0);
  //rndMatGroup

  // loop {
  //   rndLightGroup
  //   ... rndMatLayer
  //   ... rndMatLayer
  //   ... rndMatLayer
  //   ... rndMatLayer
  //   rndMatGroup
  // }

};

const int MLT_QMC_NUMBERS_CPU = 256;

void IntegratorPSSMLT::initRands()
{
  m_perThread.resize(INTEGRATOR_MAX_THREADS_NUM);

  for (size_t i = 0; i < m_perThread.size(); i++)
  {
    m_perThread[i].gen  = RandomGenInit(GetTickCount());
    m_perThread[i].gen2 = RandomGenInit(GetTickCount());
  }
}

IntegratorPSSMLT::PSSampleV IntegratorPSSMLT::initialSample_PS(int a_mutationsNum)
{
  // do burning-in
  //
  int burnIters = a_mutationsNum / 1024;
  if (burnIters < 100)
    burnIters = 100;
  if (burnIters > 1024)
    burnIters = 1024;

  auto& gen = randomGen();

  std::vector<float>      contribArray(burnIters);
  std::vector<RandomGen>  genState(burnIters);

  for (size_t i = 0; i < contribArray.size(); i++)
  {
    genState[i] = gen;

    PSSampleV x(MLT_QMC_NUMBERS_CPU);
    for (size_t j = 0; j < x.size(); j++)
      x[j] = rndFloat1_Pseudo(&gen);

    contribArray[i] = contribFunc(F(x));
  }

  // compute prefix summ
  // 
  for (size_t i = 1; i < contribArray.size(); i++)
    contribArray[i] = contribArray[i] + contribArray[i - 1];

  // now select sample from array ~ to it's contib
  //
  float choice = rndFloat1_Pseudo(&gen)*contribArray[contribArray.size() - 1];
  size_t foundIndex = 0;
  for (size_t i = 0; i < contribArray.size(); i++)
  {
    if (contribArray[i] <= choice && choice <= contribArray[i + 1])
    {
      foundIndex = i;
      break;
    }
  }
  

  //auto& gen2 = randomGen();
  RandomGen gen2 = genState[foundIndex];

  PSSampleV res(MLT_QMC_NUMBERS_CPU);
  for (size_t i = 0; i < res.size(); i++)
    res[i] = rndFloat1_Pseudo(&gen2);

  return res;
}


float3 IntegratorPSSMLT::F(const PSSampleV& x_ps)
{
  auto& gen = randomGen();

  gen.rptr       = &(x_ps[0]);
  gen.maxNumbers = MLT_QMC_NUMBERS_CPU;

  const float4 lens  = rndLens(&gen, gen.rptr, float2(1, 1), 0, 0, m_pGlobals->rmQMC);
  const float xPosPs = lens.x;
  const float yPosPs = lens.y;
  const float x      = m_width*xPosPs;
  const float y      = m_height*yPosPs;

  float3 ray_pos, ray_dir;
  std::tie(ray_pos, ray_dir) = makeEyeRay2(x, y);

  float3 color = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

  gen.rptr = nullptr;
  return color;
}


#include <time.h>


const float MLT_PLARGE = 0.25f;

void MakeProposalAsInGPUVer(RandomGen* gen, float* yVecOut, const float* xVecIn, bool forceLargeStep, const EngineGlobals* a_globals)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  float rlarge = 0.0f;
  if (!forceLargeStep)
    rlarge = rndFloat1_Pseudo(gen);

  //const float screenScaleX = a_globals->varsF[HRT_MLT_SCREEN_SCALE_X];
  //const float screenScaleY = a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y];
  const float2 lensMutateCoeff = make_float2(1.0f, 1.0f);

  if (rlarge <= MLT_PLARGE)
    gen->lazy = MUTATE_LAZY_LARGE;
  else
    gen->lazy = MUTATE_LAZY_YES;

  for (int bounceId = 0; bounceId < MLT_MAX_BOUNCE; bounceId++)
  {
    const int lightOffset = rndLightOffset(bounceId);
    const int matOffset   = rndMatOffset(bounceId);
    const int matLOffset  = rndMatLOffset(bounceId);

    float4 l_i = make_float4(0, 0, 0, 0);

    if (bounceId == 0)
      l_i = rndLens(gen, xVecIn, lensMutateCoeff, 0, 0, a_globals->rmQMC);
    else
      l_i = rndLight(gen, xVecIn, bounceId);

    yVecOut[lightOffset + 0] = l_i.x;
    yVecOut[lightOffset + 1] = l_i.y;
    yVecOut[lightOffset + 2] = l_i.z;
    yVecOut[lightOffset + 3] = l_i.w;

    for (int i = 0; i < MLT_FLOATS_PER_MLAYER; i++)
      yVecOut[matLOffset + i] = rndMatLayer(gen, xVecIn, bounceId, i,
                                            a_globals->rmQMC, 0, 0);

    const float3 m_i = rndMat(gen, xVecIn, bounceId,
                              a_globals->rmQMC, 0, nullptr);

    yVecOut[matOffset + 0] = m_i.x;
    yVecOut[matOffset + 1] = m_i.y;
    yVecOut[matOffset + 2] = m_i.z;
  }

  gen->lazy = MUTATE_LAZY_NO;
}

IntegratorPSSMLT::PSSampleV IntegratorPSSMLT::mutatePrimarySpace(const PSSampleV& a_vec, bool* pIsLargeStep)
{
  auto& gen = randomGen();
  PSSampleV v2(a_vec.size());
  MakeProposalAsInGPUVer(&gen, &v2[0], &a_vec[0], false, m_pGlobals);
  return v2;
}

void IntegratorPSSMLT::DoPassDL(HDRImage4f& a_outImage)
{
  int oldTraceDepth = m_maxDepth;
  m_maxDepth = 1;

  if (m_width*m_height != a_outImage.width()*a_outImage.height())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  float4* a_outHDR = (float4*)a_outImage.data();

  // Update HDR image
  //
  const float alpha = 1.0f / float(m_spp + 1);

  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {
      auto& gen = randomGen();
      gen.rptr = nullptr;

      float3 ray_pos, ray_dir;
      std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);

      const float3 color    = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
      const float  colorSqr = colorSquareMax3(color);

      a_outHDR[y*m_width + x] = a_outHDR[y*m_width + x] * (1.0f - alpha) + to_float4(color*alpha, colorSqr);
    }

  }

  std::cout << "IntegratorPSSMLT: mpp = " << 8*m_spp << std::endl;

  m_maxDepth = oldTraceDepth;
}


void IntegratorPSSMLT::DoPassIndirectMLT(int mutationsNum)
{
  float4* hystColored = (float4*)m_hystColored.data();
  int imageSize = m_hystColored.width()*m_hystColored.height();

  m_computeIndirectMLT = true;

  auto& gen     = randomGen2();

  auto   x      = initialSample_PS(mutationsNum);
  float3 yColor = F(x);
  float  y      = contribFunc(yColor);

  int accept = 0;

  for (int mId = 0; mId < mutationsNum; mId++)
  {
    auto xOld = x;
    auto xNew = mutatePrimarySpace(x);

    float  yOld      = y;
    float3 yOldColor = yColor;

    float3 yNewColor = F(xNew);
    float  yNew      = contribFunc(yNewColor);

    float a = (yOld == 0.0f) ? 1.0f : fminf(1.0f, yNew / yOld);

    float p = rndFloat1_Pseudo(&gen);

    if (p <= a) // accept //
    {
      x = xNew;
      y = yNew;
      yColor = yNewColor;
      accept++;
    }
    else        // reject
    {
      //x = x;
      //y = y;
      //yColor = yColor;
    }


    float xPosPs0 = xOld[0];
    float yPosPs0 = xOld[1];

    float xPosPs1 = xNew[0];
    float yPosPs1 = xNew[1];

    float fw = (float)m_width;
    float fh = (float)m_height;

    int xScreen0 = (int)(fw*xPosPs0);
    int yScreen0 = (int)(fh*yPosPs0);

    int xScreen1 = (int)(fw*xPosPs1);
    int yScreen1 = (int)(fh*yPosPs1);

    int index1 = yScreen0*m_width + xScreen0;
    int index2 = yScreen1*m_width + xScreen1;

    #pragma omp critical
    {
      if (index1 < imageSize)
      {
        //m_hyst[index1]        += yOld*(1.0f - a);
        float3 color    = yOldColor*(1.0f / fmaxf(yOld, 1e-6f))*(1.0f - a);
        float  colorSqr = colorSquareMax3(color)*(1.0f - a);

        hystColored[index1] += to_float4(color, colorSqr);
      }

      if (index2 < imageSize)
      {
        //m_hyst[index2]        += yNew*a;
        float3 color    = yNewColor*(1.0f / fmaxf(yNew, 1e-6f))*a;
        float  colorSqr = colorSquareMax3(color)*a;

        hystColored[index2] += to_float4(color, colorSqr);
      }

    }
  }

  if (omp_get_thread_num() == 0)
  {
    float acceptanceRate = float(accept) / float(mutationsNum);
    auto oldPrecition = std::cout.precision(3);
    std::cout << "[MLT]: acceptanceRate = " << 100.0f*acceptanceRate << "%" << std::endl;
    std::cout.precision(oldPrecition);
  }

  m_computeIndirectMLT = false;
}

void IntegratorPSSMLT::DoPassIndirectOrdinaryMC(int a_spp, HDRImage4f& a_outImage)
{
  float4* outHDR = (float4*)a_outImage.data();

  m_computeIndirectMLT = true;

  // Update HDR image
  //
  for (int spp = m_ordinarySpp; spp < a_spp + m_ordinarySpp; spp++)
  {
    const float alpha = 1.0f / float(spp + 1);

    #pragma omp parallel for
    for (int y = 0; y < m_height; y++)
    {
      for (int x = 0; x < m_width; x++)
      {
        auto& gen = randomGen();
        gen.rptr = nullptr;

        float3 ray_pos, ray_dir;
        std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);

        float3 color    = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
        float  colorSqr = colorSquareMax3(color);

        outHDR[y*m_width + x] = outHDR[y*m_width + x] * (1.0f - alpha) + to_float4(color*alpha, colorSqr);
      }

    }
  }

  m_computeIndirectMLT = false;
  m_ordinarySpp        += a_spp;
}


float3 EstimateAverageBrightnessRGB(const HDRImage4f& a_color)
{
  const float4* in_color = (float4*)a_color.data();
  int imageSize = a_color.width()*a_color.height();

  float4 summ(0,0,0,0);

  for (int i = 0; i < imageSize; i++)
  {
    float4 color = in_color[i];
    summ += color;  
  }

  return to_float3(summ) / float(imageSize);
}

float EstimateAverageBrightness(const HDRImage4f& a_color)
{
  const float3 color = EstimateAverageBrightnessRGB(a_color);
  const float val = contribFunc(color);
  if (val < 1e-20f)
    return 1e-20f;
  else
    return val;
}

float3 EstimateAverageBrightnessRGB(const std::vector<float4>& a_color)
{
  const float4* in_color = (float4*)a_color.data();
  int imageSize = int(a_color.size());

  float4 summ(0, 0, 0, 0);
  for (int i = 0; i < imageSize; i++)
  {
    float4 color = a_color[i];
    summ += color;
  }
  return to_float3(summ) / float(imageSize);
}

float EstimateAverageBrightness(const std::vector<float4>& a_color)
{
  const float3 color = EstimateAverageBrightnessRGB(a_color);
  const float val    = contribFunc(color);
  if (val < 1e-20f)
    return 1e-20f;
  else
    return val;
}


void IntegratorPSSMLT::DoPass(std::vector<uint>& a_imageLDR)
{
  if (m_width*m_height != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  if (m_firstPass)
  {
    HDRImage4f indirectLightOriginMC(m_width, m_height);
    std::cout << "[MLT]: estimating average luminance ... " << std::endl;
    DoPassIndirectOrdinaryMC(4, indirectLightOriginMC);
    m_averageBrightness = EstimateAverageBrightness(indirectLightOriginMC);
    m_firstPass = false;
  }

  // compute direct light with ordinary monte carlo
  //
  DoPassDL(m_directLight);

  // compute indirect light with mlt
  //
  #pragma omp parallel num_threads(8)
  DoPassIndirectMLT(m_width*m_height);

  m_spp++;

  float averageBrightness = EstimateAverageBrightness(m_hystColored);

  // get final image
  //
  float normConst = m_averageBrightness / averageBrightness;

  float4* directLight = (float4*)m_directLight.data();
  float4* hystColored = (float4*)m_hystColored.data();


  float gammaPow = 1.0f / 2.2f;  // gamma correction

  for (size_t i = 0; i < a_imageLDR.size(); i++)
  {
    //float cval = sqrtf(fmaxf(m_hyst[i] * normConst, 0.0f));
    //m_summColors[i] = float4(cval, cval, cval, 0.0f);

    float4 direct   = directLight[i];
    float4 indirect = hystColored[i] * normConst;

    float4 color4   = direct + indirect; // direct + indirect
    m_summColors[i] = color4;

    // get HDR to LDR
    //
    float4 color = ToneMapping4(color4);

    color.x = powf(color.x, gammaPow);
    color.y = powf(color.y, gammaPow);
    color.z = powf(color.z, gammaPow);

    a_imageLDR[i] = RealColorToUint32(color);
  }

}




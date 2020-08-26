#include "CPUExp_Integrators.h"
//#include <time.h>
#include <algorithm>  // sort

//////////////////////////////////////////////////////////////////////////////////////////

float Distance(const int x, const int y)
{
  return (float)sqrt(x * x + y * y);
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
  return clamp(a_nextProb / fmax(a_currProb, 1e-6F), 0.0F, 1.0F);
}

float r2Error(const float a_val, const float a_mean)    // coefficient of determination     
{
  const float diffCurrLum  = a_val - a_mean;
  return clamp((diffCurrLum * diffCurrLum) / fmax(a_mean * a_mean, 1e-8F), 0.0F, 1.0F);
}


void IntegratorMISPTLoop2Adapt::DoPass(std::vector<uint>& a_imageLDR)
{  
  // Update HDR image
  if (m_imgSize != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  const bool  multiThreads = true;

  const float imgRadius    = Distance(m_width, m_height) * 0.5F;  
  const int   maxThreads    = omp_get_max_threads();
  const int   numThreads    = multiThreads ? omp_get_max_threads() : 1;
  const int   maxSamples    = multiThreads ? m_imgSize / numThreads: m_imgSize / maxThreads;

#ifdef NDEBUG
#pragma omp parallel num_threads(numThreads)
#endif
  {    
    auto&        gen  = randomGen();
    float        step = imgRadius * 0.25F;
    int2         pos  = GetNewLocalPos(int2(0, 0), step, gen, false, true);

    for (int sample = 0; sample < maxSamples; ++sample)
    {
      //NewPositionWithAdapt(startX, startY, imgRadius);
      NewPositionWithMarkovChain(pos, step, gen, imgRadius);
      //ScreenSpaceMLT(pos, gen, imgRadius, sample);
    }
  }

  RandomizeAllGenerators();

    
  // get HDR to LDR and scale

  GetImageToLDR(a_imageLDR, true);
  
  //const float gammaPow = 1.0F / m_pGlobals->varsF[HRT_IMAGE_GAMMA];  // gamma correction
  
//  float summ = 0.0F;
//  
//  for (int i = 0; i < a_imageLDR.size(); ++i)
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

  uint meanSpp = 0;
  
  for (int i = 0; i < a_imageLDR.size(); i++)
  {
    meanSpp += m_samplePerPix[i];
  
    // for ScreenSpaceMLT()

    //const float spp = fmax((float)(m_samplePerPix[i]), 1.0F);

    //float4 color    = m_summColors[i];    
    //color.x /= spp;
    //color.y /= spp;
    //color.z /= spp;

    //color          = ToneMapping4Compress(color);
    //color.x        = powf(color.x, gammaPow);
    //color.y        = powf(color.y, gammaPow);
    //color.z        = powf(color.z, gammaPow);

    //a_imageLDR[i] = RealColorToUint32(color);
  }


  //for (y = 0; y < m_height; y++)
  //{
  //  for (x = 0; x < m_width; x++)
  //  {
  //    //if (!m_pixFinish[i])
  //    // {
  //    //   float4 color(0.5F, 0.25F, 0.25F, 0.0F);
  //    //   a_imageLDR[i] = RealColorToUint32(color);
  //    // }   
  //    const int index   = y * m_width + x;
  //    const int index2  = GetIndexSmallImage(x, y);
  //    
  //    const float color = m_summColorsArea[index2] / (1.0F + m_summColorsArea[index2]);

  //    a_imageLDR[index] = RealColorToUint32(float4(color, color, color, 0));
  //  }
  //}

  //m_progress = 0;
  //for (int i = 0; i < m_imgSize; i++)
  //  if (m_pixFinish[i])
  //    m_progress += 1;

  //m_progress = m_progress / (float)maxSample * 100.0F;
  meanSpp /= m_imgSize;
  m_spp = meanSpp;

  std::cout << "[" << this->Name() << "]: mean spp = " << meanSpp    << std::endl;
  //std::cout << "[" << this->Name() << "]: progress = " << m_progress << std::endl;

}


void IntegratorMISPTLoop2Adapt::GetStatisticsLocalWin(const int2 a_pos, const int a_sizeLocalWindow, float& a_summ, float& a_max, 
  float& a_mediana, float& a_mean, const bool a_useWeightSample) 
{
  int summSpp = 0;
  
  if (a_useWeightSample)
  {
    summSpp = GetSummSppLocalWin(a_pos, a_sizeLocalWindow);
    if (summSpp < 1)
      return;
  }

  float     summ      = 0.0F;
  const int sizeLine  = (a_sizeLocalWindow * 2 + 1);
  const int sizeArray = sizeLine * sizeLine;
  std::vector<float> pixels(sizeArray);

  for (int offsetY = -a_sizeLocalWindow; offsetY <= a_sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -a_sizeLocalWindow; offsetX <= a_sizeLocalWindow; offsetX++)
    {
      int NewX        = a_pos.x + offsetX;
      int NewY        = a_pos.y + offsetY;

      NewX            = clamp(NewX, 0, m_width  - 1);
      NewY            = clamp(NewY, 0, m_height - 1);

      const int index = NewY * m_width + NewX;

      float weight    = 1.0F;
      if (a_useWeightSample)
        weight        = (float)m_samplePerPix[index] / (float)summSpp;

      const float val = Luminance(m_summColors[index]) * weight;

      pixels.push_back(val);
      summ += val;
    }
  }

  std::sort(pixels.begin(), pixels.end());

  const int lastIndex = sizeArray - 1;
  a_summ              = summ;
  a_max               = pixels[lastIndex];
  a_mediana           = pixels[sizeArray / 2];
  a_mean              = summ / (float)sizeArray;
}

int IntegratorMISPTLoop2Adapt::GetSummSppLocalWin(const int2 a_pos, const int a_sizeLocalWindow)
{
  int summ = 0;

  for (int offsetY = -a_sizeLocalWindow; offsetY <= a_sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -a_sizeLocalWindow; offsetX <= a_sizeLocalWindow; offsetX++)
    {
      int NewX = a_pos.x + offsetX;
      int NewY = a_pos.y + offsetY;

      NewX = clamp(NewX, 0, m_width  - 1);
      NewY = clamp(NewY, 0, m_height - 1);

      const int index = NewY * m_width + NewX;
            
      summ += m_samplePerPix[index];
    }
  }

  return summ;
}

float IntegratorMISPTLoop2Adapt::MathExp(const int2 a_pos, const int a_sizeLocalWindow, const bool a_square)
{
  const int summSpp = GetSummSppLocalWin(a_pos, a_sizeLocalWindow);

  if (summSpp < 1)
    return 0.0F;

  float mathExp  = 0.0F;  

  for (int offsetY = -a_sizeLocalWindow; offsetY <= a_sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -a_sizeLocalWindow; offsetX <= a_sizeLocalWindow; offsetX++)
    {
      int NewX = a_pos.x + offsetX;
      int NewY = a_pos.y + offsetY;

      NewX = clamp(NewX, 0, m_width - 1);
      NewY = clamp(NewY, 0, m_height - 1);

      const int index = NewY * m_width + NewX;

      const float val    = Luminance(m_summColors[index]);
      const float weight = (float)m_samplePerPix[index] / (float)summSpp;      

      if (a_square)      
        mathExp += (val * val * weight);
      else
        mathExp += (val * weight);
    }
  }

  return mathExp;
}


float IntegratorMISPTLoop2Adapt::GetLocalDispers(const int2 a_pos, const int a_sizeLocalWindow)
{
  const float mathExp       = MathExp(a_pos, a_sizeLocalWindow, false);
  const float mathExpSquare = MathExp(a_pos, a_sizeLocalWindow, true);
  const float dispersion    = mathExpSquare - (mathExp * mathExp);  
  return sqrt(fmax(dispersion, 0.0F));
}

int2 IntegratorMISPTLoop2Adapt::NewPositionWithAdapt(const int2 a_pos, const float imgRadius)
{  
  // offset x or y for get near pixel
  RandomGen& gen = randomGen();
  float2 rnd                          = rndFloat2_Pseudo(&gen) * 2.0F;
  rnd                                -= 1.0F;
  const int offsetX1px                = (int)(rnd.x * 1.5F);
  const int offsetY1px                = (int)(rnd.y * 1.5F);
  int         offsetX                 = a_pos.x + offsetX1px;
  int         offsetY                 = a_pos.y + offsetY1px;
  
  if (offsetX < 0)            offsetX = 1;
  if (offsetY < 0)            offsetY = 1;
  if (offsetX > m_width  - 1) offsetX = m_width  - 2;
  if (offsetY > m_height - 1) offsetY = m_height - 2;
    
  const int   indexCurrPix = a_pos.y * m_width + a_pos.x;
  const int   indexNextPix = offsetY * m_width + offsetX;

  const float lum                     = Luminance(float3(m_summColors[indexCurrPix].x, m_summColors[indexCurrPix].y, m_summColors[indexCurrPix].z));
  const float lumOffsetPix            = Luminance(float3(m_summColors[indexNextPix].x, m_summColors[indexNextPix].y, m_summColors[indexNextPix].z));
                                      
  // calculate R2 error
  const float diffLum                 = lum  - lumOffsetPix;
  const float meanLum                 = (lum + lumOffsetPix) * 0.5F;
  const float errorR2                 = (diffLum * diffLum) / fmax(meanLum * meanLum, 1e-8F); // coefficient of determination     
  const float maxStep                 = imgRadius * 0.01F + 1.0F;
  const float step                    = (1.0F - clamp(errorR2, 0.0F, 1.0F)) * maxStep + 1.5F;  
                                        
  // offset pixels with step                                                
  offsetX                             = (int)(rnd.x * step);
  offsetY                             = (int)(rnd.y * step);
  
  int newX                            = a_pos.x + offsetX;
  int newY                            = a_pos.y + offsetY;

  if (newX < 0)            newX      -= offsetX;
  if (newY < 0)            newY      -= offsetY;
  if (newX > m_width  - 1) newX      -= offsetX;
  if (newY > m_height - 1) newY      -= offsetY;

  return int2(newX, newY);
}

void IntegratorMISPTLoop2Adapt::NewPositionWithMarkovChain(int2& a_pos, float& a_step, RandomGen& a_gen, const float a_imgRadius)
{
  // Get next screen pos 

  const int2   nextPos       = GetNewLocalPos(a_pos, a_step, a_gen, false, false);

  // Get sample

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir) = makeEyeRay(a_pos.x, a_pos.y);
  const float3 colorCurr     = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

  // add color

  const int   indexCurr      = a_pos.y  * m_width + a_pos.x;
  const float alpha          = 1.0F / float(m_samplePerPix[indexCurr] + 1);
  m_summColors[indexCurr]    = m_summColors[indexCurr] * (1.0F - alpha) + to_float4(colorCurr, maxcomp(colorCurr)) * alpha;
#pragma omp atomic
  m_samplePerPix[indexCurr]++;


  // luminance

  //const float currLum        = Luminance(colorCurr);
  //const int   nextIndex      = nextPos.y * m_width + nextPos.x;
  //const float nextLum        = Luminance(m_summColors[nextIndex]);

  // local statistics

  const int sizeLocalWin  = 1;  
  //const bool useSppWeight = true;
  //float currSumm(0);
  //float currMax(0);
  //float currMediana(0);
  //float currMean(0);  
  //float nextSumm(0);
  //float nextMax(0);
  //float nextMediana(0);
  //float nextMean(0);

  //GetStatisticsLocalWin(a_pos  , sizeLocalWin, currSumm, currMax, currMediana, currMean, useSppWeight);
  //GetStatisticsLocalWin(nextPos, sizeLocalWin, nextSumm, nextMax, nextMediana, nextMean, useSppWeight);

  // dispersions

  const float currDispLum    = GetLocalDispers( a_pos, sizeLocalWin);
  const float nextDispLum    = GetLocalDispers(nextPos, sizeLocalWin);

  // R2 error 

  //const float currErrorR2    = r2Error(currLum, currMean);
  //const float nextErrorR2    = r2Error(nextLum, nextMean);

  // probability

  //const float gamma          = 1.0F / 0.02F;
  //const float lumProb        = pow(AcceptProb(currLum, nextLum), gamma);
  const float dispProb       = AcceptProb(currDispLum, nextDispLum);
  //const float r2Prob         = AcceptProb(currErrorR2, nextErrorR2);

  // acceptance probability 

  const float acceptProb     = dispProb;
  a_step                     = (1.0F - acceptProb) * (a_imgRadius * 0.5F) + 1.5F;

  if (rndFloat1_Pseudo(&a_gen) < acceptProb || acceptProb == 0.0F)
    a_pos  = nextPos;  
}



void IntegratorMISPTLoop2Adapt::ScreenSpaceMLT(int2& a_pos, RandomGen& a_gen, const float a_imgRadius, int& a_sample)
{
  // Get first sample

  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir)  = makeEyeRay(a_pos.x, a_pos.y);
  const float3 colorCurr      = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

  // Get next local screen pos 

  const float2 rnd            = rndFloat2_Pseudo(&a_gen);  
  const float  step           = a_imgRadius * 0.01F + 1.5F;
  const int2   newPos         = GetNewLocalPos(a_pos, step, a_gen, false, false);


  // Get next sample

  std::tie(ray_pos, ray_dir)  = makeEyeRay(newPos.x, newPos.y);
  const float3 colorNext      = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);
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
  const float rnd1 = rndFloat1_Pseudo(&a_gen);
  if (rnd1 < acceptProb || acceptProb == 0.0F)
    a_pos = newPos;      
}



int2 IntegratorMISPTLoop2Adapt::GetNewLocalPos(const LiteMath::int2 a_pos, const float a_step, RandomGen& a_gen, const bool a_constantStep, const bool a_fullScreenRnd) const 
{
  const float2 rnd = rndFloat2_Pseudo(&a_gen);

  if (a_fullScreenRnd)  
    return int2((int)(rnd.x * (float)(m_width - 1)), (int)(rnd.y * (float)(m_height - 1)));
           

  float rndX;
  float rndY;

  if (a_constantStep)
  {
    const float angle = rnd.x * M_TWOPI;
    rndX = cos(angle);
    rndY = sin(angle);
  }
  else
  {
    rndX = rnd.x * 2.0F - 1.0F;
    rndY = rnd.y * 2.0F - 1.0F;
  }

  const int2 offset = int2((int)(rndX * a_step), (int)(rndY * a_step));
  int2       newPos = int2(a_pos.x + offset.x, a_pos.y + offset.y);

  if (newPos.x < 0)            newPos.x -= offset.x;
  if (newPos.y < 0)            newPos.y -= offset.y;
  if (newPos.x > m_width  - 1) newPos.x -= offset.x;
  if (newPos.y > m_height - 1) newPos.y -= offset.y;

  return newPos;
}



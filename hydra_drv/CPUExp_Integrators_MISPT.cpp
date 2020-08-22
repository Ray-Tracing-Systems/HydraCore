#include "CPUExp_Integrators.h"

//////////////////////////////////////////////////////////////////////////////////////////

float Distance(const int x, const int y)
{
  return (float)sqrt(x * x + y * y);
}


void IntegratorMISPTLoop2Adapt::DoPass(std::vector<uint>& a_imageLDR)
{  
  // Update HDR image

  if (m_imgSize != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  const float  imgRadius = Distance(m_width, m_height) * 0.5F;
  
  RandomGen&   gen       = randomGen();
  const float2 rnd       = rndFloat2_Pseudo(&gen);
  int2         pos       = int2((int)(rnd.x * (float)(m_width - 1)), (int)(rnd.y * (float)(m_height - 1)));
  
  float3       colorCurr;
  float3       colorNext;

  int          countAccept  = 0;
  const int    maxSample    = m_imgSize;

#ifdef NDEBUG
#pragma omp parallel for
#endif
  for (int sample = 0; sample < maxSample; ++sample)
  {     
    //const int newPos            = NewPositionWithAdapt(startX, startY, imgRadius);
    //NewPositionWithMarkovChain(pos, imgRadius);
    ScreenSpaceMLT(pos, colorCurr, imgRadius, countAccept);

    float3 ray_pos;
    float3 ray_dir;
    std::tie(ray_pos, ray_dir)  = makeEyeRay(pos.x, pos.y);
    colorCurr                   = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

    //const int    index          = pos.y * m_width + pos.x;
    //const float  alpha          = 1.0F / float(m_samplePerPix[index] + 1);
    //const float4 summColor      = m_summColors[index] * (1.0F - alpha) + to_float4(currColor, maxcomp(currColor)) * alpha;
    //m_summColors[index]         = summColor;                 
    //    
    //m_samplePerPix[index]++;
  }

  RandomizeAllGenerators();

  //uint meanSpp = 0;
  //for (int i = 0; i < m_imgSize; i++)
  //  meanSpp += m_samplePerPix[i];

  //meanSpp /= m_imgSize;

  //m_spp = meanSpp;


  // Scale image
  float summ = 0;
  for (int i = 0; i < m_imgSize; i++)
  {
    summ += (m_summColors[i].x + m_summColors[i].y + m_summColors[i].z) * 0.3333333F;
  }
  
  const float meanVal = summ / float(m_imgSize);

  for (int i = 0; i < m_imgSize; i++)
  {
    m_summColors[i].x /= meanVal;
    m_summColors[i].y /= meanVal;
    m_summColors[i].z /= meanVal;
  }



  GetImageToLDR(a_imageLDR, true);


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

  //std::cout << "[" << this->Name() << "]: mean spp = " << meanSpp    << std::endl;
  //std::cout << "[" << this->Name() << "]: progress = " << m_progress << std::endl;

  auto oldPrecition = std::cout.precision(3);
  std::cout << this->Name() << ": acceptanceRate = " << float(countAccept) / float(maxSample) * 100.0F << "%" << std::endl;
  std::cout.precision(oldPrecition);
}




float IntegratorMISPTLoop2Adapt::GetLocalMean(const int x, const int y, const int sizeLocalWindow)
{
  int   samples = 0;  
  float mean    = 0.0F;

  // Get 
  for (int offsetY = -sizeLocalWindow; offsetY <= sizeLocalWindow; offsetY++)
  {
    for (int offsetX = -sizeLocalWindow; offsetX <= sizeLocalWindow; offsetX++)
    {
      int NewX = x + offsetX;
      int NewY = y + offsetY;

      NewX = clamp(NewX, 0, m_width  - 1);
      NewY = clamp(NewY, 0, m_height - 1);
      
      const int index = NewY * m_width + NewX;
          
      mean += Luminance(float3(m_summColors[index].x, m_summColors[index].y, m_summColors[index].z));
      samples++;
    }
  }

  return mean / (float)samples;
}

int2 IntegratorMISPTLoop2Adapt::NewPositionWithAdapt(const int x, const int y, const float imgRadius)
{  
  // offset x or y for get near pixel
  RandomGen& gen = randomGen();
  float2 rnd                          = rndFloat2_Pseudo(&gen) * 2.0F;
  rnd                                -= 1.0F;
  const int offsetX1px                = (int)(rnd.x * 1.5F);
  const int offsetY1px                = (int)(rnd.y * 1.5F);
  int         offsetX                 = x + offsetX1px;
  int         offsetY                 = y + offsetY1px;
  
  if (offsetX < 0)            offsetX = 1;
  if (offsetY < 0)            offsetY = 1;
  if (offsetX > m_width  - 1) offsetX = m_width  - 2;
  if (offsetY > m_height - 1) offsetY = m_height - 2;
    
  const int   indexCurrPix = y       * m_width + x;
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
  
  int newX                            = x + offsetX;
  int newY                            = y + offsetY;

  if (newX < 0)            newX      -= offsetX;
  if (newY < 0)            newY      -= offsetY;
  if (newX > m_width  - 1) newX      -= offsetX;
  if (newY > m_height - 1) newY      -= offsetY;

  return int2(newX, newY);
}

void IntegratorMISPTLoop2Adapt::NewPositionWithMarkovChain(int2& pos, const float imgRadius)
{
  // local screen sampling
  RandomGen& gen                 = randomGen();
  float4 rnd                     = rndFloat4_Pseudo(&gen);
  rnd.x                          = (rnd.x * rnd.x) * 2.0F - 1.0F;
  rnd.y                          = (rnd.y * rnd.y) * 2.0F - 1.0F;
                                 
  const float step               = imgRadius * 0.01F + 1.5F;
  const int2 offset              = int2((int)(rnd.x * step), (int)(rnd.y * step));                                   
  int2       newPos              = int2(pos.x + offset.x, pos.y + offset.y);

  if (newPos.x < 0)            newPos.x -= offset.x;
  if (newPos.y < 0)            newPos.y -= offset.y;
  if (newPos.x > m_width  - 1) newPos.x -= offset.x;
  if (newPos.y > m_height - 1) newPos.y -= offset.y;

  // full screen sampling
  //RandomGen&   gen       = randomGen();
  //const float4 rnd       = rndFloat4_Pseudo(&gen);
  //const int2   newPos    = int2((int)(rnd.x * (float)(m_width - 1)), (int)(rnd.y * (float)(m_height - 1)));

  const int indexCurrPix = pos.y    * m_width + pos.x;
  const int indexNextPix = newPos.y * m_width + newPos.x;

  const float3 colorCurr = float3(m_summColors[indexCurrPix].x, m_summColors[indexCurrPix].y, m_summColors[indexCurrPix].z);
  const float3 colorNext = float3(m_summColors[indexNextPix].x, m_summColors[indexNextPix].y, m_summColors[indexNextPix].z);

  float lumCurrPix       = Luminance(colorCurr);
  float lumNextPix       = Luminance(colorNext);

  // calculate R2 error
   //Walking by error R2 is not effective. For some reason, there is a lot of noise where the error is supposed to be large.
   //Walking on an inverted error doesn't have any effect at all.  
  //const float diffLum = lumCurrPix - lumNextPix;
  //const float meanCurLum = GetLocalMean(x, y, 1);
  //const float meanNexLum = GetLocalMean(newX, newY, 1);
  //const float meanLum = (lumCurrPix + lumNextPix) * 0.5F;
  //const float errorR2 = clamp((diffLum * diffLum) / fmax(meanLum * meanLum, 1e-8F), 0.0F, 1.0F); // coefficient of determination     

  // acceptance probability with R2
  //const float acceptProb = errorR2;

  // acceptance probability with lum
  lumCurrPix          += 0.1F;
  lumNextPix          += 0.1F;
  const float probCurr = fmax(lumCurrPix, 1e-6F);
  const float probNext = lumNextPix;
  float acceptProb     = clamp(probNext / probCurr, 0.0F, 1.0F);
  acceptProb           = pow(acceptProb, 1.0F / 2.2F);// * errorR2;

  // acceptance probability
  if (rnd.z < acceptProb || acceptProb == 0.0F)  
    pos = newPos;  
}

void IntegratorMISPTLoop2Adapt::ScreenSpaceMLT(int2& a_pos, const float3 a_colorCurr, const float a_imgRadius,  int& a_countAccept)
{
  // local screen sampling
  RandomGen& gen              = randomGen();
  float4 rnd                  = rndFloat4_Pseudo(&gen);
  rnd.x                       = rnd.x * 2.0F - 1.0F;
  rnd.y                       = rnd.y * 2.0F - 1.0F;
                              
  const float step            = a_imgRadius * 0.01F + 1.5F;
  const int2 offset           = int2((int)(rnd.x * step), (int)(rnd.y * step));
  int2       newPos           = int2(a_pos.x + offset.x, a_pos.y + offset.y);

  if (newPos.x < 0)            newPos.x -= offset.x;
  if (newPos.y < 0)            newPos.y -= offset.y;
  if (newPos.x > m_width  - 1) newPos.x -= offset.x;
  if (newPos.y > m_height - 1) newPos.y -= offset.y;


  float3 ray_pos;
  float3 ray_dir;
  std::tie(ray_pos, ray_dir) = makeEyeRay(newPos.x, newPos.y);
  const float3 colorNext     = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

  float lumCurrPix            = Luminance(a_colorCurr);
  float lumNextPix            = Luminance(colorNext);


  // acceptance probability 
  const float probCurr        = lumCurrPix + 0.1F;
  const float probNext        = lumNextPix + 0.1F;
  float acceptProb            = clamp(probNext / probCurr, 0.0F, 1.0F);
  //acceptProb                  = pow(acceptProb, 1.0F / 2.2F);



  const float3 summColorCurr = a_colorCurr * (1.0F - acceptProb) / probCurr;
  const float3 summColorNext = colorNext   *         acceptProb  / probNext;

  if (dot(summColorCurr, summColorCurr) > 1e-6F)
  {
    const int index        = a_pos.y * m_width + a_pos.x;
#pragma omp atomic
    m_summColors[index].x += summColorCurr.x;
#pragma omp atomic
    m_summColors[index].y += summColorCurr.y;
#pragma omp atomic
    m_summColors[index].z += summColorCurr.z;
#pragma omp atomic
    m_summColors[index].w += (1.0F - acceptProb);
  }

  if (dot(summColorNext, summColorNext) > 1e-6F)
  {
    const int index        = newPos.y * m_width + newPos.x;
#pragma omp atomic
    m_summColors[index].x += summColorNext.x;
#pragma omp atomic
    m_summColors[index].y += summColorNext.y;
#pragma omp atomic
    m_summColors[index].z += summColorNext.z;
#pragma omp atomic
    m_summColors[index].w += acceptProb;
  }

  // acceptance probability
  if (rnd.z < acceptProb || acceptProb == 0.0F)
  {
    a_pos = newPos;
    a_countAccept++;
  }  
}



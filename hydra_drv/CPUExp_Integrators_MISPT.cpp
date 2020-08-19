#include "CPUExp_Integrators.h"

//////////////////////////////////////////////////////////////////////////////////////////

float Distance(const int x, const int y)
{
  return (float)sqrt(x * x + y * y);
}


void IntegratorMISPTLoop2Adapt::DoPass(std::vector<uint>& a_imageLDR)
{  
  if (m_imgSize != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  // Update HDR image
  //
  const float imgRadius = Distance(m_width, m_height) * 0.5F;

  RandomGen& gen        = randomGen();
  const float2 rnd      = rndFloat2_Pseudo(&gen);
  int x                 = (int)(rnd.x * (float)(m_width  - 1));
  int y                 = (int)(rnd.y * (float)(m_height - 1));

  

#ifdef NDEBUG
#pragma omp parallel for
#endif
  for (int sample = 0; sample < m_imgSize; ++sample)
  {    
    const int index             = y * m_width + x;    

    float3 ray_pos;
    float3 ray_dir;
    std::tie(ray_pos, ray_dir)  = makeEyeRay(x, y);

    const float3 color          = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

    const float  alpha          = 1.0F / float(m_samplePerPix[index] + 1);
    const float4 summColor      = m_summColors[index] * (1.0F - alpha) + to_float4(color, maxcomp(color)) * alpha;
    m_summColors[index]         = summColor;
                 
    //NewPositionWithAdapt2(x, y, m_summColors[index], imgRadius, gen);
    NewPositionWithMarkovChain(x, y, m_summColors[index], imgRadius, gen, sample);
        
    m_samplePerPix[index]++;
  }

  RandomizeAllGenerators();

  uint meanSpp = 0;
  for (int i = 0; i < m_imgSize; i++)
    meanSpp += m_samplePerPix[i];

  meanSpp /= m_imgSize;

  m_spp = meanSpp;
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

  //m_progress = m_progress / (float)m_imgSize * 100.0F;

  std::cout << "[" << this->Name() << "]: mean spp = " << meanSpp    << std::endl;
  //std::cout << "[" << this->Name() << "]: progress = " << m_progress << std::endl;
}


//int IntegratorMISPTLoop2Adapt::GetIndexSmallImage(const int x, const int y) const 
//{
//  const int newX     = (int)((float)x / (float)m_bucketWidth);
//  const int newY     = (int)((float)y / (float)m_bucketWidth);
//  
//  return newY * m_numBucketWidth + newX;
//}


//void IntegratorMISPTLoop2Adapt::NewPositionWithAdapt(int& x, int& y, const LiteMath::float4& summColor, const float& imgRadius, RandomGen& gen)
//{
//  const int   index           = GetIndexSmallImage(x, y);
//  const float alpha           = 1.0F / float(m_samplePerPixArea[index] + 1);
//  const float newLum          = Luminance(float3(summColor.x, summColor.y, summColor.z));
//  const float summColorArea   = m_summColorsArea[index] * (1.0F - alpha) + newLum * alpha;
//  m_summColorsArea[index]     = summColorArea;
//                                                              
//  const float diffLum         = summColorArea - newLum;
//  const float errorR2         = (diffLum * diffLum) / fmax(summColorArea * summColorArea, 1e-8F); // coefficient of determination     
//  const float step            = (1.0F - clamp(errorR2, 0.0F, 1.0F)) * (imgRadius * 0.5F) + 1.5F;
//                              
//  const float angle           = rndFloat1_Pseudo(&gen) * M_TWOPI;
//                             
//  const int offsetX           = (int)(cos(angle) * step);
//  const int offsetY           = (int)(sin(angle) * step);
//
//  int newX                    = x + offsetX;
//  int newY                    = y + offsetY;
//
//  while (newX < 0)            newX -= offsetX;
//  while (newY < 0)            newY -= offsetY;
//  while (newX > m_width  - 1) newX -= offsetX;
//  while (newY > m_height - 1) newY -= offsetY;
//
//  m_samplePerPixArea[index]++;
//
//  x = newX;
//  y = newY;
//}

void IntegratorMISPTLoop2Adapt::NewPositionWithAdapt2(int& x, int& y, const LiteMath::float4 summColor, const float imgRadius, RandomGen& gen) const
{  
  // offset x or y for get near pixel

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
    
  const int   indexOffsetPix          = offsetY * m_width + offsetX;

  const float lum                     = Luminance(float3(summColor.x, summColor.y, summColor.z));
  const float lumOffsetPix            = Luminance(float3(m_summColors[indexOffsetPix].x, m_summColors[indexOffsetPix].y, m_summColors[indexOffsetPix].z));
                                      
  // calculate R2 error
  const float diffLum                 = lum  - lumOffsetPix;
  const float meanLum                 = (lum + lumOffsetPix) * 0.5F;
  const float errorR2                 = (diffLum * diffLum) / fmax(meanLum * meanLum, 1e-8F); // coefficient of determination     
  const float maxStep                 = imgRadius * 0.01F;
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

  x = newX;
  y = newY;
}

void IntegratorMISPTLoop2Adapt::NewPositionWithMarkovChain(int& x, int& y, const LiteMath::float4 summColor, const float imgRadius, RandomGen& gen, const int sample) const
{
  // local screen sampling

  float2 rnd                     = rndFloat2_Pseudo(&gen) * 2.0F;
  rnd                           -= 1.0F;
                                
  const float step               = imgRadius * 0.01F + 1.5F;
                                
  const int offsetX              = (int)(rnd.x * step);
  const int offsetY              = (int)(rnd.y * step);
                                
  int newX                       = x + offsetX;
  int newY                       = y + offsetY;

  if (newX < 0)            newX -= offsetX;
  if (newY < 0)            newY -= offsetY;
  if (newX > m_width  - 1) newX -= offsetX;
  if (newY > m_height - 1) newY -= offsetY;

  // full screen sampling

  //const float2 rnd         = rndFloat2_Pseudo(&gen);
  //const int newX           = (int)(rnd.x * (float)(m_width  - 1));
  //const int newY           = (int)(rnd.y * (float)(m_height - 1));

  const int   indexNextPix = newY * m_width + newX;

  float lum          = Luminance(float3(summColor.x, summColor.y, summColor.z));
  float lumNextPix   = Luminance(float3(m_summColors[indexNextPix].x, m_summColors[indexNextPix].y, m_summColors[indexNextPix].z));

  // calculate R2 error
  const float diffLum      =  lum - lumNextPix;
  const float meanLum      = (lum + lumNextPix) * 0.5F;
  const float errorR2      = clamp((diffLum * diffLum) / fmax(meanLum * meanLum, 1e-8F), 0.0F, 1.0F); // coefficient of determination     
  
  //lum                      = pow(lum,        1.0F / 2.2F);
  //lumNextPix               = pow(lumNextPix, 1.0F / 2.2F);

  //float acceptProb         = clamp(lumNextPix / fmax(lum, 1e-6F), 0.0F, 1.0F); //acceptance probability
  float acceptProb         = errorR2; //acceptance probability
  
  bool        transition   = false;
  const float rnd1         = rndFloat1_Pseudo(&gen);

  if (acceptProb >= 1.0F || rnd1 < acceptProb || acceptProb == 0.0F)
    transition = true;
  
  if (transition)
  {
    x = newX;
    y = newY;
  }
}



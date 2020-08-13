#include "CPUExp_Integrators.h"

//////////////////////////////////////////////////////////////////////////////////////////

float Distance(const int x, const int y)
{
  return sqrt(x * x + y * y);
}


void IntegratorMISPTLoop2Adapt::DoPass(std::vector<uint>& a_imageLDR)
{
  const int sizeImage = m_width * m_height;
  
  if (sizeImage != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");

  // Update HDR image
  //

  const int maxLocalSpp   = 256;
  const float error       = 1.0F / 256.0F;
  bool pixFinish          = false;

#ifdef NDEBUG
#pragma omp parallel for
#endif
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {      
      int sample      = 0;      
      int numFinish   = 0;
      const int index = y * m_width + x;

      float3 ray_pos, ray_dir;
      std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);
      do
      {
        const float3 color     = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

        const float alpha      = 1.0F / float(m_samplePerPix[index] + 1);
        const float maxCol     = maxcomp(color);
        const float prevLum    = Luminance(float3(m_summColors[index].x, m_summColors[index].y, m_summColors[index].z));

        const float4 summColor = m_summColors[index] * (1.0F - alpha) + to_float4(color, maxCol) * alpha;
        m_summColors[index]    = summColor;

        const float newLum     = Luminance(float3(summColor.x, summColor.y, summColor.z));
        const float diffLum    = abs(prevLum - newLum);
        m_samplePerPix[index]++;                        
        sample++;

        if (diffLum < error)
        {
          pixFinish            = true;
          numFinish++;
        }
        //m_pixFinish[index]     = pixFinish;

      } while (numFinish < (maxLocalSpp * 0.1F) && sample < maxLocalSpp);
    }
  }

  RandomizeAllGenerators();

  uint meanSpp = 0;
  for (int i = 0; i < sizeImage; i++)
    meanSpp += m_samplePerPix[i];

  meanSpp /= sizeImage;

  m_spp = meanSpp;
  GetImageToLDR(a_imageLDR);

//#ifdef NDEBUG
//#pragma omp parallel for
//#endif
//  for (int i = 0; i < sizeImage; i++)
//  {
//   //if (!m_pixFinish[i])
//   // {
//   //   float4 color(0.5F, 0.25F, 0.25F, 0.0F);
//   //   a_imageLDR[i] = RealColorToUint32(color);
//   // }          
//    a_imageLDR[i] = RealColorToUint32(float4(m_r2[i], m_r2[i], m_r2[i], 0));
//  }

//  uint progress = 0;
//#ifdef NDEBUG
//#pragma omp parallel for
//#endif
//  for (int i = 0; i < sizeImage; i++)
//    if (m_pixFinish[i])
//      progress++;
//
//  progress = (float)progress / m_pixFinish.size() * 100.0F;

  //if (m_spp == 1)
    //DebugSaveGbufferImage(L"C:/[Hydra]/rendered_images/torus_gbuff");

  std::cout << "[" << this->Name() << "]: mean spp = " << meanSpp  << std::endl;
  //std::cout << "[" << this->Name() << "]: progress = " << progress << std::endl;
}

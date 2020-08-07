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


  //RandomGen& gen          = randomGen();
  //const float2 offset     = rndFloat2_Pseudo(&gen);
  //int x                   = offset.x * (m_width - 1);
  //int y                   = offset.y * (m_height - 1);
  const int maxLocalSpp   = 16;
  const float error       = 0.1F;
  //const float radiusImage = Distance(m_width, m_height) * 0.5F;
  //float step              = radiusImage * 0.5F;

#ifdef NDEBUG
#pragma omp parallel for
#endif
  for (int y = 0; y < m_width; y++)
  {
    for (int x = 0; x < m_height; x++)
    {      
      int sample      = 0;
      float prevLum   = 0.0F;
      float diffLum   = 0.0F;
      const int index = y * m_width + x;

      float3 ray_pos, ray_dir;
      std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);

      while(diffLum > error || sample < maxLocalSpp)
      {
        //const float angle          = rndFloat1_Pseudo(&gen) * M_TWOPI;
        //
        //const int offsetX          = (int)(sin(angle) * step);
        //const int offsetY          = (int)(cos(angle) * step);
        //x                         += offsetX;
        //y                         += offsetY;

        //x                          = clamp(x, 0, m_width  - 1);
        //y                          = clamp(y, 0, m_height - 1);


        const float3 color     = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0);

        const float alpha      = 1.0F / float(m_samplePerPix[index] + 1);
        const float maxCol     = maxcomp(color);

        //const float prevLum    = Luminance(float3(m_summColors[index].x, m_summColors[index].y, m_summColors[index].z));

        const float4 summColor = m_summColors[index] * (1.0F - alpha) + to_float4(color, maxCol) * alpha;
        m_summColors[index]    = summColor;

        // adapt step

        const float newLum     = Luminance(float3(summColor.x, summColor.y, summColor.z));
        const float diffLum    = abs(prevLum - newLum);

        //const float compDiffLum    = diffLum / (1.0F + diffLum);
        //step                       = lerp(max(radiusImage * 0.5F, 2.0F), 2.0F, min(diffLum, 1.0F));

        //m_summColors[index]    = float4{ diffLum, diffLum, diffLum, maxCol };
        
        m_samplePerPix[index] += 1;
                        
        prevLum = newLum;
        sample++;
      }
    }
  }

  RandomizeAllGenerators();

  const int meanSpp = m_samplePerPix[sizeImage * 0.5F];
  m_spp = meanSpp;
  GetImageToLDR(a_imageLDR);

  //if (m_spp == 1)
    //DebugSaveGbufferImage(L"C:/[Hydra]/rendered_images/torus_gbuff");

  std::cout << "[" << this->Name() << "]: mean spp = " << meanSpp << std::endl;
}

#include <omp.h>
#include "CPUExp_Integrators.h"


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
  
  if(m_spp % 17 == 0)
    RandomizeAllGenerators();
  
  std::cout << "IntegratorMISPT_QMC: spp = " << m_spp << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool HR_SaveHDRImageToFileHDR_WithFreeImage(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scale = 1.0f);

std::vector<float> PrefixSumm(const std::vector<float>& a_vec);

IntegratorMISPT_AQMC::IntegratorMISPT_AQMC(int w, int h, EngineGlobals* a_pGlobals, int a_createFlags) : IntegratorMISPT(w, h, a_pGlobals, a_createFlags) 
{
  m_summSquareColors.resize(w*h);
  m_errorMap.resize(w/4, h/4);                                 // #TODO: calc correct resolution!
  for(int i=0;i<m_errorMap.width()*m_errorMap.height()*4;i++)
    m_errorMap.data()[i] = 1.0f;
  
  std::vector<float> errorMap1f(m_errorMap.width()*m_errorMap.height()); 
  for(size_t i=0;i<errorMap1f.size();i++)
    errorMap1f[i] = m_errorMap.data()[i*4+0];
  m_errorTable = PrefixSumm(errorMap1f);

  m_tilesMin.resize(m_errorMap.width()*m_errorMap.height());

  const float tileSize = 1.0f/m_errorMap.width(); 
  for(int y=0;y<m_errorMap.height();y++)
  { 
    float fy = float(y)*tileSize;
    for(int x=0;x<m_errorMap.width();x++)
    { 
       float fx = float(x)*tileSize;
       m_tilesMin[y*m_errorMap.width() + x] = float2(fx,fy);
    }
  }

}

void IntegratorMISPT_AQMC::DoPass(std::vector<uint>& a_imageLDR)
{
  if (m_width*m_height != a_imageLDR.size())
    RUN_TIME_ERROR("DoPass: bad output bufffer size");
  
  const int loopSize   = int(m_summColors.size());
  const int qmcOffset  = int(loopSize)*m_spp;

  const float tileSize = 1.0f/m_errorMap.width(); 
  
  const int maxTileId  = m_errorMap.width()*m_errorMap.height();

  #pragma omp parallel for
  for (int i = 0; i < loopSize; ++i)
  {
    PerThread().qmcPos = qmcOffset + i;
    
    RandomGen& gen  = randomGen();
    
    float tilePdfSelector = 1.0f;
    const float tileR     = rndQmcTab(&gen, m_pGlobals->rmQMC, PerThread().qmcPos, QMC_VAR_SRC_A, (const unsigned int*)m_tableQMC);
    const int   tileId    = SelectIndexPropToOpt(tileR, &m_errorTable[0], int(m_errorTable.size()), 
                                                 &tilePdfSelector);  

    float4 lensOffs = rndLens(&gen, nullptr, float2(1,1), 
                              m_pGlobals->rmQMC, PerThread().qmcPos, (const unsigned int*)m_tableQMC);

    lensOffs.x = m_tilesMin[tileId].x + tileSize*lensOffs.x;
    lensOffs.y = m_tilesMin[tileId].y + tileSize*lensOffs.y;

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
    
    const float selectorMult = (1.0f/float(maxTileId))/tilePdfSelector;

    const float3 color = PathTrace(ray_pos, ray_dir, makeInitialMisData(), 0, 0)*selectorMult;
    const float avgCol = 0.3333333f*(color.x + color.y + color.z);
    
    #pragma omp critical
    {
      m_summColors      [y*m_width + x] = m_summColors      [y*m_width + x] + to_float4(color, 0.0f);
      m_summSquareColors[y*m_width + x] = m_summSquareColors[y*m_width + x] + avgCol*avgCol;
    }
  }

  constexpr float gammaPow = 1.0f/2.2f;
  const float scaleInv     = 1.0f / float(m_spp + 1);

  #pragma omp parallel for
  for (int i = 0; i < int(a_imageLDR.size()); i++)
  {
    float4 color = m_hdrData[i];

    color.x = powf(clamp(color.x*scaleInv, 0.0f, 1.0f), gammaPow);
    color.y = powf(clamp(color.y*scaleInv, 0.0f, 1.0f), gammaPow);
    color.z = powf(clamp(color.z*scaleInv, 0.0f, 1.0f), gammaPow);
    color.w = 1.0f;

    a_imageLDR[i] = RealColorToUint32(color);
  }

  
  m_spp++;
  
  if(m_spp % 17 == 0)
    RandomizeAllGenerators();
  
  if(m_spp % 16 == 0)
  {
    const auto N = m_summColors.size();
    const float fN = float(N);
    
    HDRImage4f testImage(m_width, m_height);
    float4* array = (float4*)testImage.data();

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      const float3 color          = to_float3(m_summColors[i]);
      const float summ            = scaleInv*0.3333333f*(color.x + color.y + color.z);
      const float summOfTheSquare = m_summSquareColors[i];
      const float D               = fmax((summOfTheSquare/fN) - SQR(summ/fN), 0.0f); 
      const float err_abs         = sqrt(D);
      const float err_rel         = err_abs/fmax(summ, 0.001f);

      array[i] = 20.0f*make_float4(err_rel, err_rel, err_rel, 0.0f);
    }

    //testImage.medianFilterInPlace(0.0f, 0.0f);
    testImage.resampleTo(m_errorMap);
    m_errorMap.medianFilterInPlace(0.0f);

    std::vector<float> errorMap1f(m_errorMap.width()*m_errorMap.height());
    for(size_t i=0;i<errorMap1f.size();i++)
      errorMap1f[i] = m_errorMap.data()[i*4+0];
    m_errorTable = PrefixSumm(errorMap1f);

    //std::ofstream fout("/home/frol/hydra/rendered_images/test.txt");
    //for(size_t i=0;i<errorMap1f.size();i++)
    //  fout << errorMap1f[i] << "\t" << m_errorTable[i] << std::endl;

    HR_SaveHDRImageToFileHDR_WithFreeImage(L"/home/frol/hydra/rendered_images/error.hdr", testImage.width(), testImage.height(), testImage.data());
    HR_SaveHDRImageToFileHDR_WithFreeImage(L"/home/frol/hydra/rendered_images/error2.hdr", m_errorMap.width(), m_errorMap.height(), m_errorMap.data());
    std::cout << "error image saves" << std::endl;

    //static bool firstTime = true;
    //if(firstTime)
    //{
    //  m_spp = 0;
    //  std::cout << "reset spp counter for test!" << std::endl;
    //  firstTime = false;
    //}
  }

  std::cout << "IntegratorMISPT_AQMC: spp = " << m_spp << std::endl;
}

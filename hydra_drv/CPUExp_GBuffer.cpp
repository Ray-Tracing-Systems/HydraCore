#include <omp.h>

#include "CPUExp_Integrators.h"
#include "ctrace.h"

#include <math.h>
#include <algorithm>
#include <unordered_map>


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GBufferAll IntegratorCommon::gbufferEval(int x, int y)
{
  const float fov = DEG_TO_RAD*90.0f;

  GBufferAll samples[GBUFFER_SAMPLES];
 
  // (0) generate hammersly samples ... 
  //
  float2 qmc[GBUFFER_SAMPLES];
  PlaneHammersley(&qmc[0].x, GBUFFER_SAMPLES);

  // (1) rotate samples may be? 
  //

  // (2) eval samples
  //

  const float sizeInvX = 1.0f / (float)(m_width);
  const float sizeInvY = 1.0f / (float)(m_width);

  for (int i = 0; i < GBUFFER_SAMPLES; i++)
  { 
    float4 lensOffs = make_float4(qmc[i].x, qmc[i].y, 0, 0);

    lensOffs.x = sizeInvX * (lensOffs.x + (float)x);
    lensOffs.y = sizeInvY * (lensOffs.y + (float)y);

    float  fx, fy;
    float3 ray_pos, ray_dir;
    MakeEyeRayFromF4Rnd(lensOffs, m_pGlobals,
                        &ray_pos, &ray_dir, &fx, &fy);

    samples[i] = gbufferSample(ray_pos, ray_dir);
  }

  // (3) find biggest cluster and eval coverage
  //
  float minDiff   = 100000000.0f; 
  int   minDiffId = 0;

  for (int i = 0; i < GBUFFER_SAMPLES; i++)
  {
    float diff     = 0.0f;
    float coverage = 0.0f;
    for (int j = 0; j < GBUFFER_SAMPLES; j++)
    {
      const float thisDiff = gbuffDiff(samples[i], samples[j], fov, m_width, m_height);
      diff += thisDiff;
      if (thisDiff < 1.0f)
        coverage += 1.0f;
    }

    coverage *= (1.0f / (float)GBUFFER_SAMPLES);
    samples[i].data1.coverage = coverage;

    if (diff < minDiff)
    {
      minDiff   = diff;
      minDiffId = i;
    }
  }

  // (4) average depth, norm and e.t.c for all samples with the same cluster
  //

  return samples[minDiffId];
}


GBufferAll IntegratorCommon::gbufferSample(float3 ray_pos, float3 ray_dir)
{
  GBufferAll result; 
  initGBufferAll(&result);

  Lite_Hit liteHit = rayTrace(ray_pos, ray_dir);

  if (HitNone(liteHit))
  {
    result.data1.rgba = float4(0, 0, 0, 1);
  }
  else
  {
    auto surfHit = surfaceEval(ray_pos, ray_dir, liteHit);
   
    result.data1.depth    = liteHit.t;
    result.data1.norm     = surfHit.normal;
    result.data1.rgba     = to_float4(evalDiffuseColor(ray_dir, surfHit), 0.0f); // #TODO: eval alpha here for transparent surfaces 
    result.data1.matId    = surfHit.matId;
    result.data1.coverage = 1.0f;

    result.data2.texCoord = surfHit.texCoord;
    result.data2.objId    = liteHit.geomId;
    result.data2.instId   = liteHit.instId;
  }

  return result;
}

bool HR_SaveLDRImageToFile(const wchar_t* a_fileName, int w, int h, int32_t* data);

void DebugGetDepthImage(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_depthImageLDR)
{
  std::vector<float> depthNormalised(a_gbuff.size());

  float minDepth = 1e+6f;
  float maxDepth = 0.0f;

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const float depth = a_gbuff[i].data1.depth;
    if (depth > 0.0f && depth < 1e+6f)
    {
      if (depth < minDepth) minDepth = depth;
      if (depth > maxDepth) maxDepth = depth;
      depthNormalised[i] = depth;
    }
    else
      depthNormalised[i] = 0.0f;
  }

  if (maxDepth <= 1e-20f)
  {
    std::cout << "DebugGetDepthImage, bad depth image!" << std::endl;
    return;
  }

  a_depthImageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const float depth2 = fmax((depthNormalised[i] - minDepth) / maxDepth, 0.0f);
    a_depthImageLDR[i] = RealColorToUint32(float4(depth2, depth2, depth2, 1));
  }

}

void GetNormalsImage(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const float3 norm = a_gbuff[i].data1.norm;
    a_imageLDR[i] = RealColorToUint32(float4(fabs(norm.x), fabs(norm.y), fabs(norm.z), 1));
  }
}

void GetTexColor(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const float4 color = a_gbuff[i].data1.rgba;
    a_imageLDR[i] = RealColorToUint32(color);
  }
}

void GetTexCooord(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const float2 texcoord = a_gbuff[i].data2.texCoord;
    a_imageLDR[i] = RealColorToUint32(make_float4(texcoord.x, texcoord.y, 0.0f, 1.0f));
  }
}


void GetCoverageImage(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const float coverage = a_gbuff[i].data1.coverage;
    a_imageLDR[i] = RealColorToUint32(float4(coverage, coverage, coverage, 1));
  }
}

void GetAlphaImage(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const float coverage = a_gbuff[i].data1.rgba.w;
    a_imageLDR[i] = RealColorToUint32(float4(coverage, coverage, coverage, 1));
  }
}

const int g_colorTable[16] = { 0x00101010, 0x0000F000, 0x000000F0, 0x00F00000,
                               0x00F0F010, 0x0010F0F0, 0x00F010F0, 0x007030C0,
                               0x00A0A0A0, 0x00902000, 0x00005020, 0x00B04010,
                               0x00600030, 0x00008010, 0x00205000, 0x00004040 };

void GetObjectId(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const int32_t id = a_gbuff[i].data2.objId;
    a_imageLDR[i] = g_colorTable[id % 16];
  }
}

void GetInstId(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const int32_t id = a_gbuff[i].data2.instId;
    a_imageLDR[i] = g_colorTable[id % 16];
  }
}

void GetMatId(const std::vector<GBufferAll>& a_gbuff, std::vector<int32_t>& a_imageLDR)
{
  a_imageLDR.resize(a_gbuff.size());

  for (size_t i = 0; i < a_gbuff.size(); i++)
  {
    const int32_t id = a_gbuff[i].data1.matId;
    a_imageLDR[i] = g_colorTable[id % 16];
  }
}

void IntegratorCommon::CalcGBufferUncompressed(std::vector<GBufferAll>& a_gbuff)
{
  // (1) calc gbuffer
  //
  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
      a_gbuff[y*m_width + x] = gbufferEval(x, y);
  }
}

void TestCompressGBuffer(std::vector<GBufferAll>& a_gbuff)
{
  std::vector<float4> gbuff1(a_gbuff.size());
  std::vector<float4> gbuff2(a_gbuff.size());
  
  #pragma omp parallel for
  for (int i = 0; i < int(a_gbuff.size()); i++)
  {
    GBufferAll all = a_gbuff[i];
    gbuff1[i] = packGBuffer1(all.data1);
    gbuff2[i] = packGBuffer2(all.data2);
  }

  #pragma omp parallel for
  for (int i = 0; i < int(a_gbuff.size()); i++)
  {
    GBufferAll all;
    all.data1 = unpackGBuffer1(gbuff1[i]);
    all.data2 = unpackGBuffer2(gbuff2[i]);
    a_gbuff[i] = all;
  }

}

void IntegratorCommon::DebugSaveGbufferImage(const wchar_t* a_path)
{
  // (1) calc gbuffer
  //
  std::vector<GBufferAll> gbuffer(m_width*m_height);
  CalcGBufferUncompressed(gbuffer);
  
  // TestCompressGBuffer(gbuffer);

  // (2) save gbuffer to different layers for debug purpose
  //
  std::cout << "saving images ... " << std::endl;
  {
    const std::wstring folder        = std::wstring(a_path) + L"/";
    const std::wstring depthFile     = folder + L"01_depth.png";
    const std::wstring normalsFile   = folder + L"02_normals.png";
    const std::wstring texcolorFile  = folder + L"03_texcolor.png";
    const std::wstring texcooordFile = folder + L"04_texcoord.png";
    const std::wstring coverageFile  = folder + L"05_coverage.png";
    const std::wstring objIdFile     = folder + L"06_objid.png";
    const std::wstring instIdFile    = folder + L"07_instid.png";
    const std::wstring matIdFile     = folder + L"08_matid.png";
    const std::wstring alphaFile     = folder + L"09_alpha.png";

    std::vector<int32_t> tmpImage(m_width*m_height);

    DebugGetDepthImage(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(depthFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetNormalsImage(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(normalsFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetTexColor(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(texcolorFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetTexCooord(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(texcooordFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetCoverageImage(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(coverageFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetObjectId(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(objIdFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetInstId(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(instIdFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetMatId(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(matIdFile.c_str(), m_width, m_height, &tmpImage[0]);

    GetAlphaImage(gbuffer, tmpImage);
    HR_SaveLDRImageToFile(alphaFile.c_str(), m_width, m_height, &tmpImage[0]);
  }
  std::cout << "end of images ... " << std::endl;


}

bool HR_SaveHDRImageToFileHDR(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scale = 1.0f);

float MedianOfMaxColorInWindow(const float4* a_data, int a_x, int a_y, int a_width, int a_height, int a_windowSize,
                               float* pAvg)
{
  std::vector<float> pixVals;  
  pixVals.reserve(a_windowSize*a_windowSize + 1);
  
  int minX = a_x - a_windowSize;
  int maxX = a_x + a_windowSize;
  int minY = a_y - a_windowSize;
  int maxY = a_y + a_windowSize;

  if (minX < 0) minX = 0;
  if (minY < 0) minY = 0;

  if (maxX >= a_width - 1)  maxX = a_width  - 1;
  if (maxY >= a_height - 1) maxY = a_height - 1;

  float avg = 0.0f;
  for (int y = minY; y <= maxY; y++)
  {
    const int offset = y*a_width;
    for (int x = minX; x <= maxX; x++)
    {
      const float val = maxcomp(to_float3(a_data[offset+x]));
      pixVals.push_back(val);
      avg += val;
    }
  }

  std::sort(pixVals.begin(), pixVals.end());
  (*pAvg) = avg / float(pixVals.size());
  return pixVals[pixVals.size()/2];
}

void IntegratorCommon::ExtractNoise(const float4* a_data, const float a_userCoeff,
                                    std::vector<float>& a_errArray, float& normConst)
{
  float maxVal = 0.0f;
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {
      const float4 data = a_data[y*m_width + x];
      const float thisVal = maxcomp(to_float3(data));
      float avg = 0.0f;
      const float median = MedianOfMaxColorInWindow(a_data, x, y, m_width, m_height, 3, &avg);
      const float err = fabs(thisVal - median); // +fabs(thisVal - avg);
      if (err > maxVal)
        maxVal = err;

      a_errArray[y*m_width + x] = err;
    }
  }

  std::vector<float> errArray = a_errArray;
  std::nth_element(errArray.begin(), errArray.begin() + errArray.size() / 2, errArray.end());
  const float median = errArray[errArray.size() / 2];

  normConst = 2.0f*a_userCoeff / (median + maxVal);
  //normConst = 1.0f*a_userCoeff / median;
}

void IntegratorCommon::DebugSaveNoiseImage(const wchar_t* a_path, const float4* a_data, const float a_userCoeff)
{
  float normConst = 1.0f;
  
  std::vector<float>  errArray(m_width*m_height);
  ExtractNoise(a_data, a_userCoeff,
               errArray, normConst);

  std::vector<float4> errImage(m_width*m_height);
  for (size_t i = 0; i < errImage.size(); i++)
  {
    const float val = errArray[i];
    errImage[i] = normConst*float4(val,val,val, 1.0f);
  }

  HR_SaveHDRImageToFileHDR(a_path, m_width, m_height, (const float*)&errImage[0]);
}

void IntegratorCommon::SpreadNoise(const std::vector<GBufferAll>& a_gbuff, std::vector<float>& a_noise)
{
  const int WINDOW_SIZE = 64;
  const float g_GaussianSigma = 1.0f / 50.0f;
  const float fov = DEG_TO_RAD*90.0f;

  //const float match = surfaceSimilarity(n0, n1, ppSize);
  //clamp(match, 0.25f, 1.0f);

  // (1) horizontal pass (a_noise => temp)
  //
  std::vector<float> temp(a_noise.size());

  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    // int minY = y - WINDOW_SIZE; 
    // int maxY = y + WINDOW_SIZE;
    // if (minY < 0)  minY = 0;
    // if (maxY >= m_height) maxY = m_height - 1;

    for (int x = 0; x < m_width; x++)
    {
      int minX = x - WINDOW_SIZE;
      int maxX = x + WINDOW_SIZE;
      if (minX < 0)  minX = 0;
      if (maxX >= m_width) maxX = m_width - 1;

      GBufferAll thisPixel = a_gbuff[y*m_width + x];

      float avgVal = 0.0f;
      float maxVal = 0.0f;
      int currCounter = 0;

      for (int x1 = minX; x1 <= maxX; x1++)
      {
        GBufferAll otherPixel = a_gbuff[y*m_width + x1];
        const float thisDiff  = gbuffDiff(thisPixel, otherPixel, fov, m_width, m_height);

        const int d = abs(x-x1);
        const float gaussW = exp(-(0.0f + (d*d) * g_GaussianSigma));

        if ((otherPixel.data1.coverage > 0.85f) && (thisDiff < 1.0f || (thisPixel.data1.coverage < 0.85f && d <= 1)))
        {
          const float val = a_noise[y*m_width + x1]*gaussW;
          maxVal = fmax(maxVal, val);
          avgVal += val;
          currCounter++;
        }
      }

      avgVal *= (1.0f / float(currCounter));
      temp[y*m_width + x] = 0.5f*(avgVal + maxVal);

    }
  }


  // (2) vertical pass (a_noise <= temp)
  //
  #pragma omp parallel for
  for (int x = 0; x < m_width; x++)
  {
    for (int y = 0; y < m_height; y++)
    {
      int minY = y - WINDOW_SIZE; 
      int maxY = y + WINDOW_SIZE;
      if (minY < 0)  minY = 0;
      if (maxY >= m_height) maxY = m_height - 1;

      GBufferAll thisPixel = a_gbuff[y*m_width + x];

      float avgVal = 0.0f;
      float maxVal = 0.0f;
      int currCounter = 0;

      for (int y1 = minY; y1 <= maxY; y1++)
      {
        GBufferAll otherPixel = a_gbuff[y1*m_width + x];
        const float thisDiff  = gbuffDiff(thisPixel, otherPixel, fov, m_width, m_height);

        const int d = abs(y - y1);
        const float gaussW = exp(-(0.0f + (d*d) * g_GaussianSigma));

        if ((otherPixel.data1.coverage > 0.85f) && thisDiff < 1.0f || (thisPixel.data1.coverage < 0.85f && d <= 1))
        {
          const float val = temp[y1*m_width + x]*gaussW;
          maxVal = fmax(maxVal, val);
          avgVal += val;
          currCounter++;
        }
      }

      avgVal *= (1.0f / float(currCounter));
      a_noise[y*m_width + x] = 0.5f*(avgVal + maxVal);

    }
  }

}

static inline uint64_t objectClassId(const GBufferAll& a_pixelG)
{
  uint64_t matId  = (uint64_t)a_pixelG.data1.matId;
  uint64_t instId = (uint64_t)a_pixelG.data2.instId;
  return (matId << 32) | instId;
}

void IntegratorCommon::SpreadNoise2(const std::vector<GBufferAll>& a_gbuff, std::vector<float>& a_noise)
{

  struct ObjectInfo
  {
    ObjectInfo() : avgNoise(0.0f), maxNoise(0.0f), numNoise(0) {}
    ObjectInfo(float a_noiseLvl) : avgNoise(a_noiseLvl), maxNoise(a_noiseLvl), numNoise(1) {}
    float avgNoise;
    float maxNoise;
    int   numNoise;
  };

  std::unordered_map<uint64_t, ObjectInfo> objHash;
  objHash.reserve(1000);

  // (1) collect per object info
  //
  for (size_t i = 0; i < a_noise.size(); i++)
  {
    const GBufferAll gval = a_gbuff[i];
    const float noiseLvl  = a_noise[i];

    if (noiseLvl < 0.1f || gval.data1.coverage < 0.85f)
      continue;

    const uint64_t idx = objectClassId(gval);
    auto p = objHash.find(idx);
    if (p != objHash.end())
    {
      p->second.avgNoise += noiseLvl;
      p->second.numNoise += 1;
      p->second.maxNoise = fmax(p->second.maxNoise, noiseLvl);
    }
    else
    {
      ObjectInfo val(noiseLvl);
      objHash[idx] = val;
    }

  }

  // (2) paint all pixels for each noisy object (or use fill like in paint instead)
  //
  float maxVal = 0.0f;
  float avgVal = 0.0f;

  for (size_t i = 0; i < a_noise.size(); i++)
  {
    const GBufferAll gval = a_gbuff[i];

    const uint64_t idx = objectClassId(gval);
    auto p = objHash.find(idx);
    if (p != objHash.end())
      a_noise[i] = 0.5f*(p->second.avgNoise/float(p->second.numNoise) + p->second.maxNoise);

    if (maxVal < a_noise[i])
      maxVal = a_noise[i];

    avgVal += a_noise[i];
  }

  avgVal /= float(a_noise.size());

  // (2.1) rescale noise image
  //
  const float scaleInv = 1.0f/maxVal; // 2.0f / (maxVal + avgVal);
  for (size_t i = 0; i < a_noise.size(); i++)
  {
    float newVal = scaleInv*a_noise[i];

    const GBufferAll gval = a_gbuff[i];
    if (gval.data1.rgba.w <= 0.5f)
      a_noise[i] = clamp(newVal, 0.1f, 1.0f);
    else
      a_noise[i] = 0.0f;

    //if(newVal <= 0.5f)
    //  a_noise[i] = clamp(newVal*newVal, 0.1f, 1.0f);
    //else
    //  a_noise[i] = clamp(powf(newVal, 0.25f), 0.1f, 1.0f);
  }

  //// (3) blur/paint coverage boundary
  ////
  //const int WINDOW_SIZE       = 16;
  //const float g_GaussianSigma = 1.0f / 50.0f;
  //
  //// horizontal pass (a_noise => temp)
  ////
  //std::vector<float> temp(a_noise.size());
  //
  //#pragma omp parallel for
  //for (int y = 0; y < m_height; y++)
  //{
  //  for (int x = 0; x < m_width; x++)
  //  {
  //    int minX = x - WINDOW_SIZE;
  //    int maxX = x + WINDOW_SIZE;
  //    if (minX < 0)  minX = 0;
  //    if (maxX >= m_width) maxX = m_width - 1;
  //
  //    float avgVal = 0.0f;
  //    float maxVal = 0.0f;
  //    int currCounter = 0;
  //
  //    for (int x1 = minX; x1 <= maxX; x1++)
  //    {
  //      const int d = abs(x-x1);
  //      const float gaussW = exp(-(0.0f + (d*d) * g_GaussianSigma));
  //
  //      const float val = a_noise[y*m_width + x1]*gaussW;
  //      avgVal += val;
  //      maxVal = fmax(maxVal, val);
  //      currCounter++; 
  //    }
  //
  //    temp[y*m_width + x] = 0.5f*(avgVal/float(currCounter) + maxVal);
  //
  //  }
  //}
  //
  //
  //// vertical pass (a_noise <= temp)
  ////
  //#pragma omp parallel for
  //for (int x = 0; x < m_width; x++)
  //{
  //  for (int y = 0; y < m_height; y++)
  //  {
  //    int minY = y - WINDOW_SIZE; 
  //    int maxY = y + WINDOW_SIZE;
  //    if (minY < 0)  minY = 0;
  //    if (maxY >= m_height) maxY = m_height - 1;
  //
  //    float avgVal = 0.0f;
  //    float maxVal = 0.0f;
  //    int currCounter = 0;
  //
  //    for (int y1 = minY; y1 <= maxY; y1++)
  //    {
  //      const int d = abs(y - y1);
  //      const float gaussW = exp(-(0.0f + (d*d) * g_GaussianSigma));
  //
  //      const float val = temp[y1*m_width + x]*gaussW;
  //      avgVal += val;
  //      maxVal = fmax(maxVal, val);
  //      currCounter++;
  //    }
  //
  //    a_noise[y*m_width + x] = clamp( 0.5f*(avgVal / float(currCounter) + maxVal), 0.1f, 1.0f);
  //  }
  //}


}
 

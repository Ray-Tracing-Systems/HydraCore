#include "IHWLayer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int clampi(int x, int a, int b)
{
  if (x < a)      return a;
  else if (x > b) return b;
  else            return x;
}

float SQRF(float x) { return x*x; }

void BilateralFilter(const HDRImage4f& inImage, HDRImage4f& outImage, int a_windowRadius, float a_smoothLvl)
{
  ////////////////////////////////////////////////////////////////////
  float g_NoiseLevel = 1.0f / (a_smoothLvl*a_smoothLvl);
  float g_GaussianSigma = 1.0f / 50.0f;
  float g_WeightThreshold = 0.03f;
  float g_LerpCoefficeint = 0.80f;
  float g_CounterThreshold = 0.05f;
  ////////////////////////////////////////////////////////////////////

  const int w = inImage.width();
  const int h = inImage.height();

  outImage.resize(w, h);

  const float4* in_buff = (const float4*)inImage.data();
  float4*       out_buff = (float4*)outImage.data();

  float windowArea = SQRF(2.0f * float(a_windowRadius) + 1.0f);

  #pragma omp parallel for
  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      const int minX = clampi(x - a_windowRadius, 0, w - 1);
      const int maxX = clampi(x + a_windowRadius, 0, w - 1);

      const int minY = clampi(y - a_windowRadius, 0, h - 1);
      const int maxY = clampi(y + a_windowRadius, 0, h - 1);

      const float4 c0 = in_buff[y*w + x];

      int counterPass = 0;

      float fSum = 0.0f;
      float4 result(0, 0, 0, 0);

      // do window
      //
      for (int y1 = minY; y1 <= maxY; y1++)
      {
        for (int x1 = minX; x1 <= maxX; x1++)
        {
          const float4 c1 = in_buff[y1*w + x1];
          const float4 dist = c1 - c0;

          const int i = x1 - x;
          const int j = y1 - y;

          const float w1 = dot3f(dist, dist);
          const float w2 = exp(-(w1 * g_NoiseLevel + (i * i + j * j) * g_GaussianSigma));

          if (w2 > g_WeightThreshold)
            counterPass++;

          fSum += w2;
          result += c1 * w2;
        }
      }

      result = result * (1.0f / fSum);

      //  Now the restored pixel is ready
      //  But maybe the area is actually edgy and so it's better to take the pixel from the original image?	
      //  This test shows if the area is smooth or not
      //
      float lerpQ = (float(counterPass) > (g_CounterThreshold * windowArea)) ? 1.0f - g_LerpCoefficeint : g_LerpCoefficeint;

      //  This is the last lerp
      //  Most common values for g_LerpCoefficient = [0.85, 1];
      //  So if the area is smooth the result will be
      //  RestoredPixel*0.85 + NoisyImage*0.15
      //  If the area is noisy
      //  RestoredPixel*0.15 + NoisyImage*0.85
      //  That allows to preserve edges more thoroughly
      //
      result = lerp(result, c0, lerpQ);

      out_buff[y*w + x] = result;
    }
  }

}

std::vector<uchar4> CPUSharedData::NormalMapFromDisplacement(int w, int h, const uchar4* a_data, float bumpAmt, bool invHeight, float smoothLvl)
{
  typedef unsigned char uchar;

  const uchar4* height = a_data;

  std::vector<float> heightDataf(w*h);
  HDRImage4f normslaImage(w, h);

  // init
  //
  for (auto i = 0; i < heightDataf.size(); i++)
    heightDataf[i] = 255.0f - fmax(float(height[i].x), fmax(float(height[i].y), float(height[i].z)));

  const float kScale = 2000.0f / fmin(float(w), float(h));

  // calculate normals
  //
  #pragma omp parallel for
  for (int y = 0; y < int(h); y++)
  {
    int offsetY = y*w;
    int offsetYPlusOne = (y + 1)*w;
    int offsetYMinusOne = (y - 1)*w;

    if (y + 1 >= h) offsetYPlusOne = 0;
    if (y - 1 <= 0) offsetYMinusOne = (h - 1)*w;

    for (int x = 0; x < int(w); x++)
    {
      int offsetXPlusOne = x + 1;
      int offsetXMinusOne = x - 1;

      if (x + 1 >= w) offsetXPlusOne = 0;
      if (x - 1 <= 0) offsetXMinusOne = w - 1;

      float diff[8];

      diff[0] = heightDataf[offsetY + x] - heightDataf[offsetYMinusOne + offsetXMinusOne];
      diff[1] = heightDataf[offsetY + x] - heightDataf[offsetYMinusOne + x];
      diff[2] = heightDataf[offsetY + x] - heightDataf[offsetYMinusOne + offsetXPlusOne];
      diff[3] = heightDataf[offsetY + x] - heightDataf[offsetY + offsetXMinusOne];
      diff[4] = heightDataf[offsetY + x] - heightDataf[offsetY + offsetXPlusOne];
      diff[5] = heightDataf[offsetY + x] - heightDataf[offsetYPlusOne + offsetXMinusOne];
      diff[6] = heightDataf[offsetY + x] - heightDataf[offsetYPlusOne + x];
      diff[7] = heightDataf[offsetY + x] - heightDataf[offsetYPlusOne + offsetXPlusOne];

      if (!invHeight)
      {
        for (int i = 0; i < 8; i++)
          diff[i] *= -1.0f;
      }

      for (int i = 0; i < 8; i++)
        diff[i] *= (bumpAmt*bumpAmt);

      float scale = kScale;// *clamp(1.0f - bumpAmt, 0.1f, 1.0f); // 100.0f

      float3 v38[8];
      v38[0] = float3(-diff[0], -diff[0], scale);
      v38[1] = float3(0.f, -diff[1], scale);
      v38[2] = float3(diff[2], -diff[2], scale);
      v38[3] = float3(-diff[3], 0.f, scale);
      v38[4] = float3(diff[4], 0.f, scale);
      v38[5] = float3(-diff[5], diff[5], scale);
      v38[6] = float3(0.f, diff[6], scale);
      v38[7] = float3(diff[7], diff[7], scale);

      float3 res(0, 0, 0);
      for (int i = 0; i < 8; i++)
        res += v38[i];

      res = res*(1.0f / 8.0f);

      res.x *= -1.0f;
      res.y *= +1.0f;
      res.z *= +1.0f;

      res = normalize(res);

      if (res.z < 0.65f)
      {
        res.z = 0.65f;
        res = normalize(res);
      }

      normslaImage.data()[offsetY * 4 + x * 4 + 0] = res.x;
      normslaImage.data()[offsetY * 4 + x * 4 + 1] = res.y;
      normslaImage.data()[offsetY * 4 + x * 4 + 2] = res.z;
      normslaImage.data()[offsetY * 4 + x * 4 + 3] = (255.0f - heightDataf[offsetY + x]) / 255.0f;
    }
  }

  // if (0)
  // {
  //   float4* pData = (float4*)normslaImage.data();
  //   SaveHDRImageToFileHDR("D:\\temp\\norms1.hdr", w, h, pData);
  // }

  //if(int(blurRadius) > 0 && blurSigma > 0.0f)
  //  normslaImage.gaussBlur(int(blurRadius), blurSigma); // shitty gauss blur (!!!)

  if (smoothLvl >= 1.0f)
  {
    int radius = 5; // smoothLvl;
    if (radius > 7) radius = 7;
    if (smoothLvl > 10.0f) smoothLvl = 10.0f;

    HDRImage4f image2(w, h);
    BilateralFilter(normslaImage, image2, radius, smoothLvl*0.1f);
    normslaImage = image2;
  }

  // if (0)
  // {
  //   float4* pData = (float4*)normslaImage.data();
  //   SaveHDRImageToFileHDR("D:\\temp\\norms2.hdr", w, h, pData);
  // }

  std::vector<uchar4> normalmapData(w*h);

  for (int y = 0; y < int(h); y++)
  {
    int offsetY = int(y)*w;

    for (int x = 0; x < w; x++)
    {
      float4 cr, res;

      res.x = normslaImage.data()[offsetY * 4 + x * 4 + 0];
      res.y = normslaImage.data()[offsetY * 4 + x * 4 + 1];
      res.z = normslaImage.data()[offsetY * 4 + x * 4 + 2];
      res.w = normslaImage.data()[offsetY * 4 + x * 4 + 3];

      cr.x = 0.5f*res.x + 0.5f;
      cr.y = 0.5f*res.y + 0.5f;
      cr.z = res.z;
      cr.w = res.w;

      cr = clamp(cr*255.0f, 0.0f, 255.0f);

      normalmapData[offsetY + x] = uchar4(uchar(cr.x), uchar(cr.y), uchar(cr.z), uchar(cr.w));
    }
  }

  return normalmapData;
}


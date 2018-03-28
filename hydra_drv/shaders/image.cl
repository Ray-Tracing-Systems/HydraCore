////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int2      make_int2(int a, int b)                         { int2   res; res.x = a; res.y = b; return res; }
inline float2    make_float2(float a, float b)                   { float2 res; res.x = a; res.y = b;                       return res; }
inline float3    make_float3(float a, float b, float c)          { float3 res; res.x = a; res.y = b; res.z = c;            return res; }
inline float4    make_float4(float a, float b, float c, float d) { float4 res; res.x = a; res.y = b; res.z = c; res.w = d; return res; }

inline float dot3(const float4 u, const float4 v) { return (u.x*v.x + u.y*v.y + u.z*v.z); }
inline float SQRF(float x) { return x*x; }

inline float4 mylerp(const float4 u, const float4 v, const float t) { return u + t * (v - u); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline float getHeight(const float4 inVal) 
{ 
  const float4 res = 255.0f*(make_float4(1,1,1,1) - inVal);
  return fmax(res.x, fmax(res.y, res.z));
}

inline float2 toNormalizedCoord(const int2 input, const float2 multInv)
{
  const float x1 = ((float)input.x) + 0.5f;
  const float y1 = ((float)input.y) + 0.5f;
  return make_float2(x1, y1)*multInv;
}

// removed CLK_ADDRESS_REPEAT. Seems new AMD wants  CLK_ADDRESS_CLAMP with CLK_NORMALIZED_COORDS_FALSE.

//#define THIS_SAMPLER_MODE (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST) 
#define THIS_SAMPLER_MODE2 (CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST) 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void NormalmapFromHeight(write_only image2d_t a_outImage, image2d_t a_inputImage, int w, int h, int invHeight, float bumpAmt)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= w || y >= h)
    return;

  const float2 multInv = make_float2(1.0f / (float)(w), 1.0f / (float)(h));

  const sampler_t FETCH_VAL = THIS_SAMPLER_MODE2;

  const float kScale     = 2000.0f / fmin((float)(w), (float)(h));
  const float hThisPixel = getHeight(read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x, y), multInv)));

  float diff[8];

  diff[0] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x-1, y-1), multInv)  ) ); 
  diff[1] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x,   y-1), multInv)  ) ); 
  diff[2] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x+1, y-1), multInv)  ) ); 
  diff[3] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x-1, y  ), multInv)  ) ); 
  diff[4] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x+1, y  ), multInv)  ) ); 
  diff[5] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x-1, y+1), multInv)  ) ); 
  diff[6] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x  , y+1), multInv)  ) ); 
  diff[7] = hThisPixel - getHeight( read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x+1, y+1), multInv)  ) ); 

  const float allDiffSumm = fabs(diff[0]) + fabs(diff[1]) + fabs(diff[2]) + fabs(diff[3]) + fabs(diff[4]) + fabs(diff[5]) + fabs(diff[6]) + fabs(diff[7]);

  if (!invHeight)
  {
    for (int i = 0; i<8; i++)
      diff[i] *= -1.0f;
  }

  for (int i = 0; i < 8; i++)
    diff[i] *= (bumpAmt*bumpAmt);

  const float scale = kScale;

  float3 v38[8];
  v38[0] = make_float3(-diff[0], -diff[0], scale);
  v38[1] = make_float3(0.f, -diff[1], scale);
  v38[2] = make_float3(diff[2], -diff[2], scale);
  v38[3] = make_float3(-diff[3], 0.f, scale);
  v38[4] = make_float3(diff[4], 0.f, scale);
  v38[5] = make_float3(-diff[5], diff[5], scale);
  v38[6] = make_float3(0.f, diff[6], scale);
  v38[7] = make_float3(diff[7], diff[7], scale);

  float3 res = make_float3(0, 0, 0);
  for (int i = 0; i<8; i++)
    res += v38[i];

  res = res*(1.0f / 8.0f);

  res.x *= -1.0f;
  res.y *= -1.0f;
  res.z *= +1.0f;

  res = normalize(res);

  if (res.z < 0.65f)
  {
    res.z = 0.65f;
    res = normalize(res);
  }

  if (allDiffSumm < 1.0f)
  {
    res.x = 0.0f;
    res.y = 0.0f;
    res.z = 1.0f;
  }

  const float zeroPointFive = 0.49995f; // this is due to on NV and AMD 0.5 saved to 8 bit textures in different way - 127 on NV and 128 on AMD

  const float resX = clamp(0.5f*res.x + zeroPointFive, 0.0f, 1.0f);
  const float resY = clamp(0.5f*res.y + zeroPointFive, 0.0f, 1.0f);
  const float resZ = clamp(1.0f*res.z + 0.0f, 0.0f, 1.0f);
  const float resW = clamp((255.0f - hThisPixel) / 255.0f, 0.0f, 1.0f);
  
  write_imagef(a_outImage, make_int2(x, y), make_float4(resX, resY, resZ, resW));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void BilateralFilter(write_only image2d_t a_outImage, image2d_t a_inputImage, int w, int h, int a_windowRadius, float a_noiseLvl)
{

  ////////////////////////////////////////////////////////////////////
  float g_NoiseLevel       = 1.0f / (a_noiseLvl*a_noiseLvl); // 1.0f / (0.15f*0.15f);
  float g_GaussianSigma    = 1.0f / 50.0f;
  float g_WeightThreshold  = 0.03f;
  float g_LerpCoefficeint  = 0.80f;
  float g_CounterThreshold = 0.05f;
  ////////////////////////////////////////////////////////////////////

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= w || y >= h)
    return;

  const float2 multInv = make_float2(1.0f / (float)(w), 1.0f / (float)(h));

  const sampler_t FETCH_VAL = THIS_SAMPLER_MODE2;

  int    counterPass = 0;
  float  fSum   = 0.0f;
  float4 result = make_float4(0, 0, 0, 0);

  const float windowArea = SQRF(2.0f * ((float)(a_windowRadius)) + 1.0f);

  const int minX = x - a_windowRadius;
  const int maxX = x + a_windowRadius;

  const int minY = y - a_windowRadius;
  const int maxY = y + a_windowRadius;

  const float4 c0 = read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x, y), multInv)); // in_buff[y*w + x];

  // do window
  //
  for (int y1 = minY; y1 <= maxY; y1++)
  {
    for (int x1 = minX; x1 <= maxX; x1++)
    {
      const float4 c1 = read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x1, y1), multInv)); // in_buff[y1*w + x1];
      const float4 dist = c1 - c0;

      const int i = x1 - x;
      const int j = y1 - y;

      const float w1 = dot3(dist, dist);
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
  float lerpQ = ((float)(counterPass) > (g_CounterThreshold * windowArea)) ? 1.0f - g_LerpCoefficeint : g_LerpCoefficeint;

  //  This is the last lerp
  //  Most common values for g_LerpCoefficient = [0.85, 1];
  //  So if the area is smooth the result will be
  //  RestoredPixel*0.85 + NoisyImage*0.15
  //  If the area is noisy
  //  RestoredPixel*0.15 + NoisyImage*0.85
  //  That allows to preserve edges more thoroughly
  //
  result = mylerp(result, c0, lerpQ);

  write_imagef(a_outImage, make_int2(x, y), result);
}


inline float NLMWeight(image2d_t a_inputImage, int w, int h, int x, int y, int x1, int y1, int a_blockRadius)
{
  const float2 multInv = make_float2(1.0f / (float)(w), 1.0f / (float)(h));

  const sampler_t FETCH_VAL = THIS_SAMPLER_MODE2;

  float w1 = 0.0f; // this is what NLM differs from KNN (bilateral)
  {
    const int minX1 = x1 - a_blockRadius;
    const int maxX1 = x1 + a_blockRadius;

    const int minY1 = y1 - a_blockRadius;
    const int maxY1 = y1 + a_blockRadius;

    for (int y2 = minY1; y2 <= maxY1; y2++)
    {
      for (int x2 = minX1; x2 <= maxX1; x2++)
      {
        const int offsX = x2 - x1;
        const int offsY = y2 - y1;

        const int x3 = x + offsX;
        const int y3 = y + offsY;

        const float4 c2 = read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x2, y2), multInv) ); 
        const float4 c3 = read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x3, y3), multInv) ); 

        const float4 dist = c2 - c3;
        w1 += dot3(dist, dist);
      }
    }

  }

  return w1 / SQRF(2.0f * (float)(a_blockRadius) + 1.0f);
}


__kernel void NonLocalMeansFilter(write_only image2d_t a_outImage, image2d_t a_inputImage, int w, int h, int a_windowRadius, float a_noiseLvl)
{

  ////////////////////////////////////////////////////////////////////
  const int   a_blockRadius      = 1;
  const float g_NoiseLevel       = 1.0f / (a_noiseLvl*a_noiseLvl); // 1.0f / (0.15f*0.15f);
  const float g_GaussianSigma    = 1.0f / 50.0f;
  const float g_WeightThreshold  = 0.03f;
  const float g_LerpCoefficeint  = 0.80f;
  const float g_CounterThreshold = 0.05f;
  ////////////////////////////////////////////////////////////////////

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= w || y >= h)
    return;

  const float2 multInv = make_float2(1.0f / (float)(w), 1.0f / (float)(h));

  const sampler_t FETCH_VAL = THIS_SAMPLER_MODE2;

  int    counterPass = 0;
  float  fSum = 0.0f;
  float4 result = make_float4(0, 0, 0, 0);

  const float windowArea = SQRF(2.0f * ((float)(a_windowRadius)) + 1.0f);

  const int minX = x - a_windowRadius;
  const int maxX = x + a_windowRadius;

  const int minY = y - a_windowRadius;
  const int maxY = y + a_windowRadius;

  const float4 c0 = read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x, y), multInv)); // in_buff[y*w + x];

  // do window
  //
  for (int y1 = minY; y1 <= maxY; y1++)
  {
    for (int x1 = minX; x1 <= maxX; x1++)
    {
      const float4 c1 = read_imagef(a_inputImage, FETCH_VAL, toNormalizedCoord(make_int2(x1, y1), multInv)); // in_buff[y1*w + x1];

      const int i = x1 - x;
      const int j = y1 - y;

      //const float4 dist = c1 - c0;
      //const float w1 = dot3(dist, dist);
      //
      const float w1 = NLMWeight(a_inputImage, w, h, x, y, x1, y1, a_blockRadius);
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
  float lerpQ = ((float)(counterPass) > (g_CounterThreshold * windowArea)) ? 1.0f - g_LerpCoefficeint : g_LerpCoefficeint;

  //  This is the last lerp
  //  Most common values for g_LerpCoefficient = [0.85, 1];
  //  So if the area is smooth the result will be
  //  RestoredPixel*0.85 + NoisyImage*0.15
  //  If the area is noisy
  //  RestoredPixel*0.15 + NoisyImage*0.85
  //  That allows to preserve edges more thoroughly
  //
  result = mylerp(result, c0, lerpQ);

  write_imagef(a_outImage, make_int2(x, y), result);
}

// change 17.07.2017 17:48;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



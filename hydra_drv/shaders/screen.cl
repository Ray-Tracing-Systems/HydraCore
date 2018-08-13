#include "cglobals.h"
#include "crandom.h"
#include "cfetch.h"


__kernel void MakeEyeRays(int offset,                                  // #TODO: this kernel is not used anymore; remove it!
                          __global float4* out_pos,
                          __global float4* out_dir,
                          int w, int h,
                          __global const EngineGlobals* a_globals)
{
  int tid = GLOBAL_ID_X;
  if (tid + offset >= w*h)
    return;

  ushort2 screenPos;
  screenPos.y = (tid + offset) / w;
  screenPos.x = (tid + offset) - screenPos.y*w;

  float4x4 a_mViewProjInv  = make_float4x4(a_globals->mProjInverse);
  float4x4 a_mWorldViewInv = make_float4x4(a_globals->mWorldViewInverse);

  float3 ray_pos = make_float3(0.0f, 0.0f, 0.0f);
  float3 ray_dir = EyeRayDir(screenPos.x, screenPos.y, w, h, a_mViewProjInv);

  ray_dir        = tiltCorrection(ray_pos, ray_dir, a_globals);

  matrix4x4f_mult_ray3(a_mWorldViewInv, &ray_pos, &ray_dir);

  out_pos[tid] = to_float4(ray_pos, 1.0f);
  out_dir[tid] = to_float4(ray_dir, 0.0f);
}

__kernel void MakeEyeRaysSPP(__global float4* out_pos,
                             __global float4* out_dir,
                             int w, int h, int b_size, int yStart,
                             __global const float2* a_qmc,
                             __global const EngineGlobals* a_globals)
{
  const int tid     = GLOBAL_ID_X;
  const int offset  = yStart * w;
  const int pixelId = (tid / b_size + offset);
  if (pixelId >= w * h)
    return;

  const float sizeInvX = 1.0f / (float)(w);
  const float sizeInvY = 1.0f / (float)(h);

  const int y = pixelId / w;
  const int x = pixelId - (y * w);

  const float2 qmc = a_qmc[tid % b_size];
  float4 lensOffs  = make_float4(qmc.x, qmc.y, 0, 0);

  lensOffs.x = sizeInvX * (lensOffs.x + (float)x);
  lensOffs.y = sizeInvY * (lensOffs.y + (float)y);

  // (2) generate random camera sample
  //
  float  fx = (float)x, fy = (float)y;
  float3 ray_pos, ray_dir;
  MakeEyeRayFromF4Rnd(lensOffs, a_globals,
                      &ray_pos, &ray_dir, &fx, &fy);

  out_pos [tid] = to_float4(ray_pos, fx);
  out_dir [tid] = to_float4(ray_dir, fy);
}

__kernel void MakeEyeRaysSamplesOnly(__global RandomGen*           restrict out_gens,
                                     __global float4*              restrict out_samples,
                                     __global int2*                restrict out_zind,
                                     __global const EngineGlobals* restrict a_globals,
                                     __constant ushort*            restrict a_mortonTable256,
                                     __constant unsigned int*      restrict a_qmcTable,
                                     int a_passNumberForQmc, int w, int h, int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  RandomGen gen             = out_gens[tid];
  const float2 mutateScale  = make_float2(a_globals->varsF[HRT_MLT_SCREEN_SCALE_X], a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y]);
  const unsigned int qmcPos = reverseBits(tid, a_size) + a_passNumberForQmc * a_size; // we use reverseBits due to neighbour thread number put in to sobol random generator are too far from each other 
  const float4 lensOffs     = rndLens(&gen, 0, mutateScale, 
                                      a_qmcTable, qmcPos, a_globals->rmQMC);
  out_gens[tid]             = gen;

  const float fwidth        = a_globals->varsF[HRT_WIDTH_F];
  const float fheight       = a_globals->varsF[HRT_HEIGHT_F];

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// related to MakeEyeRayFromF4Rnd

  unsigned short x = (unsigned short)(lensOffs.x*fwidth);
  unsigned short y = (unsigned short)(lensOffs.y*fheight);

  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;

  if (x < 0)  x = 0;
  if (y < 0)  y = 0;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// related to MakeEyeRayFromF4Rnd

  int2 indexToSort;
  indexToSort.x = ZIndex(x,y, a_mortonTable256);
  indexToSort.y = tid;
  
  out_samples[tid] = lensOffs;
  out_zind   [tid] = indexToSort;
}


__kernel void MakeEyeRaysUnifiedSampling(__global float4*              restrict out_pos, 
                                         __global float4*              restrict out_dir, 
                                         __global int*                 restrict out_packXY,

                                         int w, int h, int a_size,
                                         __global const EngineGlobals* restrict a_globals, 
                                         __global uint*                restrict a_flags,
                                         __global float4*              restrict out_color,
                                         __global float4*              restrict out_thoroughput,
                                         __global float4*              restrict out_fog,
                                         __global HitMatRef*           restrict out_hitMat,
                                         __global PerRayAcc*           restrict out_accPdf,

                                         __global const int2*          restrict in_zind,
                                         __global const float4*        restrict in_samples,
                                         __constant ushort*            restrict a_mortonTable256,
                                         __constant unsigned int*      restrict a_qmcTable, 
                                         int a_passNumberForQmc, int a_packIndexForCPU)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  // (1) generate 4 random floats
  //
  const int2 sortedIndex = in_zind[tid];
  const float4 lensOffs  = in_samples[sortedIndex.y]; // we pack lensOffs in this buffer in 'MakeEyeRaysSamplesOnly' kernel

  // (2) generate random camera sample
  //
  float  fx, fy;
  float3 ray_pos, ray_dir;
  MakeEyeRayFromF4Rnd(lensOffs, a_globals,
                      &ray_pos, &ray_dir, &fx, &fy);

  int x = (int)(fx);
  int y = (int)(fy);

  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;

  if (x < 0)  x = 0;
  if (y < 0)  y = 0;

  out_pos   [tid] = to_float4(ray_pos, fx);
  out_dir   [tid] = to_float4(ray_dir, fy);
  out_packXY[tid] = packXY1616(x, y);

  // clear all other per-ray data
  //
  HitMatRef data3;
  data3.m_data    = 0; 
  data3.accumDist = 0.0f;

  a_flags        [tid] = 0;
  out_color      [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  out_thoroughput[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  out_fog        [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  out_hitMat     [tid] = data3;
  out_accPdf     [tid] = InitialPerParAcc();
}


__kernel void ContribSampleToScreen(const __global float4* in_color, const __global int2* a_indices, __constant ushort* a_mortonTable256,
                                    const int a_samplesNum, const int w, const int h, const float a_spp, const float a_gammaInv,
                                    __global  float4* out_colorHDR, __global uint* out_colorLDR, const int alreadySorted)
{
  const int x = GLOBAL_ID_X;
  const int y = GLOBAL_ID_Y;

  if (x >= w || y >= h)
    return;

  const int pxZIndex = ZIndex(x, y, a_mortonTable256);

  // (1) run binary search to find pair in a_indices where val.x == pixelIndex // no bilinear2D.
  //
  const int samNum = a_samplesNum;
  const int beginX = binarySearchForLeftRange (a_indices, samNum, pxZIndex);
  const int endX   = samNum - 1;

  float4 color = make_float4(0, 0, 0, 0);
  
  if (beginX != -1)
  {
    int i            = beginX;
    int2 xOldNewPair = a_indices[i];
  
    while (i <= endX && xOldNewPair.x == pxZIndex)
    {
      const int offset = (alreadySorted == 1) ? i : xOldNewPair.y;

      color += in_color[offset];
      i++;
      xOldNewPair = (i <= endX) ? a_indices[i] : xOldNewPair;
    }
  }

  // (2) save HDR image
  //
  const float4 newColor = out_colorHDR[Index2D(x, y, w)] + color;
  out_colorHDR[Index2D(x, y, w)] = newColor;

  const float sppp       = (float)(a_samplesNum) / (float)(w*h);
  const float scaleConst = 1.0f / (a_spp + sppp);

  // (3) save LDR image
  //
  if (out_colorLDR != 0)
  {
    float4 color2;
    color2.x = pow(scaleConst*newColor.x, a_gammaInv);
    color2.y = pow(scaleConst*newColor.y, a_gammaInv);
    color2.z = pow(scaleConst*newColor.z, a_gammaInv);
    color2.w = pow(scaleConst*newColor.w, a_gammaInv);
    out_colorLDR[Index2D(x, y, w)] = RealColorToUint32(ToneMapping4(color2));
  }
}

__kernel void PackIndexToColorW(const __global int* a_packedId, __global float4* out_color, const int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_color[tid].w = as_float(a_packedId[tid]);
}

__kernel void RealColorToRGB256(__global   float4* in_color,
                                __global   uint*   out_color, 
                                int                a_width, 
                                int                a_height,
                                __constant ushort* a_mortonTable256, 
                                float  a_gamma)
{
  uint x = GLOBAL_ID_X;
  uint y = GLOBAL_ID_Y;

  if (x >= a_width || y >= a_height)
    return;

  float4 color = in_color[Index2D(x, y, a_width)];
    
  float gammaPow = 1.0f / a_gamma;  // gamma correction

  color.x = pow(color.x, gammaPow);
  color.y = pow(color.y, gammaPow);
  color.z = pow(color.z, gammaPow);
  color.w = pow(color.w, gammaPow);

  out_color[Index2D(x, y, a_width)] = RealColorToUint32(ToneMapping4(color));
} 

__kernel void FillColor(__global uint* out, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out[tid] = 0xFF00FFFF;
}


__kernel void MemSetu32(__global uint* out, uint a_value, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out[tid] = a_value;
}


__kernel void MemSetf4(__global float4* out, float4 a_value, int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out[tid + iOffset] = a_value;
}

__kernel void MemCopyu32(__global uint* in_buff, uint a_offset1, __global uint* out_buff, uint a_offset2, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_buff[tid + a_offset2] = in_buff[tid + a_offset1];
}


__kernel void SimpleReductionTest(__global int* a_data, int iNumElements)
{ 
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  __local int sArray[CMP_RESULTS_BLOCK_SIZE];

  // reduce all thisRayIsDead to shared allBlockRaysAreDead
  //

  sArray[LOCAL_ID_X] = a_data[tid];
  SYNCTHREADS_LOCAL;

  for (uint c = CMP_RESULTS_BLOCK_SIZE / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] += sArray[LOCAL_ID_X + c];
    SYNCTHREADS_LOCAL;
  }

  if (LOCAL_ID_X == 0)
    a_data[tid] = sArray[0];

}


__kernel void ReductionFloat4AvgSqrt256(__global const float4* in_data, __global float4* out_data, int iNumElements)
{
  int tid = GLOBAL_ID_X;

  __local float4 sArray[256];

  if (tid < iNumElements)
  {
    float4 data = in_data[tid];

    data.x = sqrt(data.x);
    data.y = sqrt(data.y);
    data.z = sqrt(data.z);
    data.w = sqrt(data.w);

    sArray[LOCAL_ID_X] = data;
  }
  else
    sArray[LOCAL_ID_X] = make_float4(0, 0, 0, 0);

  SYNCTHREADS_LOCAL;

  for (uint c = 256 / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] += sArray[LOCAL_ID_X + c];
    SYNCTHREADS_LOCAL;
  }

  if (LOCAL_ID_X == 0)
    out_data[tid / 256] = sArray[0] * (1.0f / 256.0f);
}

__kernel void ReductionFloat4Avg256(__global const float4* in_data, __global float4* out_data, int iNumElements)
{
  int tid = GLOBAL_ID_X;
 
  float4 data = make_float4(0, 0, 0, 0);

  if (tid < iNumElements)
    data = in_data[tid];

  if (!isfinite(data.x)) data.x = 0.0f;
  if (!isfinite(data.y)) data.y = 0.0f;
  if (!isfinite(data.z)) data.z = 0.0f;
  if (!isfinite(data.w)) data.w = 0.0f;
  
  __local float4 sArray[256];
  sArray[LOCAL_ID_X] = data;
  SYNCTHREADS_LOCAL;

  for (uint c = 256 / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] += sArray[LOCAL_ID_X + c];
    SYNCTHREADS_LOCAL;
  }

  if (LOCAL_ID_X == 0)
    out_data[tid / 256] = sArray[0] * (1.0f / 256.0f);
}



__kernel void ReductionFloat4Avg64(__global const float4* in_data, __global float4* out_data, int iNumElements)
{
  int tid = GLOBAL_ID_X;

  __local float4 sArray[64];

  if (tid < iNumElements)
    sArray[LOCAL_ID_X] = in_data[tid];
  else
    sArray[LOCAL_ID_X] = make_float4(0, 0, 0, 0);

  SYNCTHREADS_LOCAL;

  for (uint c = 64 / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] += sArray[LOCAL_ID_X + c];
    SYNCTHREADS_LOCAL;
  }

  if (LOCAL_ID_X == 0)
    out_data[tid / 64] = sArray[0] * (1.0f / 64.0f);
}


__kernel void ReductionFloat4Avg16(__global const float4* in_data, __global float4* out_data, int iNumElements)
{
  int tid = GLOBAL_ID_X;

  __local float4 sArray[16];

  if (tid < iNumElements)
    sArray[LOCAL_ID_X] = in_data[tid];
  else
    sArray[LOCAL_ID_X] = make_float4(0, 0, 0, 0);

  SYNCTHREADS_LOCAL;

  for (uint c = 16 / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] += sArray[LOCAL_ID_X + c];
    SYNCTHREADS_LOCAL;
  }

  if (LOCAL_ID_X == 0)
    out_data[tid / 16] = sArray[0] * (1.0f / 16.0f);
}

__kernel void ReductionGBuffer16(__global const float4* in_data, __global float4* out_data, int iNumElements)
{
  int tid  = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  GBuffer1 buffRes;

  buffRes.depth = 0.0f;
  buffRes.norm  = make_float3(0, 0, 0);
  buffRes.rgba  = make_float4(0, 0, 0, 0);
  buffRes.matId = -1;

  int normalCounter = 0;
  int depthCounter = 0;

  for (int i = 0; i < 16; i++)
  {
    int tid2 = tid * 16 + i;

    GBuffer1 buffSpp = unpackGBuffer1(in_data[tid2]);

    if (buffSpp.depth < 1000000000.0f - 100.0f)
    {
      buffRes.depth += buffSpp.depth;
      depthCounter++;
    }

    if (dot(buffSpp.norm, buffSpp.norm) > 1e-5f)
    {
      buffRes.norm += buffSpp.norm;
      normalCounter++;
    }

    buffRes.rgba += buffSpp.rgba;

    if (buffSpp.matId != -1)
      buffRes.matId = buffSpp.matId;
  }

  // #TODO: add correct reduction of matId based on max spp for target material
 
  buffRes.depth *= (1.0f / fmax((float)depthCounter, 1.0f));
  buffRes.norm  *= (1.0f / fmax((float)normalCounter, 1.0f));
  buffRes.rgba  *= (1.0f / 16.0f);

  out_data[tid] = packGBuffer1(buffRes);
}


__kernel void GetAlphaToGBuffer(__global float4* out_data, __global const float4* in_data, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  GBuffer1 buffRes  = unpackGBuffer1(out_data[tid]);
  float3 alphaColor = to_float3(in_data[tid]);
  buffRes.rgba.w    = 1.0f - dot(alphaColor, make_float3(0.35f, 0.51f, 0.14f));
  out_data[tid]     = packGBuffer1(buffRes);
}


__kernel void FloatToHalf(__global float* a_inData, __global half* a_outData, int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float data = clamp(a_inData[tid + iOffset], 0.0f, 65504.0f); // 65504.0f

  vstore_half(data, tid + iOffset, a_outData);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__kernel void ScanTest(__global int* a_inData, __global int* a_outData, int a_size)
{ 
  int tid = GLOBAL_ID_X;

  const int size = 256;
  __local int l_Data[512];


  int idata = a_inData[tid];
  int odata = 0;
  PREFIX_SUMM_MACRO(idata, odata, l_Data, size);

  if (tid < a_size)
    a_outData[tid] = odata;
}

__kernel void CompactTest(__global int* a_inData, __global int* a_outData, int a_size)
{
  int tid = GLOBAL_ID_X;

  __local int l_Data[512];

  int idata = a_inData[tid];
  int odata = 0;

  PREFIX_SUMM_MACRO(idata, odata, l_Data, 256);

  // perform compaction
  //
  
  SYNCTHREADS_LOCAL;
  if (idata != 0 && (odata - 1) < 256 && (odata - 1) >= 0)
    l_Data[odata-1] = LOCAL_ID_X;
  SYNCTHREADS_LOCAL;

  odata = l_Data[LOCAL_ID_X];
  
  if (tid < a_size)
    a_outData[tid] = odata;

}

__kernel void Texture2DTest(texture2d_t a_tex, __global float* a_outData, int a_w, int a_h)
{
  int x = GLOBAL_ID_X;
  int y = GLOBAL_ID_Y;

  if (x >= a_w || y >= a_h)
    return;

  float fx = (float)x / (float)a_w;
  float fy = (float)y / (float)a_h;

  float4 texColor4 = make_float4(2, 2, 2, 2);

  a_outData[y*a_w + x] = texColor4.x;
}



__kernel void ReadNormalsAndDepth(__global const float4* in_posNorm, __global const Lite_Hit* in_hits, __global float4* a_color, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  Lite_Hit hit = in_hits[tid];
  float3 norm  = normalize(decodeNormal(as_int(in_posNorm[tid].w)));

  if (HitNone(hit))
  {
    norm  = make_float3(0, 0, 0);
    hit.t = 1000000000.0f;
  }

  a_color[tid] = to_float4(norm, hit.t);
}


__kernel void MaxPixelSize256(__global const Lite_Hit*  in_hits,
                              __global const HitMatRef* in_matData,
                              __global const EngineGlobals* a_globals, 
                              __global float* a_result)
{
  int tid = GLOBAL_ID_X;

  float3 color = make_float3(0, 0, 0);
  Lite_Hit hit = in_hits[tid];

  float pixelSize = 0.0f;

  if (HitNone(hit))
  {
    float2 pp = projectedPixelSize2(in_matData[tid].accumDist, a_globals);
    pixelSize = fmax(fmax(pp.x, pp.y), 0.0f);
  }

  __local float sArray[256];
  sArray[LOCAL_ID_X] = pixelSize;
  SYNCTHREADS_LOCAL;

  for (uint c = 256 / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] = fmax(sArray[LOCAL_ID_X], sArray[LOCAL_ID_X + c]);
    SYNCTHREADS_LOCAL;
  }

  if (LOCAL_ID_X == 0)
    a_result[tid/256] = sArray[0];
}




__kernel void BlendFrameBuffers(__global float4* out_dst, __global const float4* in_src1, __global const float4* in_src2, 
                               float4 k1, float4 k2, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_dst[tid] = in_src1[tid] * k1 + in_src2[tid] * k2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void TestAtomicsInt(__global int4* pValues, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint offset = (tid * (tid * tid * 15731 + 74323) + 871483) % (uint)(iNumElements);

  __global int* ptr = (__global int*)(pValues + offset);

  atomic_add(ptr + 0, 1);
  atomic_add(ptr + 0, 2);
  atomic_add(ptr + 0, 3);
}


__kernel void TestAtomicsFloat(__global float4* pValues, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint offset = (tid * (tid * tid * 15731 + 74323) + 871483) % (uint)(iNumElements);

  __global float* ptr = (__global float*)(pValues + offset);

  atomic_addf(ptr + 0, 1.0f);
  atomic_addf(ptr + 1, 2.0f);
  atomic_addf(ptr + 2, 3.0f);
}


__kernel void CopyShadowTo(__global const uchar* in_shadow, __global float4* out_color, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const float shadow = (float)(in_shadow[tid]) * (1.0f / 255.0f);
  out_color[tid]     = make_float4(shadow, shadow, shadow, shadow);
}


// change 31.01.2018 15:20;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


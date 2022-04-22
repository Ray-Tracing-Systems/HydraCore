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
  float3 ray_dir = EyeRayDirNormalized((screenPos.x + 0.5f)/(float)w, 
                                       (screenPos.y + 0.5f)/(float)h, a_mViewProjInv);

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



__kernel void MakeEyeRaysSPPPixels(__global float4*              restrict out_pos,
                                   __global float4*              restrict out_dir,
                                   __global int*                 restrict out_XY,

                                   int w, int h, 
                                   __global const int*           restrict in_pixcoords,
                                   
                                   __global RandomGen*           restrict out_gens,
                                    __constant unsigned int*     restrict a_qmcTable, 
                                    int a_passNumberForQmc,
                                   __global const float2*        restrict in_qmc,
                                   __global const EngineGlobals* restrict a_globals, 
                                   int a_size)
{
  const int tid       = GLOBAL_ID_X;
  const int pixPacked = in_pixcoords[tid / PMPIX_SAMPLES];

  const int y = (pixPacked & 0xFFFF0000) >> 16;
  const int x = (pixPacked & 0x0000FFFF);

  //const float2 qmc      = in_qmc[tid % PMPIX_SAMPLES];
  //const float4 lensOffs = make_float4(qmc.x, qmc.y, 0, 0); //#TODO: add dof sampling via sobol qmc. 
  
  RandomGen gen             = out_gens[tid];
  const float2 mutateScale  = make_float2(a_globals->varsF[HRT_MLT_SCREEN_SCALE_X], a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y]);
  const unsigned int qmcPos = (tid % PMPIX_SAMPLES) + a_passNumberForQmc*PMPIX_SAMPLES; // we use reverseBits due to neighbour thread number put in to sobol random generator are too far from each other 
  float4 lensOffs           = rndLens(&gen, 0, mutateScale, a_globals->rmQMC, qmcPos, a_qmcTable);
  out_gens[tid]             = gen;

  const float sizeInvX  = 1.0f / (float)(w);
  const float sizeInvY  = 1.0f / (float)(h);

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
  out_XY  [tid] = packXY1616(x,y);
}

// make multidim sample for QMC/KMLT test
//
__kernel void MakeEyeRaysQMC(__global RandomGen*           restrict out_gens,
                             __global float*               restrict out_samples,
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
  const unsigned int qmcPos = tid + a_passNumberForQmc * a_size;
  const float4 lensOffs     = rndLens(&gen, 0, mutateScale, 
                                      a_globals->rmQMC, qmcPos, a_qmcTable);

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
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// QMC/KMLT memory layout
  {
    const int vecSize = kmltBlobSize(a_globals);
    
    out_samples[vecSize*tid + 0] = lensOffs.x;
    out_samples[vecSize*tid + 1] = lensOffs.y;
    out_samples[vecSize*tid + 2] = lensOffs.z;
    out_samples[vecSize*tid + 3] = lensOffs.w;
    
    if(a_globals->varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] > 0)
    {
      out_samples[vecSize*tid + 4] = rndQmcTab(&gen, a_globals->rmQMC, qmcPos, QMC_VAR_LGT_0, a_qmcTable);
      out_samples[vecSize*tid + 5] = rndQmcTab(&gen, a_globals->rmQMC, qmcPos, QMC_VAR_LGT_1, a_qmcTable);
      out_samples[vecSize*tid + 6] = rndQmcTab(&gen, a_globals->rmQMC, qmcPos, QMC_VAR_LGT_2, a_qmcTable);
      out_samples[vecSize*tid + 7] = rndQmcTab(&gen, a_globals->rmQMC, qmcPos, QMC_VAR_LGT_N, a_qmcTable);
    }

    int top = 8;
    for(int lightB=1; lightB < a_globals->varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES]; lightB += 4)
    {
      const float4 data = rndFloat4_Pseudo(&gen);
      out_samples[vecSize*tid + top + 0] = data.x;
      out_samples[vecSize*tid + top + 1] = data.y;
      out_samples[vecSize*tid + top + 2] = data.z;
      out_samples[vecSize*tid + top + 3] = data.w;
      top += 4;
    }
    
    if(a_globals->varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES] > 0)
    {
      float6_gr gr1f;
      gr1f.group24 = rndFloat4_Pseudo(&gen);
      gr1f.group16 = rndFloat2_Pseudo(&gen);
      float4  gr2f = rndFloat4_Pseudo(&gen);

      gr1f.group24.x  = rndQmcTab(&gen, a_globals->rmQMC, qmcPos, QMC_VAR_MAT_0, a_qmcTable);
      gr1f.group24.y  = rndQmcTab(&gen, a_globals->rmQMC, qmcPos, QMC_VAR_MAT_1, a_qmcTable);
      gr1f.group24.w  = rndQmcTab(&gen, a_globals->rmQMC, qmcPos, QMC_VAR_MAT_L, a_qmcTable); // #LOOK_AT: MMLT_FLOATS_PER_MLAYER

      const uint4 gr1 = packBounceGroup(gr1f);
      const uint2 gr2 = packBounceGroup2(gr2f);

      out_samples[vecSize*tid + top + 0] = as_float( gr1.x );
      out_samples[vecSize*tid + top + 1] = as_float( gr1.y );
      out_samples[vecSize*tid + top + 2] = as_float( gr1.z );
      out_samples[vecSize*tid + top + 3] = as_float( gr1.w );
      out_samples[vecSize*tid + top + 4] = as_float( gr2.x );
      out_samples[vecSize*tid + top + 5] = as_float( gr2.y );
      top += 6;
    }

    for(int matB=1; matB < a_globals->varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES]; matB += 6)
    { 
      float6_gr gr1f;
      gr1f.group24 = rndFloat4_Pseudo(&gen);
      gr1f.group16 = rndFloat2_Pseudo(&gen);
      float4  gr2f = rndFloat4_Pseudo(&gen);

      uint4 gr1 = packBounceGroup(gr1f);
      uint2 gr2 = packBounceGroup2(gr2f);

      out_samples[vecSize*tid + top + 0] = as_float( gr1.x );
      out_samples[vecSize*tid + top + 1] = as_float( gr1.y );
      out_samples[vecSize*tid + top + 2] = as_float( gr1.z );
      out_samples[vecSize*tid + top + 3] = as_float( gr1.w );
      out_samples[vecSize*tid + top + 4] = as_float( gr2.x );
      out_samples[vecSize*tid + top + 5] = as_float( gr2.y );
      top += 6;
    }
  } 
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// QMC/KMLT memory layout 

  out_zind[tid] = indexToSort;
  out_gens[tid] = gen;
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
  const unsigned int qmcPos = tid + a_passNumberForQmc * a_size;
  const float4 lensOffs     = rndLens(&gen, 0, mutateScale, 
                                      a_globals->rmQMC, qmcPos, a_qmcTable);
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
                                         __global float4*              restrict out_color, 
                                         __global int*                 restrict out_packXY,

                                         int w, int h, int a_size,
                                         __global const EngineGlobals* restrict a_globals, 
                                         __global const int2*          restrict in_zind,
                                         __global const float*         restrict in_samples,
                                         __constant ushort*            restrict a_mortonTable256,
                                         __constant unsigned int*      restrict a_qmcTable, 
                                         int a_passNumberForQmc, int a_packIndexForCPU)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;
  
  // (1) generate 4 random floats
  // we pack lensOffs in this buffer in 'MakeEyeRaysSamplesOnly' kernel (or other)
  //
  const int2 sortedIndex = in_zind[tid];

  float4 lensOffs;
  __global const float* pData = (in_samples + kmltBlobSize(a_globals)*sortedIndex.y); 
  lensOffs.x = pData[0];
  lensOffs.y = pData[1];
  lensOffs.z = pData[2];
  lensOffs.w = pData[3];

  // (2) generate random camera sample
  //
  float  fx, fy;
  float3 ray_pos, ray_dir;
  MakeEyeRayFromF4Rnd(lensOffs, a_globals,
                      &ray_pos, &ray_dir, &fx, &fy);

  int x = (int)(fx); // - 0.5f as were before !!!
  int y = (int)(fy); // - 0.5f as were before !!!

  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;

  if (x < 0)  x = 0;
  if (y < 0)  y = 0;

  out_pos   [tid] = to_float4(ray_pos, fx);
  out_dir   [tid] = to_float4(ray_dir, fy);
  out_color [tid] = make_float4(0,0,0,0);
  out_packXY[tid] = packXY1616(x, y);
}


__kernel void TakeHostRays(__global float4*             restrict out_pos, 
                           __global float4*             restrict out_dir, 
                           __global float4*             restrict out_color,
                           __global int*                restrict out_packXY,
                           __global RandomGen*          restrict out_gens,
                           int w, int h, int a_size,
                           __global const EngineGlobals* restrict a_globals, 
                           __global const float4*        restrict in_raysFromHost)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;
   
  __global const float4* rays1 = in_raysFromHost;
  __global const float4* rays2 = in_raysFromHost + a_size;
  
  const float4 rayPosAndXY = rays1[tid];
  float3 ray_pos  = to_float3(rayPosAndXY);
  float3 ray_dir  = to_float3(rays2[tid]);
  const int packedIndex = as_int(rayPosAndXY.w);
  
  int x = (packedIndex & 0x0000FFFF);         ///<! extract x position from color.w
  int y = (packedIndex & 0xFFFF0000) >> 16;   ///<! extract y position from color.w
  
  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;
  
  if (x < 0)  x = 0;
  if (y < 0)  y = 0;
  
  const float4x4 a_mWorldViewInv = make_float4x4(a_globals->mWorldViewInverse);
  matrix4x4f_mult_ray3(a_mWorldViewInv, &ray_pos, &ray_dir);
  
  //RandomGen gen             = out_gens[tid];
  //const float4 lensOffs     = rndFloat4_Pseudo(&gen);
  //out_gens[tid]             = gen;
  //float  fx, fy;
  //float3 ray_pos, ray_dir;
  //MakeEyeRayFromF4Rnd(lensOffs, a_globals,
  //                    &ray_pos, &ray_dir, &fx, &fy);
  //int x = (int)(fx);
  //int y = (int)(fy); // //

  out_pos   [tid] = to_float4(ray_pos, (float)(x));
  out_dir   [tid] = to_float4(ray_dir, (float)(y));
  out_color [tid] = make_float4(0,0,0,0);
  out_packXY[tid] = packXY1616(x, y);
}

__kernel void ClearAllInternalTempBuffers(__global uint*      restrict out_flags,
                                          __global float4*    restrict out_color,
                                          __global float4*    restrict out_thoroughput,
                                          __global float4*    restrict out_fog,
                                          __global float4*    restrict out_surfaceHit,
                                          __global PerRayAcc* restrict out_accPdf,
                                          int a_size)

{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  HitMatRef data3;
  data3.m_data    = 0; 
  data3.accumDist = 0.0f;

  out_flags      [tid] = 0;
  out_color      [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  out_thoroughput[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  out_fog        [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  out_accPdf     [tid] = InitialPerParAcc();
  
  WriteSurfaceHitMatId(-1, tid, a_size, 
                       out_surfaceHit);
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

__kernel void HDRToLDRWithScale(const __global float4* in_color, __global uint* out_colorLDR,
                                const float a_gammaInv, const float a_scale, const int w, const int h)
{
  const int x = GLOBAL_ID_X;
  const int y = GLOBAL_ID_Y;

  if (x >= w || y >= h)
    return;

  const float4 newColor = in_color[Index2D(x, y, w)];

  float4 color2;
  color2.x = pow(a_scale*newColor.x, a_gammaInv);
  color2.y = pow(a_scale*newColor.y, a_gammaInv);
  color2.z = pow(a_scale*newColor.z, a_gammaInv);
  color2.w = pow(a_scale*newColor.w, a_gammaInv);
  out_colorLDR[Index2D(x, y, w)] = RealColorToUint32(ToneMapping4(color2));
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


__kernel void CountNumLiveThreads(__global int* a_counter, __global uint* a_flags, int iNumElements)
{ 
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  __local int sArray[256];

  // reduce all thisRayIsDead to shared allBlockRaysAreDead
  //
  sArray[LOCAL_ID_X] = (unpackRayFlags(a_flags[tid]) & RAY_IS_DEAD) ? 0 : 1;
  SYNCTHREADS_LOCAL;

  for (uint c = 256 / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] += sArray[LOCAL_ID_X + c];
    SYNCTHREADS_LOCAL;
  }

  if (LOCAL_ID_X == 0)
    atomic_add(a_counter, sArray[0]);
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

__kernel void GetShadowToAlpha(__global float4* inout_data, __global const uchar* in_shadow, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float4 color = inout_data[tid];
  float shadow = (float)(in_shadow[tid])/255.0f;
  color.w = shadow;
  inout_data[tid] = color;
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


__kernel void ClampFloat4(__global float4* inout_color, float a_min, float a_max, int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;
  
  float4 color = inout_color[tid];
  
  color.x = fmax(fmin(color.x, a_max), a_min);
  color.y = fmax(fmin(color.y, a_max), a_min);
  color.z = fmax(fmin(color.z, a_max), a_min);

  inout_color[tid] = color;
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

__kernel void AccumColor3f(__global const float4* in_color, __global float4* out_color, int iNumElements, float a_mult)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;
  
  const float4 incomeColor = in_color[tid];
  float4       currColor   = out_color[tid];

  currColor.x += incomeColor.x*a_mult;
  currColor.y += incomeColor.y*a_mult;
  currColor.z += incomeColor.z*a_mult;
  currColor.w = incomeColor.w;

  out_color[tid] = currColor;
}


// change 21.02.2022 18:00;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


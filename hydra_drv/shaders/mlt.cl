/**
 \file
 \brief Metropolis Light Transport kernels.

 */

#include "globals.h"
#include "crandom.h"
#include "cfetch.h"

/**
\brief Evaluate contib function. Not used.
\param in_color   - input color
\param out_colors - output contrib value

*/
__kernel void MLTEvalContribFunc(__global const float4* in_color, __global float* out_colors, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_colors[tid] = contribFunc(to_float3(in_color[tid]));
}

/**
\brief contribute rays to screen; use atomics; this kernel is not used currently;
*/
__kernel void MLTContribToScreenAtomics(__global const float4* xColor,  __global const float4* yColor, 
                                        __global const float*  xVector, __global const RandomGen* old_gens,
                                        __global float4* outColor, __global const int*  a_qmcPos, __constant unsigned int* a_qmcTable,
                                        __constant ushort* a_mortonTable, int a_width, int a_height, int iNumElements,
                                        __global const EngineGlobals* a_globals) // run it for full screen wxh
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  RandomGen oldGen  = old_gens[tid];
  oldGen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  __global const float* qmcVec = xVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  const unsigned int qmcPos = (unsigned int)a_qmcPos[tid];

  const float screenScaleX = a_globals->varsF[HRT_MLT_SCREEN_SCALE_X];
  const float screenScaleY = a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y];

  const float4 oldRands = make_float4(0,0,0,0); // rndLensGroupOld(&oldGen, qmcVec);
  const float4 newRands = make_float4(0,0,0,0); // rndLensGroupNew(&oldGen, qmcVec, qmcPos, a_qmcTable, make_float2(screenScaleX, screenScaleY));
  
  const float plarge = a_globals->varsF[HRT_MLT_PLARGE];
  const float b      = a_globals->varsF[HRT_MLT_BKELEMEN];
  const bool enableKelemenMIS = false; // (b > 1e-5f);

  const float x_x0 = oldRands.x;
  const float x_x1 = oldRands.y;
  const float y_x0 = newRands.x;
  const float y_x1 = newRands.y;
  
  const int x1 = (int)(x_x0*(float)a_width);
  const int y1 = (int)(x_x1*(float)a_height);
  const int x2 = (int)(y_x0*(float)a_width);
  const int y2 = (int)(y_x1*(float)a_height);

  const float3 colorOld = to_float3(xColor[tid]);
  const float3 colorNew = to_float3(yColor[tid]);

  const float  yOld = contribFunc(colorOld);
  const float  yNew = contribFunc(colorNew);

  const float  a = (yOld == 0.0f) ? 1.0f : fmin(1.0f, yNew / yOld);

  float3 contribOld = colorOld*(1.0f - a)*(1.0f / fmax(yOld, 1e-6f));
  float3 contribNew = colorNew*a*(1.0f / fmax(yNew, 1e-6f));

  // if (enableKelemenMIS)
  // {
  //   const float p_mltOld = yOld / b;
  //   const float p_mltNew = yNew / b;
  //   const float p_pt     = 1.0f;
  // 
  //   contribOld = contribOld*(p_mltOld) / (p_mltOld + p_pt);
  //   contribNew = contribNew*(p_mltNew) / (p_mltNew + p_pt);
  // 
  //   if (largeStep)
  //     contribNew += (colorNew / plarge) * ((p_pt) / (p_mltNew + p_pt));
  // }

  const int pxOld = IndexZBlock2D(x1, y1, a_width, a_mortonTable);
  const int pxNew = IndexZBlock2D(x2, y2, a_width, a_mortonTable);

  __global float* ptr1 = (__global float*)(outColor + pxOld);
  __global float* ptr2 = (__global float*)(outColor + pxNew);

  if (!isfinite(contribOld.x)) contribOld.x = 0.0f;
  if (!isfinite(contribOld.y)) contribOld.y = 0.0f;
  if (!isfinite(contribOld.z)) contribOld.z = 0.0f;

  if (!isfinite(contribNew.x)) contribNew.x = 0.0f;
  if (!isfinite(contribNew.y)) contribNew.y = 0.0f;
  if (!isfinite(contribNew.z)) contribNew.z = 0.0f;

  atomic_addf(ptr1 + 0, contribOld.x);
  atomic_addf(ptr1 + 1, contribOld.y);
  atomic_addf(ptr1 + 2, contribOld.z);

  atomic_addf(ptr2 + 0, contribNew.x);
  atomic_addf(ptr2 + 1, contribNew.y);
  atomic_addf(ptr2 + 2, contribNew.z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


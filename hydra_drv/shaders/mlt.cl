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
\brief Generate eye rays from 4 first random values
\param out_pos   - out   ray position
\param out_dir   - out   ray direction
\param out_gens  - inout generator state; this is common/primary generator used for path generation from unit hypercube

\param a_flags   - out ray flags;  do initial clear/memset of flags;
\param out_color - out path color; do initial clear/memset of out_color;
\param out_thoroughput - out path thoroughput; do initial clear/memset of out_thoroughput;
\param out_fog   - out fog; do initial clear/memset of out_fog;

\param in_qmcSorted - in pair of <newId,oldId> used to get old index of data sorted by newId
\param in_pssVector - in vector of random variables (in unit hypercube).
\param a_qmcPos     - in ... not used
\param a_qmcTable   - in ... not used
\param a_width      - in screen width;
\param a_height     - in screen height;
\param a_seed       - in ... not used

\param a_mutateMode - in proposal mode; have to be one of (MUTATE_LAZY_YES, MUTATE_LAZY_NO); note that MUTATE_LAZY_LARGE is not used. burning-in phase use MUTATE_LAZY_NO;
\param iNumElements - in num threads
\param a_globals    - in global engine variables

*/

__kernel void MLTMakeEyeRaysFromPrimeSpaceSample(__global float4* out_pos, 
                                                 __global float4* out_dir, 
                                                 __global RandomGen* out_gens,

                                                 __global uint*           a_flags,
                                                 __global float4*         out_color,
                                                 __global float4*         out_thoroughput,
                                                 __global float4*         out_fog,
                                                 __global const int2*     in_qmcSorted,
                                                 __global const float*    in_pssVector, 
                                                 __global const int*      a_qmcPos, 
                                                 __constant unsigned int* a_qmcTable,
                                                 int a_width, int a_height, int a_seed, int a_mutateMode, int iNumElements, __global const EngineGlobals* a_globals)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

#ifdef MLT_MULTY_PROPOSAL
  __global const float* qmcVec = in_pssVector + (tid/MLT_PROPOSALS)*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
#else

  #ifndef MLT_SORT_RAYS
  __global const float* qmcVec = in_pssVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #else
  __global const float* qmcVec = in_pssVector + in_qmcSorted[tid].y*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #endif

#endif

  RandomGen gen  = out_gens[tid];
  gen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  if(a_mutateMode == MUTATE_LAZY_YES)
    gen.lazy = (rndFloat1_Pseudo(&gen) < a_globals->varsF[HRT_MLT_PLARGE]) ? MUTATE_LAZY_LARGE : MUTATE_LAZY_YES;
  else
    gen.lazy = MUTATE_LAZY_NO; // for burn in

  // get back this for QMC
  //
  // unsigned int qmcPos = (unsigned int)a_qmcPos[tid];
  // __constant unsigned int* table = a_qmcTable;

  const float screenScaleX = a_globals->varsF[HRT_MLT_SCREEN_SCALE_X];
  const float screenScaleY = a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y];

  const float4 newRands = rndLens(&gen, qmcVec, make_float2(screenScaleX, screenScaleY), 0, 0);

  const float xPosPs = newRands.x;
  const float yPosPs = newRands.y;

  const float x      = a_width*xPosPs  - 0.5f;
  const float y      = a_height*yPosPs - 0.5f;

  // no dof, add later

  const float4x4 a_mViewProjInv  = make_float4x4(a_globals->mProjInverse);
  const float4x4 a_mWorldViewInv = make_float4x4(a_globals->mWorldViewInverse);

  float3 ray_pos = make_float3(0.0f, 0.0f, 0.0f);
  float3 ray_dir = EyeRayDir(x, y, a_width, a_height, a_mViewProjInv);

  ray_dir        = tiltCorrection(ray_pos, ray_dir, a_globals);


  if (a_globals->varsI[HRT_ENABLE_DOF] == 1)
  {
    float  tFocus        = a_globals->varsF[HRT_DOF_FOCAL_PLANE_DIST] / (-ray_dir.z);
    float3 focusPosition = ray_pos + ray_dir*tFocus;
    float2 xy = a_globals->varsF[HRT_DOF_LENS_RADIUS] * MapSamplesToDisc(2.0f*make_float2(newRands.z, newRands.w) - make_float2(1.0f,1.0f));
    ray_pos.x += xy.x;
    ray_pos.y += xy.y;
    ray_dir = normalize(focusPosition - ray_pos);
  }

  matrix4x4f_mult_ray3(a_mWorldViewInv, &ray_pos, &ray_dir);

  out_gens[tid]        = gen;
  out_pos [tid]        = to_float4(ray_pos, 1.0f);
  out_dir [tid]        = to_float4(ray_dir, 0.0f);

  a_flags        [tid] = 0;
  out_color      [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  out_thoroughput[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  out_fog        [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}



#ifdef COMPRESS_RAND

/**
\brief Make MCMC proposal (not lazy). Y ~ makeProposal(X);
\param gen       - inout primary random gen (for rays/paths)
\param yVecOut   - out new point coordinate in unit hypercube y[0..N]; yVecOut can be the same as xVecIn;
\param xVecIn    - in  old point coordinate in unit hypercube x[0..N]  xVecIn  can be the same as yVecOut;
\param forceLargeStep - in this parameter force us to make large step.

\param a_globals - in global engine variables


*/

IDH_CALL void MakeProposal(__private RandomGen* gen, __global float* yVecOut, __global float* xVecIn, bool forceLargeStep, __global const EngineGlobals* a_globals)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  float rlarge = 0.0f;
  if (!forceLargeStep)
    rlarge = rndFloat1_Pseudo(gen);

  const float screenScaleX = a_globals->varsF[HRT_MLT_SCREEN_SCALE_X];
  const float screenScaleY = a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y];
  const float2 screenScale = make_float2(screenScaleX, screenScaleY);

  if (rlarge <= a_globals->varsF[HRT_MLT_PLARGE])
    gen->lazy = MUTATE_LAZY_LARGE;
  else
    gen->lazy = MUTATE_LAZY_YES;

  __global uint4* rptrIn  = (__global uint4*)xVecIn;
  __global uint4* rptrOut = (__global uint4*)yVecOut;

  if(gen->lazy == MUTATE_LAZY_LARGE)
  {
    for (int bounceId = 0; bounceId < MLT_MAX_BOUNCE; bounceId++)
    {
      float6_gr group1;
      float6_gr group2;
      
      group1.group24   = rndFloat4_Pseudo(gen);

      group2.group24.w =                        rndFloat1_Pseudo(gen);
      group2.group16.x = (unsigned short)(clamp(rndFloat1_Pseudo(gen)*65535.0f, 0.0f, 65535.0f));
      group2.group16.y = (unsigned short)(clamp(rndFloat1_Pseudo(gen)*65535.0f, 0.0f, 65535.0f));
      group1.group16.x = (unsigned short)(clamp(rndFloat1_Pseudo(gen)*65535.0f, 0.0f, 65535.0f));
      group1.group16.y = (unsigned short)(clamp(rndFloat1_Pseudo(gen)*65535.0f, 0.0f, 65535.0f));

      const float4 randsMat = rndFloat4_Pseudo(gen);

      group2.group24.x = randsMat.x; 
      group2.group24.y = randsMat.y;
      group2.group24.z = randsMat.z;

      rptrOut[rndLightOffset(bounceId)] = packBounceGroup(group1);
      rptrOut[rndMatOffset(bounceId)]   = packBounceGroup(group2);
    }
  }
  else
  {
    for (int bounceId = 0; bounceId < MLT_MAX_BOUNCE; bounceId++)
    {
      float6_gr group1 = unpackBounceGroup(rptrIn[rndLightOffset(bounceId)]);
      float6_gr group2 = unpackBounceGroup(rptrIn[rndMatOffset(bounceId)]);

      if(bounceId == 0)
      {
        group1.group24.x = MutateKelemen(group1.group24.x, gen, MUTATE_COEFF_SCREEN*screenScale.x);
        group1.group24.y = MutateKelemen(group1.group24.y, gen, MUTATE_COEFF_SCREEN*screenScale.y);
      }
      else
      {
        group1.group24.x = MutateKelemen(group1.group24.x, gen, MUTATE_COEFF_BSDF);
        group1.group24.y = MutateKelemen(group1.group24.y, gen, MUTATE_COEFF_BSDF);
      }

      group1.group24.z = MutateKelemen(group1.group24.z, gen, MUTATE_COEFF_BSDF);
      group1.group24.w = MutateKelemen(group1.group24.w, gen, MUTATE_COEFF_BSDF);

      group2.group24.x = MutateKelemen(group2.group24.x, gen, MUTATE_COEFF_BSDF);
      group2.group24.y = MutateKelemen(group2.group24.y, gen, MUTATE_COEFF_BSDF);
      group2.group24.z = MutateKelemen(group2.group24.z, gen, MUTATE_COEFF_BSDF);

      rptrOut[rndLightOffset(bounceId)] = packBounceGroup(group1);
      rptrOut[rndMatOffset(bounceId)]   = packBounceGroup(group2);
    }
  }

  gen->lazy = MUTATE_LAZY_NO;

}

#else

/**
\brief Make MCMC proposal (not lazy). Y ~ makeProposal(X);
\param gen       - inout primary random gen (for rays/paths)
\param yVecOut   - out new point coordinate in unit hypercube y[0..N]; yVecOut can be the same as xVecIn;
\param xVecIn    - in  old point coordinate in unit hypercube x[0..N]; xVecIn  can be the same as yVecOut;
\param forceLargeStep - in this parameter force us to make large step.

\param a_globals - in global engine variables

*/

IDH_CALL void MakeProposal(__private RandomGen* gen, __global float* yVecOut, __global float* xVecIn, bool forceLargeStep, __global const EngineGlobals* a_globals)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  float rlarge = 0.0f;
  if (!forceLargeStep)
    rlarge = rndFloat1_Pseudo(gen);

  const float screenScaleX     = a_globals->varsF[HRT_MLT_SCREEN_SCALE_X];
  const float screenScaleY     = a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y];
  const float2 lensMutateCoeff = make_float2(screenScaleX, screenScaleY);

  if (rlarge <= a_globals->varsF[HRT_MLT_PLARGE])
    gen->lazy = MUTATE_LAZY_LARGE;
  else
    gen->lazy = MUTATE_LAZY_YES;

  for (int bounceId = 0; bounceId < MLT_MAX_BOUNCE; bounceId++)
  {
    const int lightOffset = rndLightOffset(bounceId);
    const int matOffset   = rndMatOffset(bounceId);
    const int matLOffset  = rndMatLOffset(bounceId);

    float4 l_i = make_float4(0,0,0,0); 

    if(bounceId == 0)
      l_i = rndLens(gen, xVecIn, lensMutateCoeff, 0, 0);
    else 
      l_i = rndLight(gen, xVecIn, bounceId);

    yVecOut[lightOffset + 0] = l_i.x;
    yVecOut[lightOffset + 1] = l_i.y;
    yVecOut[lightOffset + 2] = l_i.z;
    yVecOut[lightOffset + 3] = l_i.w;

    for (int i = 0; i < MLT_FLOATS_PER_MLAYER; i++)
      yVecOut[matLOffset + i] = rndMatLayer(gen, xVecIn, bounceId, i);

    const float3 m_i = rndMat(gen, xVecIn, bounceId);

    yVecOut[matOffset + 0]   = m_i.x;
    yVecOut[matOffset + 1]   = m_i.y;
    yVecOut[matOffset + 2]   = m_i.z;
  }

  gen->lazy = MUTATE_LAZY_NO;

}

#endif

/**
\brief Kernel that make MCMC proposal (not lazy). Y ~ makeProposal(X);
\param xVector   - in old point coordinate in unit hypercube
\param yVector   - in old point coordinate in unit hypercube
\param out_gens  - input primary/rays generator
\param a_forceLargeStep - force large step probability (for burning in).
\param a_globals    - in global engine variables
\param iNumElements - in threads num

*/
__kernel void MLTMakeProposal(__global float* xVector, __global float* yVector, __global RandomGen* out_gens, int a_forceLargeStep, __global const EngineGlobals* a_globals, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  __global float* xVecIn  = xVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  __global float* yVecOut = yVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  RandomGen gen  = out_gens[tid];
  gen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  const bool forceLargeStep = (a_forceLargeStep == 1);

  MakeProposal(&gen, yVecOut, xVecIn, forceLargeStep, a_globals);

  out_gens[tid] = gen;
}

/**
\brief Kernel that generate next quiasi monte carlo sequence. Not used.

*/
__kernel void MLTEvalQMCLargeStepIndex(__global RandomGen* out_gens, __global unsigned int* out_positions, __global unsigned int* pCounter, __global const EngineGlobals* a_globals, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  RandomGen gen = out_gens[tid];

  // int largestep = (rndFloat1_Pseudo(&gen) < a_globals->varsF[HRT_MLT_PLARGE]) ? MUTATE_LAZY_LARGE : MUTATE_LAZY_YES;
  // 
  // if (largestep == MUTATE_LAZY_LARGE)
  //   out_positions[tid] = atomic_add(pCounter, (unsigned int)1);
  // else
  //   out_positions[tid] = 0;

}

/**
\brief Kernel fot testing sobol. Not used.

*/
__kernel void MLTTestSobolQMC(__global unsigned int* in_positions, __global float* out_vals, __constant unsigned int* a_qmcTable, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  // const unsigned int qmcPos = (unsigned int)a_qmcPos[tid];
  const unsigned int qmcPos = (unsigned int)tid;
  out_vals[tid] = rndQmcSobolN(qmcPos, 0, a_qmcTable);
}

/**
\brief Make pairs of (newId,oldId) for further sorting. Using pairs of (x,y) and may be ray direction further.
\param a_gens     - random gen for make proposal; just to make correct new ray;
\param xOldNewId  - pairs for old rays (current  Markov chain state)
\param yOldNewId  - pairs for new rays (proposed Markov chain state)
\param xVector    - Primary Space position  (current  Markov chain state)
\param oldXY      - not used; If no MCMC_LAZY, store (x,y) for current  Markov chain state
\param a_qmcPos   - not used;
\param a_qmcTable - not used;
\param a_mortonTable - read only morton tabe used for calculating morton codes
\param a_globals - read only engine globals

\param resW - screen image width
\param resH - screen image height
\param iNumElements - num threads

*/

__kernel void MLTMakeIdPairForSorting(__global const RandomGen* a_gens,        __global int2* xOldNewId, __global int2* yOldNewId,
                                      __global const float*     xVector,       __global const float2* oldXY,
                                      __global const int*       a_qmcPos,      __constant unsigned int* a_qmcTable,
                                      __constant ushort*        a_mortonTable, __global const EngineGlobals* a_globals, float resW, float resH, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float x_x0 = 0.0f;
  float x_x1 = 0.0f;
  float y_x0 = 0.0f;
  float y_x1 = 0.0f;

  #ifdef MLT_MULTY_PROPOSAL
  __global const float* qmcVec = xVector + (tid/MLT_PROPOSALS)*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #else
  __global const float* qmcVec = xVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #endif
  
  RandomGen dummyGen = a_gens[tid];
  #ifdef MCMC_LAZY
  dummyGen.lazy = (rndFloat1_Pseudo(&dummyGen) <= a_globals->varsF[HRT_MLT_PLARGE]) ? MUTATE_LAZY_LARGE : MUTATE_LAZY_YES;
  #else
  dummyGen.lazy = MUTATE_LAZY_NO;
  #endif

  //const unsigned int qmcPos = (unsigned int)a_qmcPos[tid];
  
  const float screenScaleX = a_globals->varsF[HRT_MLT_SCREEN_SCALE_X];
  const float screenScaleY = a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y];

#ifdef MCMC_LAZY
  const float4 oldRands = rndLensOld(qmcVec);
#else
  const float2 oldRands = oldXY[tid];
#endif

  const float4 newRands = rndLens(&dummyGen, qmcVec, make_float2(screenScaleX, screenScaleY), 0, 0);
  
  x_x0 = oldRands.x;
  x_x1 = oldRands.y;
  
  y_x0 = newRands.x;
  y_x1 = newRands.y;
  
  // a_gens[tid] = dummyGen;

  const int x_x0i = (int)(x_x0 * resW);
  const int x_x1i = (int)(x_x1 * resH);

  const int y_x0i = (int)(y_x0 * resW);
  const int y_x1i = (int)(y_x1 * resH);

  const int zIndex1 = ZIndex(x_x0i, x_x1i, a_mortonTable);
  const int zIndex2 = ZIndex(y_x0i, y_x1i, a_mortonTable);

#ifdef MLT_MULTY_PROPOSAL
  if(tid%MLT_PROPOSALS == 0)
    xOldNewId[tid/MLT_PROPOSALS] = make_int2(zIndex1, tid/MLT_PROPOSALS);
  yOldNewId[tid] = make_int2(zIndex2, tid);
#else
  xOldNewId[tid] = make_int2(zIndex1, tid);
  yOldNewId[tid] = make_int2(zIndex2, tid);
#endif
}

__kernel void MLTInitIdPairForSorting(__global int2* xOldNewId, __global int2* yOldNewId, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int2 keyVal = make_int2(tid,tid);
  
  xOldNewId[tid] = keyVal;
  yOldNewId[tid] = keyVal;
}



/**
\brief contribute rays to screen; this kernel gather contribution for each pixel using sorted indices arrays; total thread number is (a_width*a_height);
\param xOldNewId - sorted (newId,oldId) indices array for current  MC state.
\param yOldNewId - sorted (newId,oldId) indices array for proposed MC state.
\param xColor    - unsorted (indexed by oldId) array of colors for current  MC state
\param yColor    - unsorted (indexed by oldId) array of colors for proposed MC state
\param outColor  - color buffer (size of a_width*a_height);
\param a_globals - engine globals
\param a_width   - image width
\param a_height  - image height
\param arraySize - size of xOldNewId and yOldNewId in int2

*/
__kernel void MLTContribToScreen(__global const int2* xOldNewId, __global const int2* yOldNewId, 
                                 __global const float4* xColor,  __global const float4* yColor, __global float4* outColor,
                                 __constant ushort* a_mortonTable, __global const EngineGlobals* a_globals, int a_width, int a_height, int arraySize) // run it for full screen wxh
{
  uint x = GLOBAL_ID_X;
  uint y = GLOBAL_ID_Y;

  int pxRealOffset = IndexZBlock2D(x, y, a_width, a_mortonTable);
  int pxZIndex     = ZIndex(x, y, a_mortonTable);

  const int beginX = binarySearchForLeftRange(xOldNewId,  arraySize, pxZIndex);
  const int endX   = binarySearchForRightRange(xOldNewId, arraySize, pxZIndex);

  const int beginY = binarySearchForLeftRange(yOldNewId,  arraySize, pxZIndex);
  const int endY   = binarySearchForRightRange(yOldNewId, arraySize, pxZIndex);
 
  float4 accum    = outColor[pxRealOffset];
  float4 accumOld = accum;
  
  const float b = a_globals->varsF[HRT_MLT_BKELEMEN];
  const bool enableKelemenMIS = false && (b > 1e-5f);

  // find all (xOldNewId[i].x == pxZIndex), contrib them with (1.0f - a)
  //
  
  if (beginX != -1 && endX != -1)
  {
    for (int i = beginX; i <= endX; i++)
    {
      const int2 xOldNewPair = xOldNewId[i];

      const float3 colorOld  = to_float3(xColor[xOldNewPair.y]);
      const float3 colorNew  = to_float3(yColor[xOldNewPair.y]);

      const float  yOld = contribFunc(colorOld);
      const float  yNew = contribFunc(colorNew);

      const float  a = (yOld == 0.0f) ? 1.0f : fmin(1.0f, yNew / yOld);

      const float3 color = colorOld*(1.0f - a)*(1.0f / fmax(yOld, 1e-6f));

      if (enableKelemenMIS)
      {
        accum += to_float4(color, 0.0f);
      }
      else
      {
        accum += to_float4(color, 0.0f);
      }
    }

  }
  
  // find all (yOldNewId[i].x == pxZIndex), contrib them with a
  //
  if (beginY != -1 && endY != -1)
  {
    for (int i = beginY; i <= endY; i++)
    {
      const int2 yOldNewPair = yOldNewId[i];

      const float3 colorOld  = to_float3(xColor[yOldNewPair.y]);
      const float3 colorNew  = to_float3(yColor[yOldNewPair.y]);

      const float  yOld = contribFunc(colorOld);
      const float  yNew = contribFunc(colorNew);

      const float  a = (yOld == 0.0f) ? 1.0f : fmin(1.0f, yNew / yOld);

      const float3 color = colorNew*a*(1.0f / fmax(yNew, 1e-6f));

      if (enableKelemenMIS)
      {
        accum += to_float4(color, 0.0f);
      }
      else
      {
        accum += to_float4(color, 0.0f);
      }
    }
  }
  
  if (!isfinite(accum.x)) accum.x = accumOld.x;
  if (!isfinite(accum.y)) accum.y = accumOld.y;
  if (!isfinite(accum.z)) accum.z = accumOld.z;
  if (!isfinite(accum.w)) accum.w = accumOld.w;

  outColor[pxRealOffset] = accum;

}

/**
\brief contribute rays to screen; use atomics; this kernel is not used currently;


*/

__kernel void MLTContribToScreenAtomics(__global const float4* xColor,  __global const float4* yColor, 
                                        __global const float*  xVector, __global const RandomGen* old_gens,
                                        __global float4* outColor, __global const int*  a_qmcPos, __constant unsigned int* a_qmcTable,
                                        __constant ushort* a_mortonTable, int a_width, int a_height, int iNumElements,
                                        __global const int2* a_qmcSorted, __global const EngineGlobals* a_globals) // run it for full screen wxh
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  RandomGen oldGen  = old_gens[tid];
  oldGen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  
  #ifdef MLT_MULTY_PROPOSAL
  __global const float* qmcVec = xVector + (tid / MLT_PROPOSALS)*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #else
  __global const float* qmcVec = xVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #endif

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


/**
\brief perfrom Accept/Reject for each thread
\param mltAR_gens - separate random generator used for decisiof for Accept/Reject
\param gens_old   - common random generator used for make proposal

\param xVector - current  MC state
\param yVector - next     MC state (not used if MCMC_LAZY)

\param xColor  - color for current  MC state
\param yColor  - color for proposed MC state

\param a_qmcSorted - not used
\param a_qmcPos    - not used
\param a_qmcTable  - not used

\param a_globals    - engine globals
\param iNumElements - thread number

*/
__kernel void MLTAcceptReject(__global RandomGen* mltAR_gens, __global RandomGen* gens_old, 
                              __global float*  xVector, __global const float*  yVector,
                              __global float4* xColor,  __global const float4* yColor, 
                              __global const int2* a_qmcSorted, __global const int* a_qmcPos, __constant unsigned int* a_qmcTable,
                              __global const EngineGlobals* a_globals, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const float3 colorOld = to_float3(xColor[tid]);
  const float3 colorNew = to_float3(yColor[tid]);

  const float  yOld = contribFunc(colorOld);
  const float  yNew = contribFunc(colorNew);

  const float  a = (yOld == 0.0f) ? 1.0f : fmin(1.0f, yNew / yOld);

  RandomGen genm  = mltAR_gens[tid];
  const float p   = rndFloat1_Pseudo(&genm);
  mltAR_gens[tid] = genm;

  __global       float* xVec = xVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  __global const float* yVec = yVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  //const unsigned int qmcPos = (unsigned int)a_qmcPos[tid];

#ifdef MCMC_LAZY
  
  RandomGen genOld = gens_old[tid];

  if (p <= a) // accept //
  {
    MakeProposal(&genOld, xVec, xVec, false, a_globals);
    xColor[tid] = yColor[tid];
  }


#else

  if (p <= a) // accept //
  {
    for (int i = 0; i < a_globals->varsI[HRT_MLT_MAX_NUMBERS]; i++)
      xVec[i] = yVec[i];

    xColor[tid] = yColor[tid];
  }

#endif

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IDH_CALL float2 readInterval(__global const float* intervals, int a_offset, int a_size)
{
  if (a_offset >= a_size)
    return make_float2(intervals[a_size - 2], intervals[a_size - 1]);
  else if (a_offset <= 0)
    return  make_float2(0.0f, intervals[0]);
  else
    return make_float2(intervals[a_offset - 1], intervals[a_offset]);
}

IDH_CALL int binarySearch2(float x, __global const float* intervals, int a_size)
{
  int leftBound  = 0;
  int rightBound = a_size - 1;

  int currPos = -1;
  int counter = 0;

  int maxStep = 100; // (int)(log((float)a_size) / log(2.0f)) + 10;

  while (rightBound - leftBound > 4 && counter < maxStep)
  {
    int currPos1 = (rightBound + leftBound + 1) / 2;

    float2 interval1 = readInterval(intervals, currPos1, a_size);

    if (interval1.x < x && x <= interval1.y)
    {
      currPos = currPos1;
      break;
    }

    if (x <= interval1.x)
      rightBound = currPos1;
    else if (x > interval1.y)
      leftBound = currPos1;

    counter++;
  }

  if (currPos >= 0)
    return currPos;


  for (int i = leftBound; i <= rightBound; i++)
  {
    float2 interval1 = readInterval(intervals, i, a_size);
    if (interval1.x < x && x <= interval1.y)
      currPos = i;
  }

  if (currPos < 0)
    currPos = (rightBound + leftBound + 1) / 2;

  return currPos;
}


/**
\brief Select Markov Chains proportional to their contribution; Used for Burning-In process;

\param out_gens   - tempoprary random generator used just to get random variable to make selection further;
\param offset     - offset to out_gens2 (!!!); may be omit for out_gens actually; used because thica kernell is called several times; Each time with different offset;
\param old_gens   - this is the arrays of raamdom gens from which we select gens proportional to contributions
\param out_gens2  - this is the actual result; kernel select random generato states to regenerate actual MC state (long vector) further;
\param samplesLum - prefix summed contribution array
\param arraySize  - size of prefix summed contribution array
\param a_globals  - engine globals

\param a_qmcTable - not used
\param a_seed     - not used
\param iNumElements - threads number

*/
__kernel void MLTSelectSampleProportionalToContrib(__global RandomGen* out_gens, int offset, 
                                                   __global RandomGen* old_gens, __global RandomGen* out_gens2, __global const float* samplesLum, int arraySize,
                                                   __global const EngineGlobals* a_globals, __constant unsigned int* a_qmcTable, unsigned int a_seed, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  RandomGen gen = out_gens[offset + tid];

  const float p = rndFloat1_Pseudo(&gen);
  const float x = fmin(p*samplesLum[arraySize - 1], (float)arraySize);

  out_gens[offset + tid] = gen;

  int foundOffset = binarySearch2(x, samplesLum, arraySize);
  if (foundOffset < 0)
    foundOffset = 0;
  if (foundOffset > arraySize-1)
    foundOffset = arraySize-1;

  out_gens2[offset + tid] = old_gens[foundOffset]; // got it
}


__kernel void MLTContribToScreenDebug(__global const int2* xOldNewId, __global const float4* xColor, __global float4* outColor,
                                      __constant ushort* a_mortonTable, int a_width, int a_height, int arraySize) // run it for full screen wxh
{
  uint x = GLOBAL_ID_X;
  uint y = GLOBAL_ID_Y;

  int pxRealOffset = IndexZBlock2D(x, y, a_width, a_mortonTable);
  int pxZIndex     = ZIndex(x, y, a_mortonTable);

  const int beginX = binarySearchForLeftRange(xOldNewId,  arraySize, pxZIndex);
  const int endX   = binarySearchForRightRange(xOldNewId, arraySize, pxZIndex);
 
  float4 accum = outColor[pxRealOffset];

  // find all (xOldNewId[i].x == pxZIndex), contrib them with (1.0f - a)
  //
  
  if (beginX != -1 && endX != -1)
  {
    for (int i = beginX; i <= endX; i++)
    {
      const int2   xOldNewPair = xOldNewId[i];
      const float4 colorLumOld = xColor[xOldNewPair.y];

      accum += colorLumOld;
    }

  }

  outColor[pxRealOffset] = accum;

}

__kernel void MLTMoveRandStateByIndex(__global RandomGen* out_gens, __global const RandomGen* in_gens, 
                                      __global int2* out_qpos, __global const int2* in_qpos,
                                      __global const int2* a_indices, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int offset = a_indices[tid].y;

  out_gens[tid] = in_gens[offset];
  //out_qpos[tid] = in_qpos[offset];
}

__kernel void MLTMoveColorByIndexBack(__global float4* out_vals, __global const float4* in_vals, __global const int2* a_indices, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_vals[a_indices[tid].y] = in_vals[tid];
}

__kernel void MLTStoreOldXY(__global float2* out_vals, __global const float* xVector, __global const EngineGlobals* a_globals, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  #ifdef MLT_MULTY_PROPOSAL
  __global const float* qmcVec = xVector + (tid / MLT_PROPOSALS)*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #else
  __global const float* qmcVec = xVector + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #endif

  const float4 lensRnds = rndLensOld(qmcVec);

  out_vals[tid] = make_float2(lensRnds.x, lensRnds.y);
}

// change 08.11.2017 19:46;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/**
 \file
 \brief Metropolis Light Transport kernels.

 */

#include "cglobals.h"
#include "crandom.h"
#include "cfetch.h"

#include "clight.h"
#include "cmaterial.h"
#include "cbidir.h"


/**
\brief Evaluate contib function and average brightess per bounce.
\param in_color   - input color
\param out_colors - output contrib value

*/
__kernel void MMLTEvalContribFunc(__global const float4* restrict in_color,
                                  __global const int2*   restrict in_split,
                                  __global float*        restrict out_colors,
                                  __global float*        restrict out_avgb, 
                                  int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const float val = contribFunc(to_float3(in_color[tid]));
  out_colors[tid] = val;

  if(out_avgb != 0)
  {
    const int d = in_split[tid].x;
    atomic_addf(out_avgb + d, val);
  }

}

/**
\brief Init split buffer with selected depth

*/
__kernel void MMLTCopySelectedDepthToSplit(__global const int* restrict in_split, __global int2* restrict out_split, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int val  = in_split[tid];
  out_split[tid] = make_int2(val,val);
}

/**
\brief Select Markov Chains proportional to their contribution; Used for Burning-In process;

\param out_gens  - output random gen state
\param out_depth - output contrib value
\param offset    - input offset inside (out_gens, out_depth) buffers

\param in_gens   - input random generator state
\param in_split  - input split (d,s) pair; only d is used -- i.e. split.x

\param a_gens_select - random generator used to select contribution (must be different than in_gens!!!)

*/
__kernel void MMLTSelectSampleProportionalToContrib(__global RandomGen*       restrict out_gens,
                                                    __global int*             restrict out_depth, 
                                                    int offset, 
                                                    __global const RandomGen* restrict in_gens,
                                                    __global const int2*      restrict in_split,
                                                    
                                                    __global       RandomGen* restrict a_gens_select,  
                                                    __global const float*     restrict in_samplesLum, 
                                                    int arraySize,
                                                    int iNumElements)

{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  RandomGen gen      = a_gens_select[tid];
  const float r0     = rndFloat1_Pseudo(&gen);
  a_gens_select[tid] = gen;

  float pdf = 1.0f;
  const int foundIndex = SelectIndexPropToOpt(r0, in_samplesLum, arraySize-1, &pdf);

  out_gens [offset + tid] = in_gens [foundIndex];
  out_depth[offset + tid] = in_split[foundIndex].x;
}


__kernel void MMLTMakeStatesIndexToSort(__global const RandomGen* restrict in_gens,
                                        __global const int*       restrict in_depth,
                                        __global       int2*      restrict out_index,
                                        const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int d = in_depth[tid];
  const RandomGen gen = in_gens[tid];
  
  //#TODO: add lense (x,y) to the index
  out_index[tid] = make_int2(-d, tid);
}

__kernel void MMLTMoveStatesByIndex(__global const int2*      restrict in_index,
                                    __global const RandomGen* restrict in_gens,
                                    __global const int*       restrict in_depth,

                                    __global       RandomGen* restrict out_gens,
                                    __global       int*       restrict out_depth,
                                    __global       int2*      restrict out_split,
                                    const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int2 index = in_index[tid];

  const int d    = in_depth[index.y];
  out_gens [tid] = in_gens [index.y];
  out_depth[tid] = d;
  if(out_split != 0)
    out_split[tid] = make_int2(d,d);
}


inline int TabIndex(const int vertId, const int tid, const int iNumElements)
{
  return tid + vertId*iNumElements;
}

__kernel void MMLTAcceptReject(__global       float*         restrict a_xVector,
                               __global const float*         restrict a_yVector,
                               __global       float4*        restrict a_xColor,
                               __global const float4*        restrict a_yColor, 
                               
                               __global   RandomGen*         restrict a_gensAR,
                               __global       float4*        restrict out_xAlpha,
                               __global       float4*        restrict out_yAlpha,
                               const int a_maxBounce,
                               const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  RandomGen gen = a_gensAR[tid];
  const float p = rndFloat1_Pseudo(&gen);
  a_gensAR[tid] = gen;
  
  const float4 yOldColor = a_xColor[tid];
  const float4 yNewColor = a_yColor[tid];

  const float yOld = contribFunc( to_float3(yOldColor) );
  const float yNew = contribFunc( to_float3(yNewColor) );

  const float a = (yOld == 0.0f) ? 1.0f : fmin(1.0f, yNew / yOld);

  float4 contribAtX, contribAtY;
  contribAtX.xyz = yOldColor.xyz*(1.0f / fmax(yOld, 1e-6f))*(1.0f - a);  contribAtX.w = yOldColor.w;
  contribAtY.xyz = yNewColor.xyz*(1.0f / fmax(yNew, 1e-6f))*a;           contribAtY.w = yNewColor.w;

  out_xAlpha[tid] = contribAtX;
  out_yAlpha[tid] = contribAtY;

  if (p <= a) // accept 
  {
    a_xColor[tid] = yNewColor;
    
    for(int i=0;i<MMLT_HEAD_TOTAL_SIZE;i++)
      a_xVector[TabIndex(i, tid, iNumElements)] = a_yVector[TabIndex(i, tid, iNumElements)];
    
    for(int bounce = 0; bounce < a_maxBounce; bounce++)
    {
      const int bounceOffset = MMLT_HEAD_TOTAL_SIZE + MMLT_COMPRESSED_F_PERB*bounce;
      
      a_xVector[TabIndex(bounceOffset + 0, tid, iNumElements)] = a_yVector[TabIndex(bounceOffset + 0, tid, iNumElements)];
      a_xVector[TabIndex(bounceOffset + 1, tid, iNumElements)] = a_yVector[TabIndex(bounceOffset + 1, tid, iNumElements)];
      a_xVector[TabIndex(bounceOffset + 2, tid, iNumElements)] = a_yVector[TabIndex(bounceOffset + 2, tid, iNumElements)];
      a_xVector[TabIndex(bounceOffset + 3, tid, iNumElements)] = a_yVector[TabIndex(bounceOffset + 3, tid, iNumElements)];
      a_xVector[TabIndex(bounceOffset + 4, tid, iNumElements)] = a_yVector[TabIndex(bounceOffset + 4, tid, iNumElements)];
      a_xVector[TabIndex(bounceOffset + 5, tid, iNumElements)] = a_yVector[TabIndex(bounceOffset + 5, tid, iNumElements)];
    }
  }
  
}


__kernel void MMLTInitCameraPath(__global   uint*      restrict a_flags,
                                 __global float4*      restrict a_color,
                                 __global int2*        restrict a_split,
                                 __global float4*      restrict a_vertexSup,
                                 __global PdfVertex*   restrict a_pdfVert,
                                 const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int2 oldSplit = a_split[tid];
  const int  d        = oldSplit.x; // MMLT_GPU_TEST_DEPTH;

  a_flags[tid] = packBounceNum(0, 1);
  a_color[tid] = make_float4(1,1,1,1);
  
  PathVertex resVertex;
  resVertex.ray_dir      = make_float3(0,0,0); 
  resVertex.accColor     = make_float3(0,0,0);   
  resVertex.valid        = false; //(a_currDepth == a_targetDepth);     // #TODO: dunno if this is correct ... 
  resVertex.hitLight     = false;
  resVertex.wasSpecOnly  = false;
  WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                            a_vertexSup);

  PdfVertex vInitial;
  vInitial.pdfFwd = 1.0f;
  vInitial.pdfRev = 1.0f;
  for(int i=0;i<=d;i++)
    a_pdfVert[TabIndex(i, tid, iNumElements)] = vInitial;
}

__kernel void CopyAccColorTo(__global const float4* restrict in_vertexSup, 
                             __global       float4* restrict out_color,
                             const int   iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  PathVertex resVertex;
  ReadPathVertexSupplement(in_vertexSup, tid, iNumElements,
                           &resVertex);

  if(resVertex.valid && resVertex.hitLight)
    out_color[tid] = to_float4(resVertex.accColor, 0.0f);
  else
    out_color[tid] = make_float4(0,0,0,0);
}

static inline int PickSplitFromRand(const float r, const int d, float a_threshold)
{
  if(a_threshold < 0.1f)
    return mapRndFloatToInt(r, 0, d);
  else
  {
    const float threshold = a_threshold;
    const float r1        = (r - threshold)/(1.0f - threshold);
    
    if(r <= threshold)
      return 0;
    else 
      return mapRndFloatToInt(r1, 0, d);
  }
}


__kernel void MMLTMakeProposal(__global int2*            restrict a_split,
                               __global const RandomGen* restrict in_gens,
                               __global       RandomGen* restrict out_gens,     // save new random gen state here if it is not null
                               __global const float*     restrict in_numbers,
                               __global       float*     restrict out_numbers,  // save random numbers here if it is not null
                               int a_forceLargeStep, int a_maxBounce,
                               __global const EngineGlobals* restrict a_globals,
                               int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  RandomGen gen = in_gens[tid];

  bool largeStep    = false; 
  int smallStepType = 0;
  {
    const float p = rndFloat1_Pseudo(&gen);
    if(a_forceLargeStep != 1)
    {
      if(p <= 0.5f)
        largeStep = true;
      else if(0.5f < p && p <= 0.666667f)
        smallStepType = MUTATE_LIGHT;
      else if (0.666667 < p && p <= 0.85f)
        smallStepType = MUTATE_LIGHT | MUTATE_CAMERA;
      else
        smallStepType = MUTATE_CAMERA;
    }
    else
      largeStep = true;
  }
  // enum MUTATION_TYPE { MUTATE_LIGHT = 1, MUTATE_CAMERA = 2 };

  // gen head first
  //
  
  // lens
  {
    float4 lensOffs; 
    
    if(largeStep)
    {
      lensOffs = rndFloat4_Pseudo(&gen);
    }
    else if(in_numbers != 0)
    {
      const float screenScaleX = a_globals->varsF[HRT_MLT_SCREEN_SCALE_X]; // #NOTE: be sure these variables are not zero !!! 
      const float screenScaleY = a_globals->varsF[HRT_MLT_SCREEN_SCALE_Y]; // #NOTE: be sure these variables are not zero !!! 

      lensOffs.x = in_numbers[ TabIndex(MMLT_DIM_SCR_X, tid, iNumElements) ];
      lensOffs.y = in_numbers[ TabIndex(MMLT_DIM_SCR_Y, tid, iNumElements) ];
      lensOffs.z = in_numbers[ TabIndex(MMLT_DIM_DOF_X, tid, iNumElements) ];
      lensOffs.w = in_numbers[ TabIndex(MMLT_DIM_DOF_Y, tid, iNumElements) ];
      
      if(smallStepType & MUTATE_CAMERA)
      {
        const float power = a_globals->varsF[HRT_MMLT_STEP_SIZE_POWER];
        const float coeff = a_globals->varsF[HRT_MMLT_STEP_SIZE_COEFF];

        lensOffs.x = MutateKelemen(lensOffs.x, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_SCREEN*screenScaleX, 1024.0f);
        lensOffs.y = MutateKelemen(lensOffs.y, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_SCREEN*screenScaleY, 1024.0f);
        lensOffs.z = MutateKelemen(lensOffs.z, rndFloat2_Pseudo(&gen), MUTATE_COEFF_BSDF, 1024.0f);
        lensOffs.w = MutateKelemen(lensOffs.w, rndFloat2_Pseudo(&gen), MUTATE_COEFF_BSDF, 1024.0f);
      }
    }
    
    if(out_numbers != 0)
    {
      out_numbers[ TabIndex(MMLT_DIM_SCR_X, tid, iNumElements) ] = lensOffs.x;
      out_numbers[ TabIndex(MMLT_DIM_SCR_Y, tid, iNumElements) ] = lensOffs.y;
      out_numbers[ TabIndex(MMLT_DIM_DOF_X, tid, iNumElements) ] = lensOffs.z;
      out_numbers[ TabIndex(MMLT_DIM_DOF_Y, tid, iNumElements) ] = lensOffs.w;
    }
  }

  // light, split
  //
  int d,s;
  {
    float4 lsam1; float2 lsam2; float lsamN; float split;
    if(largeStep)
    {
      lsam1 = rndFloat4_Pseudo(&gen);
      lsam2 = rndFloat2_Pseudo(&gen);
      lsamN = rndFloat1_Pseudo(&gen);
      split = rndFloat1_Pseudo(&gen);
    }
    else if(in_numbers != 0)
    {
      lsam1.x = in_numbers[ TabIndex(MMLT_DIM_LGT_X, tid, iNumElements) ];
      lsam1.y = in_numbers[ TabIndex(MMLT_DIM_LGT_Y, tid, iNumElements) ];
      lsam1.z = in_numbers[ TabIndex(MMLT_DIM_LGT_Z, tid, iNumElements) ];
      lsam1.w = in_numbers[ TabIndex(MMLT_DIM_LGT_W, tid, iNumElements) ];

      lsam2.x = in_numbers[ TabIndex(MMLT_DIM_LGT_X1, tid, iNumElements) ];
      lsam2.y = in_numbers[ TabIndex(MMLT_DIM_LGT_Y1, tid, iNumElements) ];
      lsamN   = in_numbers[ TabIndex(MMLT_DIM_LGT_N,  tid, iNumElements) ];
      split   = in_numbers[ TabIndex(MMLT_DIM_SPLIT,  tid, iNumElements) ];
   
      if(smallStepType & MUTATE_LIGHT)
      {
        const float power = a_globals->varsF[HRT_MMLT_STEP_SIZE_POWER];
        const float coeff = a_globals->varsF[HRT_MMLT_STEP_SIZE_COEFF];

        lsam1.x = MutateKelemen(lsam1.x, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
        lsam1.y = MutateKelemen(lsam1.y, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
        lsam1.z = MutateKelemen(lsam1.z, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
        lsam1.w = MutateKelemen(lsam1.w, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
        lsam2.x = MutateKelemen(lsam2.x, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
        lsam2.y = MutateKelemen(lsam2.y, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power); 
      
        //#NOTE: do not mutate lsamN !!!
        //#NOTE: do not mutate split !!!
      }
    }

    const int2 oldSplit = a_split[tid];
    d  = oldSplit.x;                                                                  // MMLT_GPU_TEST_DEPTH;
    s  = PickSplitFromRand(split, d, a_globals->varsF[HRT_MMLT_IMPLICIT_FIXED_PROB]); // mapRndFloatToInt(split, 0, d); 
    a_split[tid] = make_int2(d,s);

    if(out_numbers != 0)
    {
      out_numbers[TabIndex(MMLT_DIM_LGT_X, tid, iNumElements)]  = lsam1.x;
      out_numbers[TabIndex(MMLT_DIM_LGT_Y, tid, iNumElements)]  = lsam1.y;
      out_numbers[TabIndex(MMLT_DIM_LGT_Z, tid, iNumElements)]  = lsam1.z;
      out_numbers[TabIndex(MMLT_DIM_LGT_W, tid, iNumElements)]  = lsam1.w;

      out_numbers[TabIndex(MMLT_DIM_LGT_X1, tid, iNumElements)] = lsam2.x;
      out_numbers[TabIndex(MMLT_DIM_LGT_Y1, tid, iNumElements)] = lsam2.y;
      out_numbers[TabIndex(MMLT_DIM_LGT_N,  tid, iNumElements)] = lsamN;
      out_numbers[TabIndex(MMLT_DIM_SPLIT,  tid, iNumElements)] = split;
    }

  }

  // gen tail (bounces) next
  //
  for(int bounce = 0; bounce < a_maxBounce; bounce++)
  {
    const int bounceOffset = MMLT_HEAD_TOTAL_SIZE + MMLT_COMPRESSED_F_PERB*bounce;
   
    float6_gr gr1f;
    float4    gr2f;

    if(largeStep)
    {
      gr1f.group24 = rndFloat4_Pseudo(&gen);
      gr1f.group16 = rndFloat2_Pseudo(&gen);
      gr2f         = rndFloat4_Pseudo(&gen);
    }
    else if(in_numbers != 0)
    {
      uint4 gr1; uint2 gr2;
      gr1.x = as_int( in_numbers[TabIndex(bounceOffset + 0, tid, iNumElements)] );
      gr1.y = as_int( in_numbers[TabIndex(bounceOffset + 1, tid, iNumElements)] );
      gr1.z = as_int( in_numbers[TabIndex(bounceOffset + 2, tid, iNumElements)] );
      gr1.w = as_int( in_numbers[TabIndex(bounceOffset + 3, tid, iNumElements)] );
      gr2.x = as_int( in_numbers[TabIndex(bounceOffset + 4, tid, iNumElements)] );
      gr2.y = as_int( in_numbers[TabIndex(bounceOffset + 5, tid, iNumElements)] );

      gr1f = unpackBounceGroup (gr1);
      gr2f = unpackBounceGroup2(gr2);
      
      const bool lightT = (bounce < s) && (smallStepType & MUTATE_LIGHT) != 0;
      const bool camT   = (bounce > s) && (smallStepType & MUTATE_CAMERA) != 0;

      if(lightT || camT || (bounce == s))
      {
        const float power = a_globals->varsF[HRT_MMLT_STEP_SIZE_POWER];
        const float coeff = a_globals->varsF[HRT_MMLT_STEP_SIZE_COEFF];

        gr1f.group24.x = MutateKelemen(gr1f.group24.x, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
        gr1f.group24.y = MutateKelemen(gr1f.group24.y, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
        gr1f.group24.z = MutateKelemen(gr1f.group24.z, rndFloat2_Pseudo(&gen), coeff*MUTATE_COEFF_BSDF, power);
      }
    }
 
    if(out_numbers != 0)
    {
      uint4 gr1 = packBounceGroup(gr1f);
      uint2 gr2 = packBounceGroup2(gr2f);
      out_numbers[TabIndex(bounceOffset + 0, tid, iNumElements)] = as_float(gr1.x);
      out_numbers[TabIndex(bounceOffset + 1, tid, iNumElements)] = as_float(gr1.y);
      out_numbers[TabIndex(bounceOffset + 2, tid, iNumElements)] = as_float(gr1.z);
      out_numbers[TabIndex(bounceOffset + 3, tid, iNumElements)] = as_float(gr1.w);
      out_numbers[TabIndex(bounceOffset + 4, tid, iNumElements)] = as_float(gr2.x);
      out_numbers[TabIndex(bounceOffset + 5, tid, iNumElements)] = as_float(gr2.y);
    }
  }  

  if(out_gens != 0)
    out_gens[tid] = gen;
}

static inline MMLTReadMaterialBounceRands(__global const float* restrict in_numbers, int bounce, int tid, int iNumElements,
                                          __private float a_out[MMLT_FLOATS_PER_BOUNCE])
{
  const int bounceOffset = MMLT_HEAD_TOTAL_SIZE + MMLT_COMPRESSED_F_PERB*bounce;

  uint4 gr1; uint2 gr2;
  gr1.x = as_int( in_numbers[TabIndex(bounceOffset + 0, tid, iNumElements)] );
  gr1.y = as_int( in_numbers[TabIndex(bounceOffset + 1, tid, iNumElements)] );
  gr1.z = as_int( in_numbers[TabIndex(bounceOffset + 2, tid, iNumElements)] );
  gr1.w = as_int( in_numbers[TabIndex(bounceOffset + 3, tid, iNumElements)] );
  gr2.x = as_int( in_numbers[TabIndex(bounceOffset + 4, tid, iNumElements)] );
  gr2.y = as_int( in_numbers[TabIndex(bounceOffset + 5, tid, iNumElements)] );

  const float6_gr gr1f = unpackBounceGroup(gr1);
  const float4    gr2f = unpackBounceGroup2(gr2);

  a_out[0] = gr1f.group24.x;
  a_out[1] = gr1f.group24.y;
  a_out[2] = gr1f.group24.z;
  a_out[3] = gr1f.group24.w;
  a_out[4] = gr1f.group16.x;
  a_out[5] = gr1f.group16.y;
  a_out[6] = gr2f.x;
  a_out[7] = gr2f.y;
  a_out[8] = gr2f.z;
  a_out[9] = gr2f.w;
}

__kernel void MMLTMakeEyeRays(__global const float*         restrict in_numbers,
                              __global float4*              restrict out_pos, 
                              __global float4*              restrict out_dir,
                              __global int2*                restrict out_ind, 
                              __constant ushort*            restrict a_mortonTable256,
                              __global const EngineGlobals* restrict a_globals,  
                              int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  // (1) generate 4 random floats
  //
  //const int2 sortedIndex = in_zind[tid];
  //const float4 lensOffs  = in_samples[sortedIndex.y]; 
  
  float4 lensOffs;
  lensOffs.x = in_numbers[ TabIndex(MMLT_DIM_SCR_X, tid, iNumElements) ];
  lensOffs.y = in_numbers[ TabIndex(MMLT_DIM_SCR_Y, tid, iNumElements) ];
  lensOffs.z = in_numbers[ TabIndex(MMLT_DIM_DOF_X, tid, iNumElements) ];
  lensOffs.w = in_numbers[ TabIndex(MMLT_DIM_DOF_Y, tid, iNumElements) ];

  //if(MCMC_LAZY == 1) // #TODO: implement mutate here
  //{
  //
  //}

  // (2) generate random camera sample
  //
  float  fx, fy;
  float3 ray_pos, ray_dir;
  MakeEyeRayFromF4Rnd(lensOffs, a_globals,
                      &ray_pos, &ray_dir, &fx, &fy);

  const int w = (int)a_globals->varsF[HRT_WIDTH_F];
  const int h = (int)a_globals->varsF[HRT_HEIGHT_F];

  unsigned short x = (unsigned short)(fx);
  unsigned short y = (unsigned short)(fy);

  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;

  if (x < 0)  x = 0;
  if (y < 0)  y = 0;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// related to MakeEyeRayFromF4Rnd

  int2 indexToSort;
  indexToSort.x = ZIndex(x, y, a_mortonTable256);
  indexToSort.y = tid;

  out_pos[tid] = to_float4(ray_pos, 0.0f);
  out_dir[tid] = to_float4(ray_dir, 0.0f);
  out_ind[tid] = indexToSort;
}

__kernel void MMLTCameraPathBounce(__global   float4*        restrict a_rpos,
                                   __global   float4*        restrict a_rdir,
                                   __global   uint*          restrict a_flags,
                                   __global RandomGen*       restrict out_gens,
                                  
                                   __global const float*     restrict in_numbers,
                                   __global const int2*      restrict in_splitInfo,
                                   __global const Lite_Hit*  restrict in_hits,
                                   __global const int*       restrict in_instLightInstId,
                                   __global const float4*    restrict in_surfaceHit,
                                   __global const float4*    restrict in_procTexData,

                                   __global float4*          restrict a_color,
                                   __global MisData*         restrict a_misDataPrev,
                                   __global float4*          restrict a_fog,
                                   __global PdfVertex*       restrict a_pdfVert,       // (!) MMLT pdfArray 
                                   __global float4*          restrict a_vertexSup,     // (!) MMLT out Path Vertex supplemental to surfaceHit data

                                   __global const float4*    restrict in_texStorage1,    
                                   __global const float4*    restrict in_texStorage2,
                                   __global const float4*    restrict in_mtlStorage,
                                   __global const float4*    restrict in_pdfStorage,   //

                                   __global const EngineGlobals*  restrict a_globals,
                                   const int   iNumElements, 
                                   const float mLightSubPathCount)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid]; // #NOTE: what if ray miss object just recently .. don't we need to do soem thing in MMLT? See original code.

  // (0) Ray is outside of scene, hit environment
  //
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE) // #TODO: read environment! 
  {
    float3 envColor = make_float3(0,0,0);
    
    PathVertex resVertex;
    resVertex.ray_dir     = to_float3(a_rdir[tid]);
    resVertex.accColor    = make_float3(0,0,0); // envColor*to_float3(a_color[tid]);   
    resVertex.valid       = false; //(a_currDepth == a_targetDepth);     // #TODO: dunno if this is correct ... 
    resVertex.hitLight    = true;
    resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);
    
    flags        = packRayFlags(0, RAY_IS_DEAD);
    a_flags[tid] = flags;
  } 

  if (!rayIsActiveU(flags)) 
    return;

  // (0) Read "IntegratorMMLT::CameraPath" arguments and calc ray hit
  //
  const int2 splitData = in_splitInfo[tid];
  const int  d = splitData.x;
  const int  s = splitData.y;
  const int  t = d - s;               // note that s=2 means 1 light bounce and one connection!!!

  const bool a_haveToHitLightSource = (s == 0); // when s == 0, use only camera strategy, so have to hit light at some depth level   
  const int  a_fullPathDepth        = d;       
  const int  a_targetDepth          = t;
  const int  a_currDepth            = unpackBounceNum(flags); // #NOTE: first bounce must be equal to 1                           
  const int  prevVertexId           = a_fullPathDepth - a_currDepth + 1; 

  if(a_currDepth > t)
    return;

  //__global const PlainLight* pLight = lightAt(a_globals, lightOffset);
  SurfaceHit surfElem;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &surfElem);

 
  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfElem.matId);

  const float3 ray_pos    = to_float3(a_rpos[tid]);
  const float3 ray_dir    = to_float3(a_rdir[tid]);
  const MisData a_misPrev = a_misDataPrev[tid];
  
  // (1)
  //
  const float cosHere = fabs(dot(ray_dir, surfElem.normal));
  const float cosPrev = fabs(a_misPrev.cosThetaPrev); // fabs(dot(ray_dir, a_prevNormal));
 
  float GTerm = 1.0f;
  if (a_currDepth == 1)
  {
    float3 camDirDummy; float zDepthDummy;
    const float imageToSurfaceFactor = CameraImageToSurfaceFactor(surfElem.pos, surfElem.normal, a_globals,
                                                                  &camDirDummy, &zDepthDummy);
    const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
   
    PdfVertex vertLast;
    vertLast.pdfRev = cameraPdfA;
    vertLast.pdfFwd = 1.0f;
    a_pdfVert[TabIndex(a_fullPathDepth, tid, iNumElements)] = vertLast; //  a_perThread->pdfArray[a_fullPathDepth]
  }
  else
  {
    const float dist     = length(ray_pos - surfElem.pos); 
    GTerm = cosHere*cosPrev / fmax(dist*dist, DEPSILON2);
  }

  // (2)
  //
  const Lite_Hit liteHit  = in_hits[tid];

  ProcTextureList ptl;        
  InitProcTextureList(&ptl);  
  ReadProcTextureList(in_procTexData, tid, iNumElements, 
                      &ptl);
  
  const int lightOffset   = (a_globals->lightsNum == 0 || liteHit.instId < 0) ? -1 : in_instLightInstId[liteHit.instId]; // #TODO: refactor this into function!
  __global const PlainLight* pLight = lightAt(a_globals, lightOffset);

  const float3 emission = emissionEval(ray_pos, ray_dir, &surfElem, flags, (a_misPrev.isSpecular == 1), pLight,
                                       pHitMaterial, in_texStorage1, in_pdfStorage, a_globals, &ptl);
  
  
  if (dot(emission, emission) > 1e-3f)
  {    
    if (a_currDepth == a_targetDepth && a_haveToHitLightSource)
    {
      const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, a_globals, in_texStorage1, in_pdfStorage);
      const float pdfLightWP    = lPdfFwd.pdfW           / fmax(cosHere, DEPSILON);
      const float pdfMatRevWP   = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);
      
      {
        PdfVertex v0,v1;

        v0.pdfFwd = lPdfFwd.pdfA / ((float)a_globals->lightsNum);
        v0.pdfRev = 1.0f;

        v1.pdfFwd = pdfLightWP*GTerm;
        v1.pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfMatRevWP*GTerm;

        a_pdfVert[TabIndex(0, tid, iNumElements)] = v0;
        a_pdfVert[TabIndex(1, tid, iNumElements)] = v1;
      } 
      
      PathVertex resVertex;
      resVertex.ray_dir     = ray_dir;
      resVertex.accColor    = emission*to_float3(a_color[tid]);   
      resVertex.lastGTerm   = 1.0f; ///////////////////////////////// ?????????????????????????????????????????????????
      resVertex.valid       = true;
      resVertex.hitLight    = true; 
      resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;
      WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                                a_vertexSup);

      a_flags[tid] = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
      return;
    }
    else // this branch could probably change in future, for simple emissive materials
    {
      PathVertex resVertex;
      resVertex.ray_dir     = ray_dir;
      resVertex.accColor    = make_float3(0,0,0);
      resVertex.valid       = false;
      resVertex.hitLight    = true; 
      resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;
      WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                                a_vertexSup);

      a_flags[tid] = packRayFlags(flags, RAY_IS_DEAD);
      return;
    } 
  }
  else if (a_currDepth == a_targetDepth && !a_haveToHitLightSource) // #NOTE: what if a_targetDepth == 1 ?
  {
    PathVertex resVertex;
    resVertex.ray_dir     = ray_dir;
    resVertex.accColor    = make_float3(1, 1, 1)*to_float3(a_color[tid]);
    resVertex.valid       = true;
    resVertex.hitLight    = false;
    resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false; 
 
    if (a_targetDepth != 1)
    {
      const float lastPdfWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON); // we store them to calculate fwd and rev pdf later when we connect end points
      resVertex.lastGTerm   = GTerm;                                            // because right now we can not do this until we don't know the light vertex
     
      PdfVertex vcurr;
      vcurr.pdfFwd = 1.0f; // write it later, inside ConnectShadow or ConnectEndPoints
      vcurr.pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : GTerm*lastPdfWP;
      a_pdfVert[TabIndex(prevVertexId, tid, iNumElements)] = vcurr;
    }
    else
      resVertex.lastGTerm = 1.0f;
    
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);

    a_flags[tid] = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
    return;
  }
  

  // (3) sample material, eval reverse and forward pdfs
  //  
  float allRands[MMLT_FLOATS_PER_BOUNCE];
  MMLTReadMaterialBounceRands(in_numbers, s + a_currDepth - 1, tid, iNumElements,
                              allRands);
      
  int matOffset = materialOffset(a_globals, surfElem.matId);

  MatSample matSam; int localOffset = 0; 
  MaterialSampleAndEvalBxDF(pHitMaterial, allRands, &surfElem, ray_dir, make_float3(1,1,1), flags,
                            a_globals, in_texStorage1, in_texStorage2, &ptl, 
                            &matSam, &localOffset);
  
  matOffset    = matOffset    + localOffset*(sizeof(PlainMaterial)/sizeof(float4));
  pHitMaterial = pHitMaterial + localOffset;

  const float3 bxdfVal = matSam.color; // *(1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosNext  = fabs(dot(matSam.direction, surfElem.normal));

  if (a_currDepth == 1)
  {
    if (isPureSpecular(matSam))  //  ow ... but if we met specular reflection when tracing from camera, we must put 0 because this path cannot be sample by light strategy at all.
    {                            //  a_perThread->pdfArray[a_fullPathDepth].pdfFwd = 0.0f;
      PdfVertex vertLast = a_pdfVert[TabIndex(a_fullPathDepth, tid, iNumElements)];
      vertLast.pdfFwd    = 0.0f;
      a_pdfVert[TabIndex(a_fullPathDepth, tid, iNumElements)] = vertLast;
    }
  }
  else
  {
    PdfVertex prevVert;

    if (!isPureSpecular(matSam))
    {
      ShadeContext sc;
      sc.wp = surfElem.pos;
      sc.l  = (-1.0f)*ray_dir;  // fliped; if compare to normal PT
      sc.v  = matSam.direction; // fliped; if compare to normal PT
      sc.n  = surfElem.normal;
      sc.fn = surfElem.flatNormal;
      sc.tg = surfElem.tangent;
      sc.bn = surfElem.biTangent;
      sc.tc = surfElem.texCoord;

      const float pdfFwdW  = materialEval(pHitMaterial, &sc, false, false, 
                                          a_globals, in_texStorage1, in_texStorage2, &ptl).pdfFwd;
      const float pdfFwdWP = pdfFwdW / fmax(cosHere, DEPSILON);

      prevVert.pdfFwd = pdfFwdWP*GTerm;
    }
    else
      prevVert.pdfFwd = -1.0f*GTerm;

    const float pdfCamPrevWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);
    prevVert.pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfCamPrevWP*GTerm;
    
    a_pdfVert[TabIndex(prevVertexId, tid, iNumElements)] = prevVert;
  }

  // (4) proceed to next bounce
  //  
  float3 accColor   = to_float3(a_color[tid]);
  const bool stopDL = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;

  accColor *= (bxdfVal*cosNext / fmax(matSam.pdf, DEPSILON2));
  if (stopDL && a_haveToHitLightSource && a_currDepth + 1 == a_targetDepth) // exclude direct light
    accColor = make_float3(0, 0, 0);

  flags = flagsNextBounce(flags, matSam, a_globals);
  if (maxcomp(accColor) < 0.00001f)
  {
    PathVertex resVertex;
    resVertex.ray_dir     = make_float3(0, 0, 0);
    resVertex.accColor    = make_float3(0, 0, 0);
    resVertex.valid       = false;
    resVertex.hitLight    = false;
    resVertex.wasSpecOnly = false; 
    resVertex.lastGTerm   = 1.0f;                                            
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);
    flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
  }

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  a_flags[tid] = flags;
  a_color[tid] = to_float4(accColor,    0.0f);
  a_rpos [tid] = to_float4(nextRay_pos, 0.0f);
  a_rdir [tid] = to_float4(nextRay_dir, 0.0f);

  MisData misNext            = makeInitialMisData(); 
  misNext.matSamplePdf       = matSam.pdf;
  misNext.isSpecular         = (int)isPureSpecular(matSam);
  misNext.prevMaterialOffset = matOffset;
  misNext.cosThetaPrev       = fabs(+dot(nextRay_dir, surfElem.normal)); 
  a_misDataPrev[tid]         = misNext;
}


__kernel void MMLTLightSampleForward(__global   float4*        restrict a_rpos,
                                     __global   float4*        restrict a_rdir,
                                     __global   uint*          restrict a_flags,
                                     __global RandomGen*       restrict out_gens,
                                     __global const float*     restrict in_numbers,

                                     __global float4*          restrict a_color,
                                     __global PdfVertex*       restrict a_pdfVert,       // (!) MMLT pdfArray 
                                     __global float4*          restrict a_vertexSup,     // (!) MMLT out Path Vertex supplemental to surfaceHit data
                                     __global int*             restrict a_spec,          // (!) MMLTLightPathBounce only !!! prev bounce is specular.
                                    
                                     __global const float4*        restrict in_texStorage1,    
                                     __global const float4*        restrict in_pdfStorage,   //
                                     __global const EngineGlobals* restrict a_globals,
                                     const int   iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  /////////////////////////////////////////////////// #TODO: sample if (lightTraceDepth != 0); else return immediately
  //
  
  LightGroup2 lightRands;  
  lightRands.group1.x = in_numbers[ TabIndex(MMLT_DIM_LGT_X, tid, iNumElements) ];
  lightRands.group1.y = in_numbers[ TabIndex(MMLT_DIM_LGT_Y, tid, iNumElements) ];
  lightRands.group1.z = in_numbers[ TabIndex(MMLT_DIM_LGT_Z, tid, iNumElements) ];
  lightRands.group1.w = in_numbers[ TabIndex(MMLT_DIM_LGT_W, tid, iNumElements) ];
  lightRands.group2.x = in_numbers[ TabIndex(MMLT_DIM_LGT_X1,tid, iNumElements) ];
  lightRands.group2.y = in_numbers[ TabIndex(MMLT_DIM_LGT_Y1,tid, iNumElements) ];
  lightRands.group2.z = in_numbers[ TabIndex(MMLT_DIM_LGT_N, tid, iNumElements) ]; 
  
  float lightPickProb = 1.0f;
  const int lightId = SelectRandomLightFwd(lightRands.group2.z, a_globals,
                                           &lightPickProb);
  
  __global const PlainLight* pLight = lightAt(a_globals, lightId);
  
  LightSampleFwd sample;
  LightSampleForward(pLight, lightRands.group1, make_float2(lightRands.group2.x, lightRands.group2.y), 
                     a_globals, in_texStorage1, in_pdfStorage,
                     &sample);
  
  {
    PdfVertex v0;
    v0.pdfFwd = sample.pdfA*lightPickProb;
    v0.pdfRev = 1.0f;
    a_pdfVert[TabIndex(0, tid, iNumElements)] = v0;
  }

  float3 color = (1.0f/lightPickProb)*sample.color/(sample.pdfA*sample.pdfW);

  {  
    PathVertex lv;
    InitPathVertex(&lv);
    WritePathVertexSupplement(&lv, tid, iNumElements, 
                              a_vertexSup);
  }

  a_flags[tid] = packBounceNum(0, 1);
  a_color[tid] = to_float4(color,      0.0f);
  a_rpos [tid] = to_float4(sample.pos, sample.cosTheta);
  a_rdir [tid] = to_float4(sample.dir, sample.pdfW);
  a_spec [tid] = 0;
}

__kernel void MMLTLightPathBounce (__global   float4*        restrict a_rpos,
                                   __global   float4*        restrict a_rdir,
                                   __global   uint*          restrict a_flags,
                                   __global RandomGen*       restrict out_gens,
                                  
                                   __global const float*     restrict in_numbers,
                                   __global const int2*      restrict in_splitInfo,
                                   __global const float4*    restrict in_surfaceHit,
                                   __global const float4*    restrict in_procTexData,

                                   __global float4*          restrict a_color,
                                   __global int*             restrict a_prevSpec,
                                   __global float4*          restrict a_fog,
                                   __global PdfVertex*       restrict a_pdfVert,       // (!) MMLT pdfArray 
                                   __global float4*          restrict a_vertexSup,     // (!) MMLT out Path Vertex supplemental to surfaceHit data

                                   __global const float4*    restrict in_texStorage1,    
                                   __global const float4*    restrict in_texStorage2,
                                   __global const float4*    restrict in_mtlStorage,
                                   __global const float4*    restrict in_pdfStorage,   //

                                   __global const EngineGlobals*  restrict a_globals,
                                   const int   iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];
 
  // (0) Ray is outside of scene, hit environment
  //
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE)  // #TODO: read environment! 
  {    
    PathVertex resVertex;
    resVertex.ray_dir     = make_float3(0,0,0);
    resVertex.accColor    = make_float3(0,0,0);   
    resVertex.valid       = false; //(a_currDepth == a_targetDepth);     // #TODO: dunno if this is correct ... 
    resVertex.hitLight    = true;
    resVertex.wasSpecOnly = false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);
    
    flags        = packRayFlags(flags, RAY_IS_DEAD);
    a_flags[tid] = flags;
  } 

  const int a_currDepth = unpackBounceNum(flags);
  const int2 splitData  = in_splitInfo[tid];
  const int  d = splitData.x;
  const int  s = splitData.y;
  const int  a_lightTraceDepth = s - 1;

  if (!rayIsActiveU(flags) || a_currDepth > a_lightTraceDepth) 
    return;

  SurfaceHit surfElem;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &surfElem);

  const float4 rpos_data = a_rpos[tid];
  const float4 rdir_data = a_rdir[tid];

  const float3 ray_pos = to_float3(rpos_data); const float a_prevLightCos = rpos_data.w;
  const float3 ray_dir = to_float3(rdir_data); const float a_prevPdf      = rdir_data.w;

  const float cosPrev = fabs(a_prevLightCos);
  const float cosCurr = fabs(-dot(ray_dir, surfElem.normal));
  const float dist    = length(surfElem.pos - ray_pos);

  // eval forward pdf
  //
  const float GTermPrev = (a_prevLightCos*cosCurr / fmax(dist*dist, DEPSILON2));
  const float prevPdfWP = a_prevPdf / fmax(a_prevLightCos, DEPSILON);
  
  const bool a_wasSpecular = (a_prevSpec[tid] == 1); 
  
  {
    PdfVertex vCurr;
    if (!a_wasSpecular)
      vCurr.pdfFwd = prevPdfWP*GTermPrev;
    else
      vCurr.pdfFwd = -1.0f*GTermPrev;
    vCurr.pdfRev = 1.0f;                                          //#NOTE: override it later!
    a_pdfVert[TabIndex(a_currDepth, tid, iNumElements)] = vCurr;
  }

  // are we done with LT pass ?
  //
  if (a_currDepth == a_lightTraceDepth)
  {
    PathVertex resVertex;
    resVertex.ray_dir     = ray_dir;
    resVertex.accColor    = to_float3(a_color[tid]);
    resVertex.lastGTerm   = GTermPrev;
    resVertex.valid       = true;
    resVertex.hitLight    = false; 
    resVertex.wasSpecOnly = false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);

    a_flags[tid] = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
    return;
  }

  // not done, next bounce
  // 
  ProcTextureList ptl;        
  InitProcTextureList(&ptl);  
  ReadProcTextureList(in_procTexData, tid, iNumElements, 
                      &ptl);

  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfElem.matId);
  
  float allRands[MMLT_FLOATS_PER_BOUNCE];
  MMLTReadMaterialBounceRands(in_numbers, a_currDepth - 1, tid, iNumElements,
                              allRands);
  
  int matOffset = materialOffset(a_globals, surfElem.matId);
  
  MatSample matSam; int localOffset = 0; 
  MaterialSampleAndEvalBxDF(pHitMaterial, allRands, &surfElem, ray_dir, make_float3(1,1,1), flags,
                            a_globals, in_texStorage1, in_texStorage2, &ptl, 
                            &matSam, &localOffset);

  matOffset    = matOffset    + localOffset*(sizeof(PlainMaterial)/sizeof(float4));
  pHitMaterial = pHitMaterial + localOffset;
  
  const float cosNext  = fabs(+dot(matSam.direction, surfElem.normal));

  // calc new ray
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);
  
  // If we sampled specular event, then the reverse probability
  // cannot be evaluated, but we know it is exactly the same as
  // forward probability, so just set it. If non-specular event happened,
  // we evaluate the pdf
  //
  PdfVertex vCurr = a_pdfVert[TabIndex(a_currDepth, tid, iNumElements)];

  if (!isPureSpecular(matSam))
  {
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = (-1.0f)*ray_dir;
    sc.v  = (-1.0f)*nextRay_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const float pdfW         = materialEval(pHitMaterial, &sc, false, false, a_globals, in_texStorage1, in_texStorage2, &ptl).pdfFwd;
    const float prevPdfRevWP = pdfW / fmax(cosCurr, DEPSILON);
    vCurr.pdfRev = prevPdfRevWP*GTermPrev;
  }
  else
  {
    vCurr.pdfRev = -1.0f*GTermPrev;
  }
  
  a_pdfVert[TabIndex(a_currDepth, tid, iNumElements)] = vCurr;
  
  const float3 accColor = to_float3(a_color[tid])*matSam.color*cosNext*(1.0f / fmax(matSam.pdf, DEPSILON2));
  
  a_color[tid] = to_float4(accColor, 0.0f);
  a_rpos [tid] = to_float4(nextRay_pos, cosNext);
  a_rdir [tid] = to_float4(nextRay_dir, matSam.pdf);

  flags = flagsNextBounce(flags, matSam, a_globals);
  if (maxcomp(accColor) < 0.00001f)
  {
    PathVertex resVertex;
    resVertex.ray_dir     = ray_dir;
    resVertex.accColor    = to_float3(a_color[tid]);
    resVertex.lastGTerm   = GTermPrev;
    resVertex.valid       = false;
    resVertex.hitLight    = false; 
    resVertex.wasSpecOnly = false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);

    flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
  }

  a_flags   [tid] = flags;
  a_prevSpec[tid] = isPureSpecular(matSam) ? 1 : 0;
  
  
}

__kernel void MMLTMakeShadowRay(__global const int2  *  restrict in_splitInfo,
                                __global const float4*  restrict in_lv_hit,
                                __global const float4*  restrict in_lv_sup,
                                __global const float4*  restrict in_cv_hit,
                                __global const float4*  restrict in_cv_sup,
                                __global       float4*  restrict out_ray_pos,
                                __global       float4*  restrict out_ray_dir,
                                __global       int   *  restrict out_rflags,
                                __global       float4*  restrict out_lssam,
                                
                                __global RandomGen*     restrict a_gens,
                                __global const float*   restrict in_numbers,

                                __global const float4*         restrict in_mtlStorage,
                                __global const float4*         restrict in_pdfStorage,
                                __global const float4*         restrict in_texStorage1,
                                __global const EngineGlobals*  restrict a_globals,
                                const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;
  
  const __global float* a_rptr = 0;   /////////////////////////////////////////////////// #TODO: INIT THIS POINTER WITH MMLT RANDS !!!

  const int2 splitData      = in_splitInfo[tid];
  const int d               = splitData.x;
  const int s               = splitData.y;
  const int t               = d - s;  // note that s=2 means 1 light bounce and one connection!!!
  const int lightTraceDepth = s - 1;  // because the last light path is a connection anyway - to camera or to camera path
  const int camTraceDepth   = t;      //

  const float topUp = 1e10f;

  out_ray_pos[tid] = make_float4(0,topUp, 0, 0);
  out_ray_dir[tid] = make_float4(0,1,0,0);
  out_rflags [tid] = 0;
  
  PathVertex lv;
  ReadSurfaceHit(in_lv_hit, tid, iNumElements, 
                 &lv.hit);
  ReadPathVertexSupplement(in_lv_sup, tid, iNumElements, 
                           &lv);
  
  PathVertex cv;
  ReadSurfaceHit(in_cv_hit, tid, iNumElements, 
                 &cv.hit);
  ReadPathVertexSupplement(in_cv_sup, tid, iNumElements, 
                           &cv);

  if (lightTraceDepth == -1)        // (3.1) -1 means we have full camera path, no conection is needed
  {

  }
  else
  {
    if (camTraceDepth == 0)         // (3.2) connect light vertex to camera (light tracing connection)
    {
      if (lv.valid)
      {
         float3 camDir; float zDepth;
         const float imageToSurfaceFactor = CameraImageToSurfaceFactor(lv.hit.pos, lv.hit.normal, a_globals,
                                                                       &camDir, &zDepth);
         float signOfNormal = 1.0f;
         if (lv.hit.matId >= 0)
         {
           __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, lv.hit.matId);
           if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, lv.hit.normal) < -0.01f)
             signOfNormal *= -1.0f;
         }

         out_ray_pos[tid] = to_float4(lv.hit.pos + epsilonOfPos(lv.hit.pos)*signOfNormal*lv.hit.normal, zDepth); // OffsRayPos(lv.hit.pos, lv.hit.normal, camDir);
         out_ray_dir[tid] = to_float4(camDir, as_float(-1));
      }
    }
    else if (lightTraceDepth == 0)  // (3.3) connect camera vertex to light (shadow ray)
    {
      LightGroup2 lightSelector;
      lightSelector.group1.x = in_numbers[ TabIndex(MMLT_DIM_LGT_X, tid, iNumElements) ];
      lightSelector.group1.y = in_numbers[ TabIndex(MMLT_DIM_LGT_Y, tid, iNumElements) ];
      lightSelector.group1.z = in_numbers[ TabIndex(MMLT_DIM_LGT_Z, tid, iNumElements) ];
      lightSelector.group1.w = in_numbers[ TabIndex(MMLT_DIM_LGT_W, tid, iNumElements) ];
      lightSelector.group2.x = in_numbers[ TabIndex(MMLT_DIM_LGT_X1,tid, iNumElements) ];
      lightSelector.group2.y = in_numbers[ TabIndex(MMLT_DIM_LGT_Y1,tid, iNumElements) ];
      lightSelector.group2.z = in_numbers[ TabIndex(MMLT_DIM_LGT_N, tid, iNumElements) ]; 

      if (cv.valid && !cv.wasSpecOnly) // cv.wasSpecOnly exclude direct light actually
      {
        float lightPickProb = 1.0f;
        int lightOffset = SelectRandomLightRev(lightSelector.group2.z, cv.hit.pos, a_globals,
                                               &lightPickProb);
       
        if (lightOffset >= 0)
        {
          __global const PlainLight* pLight = lightAt(a_globals, lightOffset);
        
          ShadowSample explicitSam;
          LightSampleRev(pLight, to_float3(lightSelector.group1), cv.hit.pos, a_globals, in_pdfStorage, in_texStorage1,
                         &explicitSam);
        
          const float3 shadowRayDir = normalize(explicitSam.pos - cv.hit.pos); // explicitSam.direction;
          const float3 shadowRayPos = OffsRayPos(cv.hit.pos, cv.hit.normal, shadowRayDir); 
          const float  maxDist      = length(shadowRayPos - explicitSam.pos)*lightShadowRayMaxDistScale(pLight);

          out_ray_pos[tid] = to_float4(shadowRayPos, maxDist);
          out_ray_dir[tid] = to_float4(shadowRayDir, as_float(-1));

          WriteShadowSample(&explicitSam, lightPickProb, lightOffset, tid, iNumElements,
                            out_lssam);
        }
        

      } /// 
    }
    else                            // (3.4) connect light and camera vertices (bidir connection)
    {
      if (cv.valid && lv.valid)
      {
        const float3 diff = cv.hit.pos - lv.hit.pos;
        const float dist2 = fmax(dot(diff, diff), DEPSILON2);
        const float  dist = sqrt(dist2);
        const float3 lToC = diff / dist; // normalize(a_cv.hit.pos - a_lv.hit.pos)
        
        const float3 shadowRayDir = lToC; // explicitSam.direction;
        const float3 shadowRayPos = OffsRayPos(lv.hit.pos, lv.hit.normal, shadowRayDir);
        const float maxDist       = dist*0.995f;
        
        out_ray_pos[tid] = to_float4(shadowRayPos, maxDist);
        out_ray_dir[tid] = to_float4(shadowRayDir, as_float(-1));        
      }
    }
  }
  

}

__kernel void MMLTConnect(__global const int2  *  restrict in_splitInfo,
                          __global const float4*  restrict in_lv_hit,
                          __global const float4*  restrict in_lv_sup,
                          __global const float4*  restrict in_cv_hit,
                          __global const float4*  restrict in_cv_sup,
                          __global const float4*  restrict in_procTexData,
                          __global const ushort4* restrict in_shadow,
                          __global const float4*  restrict in_lssam,

                          __global PdfVertex*     restrict a_pdfVert,
                          __global       float4*  restrict out_color,
                          __global int2*          restrict out_zind,

                          __global const float4*         restrict in_texStorage1,    
                          __global const float4*         restrict in_texStorage2,
                          __global const float4*         restrict in_mtlStorage,
                          __global const float4*         restrict in_pdfStorage,  
                          __global const EngineGlobals*  restrict a_globals,
                          __global const float*          restrict a_scaleTable,
                          __constant ushort*             restrict a_mortonTable256,
                          const int iNumElements, const float mLightSubPathCount, const int iNumElements2)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements2)
    return;

  const int2 splitData = in_splitInfo[tid];
  const int d = splitData.x;
  const int s = splitData.y;
  const int t = d - s;                // note that s=2 means 1 light bounce and one connection!!!
  const int lightTraceDepth = s - 1;  // because the last light path is a connection anyway - to camera or to camera path
  const int camTraceDepth   = t;      //
  
  PathVertex lv;
  ReadSurfaceHit(in_lv_hit, tid, iNumElements, 
                 &lv.hit);
  ReadPathVertexSupplement(in_lv_sup, tid, iNumElements, 
                           &lv);

  PathVertex cv;
  ReadSurfaceHit(in_cv_hit, tid, iNumElements, 
                 &cv.hit);
  ReadPathVertexSupplement(in_cv_sup, tid, iNumElements, 
                           &cv);

  ProcTextureList ptl;        
  InitProcTextureList(&ptl);  
  ReadProcTextureList(in_procTexData, tid, iNumElements, 
                      &ptl);


  float3 sampleColor = make_float3(0,0,0);  
  const int zid2 = out_zind[tid].x;
  int x = ExtractXFromZIndex (zid2); // #TODO: in don;t like this but may be its ok ... 
  int y = ExtractYFromZIndex (zid2); // #TODO: in don;t like this but may be its ok ... 

  const float fixImplicit = (1.0f - a_globals->varsF[HRT_MMLT_IMPLICIT_FIXED_PROB]);
  const float fixOther    = 1.0f/fixImplicit;

  if (lightTraceDepth == -1)        // (3.1) -1 means we have full camera path, no conection is needed
  {
    if(cv.valid)
      sampleColor = cv.accColor * fixImplicit;
  }
  else
  {
    if (camTraceDepth == 0)         // (3.2) connect light vertex to camera (light tracing connection)
    {
      if (lv.valid)
      {
        float3 camDir; float zDepth;
        const float imageToSurfaceFactor = CameraImageToSurfaceFactor(lv.hit.pos, lv.hit.normal, a_globals,
                                                                      &camDir, &zDepth);
      
        __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, lv.hit.matId);
        float signOfNormal = 1.0f;
        if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, lv.hit.normal) < -0.01f)
          signOfNormal = -1.0f;
    
        PdfVertex v0, v1;
        v0 = a_pdfVert[TabIndex(lightTraceDepth + 0, tid, iNumElements)];

        sampleColor = ConnectEyeP(&lv, mLightSubPathCount, camDir, imageToSurfaceFactor,
                                  a_globals, in_mtlStorage, in_texStorage1, in_texStorage2, &ptl,
                                  &v0, &v1, &x, &y)*fixOther;
       
        if (imageToSurfaceFactor <= 0.0f)
          sampleColor = make_float3(0, 0, 0);
     
        a_pdfVert[TabIndex(lightTraceDepth + 0, tid, iNumElements)] = v0;
        a_pdfVert[TabIndex(lightTraceDepth + 1, tid, iNumElements)] = v1;
      }
    }
    else if (lightTraceDepth == 0)  // (3.3) connect camera vertex to light (shadow ray)
    {
      if (cv.valid && !cv.wasSpecOnly) // cv.wasSpecOnly exclude direct light actually
      {
        ShadowSample explicitSam; float lightPickProb; int lightOffset;
        ReadShadowSample(in_lssam, tid, iNumElements,
                         &explicitSam, &lightPickProb, &lightOffset);
        
        __global const PlainLight* pLight = lightAt(a_globals, lightOffset);
          
        PdfVertex v0, v1;
        PdfVertex v2 = a_pdfVert[TabIndex(2, tid, iNumElements)];
        sampleColor  = cv.accColor*ConnectShadowP(&cv, t, pLight, explicitSam, lightPickProb,
                                                  a_globals, in_mtlStorage, in_texStorage1, in_texStorage2, in_pdfStorage, &ptl,
                                                  &v0, &v1, &v2)*fixOther;
        //sampleColor = make_float3(0,0,0);
        a_pdfVert[TabIndex(0, tid, iNumElements)] = v0;
        a_pdfVert[TabIndex(1, tid, iNumElements)] = v1;
        a_pdfVert[TabIndex(2, tid, iNumElements)] = v2;
      }
    }
    else                            // (3.4) connect light and camera vertices (bidir connection)
    {
      if (cv.valid && !cv.wasSpecOnly && lv.valid)
      {
        const float3 diff = cv.hit.pos - lv.hit.pos;
        const float dist2 = fmax(dot(diff, diff), DEPSILON2);
        const float  dist = sqrt(dist2);
        const float3 lToC = diff / dist; // normalize(a_cv.hit.pos - a_lv.hit.pos)
        
        const float cosAtLightVertex  = +dot(lv.hit.normal, lToC);
        const float cosAtCameraVertex = -dot(cv.hit.normal, lToC);
        
        const float GTerm = cosAtLightVertex*cosAtCameraVertex / dist2;
        
        if (GTerm < 0.0f)
          sampleColor = make_float3(0, 0, 0);
        else
        {
          PdfVertex vSplitBefore = a_pdfVert[TabIndex(s-1, tid, iNumElements)];
          PdfVertex vSplitAfter  = a_pdfVert[TabIndex(s+1, tid, iNumElements)];
          PdfVertex vSplit;
      
          sampleColor = cv.accColor*lv.accColor*ConnectEndPointsP(&lv, &cv, d,
                                                                  a_globals, in_mtlStorage, in_texStorage1, in_texStorage2, &ptl,
                                                                  &vSplitBefore, &vSplit, &vSplitAfter)*fixOther;

          a_pdfVert[TabIndex(s-1, tid, iNumElements)] = vSplitBefore;
          a_pdfVert[TabIndex(s+0, tid, iNumElements)] = vSplit;
          a_pdfVert[TabIndex(s+1, tid, iNumElements)] = vSplitAfter;
        }
      }
    }
  }
  
  if (lightTraceDepth != -1)
    sampleColor *= decompressShadow(in_shadow[tid]);

  // calc MIS weight
  //
  float misWeight = 1.0f;
  if (dot(sampleColor, sampleColor) > 1e-12f)
  {
    float pdfThisWay = 1.0f;
    float pdfSumm    = 0.0f;

    const PdfVertex vD = a_pdfVert[TabIndex(d, tid, iNumElements)];

    for (int split = 0; split <= d; split++)
    {
      const int s1 = split;
      const int t1 = d - split;
      
      const PdfVertex vS = a_pdfVert[TabIndex(split, tid, iNumElements)];

      const bool specularMet = (split > 0) && (split < d) && (vS.pdfRev < 0.0f || vS.pdfFwd < 0.0f);
      float pdfOtherWay = specularMet ? 0.0f : 1.0f;
      if (split == d)
        pdfOtherWay = misHeuristicPower1(vD.pdfFwd);

      for (int i = 0; i < s1; i++)
      {
        const PdfVertex vI = a_pdfVert[TabIndex(i, tid, iNumElements)];
        pdfOtherWay *= misHeuristicPower1(vI.pdfFwd);
      }

      for (int i = s1 + 1; i <= d; i++)
      {
        const PdfVertex vI = a_pdfVert[TabIndex(i, tid, iNumElements)];
        pdfOtherWay *= misHeuristicPower1(vI.pdfRev);
      }

      if (s1 == s && t1 == t)
        pdfThisWay = pdfOtherWay;

      pdfSumm += pdfOtherWay;
    }

    misWeight = pdfThisWay / fmax(pdfSumm, DEPSILON2);
  }

  sampleColor *= misWeight;
  sampleColor *= a_scaleTable[d];

  if (!isfinite(sampleColor.x) || !isfinite(sampleColor.y) || !isfinite(sampleColor.z))
    sampleColor = make_float3(0, 0, 0);

  if (tid >= iNumElements) // if threads num was less than MEGABLOCKSIZE
  {
    x = 65535; y = 65535;
    sampleColor = make_float3(0,0,0);
  }

  const int zid = (int)ZIndex(x, y, a_mortonTable256);
  if(out_zind != 0)
    out_zind[tid] = make_int2(zid, tid);  
  out_color [tid] = to_float4(sampleColor, as_float(packXY1616(x,y)));
}

__kernel void UpdateZIndexFromColorW(__global const float4*  restrict in_color,
                                     __global       int2*    restrict out_zind,
                                     __constant ushort*      restrict a_mortonTable256,
                                     const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int packedXY = as_int(in_color[tid].w);
  const int screenX  = (packedXY & 0x0000FFFF);
  const int screenY  = (packedXY & 0xFFFF0000) >> 16;
  const int zid      = (int)ZIndex(screenX, screenY, a_mortonTable256);
  out_zind[tid]      = make_int2(zid, tid);
}

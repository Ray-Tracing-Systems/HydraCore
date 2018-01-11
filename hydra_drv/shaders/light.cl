#include "globals.h"
#include "cfetch.h"
#include "crandom.h"
#include "clight.h"

__kernel void LightSampleForwardKernel(__global float4*        restrict out_rpos,
                                       __global float4*        restrict out_rdir,
                                       __global RandomGen*     restrict out_gens,
                                       
                                       __global float4*        restrict out_data1,
                                       __global float4*        restrict out_data2,
                                       __global float4*        restrict out_data3,

                                       __global int*           restrict out_flags,         // just for clearing them
                                       __global float4*        restrict out_color,         // just for clearing them
                                       __global float4*        restrict out_thoroughput,   // just for clearing them
                                       __global float4*        restrict out_fog,           // just for clearing them
                                       __global HitMatRef*     restrict out_hitMat,        // just for clearing them
                                                                                        
                                       __global const float4*  restrict a_texStorage1,     // 
                                       __global const float4*  restrict a_pdfStorage, 
                                       __global const EngineGlobals* restrict a_globals,
                                       const int iNumElements)
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  // (1) generate light sample
  //
  RandomGen gen       = out_gens[tid];
  gen.maxNumbers      = a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  const float4 rands1 = rndFloat4_Pseudo(&gen);             // #TODO: change this for MMLT
  const float2 rands2 = rndFloat2_Pseudo(&gen);             // #TODO: change this for MMLT
  const float2 rands3 = rndFloat2_Pseudo(&gen);
  out_gens[tid]       = gen;

  float lightPickProb = 1.0f;
  const int lightId   = SelectRandomLightFwd(rands3, a_globals,
                                             &lightPickProb);
   
  __global const PlainLight* pLight = lightAt(a_globals, lightId);
  
  LightSampleFwd sample;
  LightSampleForward(pLight, rands1, rands2, a_globals, a_texStorage1, a_pdfStorage,
                     &sample);

  const float invPdf = 1.0f / (lightPickProb*fmax(sample.pdfA*sample.pdfW, DEPSILON2));

  // (2) put light sample to global memory
  //
  const float isPoint = sample.isPoint ? 1.0f : -1.0f;

  out_data1[tid] = to_float4(sample.pos,   sample.pdfA*isPoint); // float3 pos; float  pdfA;     => float4 (1) and isPoint as sign!
  out_data2[tid] = to_float4(sample.dir,   sample.pdfW);         // float3 dir; float  pdfW;     => float4 (2)
  out_data3[tid] = to_float4(sample.norm,  sample.cosTheta);     // float3 norm; float cosTheta; => float4 (3)

  out_rpos [tid] = to_float4(sample.pos, 0.0f);                  // #TODO: do we really need so many out buffers ?
  out_rdir [tid] = to_float4(sample.dir, 0.0f);                  // #TODO: do we really need so many out buffers ?

  // (3) clear temporary per ray data
  //
  HitMatRef data3;
  data3.m_data    = 0;
  data3.accumDist = 0.0f;

  out_flags      [tid] = 0;
  out_color      [tid] = to_float4(sample.color*invPdf, 0.0f);
  out_thoroughput[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  out_fog        [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  out_hitMat     [tid] = data3;

}



__kernel void LightSample(__global const float4*  restrict a_rpos,
                          __global const float4*  restrict a_rdir,
                          __global const uint*    restrict a_flags,
                          __global RandomGen*     restrict out_gens,
                                                  
                          __global const float4*  restrict in_hitPosNorm,
                          __global float4*        restrict out_data1,
                          __global float4*        restrict out_data2,
                          __global float*         restrict out_lightPickProb,  
                          __global float4*        restrict out_srpos,
                          __global float4*        restrict out_srdir,
                                                 
                          __global const float4*  restrict a_texStorage1,  // 
                          __global const float4*  restrict a_texStorage2,  //
                          __global const float4*  restrict a_pdfStorage,   //

                          int iNumElements,

                          __global const EngineGlobals* restrict a_globals,
                          __global const float*         restrict a_qmcVec,
                          __global const int2*          restrict a_qmcSorted)
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

#ifdef MLT_MULTY_PROPOSAL
  __global const float* qmcVec = (a_qmcVec == 0) ? 0 : a_qmcVec + (tid/MLT_PROPOSALS)*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
#else

  #ifndef MLT_SORT_RAYS
  __global const float* qmcVec = (a_qmcVec == 0) ? 0 : a_qmcVec + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #else
  __global const float* qmcVec = (a_qmcVec == 0) ? 0 : a_qmcVec + a_qmcSorted[tid].y*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #endif

#endif

  uint flags = a_flags[tid];

  if (!rayIsActiveU(flags))
    return;

  const bool wasGlossyOrDiffuse = (unpackRayFlags(flags) & RAY_GRAMMAR_GLOSSY_REFLECTION) || (unpackRayFlags(flags) & RAY_GRAMMAR_DIFFUSE_REFLECTION);
  const uint rayBounceNum       = unpackBounceNum(flags);

  if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
  {
    if (wasGlossyOrDiffuse && rayBounceNum >= 1)
      return;
  }

  if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
  {
    if (a_globals->g_flags & HRT_ENABLE_MLT)
    {
      if (rayBounceNum < 1) // well, this is important for MLT because it MUST sample light each bounce except the first one. 
        return;             // Even if light sample don't used further -- just to keep the right random sequence.
    }
    else
    {
      if (!wasGlossyOrDiffuse) // this works fine for PT
        return;
    }
  }

  if (a_globals->lightsNum == 0)
    return;

  const float4 data1   = in_hitPosNorm[tid];
  const float3 hitPos  = to_float3(data1);
  const float3 hitNorm = normalize(decodeNormal(as_int(data1.w)));
  const float3 ray_pos = to_float3(a_rpos[tid]);
  const float3 ray_dir = to_float3(a_rdir[tid]);

  RandomGen gen       = out_gens[tid];
  gen.maxNumbers      = a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  const float3 rands3 = to_float3(rndFloat4_Pseudo(&gen));
  const float2 rands2 = rndFloat2_Pseudo(&gen);
  out_gens[tid]       = gen;

  // (1) generate light sample
  //
  float lightPickProb = 1.0f;
  const int lightOffset = SelectRandomLightRev(rands2, hitPos, a_globals,
                                               &lightPickProb);

  __global const PlainLight* pLight = lightAt(a_globals, lightOffset); // in_plainData1 + visiableLightsOffsets[lightOffset];

  ShadowSample explicitSam;
  LightSampleRev(pLight, rands3, hitPos, a_globals, a_pdfStorage, a_texStorage1,
                 &explicitSam);

  if (explicitSam.isPoint)
    explicitSam.pdf *= -1.0f; // just to pack 'isPont' flag in pdf

  out_data1[tid]         = make_float4(explicitSam.pos.x, explicitSam.pos.y, explicitSam.pos.z, explicitSam.pdf);
  out_data2[tid]         = make_float4(explicitSam.color.x, explicitSam.color.y, explicitSam.color.z, explicitSam.maxDist);
  out_lightPickProb[tid] = lightPickProb;

  // (2) generate shadow ray
  //
  const float3 shadowRayDir = normalize(explicitSam.pos - hitPos);
  const float3 shadowRayPos = OffsRayPos(hitPos, hitNorm, shadowRayDir);
  const float maxDist       = length(shadowRayPos - explicitSam.pos); // recompute max dist based on real (shifted with offset) shadowRayPos

  out_srpos[tid] = to_float4(shadowRayPos, maxDist);
  out_srdir[tid] = to_float4(shadowRayDir, 0.0f);   // or explicitSam.cosAtLight

}

// change 11.01.2018 13:44;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

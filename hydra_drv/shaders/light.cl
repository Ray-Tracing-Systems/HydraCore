#include "globals.h"
#include "cfetch.h"
#include "crandom.h"
#include "clight.h"

__kernel void LightSampleForwardCreate(__global float4*              restrict out_data1,
                                       __global float4*              restrict out_data2,
                                       __global int2*                restrict out_index,
                                       __global RandomGen*           restrict out_gens,
                                       __constant ushort*            restrict a_mortonTable256,
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

  unsigned int sortId = (((unsigned int)lightId) << 24) & 0xFF000000; // this mask is nor reallly neede, i just wat to have it here to understand bit layout

  const unsigned int x1 = (unsigned int)( fmin(rands1.x*16.0f,  16.0f)  );
  const unsigned int y1 = (unsigned int)( fmin(rands1.y*16.0f,  16.0f)  );
  const unsigned int x2 = (unsigned int)( fmin(rands1.z*256.0f, 256.0f) );
  const unsigned int y2 = (unsigned int)( fmin(rands1.w*256.0f, 256.0f) );

  __global const PlainLight* pLight = lightAt(a_globals, lightId);

  if (lightType(pLight) == PLAIN_LIGHT_TYPE_SPHERE)                   // when sample spheres positions are more important
  {
    sortId |= (ZIndex(x2, y2, a_mortonTable256)) & 0x000000FF;
    sortId |= (ZIndex(x1, y1, a_mortonTable256) << 8) & 0x00FFFF00;
  }
  else                                                                // when sarea lights and other directions are more important
  {
    sortId |= (ZIndex(x1, y1, a_mortonTable256)) & 0x000000FF;
    sortId |= (ZIndex(x2, y2, a_mortonTable256) << 8) & 0x00FFFF00;
  }

  out_data1[tid] = rands1;
  out_data2[tid] = make_float4(rands2.x, rands2.y, lightPickProb, as_float(lightId));
  out_index[tid] = make_int2(sortId, tid);
}


__kernel void LightSampleForwardKernel(__global float4*        restrict out_rpos,
                                       __global float4*        restrict out_rdir,
                                       //__global RandomGen*     restrict out_gens,
                                       __global const float4*  restrict in_data1,
                                       __global const float4*  restrict in_data2,
                                       __global const int2*    restrict in_index,

                                       __global float4*        restrict out_data1,
                                       __global float4*        restrict out_data2,
                                       __global float4*        restrict out_data3,
                                       __global PerRayAcc*     restrict out_pdfAcc,
                                       __global MisData*       restrict out_cosAndOther,
                                       __global int*           restrict out_lightId,

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
  // RandomGen gen       = out_gens[tid];
  // gen.maxNumbers      = a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  // const float4 rands1 = rndFloat4_Pseudo(&gen);             // #TODO: change this for MMLT
  // const float2 rands2 = rndFloat2_Pseudo(&gen);             // #TODO: change this for MMLT
  // const float2 rands3 = rndFloat2_Pseudo(&gen);
  // out_gens[tid]       = gen;
  // 
  // float lightPickProb = 1.0f;
  // const int lightId   = SelectRandomLightFwd(rands3, a_globals,
  //                                            &lightPickProb);
  
  const int2 index    = in_index[tid];
  const float4 rands1 = in_data1[index.y];
  const float4 data2  = in_data2[index.y];
  const float2 rands2 = make_float2(data2.x, data2.y);

  const float lightPickProb = data2.z;
  const int   lightId       = as_int(data2.w);

  __global const PlainLight* pLight = lightAt(a_globals, lightId);
  
  LightSampleFwd sample;
  LightSampleForward(pLight, rands1, rands2, a_globals, a_texStorage1, a_pdfStorage,
                     &sample);

  const float invPdf = 1.0f / (lightPickProb*fmax(sample.pdfA*sample.pdfW, DEPSILON2));

  // (2) put light sample to global memory
  //
  const float isPoint = sample.isPoint ? 1.0f : -1.0f;

  out_data1[tid] = to_float4(sample.pos,  as_float(lightType(pLight)));
  out_data2[tid] = to_float4(sample.dir,  sample.pdfA);          
  out_data3[tid] = to_float4(sample.norm, as_float(lightId));    

  out_rpos [tid] = to_float4(sample.pos, 0.0f);                 // #TODO: do we really need so many out buffers ?
  out_rdir [tid] = to_float4(sample.dir, 0.0f);                 // #TODO: do we really need so many out buffers ?

  out_lightId[tid] = lightId;

  PerRayAcc acc0  = InitialPerParAcc();
  acc0.pdfLightWP = sample.pdfW / fmax(sample.cosTheta, DEPSILON);
  out_pdfAcc[tid] = acc0;

  // store sample.cosTheta like 'a_prevLightCos' as argument (per bounce) 
  //
  MisData misdata      = makeInitialMisData();
  misdata.cosThetaPrev = sample.cosTheta;
  out_cosAndOther[tid] = misdata;

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
                          __global float*         restrict out_lPP,
                          __global float*         restrict out_lcos,

                          __global float4*        restrict out_srpos,
                          __global float4*        restrict out_srdir,

                          __global const float4*  restrict a_texStorage1,  //
                          __global const float4*  restrict a_texStorage2,  //
                          __global const float4*  restrict a_pdfStorage,   //
                          
                          int iNumElements,
                          __global const EngineGlobals* restrict a_globals)
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];

  if (!rayIsActiveU(flags))
    return;

  const bool wasGlossyOrDiffuse = (unpackRayFlags(flags) & RAY_EVENT_G) || (unpackRayFlags(flags) & RAY_EVENT_D);
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

  out_data1[tid] = make_float4(explicitSam.pos.x, explicitSam.pos.y, explicitSam.pos.z, explicitSam.pdf);
  out_data2[tid] = make_float4(explicitSam.color.x, explicitSam.color.y, explicitSam.color.z, explicitSam.maxDist);
  out_lPP  [tid] = lightPickProb;
  out_lcos [tid] = explicitSam.cosAtLight;

  // (2) generate shadow ray
  //
  const float3 shadowRayDir = normalize(explicitSam.pos - hitPos);
  const float3 shadowRayPos = OffsRayPos(hitPos, hitNorm, shadowRayDir);
  const float maxDist       = length(shadowRayPos - explicitSam.pos); // recompute max dist based on real (shifted with offset) shadowRayPos

  out_srpos[tid] = to_float4(shadowRayPos, maxDist);
  out_srdir[tid] = to_float4(shadowRayDir, as_float(lightOffset)); 

}

__kernel void CopyAndPackForConnectEye(__global const uint*    restrict in_flags,
                                       __global const float4*  restrict in_raydir, 
                                       __global const float4*  restrict in_colors,
         
                                       __global uint*          restrict out_flags,
                                       __global float4*        restrict out_raydir,
                                       __global float4*        restrict out_colors,
                                      int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_flags [tid] = in_flags[tid];
  out_colors[tid] = in_colors[tid];
  out_raydir[tid] = in_raydir[tid];
}

// change 31.01.2018 15:20;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

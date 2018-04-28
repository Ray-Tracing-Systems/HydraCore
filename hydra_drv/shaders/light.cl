#include "cglobals.h"
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
                          __global const float4*  restrict in_hitNormAndShadowOffs,

                          __global float4*        restrict out_data1,
                          __global float4*        restrict out_data2,
                          __global float*         restrict out_lPP,
                          __global float*         restrict out_lcos,

                          __global float4*        restrict out_srpos,
                          __global float4*        restrict out_srdir,
                          __global int*           restrict out_loffs,

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

  if (a_globals->lightsNum == 0)
    return;

  const float4 data1   = in_hitPosNorm[tid];
  const float4 data2   = in_hitNormAndShadowOffs[tid];
  const float3 hitPos  = to_float3(data1);
  const float3 hitNorm = to_float3(data2); // normalize(decodeNormal(as_int(data1.w)));
  const float  sRayOff = data2.w;
  const float3 ray_pos = to_float3(a_rpos[tid]);
  const float3 ray_dir = to_float3(a_rdir[tid]);

  RandomGen gen       = out_gens[tid];
  gen.maxNumbers      = a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  const float3 rands3 = to_float3(rndFloat4_Pseudo(&gen));
  const float2 rands2 = rndFloat2_Pseudo(&gen);
  out_gens[tid]       = gen;

  // (1) generate light sample
  //
  float lightPickProb   = 1.0f;
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
  const float3 shadowRayPos = OffsShadowRayPos(hitPos, hitNorm, shadowRayDir, sRayOff); // sRayOff
  const float  maxDist      = length(shadowRayPos - explicitSam.pos)*0.995f;            // recompute max dist based on real (shifted with offset) shadowRayPos

  out_srpos[tid] = to_float4(shadowRayPos, maxDist);
  out_srdir[tid] = to_float4(shadowRayDir, as_float(-1)); 
  out_loffs[tid] = lightOffset;
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


__kernel void MakeAORays(__global const uint*      restrict in_flags,
                         __global RandomGen*       restrict a_gens,

                         __global const Lite_Hit*  restrict in_hits,
                         __global const float4*    restrict in_hitPosNorm,
                         __global const float2*    restrict in_hitTexCoord,
                         __global const HitMatRef* restrict in_matData,

                         __global float4*          restrict out_rpos,
                         __global float4*          restrict out_rdir,
  
                         __global const float4*    restrict in_texStorage1,
                         __global const float4*    restrict in_mtlStorage,
                         __global const EngineGlobals* restrict a_globals,
                         int iterNum, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const uint flags = in_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  const float4 data1   = in_hitPosNorm[tid];
  const float3 hitPos  = to_float3(data1);
  const float3 hitNorm = decodeNormal(as_int(data1.w));

  __global const PlainMaterial* pMaterialHead = materialAt(a_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));

  float3 sRayPos    = make_float3(0, 1e30f, 0);
  float3 sRayDir    = make_float3(0, 1, 0);
  float  sRayLength = pMaterialHead->data[PROC_TEX_AO_LENGTH];
  int targetInstId  = -1;

  if (MaterialHaveAO(pMaterialHead))
  {
    RandomGen gen = a_gens[tid];
    float3 rands  = to_float3(rndFloat4_Pseudo(&gen));
    a_gens[tid]   = gen;

    sRayPos       = hitPos;

    int aoType = MaterialAOType(pMaterialHead);
    if (aoType == AO_TYPE_UP)
    {
      sRayDir = MapSampleToCosineDistribution(rands.x, rands.y, hitNorm, hitNorm, 1.0f);
      sRayPos = OffsRayPos(sRayPos, hitNorm, sRayDir);
    }
    else if (aoType == AO_TYPE_DOWN)
    {
      sRayDir = MapSampleToCosineDistribution(rands.x, rands.y, (-1.0f)*hitNorm, (-1.0f)*hitNorm, 1.0f);
      sRayPos = OffsRayPos(sRayPos, (-1.0f)*hitNorm, sRayDir);
    }
    else if (aoType == AO_TYPE_BOTH)
    {
      if (rands.z <= 0.5f)
      {
        sRayDir = MapSampleToCosineDistribution(rands.x, rands.y, hitNorm, hitNorm, 1.0f);
        sRayPos = OffsRayPos(sRayPos, hitNorm, sRayDir);
      }
      else
      {
        sRayDir = MapSampleToCosineDistribution(rands.x, rands.y, (-1.0f)*hitNorm, (-1.0f)*hitNorm, 1.0f);
        sRayPos = OffsRayPos(sRayPos, (-1.0f)*hitNorm, sRayDir);
      }
    }

    if (materialGetFlags(pMaterialHead) & PLAIN_MATERIAL_LOCAL_AO)
      targetInstId = in_hits[tid].instId;
  }

  if(iterNum == 0)
    out_rpos[tid] = to_float4(sRayPos, sRayLength);

  out_rdir[tid] = to_float4(sRayDir, as_float(targetInstId));
}

static inline float3 decompressShadow(ushort4 shadowCompressed)
{
  const float invNormCoeff = 1.0f / 65535.0f;
  return invNormCoeff * make_float3((float)shadowCompressed.x, (float)shadowCompressed.y, (float)shadowCompressed.z);
}

__kernel void PackAO(__global const uint*      restrict in_flags,
                     __global const ushort4*   restrict in_shadowAO,
                     __global       uchar*     restrict out_ao,
                     int i, int numIter, int iNumElements)

{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const uint flags = in_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  const float3 shadow = decompressShadow(in_shadowAO[tid]);
  const float  aoIn   = 0.333334f*(shadow.x + shadow.y + shadow.z);

  const uchar  aoChaCur = (aoIn > 0.5f) ? 1 : 0;
  const uchar  aoChaOld = (i == 0) ? 0 : out_ao[tid];
  const uchar  aoSumm   = aoChaCur + aoChaOld;

  const float  aoFinRes = (float)(aoSumm) / ((float)numIter);

  if (i == numIter - 1)
    out_ao[tid] = (uchar)(aoFinRes*255.0f);
  else
    out_ao[tid] = aoSumm;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

  const float4 rands1 = rndFloat4_Pseudo(&gen);             // #TODO: change this for MMLT
  const float2 rands2 = rndFloat2_Pseudo(&gen);             // #TODO: change this for MMLT
  const float  rands3 = rndFloat1_Pseudo(&gen);
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
                                       __global const float4*  restrict in_data1,
                                       __global const float4*  restrict in_data2,
                                       __global const int2*    restrict in_index,

                                       __global PerRayAcc*     restrict out_pdfAcc,
                                       __global MisData*       restrict out_cosAndOther,
                                       __global int*           restrict out_lightId,
                                       __global float*         restrict out_lsam2,

                                       __global int*           restrict out_flags,         // just for clearing them
                                       __global float4*        restrict out_color,         // just for clearing them
                                       __global float4*        restrict out_thoroughput,   // just for clearing them
                                       __global float4*        restrict out_fog,           // just for clearing them
                                                                                        
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
  
  out_lsam2[tid] = sample.pdfA;
  out_rpos [tid] = to_float4(sample.pos, as_float(encodeNormal(sample.norm)));
  out_rdir [tid] = to_float4(sample.dir, 0.0f);

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
  out_flags      [tid] = 0;
  out_color      [tid] = to_float4(sample.color*invPdf, 0.0f);
  out_thoroughput[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  out_fog        [tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__kernel void LightSample(__global const int2*    restrict in_zind,
                          __global const float*   restrict in_xvector,
  
                          __global const float4*  restrict in_rpos,
                          __global const float4*  restrict in_rdir,

                          __global const uint*    restrict in_flags,
                          __global const float4*  restrict in_surfaceHit,

                          __global RandomGen*     restrict out_gens,
                          __global float4*        restrict out_lrev,

                          __global float4*        restrict out_srpos,
                          __global float4*        restrict out_srdir,

                          __global const float4*  restrict a_texStorage1,  //
                          __global const float4*  restrict a_texStorage2,  //
                          __global const float4*  restrict a_pdfStorage,   //
                          
                          __global const EngineGlobals* restrict a_globals,
                          int iNumElements)
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = in_flags[tid];

  if (!rayIsActiveU(flags))
    return;

  if (a_globals->lightsNum == 0)
    return;

  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &sHit);

  const float3 ray_pos = to_float3(in_rpos[tid]);
  const float3 ray_dir = to_float3(in_rdir[tid]);

  // read (QMC/KMLT) or generate (PT) rands
  //
  float4 rands;
  {
    RandomGen gen = out_gens[tid];
    
    const int depth = unpackBounceNum(flags);
    
    if(depth+1 <= a_globals->varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] && in_zind != 0 && in_xvector != 0)
    {
      const int2 sortedIndex = in_zind[tid];
      const int lightsOffset = kmltBlobSize(a_globals)*sortedIndex.y + kmltLightOffset(depth);
      
      __global const float* pData = (in_xvector + lightsOffset); 
      rands.x = pData[0];
      rands.y = pData[1];
      rands.z = pData[2];
      rands.w = pData[3];
    }
    else
    {
      rands = rndFloat4_Pseudo(&gen);
    }
  
    out_gens[tid] = gen;
  }

  // (1) generate light sample
  //
  float lightPickProb   = 1.0f;
  const int lightOffset = SelectRandomLightRev(rands.w, sHit.pos, a_globals,
                                               &lightPickProb);

  __global const PlainLight* pLight = lightAt(a_globals, lightOffset); // in_plainData1 + visiableLightsOffsets[lightOffset];

  ShadowSample explicitSam;
  LightSampleRev(pLight, to_float3(rands), sHit.pos, a_globals, a_pdfStorage, a_texStorage1,
                 &explicitSam);

  WriteShadowSample(&explicitSam, lightPickProb, lightOffset, tid, iNumElements,
                    out_lrev);

  // (2) generate shadow ray
  //
  const float3 shadowRayDir = normalize(explicitSam.pos - sHit.pos);
  const float3 shadowRayPos = OffsShadowRayPos(sHit.pos, sHit.normal, shadowRayDir, sHit.sRayOff);       // sRayOff
  const float  maxDist      = length(shadowRayPos - explicitSam.pos)*lightShadowRayMaxDistScale(pLight); // recompute max dist based on real (shifted with offset) shadowRayPos

  out_srpos[tid] = to_float4(shadowRayPos, maxDist);
  out_srdir[tid] = to_float4(shadowRayDir, as_float(-1)); 
}


__kernel void CopyLightSampleToSurfaceHit(__global const float4* restrict in_raypos, 
                                          __global       float4* restrict out_hits,
                                          int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  SurfaceHit surfHitWS;

  const float4 data    = in_raypos[tid];

  surfHitWS.pos        = to_float3(data);
  surfHitWS.normal     = decodeNormal(as_int(data.w));
  surfHitWS.matId      = -1;

  WriteSurfaceHit(&surfHitWS, tid, iNumElements, 
                  out_hits);
  
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
                         __global const float4*    restrict in_surfaceHit,

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

  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &sHit);

  __global const PlainMaterial* pMaterialHead = materialAt(a_globals, in_mtlStorage, sHit.matId);
  if(pMaterialHead == 0)
    return;

  float3 sRayPos    = make_float3(0, 1e30f, 0);
  float3 sRayDir    = make_float3(0, 1, 0);
  float  sRayLength = pMaterialHead->data[PROC_TEX_AO_LENGTH];
  int targetInstId  = -1;

  const float3 lenTexColor = sample2D(as_int(pMaterialHead->data[PROC_TEXMATRIX_ID]), sHit.texCoord, (__global const int4*)pMaterialHead, in_texStorage1, a_globals);
  sRayLength *= maxcomp(lenTexColor);

  if (MaterialHaveAO(pMaterialHead))
  {
    RandomGen gen = a_gens[tid];
    float3 rands  = to_float3(rndFloat4_Pseudo(&gen));
    a_gens[tid]   = gen;

    const float3 aoDir = (as_int(pMaterialHead->data[PROC_TEX_AO_TYPE]) == AO_TYPE_UP) ? sHit.normal : -1.0f*(sHit.normal);

    sRayDir = MapSampleToCosineDistribution(rands.x, rands.y, aoDir, aoDir, 1.0f);
    sRayPos = OffsRayPos(sHit.pos, aoDir, aoDir);

    if (materialGetFlags(pMaterialHead) & PLAIN_MATERIAL_LOCAL_AO1)
      targetInstId = in_hits[tid].instId;

    out_rdir[tid] = to_float4(sRayDir, as_float(targetInstId));
  }
  else
  {
    sRayLength = 0.0f;
  }

  if(iterNum == 0)
    out_rpos[tid] = to_float4(sRayPos, sRayLength);
}


__kernel void MakeAORaysPacked4(__global const uint*      restrict in_flags,
                                __global RandomGen*       restrict a_gens,
                                
                                __global const Lite_Hit*  restrict in_hits,
                                __global const float4*    restrict in_surfaceHit,
                                
                                __global float4*          restrict out_rpos,
                                __global int4*            restrict out_rdir,
                                __global int*             restrict out_instId,
                                
                                __global const float4*    restrict in_texStorage1,
                                __global const float4*    restrict in_mtlStorage,
                                __global const EngineGlobals* restrict a_globals,
                                int aoId, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const uint flags = in_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements,                                                //#TODO: opt surface read for this kernel !!!
                 &sHit);

  __global const PlainMaterial* pMaterialHead = materialAt(a_globals, in_mtlStorage, sHit.matId);
  if(pMaterialHead == 0)
    return;

  float3 sRayPos  = make_float3(0, 1e30f, 0);
  float3 sRayDir1 = make_float3(0, 1, 0);
  float3 sRayDir2 = make_float3(0, 1, 0);
  float3 sRayDir3 = make_float3(0, 1, 0);
  float3 sRayDir4 = make_float3(0, 1, 0);

  const int mflags    = materialGetFlags(pMaterialHead);
  float  sRayLength   = (aoId == 1) ? pMaterialHead->data[PROC_TEX_AO_LENGTH2]        : pMaterialHead->data[PROC_TEX_AO_LENGTH];
  const int texId     = (aoId == 1) ? as_int(pMaterialHead->data[PROC_TEXMATRIX_ID2]) : as_int(pMaterialHead->data[PROC_TEXMATRIX_ID]);
  const int flagLocal = (aoId == 1) ? (mflags & PLAIN_MATERIAL_LOCAL_AO2)             : (mflags & PLAIN_MATERIAL_LOCAL_AO1);
  const int aoType    = (aoId == 1) ? as_int(pMaterialHead->data[PROC_TEX_AO_TYPE2])  : as_int(pMaterialHead->data[PROC_TEX_AO_TYPE]);

  int targetInstId    = -1;

  const float3 lenTexColor = sample2D(texId, sHit.texCoord, (__global const int4*)pMaterialHead, in_texStorage1, a_globals);
  sRayLength *= maxcomp(lenTexColor);

  if (MaterialHaveAO(pMaterialHead))
  {
    RandomGen gen = a_gens[tid];
    const float3 rands1 = to_float3(rndFloat4_Pseudo(&gen));
    const float3 rands2 = to_float3(rndFloat4_Pseudo(&gen));
    const float3 rands3 = to_float3(rndFloat4_Pseudo(&gen));
    const float3 rands4 = to_float3(rndFloat4_Pseudo(&gen));
    a_gens[tid]   = gen;

    const float3 aoDir = (aoType == AO_TYPE_UP) ? sHit.normal : -1.0f*(sHit.normal);

    sRayDir1 = MapSampleToCosineDistribution(rands1.x, rands1.y, aoDir, aoDir, 1.0f);
    sRayDir2 = MapSampleToCosineDistribution(rands2.x, rands2.y, aoDir, aoDir, 1.0f);
    sRayDir3 = MapSampleToCosineDistribution(rands3.x, rands3.y, aoDir, aoDir, 1.0f);
    sRayDir4 = MapSampleToCosineDistribution(rands4.x, rands4.y, aoDir, aoDir, 1.0f);

    sRayPos  = OffsRayPos(sHit.pos, aoDir, aoDir);

    if (flagLocal)
      targetInstId = in_hits[tid].instId;

    out_rdir  [tid] = make_int4(encodeNormal(sRayDir1), encodeNormal(sRayDir2), encodeNormal(sRayDir3), encodeNormal(sRayDir4));
    out_instId[tid] = targetInstId;
  }
  else
  {
    sRayLength = 0.0f;
  }

  out_rpos[tid] = to_float4(sRayPos, sRayLength);
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

__kernel void PackAO4(__global const uint*     restrict in_flags,
                      __global const ushort*   restrict in_shadowAO,
                      __global       uchar*    restrict out_ao,
                     int iNumElements)

{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const uint flags = in_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  const float  aoIn0 = (float)(in_shadowAO[tid * 4 + 0]) * (1.0f / 65535.0f);
  const float  aoIn1 = (float)(in_shadowAO[tid * 4 + 1]) * (1.0f / 65535.0f);
  const float  aoIn2 = (float)(in_shadowAO[tid * 4 + 2]) * (1.0f / 65535.0f);
  const float  aoIn3 = (float)(in_shadowAO[tid * 4 + 3]) * (1.0f / 65535.0f);
  const float  aoRes = 0.25f*(aoIn0 + aoIn1 + aoIn2 + aoIn3);

  out_ao[tid] = (uchar)(aoRes*255.0f);
}

// change 22.03.2020 11:08;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#include "globals.h"
//#include "cfetch.h"
//#include "crandom.h"
#include "cmaterial.h"

__kernel void ProcTexExec(__global       uint*          restrict a_flags,
                                                        
                          __global const float4*        restrict in_hitPosNorm,
                          __global const float2*        restrict in_hitTexCoord,
                          __global const uint*          restrict in_flatNorm,
                          __global const HitMatRef*     restrict in_matData,
                          __global const Hit_Part4*     restrict in_hitTangent,  
                          __global const float4*        restrict in_normalsFull,
                                                        
                          __global       float4*        restrict out_procTexData,
                                                        
                          __global const float4*        restrict in_texStorage1,
                          __global const float4*        restrict in_mtlStorage,
                          __global const float4*        restrict in_pdfStorage,
                          __global const EngineGlobals* restrict in_globals,
                          int iNumElements)
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const uint flags  = a_flags[tid];

  if (!rayIsActiveU(flags))
    return;

  __global const PlainMaterial* pHitMaterial = materialAt(in_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));

  if (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_PROC_TEXTURES)
  {
    // read surfaceHit
    //
    const Hit_Part4 btanAndN = in_hitTangent[tid];
    const float3 hitPos      = to_float3(in_hitPosNorm[tid]);
    const float3 hitNorm     = to_float3(in_normalsFull[tid]); // normalize(decodeNormal(as_int(data.w)));
    const float2 hitTexCoord = in_hitTexCoord[tid];
    const float3 hitBiTang   = decodeNormal(btanAndN.tangentCompressed);
    const float3 hitBiNorm   = decodeNormal(btanAndN.bitangentCompressed);
    const float3 flatN       = decodeNormal(in_flatNorm[tid]);

    ProcTextureList ptl;
    InitProcTextureList(&ptl);

    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptl);

    ptl.fdata4[0] = make_float4(hitTexCoord.x, hitTexCoord.y, 0.0f, 0.0f);

    WriteProcTextureList(out_procTexData, tid, iNumElements, &ptl);
  }



} 


// change 31.01.2018 15:20;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

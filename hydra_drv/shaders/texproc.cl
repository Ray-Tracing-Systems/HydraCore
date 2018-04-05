//#include "globals.h"
//#include "cfetch.h"
//#include "crandom.h"
#include "cmaterial.h"

static inline float4 InternalFetch(int a_texId, const float2 texCoord, const int a_flags, 
                                   __global const float4* restrict in_texStorage1, __global const EngineGlobals* restrict in_globals)
{
  const int offset = textureHeaderOffset(in_globals, a_texId);
  return read_imagef_sw4(in_texStorage1 + offset, texCoord, a_flags);
}

#define texture2D(texName, texCoord, flags) InternalFetch((texName), (texCoord), (flags), in_texStorage1, in_globals)


typedef struct SurfaceInfoT
{
  float3 wp;
  float3 n;
  float2 tc0;

  //#TODO: add custom attributes

} SurfaceInfo;

#define readAttr_WorldPos (sHit) (sHit->wp);
#define readAttr_ShadeNorm(sHit) (sHit->n);
#define readAttr_TexCoord0(sHit) (sHit->tc0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define TEX_POINT_SAM 0x10000000
//#define TEX_CLAMP_U   0x40000000
//#define TEX_CLAMP_V   0x80000000

float4 userProc(const SurfaceInfo* sHit, __global const float4* restrict in_texStorage1, __global const EngineGlobals* restrict in_globals)
{
  const float2 texCoord = readAttr_TexCoord0(sHit);
  const float4 texColor = texture2D(1, texCoord, TEX_CLAMP_U | TEX_CLAMP_V);
  return make_float4(texCoord.x, texCoord.y, 0.0f, 0.0f)*texColor;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void ProcTexExec(__global       uint*          restrict a_flags,
                                                        
                          __global const float4*        restrict in_hitPosNorm,
                          __global const float2*        restrict in_hitTexCoord,
                          __global const HitMatRef*     restrict in_matData,
                          __global const float4*        restrict in_normalsFull,
                                                        
                          __global       float4*        restrict out_procTexData,
                                                        
                          __global const float4*        restrict in_texStorage1,
                          __global const float4*        restrict in_mtlStorage,
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
    // (1) read common attributes to 'surfaceHit'
    //
    SurfaceInfo surfHit;

    surfHit.wp  = to_float3(in_hitPosNorm[tid]);
    surfHit.n   = to_float3(in_normalsFull[tid]); // normalize(decodeNormal(as_int(data.w)));
    surfHit.tc0 = in_hitTexCoord[tid];

    // (2) read custom attributes to 'surfHit' if target mesh have them.
    //

    ProcTextureList ptl;
    InitProcTextureList(&ptl);

    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptl);

    // (3) evaluate all proc textures
    //

    // (4) take what we need from all array
    //
    ptl.fdata4[0] = userProc(&surfHit, in_texStorage1, in_globals);

    WriteProcTextureList(out_procTexData, tid, iNumElements, &ptl);
  }



} 


// change 31.01.2018 15:20;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

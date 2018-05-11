#include "cglobals.h"
#include "cfetch.h"

static inline float4 InternalFetch(int a_texId, const float2 texCoord, const int a_flags, 
                                   __global const float4* restrict in_texStorage1, __global const EngineGlobals* restrict in_globals)
{
  if (a_texId < 0)
    return make_float4(1, 1, 1, 1);
  else
  {
    const int offset = textureHeaderOffset(in_globals, a_texId);
    return read_imagef_sw4(in_texStorage1 + offset, texCoord, a_flags);
  }
}

#define texture2D(texName, texCoord, flags) InternalFetch((texName), (texCoord), (flags), in_texStorage1, in_globals)

typedef int sampler2D;

typedef struct SurfaceInfoT
{
  float3 wp;
  float3 lp;
  float3 n;
  float2 tc0;
  float  ao;
  float  ao2;

  //#TODO: add custom attributes

} SurfaceInfo;

#define readAttr_WorldPos(sHit) (sHit->wp)
#define readAttr_LocalPos(sHit) (sHit->lp)
#define readAttr_ShadeNorm(sHit) (sHit->n)
#define readAttr_TexCoord0(sHit) (sHit->tc0)
#define readAttr_AO(sHit) (sHit->ao)
#define readAttr_AO1(sHit) (sHit->ao2)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define TEX_POINT_SAM 0x10000000
//#define TEX_CLAMP_U   0x40000000
//#define TEX_CLAMP_V   0x80000000

//#PUT_YOUR_PROCEDURAL_TEXTURES_HERE:


///////////////////////////////////////////////////////////////

const int findArgDataOffsetInTable(int a_texId, __global const int* a_table)
{
  const int totalTexNum = a_table[PLAIN_MATERIAL_DATA_SIZE-1];
  
  int offset = 0;
  for (int i = 0; i < totalTexNum; i++)
  {
    if (a_table[i * 2 + 0] == a_texId)
    {
      offset = a_table[i * 2 + 1];
      break;
    }
  }

  return offset;
}

static inline float3 decompressShadow(ushort4 shadowCompressed)
{
  const float invNormCoeff = 1.0f / 65535.0f;
  return invNormCoeff * make_float3((float)shadowCompressed.x, (float)shadowCompressed.y, (float)shadowCompressed.z);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void ProcTexExec(__global       uint*          restrict a_flags,
                          __global const float4*        restrict in_rdir,                                                      

                          __global const float4*        restrict in_hitPosNorm,
                          __global const float2*        restrict in_hitTexCoord,
                          __global const HitMatRef*     restrict in_matData,
 
                          __global const uchar*         restrict in_shadowAOCompressed1,
                          __global const uchar*         restrict in_shadowAOCompressed2,
                          __global const Lite_Hit*      restrict in_hits,
                          __global const float4*        restrict in_matrices,

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
    float shadow1 = 1.0f;
    float shadow2 = 1.0f;

    if(in_shadowAOCompressed1 != 0)
      shadow1 = ((float)in_shadowAOCompressed1[tid]) / 255.0f;

    if(in_shadowAOCompressed2 != 0 )
      shadow2 = ((float)in_shadowAOCompressed2[tid]) / 255.0f;

    const Lite_Hit hit = in_hits[tid];
    const float4x4 instanceMatrixInv = fetchMatrix(hit, in_matrices);

    const float4 dataPos = in_hitPosNorm[tid];

    SurfaceInfo surfHit;
    surfHit.wp  = to_float3(in_hitPosNorm[tid]);
    surfHit.lp  = mul4x3(instanceMatrixInv, surfHit.wp);
    surfHit.n   = normalize(decodeNormal(as_int(dataPos.w)));
    surfHit.tc0 = in_hitTexCoord[tid];
    surfHit.ao  = shadow1; 
    surfHit.ao2 = shadow2;
    __private const SurfaceInfo* sHit = &surfHit;

    const float3 hr_viewVectorHack = to_float3(in_rdir[tid]); // make dependence of this vector is not physycally correct but useful for artists ... 

    // (2) read custom attributes to 'surfHit' if target mesh have them.
    //

    // read proc texture list for target 'pHitMaterial' material
    // 
    ProcTextureList ptl;
    InitProcTextureList(&ptl);

    GetProcTexturesIdListFromMaterialHead(pHitMaterial, &ptl);

    // get arg data pointers from material
    //
    __global const int* head    = (__global const int*)pHitMaterial;
    __global const int* table   = head  + as_int(pHitMaterial->data[PROC_TEX_TABLE_OFFSET]);
    __global const int* argdata = table + PLAIN_MATERIAL_DATA_SIZE;
    __global const float* fdata = (__global const float*)argdata;

    ptl.fdata4[0] = make_float3(0, 0, 1);
    ptl.fdata4[1] = make_float3(0, 0, 1);
    ptl.fdata4[2] = make_float3(0, 0, 1);
    ptl.fdata4[3] = make_float3(0, 0, 1);
    ptl.fdata4[4] = make_float3(0, 0, 1);

    // (3) evaluate all proc textures
    //
    //#PUT_YOUR_PROCEDURAL_TEXTURES_EVAL_HERE:

    // (4) take what we need from all array
    //
    WriteProcTextureList(out_procTexData, tid, iNumElements, &ptl);
  }



} 



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

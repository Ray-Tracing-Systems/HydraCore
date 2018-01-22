#include "globals.h"
#include "cfetch.h"
#include "ctrace.h"
#include "crandom.h"

__kernel void InitRandomGen(__global RandomGen* restrict out_gens, int a_seed, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_gens[tid] = RandomGenInit(a_seed + tid);
}

__kernel void FillColorTest(__global float4* restrict color, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  color[tid] = make_float4(1, 1, 0, 0);
}
 

__kernel void BVH4TraversalKernel(__global const float4* restrict rpos,     __global const float4* restrict  rdir, 
                                  __global const float4* restrict a_bvh,    __global const float4* restrict  a_tris,
                                  __global const uint*   restrict in_flags, __global Lite_Hit*     restrict  out_hits, int iRunId, int iNumElements)
{
  const int tid     = GLOBAL_ID_X;
  const int tid2    = (tid < iNumElements) ? tid : iNumElements - 1;
  const uint flags  = in_flags[tid2];
  const bool active = (rayIsActiveU(flags) && (tid < iNumElements));

  if (active)
  {
    const float3 ray_pos = to_float3(rpos[tid]); 
    const float3 ray_dir = to_float3(rdir[tid]);

    Lite_Hit liteHit = out_hits[tid];

    liteHit = (iRunId == 0) ? Make_Lite_Hit(MAXFLOAT, -1) : liteHit;
    liteHit = BVH4Traverse(ray_pos, ray_dir, 0.0f, liteHit, a_bvh, a_tris); // abab 14:53

    out_hits[tid] = liteHit;   // store final result
  }
  

}

__kernel void BVH4TraversalInstKernel(__global const float4* restrict  rpos,     __global const float4* restrict  rdir, 
                                      __global const float4* restrict  a_bvh,    __global const float4* restrict  a_tris,
                                      __global const uint*   restrict  in_flags, __global Lite_Hit*     restrict  out_hits, int iRunId, int iNumElements)
{
  const int tid     = GLOBAL_ID_X;
  const int tid2    = (tid < iNumElements) ? tid : iNumElements - 1;
  const uint flags  = in_flags[tid2];
  const bool active = (rayIsActiveU(flags) && (tid < iNumElements));

  if (active)
  {
    const float3 ray_pos = to_float3(rpos[tid]); 
    const float3 ray_dir = to_float3(rdir[tid]); 

    Lite_Hit liteHit = out_hits[tid];

    liteHit = (iRunId == 0) ? Make_Lite_Hit(MAXFLOAT, -1) : liteHit;
    liteHit = BVH4InstTraverse(ray_pos, ray_dir, 0.0f, liteHit, a_bvh, a_tris);

    out_hits[tid] = liteHit;   // store final result
  }

}

__kernel void BVH4TraversalInstKernelA(__global const float4* restrict  rpos,     __global const float4* restrict  rdir, 
                                       __global const float4* restrict  a_bvh,    __global const float4* restrict  a_tris, __global const uint2*  restrict a_alpha,  
                                       __global const float4* restrict  a_texStorage, __global const EngineGlobals* restrict a_globals,
                                       __global const uint*   restrict  in_flags, __global Lite_Hit*     restrict  out_hits, int iRunId, int iNumElements)
{
  const int tid     = GLOBAL_ID_X;
  const int tid2    = (tid < iNumElements) ? tid : iNumElements - 1;
  const uint flags  = in_flags[tid2];
  const bool active = (rayIsActiveU(flags) && (tid < iNumElements));

  if (active)
  {
    const float3 ray_pos = to_float3(rpos[tid]); 
    const float3 ray_dir = to_float3(rdir[tid]); 

    Lite_Hit liteHit = out_hits[tid];

    liteHit = (iRunId == 0) ? Make_Lite_Hit(MAXFLOAT, -1) : liteHit;
    liteHit = BVH4InstTraverseAlpha(ray_pos, ray_dir, 0.0f, liteHit, a_bvh, a_tris, a_alpha, a_texStorage, a_globals);

    out_hits[tid] = liteHit;   // store final result
  }

}


__kernel void BVH4TraversalInstKernelAS(__global const float4* restrict  rpos,     __global const float4* restrict  rdir, 
                                        __global const float4* restrict  a_bvh,    __global const float4* restrict  a_tris, __global const uint2*  restrict a_alpha,  
                                        __global const float4* restrict  a_texStorage, __global const EngineGlobals* restrict a_globals,
                                        __global const uint*   restrict  in_flags, __global Lite_Hit* restrict  out_hits, __global RandomGen* restrict out_gens,
                                        int iRunId, int iNumElements)
{
  const int tid     = GLOBAL_ID_X;
  const int tid2    = (tid < iNumElements) ? tid : iNumElements - 1;
  const uint flags  = in_flags[tid2];
  const bool active = (rayIsActiveU(flags) && (tid < iNumElements));

  if (active)
  {
    const float3 ray_pos = to_float3(rpos[tid]); 
    const float3 ray_dir = to_float3(rdir[tid]); 

    RandomGen rgen = out_gens[tid];

    Lite_Hit liteHit = out_hits[tid];

    liteHit = (iRunId == 0) ? Make_Lite_Hit(MAXFLOAT, -1) : liteHit;
    liteHit = BVH4InstTraverseAlphaS(ray_pos, ray_dir, 0.0f, liteHit, &rgen, a_bvh, a_tris, a_alpha, a_texStorage, a_globals);

    out_hits[tid] = liteHit;   // store final result
    out_gens[tid] = rgen;
  }

}

static inline float4x4 fetchMatrix(const Lite_Hit hit, __global const float4* in_matrices)
{
  float4x4 res;
  res.row[0] = in_matrices[hit.instId * 4 + 0];
  res.row[1] = in_matrices[hit.instId * 4 + 1];
  res.row[2] = in_matrices[hit.instId * 4 + 2];
  res.row[3] = in_matrices[hit.instId * 4 + 3];
  return res;
}

__kernel void ComputeHit(__global const float4*   restrict rpos, 
                         __global const float4*   restrict rdir, 
                         __global const Lite_Hit* restrict in_hits,

                         __global const float4*   restrict in_matrices,
                         __global const float4*   restrict in_geomStorage,
                         __global const float4*   restrict in_materialStorage,
                         
                         __global uint*           restrict out_flags,
                         __global float4*         restrict out_hitPosNorm,
                         __global float2*         restrict out_hitTexCoord,
                         __global uint*           restrict out_flatNorm,
                         __global HitMatRef*      restrict out_matData,
                         __global Hit_Part4*      restrict out_hitTangent,
                         __global float4*         restrict out_normalsFull,

                         __global const EngineGlobals* restrict a_globals,
                         int a_size)
{
  ///////////////////////////////////////////////////////////
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  uint flags = out_flags[tid];
  if (!rayIsActiveU(flags))
    return;
  ///////////////////////////////////////////////////////////

  Lite_Hit hit = in_hits[tid];

  if (HitNone(hit))
  {
    HitMatRef data3;
    data3.m_data        = -1; // ray is out of scene (or rc alpha range)
    data3.accumDist     = 0.0f;
    out_matData[tid]    = data3;
    out_hitPosNorm[tid] = make_float4(0, 0, 0, 0);

    uint rayOtherFlags = unpackRayFlags(flags);
    rayOtherFlags     |= RAY_GRAMMAR_OUT_OF_SCENE;
    out_flags[tid]     = packRayFlags(flags, rayOtherFlags);
    return;
  }

  float3 ray_pos = to_float3(rpos[tid]);
  float3 ray_dir = to_float3(rdir[tid]);

  // hit data
  //
  SurfaceHit surfHitWS;
  surfHitWS.pos        = make_float3(0, 0, 0);
  surfHitWS.normal     = normalize(make_float3(0, 1, 1));
  surfHitWS.flatNormal = make_float3(0, 0, 1);
  surfHitWS.tangent    = make_float3(0, 0, 0);
  surfHitWS.biTangent  = make_float3(0, 0, 0);
  surfHitWS.texCoord   = make_float2(0, 0);
  surfHitWS.matId      = -1;
  surfHitWS.t          = 0.0f;
  float hit_poly_size  = 0.0f;
  // \\ hit data

  {
    // (1) mul ray with instanceMatrixInv
    //
    const float4x4 instanceMatrixInv = fetchMatrix(hit, in_matrices);
  
    const float3 rayPosLS = mul4x3(instanceMatrixInv, ray_pos);
    const float3 rayDirLS = mul3x3(instanceMatrixInv, ray_dir);
  
    // (2) get pointer to PlainMesh via hit.geomId
    //
    __global const PlainMesh* mesh = fetchMeshHeader(hit, in_geomStorage, a_globals);
    
    // (3) intersect transformed ray with triangle and get SurfaceHit in local space
    //
    const SurfaceHit surfHit = surfaceEvalLS(rayPosLS, rayDirLS, hit, mesh);
    
    // (4) get transformation to wold space
    //
    const float4x4 instanceMatrix = inverse4x4(instanceMatrixInv);
    
    // (5) transform SurfaceHit to world space with instanceMatrix
    //
    surfHitWS = surfHit;
    
    surfHitWS.pos        = mul4x3(instanceMatrix, surfHit.pos);
    surfHitWS.normal     = normalize( mul3x3(instanceMatrix, surfHit.normal));
    surfHitWS.flatNormal = normalize( mul3x3(instanceMatrix, surfHit.flatNormal));
    surfHitWS.tangent    = normalize( mul3x3(instanceMatrix, surfHit.tangent));
    surfHitWS.biTangent  = normalize( mul3x3(instanceMatrix, surfHit.biTangent));
    surfHitWS.t          = length(surfHitWS.pos - ray_pos); // seems this is more precise. VERY strange !!!
  } 

  //__global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_materialStorage, surfHitWS.matId);
  //const float boundingSphereRadius = a_globals->varsF[HRT_BSPHERE_RADIUS];                             // #TODO: fix
  //curvedSurvace = (length(A_norm - B_norm) + length(A_norm - C_norm)) > (0.001f*boundingSphereRadius); // #TODO: fix
  //const bool  curvedSurvace   = true;
  //const bool  transparent     = materialHasTransparency(pHitMaterial);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// THIS IS FUCKING CRAZY !!!!
  {
    int rayOtherFlags = unpackRayFlags(flags);
    rayOtherFlags = rayOtherFlags & (~RAY_HIT_SURFACE_FROM_OTHER_SIDE);
    if (surfHitWS.hfi)
      rayOtherFlags |= RAY_HIT_SURFACE_FROM_OTHER_SIDE;
    out_flags[tid] = packRayFlags(flags, rayOtherFlags);
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  out_hitPosNorm [tid] = to_float4(surfHitWS.pos, as_float(encodeNormal(surfHitWS.normal)));
  out_hitTexCoord[tid] = surfHitWS.texCoord;
  out_normalsFull[tid] = to_float4(surfHitWS.normal, hit_poly_size);

  HitMatRef data3 = out_matData[tid];
  {
    SetMaterialId(&data3, surfHitWS.matId);
    if (unpackBounceNum(flags) == 0)
      data3.accumDist = hit.t;
    else
      data3.accumDist += hit.t;
  }

  Hit_Part4 data4;
  data4.tangentCompressed   = encodeNormal(surfHitWS.tangent);
  data4.bitangentCompressed = encodeNormal(surfHitWS.biTangent);

  out_matData   [tid] = data3;
  out_hitTangent[tid] = data4;
  out_flatNorm  [tid] = encodeNormal(surfHitWS.flatNormal);
}



IDH_CALL bool directLightFromPhotonMap(uint flags, uint gflags, __global const int* a_varsI)
{ 
  uint rayBounceNum  = unpackBounceNum(flags);
  uint diffBounceNum = unpackBounceNumDiff(flags);
  uint otherFlags    = unpackRayFlags(flags);

  return (a_varsI[HRT_PHOTONS_STORE_BOUNCE] == 0) && (diffBounceNum >= a_varsI[HRT_PHOTONS_GARTHER_BOUNCE]);
}


IDH_CALL ushort4 compressShadow(float3 shadow)
{
  ushort4 shadowCompressed;

  shadowCompressed.x = (ushort)(65535.0f * shadow.x);
  shadowCompressed.y = (ushort)(65535.0f * shadow.y);
  shadowCompressed.z = (ushort)(65535.0f * shadow.z);
  shadowCompressed.w = 0;

  return shadowCompressed;
}

IDH_CALL float3 decompressShadow(ushort4 shadowCompressed)
{
  const float invNormCoeff = 1.0f / 65535.0f;
  return invNormCoeff*make_float3((float)shadowCompressed.x, (float)shadowCompressed.y, (float)shadowCompressed.z);
}

__kernel void NoShadow(__global ushort4* a_shadow, int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  a_shadow[tid] = compressShadow(make_float3(1.0f,1.0f,1.0f));
}


__kernel void BVH4TraversalShadowKenrel(__global const uint*         restrict in_flags,
                                        __global const float4*       restrict in_sraypos,
                                        __global const float4*       restrict in_sraydir,
                                        __global       ushort4*      restrict a_shadow,
                                       int a_runId,    
                                       int a_size,                    
                                       __global const float4*        restrict a_bvh,
                                       __global const float4*        restrict a_tris,
                                       __global const EngineGlobals* restrict a_globals)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid];
  
  ////#ifdef RAYTR_THREAD_COMPACTION  // gives 1% on NV
  //__local int l_Data[512];
  //{
  //  int idata = rayIsActiveU(flags) ? 1 : 0;
  //  int odata = 0;
  //
  //  PREFIX_SUMM_MACRO(idata, odata, l_Data, 256);
  //
  //  SYNCTHREADS_LOCAL;
  //  if (idata != 0 && (odata - 1) < 256)
  //    l_Data[odata - 1] = LOCAL_ID_X;
  //  SYNCTHREADS_LOCAL;
  //
  //  int localOffset = l_Data[LOCAL_ID_X];
  //
  //  tid   = ((tid >> 8) << 8) + localOffset;
  //  flags = in_flags[tid];
  //
  //  activeAfterCompaction = (LOCAL_ID_X == 0) || (localOffset != 0);
  //}
  ////#endif

  bool disableThread = !rayIsActiveU(flags);
  
  const bool wasGlossyOrDiffuse = (unpackRayFlags(flags) & RAY_EVENT_G) || (unpackRayFlags(flags) & RAY_EVENT_D);
  const uint rayBounceNum       = unpackBounceNum(flags);

  if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
  {
    if (wasGlossyOrDiffuse && rayBounceNum >= 1)
      disableThread = true;
  }

  if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
  {
    if (!wasGlossyOrDiffuse)
      disableThread = true;
  }
 
  const int a_gflags = a_globals->g_flags;

  if (!disableThread && activeAfterCompaction)
  {
    const float4 data1        = in_sraypos[tid];
    const float4 data2        = in_sraydir[tid];
    const float3 shadowRayPos = to_float3(data1);
    const float3 shadowRayDir = to_float3(data2);
    const float  maxDist      = data1.w;

    const Lite_Hit hit  = BVH4Traverse(shadowRayPos, shadowRayDir, 0.0f, Make_Lite_Hit(maxDist, -1), a_bvh, a_tris);
    const float3 shadow = (HitSome(hit) && hit.t > 0.0f && hit.t < maxDist*0.999f) ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(1.0f, 1.0f, 1.0f);
    
    a_shadow[tid] = compressShadow(shadow);
  }

}

__kernel void BVH4TraversalInstShadowKenrel(__global const uint*         restrict in_flags,
                                            __global const float4*       restrict in_sraypos,
                                            __global const float4*       restrict in_sraydir,
                                            __global       ushort4*      restrict a_shadow,
                                            int a_runId,
                                            int a_size,                    
                                            __global const float4*        restrict a_bvh,
                                            __global const float4*        restrict a_tris,
                                            __global const EngineGlobals* restrict a_globals)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid];

  bool disableThread = !rayIsActiveU(flags);
  
  const bool wasGlossyOrDiffuse = (unpackRayFlags(flags) & RAY_EVENT_G) || (unpackRayFlags(flags) & RAY_EVENT_D);
  const uint rayBounceNum       = unpackBounceNum(flags);

  if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
  {
    if (wasGlossyOrDiffuse && rayBounceNum >= 1)
      disableThread = true;
  }

  if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
  {
    if (!wasGlossyOrDiffuse)
      disableThread = true;
  }
 
  const int a_gflags = a_globals->g_flags;

  if (a_runId > 0)
  {
    const float3 shadowOld = decompressShadow(a_shadow[tid]);
    disableThread = disableThread || (dot(shadowOld, shadowOld) < 0.001f);
  }

  if (!disableThread && activeAfterCompaction)
  {
    const float4 data1        = in_sraypos[tid];
    const float4 data2        = in_sraydir[tid];
    const float3 shadowRayPos = to_float3(data1);
    const float3 shadowRayDir = to_float3(data2);
    const float  maxDist      = data1.w;

    const Lite_Hit hit  = Make_Lite_Hit(maxDist*0.999f, -1);
    const float3 shadow = BVH4InstTraverseShadow(shadowRayPos, shadowRayDir, 0.0f, hit, a_bvh, a_tris);
                        
    a_shadow[tid] = compressShadow(shadow);
  }

}

__kernel void BVH4TraversalInstShadowKenrelAS(__global const uint*         restrict in_flags,
                                              __global const float4*       restrict in_sraypos,
                                              __global const float4*       restrict in_sraydir,
                                              __global       ushort4*      restrict a_shadow,
                                              int a_runId,
                                              int a_size,                    
                                              __global const float4*        restrict a_bvh,
                                              __global const float4*        restrict a_tris,
                                              __global const uint2*         restrict a_alpha,
                                              __global const float4*        restrict a_texStorage, 
                                              __global const EngineGlobals* restrict a_globals)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid];

  bool disableThread = !rayIsActiveU(flags);
  
  const bool wasGlossyOrDiffuse = (unpackRayFlags(flags) & RAY_EVENT_G) || (unpackRayFlags(flags) & RAY_EVENT_D);
  const uint rayBounceNum       = unpackBounceNum(flags);

  if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
  {
    if (wasGlossyOrDiffuse && rayBounceNum >= 1)
      disableThread = true;
  }

  if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
  {
    if (!wasGlossyOrDiffuse)
      disableThread = true;
  }
 
  const int a_gflags = a_globals->g_flags;

  if(a_runId > 0)
  { 
    const float3 shadowOld = decompressShadow(a_shadow[tid]);
    disableThread = disableThread || (dot(shadowOld, shadowOld) < 0.001f);
  }

  if (!disableThread && activeAfterCompaction)
  {
    const float4 data1        = in_sraypos[tid];
    const float4 data2        = in_sraydir[tid];
    const float3 shadowRayPos = to_float3(data1);
    const float3 shadowRayDir = to_float3(data2);
    const float  maxDist      = data1.w;

    const float3 shadow = BVH4InstTraverseShadowAlphaS(shadowRayPos, shadowRayDir, 0.0f, maxDist*0.999f, a_bvh, a_tris, a_alpha, a_texStorage, a_globals);
    
    a_shadow[tid] = compressShadow(shadow);
  }

}



__kernel void ShowNormals(__global float4* in_posNorm, __global float4* a_color, int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float3 norm = normalize( decodeNormal(as_int(in_posNorm[tid].w)) );

  norm.x = fabs(norm.x);
  norm.y = fabs(norm.y);
  norm.z = fabs(norm.z);

  a_color[tid + iOffset] = to_float4(norm, 0.0f);
}



__kernel void ShowTexCoord(__global float2* in_texCoord, __global float4* a_color, int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float2 texCoord = in_texCoord[tid];

  a_color[tid + iOffset] = make_float4(texCoord.x, texCoord.y, 0.0f, 0.0f);
}

__kernel void ColorIndexTriangles(__global Lite_Hit* in_hits, 
                                  __global float4*   a_color, 
                                  int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  Lite_Hit hit = in_hits[tid];
  float4 color = make_float4(0, 0, 0, 0);

  if (HitSome(hit))
  {
    
    switch(hit.primId % 8)
    {
    case 0: color = make_float4(1, 0, 0, 0); break;
    case 1: color = make_float4(0, 1, 0, 0); break;
    case 2: color = make_float4(0, 0, 1, 0); break;
    case 3: color = make_float4(1, 1, 0, 0); break;
    case 4: color = make_float4(1, 0, 1, 0); break;
    case 5: color = make_float4(1, 1, 1, 0); break;
    case 6: color = make_float4(1, 0.5, 0.5, 0); break;
    case 7: color = make_float4(1, 0.5, 0, 0); break;
    default: break;
    };
    
    /*
    if (sizeof(HitPosNorm) == sizeof(float4))
      color = make_float4(1, 0, 0, 0);
    else
      color = make_float4(0, 0, 1, 0);
    */
  } 

  a_color[tid + iOffset] = color;
  
}

/*
__kernel void ReadDiffuseColor(__global const float4*    a_rdir, 
                               __global const Lite_Hit*  in_hits,
                               __global const float4*    in_posNorm,
                               __global const float2*    in_texCoord,
                               __global const HitMatRef* in_matData,

                               texture2d_t                   a_shadingTexture,
                               __global const EngineGlobals* a_globals, 
                               __global float4* a_color, int iNumElements)
{
  __global const float4* a_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float3 color = make_float3(0, 0, 0);
  Lite_Hit hit = in_hits[tid];

  if (HitSome(hit))
  {
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mtlStorage, GetMaterialId(in_matData[tid]));

    float2 txcrd = in_texCoord[tid];
    float3 norm  = normalize(decodeNormal(as_int(in_posNorm[tid].w)));

    color = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), norm, txcrd, a_globals, a_shadingTexture);
  }

  a_color[tid] = to_float4(color, 0.0f);
}

__kernel void GetGBufferFirstBounce(__global const uint*      a_flags,
                                    __global const float4*    a_rdir, 
                                    __global const Lite_Hit*  in_hits, 
                                    __global const float4*    in_posNorm,
                                    __global const float2*    in_texCoord,
                                    __global const HitMatRef* in_matData,

                                    texture2d_t                   a_shadingTexture,
                                    __global const EngineGlobals* a_globals,
                                    __global float4*              a_color, 
                                    int iNumElements)
{
  __global const float4* a_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint     flags     = a_flags[tid];
  Lite_Hit hit       = in_hits[tid];
  float3   norm      = normalize(decodeNormal(as_int(in_posNorm[tid].w)));
  float3   diffColor = make_float3(0,0,0);
  int      matIndex  = -1;

  if (HitNone(hit))
  {
    norm = make_float3(0, 0, 0);
    hit.t = 1000000000.0f;
  }
  else
  {
    matIndex = GetMaterialId(in_matData[tid]);
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mtlStorage, matIndex);

    float2 txcrd = in_texCoord[tid];

    diffColor = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), norm, txcrd, a_globals, a_shadingTexture);
  }

  float alpha = (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE) ? 0.0f : 1.0f;

  GBuffer1 buffData;
  buffData.depth = hit.t;
  buffData.norm  = norm;
  buffData.matId = matIndex;
  buffData.rgba  = to_float4(diffColor, alpha);
  a_color[tid]   = packGBuffer1(buffData);
}

*/

// change 20.01.2018 16:30;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cglobals.h"
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


__kernel void ComputeHit(__global const float4*   restrict rpos, 
                         __global const float4*   restrict rdir, 
                         __global const Lite_Hit* restrict in_hits,

                         __global const float4*   restrict in_matrices,
                         __global const float4*   restrict in_geomStorage,
                         __global const float4*   restrict in_materialStorage,
                         
                         __global const int*      restrict in_allMatRemapLists,
                         __global const int2*     restrict in_remapTable,
                         __global const int*      restrict in_remapInst,

                         __global uint*           restrict out_flags,
                         __global float4*         restrict out_surfaceHit,

                         __global const EngineGlobals* restrict a_globals,
                         int a_remapTableSize, int a_totalInstNumber,  int a_size)
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
    WriteSurfaceHitMatId(-1, tid, a_size, 
                         out_surfaceHit);

    uint rayOtherFlags = unpackRayFlags(flags);
    rayOtherFlags     |= RAY_GRAMMAR_OUT_OF_SCENE;
    out_flags[tid]     = packRayFlags(flags, rayOtherFlags);
    return;
  }

  const float3 ray_pos = to_float3(rpos[tid]);
  const float3 ray_dir = to_float3(rdir[tid]);

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
  SurfaceHit surfHitWS = surfHit;
  
  const float multInv            = 1.0f/sqrt(3.0f);
  const float3 shadowStartPos    = multInv*mul3x3(instanceMatrix, make_float3(surfHitWS.sRayOff, surfHitWS.sRayOff, surfHitWS.sRayOff));
  
  // gl_NormalMatrix is transpose(inverse(gl_ModelViewMatrix))
  //
  const float4x4 normalMatrix = transpose(instanceMatrixInv);  // gl_NormalMatrix is transpose(inverse(gl_ModelViewMatrix))
  
  surfHitWS.pos        = mul4x3(instanceMatrix, surfHit.pos);
  surfHitWS.normal     = normalize( ( mul3x3(normalMatrix, surfHit.normal)     ));
  surfHitWS.flatNormal = normalize( ( mul3x3(normalMatrix, surfHit.flatNormal) ));
  surfHitWS.tangent    = normalize( ( mul3x3(normalMatrix, surfHit.tangent)    ));
  surfHitWS.biTangent  = normalize( ( mul3x3(normalMatrix, surfHit.biTangent)  ));
  surfHitWS.t          = length(surfHitWS.pos - ray_pos); // seems this is more precise. VERY strange !!!
  surfHitWS.sRayOff    = length(shadowStartPos);
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// THIS IS FUCKING CRAZY !!!!
  {
    int rayOtherFlags = unpackRayFlags(flags);
    rayOtherFlags = rayOtherFlags & (~RAY_HIT_SURFACE_FROM_OTHER_SIDE);
    if (surfHitWS.hfi)
      rayOtherFlags |= RAY_HIT_SURFACE_FROM_OTHER_SIDE;
    out_flags[tid] = packRayFlags(flags, rayOtherFlags);
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  surfHitWS.matId  = remapMaterialId(surfHitWS.matId, hit.instId, 
                                     in_remapInst, a_totalInstNumber, in_allMatRemapLists, in_remapTable, a_remapTableSize);

  WriteSurfaceHit(&surfHitWS, tid, a_size, 
                  out_surfaceHit);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
                                                          
                                        __global const float4*        restrict a_bvh,
                                        __global const float4*        restrict a_tris,
                                        __global const EngineGlobals* restrict a_globals,
                                        int a_runId, int a_size)
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

  if (!disableThread && activeAfterCompaction)
  { 
    const float4 data1 = in_sraypos[tid];
    const float4 data2 = in_sraydir[tid];
    
    const float3 shadowRayPos = to_float3(data1);
    const float3 shadowRayDir = to_float3(data2);
    const float maxDist       = fabs(data1.w);
    const int targetInstId    = as_int(data2.w);
    

    if (maxDist > 0.0f)
    {
      const Lite_Hit hit  = BVH4Traverse(shadowRayPos, shadowRayDir, 0.0f, Make_Lite_Hit(maxDist, -1), a_bvh, a_tris);
      const float3 shadow = (HitSome(hit) && hit.t > 0.0f && hit.t < maxDist) ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(1.0f, 1.0f, 1.0f);

      a_shadow[tid] = compressShadow(shadow);
    }
    else
      a_shadow[tid] = compressShadow(make_float3(1, 1, 1));
  }

}

__kernel void BVH4TraversalInstShadowKenrel(__global const uint*         restrict in_flags,
                                            __global const float4*       restrict in_sraypos,
                                            __global const float4*       restrict in_sraydir,
                                            __global       ushort4*      restrict a_shadow,

                                            __global const float4*        restrict a_bvh,
                                            __global const float4*        restrict a_tris,
                                            __global const EngineGlobals* restrict a_globals,
                                            int a_runId, int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid];

  bool disableThread = !rayIsActiveU(flags);

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
    const float  maxDist      = fabs(data1.w);
    const int    targetInstId = as_int(data2.w); // single instance id, this is for dirtAO computations; 

    if (maxDist > 0.0f)
    {
      const Lite_Hit hit  = Make_Lite_Hit(maxDist, -1);
      const float3 shadow = BVH4InstTraverseShadow(shadowRayPos, shadowRayDir, 0.0f, hit, a_bvh, a_tris, targetInstId);

      a_shadow[tid] = compressShadow(shadow);
    }
    else
      a_shadow[tid] = compressShadow(make_float3(1, 1, 1));
  }

}

__kernel void BVH4TraversalInstShadowKenrelAS(__global const uint*         restrict in_flags,
                                              __global const float4*       restrict in_sraypos,
                                              __global const float4*       restrict in_sraydir,
                                              __global       ushort4*      restrict a_shadow,
                                                      
                                              __global const float4*        restrict a_bvh,
                                              __global const float4*        restrict a_tris,
                                              __global const uint2*         restrict a_alpha,
                                              __global const float4*        restrict a_texStorage, 
                                              __global const EngineGlobals* restrict a_globals,
                                               int a_runId, int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid];

  bool disableThread = !rayIsActiveU(flags);

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
    const float  maxDist      = fabs(data1.w);
    const int    targetInstId = as_int(data2.w); // single instance id, this is for dirtAO computations; 

    if (maxDist > 0.0f)
    {
      const float3 shadow = BVH4InstTraverseShadowAlphaS(shadowRayPos, shadowRayDir, 0.0f, maxDist, a_bvh, a_tris, a_alpha, a_texStorage, a_globals, targetInstId);
      a_shadow[tid] = compressShadow(shadow);
    }
    else
      a_shadow[tid] = compressShadow(make_float3(1,1,1));
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void BVH4TraversalShadowKenrel_Packed(__global const uint*         restrict in_flags,
                                               __global const float4*       restrict in_sraypos,
                                               __global const int*          restrict in_sraydirI,
                                               __global const int*          restrict in_inst_id,
                                               __global       ushort*       restrict a_shadow,
                
                                              __global const float4*        restrict a_bvh,
                                              __global const float4*        restrict a_tris,
                                              __global const EngineGlobals* restrict a_globals,
                                              int a_runId, int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid / AO_RAYS_PACKED];

  bool disableThread = !rayIsActiveU(flags);

  if (!disableThread && activeAfterCompaction)
  {     
    const float4 data1  = in_sraypos[tid / AO_RAYS_PACKED];
    int targetInstId    = in_inst_id[tid / AO_RAYS_PACKED]; 
    float3 shadowRayPos = to_float3(data1);
    float maxDist       = fabs(data1.w);
    float3 shadowRayDir = decodeNormal(in_sraydirI[tid]);
   
    if (maxDist > 0.0f)
    {
      const Lite_Hit hit  = BVH4Traverse(shadowRayPos, shadowRayDir, 0.0f, Make_Lite_Hit(maxDist, -1), a_bvh, a_tris);
      const float3 shadow = (HitSome(hit) && hit.t > 0.0f && hit.t < maxDist) ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(1.0f, 1.0f, 1.0f);

      a_shadow[tid] = (ushort)fmin(65535.0f*0.333334f*(shadow.x + shadow.y + shadow.z), 65535.0f);
    }
    else
      a_shadow[tid] = 65535;
  }

}



__kernel void BVH4TraversalInstShadowKenrel_Packed(__global const uint*         restrict in_flags,
                                                   __global const float4*       restrict in_sraypos,
                                                   __global const int*          restrict in_sraydirI,
                                                   __global const int*          restrict in_inst_id,
                                                   __global       ushort*       restrict a_shadow,
                                                   
                                                   __global const float4*        restrict a_bvh,
                                                   __global const float4*        restrict a_tris,
                                                   __global const EngineGlobals* restrict a_globals,
                                                   int a_runId, int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid / AO_RAYS_PACKED];

  bool disableThread = !rayIsActiveU(flags);

  if (a_runId > 0)
  {
    float a_shadowVal = (float)(a_shadow[tid])*(1.0f / 65535.0f);
    disableThread = disableThread || (a_shadowVal < 0.001f);
  }

  if (!disableThread && activeAfterCompaction)
  {
    const float4 data1  = in_sraypos[tid / AO_RAYS_PACKED];
    int targetInstId    = in_inst_id[tid / AO_RAYS_PACKED]; 
    float3 shadowRayPos = to_float3(data1);
    float maxDist       = fabs(data1.w);
    float3 shadowRayDir = decodeNormal(in_sraydirI[tid]);

    if (maxDist > 0.0f)
    {
      const Lite_Hit hit  = Make_Lite_Hit(maxDist, -1);
      const float3 shadow = BVH4InstTraverseShadow(shadowRayPos, shadowRayDir, 0.0f, hit, a_bvh, a_tris, targetInstId);

      a_shadow[tid] = (ushort)fmin(65535.0f*0.333334f*(shadow.x + shadow.y + shadow.z), 65535.0f);
    }
    else
      a_shadow[tid] = 65535;
  }

}


__kernel void BVH4TraversalInstShadowKenrelAS_Packed(__global const uint*         restrict in_flags,
                                                     __global const float4*       restrict in_sraypos,
                                                     __global const int*          restrict in_sraydirI,
                                                     __global const int*          restrict in_inst_id,
                                                     __global       ushort*       restrict a_shadow,
                                                     
                                                     __global const float4*        restrict a_bvh,
                                                     __global const float4*        restrict a_tris,
                                                     __global const uint2*         restrict a_alpha,
                                                     __global const float4*        restrict a_texStorage, 
                                                     __global const EngineGlobals* restrict a_globals,
                                                     int a_runId, int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    tid = a_size - 1;

  bool activeAfterCompaction = (GLOBAL_ID_X < a_size);
  uint flags = in_flags[tid / AO_RAYS_PACKED];

  bool disableThread = !rayIsActiveU(flags);

  if(a_runId > 0)
  { 
    float a_shadowVal = (float)(a_shadow[tid])*(1.0f / 65535.0f);
    disableThread = disableThread || (a_shadowVal < 0.001f);
  }

  if (!disableThread && activeAfterCompaction)
  {
    const float4 data1  = in_sraypos[tid / AO_RAYS_PACKED];
    int targetInstId    = in_inst_id[tid / AO_RAYS_PACKED]; 
    float3 shadowRayPos = to_float3(data1);
    float maxDist       = fabs(data1.w);
    float3 shadowRayDir = decodeNormal(in_sraydirI[tid]); 

    if (maxDist > 0.0f)
    {
      const float3 shadow = BVH4InstTraverseShadowAlphaS(shadowRayPos, shadowRayDir, 0.0f, maxDist, a_bvh, a_tris, a_alpha, a_texStorage, a_globals, targetInstId);
      a_shadow[tid] = (ushort)fmin(65535.0f*0.333334f*(shadow.x + shadow.y + shadow.z), 65535.0f);
    }
    else
      a_shadow[tid] = 65535;
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__kernel void ShowNormals(__global const float4* restrict in_surfaceHit, __global float4* restrict a_color, int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &sHit);

  float3 norm = sHit.normal;

  norm.x = fabs(norm.x);
  norm.y = fabs(norm.y);
  norm.z = fabs(norm.z);

  a_color[tid + iOffset] = to_float4(norm, 0.0f);
}

__kernel void ShowTangent(__global const float4* restrict in_surfaceHit, __global float4* restrict a_color, int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &sHit);

  float3 norm = sHit.tangent;

  norm.x = fabs(norm.x);
  norm.y = fabs(norm.y);
  norm.z = fabs(norm.z);

  a_color[tid + iOffset] = to_float4(norm, 0.0f);
}


__kernel void ShowTexCoord(__global const float4* restrict in_surfaceHit, __global float4* a_color, int iNumElements, int iOffset)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &sHit);

  float2 texCoord = sHit.texCoord;

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

// change 22.03.2020 11:08;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

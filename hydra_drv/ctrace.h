#ifndef RTCTRACE
#define RTCTRACE

#include "globals.h"
#include "cfetch.h"
#include "crandom.h"

IDH_CALL bool RayBoxIntersectionLite(float3 ray_pos, float3 ray_dir, float3 boxMin, float3 boxMax, __private float*  tmin, __private float* tmax)
{
  float lo = ray_dir.x*(boxMin.x - ray_pos.x);
  float hi = ray_dir.x*(boxMax.x - ray_pos.x);

  (*tmin) = fmin(lo, hi);
  (*tmax) = fmax(lo, hi);

  float lo1 = ray_dir.y*(boxMin.y - ray_pos.y);
  float hi1 = ray_dir.y*(boxMax.y - ray_pos.y);

  (*tmin) = fmax((*tmin), fmin(lo1, hi1));
  (*tmax) = fmin((*tmax), fmax(lo1, hi1));

  float lo2 = ray_dir.z*(boxMin.z - ray_pos.z);
  float hi2 = ray_dir.z*(boxMax.z - ray_pos.z);

  (*tmin) = fmax((*tmin), fmin(lo2, hi2));
  (*tmax) = fmin((*tmax), fmax(lo2, hi2));

  return ((*tmin) <= (*tmax));
}


IDH_CALL float2 RayBoxIntersectionLite2(float3 ray_pos, float3 ray_dir, float3 boxMin, float3 boxMax)
{
  const float lo  = ray_dir.x*(boxMin.x - ray_pos.x);
  const float hi  = ray_dir.x*(boxMax.x - ray_pos.x);

  const float lo1 = ray_dir.y*(boxMin.y - ray_pos.y);
  const float hi1 = ray_dir.y*(boxMax.y - ray_pos.y);

  const float lo2 = ray_dir.z*(boxMin.z - ray_pos.z);
  const float hi2 = ray_dir.z*(boxMax.z - ray_pos.z);

  float tmin = fmin(lo, hi);
  float tmax = fmax(lo, hi);

  tmin = fmax(tmin, fmin(lo1, hi1));
  tmax = fmin(tmax, fmax(lo1, hi1));

  tmin = fmax(tmin, fmin(lo2, hi2));
  tmax = fmin(tmax, fmax(lo2, hi2));

  return make_float2(tmin, tmax);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static inline Lite_Hit IntersectAllPrimitivesInLeaf1(const float3 ray_pos, const float3 ray_dir,
                                                    const int leaf_offset, const float t_min, 
                                                    Lite_Hit a_result,
                                                  #ifdef USE_1D_TEXTURES
                                                     __read_only image1d_buffer_t a_objListTex,
                                                  #else
                                                     __global const float4* a_objListTex
                                                  #endif
                                                    )
{
  const int2 objectListInfo = getObjectList(leaf_offset, a_objListTex);

  const int NUM_FETCHES_TRI = 3; // sizeof(struct ObjectListTriangle) / sizeof(float4);
  const int triAddressStart = objectListInfo.x; 
  const int triAddressEnd   = triAddressStart + objectListInfo.y*NUM_FETCHES_TRI;
 
  for (int triAddress = triAddressStart; triAddress < triAddressEnd; triAddress += NUM_FETCHES_TRI)
  {
   #ifdef USE_1D_TEXTURES
    const float4 data1 = read_imagef(a_objListTex, triAddress + 0);
    const float4 data2 = read_imagef(a_objListTex, triAddress + 1);
    const float4 data3 = read_imagef(a_objListTex, triAddress + 2);
   #else
    const float4 data1 = a_objListTex[triAddress + 0]; 
    const float4 data2 = a_objListTex[triAddress + 1]; 
    const float4 data3 = a_objListTex[triAddress + 2]; 
   #endif

    const float3 A_pos = to_float3(data1);
    const float3 B_pos = to_float3(data2);
    const float3 C_pos = to_float3(data3);

    const int primId   = as_int(data1.w);
    const int geomId   = as_int(data2.w);
    const int instId   = as_int(data3.w);

    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec  = cross(ray_dir, edge2);
    const float3 tvec  = ray_pos - A_pos;
    const float3 qvec  = cross(tvec, edge1);
    const float invDet = 1.0f / dot(edge1, pvec);

    const float v = dot(tvec, pvec)*invDet;
    const float u = dot(qvec, ray_dir)*invDet;
    const float t = dot(edge2, qvec)*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > t_min && t < a_result.t)
    {
      a_result.t      = t;
      a_result.primId = primId;
      a_result.geomId = geomId;
      a_result.instId = instId;
    }

  }

  return a_result;
}

static inline Lite_Hit IntersectAllPrimitivesInLeaf(const float3 ray_pos, const float3 ray_dir,
                                                    const int leaf_offset, const float t_min, 
                                                    Lite_Hit a_result,
                                                  #ifdef USE_1D_TEXTURES
                                                     __read_only image1d_buffer_t a_objListTex,
                                                  #else
                                                     __global const float4* a_objListTex,
                                                  #endif
                                                     const int a_instId)
{
  const int2 objectListInfo = getObjectList(leaf_offset, a_objListTex);

  const int NUM_FETCHES_TRI = 3; // sizeof(struct ObjectListTriangle) / sizeof(float4);
  const int triAddressStart = objectListInfo.x; 
  const int triAddressEnd   = triAddressStart + objectListInfo.y*NUM_FETCHES_TRI;
 
  for (int triAddress = triAddressStart; triAddress < triAddressEnd; triAddress += NUM_FETCHES_TRI)
  {
   #ifdef USE_1D_TEXTURES
    const float4 data1 = read_imagef(a_objListTex, triAddress + 0);
    const float4 data2 = read_imagef(a_objListTex, triAddress + 1);
    const float4 data3 = read_imagef(a_objListTex, triAddress + 2);
   #else
    const float4 data1 = a_objListTex[triAddress + 0]; 
    const float4 data2 = a_objListTex[triAddress + 1]; 
    const float4 data3 = a_objListTex[triAddress + 2]; 
   #endif

    const float3 A_pos = to_float3(data1);
    const float3 B_pos = to_float3(data2);
    const float3 C_pos = to_float3(data3);

    const int primId   = as_int(data1.w);
    const int geomId   = as_int(data2.w);
    //const int instId   = as_int(data3.w);

    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec  = cross(ray_dir, edge2);
    const float3 tvec  = ray_pos - A_pos;
    const float3 qvec  = cross(tvec, edge1);
    const float invDet = 1.0f / dot(edge1, pvec);

    const float v = dot(tvec, pvec)*invDet;
    const float u = dot(qvec, ray_dir)*invDet;
    const float t = dot(edge2, qvec)*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > t_min && t < a_result.t)
    {
      a_result.t      = t;
      a_result.primId = primId;
      a_result.geomId = geomId;
      a_result.instId = a_instId;
    }

  }

  return a_result;
}


static inline float2 decompressTexCoord16(unsigned int packed)
{
  const unsigned int ix =  packed & 0x0000FFFF;
  const unsigned int iy = (packed & 0xFFFF0000) >> 16;

  const float fx  = (1.0f / 65535.0f)*(float)ix;
  const float fy  = (1.0f / 65535.0f)*(float)iy;

  return make_float2(2.0f*fx - 1.0f, 2.0f*fy - 1.0f);
}

static inline Lite_Hit IntersectAllPrimitivesInLeafAlpha(const float3 ray_pos, const float3 ray_dir,
                                                         const int leaf_offset, const float t_min, 
                                                         Lite_Hit a_result,
                                                         #ifdef USE_1D_TEXTURES
                                                           __read_only image1d_buffer_t a_objListTex,
                                                         #else
                                                           __global const float4* a_objListTex,
                                                         #endif
                                                         const int a_instId,
                                                         __global const uint2* a_alphaTable, __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  const int2 objectListInfo = getObjectList(leaf_offset, a_objListTex);

  const int NUM_FETCHES_TRI = 3; // sizeof(struct ObjectListTriangle) / sizeof(float4);
  const int triAddressStart = objectListInfo.x; 
  const int triAddressEnd   = triAddressStart + objectListInfo.y*NUM_FETCHES_TRI;
 
  for (int triAddress = triAddressStart; triAddress < triAddressEnd; triAddress += NUM_FETCHES_TRI)
  {
   #ifdef USE_1D_TEXTURES
    const float4 data1 = read_imagef(a_objListTex, triAddress + 0);
    const float4 data2 = read_imagef(a_objListTex, triAddress + 1);
    const float4 data3 = read_imagef(a_objListTex, triAddress + 2);
   #else
    const float4 data1 = a_objListTex[triAddress + 0]; 
    const float4 data2 = a_objListTex[triAddress + 1]; 
    const float4 data3 = a_objListTex[triAddress + 2]; 
   #endif

    const float3 A_pos = to_float3(data1);
    const float3 B_pos = to_float3(data2);
    const float3 C_pos = to_float3(data3);

    const int primId   = as_int(data1.w);
    const int geomId   = as_int(data2.w);
    //const int instId   = as_int(data3.w);

    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec  = cross(ray_dir, edge2);
    const float3 tvec  = ray_pos - A_pos;
    const float3 qvec  = cross(tvec, edge1);
    const float invDet = 1.0f / dot(edge1, pvec);

    const float v = dot(tvec, pvec)*invDet;
    const float u = dot(qvec, ray_dir)*invDet;
    const float t = dot(edge2, qvec)*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > t_min && t < a_result.t)
    {
      const uint2 alphaId0 = a_alphaTable[triAddress+0];
      const uint2 alphaId1 = a_alphaTable[triAddress+1];
      const uint2 alphaId2 = a_alphaTable[triAddress+2];
      
      const float2 A_tex   = decompressTexCoord16(alphaId0.y);
      const float2 B_tex   = decompressTexCoord16(alphaId1.y);
      const float2 C_tex   = decompressTexCoord16(alphaId2.y);
      
      const float2 texCoord   = (1.0f - u - v)*A_tex + v*B_tex + u*C_tex;
      const int samplerOffset = (alphaId0.x == 0xFFFFFFFF || alphaId0.x == INVALID_TEXTURE || (int)(alphaId0.x) <= 0) ? INVALID_TEXTURE : 0;

      const float3 alphaColor = sample2DLite(samplerOffset, texCoord, (a_alphaTable + alphaId0.x), a_texStorage, a_globals);
      const float selector    = fmax(alphaColor.x, fmax(alphaColor.y, alphaColor.z));

      if (selector > 0.5f)
      {
        a_result.t      = t;
        a_result.primId = primId;
        a_result.geomId = geomId;
        a_result.instId = a_instId;
      }
    }

  }

  return a_result;
}



static inline Lite_Hit IntersectAllPrimitivesInLeafAlphaS(const float3 ray_pos, const float3 ray_dir,
                                                          const int leaf_offset, const float t_min, 
                                                          Lite_Hit a_result, __private RandomGen* pGen,
                                                          #ifdef USE_1D_TEXTURES
                                                            __read_only image1d_buffer_t a_objListTex,
                                                          #else
                                                            __global const float4* a_objListTex,
                                                          #endif
                                                          const int a_instId,
                                                          __global const uint2* a_alphaTable, __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  const int2 objectListInfo = getObjectList(leaf_offset, a_objListTex);

  const int NUM_FETCHES_TRI = 3; // sizeof(struct ObjectListTriangle) / sizeof(float4);
  const int triAddressStart = objectListInfo.x; 
  const int triAddressEnd   = triAddressStart + objectListInfo.y*NUM_FETCHES_TRI;
 
  for (int triAddress = triAddressStart; triAddress < triAddressEnd; triAddress += NUM_FETCHES_TRI)
  {
   #ifdef USE_1D_TEXTURES
    const float4 data1 = read_imagef(a_objListTex, triAddress + 0);
    const float4 data2 = read_imagef(a_objListTex, triAddress + 1);
    const float4 data3 = read_imagef(a_objListTex, triAddress + 2);
   #else
    const float4 data1 = a_objListTex[triAddress + 0]; 
    const float4 data2 = a_objListTex[triAddress + 1]; 
    const float4 data3 = a_objListTex[triAddress + 2]; 
   #endif

    const float3 A_pos = to_float3(data1);
    const float3 B_pos = to_float3(data2);
    const float3 C_pos = to_float3(data3);

    const int primId   = as_int(data1.w);
    const int geomId   = as_int(data2.w);
    //const int instId   = as_int(data3.w);

    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec  = cross(ray_dir, edge2);
    const float3 tvec  = ray_pos - A_pos;
    const float3 qvec  = cross(tvec, edge1);
    const float invDet = 1.0f / dot(edge1, pvec);

    const float v = dot(tvec, pvec)*invDet;
    const float u = dot(qvec, ray_dir)*invDet;
    const float t = dot(edge2, qvec)*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > t_min && t < a_result.t)
    {
      const uint2 alphaId0 = a_alphaTable[triAddress+0];
      const uint2 alphaId1 = a_alphaTable[triAddress+1];
      const uint2 alphaId2 = a_alphaTable[triAddress+2];

      const float2 A_tex = decompressTexCoord16(alphaId0.y);
      const float2 B_tex = decompressTexCoord16(alphaId1.y);
      const float2 C_tex = decompressTexCoord16(alphaId2.y);
      
      const float2 texCoord   = (1.0f - u - v)*A_tex + v*B_tex + u*C_tex;
      const int samplerOffset = (alphaId0.x == 0xFFFFFFFF || alphaId0.x == INVALID_TEXTURE || (int)(alphaId0.x) <= 0) ? INVALID_TEXTURE : 0;
      const float3 alphaColor = sample2DUI2(samplerOffset, texCoord, (a_alphaTable + alphaId0.x), a_texStorage, a_globals);

      const float selector = fmax(alphaColor.x, fmax(alphaColor.y, alphaColor.z));

      bool acceptHit = (selector > 0.5f); 
      if (alphaId1.x == 1) // smooth opacity enabled
        acceptHit = (rndFloat1_Pseudo(pGen) < selector);

      if (acceptHit)
      {
        a_result.t      = t;
        a_result.primId = primId;
        a_result.geomId = geomId;
        a_result.instId = a_instId;
      }
    }

  }

  return a_result;
}



static inline void IntersectAllPrimitivesInLeafShadowAlphaS(const float3 ray_pos, const float3 ray_dir,
                                                            const int leaf_offset, const float t_min, 
                                                            const float t_max, __private float* pShadow,
                                                            #ifdef USE_1D_TEXTURES
                                                              __read_only image1d_buffer_t a_objListTex,
                                                            #else
                                                              __global const float4* a_objListTex,
                                                            #endif
                                                            const int a_instId,
                                                            __global const uint2* a_alphaTable, __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  const int2 objectListInfo = getObjectList(leaf_offset, a_objListTex);

  const int NUM_FETCHES_TRI = 3; // sizeof(struct ObjectListTriangle) / sizeof(float4);
  const int triAddressStart = objectListInfo.x; 
  const int triAddressEnd   = triAddressStart + objectListInfo.y*NUM_FETCHES_TRI;
 
  for (int triAddress = triAddressStart; triAddress < triAddressEnd; triAddress += NUM_FETCHES_TRI)
  {
   #ifdef USE_1D_TEXTURES
    const float4 data1 = read_imagef(a_objListTex, triAddress + 0);
    const float4 data2 = read_imagef(a_objListTex, triAddress + 1);
    const float4 data3 = read_imagef(a_objListTex, triAddress + 2);
   #else
    const float4 data1 = a_objListTex[triAddress + 0]; 
    const float4 data2 = a_objListTex[triAddress + 1]; 
    const float4 data3 = a_objListTex[triAddress + 2]; 
   #endif

    const float3 A_pos = to_float3(data1);
    const float3 B_pos = to_float3(data2);
    const float3 C_pos = to_float3(data3);

    const int primId   = as_int(data1.w);
    const int geomId   = as_int(data2.w);
    //const int instId   = as_int(data3.w);

    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec  = cross(ray_dir, edge2);
    const float3 tvec  = ray_pos - A_pos;
    const float3 qvec  = cross(tvec, edge1);
    const float invDet = 1.0f / dot(edge1, pvec);

    const float v = dot(tvec, pvec)*invDet;
    const float u = dot(qvec, ray_dir)*invDet;
    const float t = dot(edge2, qvec)*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > t_min && t < t_max)
    {
      const uint2 alphaId0 = a_alphaTable[triAddress+0];
      const uint2 alphaId1 = a_alphaTable[triAddress+1];
      const uint2 alphaId2 = a_alphaTable[triAddress+2];

      const float2 A_tex = decompressTexCoord16(alphaId0.y);
      const float2 B_tex = decompressTexCoord16(alphaId1.y);
      const float2 C_tex = decompressTexCoord16(alphaId2.y);
      
      const float2 texCoord   = (1.0f - u - v)*A_tex + v*B_tex + u*C_tex;
      const int samplerOffset = (alphaId0.x == 0xFFFFFFFF || alphaId0.x == INVALID_TEXTURE || (int)(alphaId0.x) <= 0 || alphaId2.x == 1) ? INVALID_TEXTURE : 0;
      const float3 alphaColor = sample2DUI2(samplerOffset, texCoord, (a_alphaTable + alphaId0.x), a_texStorage, a_globals);

      const float selector    = fmax(alphaColor.x, fmax(alphaColor.y, alphaColor.z));

      if (alphaId2.x != 1)   // skip shadow
      {
        if (alphaId1.x == 1) // smooth opacity enabled
          (*pShadow) *= (1.0f - selector);
        else
        {
          const float mult = (selector > 0.5f) ? 0.0f : 1.0f;
          (*pShadow) *= mult;
        }
      }
    }

  }

}



#define STACK_SIZE 80

ID_CALL unsigned int BVHTraversal(float3 ray_pos, float3 ray_dir, float t_rayMin, __private Lite_Hit* pHit, 

                              #ifdef USE_1D_TEXTURES
                                 __read_only image1d_buffer_t  bvhTex,
                                 __read_only image1d_buffer_t  objListTex
                              #else
                                 __global const float4* bvhTex,
                                 __global const float4* objListTex,
                              #endif
                                 const bool shadowRay // indicets if we trace shadow rays or common
                                   )
{
  int  local_stack[STACK_SIZE]; local_stack[0] = 0xFFFFFFFF;
  int* stack = local_stack + 1;

  int top = 0;
  int leftNodeOffset = 1;
  int triAlphaTest = 0xFFFFFFFF;

  bool searchingForLeaf = true;
  float3 invDir = SafeInverse(ray_dir);

  while (top >= 0)
  {
    // BVH traversal
    //
    while (searchingForLeaf)
    {
      float t_minLeft = 0.0f;
      float t_maxLeft = 0.0f;

      BVHNode leftNode         = GetBVHNode(leftNodeOffset, bvhTex);
      bool traverseChild0      = RayBoxIntersectionLite(ray_pos, invDir, leftNode.m_boxMin, leftNode.m_boxMax, &t_minLeft, &t_maxLeft);
      traverseChild0           = traverseChild0 && (t_maxLeft >= t_rayMin) && (t_minLeft <= pHit->t); // t_rayMax is hit.t

      float t_minRight = 0.0f;
      float t_maxRight = 0.0f;

      BVHNode rightNode        = GetBVHNode(leftNodeOffset + 1, bvhTex);
      bool traverseChild1      = RayBoxIntersectionLite(ray_pos, invDir, rightNode.m_boxMin, rightNode.m_boxMax, &t_minRight, &t_maxRight);
      traverseChild1           = traverseChild1 && (t_maxRight >= t_rayMin) && (t_minRight <= pHit->t); // t_rayMax is hit.t

      // traversal decision
      //
      leftNodeOffset = traverseChild0 ? leftNode.m_leftOffsetAndLeaf : rightNode.m_leftOffsetAndLeaf;

      if (traverseChild0 && traverseChild1)
      {
        leftNodeOffset = (t_minLeft <= t_minRight) ? leftNode.m_leftOffsetAndLeaf  : rightNode.m_leftOffsetAndLeaf;
        stack[top]     = (t_minLeft <= t_minRight) ? rightNode.m_leftOffsetAndLeaf : leftNode.m_leftOffsetAndLeaf;
        top++;
      }

      if (!traverseChild0 && !traverseChild1) // both miss, stack.pop()
      {
        top--;
        leftNodeOffset = stack[top];
      }

      searchingForLeaf = !IS_LEAF(leftNodeOffset) && (top >= 0);
      leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);
    }
    

    // intersection in leaf
    //
    if (leftNodeOffset != 0x7fffffff)
    {
      (*pHit) = IntersectAllPrimitivesInLeaf1(ray_pos, ray_dir, leftNodeOffset, t_rayMin, (*pHit), objListTex);
      //if (shadowRay && triAlphaTest == 0xFFFFFFFF && pHit->primId != -1) // this seems to be not strictly correct, but speeds up shadow rays up to 50%
      //  top = 0;
    }
    
    // continue BVH traversal
    //
    top--;
    leftNodeOffset = stack[top];

    searchingForLeaf = !IS_LEAF(leftNodeOffset);
    leftNodeOffset = EXTRACT_OFFSET(leftNodeOffset);
  }
  

  return triAlphaTest;
}


#ifndef MAXFLOAT
#define MAXFLOAT 1e37f
#endif

static inline Lite_Hit BVH4Traverse(const float3 ray_pos, const float3 ray_dir, float t_rayMin, Lite_Hit a_hit, 
                                    __global const float4* a_bvh, __global const float4* a_tris)
{
  const float3 invDir = SafeInverse(ray_dir);

  int  stackData[STACK_SIZE];
  int* stack = stackData+2;

  int top               = 0;
  int leftNodeOffset    = 1;
  bool searchingForLeaf = true;

  while (top >= 0)
  {

    while (searchingForLeaf)
    {
      const BVHNode node0 = GetBVHNode(4 * leftNodeOffset + 0, a_bvh);
      const bool    vald0 = IsValidNode(node0);
      const int     loal0 = node0.m_leftOffsetAndLeaf;
      const float2  tm0   = RayBoxIntersectionLite2(ray_pos, invDir, node0.m_boxMin, node0.m_boxMax);
 
      const BVHNode node1 = GetBVHNode(4 * leftNodeOffset + 1, a_bvh);
      const bool    vald1 = IsValidNode(node1);
      const int     loal1 = node1.m_leftOffsetAndLeaf;
      const float2  tm1   = RayBoxIntersectionLite2(ray_pos, invDir, node1.m_boxMin, node1.m_boxMax);
     
      const BVHNode node2 = GetBVHNode(4 * leftNodeOffset + 2, a_bvh);
      const bool    vald2 = IsValidNode(node2);
      const int     loal2 = node2.m_leftOffsetAndLeaf;
      const float2  tm2   = RayBoxIntersectionLite2(ray_pos, invDir, node2.m_boxMin, node2.m_boxMax);

      const BVHNode node3 = GetBVHNode(4 * leftNodeOffset + 3, a_bvh);
      const bool    vald3 = IsValidNode(node3);
      const int     loal3 = node3.m_leftOffsetAndLeaf;
      const float2  tm3   = RayBoxIntersectionLite2(ray_pos, invDir, node3.m_boxMin, node3.m_boxMax);

      int4 children = make_int4(loal0, loal1, loal2, loal3);

      const bool hitChild0 = (tm0.x <= tm0.y) && (tm0.y >= t_rayMin) && (tm0.x <= a_hit.t) && vald0;
      const bool hitChild1 = (tm1.x <= tm1.y) && (tm1.y >= t_rayMin) && (tm1.x <= a_hit.t) && vald1;
      const bool hitChild2 = (tm2.x <= tm2.y) && (tm2.y >= t_rayMin) && (tm2.x <= a_hit.t) && vald2;
      const bool hitChild3 = (tm3.x <= tm3.y) && (tm3.y >= t_rayMin) && (tm3.x <= a_hit.t) && vald3;

      float4 hitMinD = make_float4(hitChild0 ? tm0.x : MAXFLOAT,
                                   hitChild1 ? tm1.x : MAXFLOAT,
                                   hitChild2 ? tm2.x : MAXFLOAT,
                                   hitChild3 ? tm3.x : MAXFLOAT);

      // sort tHit and children
      //
      const bool lessXY = (hitMinD.y < hitMinD.x);
      const bool lessWZ = (hitMinD.w < hitMinD.z);
      {
        const int   w_childrenX = lessXY ? children.y : children.x;
        const int   w_childrenY = lessXY ? children.x : children.y;
        const float w_hitTimesX = lessXY ? hitMinD.y  : hitMinD.x;
        const float w_hitTimesY = lessXY ? hitMinD.x  : hitMinD.y;

        const int   w_childrenZ = lessWZ ? children.w : children.z;
        const int   w_childrenW = lessWZ ? children.z : children.w;
        const float w_hitTimesZ = lessWZ ? hitMinD.w  : hitMinD.z;
        const float w_hitTimesW = lessWZ ? hitMinD.z  : hitMinD.w;

        children.x = w_childrenX;
        children.y = w_childrenY;
        hitMinD.x  = w_hitTimesX;
        hitMinD.y  = w_hitTimesY;

        children.z = w_childrenZ;
        children.w = w_childrenW;
        hitMinD.z  = w_hitTimesZ;
        hitMinD.w  = w_hitTimesW;
      }

      const bool lessZX = (hitMinD.z < hitMinD.x);
      const bool lessWY = (hitMinD.w < hitMinD.y);
      {
        const int   w_childrenX = lessZX ? children.z : children.x;
        const int   w_childrenZ = lessZX ? children.x : children.z;
        const float w_hitTimesX = lessZX ? hitMinD.z  : hitMinD.x;
        const float w_hitTimesZ = lessZX ? hitMinD.x  : hitMinD.z;

        const int   w_childrenY = lessWY ? children.w : children.y;
        const int   w_childrenW = lessWY ? children.y : children.w;
        const float w_hitTimesY = lessWY ? hitMinD.w  : hitMinD.y;
        const float w_hitTimesW = lessWY ? hitMinD.y  : hitMinD.w;

        children.x = w_childrenX;
        children.z = w_childrenZ;
        hitMinD.x  = w_hitTimesX; 
        hitMinD.z  = w_hitTimesZ; 

        children.y = w_childrenY;
        children.w = w_childrenW;
        hitMinD.y  = w_hitTimesY;
        hitMinD.w  = w_hitTimesW;
      }

      const bool lessZY = (hitMinD.z < hitMinD.y);
      {
        const int   w_childrenY = lessZY ? children.z : children.y;
        const int   w_childrenZ = lessZY ? children.y : children.z;
        const float w_hitTimesY = lessZY ? hitMinD.z  : hitMinD.y;
        const float w_hitTimesZ = lessZY ? hitMinD.y  : hitMinD.z;

        children.y = w_childrenY;
        children.z = w_childrenZ;
        hitMinD.y  = w_hitTimesY;
        hitMinD.z  = w_hitTimesZ;
      }

      const bool stackHaveSpace = (top < STACK_SIZE);

      // push/pop stack games
      //
      if (hitMinD.w < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.w;
        top++;
      }

      if (hitMinD.z < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.z;
        top++;
      }

      if (hitMinD.y < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.y;
        top++;
      }

      if (hitMinD.x < MAXFLOAT)
      {
        leftNodeOffset = children.x;
      }
      else if (top >= 0)
      {
        top--;
        leftNodeOffset = stack[top];
      }

      searchingForLeaf = !IS_LEAF(leftNodeOffset) && (top >= 0);
      leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

    } //  while (searchingForLeaf)

   
    // leaf node, intersect triangles
    //
    if (top >= 0)
    {
      a_hit = IntersectAllPrimitivesInLeaf1(ray_pos, ray_dir, leftNodeOffset, t_rayMin, a_hit, a_tris);
    }

    // pop next node from stack
    //
    top--;
    leftNodeOffset   = stack[top];
    
    searchingForLeaf = !IS_LEAF(leftNodeOffset);
    leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);
    

  } // while (top >= 0)

  return a_hit;
}


static inline Lite_Hit BVH4InstTraverse(float3 ray_pos, float3 ray_dir, float t_rayMin, Lite_Hit a_hit, 
                                        __global const float4* a_bvh, __global const float4* a_tris)
{
  float3 invDir = SafeInverse(ray_dir);

  int  stackData[STACK_SIZE];
  int* stack = stackData + 2;

  int top               = 0;
  int leftNodeOffset    = 1;
  bool searchingForLeaf = true;

  ////////////////////////////////////////////////////////////////////////////// instancing variables
  int instDeep        = 0;
  int instTop         = 0;
  float3 old_ray_pos  = make_float3(0, 0, 0);
  float3 old_ray_dir  = make_float3(0, 0, 0);
  int instId          = -1;
  ////////////////////////////////////////////////////////////////////////////// instancing variables

  while (top >= 0)
  {

    while (searchingForLeaf)
    {
      const BVHNode node0 = GetBVHNode(4 * leftNodeOffset + 0, a_bvh);
      const bool    vald0 = IsValidNode(node0);
      const int     loal0 = node0.m_leftOffsetAndLeaf;
      const float2  tm0   = RayBoxIntersectionLite2(ray_pos, invDir, node0.m_boxMin, node0.m_boxMax);
 
      const BVHNode node1 = GetBVHNode(4 * leftNodeOffset + 1, a_bvh);
      const bool    vald1 = IsValidNode(node1);
      const int     loal1 = node1.m_leftOffsetAndLeaf;
      const float2  tm1   = RayBoxIntersectionLite2(ray_pos, invDir, node1.m_boxMin, node1.m_boxMax);
     
      const BVHNode node2 = GetBVHNode(4 * leftNodeOffset + 2, a_bvh);
      const bool    vald2 = IsValidNode(node2);
      const int     loal2 = node2.m_leftOffsetAndLeaf;
      const float2  tm2   = RayBoxIntersectionLite2(ray_pos, invDir, node2.m_boxMin, node2.m_boxMax);

      const BVHNode node3 = GetBVHNode(4 * leftNodeOffset + 3, a_bvh);
      const bool    vald3 = IsValidNode(node3);
      const int     loal3 = node3.m_leftOffsetAndLeaf;
      const float2  tm3   = RayBoxIntersectionLite2(ray_pos, invDir, node3.m_boxMin, node3.m_boxMax);

      int4 children = make_int4(loal0, loal1, loal2, loal3);

      const bool hitChild0 = (tm0.x <= tm0.y) && (tm0.y >= t_rayMin) && (tm0.x <= a_hit.t) && vald0;
      const bool hitChild1 = (tm1.x <= tm1.y) && (tm1.y >= t_rayMin) && (tm1.x <= a_hit.t) && vald1;
      const bool hitChild2 = (tm2.x <= tm2.y) && (tm2.y >= t_rayMin) && (tm2.x <= a_hit.t) && vald2;
      const bool hitChild3 = (tm3.x <= tm3.y) && (tm3.y >= t_rayMin) && (tm3.x <= a_hit.t) && vald3;

      float4 hitMinD = make_float4(hitChild0 ? tm0.x : MAXFLOAT,
                                   hitChild1 ? tm1.x : MAXFLOAT,
                                   hitChild2 ? tm2.x : MAXFLOAT,
                                   hitChild3 ? tm3.x : MAXFLOAT);

      // sort tHit and children
      //
      {
        const bool lessXY = (hitMinD.y < hitMinD.x);
        const bool lessWZ = (hitMinD.w < hitMinD.z);

        const int   w_childrenX = lessXY ? children.y : children.x;
        const int   w_childrenY = lessXY ? children.x : children.y;
        const float w_hitTimesX = lessXY ? hitMinD.y  : hitMinD.x;
        const float w_hitTimesY = lessXY ? hitMinD.x  : hitMinD.y;

        const int   w_childrenZ = lessWZ ? children.w : children.z;
        const int   w_childrenW = lessWZ ? children.z : children.w;
        const float w_hitTimesZ = lessWZ ? hitMinD.w  : hitMinD.z;
        const float w_hitTimesW = lessWZ ? hitMinD.z  : hitMinD.w;

        children.x = w_childrenX;
        children.y = w_childrenY;
        hitMinD.x  = w_hitTimesX;
        hitMinD.y  = w_hitTimesY;

        children.z = w_childrenZ;
        children.w = w_childrenW;
        hitMinD.z  = w_hitTimesZ;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZX = (hitMinD.z < hitMinD.x);
        const bool lessWY = (hitMinD.w < hitMinD.y);

        const int   w_childrenX = lessZX ? children.z : children.x;
        const int   w_childrenZ = lessZX ? children.x : children.z;
        const float w_hitTimesX = lessZX ? hitMinD.z  : hitMinD.x;
        const float w_hitTimesZ = lessZX ? hitMinD.x  : hitMinD.z;

        const int   w_childrenY = lessWY ? children.w : children.y;
        const int   w_childrenW = lessWY ? children.y : children.w;
        const float w_hitTimesY = lessWY ? hitMinD.w  : hitMinD.y;
        const float w_hitTimesW = lessWY ? hitMinD.y  : hitMinD.w;

        children.x = w_childrenX;
        children.z = w_childrenZ;
        hitMinD.x  = w_hitTimesX;
        hitMinD.z  = w_hitTimesZ;

        children.y = w_childrenY;
        children.w = w_childrenW;
        hitMinD.y  = w_hitTimesY;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZY = (hitMinD.z < hitMinD.y);

        const int   w_childrenY = lessZY ? children.z : children.y;
        const int   w_childrenZ = lessZY ? children.y : children.z;
        const float w_hitTimesY = lessZY ? hitMinD.z  : hitMinD.y;
        const float w_hitTimesZ = lessZY ? hitMinD.y  : hitMinD.z;

        children.y = w_childrenY;
        children.z = w_childrenZ;
        hitMinD.y  = w_hitTimesY;
        hitMinD.z  = w_hitTimesZ;
      }

      const bool stackHaveSpace = (top < STACK_SIZE);

      // push/pop stack games
      //
      if (hitMinD.w < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.w;
        top++;
      }

      if (hitMinD.z < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.z;
        top++;
      }

      if (hitMinD.y < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.y;
        top++;
      }

      if (hitMinD.x < MAXFLOAT)
      {
        leftNodeOffset = children.x;
      }
      else if (top >= 0)
      {
        top--;
        leftNodeOffset = stack[top];
      }

      searchingForLeaf = !IS_LEAF(leftNodeOffset) && (top >= 0);
      leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

      if (top < instTop && instDeep == 1)
      {
        ray_pos  = old_ray_pos;
        ray_dir  = old_ray_dir;
        invDir   = SafeInverse(ray_dir);
        instDeep = 0;
      }

    } //  while (searchingForLeaf)


    // leaf node, intersect triangles
    //
    if (top >= 0 && instDeep == 1)
    {
      a_hit = IntersectAllPrimitivesInLeaf(ray_pos, ray_dir, leftNodeOffset, t_rayMin, a_hit, a_tris, instId);

      top--;
      leftNodeOffset = stack[top];
    }
    else if (top >= 0 && instDeep == 0)
    {
      instDeep    = 1;
      old_ray_pos = ray_pos;
      old_ray_dir = ray_dir;

      // (1) read matrix and next offset
      //
      const int nextOffset = as_int(a_bvh[leftNodeOffset * 8 + 0].w);

      float4x4 matrix;
      matrix.row[0] = a_bvh[leftNodeOffset * 8 + 2];
      matrix.row[1] = a_bvh[leftNodeOffset * 8 + 3];
      matrix.row[2] = a_bvh[leftNodeOffset * 8 + 4];
      matrix.row[3] = a_bvh[leftNodeOffset * 8 + 5];

      instId = as_int(a_bvh[leftNodeOffset * 8 + 6].x);
      //instId = leftNodeOffset * 8 + 2; // save instAddr instead of instId

      // (2) mult ray with matrix
      //    
      ray_pos = mul4x3(matrix, ray_pos);
      ray_dir = mul3x3(matrix, ray_dir); // DON'T NORMALIZE IT !!!! When we transform to local space of node, ray_dir must be unnormalized!!!
      invDir  = SafeInverse(ray_dir);
      instTop = top;

      leftNodeOffset = nextOffset;
    }

    searchingForLeaf = !IS_LEAF(leftNodeOffset);
    leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

    if (top < instTop && instDeep == 1)
    {
      ray_pos  = old_ray_pos;
      ray_dir  = old_ray_dir;
      invDir   = SafeInverse(ray_dir);
      instDeep = 0;
    }

  } // while (top >= 0)

  return a_hit;
}


static inline float3 BVH4InstTraverseShadow(float3 ray_pos, float3 ray_dir, float t_rayMin, Lite_Hit a_hit, 
                                            __global const float4* a_bvh, __global const float4* a_tris)
{
  float3 invDir = SafeInverse(ray_dir);

  int  stackData[STACK_SIZE];
  int* stack = stackData + 2;

  int top               = 0;
  int leftNodeOffset    = 1;
  bool searchingForLeaf = true;

  ////////////////////////////////////////////////////////////////////////////// instancing variables
  int instDeep        = 0;
  int instTop         = 0;
  float3 old_ray_pos  = make_float3(0, 0, 0);
  float3 old_ray_dir  = make_float3(0, 0, 0);
  int instId          = -1;
  ////////////////////////////////////////////////////////////////////////////// instancing variables

  while (top >= 0)
  {

    while (searchingForLeaf)
    {
      const BVHNode node0 = GetBVHNode(4 * leftNodeOffset + 0, a_bvh);
      const bool    vald0 = IsValidNode(node0);
      const int     loal0 = node0.m_leftOffsetAndLeaf;
      const float2  tm0   = RayBoxIntersectionLite2(ray_pos, invDir, node0.m_boxMin, node0.m_boxMax);
 
      const BVHNode node1 = GetBVHNode(4 * leftNodeOffset + 1, a_bvh);
      const bool    vald1 = IsValidNode(node1);
      const int     loal1 = node1.m_leftOffsetAndLeaf;
      const float2  tm1   = RayBoxIntersectionLite2(ray_pos, invDir, node1.m_boxMin, node1.m_boxMax);
     
      const BVHNode node2 = GetBVHNode(4 * leftNodeOffset + 2, a_bvh);
      const bool    vald2 = IsValidNode(node2);
      const int     loal2 = node2.m_leftOffsetAndLeaf;
      const float2  tm2   = RayBoxIntersectionLite2(ray_pos, invDir, node2.m_boxMin, node2.m_boxMax);

      const BVHNode node3 = GetBVHNode(4 * leftNodeOffset + 3, a_bvh);
      const bool    vald3 = IsValidNode(node3);
      const int     loal3 = node3.m_leftOffsetAndLeaf;
      const float2  tm3   = RayBoxIntersectionLite2(ray_pos, invDir, node3.m_boxMin, node3.m_boxMax);

      int4 children = make_int4(loal0, loal1, loal2, loal3);

      const bool hitChild0 = (tm0.x <= tm0.y) && (tm0.y >= t_rayMin) && (tm0.x <= a_hit.t) && vald0;
      const bool hitChild1 = (tm1.x <= tm1.y) && (tm1.y >= t_rayMin) && (tm1.x <= a_hit.t) && vald1;
      const bool hitChild2 = (tm2.x <= tm2.y) && (tm2.y >= t_rayMin) && (tm2.x <= a_hit.t) && vald2;
      const bool hitChild3 = (tm3.x <= tm3.y) && (tm3.y >= t_rayMin) && (tm3.x <= a_hit.t) && vald3;

      float4 hitMinD = make_float4(hitChild0 ? tm0.x : MAXFLOAT,
                                   hitChild1 ? tm1.x : MAXFLOAT,
                                   hitChild2 ? tm2.x : MAXFLOAT,
                                   hitChild3 ? tm3.x : MAXFLOAT);

      // sort tHit and children
      //
      {
        const bool lessXY = (hitMinD.y < hitMinD.x);
        const bool lessWZ = (hitMinD.w < hitMinD.z);

        const int   w_childrenX = lessXY ? children.y : children.x;
        const int   w_childrenY = lessXY ? children.x : children.y;
        const float w_hitTimesX = lessXY ? hitMinD.y  : hitMinD.x;
        const float w_hitTimesY = lessXY ? hitMinD.x  : hitMinD.y;

        const int   w_childrenZ = lessWZ ? children.w : children.z;
        const int   w_childrenW = lessWZ ? children.z : children.w;
        const float w_hitTimesZ = lessWZ ? hitMinD.w  : hitMinD.z;
        const float w_hitTimesW = lessWZ ? hitMinD.z  : hitMinD.w;

        children.x = w_childrenX;
        children.y = w_childrenY;
        hitMinD.x  = w_hitTimesX;
        hitMinD.y  = w_hitTimesY;

        children.z = w_childrenZ;
        children.w = w_childrenW;
        hitMinD.z  = w_hitTimesZ;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZX = (hitMinD.z < hitMinD.x);
        const bool lessWY = (hitMinD.w < hitMinD.y);

        const int   w_childrenX = lessZX ? children.z : children.x;
        const int   w_childrenZ = lessZX ? children.x : children.z;
        const float w_hitTimesX = lessZX ? hitMinD.z  : hitMinD.x;
        const float w_hitTimesZ = lessZX ? hitMinD.x  : hitMinD.z;

        const int   w_childrenY = lessWY ? children.w : children.y;
        const int   w_childrenW = lessWY ? children.y : children.w;
        const float w_hitTimesY = lessWY ? hitMinD.w  : hitMinD.y;
        const float w_hitTimesW = lessWY ? hitMinD.y  : hitMinD.w;

        children.x = w_childrenX;
        children.z = w_childrenZ;
        hitMinD.x  = w_hitTimesX;
        hitMinD.z  = w_hitTimesZ;

        children.y = w_childrenY;
        children.w = w_childrenW;
        hitMinD.y  = w_hitTimesY;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZY = (hitMinD.z < hitMinD.y);

        const int   w_childrenY = lessZY ? children.z : children.y;
        const int   w_childrenZ = lessZY ? children.y : children.z;
        const float w_hitTimesY = lessZY ? hitMinD.z  : hitMinD.y;
        const float w_hitTimesZ = lessZY ? hitMinD.y  : hitMinD.z;

        children.y = w_childrenY;
        children.z = w_childrenZ;
        hitMinD.y  = w_hitTimesY;
        hitMinD.z  = w_hitTimesZ;
      }

      const bool stackHaveSpace = (top < STACK_SIZE);

      // push/pop stack games
      //
      if (hitMinD.w < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.w;
        top++;
      }

      if (hitMinD.z < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.z;
        top++;
      }

      if (hitMinD.y < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.y;
        top++;
      }

      if (hitMinD.x < MAXFLOAT)
      {
        leftNodeOffset = children.x;
      }
      else if (top >= 0)
      {
        top--;
        leftNodeOffset = stack[top];
      }

      searchingForLeaf = !IS_LEAF(leftNodeOffset) && (top >= 0);
      leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

      if (top < instTop && instDeep == 1)
      {
        ray_pos  = old_ray_pos;
        ray_dir  = old_ray_dir;
        invDir   = SafeInverse(ray_dir);
        instDeep = 0;
      }

    } //  while (searchingForLeaf)


    // leaf node, intersect triangles
    //
    if (top >= 0 && instDeep == 1)
    {
      const float maxHit = a_hit.t;
      a_hit = IntersectAllPrimitivesInLeaf(ray_pos, ray_dir, leftNodeOffset, t_rayMin, a_hit, a_tris, instId);

      if (HitSome(a_hit) && a_hit.t > 0.0f && a_hit.t < maxHit)
        top = 0;
      else
        a_hit.t = maxHit;

      top--;
      leftNodeOffset = stack[top];
    }
    else if (top >= 0 && instDeep == 0)
    {
      instDeep    = 1;
      old_ray_pos = ray_pos;
      old_ray_dir = ray_dir;

      // (1) read matrix and next offset
      //
      const int nextOffset = as_int(a_bvh[leftNodeOffset * 8 + 0].w);

      float4x4 matrix;
      matrix.row[0] = a_bvh[leftNodeOffset * 8 + 2];
      matrix.row[1] = a_bvh[leftNodeOffset * 8 + 3];
      matrix.row[2] = a_bvh[leftNodeOffset * 8 + 4];
      matrix.row[3] = a_bvh[leftNodeOffset * 8 + 5];

      instId = as_int(a_bvh[leftNodeOffset * 8 + 6].x);
      //instId = leftNodeOffset * 8 + 2; // save instAddr instead of instId

      // (2) mult ray with matrix
      //    
      ray_pos = mul4x3(matrix, ray_pos);
      ray_dir = mul3x3(matrix, ray_dir); // DON'T NORMALIZE IT !!!! When we transform to local space of node, ray_dir must be unnormalized!!!
      invDir  = SafeInverse(ray_dir);
      instTop = top;

      leftNodeOffset = nextOffset;
    }

    searchingForLeaf = !IS_LEAF(leftNodeOffset);
    leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

    if (top < instTop && instDeep == 1)
    {
      ray_pos  = old_ray_pos;
      ray_dir  = old_ray_dir;
      invDir   = SafeInverse(ray_dir);
      instDeep = 0;
    }

  } // while (top >= 0)

  return HitSome(a_hit) ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(1.0f, 1.0f, 1.0f);;
}


static inline Lite_Hit BVH4InstTraverseAlpha(float3 ray_pos, float3 ray_dir, float t_rayMin, Lite_Hit a_hit, 
                                             __global const float4* a_bvh, __global const float4* a_tris, __global const uint2* a_alpha,
                                             __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  float3 invDir = SafeInverse(ray_dir);

  int  stackData[STACK_SIZE];
  int* stack = stackData + 2;

  int top               = 0;
  int leftNodeOffset    = 1;
  bool searchingForLeaf = true;

  ////////////////////////////////////////////////////////////////////////////// instancing variables
  int instDeep        = 0;
  int instTop         = 0;
  float3 old_ray_pos  = make_float3(0, 0, 0);
  float3 old_ray_dir  = make_float3(0, 0, 0);
  int instId          = -1;
  ////////////////////////////////////////////////////////////////////////////// instancing variables

  while (top >= 0)
  {

    while (searchingForLeaf)
    {
      const BVHNode node0 = GetBVHNode(4 * leftNodeOffset + 0, a_bvh);
      const bool    vald0 = IsValidNode(node0);
      const int     loal0 = node0.m_leftOffsetAndLeaf;
      const float2  tm0   = RayBoxIntersectionLite2(ray_pos, invDir, node0.m_boxMin, node0.m_boxMax);
 
      const BVHNode node1 = GetBVHNode(4 * leftNodeOffset + 1, a_bvh);
      const bool    vald1 = IsValidNode(node1);
      const int     loal1 = node1.m_leftOffsetAndLeaf;
      const float2  tm1   = RayBoxIntersectionLite2(ray_pos, invDir, node1.m_boxMin, node1.m_boxMax);
     
      const BVHNode node2 = GetBVHNode(4 * leftNodeOffset + 2, a_bvh);
      const bool    vald2 = IsValidNode(node2);
      const int     loal2 = node2.m_leftOffsetAndLeaf;
      const float2  tm2   = RayBoxIntersectionLite2(ray_pos, invDir, node2.m_boxMin, node2.m_boxMax);

      const BVHNode node3 = GetBVHNode(4 * leftNodeOffset + 3, a_bvh);
      const bool    vald3 = IsValidNode(node3);
      const int     loal3 = node3.m_leftOffsetAndLeaf;
      const float2  tm3   = RayBoxIntersectionLite2(ray_pos, invDir, node3.m_boxMin, node3.m_boxMax);

      int4 children = make_int4(loal0, loal1, loal2, loal3);

      const bool hitChild0 = (tm0.x <= tm0.y) && (tm0.y >= t_rayMin) && (tm0.x <= a_hit.t) && vald0;
      const bool hitChild1 = (tm1.x <= tm1.y) && (tm1.y >= t_rayMin) && (tm1.x <= a_hit.t) && vald1;
      const bool hitChild2 = (tm2.x <= tm2.y) && (tm2.y >= t_rayMin) && (tm2.x <= a_hit.t) && vald2;
      const bool hitChild3 = (tm3.x <= tm3.y) && (tm3.y >= t_rayMin) && (tm3.x <= a_hit.t) && vald3;

      float4 hitMinD = make_float4(hitChild0 ? tm0.x : MAXFLOAT,
                                   hitChild1 ? tm1.x : MAXFLOAT,
                                   hitChild2 ? tm2.x : MAXFLOAT,
                                   hitChild3 ? tm3.x : MAXFLOAT);

      // sort tHit and children
      //
      {
        const bool lessXY = (hitMinD.y < hitMinD.x);
        const bool lessWZ = (hitMinD.w < hitMinD.z);

        const int   w_childrenX = lessXY ? children.y : children.x;
        const int   w_childrenY = lessXY ? children.x : children.y;
        const float w_hitTimesX = lessXY ? hitMinD.y  : hitMinD.x;
        const float w_hitTimesY = lessXY ? hitMinD.x  : hitMinD.y;

        const int   w_childrenZ = lessWZ ? children.w : children.z;
        const int   w_childrenW = lessWZ ? children.z : children.w;
        const float w_hitTimesZ = lessWZ ? hitMinD.w  : hitMinD.z;
        const float w_hitTimesW = lessWZ ? hitMinD.z  : hitMinD.w;

        children.x = w_childrenX;
        children.y = w_childrenY;
        hitMinD.x  = w_hitTimesX;
        hitMinD.y  = w_hitTimesY;

        children.z = w_childrenZ;
        children.w = w_childrenW;
        hitMinD.z  = w_hitTimesZ;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZX = (hitMinD.z < hitMinD.x);
        const bool lessWY = (hitMinD.w < hitMinD.y);

        const int   w_childrenX = lessZX ? children.z : children.x;
        const int   w_childrenZ = lessZX ? children.x : children.z;
        const float w_hitTimesX = lessZX ? hitMinD.z  : hitMinD.x;
        const float w_hitTimesZ = lessZX ? hitMinD.x  : hitMinD.z;

        const int   w_childrenY = lessWY ? children.w : children.y;
        const int   w_childrenW = lessWY ? children.y : children.w;
        const float w_hitTimesY = lessWY ? hitMinD.w  : hitMinD.y;
        const float w_hitTimesW = lessWY ? hitMinD.y  : hitMinD.w;

        children.x = w_childrenX;
        children.z = w_childrenZ;
        hitMinD.x  = w_hitTimesX;
        hitMinD.z  = w_hitTimesZ;

        children.y = w_childrenY;
        children.w = w_childrenW;
        hitMinD.y  = w_hitTimesY;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZY = (hitMinD.z < hitMinD.y);

        const int   w_childrenY = lessZY ? children.z : children.y;
        const int   w_childrenZ = lessZY ? children.y : children.z;
        const float w_hitTimesY = lessZY ? hitMinD.z  : hitMinD.y;
        const float w_hitTimesZ = lessZY ? hitMinD.y  : hitMinD.z;

        children.y = w_childrenY;
        children.z = w_childrenZ;
        hitMinD.y  = w_hitTimesY;
        hitMinD.z  = w_hitTimesZ;
      }

      const bool stackHaveSpace = (top < STACK_SIZE);

      // push/pop stack games
      //
      if (hitMinD.w < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.w;
        top++;
      }

      if (hitMinD.z < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.z;
        top++;
      }

      if (hitMinD.y < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.y;
        top++;
      }

      if (hitMinD.x < MAXFLOAT)
      {
        leftNodeOffset = children.x;
      }
      else if (top >= 0)
      {
        top--;
        leftNodeOffset = stack[top];
      }

      searchingForLeaf = !IS_LEAF(leftNodeOffset) && (top >= 0);
      leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

      if (top < instTop && instDeep == 1)
      {
        ray_pos  = old_ray_pos;
        ray_dir  = old_ray_dir;
        invDir   = SafeInverse(ray_dir);
        instDeep = 0;
      }

    } //  while (searchingForLeaf)


    // leaf node, intersect triangles
    //
    if (top >= 0 && instDeep == 1)
    {
      a_hit = IntersectAllPrimitivesInLeafAlpha(ray_pos, ray_dir, leftNodeOffset, t_rayMin, a_hit, a_tris, instId, a_alpha, a_texStorage, a_globals);

      top--;
      leftNodeOffset = stack[top];
    }
    else if (top >= 0 && instDeep == 0)
    {
      instDeep    = 1;
      old_ray_pos = ray_pos;
      old_ray_dir = ray_dir;

      // (1) read matrix and next offset
      //
      const int nextOffset = as_int(a_bvh[leftNodeOffset * 8 + 0].w);

      float4x4 matrix;
      matrix.row[0] = a_bvh[leftNodeOffset * 8 + 2];
      matrix.row[1] = a_bvh[leftNodeOffset * 8 + 3];
      matrix.row[2] = a_bvh[leftNodeOffset * 8 + 4];
      matrix.row[3] = a_bvh[leftNodeOffset * 8 + 5];

      instId = as_int(a_bvh[leftNodeOffset * 8 + 6].x);
      //instId = leftNodeOffset * 8 + 2; // save instAddr instead of instId

      // (2) mult ray with matrix
      //    
      ray_pos = mul4x3(matrix, ray_pos);
      ray_dir = mul3x3(matrix, ray_dir); // DON'T NORMALIZE IT !!!! When we transform to local space of node, ray_dir must be unnormalized!!!
      invDir  = SafeInverse(ray_dir);
      instTop = top;

      leftNodeOffset = nextOffset;
    }

    searchingForLeaf = !IS_LEAF(leftNodeOffset);
    leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

    if (top < instTop && instDeep == 1)
    {
      ray_pos  = old_ray_pos;
      ray_dir  = old_ray_dir;
      invDir   = SafeInverse(ray_dir);
      instDeep = 0;
    }

  } // while (top >= 0)

  return a_hit;
}



static inline Lite_Hit BVH4InstTraverseAlphaS(float3 ray_pos, float3 ray_dir, float t_rayMin, Lite_Hit a_hit, __private RandomGen* pGen,
                                              __global const float4* a_bvh, __global const float4* a_tris, __global const uint2* a_alpha,
                                              __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  float3 invDir = SafeInverse(ray_dir);

  int  stackData[STACK_SIZE];
  int* stack = stackData + 2;

  int top               = 0;
  int leftNodeOffset    = 1;
  bool searchingForLeaf = true;

  ////////////////////////////////////////////////////////////////////////////// instancing variables
  int instDeep        = 0;
  int instTop         = 0;
  float3 old_ray_pos  = make_float3(0, 0, 0);
  float3 old_ray_dir  = make_float3(0, 0, 0);
  int instId          = -1;
  ////////////////////////////////////////////////////////////////////////////// instancing variables

  while (top >= 0)
  {

    while (searchingForLeaf)
    {
      const BVHNode node0 = GetBVHNode(4 * leftNodeOffset + 0, a_bvh);
      const bool    vald0 = IsValidNode(node0);
      const int     loal0 = node0.m_leftOffsetAndLeaf;
      const float2  tm0   = RayBoxIntersectionLite2(ray_pos, invDir, node0.m_boxMin, node0.m_boxMax);
 
      const BVHNode node1 = GetBVHNode(4 * leftNodeOffset + 1, a_bvh);
      const bool    vald1 = IsValidNode(node1);
      const int     loal1 = node1.m_leftOffsetAndLeaf;
      const float2  tm1   = RayBoxIntersectionLite2(ray_pos, invDir, node1.m_boxMin, node1.m_boxMax);
     
      const BVHNode node2 = GetBVHNode(4 * leftNodeOffset + 2, a_bvh);
      const bool    vald2 = IsValidNode(node2);
      const int     loal2 = node2.m_leftOffsetAndLeaf;
      const float2  tm2   = RayBoxIntersectionLite2(ray_pos, invDir, node2.m_boxMin, node2.m_boxMax);

      const BVHNode node3 = GetBVHNode(4 * leftNodeOffset + 3, a_bvh);
      const bool    vald3 = IsValidNode(node3);
      const int     loal3 = node3.m_leftOffsetAndLeaf;
      const float2  tm3   = RayBoxIntersectionLite2(ray_pos, invDir, node3.m_boxMin, node3.m_boxMax);

      int4 children = make_int4(loal0, loal1, loal2, loal3);

      const bool hitChild0 = (tm0.x <= tm0.y) && (tm0.y >= t_rayMin) && (tm0.x <= a_hit.t) && vald0;
      const bool hitChild1 = (tm1.x <= tm1.y) && (tm1.y >= t_rayMin) && (tm1.x <= a_hit.t) && vald1;
      const bool hitChild2 = (tm2.x <= tm2.y) && (tm2.y >= t_rayMin) && (tm2.x <= a_hit.t) && vald2;
      const bool hitChild3 = (tm3.x <= tm3.y) && (tm3.y >= t_rayMin) && (tm3.x <= a_hit.t) && vald3;

      float4 hitMinD = make_float4(hitChild0 ? tm0.x : MAXFLOAT,
                                   hitChild1 ? tm1.x : MAXFLOAT,
                                   hitChild2 ? tm2.x : MAXFLOAT,
                                   hitChild3 ? tm3.x : MAXFLOAT);

      // sort tHit and children
      //
      {
        const bool lessXY = (hitMinD.y < hitMinD.x);
        const bool lessWZ = (hitMinD.w < hitMinD.z);

        const int   w_childrenX = lessXY ? children.y : children.x;
        const int   w_childrenY = lessXY ? children.x : children.y;
        const float w_hitTimesX = lessXY ? hitMinD.y  : hitMinD.x;
        const float w_hitTimesY = lessXY ? hitMinD.x  : hitMinD.y;

        const int   w_childrenZ = lessWZ ? children.w : children.z;
        const int   w_childrenW = lessWZ ? children.z : children.w;
        const float w_hitTimesZ = lessWZ ? hitMinD.w  : hitMinD.z;
        const float w_hitTimesW = lessWZ ? hitMinD.z  : hitMinD.w;

        children.x = w_childrenX;
        children.y = w_childrenY;
        hitMinD.x  = w_hitTimesX;
        hitMinD.y  = w_hitTimesY;

        children.z = w_childrenZ;
        children.w = w_childrenW;
        hitMinD.z  = w_hitTimesZ;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZX = (hitMinD.z < hitMinD.x);
        const bool lessWY = (hitMinD.w < hitMinD.y);

        const int   w_childrenX = lessZX ? children.z : children.x;
        const int   w_childrenZ = lessZX ? children.x : children.z;
        const float w_hitTimesX = lessZX ? hitMinD.z  : hitMinD.x;
        const float w_hitTimesZ = lessZX ? hitMinD.x  : hitMinD.z;

        const int   w_childrenY = lessWY ? children.w : children.y;
        const int   w_childrenW = lessWY ? children.y : children.w;
        const float w_hitTimesY = lessWY ? hitMinD.w  : hitMinD.y;
        const float w_hitTimesW = lessWY ? hitMinD.y  : hitMinD.w;

        children.x = w_childrenX;
        children.z = w_childrenZ;
        hitMinD.x  = w_hitTimesX;
        hitMinD.z  = w_hitTimesZ;

        children.y = w_childrenY;
        children.w = w_childrenW;
        hitMinD.y  = w_hitTimesY;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZY = (hitMinD.z < hitMinD.y);

        const int   w_childrenY = lessZY ? children.z : children.y;
        const int   w_childrenZ = lessZY ? children.y : children.z;
        const float w_hitTimesY = lessZY ? hitMinD.z  : hitMinD.y;
        const float w_hitTimesZ = lessZY ? hitMinD.y  : hitMinD.z;

        children.y = w_childrenY;
        children.z = w_childrenZ;
        hitMinD.y  = w_hitTimesY;
        hitMinD.z  = w_hitTimesZ;
      }

      const bool stackHaveSpace = (top < STACK_SIZE);

      // push/pop stack games
      //
      if (hitMinD.w < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.w;
        top++;
      }

      if (hitMinD.z < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.z;
        top++;
      }

      if (hitMinD.y < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.y;
        top++;
      }

      if (hitMinD.x < MAXFLOAT)
      {
        leftNodeOffset = children.x;
      }
      else if (top >= 0)
      {
        top--;
        leftNodeOffset = stack[top];
      }

      searchingForLeaf = !IS_LEAF(leftNodeOffset) && (top >= 0);
      leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

      if (top < instTop && instDeep == 1)
      {
        ray_pos  = old_ray_pos;
        ray_dir  = old_ray_dir;
        invDir   = SafeInverse(ray_dir);
        instDeep = 0;
      }

    } //  while (searchingForLeaf)


    // leaf node, intersect triangles
    //
    if (top >= 0 && instDeep == 1)
    {
      a_hit = IntersectAllPrimitivesInLeafAlphaS(ray_pos, ray_dir, leftNodeOffset, t_rayMin, a_hit, pGen, a_tris, instId, a_alpha, a_texStorage, a_globals);

      top--;
      leftNodeOffset = stack[top];
    }
    else if (top >= 0 && instDeep == 0)
    {
      instDeep    = 1;
      old_ray_pos = ray_pos;
      old_ray_dir = ray_dir;

      // (1) read matrix and next offset
      //
      const int nextOffset = as_int(a_bvh[leftNodeOffset * 8 + 0].w);

      float4x4 matrix;
      matrix.row[0] = a_bvh[leftNodeOffset * 8 + 2];
      matrix.row[1] = a_bvh[leftNodeOffset * 8 + 3];
      matrix.row[2] = a_bvh[leftNodeOffset * 8 + 4];
      matrix.row[3] = a_bvh[leftNodeOffset * 8 + 5];

      instId = as_int(a_bvh[leftNodeOffset * 8 + 6].x);
      //instId = leftNodeOffset * 8 + 2; // save instAddr instead of instId

      // (2) mult ray with matrix
      //    
      ray_pos = mul4x3(matrix, ray_pos);
      ray_dir = mul3x3(matrix, ray_dir); // DON'T NORMALIZE IT !!!! When we transform to local space of node, ray_dir must be unnormalized!!!
      invDir  = SafeInverse(ray_dir);
      instTop = top;

      leftNodeOffset = nextOffset;
    }

    searchingForLeaf = !IS_LEAF(leftNodeOffset);
    leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

    if (top < instTop && instDeep == 1)
    {
      ray_pos  = old_ray_pos;
      ray_dir  = old_ray_dir;
      invDir   = SafeInverse(ray_dir);
      instDeep = 0;
    }

  } // while (top >= 0)

  return a_hit;
}


static inline float3 BVH4InstTraverseShadowAlphaS(float3 ray_pos, float3 ray_dir, float t_rayMin, float t_rayMax,
                                                  __global const float4* a_bvh, __global const float4* a_tris, __global const uint2* a_alpha,
                                                  __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  float3 invDir = SafeInverse(ray_dir);

  int  stackData[STACK_SIZE];
  int* stack = stackData + 2;

  int top               = 0;
  int leftNodeOffset    = 1;
  bool searchingForLeaf = true;

  ////////////////////////////////////////////////////////////////////////////// instancing variables
  int instDeep        = 0;
  int instTop         = 0;
  float3 old_ray_pos  = make_float3(0, 0, 0);
  float3 old_ray_dir  = make_float3(0, 0, 0);
  int instId          = -1;
  ////////////////////////////////////////////////////////////////////////////// instancing variables

  float shadow = 1.0f;

  while (top >= 0)
  {

    while (searchingForLeaf)
    {
      const BVHNode node0 = GetBVHNode(4 * leftNodeOffset + 0, a_bvh);
      const bool    vald0 = IsValidNode(node0);
      const int     loal0 = node0.m_leftOffsetAndLeaf;
      const float2  tm0   = RayBoxIntersectionLite2(ray_pos, invDir, node0.m_boxMin, node0.m_boxMax);
 
      const BVHNode node1 = GetBVHNode(4 * leftNodeOffset + 1, a_bvh);
      const bool    vald1 = IsValidNode(node1);
      const int     loal1 = node1.m_leftOffsetAndLeaf;
      const float2  tm1   = RayBoxIntersectionLite2(ray_pos, invDir, node1.m_boxMin, node1.m_boxMax);
     
      const BVHNode node2 = GetBVHNode(4 * leftNodeOffset + 2, a_bvh);
      const bool    vald2 = IsValidNode(node2);
      const int     loal2 = node2.m_leftOffsetAndLeaf;
      const float2  tm2   = RayBoxIntersectionLite2(ray_pos, invDir, node2.m_boxMin, node2.m_boxMax);

      const BVHNode node3 = GetBVHNode(4 * leftNodeOffset + 3, a_bvh);
      const bool    vald3 = IsValidNode(node3);
      const int     loal3 = node3.m_leftOffsetAndLeaf;
      const float2  tm3   = RayBoxIntersectionLite2(ray_pos, invDir, node3.m_boxMin, node3.m_boxMax);

      int4 children = make_int4(loal0, loal1, loal2, loal3);

      const bool hitChild0 = (tm0.x <= tm0.y) && (tm0.y >= t_rayMin) && (tm0.x <= t_rayMax) && vald0;
      const bool hitChild1 = (tm1.x <= tm1.y) && (tm1.y >= t_rayMin) && (tm1.x <= t_rayMax) && vald1;
      const bool hitChild2 = (tm2.x <= tm2.y) && (tm2.y >= t_rayMin) && (tm2.x <= t_rayMax) && vald2;
      const bool hitChild3 = (tm3.x <= tm3.y) && (tm3.y >= t_rayMin) && (tm3.x <= t_rayMax) && vald3;

      float4 hitMinD = make_float4(hitChild0 ? tm0.x : MAXFLOAT,
                                   hitChild1 ? tm1.x : MAXFLOAT,
                                   hitChild2 ? tm2.x : MAXFLOAT,
                                   hitChild3 ? tm3.x : MAXFLOAT);

      // sort tHit and children
      //
      {
        const bool lessXY = (hitMinD.y < hitMinD.x);
        const bool lessWZ = (hitMinD.w < hitMinD.z);

        const int   w_childrenX = lessXY ? children.y : children.x;
        const int   w_childrenY = lessXY ? children.x : children.y;
        const float w_hitTimesX = lessXY ? hitMinD.y  : hitMinD.x;
        const float w_hitTimesY = lessXY ? hitMinD.x  : hitMinD.y;

        const int   w_childrenZ = lessWZ ? children.w : children.z;
        const int   w_childrenW = lessWZ ? children.z : children.w;
        const float w_hitTimesZ = lessWZ ? hitMinD.w  : hitMinD.z;
        const float w_hitTimesW = lessWZ ? hitMinD.z  : hitMinD.w;

        children.x = w_childrenX;
        children.y = w_childrenY;
        hitMinD.x  = w_hitTimesX;
        hitMinD.y  = w_hitTimesY;

        children.z = w_childrenZ;
        children.w = w_childrenW;
        hitMinD.z  = w_hitTimesZ;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZX = (hitMinD.z < hitMinD.x);
        const bool lessWY = (hitMinD.w < hitMinD.y);

        const int   w_childrenX = lessZX ? children.z : children.x;
        const int   w_childrenZ = lessZX ? children.x : children.z;
        const float w_hitTimesX = lessZX ? hitMinD.z  : hitMinD.x;
        const float w_hitTimesZ = lessZX ? hitMinD.x  : hitMinD.z;

        const int   w_childrenY = lessWY ? children.w : children.y;
        const int   w_childrenW = lessWY ? children.y : children.w;
        const float w_hitTimesY = lessWY ? hitMinD.w  : hitMinD.y;
        const float w_hitTimesW = lessWY ? hitMinD.y  : hitMinD.w;

        children.x = w_childrenX;
        children.z = w_childrenZ;
        hitMinD.x  = w_hitTimesX;
        hitMinD.z  = w_hitTimesZ;

        children.y = w_childrenY;
        children.w = w_childrenW;
        hitMinD.y  = w_hitTimesY;
        hitMinD.w  = w_hitTimesW;
      }

      {
        const bool lessZY = (hitMinD.z < hitMinD.y);

        const int   w_childrenY = lessZY ? children.z : children.y;
        const int   w_childrenZ = lessZY ? children.y : children.z;
        const float w_hitTimesY = lessZY ? hitMinD.z  : hitMinD.y;
        const float w_hitTimesZ = lessZY ? hitMinD.y  : hitMinD.z;

        children.y = w_childrenY;
        children.z = w_childrenZ;
        hitMinD.y  = w_hitTimesY;
        hitMinD.z  = w_hitTimesZ;
      }

      const bool stackHaveSpace = (top < STACK_SIZE);

      // push/pop stack games
      //
      if (hitMinD.w < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.w;
        top++;
      }

      if (hitMinD.z < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.z;
        top++;
      }

      if (hitMinD.y < MAXFLOAT && stackHaveSpace)
      {
        stack[top] = children.y;
        top++;
      }

      if (hitMinD.x < MAXFLOAT)
      {
        leftNodeOffset = children.x;
      }
      else if (top >= 0)
      {
        top--;
        leftNodeOffset = stack[top];
      }

      searchingForLeaf = !IS_LEAF(leftNodeOffset) && (top >= 0);
      leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

      if (top < instTop && instDeep == 1)
      {
        ray_pos  = old_ray_pos;
        ray_dir  = old_ray_dir;
        invDir   = SafeInverse(ray_dir);
        instDeep = 0;
      }

    } //  while (searchingForLeaf)


    // leaf node, intersect triangles
    //
    if (top >= 0 && instDeep == 1)
    {
      IntersectAllPrimitivesInLeafShadowAlphaS(ray_pos, ray_dir, leftNodeOffset, t_rayMin, t_rayMax, &shadow, a_tris, instId, a_alpha, a_texStorage, a_globals);
      if (shadow < 0.0001f)
        top = 0;
     
      top--;
      leftNodeOffset = stack[top];
    }
    else if (top >= 0 && instDeep == 0)
    {
      instDeep    = 1;
      old_ray_pos = ray_pos;
      old_ray_dir = ray_dir;

      // (1) read matrix and next offset
      //
      const int nextOffset = as_int(a_bvh[leftNodeOffset * 8 + 0].w);

      float4x4 matrix;
      matrix.row[0] = a_bvh[leftNodeOffset * 8 + 2];
      matrix.row[1] = a_bvh[leftNodeOffset * 8 + 3];
      matrix.row[2] = a_bvh[leftNodeOffset * 8 + 4];
      matrix.row[3] = a_bvh[leftNodeOffset * 8 + 5];

      instId = as_int(a_bvh[leftNodeOffset * 8 + 6].x);
      //instId = leftNodeOffset * 8 + 2; // save instAddr instead of instId

      // (2) mult ray with matrix
      //    
      ray_pos = mul4x3(matrix, ray_pos);
      ray_dir = mul3x3(matrix, ray_dir); // DON'T NORMALIZE IT !!!! When we transform to local space of node, ray_dir must be unnormalized!!!
      invDir  = SafeInverse(ray_dir);
      instTop = top;

      leftNodeOffset = nextOffset;
    }

    searchingForLeaf = !IS_LEAF(leftNodeOffset);
    leftNodeOffset   = EXTRACT_OFFSET(leftNodeOffset);

    if (top < instTop && instDeep == 1)
    {
      ray_pos  = old_ray_pos;
      ray_dir  = old_ray_dir;
      invDir   = SafeInverse(ray_dir);
      instDeep = 0;
    }

  } // while (top >= 0)

  return make_float3(shadow, shadow, shadow);
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//// other utils functions
//

static inline float2 triBaricentrics(float3 ray_pos, float3 ray_dir, float3 A_pos, float3 B_pos, float3 C_pos)
{
  const float3 edge1 = B_pos - A_pos;
  const float3 edge2 = C_pos - A_pos;
  const float3 pvec  = cross(ray_dir, edge2);
  const float  det   = dot(edge1, pvec);

  const float  inv_det = 1.0f / det;
  const float3 tvec    = ray_pos - A_pos;
  const float  v       = dot(tvec, pvec)*inv_det;

  const float3 qvec = cross(tvec, edge1);
  const float  u    = dot(ray_dir, qvec)*inv_det;

  return make_float2(u, v);
}

static inline SurfaceHit surfaceEvalLS(const float3 a_rpos, const float3 a_rdir, const Lite_Hit hit, __global const PlainMesh* mesh)
{
  __global const float4* vertPos      = meshVerts(mesh);
  __global const float4* vertNorm     = meshNorms(mesh);
  __global const float2* vertTexCoord = meshTexCoords(mesh);
  __global const uint*   vertTangent  = meshTangentsCompressed(mesh);
  __global const int*    vertIndices  = meshTriIndices(mesh);
  __global const int*    matIndices   = meshMatIndices(mesh);
  
  SurfaceHit surfHit;
  surfHit.matId       = matIndices[hit.primId];
  surfHit.alphaMatId  = 0;
  
  const int offset    = hit.primId * 3; 
                      
  const int offs_A    = vertIndices[offset + 0];
  const int offs_B    = vertIndices[offset + 1];
  const int offs_C    = vertIndices[offset + 2];

  const float3 A_pos  = to_float3(vertPos[offs_A]);
  const float3 B_pos  = to_float3(vertPos[offs_B]);
  const float3 C_pos  = to_float3(vertPos[offs_C]);
  
  
  const float3 A_norm = to_float3(vertNorm[offs_A]);
  const float3 B_norm = to_float3(vertNorm[offs_B]);
  const float3 C_norm = to_float3(vertNorm[offs_C]);
  
  const float2 A_tex  = vertTexCoord[offs_A];
  const float2 B_tex  = vertTexCoord[offs_B];
  const float2 C_tex  = vertTexCoord[offs_C];
  
  const float2 uv     = triBaricentrics(a_rpos, a_rdir, A_pos, B_pos, C_pos);

  surfHit.pos         = (1.0f - uv.x - uv.y)*A_pos  + uv.y*B_pos  + uv.x*C_pos;
  surfHit.texCoord    = (1.0f - uv.x - uv.y)*A_tex  + uv.y*B_tex  + uv.x*C_tex;
  surfHit.normal      = (1.0f - uv.x - uv.y)*A_norm + uv.y*B_norm + uv.x*C_norm;
  surfHit.t           = hit.t;
  
  const float3 A_tang = decodeNormal(vertTangent[offs_A]); // GetVertexNorm(offs_A);
  const float3 B_tang = decodeNormal(vertTangent[offs_B]); // GetVertexNorm(offs_B);
  const float3 C_tang = decodeNormal(vertTangent[offs_C]); // GetVertexNorm(offs_C);

  surfHit.flatNormal  = normalize(cross(A_pos - B_pos, A_pos - C_pos));
  if (dot(a_rdir, surfHit.flatNormal) > 0.0f)
    surfHit.flatNormal = surfHit.flatNormal*(-1.0f);

  if (dot(a_rdir, surfHit.normal) > 0.0f)
  {
    surfHit.normal = surfHit.normal*(-1.0f);
    surfHit.hfi    = true;
  }
  else
  {
    surfHit.hfi = false;
  }

  surfHit.tangent     = normalize((1.0f - uv.x - uv.y)*A_tang + uv.y*B_tang + uv.x*C_tang);
  surfHit.biTangent   = normalize(cross(surfHit.normal, surfHit.tangent));
  
  return surfHit;
}


#endif // file guardian

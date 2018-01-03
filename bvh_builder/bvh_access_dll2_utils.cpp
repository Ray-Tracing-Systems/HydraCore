#include "bvh_access.h"

RTCScene EmbreeBVH4_2::TriMesh::CreateInternalGeom(const float* a_vert4f, int a_numVert, const int* a_indices, const int a_numIndices, int a_meshId, RTCScene a_pScene)
{
  const int numTriangles = a_numIndices / 3;

  RTCScene scene = nullptr;

  if (a_pScene == nullptr)
    scene = rtcDeviceNewScene(pSelf->m_device, BUILD_FLAGS, pSelf->m_algorithmFlags);
  else
    scene = a_pScene;

  //////////////////////////////////////////////////////////////////////////////////

  const unsigned int meshId = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, numTriangles, a_numVert);

  Vertex* vertices = (Vertex*)rtcMapBuffer(scene, meshId, RTC_VERTEX_BUFFER);
  memcpy(vertices, a_vert4f, a_numVert * sizeof(float) * 4);
  rtcUnmapBuffer(scene, meshId, RTC_VERTEX_BUFFER);

  Triangle* triangles = (Triangle*)rtcMapBuffer(scene, meshId, RTC_INDEX_BUFFER);
  memcpy(triangles, a_indices, a_numIndices * sizeof(int));
  rtcUnmapBuffer(scene, meshId, RTC_INDEX_BUFFER);

  return scene;
}

RTCScene EmbreeBVH4_2::RefMesh::CreateInternalGeom(const float* a_vert4f, int a_numVert, const int* a_indices, const int a_numIndices, int a_meshId, RTCScene a_pScene)
{
  const int numTriangles = a_numIndices / 3;

  RTCScene scene = nullptr;

  if (a_pScene == nullptr)
    scene = rtcDeviceNewScene(pSelf->m_device, BUILD_FLAGS, pSelf->m_algorithmFlags);
  else
    scene = a_pScene;

  //////////////////////////////////////////////////////////////////////////////////

  // create reference array
  //
  auto& triRefs = pSelf->m_refsHash[a_meshId];

  triRefs.resize(numTriangles);
  for (size_t i = 0; i < triRefs.size(); i++)
  {
    const int iA = a_indices[3 * i + 0];
    const int iB = a_indices[3 * i + 1];
    const int iC = a_indices[3 * i + 2];

    EarlySplit::Triangle tri;

    tri.A = float4(a_vert4f[iA * 4 + 0], a_vert4f[iA * 4 + 1], a_vert4f[iA * 4 + 2], 0.0f);
    tri.B = float4(a_vert4f[iB * 4 + 0], a_vert4f[iB * 4 + 1], a_vert4f[iB * 4 + 2], 0.0f);
    tri.C = float4(a_vert4f[iC * 4 + 0], a_vert4f[iC * 4 + 1], a_vert4f[iC * 4 + 2], 0.0f);

    triRefs[i].triId  = i;
    triRefs[i].box    = EarlySplit::TriBounds(tri);
    triRefs[i].metric = EarlySplit::SubdivMetric(tri, triRefs[i].box);
  }

  const int debugSize = sizeof(EarlySplit::TriRef);

  //////////////////////////////////////////////////////////////////////////////////

  const int N = int(triRefs.size());

  unsigned int geomID = rtcNewUserGeometry(scene, N);
   
  auto refsPointer = (EarlySplit::TriRef*)alignedMalloc(N * sizeof(EarlySplit::TriRef));

  for (size_t i = 0; i < N; i++)
  {
    refsPointer[i]        = triRefs[i];
    refsPointer[i].geomID = geomID;
  }

  rtcSetUserData(scene, geomID, refsPointer);
  rtcSetBoundsFunction(scene, geomID, (RTCBoundsFunc)&EarlySplit::TriRefBoundsFunc);

  return scene;
}

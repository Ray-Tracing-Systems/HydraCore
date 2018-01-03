#include "../../HydraAPP/hydra_drv/IBVHBuilderAPI.h"

#include "../common/tutorial/tutorial.h"
#include "../common/tutorial/tutorial_device.h"
#include "../../include/embree2/rtcore.h"
#include "../../kernels/bvh/bvh.h"
#include "../../kernels/geometry/trianglev.h"

#include <unordered_map>

using namespace embree;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct EmbreeBVH4 : public IBVHBuilder
{
  EmbreeBVH4();
  ~EmbreeBVH4();

  void Init(char* cgf);
  void Destroy();
  
  void ClearData();
  int  CreateTriangleMesh(const float* a_vert4f, int a_numVert, const int* a_indices, int a_numIndices);
  void ReleaseTriangleMesh(int a_meshId);
  
  void ClearScene();
  void CommitScene();
  int  CreateInstance(int a_meshId, const float* a_matrix);
  void UpdateInstance(int a_instanceId, const float* a_matrix);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////


  ConvertionResult ConvertMap();   // do actual converstion to our format
  void             ConvertUnmap(); // free memory
  
  Lite_Hit RayTrace(float3 ray_pos, float3 ray_dir);                   // for CPU engine and test only
  float3   ShadowTrace(float3 ray_pos, float3 ray_dir, float t_far);   // for CPU engine and test only

protected:

  void print_bvh4_triangle4v_v2(BVH4::NodeRef node, size_t depth, size_t instDepth);
  std::ofstream m_bvhOutputDebug;

  RTCDevice m_device;
  RTCScene  m_sceneTopLevel;

  std::vector<RTCScene>     m_objects;
  std::vector<float4x4>     m_instMatrices;
  std::vector<int>          m_instMeshesId;
  RTCAlgorithmFlags         m_aflags;

  size_t ConvertBvh4TwoLevel(BVH4::NodeRef node, size_t currNodeOffset, int depth, int instDepth);

  size_t Alloc4BVHNodes(std::vector<BVHNode>& a_vector);
  size_t Alloc3Float4(std::vector<float4>& a_vector);

  std::vector<BVHNode>      m_convertedLayout;
  std::vector<float4>       m_convertedTrinagles;
  size_t                    m_totalMeshTriangleCount;

  std::vector<BVHNode> m_dummy4bvh;
  std::vector<float4>  m_dummy3f4;

  struct InstanceNode
  {
    size_t currNodeOffset;
    size_t leftNodeOffset;
    BVH4::NodeRef tree;
    int32_t instanceId;
    int32_t meshId;
  };

  std::vector<InstanceNode>            m_instNodesConnections;
  std::unordered_map<int32_t, size_t>  m_instNodeMeshesRef;
}; 


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EmbreeBVH4::EmbreeBVH4() : m_device(nullptr), m_aflags(RTC_INTERSECT1), m_totalMeshTriangleCount(0)
{
  m_objects.reserve(1000);
  m_dummy4bvh.resize(4);
  m_dummy3f4.resize(3);
}

EmbreeBVH4::~EmbreeBVH4()
{

}

void EmbreeBVH4::Init(char* cfg)
{
  m_device = rtcNewDevice("tri_accel=bvh4.triangle4v");

  rtcDeviceSetErrorFunction2(m_device, error_handler, nullptr);

  m_sceneTopLevel = rtcDeviceNewScene(m_device, RTC_SCENE_STATIC, m_aflags);
}

void EmbreeBVH4::Destroy()
{
  ClearData();

  rtcDeleteScene(m_sceneTopLevel);
  m_sceneTopLevel = nullptr;

  rtcDeleteDevice(m_device); 
  m_device = nullptr;
}

void EmbreeBVH4::ClearData()
{
  for (auto scn : m_objects)
    rtcDeleteScene(scn);
  m_objects.clear();

  rtcDeleteScene(m_sceneTopLevel);
  m_sceneTopLevel = rtcDeviceNewScene(m_device, RTC_SCENE_STATIC, m_aflags);

  // TODO: free mem for all of our vectors
  //

}

int  EmbreeBVH4::CreateTriangleMesh(const float* a_vert4f, int a_numVert, const int* a_indices, int a_numIndices)
{
  RTCScene scene = rtcDeviceNewScene(m_device, RTC_SCENE_STATIC, m_aflags);

  int numTriangles = a_numIndices / 3;

  unsigned int meshId = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, numTriangles, a_numVert);

  Vertex* vertices = (Vertex*)rtcMapBuffer(scene, meshId, RTC_VERTEX_BUFFER);
  memcpy(vertices, a_vert4f, a_numVert * sizeof(float) * 4);
  rtcUnmapBuffer(scene, meshId, RTC_VERTEX_BUFFER);
  
  Triangle* triangles = (Triangle*)rtcMapBuffer(scene, meshId, RTC_INDEX_BUFFER);
  memcpy(triangles, a_indices, a_numIndices * sizeof(int));
  rtcUnmapBuffer(scene, meshId, RTC_INDEX_BUFFER);

  rtcCommit(scene);

  m_totalMeshTriangleCount += (a_numIndices / 3);

  m_objects.push_back(scene);
  return int(m_objects.size()-1);
}

void EmbreeBVH4::ReleaseTriangleMesh(int a_meshId)
{
  if (a_meshId < m_objects.size())
  {
    rtcDeleteScene(m_objects[a_meshId]);
    m_objects[a_meshId] = 0;
  }
}

void EmbreeBVH4::ClearScene()
{
  rtcDeleteScene(m_sceneTopLevel);
  m_sceneTopLevel = rtcDeviceNewScene(m_device, RTC_SCENE_STATIC, m_aflags);

  m_instMeshesId.resize(0);
  m_instMatrices.resize(0);
}

void EmbreeBVH4::CommitScene()
{
  rtcCommit(m_sceneTopLevel);
}

int  EmbreeBVH4::CreateInstance(int a_meshId, const float* a_matrix)
{
  if (a_meshId > m_objects.size())
    return -1;

  auto myInstanceId = rtcNewInstance(m_sceneTopLevel, m_objects[a_meshId]);

  rtcSetTransform2(m_sceneTopLevel, myInstanceId, RTC_MATRIX_ROW_MAJOR, a_matrix, 0);

  while (m_instMatrices.size() <= myInstanceId)
  {
    m_instMatrices.push_back(float4x4());
    m_instMeshesId.push_back(-1);
  }

  m_instMatrices[myInstanceId] = float4x4(a_matrix);
  m_instMeshesId[myInstanceId] = a_meshId;

  return int(myInstanceId);
}

void EmbreeBVH4::UpdateInstance(int a_instanceId, const float* a_matrix)
{
  rtcSetTransform2(m_sceneTopLevel, (unsigned int)(a_instanceId), RTC_MATRIX_ROW_MAJOR, a_matrix, 0);
  m_instMatrices[a_instanceId] = float4x4(a_matrix);
}

const bool IsThisFuckingEmbreeLeafWithTriangles(const int* pInstId)
{
  bool isZero = true;

  for (int i = 1; i <= 3; i++)
    if (pInstId[i] != 0)
      isZero = false;

  return !isZero;
}


void EmbreeBVH4::print_bvh4_triangle4v_v2(BVH4::NodeRef node, size_t depth, size_t a_level) // this function is incorrect
{
  auto nodeType = node.type();

  if (node.isAlignedNode())
  {
    BVH4::AlignedNode* n = node.alignedNode();
    
    m_bvhOutputDebug << "AlignedNode {" << std::endl;
    for (size_t i=0; i<4; i++)
    {
      for (size_t k=0; k<depth; k++) m_bvhOutputDebug << "  ";
      m_bvhOutputDebug << "  bounds" << i << " = " << n->bounds(i) << std::endl;
    }
  
    for (size_t i=0; i<4; i++)
    {
      if (n->child(i) == BVH4::emptyNode)
        continue;

      for (size_t k=0; k<depth; k++) m_bvhOutputDebug << "  ";
      m_bvhOutputDebug << "  child" << i << " = ";
      print_bvh4_triangle4v_v2(n->child(i),depth+1, a_level);
    }
    for (size_t k=0; k<depth; k++) m_bvhOutputDebug << "  ";
    m_bvhOutputDebug << "}" << std::endl;
  }
  else if(node.isLeaf())
  {

    size_t num = 0;
    const int* pInstId = (const int*)node.leaf(num);
    const Triangle4v* tri = (const Triangle4v*)pInstId;

    if (nodeType == 9 && a_level == 0 && (*pInstId) < m_instMeshesId.size() && !IsThisFuckingEmbreeLeafWithTriangles(pInstId)) // instance
    {
      if ((*pInstId) < m_instMeshesId.size())
      {
        int meshId = m_instMeshesId[(*pInstId)];
        if (meshId < m_objects.size())
        {
          // now get our matrix and new bvh scene pointer
          //
          const RTCScene& scn = m_objects[meshId];

          BVH4* bvh4 = nullptr;
          AccelData* accel = ((Accel*)scn)->intersectors.ptr;
          if (accel->type == AccelData::TY_BVH4)
            bvh4 = (BVH4*)accel;

          if (bvh4 != nullptr && a_level == 0)
          {
            m_bvhOutputDebug << "Instance {" << std::endl;
            BVH4::NodeRef root = bvh4->root;
            print_bvh4_triangle4v_v2(root, depth + 1, a_level + 1);
            for (size_t k = 0; k < depth; k++) m_bvhOutputDebug << "  ";
            m_bvhOutputDebug << "}" << std::endl;
          }
        }
      }
    }
    else
    {
      size_t num;
      const Triangle4v* tri = (const Triangle4v*)node.leaf(num);

      for (size_t k = 0; k < depth; k++) m_bvhOutputDebug << "  ";
      m_bvhOutputDebug << "Leaf { triListNum = " << num << "} " << std::endl;

      size_t triNum = 0;
      for (size_t i = 0; i < num; i++)
      {
        Triangle4v tlist = tri[i];

        for (size_t j = 0; j < tlist.size(); j++)
        {
          for (size_t k = 0; k < depth; k++) m_bvhOutputDebug << "  ";
          m_bvhOutputDebug << "  Triangle { v0 = (" << tlist.v0.x[j] << ", " << tlist.v0.y[j] << ", " << tlist.v0.z[j] << "),  "
            "v1 = (" << tlist.v1.x[j] << ", " << tlist.v1.y[j] << ", " << tlist.v1.z[j] << "), "
            "v2 = (" << tlist.v2.x[j] << ", " << tlist.v2.y[j] << ", " << tlist.v2.z[j] << "), "
            "geomID = " << tlist.geomID(j) << ", primID = " << tlist.primID(j) << " }" << std::endl;
          triNum++;
        }

        for (size_t k = 0; k < depth; k++) m_bvhOutputDebug << "  ";
        m_bvhOutputDebug << "  -------------------------------------------------------------------------------------" << std::endl;
      }

      for (size_t k = 0; k < depth; k++) m_bvhOutputDebug << "  ";
      m_bvhOutputDebug << "}" << std::endl;
      for (size_t k = 0; k < depth; k++) m_bvhOutputDebug << "  ";
      m_bvhOutputDebug << "TriNum = " << triNum << "  " << std::endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t EmbreeBVH4::Alloc4BVHNodes(std::vector<BVHNode>& a_vector)
{
  size_t currSize = a_vector.size();
  a_vector.insert(a_vector.end(), m_dummy4bvh.begin(), m_dummy4bvh.end());
  return currSize;
}

size_t EmbreeBVH4::Alloc3Float4(std::vector<float4>& a_vector)
{
  size_t currSize = a_vector.size();
  a_vector.insert(a_vector.end(), m_dummy3f4.begin(), m_dummy3f4.end());
  return currSize;
}


inline void CopyBounds(BVHNode* pNode, embree::BBox3fa box)
{
  pNode->m_boxMin.x = box.lower.x;
  pNode->m_boxMin.y = box.lower.y;
  pNode->m_boxMin.z = box.lower.z;

  pNode->m_boxMax.x = box.upper.x;
  pNode->m_boxMax.y = box.upper.y;
  pNode->m_boxMax.z = box.upper.z;
}

embree::BBox3fa CalcBoundingBoxOfChilds(BVH4::NodeRef node)
{
  embree::BBox3fa box(embree::Vec3fa(inf,inf,inf), embree::Vec3fa(neg_inf, neg_inf, neg_inf));
  
  if (node.isAlignedNode())
  {
    BVH4::AlignedNode* n = node.alignedNode();

    for (size_t i = 0; i<4; i++)
    {
      if (n->child(i) == BVH4::emptyNode)
        continue;

      auto childBox = n->bounds(i);

      box.lower.x = fminf(box.lower.x, childBox.lower.x);
      box.lower.y = fminf(box.lower.y, childBox.lower.y);
      box.lower.z = fminf(box.lower.z, childBox.lower.z);

      box.upper.x = fmaxf(box.upper.x, childBox.upper.x);
      box.upper.y = fmaxf(box.upper.y, childBox.upper.y);
      box.upper.z = fmaxf(box.upper.z, childBox.upper.z);
    }
  }

  return box;
}


size_t EmbreeBVH4::ConvertBvh4TwoLevel(BVH4::NodeRef node, size_t currNodeOffset, int a_depth, int a_level)
{
  size_t treeOffset = size_t(-1);

  auto nodeType = node.type();

  if (node.isAlignedNode())
  {
    BVH4::AlignedNode* n = node.alignedNode();

    size_t offsets[4] = { 0,0,0,0 };

    offsets[0] = Alloc4BVHNodes(m_convertedLayout);
    offsets[1] = offsets[0] + 1;
    offsets[2] = offsets[0] + 2;
    offsets[3] = offsets[0] + 3;

    for (size_t i = 0; i<4; i++)
    {
      if (n->child(i) == BVH4::emptyNode)
        continue;

      auto box = n->bounds(i);
      auto pNode = &m_convertedLayout[0] + offsets[i];

      CopyBounds(pNode, box);
    }

    m_convertedLayout[currNodeOffset].SetLeaf(0);
    m_convertedLayout[currNodeOffset].SetLeftOffset(offsets[0] / 4);

    for (size_t i = 0; i < 4; i++)
    {
      if (n->child(i) == BVH4::emptyNode)
        continue;

      ConvertBvh4TwoLevel(n->child(i), offsets[i], a_depth + 1, a_level);
    }

    treeOffset = offsets[0];
  }
  else if (node.isLeaf())
  {
    size_t num = 0;
    const int* pInstId = (const int*)node.leaf(num);
    const Triangle4v* tri = (const Triangle4v*)pInstId;

    if (nodeType == 9 && a_level == 0 && (*pInstId) < m_instMeshesId.size() && !IsThisFuckingEmbreeLeafWithTriangles(pInstId)) // instance
    {
      int meshId = m_instMeshesId[(*pInstId)];
      if (meshId < m_objects.size())
      {
        const RTCScene& scn = m_objects[meshId];

        BVH4* bvh4 = nullptr;
        AccelData* accel = ((Accel*)scn)->intersectors.ptr;
        if (accel->type == AccelData::TY_BVH4)
          bvh4 = (BVH4*)accel;

        if (bvh4 != nullptr)
        {
          BVH4::NodeRef root = bvh4->root;

          size_t offsets[4] = { 0,0,0,0 };

          offsets[0] = Alloc4BVHNodes(m_convertedLayout);
          offsets[1] = offsets[0] + 1;
          offsets[2] = offsets[0] + 2;
          offsets[3] = offsets[0] + 3;

          m_convertedLayout[currNodeOffset].SetLeaf(0);
          m_convertedLayout[currNodeOffset].SetLeftOffset(offsets[0] / 4);
          m_convertedLayout[currNodeOffset].m_escapeIndex = 1; // this is an instance (!!!)

          // m_convertedLayout[offsets[0]] -> new root;
          // m_convertedLayout[offsets[1]] -> first  half of matrix
          // m_convertedLayout[offsets[2]] -> second half of matrix
          // m_convertedLayout[offsets[3]] -> not used
          //
          float4x4 mInverse = inverse4x4(m_instMatrices[(*pInstId)]);
          float4x4* pMatrix = (float4x4*)(&m_convertedLayout[offsets[1]]);
          if (root.isLeaf())
            mInverse = float4x4();
          (*pMatrix) = mInverse;

          // get bounding box
          //
          if (!root.isAlignedNode())  // get box from current node
          {
            m_convertedLayout[offsets[0]].m_boxMin = m_convertedLayout[currNodeOffset].m_boxMin;
            m_convertedLayout[offsets[0]].m_boxMax = m_convertedLayout[currNodeOffset].m_boxMax;
          }
          else // get box by calculating it from childs
          {
            embree::BBox3fa rootBox = CalcBoundingBoxOfChilds(root);
            CopyBounds(&m_convertedLayout[offsets[0]], rootBox);
          }

          InstanceNode nextTree;

          nextTree.currNodeOffset = currNodeOffset;
          nextTree.leftNodeOffset = offsets[0];
          nextTree.instanceId     = (*pInstId);
          nextTree.meshId         = meshId;
          nextTree.tree           = root;
          
          //ConvertBvh4TwoLevel(root, offsets[0], a_depth + 1, a_level + 1);
          m_instNodesConnections.push_back(nextTree);
        }
      }
    }
    else // triangle leaf
    {
      m_convertedLayout[currNodeOffset].SetLeaf(1); // this is leaf with triangles
      m_convertedLayout[currNodeOffset].m_escapeIndex = 0;

      // read data from tri ... :)

    }
  }

  return treeOffset;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ConvertionResult EmbreeBVH4::ConvertMap()
{
  BVH4* bvh4 = nullptr;

  AccelData* accel = ((Accel*)m_sceneTopLevel)->intersectors.ptr;
  if (accel->type == AccelData::TY_BVH4)
    bvh4 = (BVH4*)accel;
  else 
    return ConvertionResult();

  BVH4::NodeRef root = bvh4->root;

  if (true)
  {
    //
    //
    m_convertedLayout.resize(0);
    m_convertedLayout.reserve(m_instMatrices.size() * 100 + m_totalMeshTriangleCount);

    m_convertedTrinagles.resize(0);
    m_convertedTrinagles.reserve(3 * m_totalMeshTriangleCount * 2);

    m_instNodesConnections.resize(0);
    m_instNodesConnections.reserve(m_instMatrices.size() + 10);

    //
    //

    size_t rootOffset = Alloc4BVHNodes(m_convertedLayout); // alloc top level bvh node (only one part of quad is used)

    float4x4 mIdentityMatrix;                                            // not used actually, just filled this because i want
    float4x4* pMatrix = (float4x4*)(&m_convertedLayout[rootOffset + 1]); // not used actually, just filled this because i want
    (*pMatrix) = mIdentityMatrix;                                        // not used actually, just filled this because i want

    embree::BBox3fa rootBox = CalcBoundingBoxOfChilds(root);
    CopyBounds(&m_convertedLayout[rootOffset], rootBox);

    // convert top level tree first
    //
    ConvertBvh4TwoLevel(root, rootOffset, 0, 0);
    //for (auto subtree : m_instNodesConnections)
    //  ConvertBvh4TwoLevel(subtree.tree, subtree.leftNodeOffset, 0, 1);

    // convert bottom level instances
    //
    m_instNodeMeshesRef.clear();
    
    for (auto subtree : m_instNodesConnections)
    {
      auto p = m_instNodeMeshesRef.find(subtree.meshId);
      if (p != m_instNodeMeshesRef.end())
      {
        const size_t  instNodeOffset = subtree.leftNodeOffset; // 4 'float8' nodes with matrix at [1] and [2]
        const size_t  subtreeOffset  = p->second;

        if (subtree.tree.isAlignedNode())
        {
          m_convertedLayout[instNodeOffset].SetLeaf(0);
          m_convertedLayout[instNodeOffset].SetLeftOffset(subtreeOffset / 4);
        }
        else
        {
          m_convertedLayout[instNodeOffset].SetLeaf(1);
          //m_convertedLayout[instNodeOffset].SetLeftOffset(offsets[0] / 4);
        }
    
      }
      else
      {
        size_t treeOffset = ConvertBvh4TwoLevel(subtree.tree, subtree.leftNodeOffset, 0, 1);
        m_instNodeMeshesRef[subtree.meshId] = treeOffset;

        if(treeOffset == size_t(-1))
          m_convertedLayout[subtree.leftNodeOffset].SetLeaf(1);
        else
          m_convertedLayout[subtree.leftNodeOffset].SetLeaf(0);

      }
    }

  }

  //
  //
  ConvertionResult res;

  if (m_convertedLayout.size() > 0)
  {
    res.pBVH          = &m_convertedLayout[0];
    res.pTriangleData = nullptr;
  }

  return res;
}

void  EmbreeBVH4::ConvertUnmap()
{
  m_convertedLayout    = std::vector<BVHNode>(); // free memory
  m_convertedTrinagles = std::vector<float4>();
}

Lite_Hit EmbreeBVH4::RayTrace(float3 ray_pos, float3 ray_dir)
{
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  Lite_Hit hit = Make_Lite_Hit(1e38f, 0xFFFFFFFF, HIT_NONE);

  RTCRay ray;

  ray.org[0] = ray_pos.x;
  ray.org[1] = ray_pos.y;
  ray.org[2] = ray_pos.z;

  ray.dir[0] = ray_dir.x;
  ray.dir[1] = ray_dir.y;
  ray.dir[2] = ray_dir.z;

  ray.tnear  = 0.0f;
  ray.tfar   = 1e38f;
  ray.geomID = RTC_INVALID_GEOMETRY_ID;
  ray.primID = RTC_INVALID_GEOMETRY_ID;
  ray.instID = RTC_INVALID_GEOMETRY_ID;
  ray.mask   = -1;
  ray.time   = 0;

  rtcIntersect(m_sceneTopLevel, ray);

  if (ray.geomID != RTC_INVALID_GEOMETRY_ID && ray.primID != RTC_INVALID_GEOMETRY_ID)
    return Make_Lite_Hit(ray.tfar, ray.primID, HIT_TRIANGLE);
  else
    return Make_Lite_Hit(1e38f, 0xFFFFFFFF, HIT_NONE);
}

float3 EmbreeBVH4::ShadowTrace(float3 ray_pos, float3 ray_dir, float t_far)
{
  RTCRay ray;

  ray.org[0] = ray_pos.x;
  ray.org[1] = ray_pos.y;
  ray.org[2] = ray_pos.z;

  ray.dir[0] = ray_dir.x;
  ray.dir[1] = ray_dir.y;
  ray.dir[2] = ray_dir.z;

  ray.tnear  = 0.0f;
  ray.tfar   = t_far;
  ray.geomID = RTC_INVALID_GEOMETRY_ID;
  ray.primID = RTC_INVALID_GEOMETRY_ID;
  ray.instID = RTC_INVALID_GEOMETRY_ID;
  ray.mask   = -1;
  ray.time   = 0;

  rtcOccluded(m_sceneTopLevel, ray);

  if (ray.geomID == RTC_INVALID_GEOMETRY_ID)
    return float3(1, 1, 1);
  else
    return float3(0, 0, 0);
}

extern "C" __declspec(dllexport) IBVHBuilder* CreateBuilder(char* cfg) { return new EmbreeBVH4; }

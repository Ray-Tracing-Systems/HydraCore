#include "bvh_access.h"

#include<memory>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EmbreeBVH4_2::EmbreeBVH4_2() : m_device(nullptr), m_algorithmFlags(RTC_INTERSECT1)
{
  m_dummy4bvh.resize(4);
  m_dummy3f4.resize(3);

  m_device = nullptr;

  for (int i = 0; i < MAXBVHTREES; i++)
  {
    m_tree[i].m_sceneTopLevel = nullptr;
    m_tree[i].m_sceneTriNum   = 0;

    m_tree[i].m_rtObjByMeshId.reserve(1000);
    m_tree[i].m_matByInstId.reserve(1000);
    m_tree[i].m_meshIdByInstId.reserve(1000);
  }
 
  m_ltrees.resize(1);
  m_ltrees[0].m_totalMeshTriangleCount = 0;
  m_ltreeId    = 0;
  m_earlySplit = false;
}

EmbreeBVH4_2::~EmbreeBVH4_2()
{
 
}

void EmbreeBVH4_2::Init(const char* cfg)
{
  m_device = rtcNewDevice("tri_accel=bvh4.triangle4v");
  //m_device = rtcNewDevice(nullptr);

  //rtcDeviceSetErrorFunction2(m_device, error_handler, nullptr); // #TODO: GO THIS BACK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! <============== !!!!!!!!!!!!!!!!!!!!!!

  for (int i = 0; i < MAXBVHTREES; i++)
  {
    m_tree[i].m_sceneTopLevel = rtcDeviceNewScene(m_device, BUILD_FLAGS, m_algorithmFlags);
    m_tree[i].m_sceneTriNum   = 0;
  }

  //if (cfg != nullptr && std::string(cfg) == "-allow_insert_copy 1")
  //  m_allowInsertCopies = true;
  //else
  //  m_allowInsertCopies = false;
}

void EmbreeBVH4_2::Destroy()
{
  ClearData();

  for (int i = 0; i < MAXBVHTREES; i++)
    m_tree[i].destroy();

  rtcDeleteDevice(m_device); 
  m_device = nullptr;
}

void EmbreeBVH4_2::ClearData()
{
  for (int i = 0; i < MAXBVHTREES; i++)
  {
    rtcDeleteScene(m_tree[i].m_sceneTopLevel);
    m_tree[i].m_sceneTopLevel = rtcDeviceNewScene(m_device, BUILD_FLAGS, m_algorithmFlags);
    m_tree[i].m_sceneTriNum   = 0;

    for (auto scn : m_tree[i].m_rtObjByMeshId)
      rtcDeleteScene(scn.second);

    m_tree[i].m_rtObjByMeshId.clear();
    m_tree[i].m_matByInstId.clear();
    m_tree[i].m_meshIdByInstId.clear();
  }
}

void EmbreeBVH4_2::ClearScene()
{
  for (int i = 0; i < MAXBVHTREES; i++)
  {
    for (auto scn : m_tree[i].m_rtObjByMeshId)
      rtcDeleteScene(scn.second);

    m_tree[i].m_rtObjByMeshId.clear();
    m_tree[i].m_matByInstId.clear();
    m_tree[i].m_meshIdByInstId.clear();

    rtcDeleteScene(m_tree[i].m_sceneTopLevel);
    m_tree[i].m_sceneTopLevel = rtcDeviceNewScene(m_device, BUILD_FLAGS, m_algorithmFlags);
    m_tree[i].m_sceneTriNum = 0;
  }

  if (m_earlySplit)
    m_pRep = std::make_unique<RefMesh>(this);
  else
    m_pRep = std::make_unique<TriMesh>(this);

  m_refsHash.clear();
  m_inputMeshData.clear();

  for (auto& ltree : m_ltrees)
    ltree.clear();
}


void EmbreeBVH4_2::GetBounds(float a_bMin[3], float a_bMax[3])
{
  a_bMin[0] = inf;
  a_bMin[1] = inf;
  a_bMin[2] = inf;

  a_bMax[0] = neg_inf;
  a_bMax[1] = neg_inf;
  a_bMax[2] = neg_inf;

  for (int i = 0; i < MAXBVHTREES; i++)
  {
    if (m_tree[i].m_sceneTriNum == 0)
      continue;

    RTCBounds bounds;
    rtcGetBounds(m_tree[i].m_sceneTopLevel, bounds);

    a_bMin[0] = fminf(a_bMin[0], bounds.lower_x);
    a_bMin[1] = fminf(a_bMin[1], bounds.lower_y);
    a_bMin[2] = fminf(a_bMin[2], bounds.lower_z);

    a_bMax[0] = fmaxf(a_bMax[0], bounds.upper_x);
    a_bMax[1] = fmaxf(a_bMax[1], bounds.upper_y);
    a_bMax[2] = fmaxf(a_bMax[2], bounds.upper_z);
  }
}

int EmbreeBVH4_2::InstanceTriangleMeshes(InstanceInputData a_data, int a_treeId, int a_realInstIdBase)
{
  if (a_treeId >= MAXBVHTREES)
    return -1;

  m_tree[a_treeId].m_sceneTriNum += (a_data.numIndices / 3);

  m_inputMeshData[a_data.meshId] = a_data;

  const int a_numIndices = a_data.numIndices;
  const int a_numVert    = a_data.numVert;

  const float* a_vert4f  = a_data.vert4f;
  const int* a_indices   = a_data.indices;
  

  RTCScene scene = m_pRep->CreateInternalGeom(a_vert4f, a_numVert, a_indices, a_numIndices, a_data.meshId);
  rtcCommit(scene);
  
  m_ltrees[m_ltreeId].m_totalMeshTriangleCount += (a_numIndices / 3);

  m_tree[a_treeId].m_rtObjByMeshId[a_data.meshId] = scene;
  
  for (int matrixId = 0; matrixId < a_data.numInst; matrixId++)
  {
    const float* matrix     = a_data.matrices + 16 * matrixId;
    const auto myInstanceId = rtcNewInstance2(m_tree[a_treeId].m_sceneTopLevel, scene);
  
    rtcSetTransform2(m_tree[a_treeId].m_sceneTopLevel, myInstanceId, RTC_MATRIX_ROW_MAJOR, matrix, 0);

    m_tree[a_treeId].m_matByInstId   [myInstanceId] = float4x4(matrix);
    m_tree[a_treeId].m_meshIdByInstId[myInstanceId] = a_data.meshId;
    m_tree[a_treeId].m_realInstId.push_back(a_realInstIdBase + matrixId);
  }

  return int(m_tree[a_treeId].m_matByInstId.size());
}

void EmbreeBVH4_2::CommitScene()
{  
  for (int i = 0; i < MAXBVHTREES; i++)
    rtcCommit(m_tree[i].m_sceneTopLevel);
}

inline static const bool IsThisFuckingEmbreeLeafWithTriangles(const int* pInstId)
{
  bool isZero = true;
  for (int i = 1; i <= 3; i++)
    if (pInstId[i] != 0)
      isZero = false;
  return !isZero;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t EmbreeBVH4_2::Alloc4BVHNodes(cvex::vector<BVHNode>& a_vector)
{
  size_t currSize = a_vector.size();
  a_vector.insert(a_vector.end(), m_dummy4bvh.begin(), m_dummy4bvh.end());
  return currSize;
}

size_t EmbreeBVH4_2::Alloc3Float4(cvex::vector<float4>& a_vector)
{
  size_t currSize = a_vector.size();
  a_vector.insert(a_vector.end(), m_dummy3f4.begin(), m_dummy3f4.end());
  return currSize;
}

size_t EmbreeBVH4_2::Alloc1Float4(cvex::vector<float4>& a_vector)
{
  size_t currSize = a_vector.size();
  a_vector.push_back(float4(0,0,0,0));
  return currSize;
}

inline static void CopyBounds(BVHNode* pNode, embree::BBox3fa box)
{
  pNode->m_boxMin.x = box.lower.x;
  pNode->m_boxMin.y = box.lower.y;
  pNode->m_boxMin.z = box.lower.z;

  pNode->m_boxMax.x = box.upper.x;
  pNode->m_boxMax.y = box.upper.y;
  pNode->m_boxMax.z = box.upper.z;
}

inline static embree::BBox3fa CalcBoundingBoxOfChilds(BVH4::NodeRef node)
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

#include <unordered_set>

#include "../../kernels/geometry/object_intersector.h"

void EmbreeBVH4_2::InsertTrainglesInLeaf(size_t currNodeOffset, BVH4::NodeRef node, EmbreeBVH4_2::LinearTree& lt, int a_meshId, const char* a_treeType)
{
  size_t objListOffset = Alloc1Float4(lt.m_convertedTrinagles);
  lt.m_convertedLayout[currNodeOffset].SetLeftOffset((unsigned int)objListOffset);

  // read data from tris
  //
  int totalTriNum = 0;
  size_t num;
  const Triangle4v* tri = (const Triangle4v*)node.leaf(num);

  auto laName = m_pRep->LayoutName();

  //std::unordered_set<int> trisWereAdded;
  const float3 leafMin = lt.m_convertedLayout[currNodeOffset].m_boxMin;
  const float3 leafMax = lt.m_convertedLayout[currNodeOffset].m_boxMax;

  if (laName == "custom")
  {
    using PrimType = embree::sse2::ObjectIntersector1<0>::Primitive;
    const PrimType* pdata = (const PrimType*)node.leaf(num);

    auto& triRefs       = m_refsHash[a_meshId];
    auto& inputMeshData = m_inputMeshData[a_meshId];

    const float4* vert4f = (const float4*)inputMeshData.vert4f;
    const int*    ind    = inputMeshData.indices;

    for (size_t i = 0; i < num; i++)
    {
      const PrimType prim = pdata[i];
      auto triRef = triRefs[prim.primID()];

      const int iA = ind[triRef.triId * 3 + 0];
      const int iB = ind[triRef.triId * 3 + 1];
      const int iC = ind[triRef.triId * 3 + 2];
      
      const float4 A = vert4f[iA];
      const float4 B = vert4f[iB];
      const float4 C = vert4f[iC];

      int triId = triRef.triId;
      int objId = a_meshId; // 
      int insId = -1;       // instance matrix id/offset 

      const int oldTriId = triId;

      size_t triOffsetF4 = Alloc3Float4(lt.m_convertedTrinagles);

      lt.m_convertedTrinagles[triOffsetF4 + 0] = float4(A.x, A.y, A.z, as_float(triId));
      lt.m_convertedTrinagles[triOffsetF4 + 1] = float4(B.x, B.y, B.z, as_float(objId));
      lt.m_convertedTrinagles[triOffsetF4 + 2] = float4(C.x, C.y, C.z, as_float(insId));

      totalTriNum++;
    }
  }
  else
  {
    for (size_t i = 0; i < num; i++)
    {
      for (size_t j = 0; j < tri[i].size(); j++)
      {
        size_t triOffsetF4 = Alloc3Float4(lt.m_convertedTrinagles);

        int triId = tri[i].primID(j);
        int objId = a_meshId; // 
        int insId = -1;       // instance matrix id/offset 

        const int oldTriId = triId;

        // you can put some flags here ... 
        //
        lt.m_convertedTrinagles[triOffsetF4 + 0] = float4(tri[i].v0.x[j], tri[i].v0.y[j], tri[i].v0.z[j], as_float(triId));
        lt.m_convertedTrinagles[triOffsetF4 + 1] = float4(tri[i].v1.x[j], tri[i].v1.y[j], tri[i].v1.z[j], as_float(objId));
        lt.m_convertedTrinagles[triOffsetF4 + 2] = float4(tri[i].v2.x[j], tri[i].v2.y[j], tri[i].v2.z[j], as_float(insId));

        totalTriNum++;
      }
    }
  }

  int* pTriData       = (int*)&(lt.m_convertedTrinagles[0]);
  int* pTriNumber     = (int*)(&lt.m_convertedTrinagles[objListOffset]); // put tri number to objListData.x .... 

  (*(pTriNumber + 0)) = int(objListOffset+1);
  (*(pTriNumber + 1)) = totalTriNum;
  (*(pTriNumber + 2)) = -1;
  (*(pTriNumber + 3)) = -1;

}

size_t EmbreeBVH4_2::ConvertBvh4TwoLevel(BVH4::NodeRef node, size_t currNodeOffset, int a_depth, int a_level, int a_meshId, const char* a_treeType, int a_treeId)
{
  if (m_ltrees.size() == 0)
    return 0;

  auto& lt = m_ltrees[m_ltreeId];

  size_t treeOffset = size_t(-1);

  auto nodeType = node.type();

  if (node.isAlignedNode())
  {
    BVH4::AlignedNode* n = node.alignedNode();

    size_t offsets[4] = { 0,0,0,0 };

    offsets[0] = Alloc4BVHNodes(lt.m_convertedLayout);
    offsets[1] = offsets[0] + 1;
    offsets[2] = offsets[0] + 2;
    offsets[3] = offsets[0] + 3;

    for (size_t i = 0; i<4; i++)
    {
      if (n->child(i) == BVH4::emptyNode)
        continue;

      auto box = n->bounds(i);
      auto pNode = &lt.m_convertedLayout[0] + offsets[i];

      CopyBounds(pNode, box);
    }

    lt.m_convertedLayout[currNodeOffset].SetLeaf(0);
    lt.m_convertedLayout[currNodeOffset].SetLeftOffset((unsigned int)(offsets[0] / 4));

    for (size_t i = 0; i < 4; i++)
    {
      if (n->child(i) == BVH4::emptyNode)
        continue;

      ConvertBvh4TwoLevel(n->child(i), offsets[i], a_depth + 1, a_level, a_meshId, a_treeType, a_treeId);
    }

    treeOffset = offsets[0];
  }
  else if (node.isLeaf())
  {
    size_t num = 0;
    const int* pInstId    = (const int*)node.leaf(num);
    const Triangle4v* tri = (const Triangle4v*)pInstId;
  
    //const bool leafWithTri = IsThisFuckingEmbreeLeafWithTriangles(pInstId);

    if (nodeType == 9 && a_level == 0 && (*(pInstId) < m_tree[m_ltreeId].m_realInstId.size()) ) // instance
    {
      const int meshId     = m_tree[m_ltreeId].m_meshIdByInstId[(*pInstId)];
      const int realInstId = m_tree[m_ltreeId].m_realInstId[*(pInstId)];
      const auto scn       = m_tree[m_ltreeId].m_rtObjByMeshId[meshId];
      
      BVH4* bvh4 = nullptr;
      AccelData* accel = ((Accel*)scn)->intersectors.ptr;
      if (accel->type == AccelData::TY_BVH4)
        bvh4 = (BVH4*)accel;
      
      if (bvh4 != nullptr)
      {
        BVH4::NodeRef root = bvh4->root;
      
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (currNodeOffset == 0) // special case: fix for single instance scene
        {
          size_t offsets[4] = { 0,0,0,0 };
      
          offsets[0] = Alloc4BVHNodes(lt.m_convertedLayout);
          offsets[1] = offsets[0] + 1;
          offsets[2] = offsets[0] + 2;
          offsets[3] = offsets[0] + 3;
      
          lt.m_convertedLayout[0].SetLeaf(0);
          lt.m_convertedLayout[0].SetLeftOffset(4 / 4);
          //lt.m_convertedLayout[0].m_boxMin = float3(0, 0, 0); 
          //lt.m_convertedLayout[0].m_boxMax = float3(0, 0, 0); 
          embree::BBox3fa rootBox = CalcBoundingBoxOfChilds(root);
          CopyBounds(&lt.m_convertedLayout[0], rootBox);
      
          for (size_t i = 1; i < 4; i++)
          {
            const auto nodeOffset = offsets[i];
            lt.m_convertedLayout[nodeOffset].SetLeaf(1);
            lt.m_convertedLayout[nodeOffset].SetLeftOffset(-1);
            lt.m_convertedLayout[nodeOffset].m_boxMin = float3(0, 0, 0);
            lt.m_convertedLayout[nodeOffset].m_boxMax = float3(0, 0, 0);
          }
      
          currNodeOffset = 4; // offsets[3] + 1;
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
      
        size_t offsets[4] = { 0,0,0,0 };
      
        offsets[0] = Alloc4BVHNodes(lt.m_convertedLayout);
        offsets[1] = offsets[0] + 1;
        offsets[2] = offsets[0] + 2;
        offsets[3] = offsets[0] + 3;
      
        lt.m_convertedLayout[currNodeOffset].SetLeaf(1);     // #watch this (!!!) different for current ray trace and OpenGL code
        lt.m_convertedLayout[currNodeOffset].SetInstance(1); // this is an instance (!!!)
        lt.m_convertedLayout[currNodeOffset].SetLeftOffset((unsigned int)(offsets[0] / 4));
      
        // lt.m_convertedLayout[offsets[0]] -> new root;
        // lt.m_convertedLayout[offsets[1]] -> first  half of matrix
        // lt.m_convertedLayout[offsets[2]] -> second half of matrix
        // lt.m_convertedLayout[offsets[3]] -> put instance id
        //
        float4x4 mInverse = inverse4x4(m_tree[m_ltreeId].m_matByInstId[(*pInstId)]);     // put matrix
        float4x4* pMatrix = (float4x4*)(&lt.m_convertedLayout[offsets[1]]);
        (*pMatrix) = mInverse;
      
        int4* pInstId2 = (int4*)(&lt.m_convertedLayout[offsets[3]]);    // put instance id
        (*pInstId2)    = int4(realInstId, meshId, 0, 0);
      
        // get bounding box
        //
        if (!root.isAlignedNode())  // get box from current node
        {
          lt.m_convertedLayout[offsets[0]].m_boxMin = lt.m_convertedLayout[currNodeOffset].m_boxMin;
          lt.m_convertedLayout[offsets[0]].m_boxMax = lt.m_convertedLayout[currNodeOffset].m_boxMax;
        }
        else // get box by calculating it from childs
        {
          embree::BBox3fa rootBox = CalcBoundingBoxOfChilds(root);
          CopyBounds(&lt.m_convertedLayout[offsets[0]], rootBox);
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
    else // triangle leaf
    {
      lt.m_convertedLayout[currNodeOffset].SetLeaf(1); // this is leaf with triangles
      lt.m_convertedLayout[currNodeOffset].SetInstance(0);
      InsertTrainglesInLeaf(currNodeOffset, node, lt, a_meshId, a_treeType);
    }
  }

  return treeOffset;
}

std::vector<BVH4*> EmbreeBVH4_2::ExtractBVH4Pointers()
{
  std::vector<BVH4*> trees2;

  trees2.resize(0);
  m_ltrees.resize(0);

  for (int bvhId = 0; bvhId < MAXBVHTREES; bvhId++)
  {
    if (m_tree[bvhId].m_sceneTriNum == 0)
    {
      m_ltrees.push_back(LinearTree("empty"));
      continue;
    }

    auto pSceneTop   = m_tree[bvhId].m_sceneTopLevel;
    AccelData* accel = ((Accel*)pSceneTop)->intersectors.ptr;

    if (accel->type == AccelData::TY_BVH4)
    {
      BVH4* bvh4 = (BVH4*)accel;
      if (bvh4 != nullptr)
      {
        trees2.push_back(bvh4);
        m_ltrees.push_back(LinearTree(bvh4->primTy->name));
      }
    }
    else if (accel->type == AccelData::TY_ACCELN)
    {
      m_ltrees.resize(0);
      AccelN* accelN = (AccelN*)(accel);
      for (size_t i = 0; i < accelN->accels.size(); i++)
      {
        if (accelN->accels[i]->intersectors.ptr->type == AccelData::TY_BVH4)
        {
          BVH4* bvh4 = (BVH4*)accelN->accels[i]->intersectors.ptr;
          if (bvh4 != nullptr)
          {
            auto laName = m_pRep->LayoutName();
            if ((bvh4->primTy->name == laName || bvh4->primTy->name == "object") && m_ltrees.size() < MAXBVHTREES)
            {
              trees2.push_back(bvh4);
              m_ltrees.push_back(LinearTree(bvh4->primTy->name));
            }
          }
        }
      }
    }

  } // for all MAXBVHTREES

  return trees2;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ConvertionResult EmbreeBVH4_2::ConvertMap()
{
  std::vector<BVH4*> trees = ExtractBVH4Pointers();

  if(trees.size() == 0)
    return ConvertionResult();

  int realTreeId = 0;
  for(m_ltreeId=0; m_ltreeId < m_ltrees.size(); m_ltreeId++)
  {
    if (realTreeId >= trees.size())
      break;

    BVH4* bvh4         = trees[realTreeId];
    BVH4::NodeRef root = bvh4->root;

    auto& lt = m_ltrees[m_ltreeId];

    if (m_tree[m_ltreeId].m_sceneTriNum == 0)
      continue;

    lt.m_totalMeshTriangleCount = 0;
    lt.m_convertedLayout.resize(0);
    lt.m_convertedLayout.reserve(m_tree[realTreeId].m_matByInstId.size() * 100 + lt.m_totalMeshTriangleCount + 100);

    lt.m_convertedTrinagles.resize(0);
    lt.m_convertedTrinagles.reserve(3 * lt.m_totalMeshTriangleCount * 2);

    m_instNodesConnections.resize(0);
    m_instNodesConnections.reserve(m_tree[realTreeId].m_matByInstId.size() + 10);

    //
    //
    size_t rootOffset = Alloc4BVHNodes(lt.m_convertedLayout); // alloc top level bvh node (only one part of quad is used)

    float4x4 mIdentityMatrix;                                               // not used actually, just filled this because i want
    float4x4* pMatrix = (float4x4*)(&lt.m_convertedLayout[rootOffset + 1]); // not used actually, just filled this because i want
    (*pMatrix) = mIdentityMatrix;                                           // not used actually, just filled this because i want

    embree::BBox3fa rootBox = CalcBoundingBoxOfChilds(root);
    CopyBounds(&lt.m_convertedLayout[rootOffset], rootBox);

    // convert top level tree first
    //
    ConvertBvh4TwoLevel(root, rootOffset, 0, 0, -1, bvh4->primTy->name.c_str(), realTreeId);
  
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
          lt.m_convertedLayout[instNodeOffset].SetLeaf(0);
          lt.m_convertedLayout[instNodeOffset].SetLeftOffset((unsigned int)(subtreeOffset / 4));
        }
        else
        {
          lt.m_convertedLayout[instNodeOffset].SetLeaf(1);
          InsertTrainglesInLeaf(instNodeOffset, subtree.tree, m_ltrees[m_ltreeId], subtree.meshId, ""); // don't pass bvh4->primTy.name.c_str() !
        }
    
      }
      else
      {
        size_t treeOffset = ConvertBvh4TwoLevel(subtree.tree, subtree.leftNodeOffset, 0, 1, subtree.meshId, "", realTreeId); // don't pass bvh4->primTy.name.c_str() !
        m_instNodeMeshesRef[subtree.meshId] = treeOffset;

        if (treeOffset == size_t(-1))
        {
          lt.m_convertedLayout[subtree.leftNodeOffset].SetLeaf(1);
          InsertTrainglesInLeaf(subtree.leftNodeOffset, subtree.tree, lt, subtree.meshId, ""); // don't pass bvh4->primTy.name.c_str() !
        }
        else
          lt.m_convertedLayout[subtree.leftNodeOffset].SetLeaf(0);

      }
    }

    realTreeId++;
  }

  // got the final result
  //
  ConvertionResult res;
  res.treesNum = int(m_ltrees.size());

  int finalBvhNumber = 0;

  for(size_t i=0;i<m_ltrees.size();i++)
  {
    if (m_ltrees[i].empty())
      continue;

    res.bvhType[finalBvhNumber]       = m_ltrees[i].embreeFormat.c_str();
    res.pBVH   [finalBvhNumber]       = &(m_ltrees[i].m_convertedLayout[0]);
    res.pTriangleData[finalBvhNumber] = (float*)&(m_ltrees[i].m_convertedTrinagles[0]);
    
    res.nodesNum[finalBvhNumber]      = int(m_ltrees[i].m_convertedLayout.size());
    res.trif4Num[finalBvhNumber]      = int(m_ltrees[i].m_convertedTrinagles.size());
    
    finalBvhNumber++;
  }

  res.treesNum = finalBvhNumber;
  return res;
}

void EmbreeBVH4_2::ConvertUnmap()
{
  for (auto& lt : m_ltrees)
    lt.m_convertedLayout.clear();

  m_instNodesConnections = std::vector<InstanceNode>();
}

constexpr int CURR_TEST_SCENE = 0;

Lite_Hit EmbreeBVH4_2::RayTrace(float3 ray_pos, float3 ray_dir)
{
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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

  rtcIntersect(m_tree[CURR_TEST_SCENE].m_sceneTopLevel, ray);

  if (ray.geomID != RTC_INVALID_GEOMETRY_ID && ray.primID != RTC_INVALID_GEOMETRY_ID)
  {
    Lite_Hit result;
    result.t      = ray.tfar;
    result.primId = ray.primID;
    result.instId = ray.instID;

    if (ray.instID != -1)
      result.geomId = m_tree[CURR_TEST_SCENE].m_meshIdByInstId[ray.instID]; // ray.geomID;
   
    return result;
  }
  else
    return Make_Lite_Hit(1e38f, 0xFFFFFFFF);
}

float3 EmbreeBVH4_2::ShadowTrace(float3 ray_pos, float3 ray_dir, float t_far)
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

  rtcOccluded(m_tree[CURR_TEST_SCENE].m_sceneTopLevel, ray);

  if (ray.geomID == RTC_INVALID_GEOMETRY_ID)
    return float3(1, 1, 1);
  else
    return float3(0, 0, 0);
}

#ifdef WIN32
extern "C" __declspec(dllexport) IBVHBuilder2* CreateBuilder2(char* cfg) { return new EmbreeBVH4_2; }
#else
extern "C" IBVHBuilder2* CreateBuilder2(char* cfg) { return new EmbreeBVH4_2; }
#endif



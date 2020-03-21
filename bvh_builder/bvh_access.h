#pragma once

#include "early_split.h"

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

static const RTCSceneFlags BUILD_FLAGS = (RTC_SCENE_STATIC | RTC_SCENE_HIGH_QUALITY | RTC_SCENE_INCOHERENT);

struct EmbreeBVH4_2 : public IBVHBuilder2
{
  EmbreeBVH4_2();
  ~EmbreeBVH4_2() override;

  void Init(const char* cgf) override;
  void Destroy() override;
  void GetBounds(float a_bMin[3], float a_bMax[3]) override;

  void ClearScene() override;
  void CommitScene() override;

  int  InstanceTriangleMeshes(InstanceInputData a_data, int a_treeId, int a_realInstIdBase) override;

  Lite_Hit RayTrace(float3 ray_pos, float3 ray_dir) override;                   // for CPU engine and test only
  float3   ShadowTrace(float3 ray_pos, float3 ray_dir, float t_far) override;   // for CPU engine and test only

  ConvertionResult ConvertMap() override;   // do actual converstion to our format
  void             ConvertUnmap() override; // free memory

protected:

  void ClearData();

  RTCDevice m_device;

  struct SingleTree
  {
    RTCScene              m_sceneTopLevel;
    int                   m_sceneTriNum;

    std::unordered_map<int, RTCScene> m_rtObjByMeshId;  ///< get embree scenes by meshId
    std::unordered_map<int, float4x4> m_matByInstId;    ///< get matrix by instanceId
    std::unordered_map<int, int>      m_meshIdByInstId; ///< get meshId by instanceId
    std::vector<int>                  m_realInstId;     ///< get actual instance id

    void destroy()
    {
      rtcDeleteScene(m_sceneTopLevel);
      m_sceneTopLevel = nullptr;
      m_sceneTriNum   = 0;

      m_rtObjByMeshId.clear();
      m_matByInstId.clear();
      m_meshIdByInstId.clear();
    }

  } m_tree[MAXBVHTREES];

  RTCAlgorithmFlags  m_algorithmFlags;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  size_t Alloc4BVHNodes(std::vector<BVHNode>& a_vector);
  size_t Alloc3Float4(cvex::vector<float4>& a_vector);
  size_t Alloc1Float4(cvex::vector<float4>& a_vector);

  struct LinearTree
  {
    LinearTree() { clear(); }
    LinearTree(const std::string& a_fmt) : embreeFormat(a_fmt) {}

    void clear() 
    { 
      m_convertedLayout.clear(); 
      m_convertedTrinagles.clear(); 
      m_totalMeshTriangleCount = 0; 
      embreeFormat = ""; 
    }

    bool empty() const { return (m_convertedLayout.size() <= 4); }

    cvex::vector<BVHNode> m_convertedLayout;
    cvex::vector<float4>  m_convertedTrinagles;
    size_t                m_totalMeshTriangleCount;
    std::string           embreeFormat;
  };

  bool m_earlySplit;

  std::vector<LinearTree>   m_ltrees;
  int                       m_ltreeId;

  cvex::vector<BVHNode>      m_dummy4bvh;
  cvex::vector<float4>       m_dummy3f4;

  size_t ConvertBvh4TwoLevel(BVH4::NodeRef node, size_t currNodeOffset, int depth, int instDepth, int a_meshId, const char* a_treeType, int a_treeId);
  void InsertTrainglesInLeaf(size_t currNodeOffset, BVH4::NodeRef node, EmbreeBVH4_2::LinearTree& lt, int a_meshId, const char* a_treeType);


  /////////////////////////////////////////////////////////////////////////////////////////////////////////

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

  std::vector<BVH4*> ExtractBVH4Pointers();

  struct PrimMesh
  {
    PrimMesh() {}
    virtual ~PrimMesh() {}

    virtual RTCScene CreateInternalGeom(const float* a_vert4f, int a_vertNum, const int* a_indices, const int a_indNum, int a_meshId, RTCScene a_pScene = nullptr) = 0;
    virtual std::string LayoutName() const = 0;
  };

  struct TriMesh : public PrimMesh
  {
    TriMesh(EmbreeBVH4_2* a_self) : pSelf(a_self) {}

    RTCScene CreateInternalGeom(const float* a_vert4f, int a_vertNum, const int* a_indices, const int a_indNum, int a_meshId, RTCScene a_pScene) override;
    virtual std::string LayoutName() const override { return std::string("triangle4v"); }

    EmbreeBVH4_2* pSelf;
  };

  struct RefMesh : public PrimMesh
  {
    RefMesh(EmbreeBVH4_2* a_self) : pSelf(a_self) {}

    RTCScene CreateInternalGeom(const float* a_vert4f, int a_vertNum, const int* a_indices, const int a_indNum, int a_meshId, RTCScene a_pScene) override;
    virtual std::string LayoutName() const override { return std::string("custom"); }

    EmbreeBVH4_2* pSelf;
  };

  std::unique_ptr<PrimMesh> m_pRep;
  std::unordered_map<int, std::vector<EarlySplit::TriRef> > m_refsHash;
  std::unordered_map<int, InstanceInputData>                m_inputMeshData;

};


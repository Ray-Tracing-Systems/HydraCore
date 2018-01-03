#pragma once

#include "globals.h"

#define MAXBVHTREES 4

struct ConvertionResult
{
  ConvertionResult() : treesNum(0)
  {
    for (int i = 0; i < MAXBVHTREES; i++)
    {
      pBVH[i]           = nullptr;
      pTriangleData[i]  = nullptr;
      pTriangleAlpha[i] = nullptr;
      nodesNum[i]       = 0;
      trif4Num[i]       = 0;
      triAfNum[i]       = 0;
      bvhType [i]       = nullptr;
    }
  }

  const char*    bvhType[MAXBVHTREES];
  const BVHNode* pBVH[MAXBVHTREES];
  const float*   pTriangleData[MAXBVHTREES];
  const uint2*   pTriangleAlpha[MAXBVHTREES];

  int            nodesNum[MAXBVHTREES];
  int            trif4Num[MAXBVHTREES];
  int            triAfNum[MAXBVHTREES];

  int            treesNum;
};

struct IBVHBuilder2 
{
  IBVHBuilder2() {}
  virtual ~IBVHBuilder2() {}

  virtual void Init(char* cgf) = 0;
  virtual void Destroy()       = 0;

  virtual void GetBounds(float a_bMin[3], float a_bMax[3]) = 0;

  struct InstanceInputData
  {
    int meshId;
    int numInst;
    const float* matrices;

    int numVert;
    int numIndices;
    const float* vert4f;
    const int*   indices;
  };

  virtual void ClearScene() = 0;
  virtual void CommitScene() = 0;

  virtual int InstanceTriangleMeshes(InstanceInputData a_data, int a_treeId, int a_realInstIdBase) = 0;

  virtual ConvertionResult ConvertMap() = 0;   // do actual converstion to our format
  virtual void             ConvertUnmap() = 0; // free memory

  virtual Lite_Hit RayTrace(float3 ray_pos, float3 ray_dir) = 0;                   // for CPU engine and test only
  virtual float3   ShadowTrace(float3 ray_pos, float3 ray_dir, float t_far) = 0;   // for CPU engine and test only


};

IBVHBuilder2* CreateBuilderFromDLL(const wchar_t* a_path, char* a_cfg);

#include "RenderDriverRTE.h"

#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <string>


/////////////////////////////////////////////////////////////////////////////////////////////////// Test & Debug

struct NodeWithMatrix
{
  NodeWithMatrix() : pNode(nullptr) {}
  NodeWithMatrix(const BVHNode* a_pNode) : pNode(a_pNode) {}
  NodeWithMatrix(const BVHNode* a_pNode, float4x4 a_mat) : pNode(a_pNode), matrix(a_mat) {}

  const BVHNode* pNode;
  float4x4       matrix;
};

void DebugSaveBVHNodesToArray4f(std::deque<NodeWithMatrix>& a_nodes, std::deque<NodeWithMatrix>& a_leafes, const int a_currLevel, const char* a_path)
{
  // (0) open file 'a_currLevel'
  //
  std::stringstream fileNameStream;
  fileNameStream << a_path << "/level_" << a_currLevel << ".array4f";
  std::stringstream fileNameStream2;
  fileNameStream2 << a_path << "/matrix_level_" << a_currLevel << ".array4f";

  const std::string fileName1 = fileNameStream.str();
  const std::string fileName2 = fileNameStream2.str();

  cvex::vector<float4> dataB(a_leafes.size() * 2 + a_nodes.size() * 2);
  cvex::vector<float4> dataM((a_leafes.size() + a_nodes.size()) * 4);

  size_t counter = 0;
  size_t counter2 = 0;

  // (1) append curr a_nodes to file 'a_currLevel'
  //
  if (a_leafes.size() > 0)
  {
    for (auto p = a_leafes.begin(); p != a_leafes.end(); ++p)
    {
      const BVHNode* node = p->pNode;
      const float4x4 matrix = p->matrix;

      dataB[counter + 0] = to_float4(node->m_boxMin, 0.0f);
      dataB[counter + 1] = to_float4(node->m_boxMax, 0.0f);

      dataM[counter2 + 0] = matrix.m_col[0];
      dataM[counter2 + 1] = matrix.m_col[1];
      dataM[counter2 + 2] = matrix.m_col[2];
      dataM[counter2 + 3] = matrix.m_col[3];

      counter += 2;
      counter2 += 4;
    }
  }

  // (2) append curr a_leafes filr 'a_currLevel'
  //
  if (a_nodes.size() > 0)
  {
    for (auto p = a_nodes.begin(); p != a_nodes.end(); ++p)
    {
      const BVHNode* node = p->pNode;
      const float4x4 matrix = p->matrix;

      dataB[counter + 0] = to_float4(node->m_boxMin, 0.0f);
      dataB[counter + 1] = to_float4(node->m_boxMax, 0.0f);

      dataM[counter2 + 0] = matrix.m_col[0];
      dataM[counter2 + 1] = matrix.m_col[1];
      dataM[counter2 + 2] = matrix.m_col[2];
      dataM[counter2 + 3] = matrix.m_col[3];

      counter += 2;
      counter2 += 4;
    }
  }

  int iSize = 0;

  // write boxes
  //
  {
    iSize = int(dataB.size());
    std::ofstream fout(fileName1.c_str(), std::ios::binary);
    fout.write((const char*)&iSize, sizeof(int));
    fout.write((const char*)&dataB[0], sizeof(float4)*dataB.size());
    fout.flush();
    fout.close();
  }

  // write matrices 
  //
  {
    iSize = int(dataM.size());
    std::ofstream fout2(fileName2.c_str(), std::ios::binary);
    fout2.write((const char*)&iSize, sizeof(int));
    fout2.write((const char*)&dataM[0], sizeof(float4)*dataM.size());
    fout2.flush();
    fout2.close();
  }
}

bool InvalidNode(const BVHNode* a_node)
{
  return (a_node->m_leftOffsetAndLeaf == 0xFFFFFFFF && a_node->m_escapeIndex == 0xFFFFFFFF);
}


void BFSTravsersal(NodeWithMatrix a_root, const char* a_path, int a_depth, std::deque<NodeWithMatrix> a_nodes, std::deque<NodeWithMatrix> a_leafes)
{
  DebugSaveBVHNodesToArray4f(a_nodes, a_leafes, a_depth, a_path);

  const size_t currSize = a_nodes.size();

  for (size_t nodesProcessed = 0; nodesProcessed < currSize; nodesProcessed++)
  {
    NodeWithMatrix currNode = a_nodes.front();

    const BVHNode* node = currNode.pNode;
    a_nodes.pop_front();

    if (node->Leaf() && !node->Instance())
      a_leafes.push_back(currNode);
    else if (node->Instance()) // instance
    {
      // extract instance matrix from node

      const size_t offset0 = node->GetLeftOffset() * 4 + 0;

      const BVHNode* child0 = a_root.pNode + offset0;

      const float4x4* pMatrix = (const float4x4*)(child0 + 1);

      float4x4 mTransform = mul(inverse4x4(*pMatrix), currNode.matrix);
      a_nodes.push_back(NodeWithMatrix(child0, mTransform));
    }
    else
    {
      const size_t offset0 = node->GetLeftOffset() * 4 + 0;
      const size_t offset1 = node->GetLeftOffset() * 4 + 1;
      const size_t offset2 = node->GetLeftOffset() * 4 + 2;
      const size_t offset3 = node->GetLeftOffset() * 4 + 3;

      const BVHNode* child0 = a_root.pNode + offset0;
      const BVHNode* child1 = a_root.pNode + offset1;
      const BVHNode* child2 = a_root.pNode + offset2;
      const BVHNode* child3 = a_root.pNode + offset3;

      if (!InvalidNode(child0)) a_nodes.push_back(NodeWithMatrix(child0, currNode.matrix));
      if (!InvalidNode(child1)) a_nodes.push_back(NodeWithMatrix(child1, currNode.matrix));
      if (!InvalidNode(child2)) a_nodes.push_back(NodeWithMatrix(child2, currNode.matrix));
      if (!InvalidNode(child3)) a_nodes.push_back(NodeWithMatrix(child3, currNode.matrix));
    }

  }

  if (a_nodes.size() > 0)
    BFSTravsersal(a_root, a_path, a_depth + 1, a_nodes, a_leafes);
}


void RenderDriverRTE::DebugSaveBVH(const std::string& a_folderName, const ConvertionResult& a_inBVH)
{
  std::deque<NodeWithMatrix> nodes, leafes;
  nodes.push_back(NodeWithMatrix(a_inBVH.pBVH[0]));
  
  BFSTravsersal(a_inBVH.pBVH[0], a_folderName.c_str(), 0, nodes, leafes);
}


struct BVHStat
{
  BVHStat() { memset(this, 0, sizeof(BVHStat)); }

  double bytesForBoxes;
  double bytesForTriList;
  double bytesTotal;

  int maxDeep;
  int maxTrianglePerLeaf;
  float avgTrianglePerLeaf;

  int leafesNum;
  int trianglesNum;
};


void ScanBVH(BVHStat* a_out, const BVHNode* a_root, const BVHNode* node, const float4* a_objList, int a_currLevel)
{

  if (node->Leaf() && !node->Instance())
  {
    if (a_currLevel > a_out->maxDeep)
      a_out->maxDeep = a_currLevel;

    if (node->m_leftOffsetAndLeaf != 0xFFFFFFFF)
    {
      const int leaf_offset     = EXTRACT_OFFSET(node->m_leftOffsetAndLeaf);
      const int2 objectListInfo = getObjectList(leaf_offset, a_objList);
      const int triNum          = objectListInfo.y;

      a_out->avgTrianglePerLeaf += triNum;
      if (triNum > a_out->maxTrianglePerLeaf)
        a_out->maxTrianglePerLeaf = triNum;

      a_out->leafesNum++;
    }
  }
  else if (node->Instance()) // instance
  {
    // extract instance matrix from node
    //
    const size_t offset0    = node->GetLeftOffset() * 4 + 0;
    const BVHNode* child0   = a_root + offset0;
    const float4x4* pMatrix = (const float4x4*)(child0 + 1);
  
    ScanBVH(a_out, a_root, child0, a_objList, a_currLevel + 1);
  }
  else
  {
    const size_t offset0 = node->GetLeftOffset() * 4 + 0;
    const size_t offset1 = node->GetLeftOffset() * 4 + 1;
    const size_t offset2 = node->GetLeftOffset() * 4 + 2;
    const size_t offset3 = node->GetLeftOffset() * 4 + 3;
  
    const BVHNode* child0 = a_root + offset0;
    const BVHNode* child1 = a_root + offset1;
    const BVHNode* child2 = a_root + offset2;
    const BVHNode* child3 = a_root + offset3;
  
    ScanBVH(a_out, a_root, child0, a_objList, a_currLevel + 1);
    ScanBVH(a_out, a_root, child1, a_objList, a_currLevel + 1);
    ScanBVH(a_out, a_root, child2, a_objList, a_currLevel + 1);
    ScanBVH(a_out, a_root, child3, a_objList, a_currLevel + 1);
  }

 
}

void RenderDriverRTE::PrintBVHStat(const ConvertionResult& a_inBVH, bool traverseThem)
{
  BVHStat stat;

  for (int bvhId = 0; bvhId < a_inBVH.treesNum; bvhId++)
  {
    stat.bytesForBoxes   += double(a_inBVH.nodesNum[bvhId]*sizeof(BVHNode));
    stat.bytesForTriList += double(a_inBVH.trif4Num[bvhId]*sizeof(float4)) + double(a_inBVH.triAfNum[bvhId]*2*sizeof(int));

    if(traverseThem)
      ScanBVH(&stat, a_inBVH.pBVH[bvhId], a_inBVH.pBVH[bvhId], (const float4*)a_inBVH.pTriangleData[bvhId], 0);
  }

  stat.avgTrianglePerLeaf = stat.avgTrianglePerLeaf / float(stat.leafesNum);
  stat.bytesTotal         = stat.bytesForBoxes + stat.bytesForTriList;

  auto oldPrecition = std::cout.precision(4);


  std::vector<std::string> treeTypes;
  for (int bvhId = 0; bvhId < a_inBVH.treesNum; bvhId++)
  {
    const bool plain     = (std::string(a_inBVH.bvhType[bvhId]) == "triangle4v");
    const bool haveAlpha = (a_inBVH.pTriangleAlpha[bvhId] != nullptr);

    if (plain && haveAlpha)
      treeTypes.push_back("TravA");
    else if(plain && !haveAlpha)
      treeTypes.push_back("TravS");
    else if (!plain && haveAlpha)
      treeTypes.push_back("TravInstA");
    else
      treeTypes.push_back("TravInstS");
  }

  //std::cout << std::endl;
  std::cout << "bvh trees   num = " << a_inBVH.treesNum << "\ttypes are (";
  for (auto name : treeTypes)
    std::cout << name.c_str() << ", ";
  std::cout << ")" << std::endl;
  std::cout << "bvh box (%) mem = " << 100.0f*float(stat.bytesForBoxes   / stat.bytesTotal) << " %" << std::endl;
  std::cout << "bvh tri (%) mem = " << 100.0f*float(stat.bytesForTriList / stat.bytesTotal) << " %" << std::endl;
  std::cout << "bvh total   mem = " << float(stat.bytesTotal) / float(1024 * 1024) << " MB" << std::endl;

  if (traverseThem)
  {
    //std::cout << std::endl;
    std::cout << "bvh max deep    = " << stat.maxDeep << std::endl;
    std::cout << "avg tri/leaf    = " << stat.avgTrianglePerLeaf << std::endl;
    std::cout << "max tri/leaf    = " << stat.maxTrianglePerLeaf << std::endl;
  }

  std::cout.precision(oldPrecition);
}


int FindMeshIdByTreeScan(const BVHNode* a_root, const float4* a_objList, const BVHNode* node)
{
  if (!IsValidNode(*node))
    return -1;

  if (node->Leaf() && !node->Instance())
  {
    if (node->m_leftOffsetAndLeaf != 0xFFFFFFFF)
    {
      const int leaf_offset     = EXTRACT_OFFSET(node->m_leftOffsetAndLeaf);
      const int2 objectListInfo = getObjectList(leaf_offset, a_objList);
      const int triNum          = objectListInfo.y;

      const int NUM_FETCHES_TRI = 3; // sizeof(struct ObjectListTriangle) / sizeof(float4);
      const int triAddressStart = objectListInfo.x;
      const int triAddressEnd   = triAddressStart + objectListInfo.y*NUM_FETCHES_TRI;

      for (int triAddress = triAddressStart; triAddress < triAddressEnd; triAddress += NUM_FETCHES_TRI)
      {
        const float4 data1 = a_objList[triAddress + 0];
        const float4 data2 = a_objList[triAddress + 1];
        const float4 data3 = a_objList[triAddress + 2];

        const float3 A_pos = to_float3(data1);
        const float3 B_pos = to_float3(data2);
        const float3 C_pos = to_float3(data3);

        const int primId = as_int(data1.w);
        const int geomId = as_int(data2.w);
        //const int instId   = as_int(data3.w);

        return geomId;
      }
      
      return -1;
    }
  }
  else
  {
    const size_t offset0 = node->GetLeftOffset() * 4 + 0;
    const size_t offset1 = node->GetLeftOffset() * 4 + 1;
    const size_t offset2 = node->GetLeftOffset() * 4 + 2;
    const size_t offset3 = node->GetLeftOffset() * 4 + 3;

    const BVHNode* child0 = a_root + offset0;
    const BVHNode* child1 = a_root + offset1;
    const BVHNode* child2 = a_root + offset2;
    const BVHNode* child3 = a_root + offset3;

    int child0Id = FindMeshIdByTreeScan(a_root, a_objList, child0);
    if (child0Id >= 0)
      return child0Id;

    int child1Id = FindMeshIdByTreeScan(a_root, a_objList, child1);
    if (child1Id >= 0)
      return child1Id;

    int child2Id = FindMeshIdByTreeScan(a_root, a_objList, child2);
    if (child2Id >= 0)
      return child2Id;

    int child3Id = FindMeshIdByTreeScan(a_root, a_objList, child3);
    if (child3Id >= 0)
      return child3Id;

    return -1;
  }

  return -1;
}

void ScanBVHToListAllInstances(std::vector<float4x4>& a_outMatrices, 
                               std::vector<int>& a_instId, 
                               std::vector<int>& a_meshId, 
                               const BVHNode* a_root, const float4* a_objList, const BVHNode* node, int a_currLevel)
{

  if (node->Leaf() && !node->Instance())
  {
    
  }
  else if (node->Instance()) // instance
  {
    // extract instance matrix from node
    //
    const size_t offset0    = node->GetLeftOffset() * 4 + 0;
    const BVHNode* child0   = a_root + offset0;
    const float4x4* pMatrix = (const float4x4*)(child0 + 1);
    const int*      pInstId = (const int*)    (pMatrix + 1);

    a_outMatrices.push_back(*pMatrix);
    a_instId.push_back(*pInstId);

    int meshId = FindMeshIdByTreeScan(a_root, a_objList, child0);
    a_meshId.push_back(meshId);
  }
  else
  {
    const size_t offset0 = node->GetLeftOffset() * 4 + 0;
    const size_t offset1 = node->GetLeftOffset() * 4 + 1;
    const size_t offset2 = node->GetLeftOffset() * 4 + 2;
    const size_t offset3 = node->GetLeftOffset() * 4 + 3;

    const BVHNode* child0 = a_root + offset0;
    const BVHNode* child1 = a_root + offset1;
    const BVHNode* child2 = a_root + offset2;
    const BVHNode* child3 = a_root + offset3;

    ScanBVHToListAllInstances(a_outMatrices, a_instId, a_meshId, a_root, a_objList, child0, a_currLevel + 1);
    ScanBVHToListAllInstances(a_outMatrices, a_instId, a_meshId, a_root, a_objList, child1, a_currLevel + 1);
    ScanBVHToListAllInstances(a_outMatrices, a_instId, a_meshId, a_root, a_objList, child2, a_currLevel + 1);
    ScanBVHToListAllInstances(a_outMatrices, a_instId, a_meshId, a_root, a_objList, child3, a_currLevel + 1);
  }
}


void RenderDriverRTE::DebugPrintBVHInfo(const ConvertionResult& a_inBVH, const char* a_fileName)
{
  std::ofstream fout(a_fileName);

  fout << "bvhtrees num = " << a_inBVH.treesNum << std::endl;

  for (int treeId = 0; treeId < a_inBVH.treesNum; treeId++)
  {
    fout << std::endl;
    fout << "bvh[" << treeId << "] = {" << std::endl;

    std::vector<float4x4> instMat;
    std::vector<int>      instId;
    std::vector<int>      meshId;
    ScanBVHToListAllInstances(instMat, instId, meshId, 
                              a_inBVH.pBVH[treeId], (const float4*)a_inBVH.pTriangleData[treeId], a_inBVH.pBVH[treeId], 0);

    fout << "  nodes_num     = " << a_inBVH.nodesNum[treeId] << std::endl;
    fout << "  tri_data_size = " << a_inBVH.trif4Num[treeId] << std::endl;
    fout << "  instance_list = [";
    for (auto index : instId)
      fout << index << ", ";
    fout << "]" << std::endl;

    fout << "  meshid11_list = [";
    for (auto index : meshId)
      fout << index << ", ";
    fout << "]" << std::endl;

    fout << "  meshid22_list = [";
    for (auto index : instId)
      fout << m_meshIdByInstId[index] << ", ";
    fout << "]" << std::endl;

    fout << "}" << std::endl;
  }

  fout.close();
}

void RenderDriverRTE::DebugTestAlphaTestTable(const std::vector<uint2>& a_alphaTable, int a_trif4Num)
{
  m_pHWLayer->PrepareEngineGlobals();
  const EngineGlobals* a_globals = m_pHWLayer->GetEngineGlobals();
  const int4* a_texStorage = (const int4*)m_pTexStorage->GetBegin();

  for (int triAddress = 0; triAddress < a_trif4Num; triAddress++ )
  {
    const uint2 alphaId0 = a_alphaTable[triAddress + 0];

    const float2 A_tex = decompressTexCoord16(alphaId0.y);

    if (int(alphaId0.x) < 0)
    {
      int a = 2;
    }

    const float2 texCoord   = A_tex;
    const int samplerOffset = (alphaId0.x == 0xFFFFFFFF || alphaId0.x == INVALID_TEXTURE || int(alphaId0.x) < 0) ? INVALID_TEXTURE : 0;

    int test = int(alphaId0.x);

    const float3 alphaColor = sample2DLite(samplerOffset, texCoord, &a_alphaTable[0] + alphaId0.x, a_texStorage, a_globals);
    const float selector = fmax(alphaColor.x, fmax(alphaColor.y, alphaColor.z));

    std::cout << "triAddress = " << triAddress << std::endl;
  }
}

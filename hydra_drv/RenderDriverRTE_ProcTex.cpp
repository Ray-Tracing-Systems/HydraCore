#include "RenderDriverRTE.h"
#pragma warning(disable:4996) // for wcsncpy to be ok

#include <iostream>
#include <queue>
#include <string>
#include <regex>
#include <algorithm>

#include "../../HydraAPI/hydra_api/HydraXMLHelpers.h"
#include "../../HydraAPI/hydra_api/HydraInternal.h"

using ProcTexInfo   = RenderDriverRTE::ProcTexInfo;
using AOProcTexInfo = RenderDriverRTE::AOProcTexInfo;

bool MaterialNodeHaveProceduralTextures(pugi::xml_node a_node, const std::unordered_map<int, ProcTexInfo>& a_ids, const std::unordered_map<int, pugi::xml_node >& a_matNodes)
{
  if (a_node.name() == std::wstring(L"material") && a_node.attribute(L"type").as_string() == std::wstring(L"hydra_blend"))
  {
    int mid1 = a_node.attribute(L"node_top").as_int();
    int mid2 = a_node.attribute(L"node_bottom").as_int();

    auto p1 = a_matNodes.find(mid1);
    auto p2 = a_matNodes.find(mid2);

    if (p1 != a_matNodes.end())
    {
      bool childHaveProcTextures = MaterialNodeHaveProceduralTextures(p1->second, a_ids, a_matNodes);
      if (childHaveProcTextures)
        return true;
    }

    if (p2 != a_matNodes.end())
    {
      bool childHaveProcTextures = MaterialNodeHaveProceduralTextures(p2->second, a_ids, a_matNodes);
      if (childHaveProcTextures)
        return true;
    }
  }

  if (std::wstring(a_node.name()) == L"texture")
  {
    int32_t id = a_node.attribute(L"id").as_int();
    return (a_ids.find(id) != a_ids.end()) || (std::wstring(a_node.attribute(L"type").as_string()) == L"texref_proc");
  }

  bool childHaveProc = false;

  for (auto child : a_node.children())
  {
    childHaveProc = childHaveProc || MaterialNodeHaveProceduralTextures(child, a_ids, a_matNodes);
    if (childHaveProc)
      break;
  }

  return childHaveProc;
}

void FindAllProcTextures(pugi::xml_node a_node, const std::unordered_map<int, ProcTexInfo>& a_ids, const std::unordered_map<int, pugi::xml_node >& a_matNodes,
                         std::vector<std::tuple<int, ProcTexInfo> >& a_outVector)
{

  if (a_node.name() == std::wstring(L"material") && a_node.attribute(L"type").as_string() == std::wstring(L"hydra_blend"))
  {
    int mid1 = a_node.attribute(L"node_top").as_int();
    int mid2 = a_node.attribute(L"node_bottom").as_int();

    auto p1 = a_matNodes.find(mid1);
    auto p2 = a_matNodes.find(mid2);

    if(p1!= a_matNodes.end())
      FindAllProcTextures(p1->second, a_ids, a_matNodes, a_outVector);

    if (p2 != a_matNodes.end())
      FindAllProcTextures(p2->second, a_ids, a_matNodes, a_outVector);
  }

  if (std::wstring(a_node.name()) == L"texture")
  {
    const int32_t id = a_node.attribute(L"id").as_int();

    auto p = a_ids.find(id);

    if (p != a_ids.end() || (std::wstring(a_node.attribute(L"type").as_string()) == L"texref_proc"))
    {
      a_outVector.push_back(std::tuple<int, ProcTexInfo>(p->first, p->second));
    }
  }

  for (auto child : a_node.children())
    FindAllProcTextures(child, a_ids, a_matNodes, a_outVector);

}


void ReadAllProcTexArgsFromMaterialNode(pugi::xml_node a_node, const std::unordered_map<int, pugi::xml_node >& a_matNodes, 
                                        std::vector<ProcTexParams>& a_procTexParams)
{
  if (a_node.name() == std::wstring(L"material") && a_node.attribute(L"type").as_string() == std::wstring(L"hydra_blend"))
  {
    int mid1 = a_node.attribute(L"node_top").as_int();
    int mid2 = a_node.attribute(L"node_bottom").as_int();

    auto p1 = a_matNodes.find(mid1);
    auto p2 = a_matNodes.find(mid2);

    if (p1 != a_matNodes.end())
      ReadAllProcTexArgsFromMaterialNode(p1->second, a_matNodes, a_procTexParams);

    if (p2 != a_matNodes.end())
      ReadAllProcTexArgsFromMaterialNode(p2->second, a_matNodes, a_procTexParams);
  }

  if (std::wstring(a_node.name()) == L"texture")
  {
    if (std::wstring(a_node.attribute(L"type").as_string()) == L"texref_proc")
    {
      std::vector<float> datav;
      for (pugi::xml_node arg : a_node.children(L"arg"))
      {
        std::wstring type    = arg.attribute(L"type").as_string();
        int32_t arraySize    = arg.attribute(L"size").as_int();
        const wchar_t* value = arg.attribute(L"val").as_string();

        std::wstringstream strIn(value);

        if (type == L"sampler2D" || type == L"int")
        {
          for (int i = 0; i < arraySize; i++)
          {
            int x = 0;
            strIn >> x;
            datav.push_back(as_float(x));
          }
        }
        else if (type == L"float")
        {
          for (int i = 0; i < arraySize; i++)
          {
            float x = 0;
            strIn >> x;
            datav.push_back(x);
          }
        }
        else if (type == L"float2")
        {
          for (int i = 0; i < 2*arraySize; i++)
          {
            float x = 0;
            strIn >> x;
            datav.push_back(x);
          }
        }
        else if (type == L"float3")
        {
          for (int i = 0; i < 3 * arraySize; i++)
          {
            float x = 0;
            strIn >> x;
            datav.push_back(x);
          }
        }
        else if (type == L"float4")
        {
          for (int i = 0; i < 4 * arraySize; i++)
          {
            float x = 0;
            strIn >> x;
            datav.push_back(x);
          }
        }

      } // for for (pugi::xml_node arg : a_node.children(L"arg"))

      ProcTexParams res;
      res.data     = datav;
      res.texId    = a_node.attribute(L"id").as_int();
      a_procTexParams.push_back(res);
    }
  }

  for (auto child : a_node.children())
    ReadAllProcTexArgsFromMaterialNode(child, a_matNodes, a_procTexParams);
}


void PutTexParamsToMaterialWithDamnTable(std::vector<ProcTexParams>& a_procTexParams, const std::unordered_map<int, ProcTexInfo>& a_allProcTextures,
                                         std::shared_ptr<RAYTR::IMaterial> a_pMaterial)
{
  if (a_procTexParams.size() == 0)
    return;
  
  // estimate needed size
  //
  int allSize = 0;
  for (auto procTex : a_allProcTextures)
  {
    auto p = std::find_if(a_procTexParams.begin(), a_procTexParams.end(),
                          [procTex](ProcTexParams x) { return (x.texId == procTex.first); });

    if (p != a_procTexParams.end())
      allSize += int(p->data.size());
  }

  // allocate memory in 'prtexDataTail.data' and get pointer
  //
  const int numOfMaterialPagesForArgData = allSize / PLAIN_MATERIAL_DATA_SIZE + 1;
  a_pMaterial->prtexDataTail.data.resize(numOfMaterialPagesForArgData);
  float* data = &(a_pMaterial->prtexDataTail.data[0].data[0]);

  // fill table and put args
  //
  int* table  = (int*)(&a_pMaterial->prtexDataTail.offsetTable.data[0]);

  const int MAX_TABLE_SIZE  = PLAIN_MATERIAL_DATA_SIZE;
  const int MAX_TABLE_ELEMS = MAX_TABLE_SIZE/2;

  int counter    = 0;
  int currOffset = 0;

  for (auto procTex : a_allProcTextures)
  { 
    if (counter < MAX_TABLE_ELEMS)
    {
      auto p = std::find_if(a_procTexParams.begin(), a_procTexParams.end(), 
                           [procTex](ProcTexParams x) { return (x.texId == procTex.first); });

      if (p != a_procTexParams.end())
      {
        table[counter * 2 + 0] = procTex.first;
        table[counter * 2 + 1] = currOffset;

        for (int i = 0; i < p->data.size(); i++)
          data[currOffset + i] = p->data[i];

        currOffset += int(p->data.size());
      }
      else
      {
        table[counter * 2 + 0] = procTex.first;
        table[counter * 2 + 1] = -1;
      }

      counter++;
    }
  }

  table[MAX_TABLE_SIZE - 1] = counter; // write total proc textures number here

}


ProcTextureList MakePTListFromTupleArray(const std::vector<std::tuple<int, ProcTexInfo> >& procTextureIds)
{
  ProcTextureList ptl;
  InitProcTextureList(&ptl);

  int counterf4 = 0;

  for (auto texIdAndType : procTextureIds)
  {
    int texId = std::get<0>(texIdAndType);

    if (counterf4 < MAXPROCTEX)
    {
      ptl.id_f4[counterf4] = texId;
      counterf4++;
    }
    else
      std::cerr << "[RTE]: too many float4 procedural textures for tex id = " << texId << std::endl;
  }

  ptl.currMaxProcTex = counterf4;

  return ptl;
}

SWTexSampler SamplerFromTexref(const pugi::xml_node a_node, bool allowAlphaToRGB = false);
const pugi::xml_node SamplerNode(const pugi::xml_node a_node);

AOProcTexInfo ReadAOFromNode(pugi::xml_node aoNode)
{
  const std::wstring aoType = aoNode.attribute(L"hemisphere").as_string();

  float aoLength              = aoNode.attribute(L"length").as_float();
  int   aoUpDownType          = AO_TYPE_NONE;
  bool  aoHitOnlySameInstance = (aoNode.attribute(L"local").as_int() == 1);

  if (aoType == L"up" || aoType == L"corner")
    aoUpDownType = AO_TYPE_UP;
  else if (aoType == L"down" || aoType == L"edge")
    aoUpDownType = AO_TYPE_DOWN;
  else if (aoType == L"both")
    aoUpDownType = AO_TYPE_BOTH;

  // put them in material head
  //
  if (aoUpDownType == AO_TYPE_NONE)
    return AOProcTexInfo();

  SWTexSampler samplerAO = DummySampler(); 
  if (SamplerNode(aoNode) != nullptr)
    samplerAO = SamplerFromTexref(SamplerNode(aoNode));

  AOProcTexInfo res;
  res.hitOnlySameInstance = aoHitOnlySameInstance;
  res.rayLen              = aoLength;
  res.rayLenSam           = samplerAO;
  res.upDownType          = aoUpDownType;
  return res;
}

void PutAOToMaterialSlot(std::shared_ptr<RAYTR::IMaterial> a_pMaterial, AOProcTexInfo a_aoInfo, int a_slotId)
{

  if (a_slotId == 0)
  {
    ((int*)(a_pMaterial->m_plain.data))[PROC_TEX_AO_TYPE] = a_aoInfo.upDownType;
    a_pMaterial->m_plain.data[PROC_TEX_AO_LENGTH]         = a_aoInfo.rayLen;

    SWTexSampler samplerAO = a_aoInfo.rayLenSam;

    if (a_aoInfo.hitOnlySameInstance)
      a_pMaterial->AddFlags(PLAIN_MATERIAL_LOCAL_AO1);
    else
      a_pMaterial->AddFlags(a_pMaterial->GetFlags() & (~PLAIN_MATERIAL_LOCAL_AO1));

    a_pMaterial->PutSamplerAt(samplerAO.texId, samplerAO, PROC_TEX_TEX_ID, PROC_TEXMATRIX_ID, PROC_TEX_AO_SAMPLER);
  }
  else if (a_slotId == 1)
  {
    ((int*)(a_pMaterial->m_plain.data))[PROC_TEX_AO_TYPE2] = a_aoInfo.upDownType;
    a_pMaterial->m_plain.data[PROC_TEX_AO_LENGTH2]         = a_aoInfo.rayLen;

    SWTexSampler samplerAO = a_aoInfo.rayLenSam;

    if (a_aoInfo.hitOnlySameInstance)
      a_pMaterial->AddFlags(PLAIN_MATERIAL_LOCAL_AO2); // #TODO: Add different flags for second ao slot !!!
    else
      a_pMaterial->AddFlags(a_pMaterial->GetFlags() & (~PLAIN_MATERIAL_LOCAL_AO2));

    a_pMaterial->PutSamplerAt(samplerAO.texId, samplerAO, PROC_TEX_TEX_ID2, PROC_TEXMATRIX_ID2, PROC_TEX_AO_SAMPLER2);
  }


}


void PutAOToMaterialHead(const std::vector< std::tuple<int, ProcTexInfo> >& a_procTextureIds, std::shared_ptr<RAYTR::IMaterial> a_pMaterial)
{
  ProcTexInfo textureThatUsesAO;

  for (auto ptex : a_procTextureIds)
  {
    auto texInfo = std::get<1>(ptex);

    if (texInfo.ao.upDownType != AO_TYPE_NONE)
    {
      textureThatUsesAO = texInfo;
      break;
    }
  }

  if (textureThatUsesAO.ao.upDownType == AO_TYPE_NONE)
    return;

  ((int*)(a_pMaterial->m_plain.data))[PROC_TEX_AO_TYPE] = textureThatUsesAO.ao.upDownType;
  a_pMaterial->m_plain.data[PROC_TEX_AO_LENGTH]         = textureThatUsesAO.ao.rayLen;

  SWTexSampler samplerAO = textureThatUsesAO.ao.rayLenSam;

  if (textureThatUsesAO.ao.hitOnlySameInstance)
    a_pMaterial->AddFlags(PLAIN_MATERIAL_LOCAL_AO1);

  a_pMaterial->PutSamplerAt(samplerAO.texId, samplerAO, PROC_TEX_TEX_ID, PROC_TEXMATRIX_ID, PROC_TEX_AO_SAMPLER);
}


static void FindAllAONodes(pugi::xml_node a_node, const std::unordered_map<int, pugi::xml_node >& a_matNodes, 
                           std::vector<pugi::xml_node>& out_AONodes)
{

  if (a_node.name() == std::wstring(L"ao"))
    out_AONodes.push_back(a_node);
  else
  {
    for (auto child : a_node.children())
      FindAllAONodes(child, a_matNodes, out_AONodes);
  }

  if (a_node.name() == std::wstring(L"material") && a_node.attribute(L"type").as_string() == std::wstring(L"hydra_blend"))
  {
    int mid1 = a_node.attribute(L"node_top").as_int();
    int mid2 = a_node.attribute(L"node_bottom").as_int();

    auto p1 = a_matNodes.find(mid1);
    auto p2 = a_matNodes.find(mid2);

    if (p1 != a_matNodes.end())
      FindAllAONodes(p1->second, a_matNodes, out_AONodes);

    if (p2 != a_matNodes.end())
      FindAllAONodes(p2->second, a_matNodes, out_AONodes);
  }

}


void OverrideAOInMaterialHead(pugi::xml_node a_materialNode, const std::unordered_map<int, pugi::xml_node >& a_matNodes, 
                              std::shared_ptr<RAYTR::IMaterial> a_pMaterial)
{ 
  // (1) get all ao nodes from material
  //
  std::vector<pugi::xml_node> aoNodes;
  FindAllAONodes(a_materialNode, a_matNodes, aoNodes);

  if (aoNodes.size() == 0)
    return;

  // (2) group all nodes by their hemisphere.
  //
  std::vector<pugi::xml_node> groupedNodes = aoNodes;  //#TODO: implement this carefully !!!

  // (3) write droup data to 2 ao slots in material head
  //
  if (groupedNodes.size() == 1)
  {
    auto aoParams = ReadAOFromNode(groupedNodes[0]);

    if (aoParams.upDownType == AO_TYPE_NONE)
      return;

    PutAOToMaterialSlot(a_pMaterial, aoParams, 0);
  }
  else if (groupedNodes.size() >= 2)
  {
    auto aoParams0 = ReadAOFromNode(groupedNodes[0]);
    auto aoParams1 = ReadAOFromNode(groupedNodes[1]);

    if (aoParams0.upDownType == AO_TYPE_NONE)
      return;

    PutAOToMaterialSlot(a_pMaterial, aoParams0, 0);
    PutAOToMaterialSlot(a_pMaterial, aoParams1, 1);
  }

}


//#TODO: move m_texShadersWasRecompiled outside of this, don't call BeginTexturesUpdate/EndTexturesUpdate if don't have textures tp update
//

void RenderDriverRTE::BeginTexturesUpdate()
{
  if (m_texShadersWasRecompiled)
    return;

  std::string pathIn  = "../hydra_drv/shaders/texproc.cl";
  std::string pathOut = "../hydra_drv/shaders/texproc_generated.cl";

  const std::string installPath2 = HydraInstallPath();

  if (!isFileExists(pathIn))   pathIn  = installPath2 + "shaders/texproc.cl";
  if (!isFileExists(pathOut))  pathOut = installPath2 + "shaders/texproc_generated.cl";

  m_inProcTexFile.open(pathIn.c_str());
  #ifndef RECOMPILE_PROCTEX_FROM_STRING
  m_outProcTexFile.open(pathOut.c_str());
  #endif

  m_outProcTexFileName = pathOut;

  if (!m_inProcTexFile.is_open())
    std::cerr << "RenderDriverRTE::BeginTexturesUpdate(): can't open in texproc file";

  #ifndef RECOMPILE_PROCTEX_FROM_STRING
  if (!m_outProcTexFile.is_open())
    std::cerr << "RenderDriverRTE::BeginTexturesUpdate(): can't open out texproc file";
  #endif

  std::string line;
  while (std::getline(m_inProcTexFile, line))
  {
    m_outProcTexFile << line.c_str() << std::endl;

    if (line.find("#PUT_YOUR_PROCEDURAL_TEXTURES_HERE:") != std::string::npos)
      break;
  }

  m_outProcTexFile << std::endl;

  m_procTextures.clear();
}

static const std::string currentDateTime()
{
  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
  // for more information about date/time format
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
  return buf;
}

void RenderDriverRTE::EndTexturesUpdate()
{
  if (m_texShadersWasRecompiled) 
    return;

  if (m_procTextures.size() == 0)
  {
    m_inProcTexFile.close();
    #ifndef RECOMPILE_PROCTEX_FROM_STRING
    m_outProcTexFile.close();
    #endif
    return;
  }

  const std::string spaces = "    ";

  std::string line;
  while (std::getline(m_inProcTexFile, line))
  {
    m_outProcTexFile << line.c_str() << std::endl;
    if (line.find("#PUT_YOUR_PROCEDURAL_TEXTURES_EVAL_HERE:") != std::string::npos)
    {
      m_outProcTexFile << std::endl;
      m_outProcTexFile << spaces.c_str() << "int counter = 0; " << std::endl;
      int counter = 0;
      for (auto ptex : m_procTextures)
      {
        if(ptex.second.call == "")
        {
          std::cerr << "[HydraCore]: RenderDriverRTE::EndTexturesUpdate, empty texture call code, id =  " << ptex.first << std::endl;
        }

        m_outProcTexFile << "    if(materialHeadHaveTargetProcTex(pHitMaterial," << ptex.first << ") && counter < MAXPROCTEX)" << std::endl;
        m_outProcTexFile << "    {" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  __global const float* stack = fdata + findArgDataOffsetInTable(" << ptex.first << ", table);" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  " << "ptl.fdata4[counter] = to_float3(" << ptex.second.call.c_str() << ");" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  " << "ptl.id_f4 [counter] = "           << ptex.first << ";" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  " << "counter++;" << std::endl;
        m_outProcTexFile << "    }" << std::endl;
        m_outProcTexFile << std::endl;   
        counter++;
      }

      m_outProcTexFile << spaces.c_str() << "ptl.currMaxProcTex = counter;";
      m_outProcTexFile << std::endl;

      std::string currtime = currentDateTime();
      m_outProcTexFile << "    // BREAK SHADER CACHE AT: " << currtime << "\n";
    }
  }

  m_inProcTexFile.close();

  #ifndef RECOMPILE_PROCTEX_FROM_STRING
  m_outProcTexFile.flush();
  m_outProcTexFile.close();
  m_pHWLayer->RecompileProcTexShaders(m_outProcTexFileName);
  #else
  m_pHWLayer->RecompileProcTexShaders(m_outProcTexFile.str());
  if(true)
  {
    std::ofstream fout(m_outProcTexFileName);
    fout << m_outProcTexFile.str();
    fout.close();
  }
  #endif

  m_texShadersWasRecompiled = true;
}

bool RenderDriverRTE::UpdateImageProc(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode)
{
  std::regex tail("_PROCTEXTAILTAG_");
  std::regex stack("stack\\[");

  const std::wstring returnType = a_texNode.child(L"code").child(L"generated").child(L"return").attribute(L"type").as_string();
  const int retT = (returnType == L"float4") ? 4 : 1;

  // (1) remember proc tex id, ret type and call string
  //
  const std::wstring callW = a_texNode.child(L"code").child(L"generated").child(L"call").text().as_string();

  std::string callS(callW.begin(), callW.end());
  callS = std::regex_replace(callS, tail,  "in_texStorage1, in_globals, hr_viewVectorHack");

  ProcTexInfo prTexInfo;
  prTexInfo.call = callS;
  prTexInfo.retT = retT;
  prTexInfo.ao   = ReadAOFromNode(a_texNode.child(L"ao"));

  m_procTextures[a_texId] = prTexInfo;

  // (2) insert code inside opencl program;
  //
  const std::wstring fileName = m_libPath + L"/" + a_texNode.child(L"code").attribute(L"loc").as_string();
  const std::string  fileNameS(fileName.begin(), fileName.end());

  std::ifstream procTexIn(fileNameS.c_str());

  if (procTexIn.is_open())
  {
    std::string line;
    while (std::getline(procTexIn, line))
    {
      line = std::regex_replace(line.c_str(), tail, 
                                " __global const float4* restrict in_texStorage1, __global const EngineGlobals* restrict in_globals, const float3 hr_viewVectorHack");
      m_outProcTexFile << line.c_str() << std::endl;
    }
  }

  // (3) end
  //
  return true;
}

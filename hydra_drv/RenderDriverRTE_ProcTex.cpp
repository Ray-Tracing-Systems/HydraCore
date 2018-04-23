#include "RenderDriverRTE.h"
#pragma warning(disable:4996) // for wcsncpy to be ok

#include <iostream>
#include <queue>
#include <string>
#include <regex>
#include <algorithm>

#include "../../HydraAPI/hydra_api/HydraXMLHelpers.h"
#include "../../HydraAPI/hydra_api/HydraInternal.h"


bool MaterialNodeHaveProceduralTextures(pugi::xml_node a_node, const std::unordered_map<int, int>& a_ids)
{
  if (std::wstring(a_node.name()) == L"texture")
  {
    int32_t id = a_node.attribute(L"id").as_int();
    return (a_ids.find(id) != a_ids.end()) || (std::wstring(a_node.attribute(L"type").as_string()) == L"texref_proc");
  }

  bool childHaveProc = false;

  for (auto child : a_node.children())
  {
    childHaveProc = childHaveProc || MaterialNodeHaveProceduralTextures(child, a_ids);
    if (childHaveProc)
      break;
  }

  return childHaveProc;
}

void FindAllProcTextures(pugi::xml_node a_node, const std::unordered_map<int, int>& a_ids, std::vector<std::tuple<int, int> >& a_outVector)
{
  if (std::wstring(a_node.name()) == L"texture")
  {
    const int32_t id = a_node.attribute(L"id").as_int();

    auto p = a_ids.find(id);

    if (p != a_ids.end() || (std::wstring(a_node.attribute(L"type").as_string()) == L"texref_proc"))
    {
      a_outVector.push_back(std::tuple<int, int>(p->first, p->second));
    }
  }

  for (auto child : a_node.children())
    FindAllProcTextures(child, a_ids, a_outVector);

}


void ReadAllProcTexArgsFromMaterialNode(pugi::xml_node a_node, std::vector<ProcTexParams>& a_procTexParams)
{
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
    ReadAllProcTexArgsFromMaterialNode(child, a_procTexParams);
}


void PutTexParamsToMaterialWithDamnTable(std::vector<ProcTexParams>& a_procTexParams, const std::unordered_map<int, int>& a_allProcTextures, 
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
      allSize += p->data.size();
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

        currOffset += p->data.size();
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


ProcTextureList MakePTListFromTupleArray(const std::vector<std::tuple<int, int> >& procTextureIds)
{
  ProcTextureList ptl;
  InitProcTextureList(&ptl);

  int counterf4 = 0;

  for (auto texIdAndType : procTextureIds)
  {
    int texId = std::get<0>(texIdAndType);
    int texTy = std::get<1>(texIdAndType);

    if (counterf4 < MAXPROCTEX)
    {
      ptl.id_f4[counterf4] = texId;
      counterf4++;
    }
    else
      std::cerr << "[RTE]: too many float4 procedural textures for tex id = " << texId << std::endl;
  }

  return ptl;
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
  m_outProcTexFile.open(pathOut.c_str());

  m_outProcTexFileName = pathOut;

  if (!m_inProcTexFile.is_open())
    std::cerr << "RenderDriverRTE::BeginTexturesUpdate(): can't open in texproc file";

  if (!m_outProcTexFile.is_open())
    std::cerr << "RenderDriverRTE::BeginTexturesUpdate(): can't open out texproc file";

  std::string line;
  while (std::getline(m_inProcTexFile, line))
  {
    m_outProcTexFile << line.c_str() << std::endl;

    if (line.find("#PUT_YOUR_PROCEDURAL_TEXTURES_HERE:") != std::string::npos)
      break;
  }

  m_outProcTexFile << std::endl;

  m_procTexturesCall.clear();
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

  if (m_procTexturesCall.size() == 0)
  {
    m_inProcTexFile.close();
    m_outProcTexFile.close();
    return;
  }

  const std::string spaces = "    ";

  std::string line;
  while (std::getline(m_inProcTexFile, line))
  {
    m_outProcTexFile << line.c_str() << std::endl;
    if (line.find("#PUT_YOUR_PROCEDURAL_TEXTURES_EVAL_HERE:") != std::string::npos)
    {
      m_outProcTexFile << spaces.c_str() << "float3 texcolor[" << m_procTexturesCall.size() << "];" << std::endl;
      m_outProcTexFile << spaces.c_str() << "int    texid[" << m_procTexturesCall.size() << "];" << std::endl;
      m_outProcTexFile << std::endl;

      int counter = 0;
      for (auto ptex : m_procTexturesCall)
      {
        m_outProcTexFile << "    if(materialHeadHaveTargetProcTex(pHitMaterial," << ptex.first << "))" << std::endl;
        m_outProcTexFile << "    {" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  __global const float* stack = fdata + findArgDataOffsetInTable(" << ptex.first << ", table);" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  " << "texcolor[" << counter << "] = to_float3(" << ptex.second.c_str() << ");" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  " << "texid   [" << counter << "] = "           << ptex.first << ";" << std::endl;
        m_outProcTexFile << "    }" << std::endl;
        m_outProcTexFile << "    else" << std::endl;
        m_outProcTexFile << "    {" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  " << "texcolor[" << counter << "] = " << "make_float3(0,0,1)" << ";" << std::endl;
        m_outProcTexFile << spaces.c_str() << "  " << "texid   [" << counter << "] = " << ptex.first << ";" << std::endl;
        m_outProcTexFile << "    }" << std::endl << std::endl; 
        counter++;
      }

      m_outProcTexFile << std::endl;
      
      counter = 0;
      for (auto ptex : m_procTexturesCall)
      {
        for (int j = 0; j < MAXPROCTEX; j++)
        {
          m_outProcTexFile << spaces.c_str() << "if(ptl.id_f4[" << j << "] == texid[" << counter << "])" << std::endl;
          m_outProcTexFile << spaces.c_str() << "  ptl.fdata4[" << j << "] = texcolor[" << counter << "];" << std::endl;
          m_outProcTexFile << std::endl;
        }

        counter++;
      }

      std::string currtime = currentDateTime();
      m_outProcTexFile << "    // BREAK SHADER CACHE AT: " << currtime << "\n";
    }
  }

  m_inProcTexFile.close();
  m_outProcTexFile.close();

  m_pHWLayer->RecompileProcTexShaders(m_outProcTexFileName.c_str());

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
  callS = std::regex_replace(callS, tail,  "in_texStorage1, in_globals");
  //callS = std::regex_replace(callS, stack, "pMat->data[");

  m_procTexturesRetT[a_texId] = retT;
  m_procTexturesCall[a_texId] = callS;

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
      line = std::regex_replace(line.c_str(), tail, " __global const float4* restrict in_texStorage1, __global const EngineGlobals* restrict in_globals");
      m_outProcTexFile << line.c_str() << std::endl;
    }
  }

  // (3) end
  //
  return true;
}
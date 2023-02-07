#include "RenderDriverRTE.h"

#include <iostream>
#include <queue>
#include <string>

using RAYTR::BumpParameters;

pugi::xml_attribute SmoothLvlAttr(const pugi::xml_node a_heightNode);

BumpParameters BumpAmtAndLvl(const pugi::xml_node a_materialNode)
{
  float bumpAmt  = 0.0f;
  float smothLvl = 0.0f;

  const pugi::xml_node heightNode   = a_materialNode.child(L"displacement").child(L"height_map");
  const pugi::xml_attribute amtAttr = heightNode.attribute(L"amount");
  const pugi::xml_attribute lvlAttr = SmoothLvlAttr(heightNode);

  const std::wstring btype = a_materialNode.child(L"displacement").attribute(L"type").as_string();
  if (btype == L"height_bump" || btype == L"bump")
  {
    if (amtAttr != nullptr)
      bumpAmt = 0.5f*amtAttr.as_float();

    if (lvlAttr != nullptr)
      smothLvl = 10.0f*lvlAttr.as_float();
  }

  return BumpParameters(bumpAmt, smothLvl);
}


std::wstring RenderDriverRTE::GetNormalMapParameterStringForCache(int textureIdNM, pugi::xml_node a_materialNode)
{
  std::wstringstream strOut;
  strOut << textureIdNM;

  auto params = BumpAmtAndLvl(a_materialNode);
  strOut << L" " << params.x << L" " << params.y;
  return strOut.str();
}

//bool HR_SaveLDRImageToFile(const wchar_t* a_fileName, int w, int h, int32_t* data);

int32_t RenderDriverRTE::GetCachedAuxNormalMatId(int32_t a_matId, const PlainMaterial& a_mat, int textureIdNM, pugi::xml_node a_materialNode)
{
  int32_t auxTexId = INVALID_TEXTURE; // AuxNormalTexPerMaterial(a_matId, textureIdNM);

  std::wstring normalMapHashStr = GetNormalMapParameterStringForCache(textureIdNM, a_materialNode);

  if (m_texturesProcessedNM.find(normalMapHashStr) == m_texturesProcessedNM.end() && textureIdNM >= 0)
  {
    auxTexId = AuxNormalTexPerMaterial(a_matId, textureIdNM);

    int w, h;
    std::vector<uchar4> noramsDataTemp;
    const uchar4* pNormals = GetAuxNormalMapFromDisaplacement(noramsDataTemp, a_mat, textureIdNM, a_materialNode, &w, &h);

    //HR_SaveLDRImageToFile(L"D:/temp/Cells_Balls_n_calculated.png", w, h, (int32_t*)pNormals);

    if (pNormals != nullptr)
      UpdateImageAux(auxTexId, w, h, 4, pNormals);  // #TODO: add normal map compression  here ... ?
    else
      auxTexId = INVALID_TEXTURE;

    m_texturesProcessedNM[normalMapHashStr] = auxTexId;
  }
  else
    auxTexId = m_texturesProcessedNM[normalMapHashStr];

  return auxTexId;
}

extern int g_maxCPUThreads;

const uchar4* RenderDriverRTE::GetAuxNormalMapFromDisaplacement(std::vector<uchar4>& normals, const PlainMaterial& mat, int textureIdNM, pugi::xml_node a_materialNode, int* pW, int* pH)
{
  const std::wstring btype = a_materialNode.child(L"displacement").attribute(L"type").as_string();
  const uchar4* pNormals = nullptr;

  if (m_texTable.size() <= textureIdNM)
    m_texTable = m_pTexStorage->GetTable();

  assert(m_texTable.size() > textureIdNM);

  const int texOffset = m_texTable[textureIdNM];
  if (texOffset < 0)
    return nullptr;

  const int4* begin   = (const int4*)m_pTexStorage->GetBegin();
  const int4* header  = begin + texOffset;

  (*pW) = header->x;
  (*pH) = header->y;

  const uchar4* pBumpData = (const uchar4*)(header + 1);
  std::vector<uchar4> dataConverted;

  if(header->z == 1 && header->w == 1)
  {
    dataConverted.resize(header->x*header->y);
    const int totalSize = int(dataConverted.size());
    const unsigned char* dataIn = (const unsigned char*)(header + 1);

    #pragma omp parallel for num_threads(g_maxCPUThreads)
    for (int i = 0; i < totalSize; i++)
    {
      const unsigned char pixIn = dataIn[i];
      uchar4 res;
      res.x = pixIn;
      res.y = pixIn;
      res.z = pixIn;
      res.w = pixIn;
      dataConverted[i] = res;
    }
    
    pBumpData = dataConverted.data();
  }
  else if (header->w == 16) // convert to 8 bit
  {
    dataConverted.resize(header->x*header->y);
    const float4* dataIn  = (const float4*)(header + 1);
    uchar4*       dataOut = dataConverted.data();
    
    const int totalSize = int(dataConverted.size());

    #pragma omp parallel for num_threads(g_maxCPUThreads)
    for (int i = 0; i < totalSize; i++)
    {
      const float4 pixIn  = dataIn[i];
      const float4 pixOut = clamp(pixIn*255.0f, 0.0f, 255.0f);

      uchar4 res;
      res.x = (unsigned char)pixOut.x;
      res.y = (unsigned char)pixOut.y;
      res.z = (unsigned char)pixOut.z;
      res.w = (unsigned char)pixOut.w;
      dataOut[i] = res;
    }

    pBumpData = dataConverted.data();
  }

  if (btype == L"height_bump")
  {
    const int  flags     = materialGetFlags(&mat);
    const bool invHeight = (flags & PLAIN_MATERIAL_INVERT_HEIGHT) != 0;

    auto params = BumpAmtAndLvl(a_materialNode);
    normals     = m_pHWLayer->NormalMapFromDisplacement(header->x, header->y, pBumpData, params.x, invHeight, params.y);
    pNormals    = normals.data();
  }
  else if (btype == L"normal_bump") // #TODO: process (header->w == 16) case ... 
  {
    pNormals = (const uchar4*)(header + 1);
  }

  return pNormals;
}

void SaveBMP(const wchar_t* fname, const int* pixels, int w, int h);

bool RenderDriverRTE::UpdateImageAux(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data)
{
  SWTextureHeader texheader;

  texheader.width  = w;
  texheader.height = h;
  texheader.depth  = 4;
  texheader.bpp    = bpp;

  const size_t inDataBSz  = size_t(w)*size_t(h)*size_t(bpp);
  const int    align      = int(m_pTexStorageAux->GetAlignSizeInBytes());
  const size_t headerSize = roundBlocks(sizeof(SWTextureHeader), align);
  const size_t totalSize  = roundBlocks(inDataBSz, align) + headerSize;

  m_pTexStorageAux->Update(a_texId, nullptr, totalSize);

  m_pTexStorageAux->UpdatePartial(a_texId, &texheader, 0, sizeof(SWTextureHeader));
  m_pTexStorageAux->UpdatePartial(a_texId, a_data, headerSize, inDataBSz);
  
  if (false)
  {
    std::wstringstream nameOut;
    nameOut << L"out/normalmap_" << a_texId << L".bmp";
    const std::wstring fname = nameOut.str();
    SaveBMP(fname.c_str(), (const int*)a_data, w, h);
  }

  return true;
}

void RenderDriverRTE::AverageStats(const MRaysStat& a_stats, MRaysStat& a_statsRes, int& counter)
{
  const float alpha = 1.0f / float(counter + 1);

  a_statsRes.bounceTimeMS = (1.0f - alpha)*a_statsRes.bounceTimeMS + a_stats.bounceTimeMS*alpha;
  a_statsRes.evalHitMs    = (1.0f - alpha)*a_statsRes.evalHitMs    + a_stats.evalHitMs*alpha;
  a_statsRes.nextBounceMs = (1.0f - alpha)*a_statsRes.nextBounceMs + a_stats.nextBounceMs*alpha;
  a_statsRes.raysPerSec   = (1.0f - alpha)*a_statsRes.raysPerSec   + a_stats.raysPerSec*alpha;

  a_statsRes.reorderTimeMs  = (1.0f - alpha)*a_statsRes.reorderTimeMs + a_stats.reorderTimeMs*alpha;
  a_statsRes.samplesPerSec  = (1.0f - alpha)*a_statsRes.samplesPerSec + a_stats.samplesPerSec*alpha;
  a_statsRes.sampleTimeMS   = (1.0f - alpha)*a_statsRes.sampleTimeMS  + a_stats.sampleTimeMS*alpha;
  a_statsRes.shadowTimeMs   = (1.0f - alpha)*a_statsRes.shadowTimeMs  + a_stats.shadowTimeMs*alpha;
  a_statsRes.samLightTimeMs = (1.0f - alpha)*a_statsRes.samLightTimeMs + a_stats.samLightTimeMs*alpha;
  a_statsRes.shadeTimeMs    = (1.0f - alpha)*a_statsRes.shadeTimeMs + a_stats.shadeTimeMs*alpha;

  a_statsRes.procTexMs     = (1.0f - alpha)*a_statsRes.procTexMs + a_stats.procTexMs*alpha;

  a_statsRes.traceTimePerCent = int( (1.0f - alpha)*float(a_statsRes.traceTimePerCent) + float(a_stats.traceTimePerCent*alpha) );
  a_statsRes.traversalTimeMs  = (1.0f - alpha)*a_statsRes.traversalTimeMs + a_stats.traversalTimeMs*alpha;

  counter++;
}

#include "RenderDriverRTE.h"
#pragma warning(disable:4996) // for wcsncpy to be ok

#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <string>

#include "HDRImageLite.h"

HDRImageLite::HDRImageLite() : m_width(0), m_height(0), m_channels(4)
{

}

HDRImageLite::HDRImageLite(int w, int h, int channels, const float* data) : m_width(w), m_height(h), m_channels(channels)
{
  m_data.resize(w*h*channels);

  if (data != NULL)
    memcpy(&m_data[0], data, w*h*channels * sizeof(float));
  else
    memset(&m_data[0], 0, w*h*channels * sizeof(float));
}

HDRImageLite::~HDRImageLite()
{

}

void HDRImageLite::gaussBlur(int radius, float sigma)
{
  std::vector<float> gKernel;
  createGaussKernelWeights1D(2 * radius + 1, gKernel, sigma);

  //std::cerr << "gKernel = ";
  //for(auto i=0;i<gKernel.size();i++)
  //  std::cerr << gKernel[i] << " ";
  //std::cerr << std::endl;

  std::vector<float> tempData(m_width*m_height*m_channels);

  float tempColor[4];

  int h = m_height;
  int w = m_width;

  // blur rows
  //
  for (int y = 0; y < h; y++)
  {
    int offsetY = y*w*m_channels;

    for (int x = 0; x < w; x++)
    {
      for (int k = 0; k<m_channels; k++)
        tempColor[k] = 0.0f;

      int minX = x - radius;
      int maxX = x + radius;
      if (minX < 0) minX = 0;
      if (maxX > w-1) maxX = w-1;

      float summW = 0.0f;
      for (int p = minX; p <= maxX; p++)
      {
        float weight = gKernel[p + radius - x];
        for (int k = 0; k<m_channels; k++)
          tempColor[k] += m_data[offsetY + p*m_channels + k] * weight;
        summW += weight;
      }

      for (int k = 0; k<m_channels; k++)
        tempData[offsetY + x*m_channels + k] = tempColor[k] / (summW + 1e-5f);
    }
  }

  if (h == 1)
    m_data = tempData;
  else
  {
    // blur cols
    //
    for (int x = 0; x < w; x++)
    {
      for (int y = 0; y < h; y++)
      {
        int minY = y - radius;
        int maxY = y + radius;
        if (minY < 0) minY = 0;
        if (maxY > h-1) maxY = h-1;

        for (int k = 0; k < m_channels; k++)
          tempColor[k] = 0.0f;

        float summW = 0.0f;
        for (int p = minY; p <= maxY; p++)
        {
          float weight = gKernel[p + radius - y];
          for (int k = 0; k < m_channels; k++)
            tempColor[k] += tempData[p*w*m_channels + x*m_channels + k] * weight;
          summW += weight;
        }

        for (int k = 0; k < m_channels; k++)
          m_data[y*w*m_channels + x*m_channels + k] = tempColor[k] / (summW + 1e-5f);
      }
    }
  }

}




void createGaussKernelWeights(int size, std::vector<float>& gKernel, float a_sigma)
{
  gKernel.resize(size*size);

  // set standard deviation to 1.0
  float sigma = a_sigma;
  float s = 2.0f * sigma * sigma;

  // sum is for normalization
  float sum = 0.0f;

  int halfSize = size / 2;

  // generate 5x5 kernel
  //
  for (int x = -halfSize; x <= halfSize; x++)
  {
    for (int y = -halfSize; y <= halfSize; y++)
    {
      float r = sqrtf(float(x*x + y*y));
      int index = (y + halfSize)*size + x + halfSize;
      gKernel[index] = (exp(-(r*r) / s)) / (3.141592654f * s);
      sum += gKernel[index];
    }
  }

  // normalize the Kernel
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j)
      gKernel[i*size + j] /= sum;

}


void createGaussKernelWeights1D(int size, std::vector<float>& gKernel, float a_sigma)
{
  gKernel.resize(size);

  // set standard deviation to 1.0
  //
  const float sigma = a_sigma;
  const float s = 2.0f * sigma * sigma;

  // sum is for normalization
  float sum = 0.0;
  int halfSize = size / 2;

  for (int x = -halfSize; x <= halfSize; x++)
  {
    float r = sqrtf(float(x*x));
    int index = x + halfSize;
    gKernel[index] = (exp(-(r) / s)) / (3.141592654f * s);
    sum += gKernel[index];
  }

  // normalize the Kernel
  for (int i = 0; i < size; ++i)
    gKernel[i] /= sum;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float maxcolorc(float3 v) { return fmax(v.x, fmax(v.y, v.z)); }

static cvex::vector<float4> resizeToHalfSizef4(const float4* pixels, int width, int height)
{
  width  = width / 2;
  height = height / 2;
  if (width == 0)  width = 1;
  if (height == 0) height = 1;

  cvex::vector<float4> copy(width*height);

  float4* oldArray = (float4*)(&pixels[0]);
  float4* newArray = (float4*)(&copy[0]);

  for (int y = 0; y < height; y++)
  {
    int offset1 = (2 * y + 0) * 2 * width;
    int offset2 = (2 * y + 1) * 2 * width;

    for (int x = 0; x < width; x++)
    {
      float4 x0 = oldArray[offset1 + 2 * x + 0];
      float4 x1 = oldArray[offset1 + 2 * x + 1];
      float4 x2 = oldArray[offset2 + 2 * x + 0];
      float4 x3 = oldArray[offset2 + 2 * x + 1];

      newArray[y*width + x] = 0.25f*(x0 + x1 + x2 + x3);
    }
  }

  return copy;
}



/**
\brief  create luminance image for futher construction of pdf table. Doadditional blur and add 0.1 to prevent zero length intervals;
\param  pixels  - input random variable in rage [0, 1]
\param  width   - input float array. it must be a result of prefix summ - i.e. it must be sorted.
\param  height  - size of extended array - i.e. a_accum[N-1] == summ(a_accum[0 .. N-2]).
\return luminance image

*/

std::vector<float> LuminanceFromFloat4Image(const float4* pixels, int& width, int& height)
{
  cvex::vector<float4> imageCopy;
  const int maxResolution = MAX_ENV_LIGHT_PDF_SIZE;

  for (int i = 0; i < 4; i++)
  {
    if (width > maxResolution || height > maxResolution)
    {
      imageCopy = resizeToHalfSizef4(pixels, width, height);
      width  = width / 2;
      height = height / 2;
      pixels = &imageCopy[0];
      if (width == 0)  width = 1;
      if (height == 0) height = 1;
    }
  }

  std::vector<float> luminanceData(width*height);

  float avg = 0.0f;
  for (size_t i = 0; i < luminanceData.size(); i++)
  {
    float lum = maxcolorc(to_float3(pixels[i]));
    luminanceData[i] = lum;
    avg += lum;
  }
  avg /= float(luminanceData.size());
  avg = fmax(avg, 1.0f);

  HDRImageLite tempImage(width, height, 1, &luminanceData[0]);
  luminanceData = std::vector<float>();

  tempImage.gaussBlur(2, 1.5f); //#TODO: opt gauss blur !!!
  for (int i = 0; i < tempImage.width()*tempImage.height(); i++) // prevent pixels with zero pdf
    tempImage.data()[i] += 0.05f*avg;

  return tempImage.data();
}

static std::vector<uchar4> resizeToHalfSizeUB4(const uchar4* pixels, int width, int height)
{
  width = width / 2;
  height = height / 2;
  if (width == 0)  width = 1;
  if (height == 0) height = 1;

  std::vector<uchar4> copy(width*height);

  uchar4* oldArray = (uchar4*)(&pixels[0]);
  uchar4* newArray = (uchar4*)(&copy[0]);

  for (int y = 0; y < height; y++)
  {
    int offset1 = (2 * y + 0) * 2 * width;
    int offset2 = (2 * y + 1) * 2 * width;

    for (int x = 0; x < width; x++)
    {
      uchar4 x0 = oldArray[offset1 + 2 * x + 0];
      uchar4 x1 = oldArray[offset1 + 2 * x + 1];
      uchar4 x2 = oldArray[offset2 + 2 * x + 0];
      uchar4 x3 = oldArray[offset2 + 2 * x + 1];

      int4 summ;

      summ.x = x0.x + x1.x + x2.x + x3.x;
      summ.y = x0.y + x1.y + x2.y + x3.y;
      summ.z = x0.z + x1.z + x2.z + x3.z;
      summ.w = x0.w + x1.w + x2.w + x3.w;

      summ.x = summ.x >> 2; // div by 4
      summ.y = summ.y >> 2; // div by 4
      summ.z = summ.z >> 2; // div by 4
      summ.w = summ.w >> 2; // div by 4

      if (summ.x > 255) summ.x = 255;
      if (summ.y > 255) summ.y = 255;
      if (summ.z > 255) summ.z = 255;
      if (summ.w > 255) summ.w = 255;

      newArray[width*y + x] = uchar4(summ.x, summ.y, summ.z, summ.w);
    }
  }

  return copy;
}

std::vector<float> LuminanceFromUchar4Image(const uchar4* pixels, int& width, int& height)
{
  // resize at least once. This is important for low res actually!
  //
  std::vector<uchar4> imageCopy;
 
  const int maxResolution = 256;
	for (int i = 0; i < 4; i++)
	{
		if (width > maxResolution || height > maxResolution)
		{
			imageCopy = resizeToHalfSizeUB4(pixels, width, height);
			width = width / 2;
			height = height / 2;
			pixels = &imageCopy[0];
      if (width == 0)  width = 1;
      if (height == 0) height = 1;
		}
	}
 
  std::vector<float> luminanceData(width*height);

  float avg = 0.0f;

  for (size_t i = 0; i < luminanceData.size(); i++)
  {
    float r = pixels[i].x*(1.0f / 255.0f);
    float g = pixels[i].y*(1.0f / 255.0f);
    float b = pixels[i].z*(1.0f / 255.0f);

    luminanceData[i] = maxcolorc(float3(r, g, b));
    avg += luminanceData[i];
  }

  avg /= float(luminanceData.size());
  avg = fmax(avg, 1.0f);

  for (int i = 0; i < luminanceData.size(); i++) // prevent pixels with zero pdf
    luminanceData[i] += 0.1f*avg;

  return luminanceData;
}

std::vector<float> PrefixSumm(const std::vector<float>& a_vec)
{
  float accum = 0.0f;
  std::vector<float> avgBAccum(a_vec.size() + 1);
  for (size_t i = 0; i < a_vec.size(); i++)
  {
    avgBAccum[i] = accum;
    accum += a_vec[i];
  }
  avgBAccum[avgBAccum.size() - 1] = accum;
  return avgBAccum;
}

std::string ws2s(const std::wstring& s);
std::vector<float> CreateSphericalTextureFromIES(const std::string& a_iesData, int* pW, int* pH);

/**
\brief  Create 2 tables from ies file in single float1 storage and return pair of their ids.
\param  pathW      - path to ies file
\param  a_storage  - storage of floats 
\param  a_iesCache - explicit cache for already processed IES files.
\param  a_libPath  - input path to scene library; used to get full path of ies.
\return pair of (texTableId, pdfTbaleId)

*/

int2 AddIesTexTableToStorage(const std::wstring pathW, IMemoryStorage* a_storage, 
                             std::unordered_map<std::wstring, int2>& a_iesCache, const std::wstring& a_libPath)
{
  int32_t iesTexId = INVALID_TEXTURE;
  int32_t iesPdfId = INVALID_TEXTURE;

  const std::wstring pathW1 = a_libPath + std::wstring(L"/") + pathW;
  const std::string  pathA1 = ws2s(pathW1);

  auto p = a_iesCache.find(pathW);
  if (p == a_iesCache.end())
  {
    int w, h;
    std::vector<float> sphericalTexture;
    
    iesTexId         = a_storage->GetMaxObjectId() + 1;
    sphericalTexture = CreateSphericalTextureFromIES(pathA1, &w, &h);
    
    if(sphericalTexture.size() == 1)
      return int2(-1,-1);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    float maxVal = 0.0f;
    for (auto i = 0; i < sphericalTexture.size(); i++)
      maxVal = fmax(maxVal, sphericalTexture[i]);

    if(maxVal == 0.0f)
    {
      std::cerr << "[ERROR]: broken IES file (maxVal = 0.0): " << pathA1.c_str() << std::endl;
      return int2(-1, -1);
    }

    float invMax = 1.0f / maxVal;
    for (auto i = 0; i < sphericalTexture.size(); i++)
    {
      float val = invMax*sphericalTexture[i];
      sphericalTexture[i] = val;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

    std::vector<float> data2(sphericalTexture.size() + 5);

    data2[0] = as_float(w);
    data2[1] = as_float(h);
    data2[2] = as_float(1);
    data2[3] = as_float(4);
    //data2[3] = as_float(int(sphericalTexture.size() + 1));

    double avgVal = 0.0f;
    for (size_t i = 0; i < sphericalTexture.size(); i++)
    {
      avgVal      += double(sphericalTexture[i]);
      data2[i + 4] = sphericalTexture[i];
    }
    avgVal /= double(sphericalTexture.size());

    a_storage->Update(iesTexId, &data2[0], data2.size() * sizeof(float));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    HDRImageLite tempImage(w, h, 1, sphericalTexture.data());
    tempImage.gaussBlur(2, 1.5f);
    for (int i = 0; i < tempImage.width()*tempImage.height(); i++) // prevent pixels with zero pdf
      sphericalTexture[i] = tempImage.data()[i] + 0.05f*float(avgVal);
    
    data2 = PrefixSumm(sphericalTexture);
    
    std::vector<float> data3(data2.size() + 4);
    
    data3[0] = as_float(w);
    data3[1] = as_float(h);
    data3[2] = as_float(1);
    data3[3] = as_float(4);
    //data3[3] = as_float(int(data2.size()));
    for (size_t i = 0; i < data2.size(); i++)
      data3[i + 4] = data2[i];
    
    iesPdfId = a_storage->GetMaxObjectId() + 1;
    a_storage->Update(iesPdfId, &data3[0], data3.size() * sizeof(float));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    a_iesCache[pathW] = int2(iesTexId, iesPdfId);
  }
  else
  {
    iesTexId = p->second.x;
    iesPdfId = p->second.y;
  }

  return int2(iesTexId, iesPdfId);
}


void RenderDriverRTE::UpdatePdfTablesForLight(int32_t a_lightId)
{
  // (1) list all tex id
  //
  int32_t a_outIds[2] = { -1,-1 };
  int32_t texturesNum = m_lights[a_lightId]->RelatedTextureIds(a_outIds, 2);

  // (2) get texture data
  //
  const int4* texData = (const int4*)m_pTexStorage->GetBegin();

  const std::vector<int32_t> texOffsets = m_pTexStorage->GetTable();


  int32_t pdfTabId[2];

  auto pElem = m_lightHavePdfTable.find(a_lightId);
  if (pElem == m_lightHavePdfTable.end())
  {
    pdfTabId[0] = m_pPdfStorage->GetMaxObjectId() + 1;
    pdfTabId[1] = m_pPdfStorage->GetMaxObjectId() + 2;
    m_lightHavePdfTable.insert(a_lightId);
  }
  else
  {
    pdfTabId[0] = m_lights[a_lightId]->GetPdfTableId(0);
    pdfTabId[1] = m_lights[a_lightId]->GetPdfTableId(1);
  }

  for (int32_t i = 0; i < texturesNum; i++)
  {
    const int   texId = a_outIds[i];

    // (3) calc lum image
    //
    std::vector<float> lumImage;

    int w, h, bpp;
    const void* pData = nullptr;

    if (texId != INVALID_TEXTURE)
    {
      assert(texData != nullptr);

      const int4* pHeader = texData + texOffsets[texId];
      w     = pHeader->x;
      h     = pHeader->y;
      bpp   = pHeader->w;
      pData = (pHeader + 1);

      if (bpp == 4)
        lumImage = LuminanceFromUchar4Image((const uchar4*)pData, w, h);
      else if (bpp == 16)
        lumImage = LuminanceFromFloat4Image((const float4*)pData, w, h);
    }
    else
    {
      w = 2;
      h = 2;
      bpp = 16;
      lumImage.resize(4);
      lumImage[0] = 0.25f;
      lumImage[1] = 0.25f;
      lumImage[2] = 0.25f;
      lumImage[3] = 0.25f;
    }

    // (4) calc pdf table and find correct id of it for current light - m_lights[a_lightId]
    //
    std::vector<float> pdfTable = PrefixSumm(lumImage);
    

    // (5) update pdf table
    //
    std::vector<float> data(pdfTable.size() + 5);

    data[0] = as_float(w);
    data[1] = as_float(h);
    data[2] = as_float(1);
    data[3] = as_float(int(pdfTable.size()+1));

    for (size_t i = 0; i < pdfTable.size(); i++)
      data[i + 4] = pdfTable[i];
    
    data[data.size() - 1] = 1.0f;

    m_pPdfStorage->Update(pdfTabId[i], &data[0], data.size() * sizeof(float));

    // (6) set pdfTabId to light
    //
    m_lights[a_lightId]->SetPdfTableId(i, pdfTabId[i]);
  
  } // for

}

std::vector<float> RenderDriverRTE::CalcLightPickProbTable(std::vector<PlainLight>& a_inOutLights, const bool a_fwd)
{
  std::vector<float> pickProb(a_inOutLights.size());

  std::unordered_map<int, int> groups;
  std::unordered_set<int>      disableSky;

  int lightNumberNoGroups = 0;
  for (size_t i = 0; i < a_inOutLights.size(); i++)
  {
    int groupId = as_int(a_inOutLights[i].data[PLIGHT_GROUP_ID]);

    if (groupId == -1)
      lightNumberNoGroups++;
    else
    {
      auto p = groups.find(groupId);
      if (p == groups.end())
        groups[groupId] = 1;
      else
        p->second++;
    }

    if (lightFlags(&a_inOutLights[i]) & AREA_LIGHT_SKY_PORTAL)
    {
      int skyId  = as_int(a_inOutLights[i].data[AREA_LIGHT_SKY_SOURCE]);
      disableSky.insert(skyId);
    }

  }

  for (auto p = groups.begin(); p != groups.end(); ++p)
    lightNumberNoGroups++;

  const float pickGroupprob = 1.0f / float(lightNumberNoGroups);

  for (size_t i = 0; i < a_inOutLights.size(); i++)
  {
    int groupId = as_int(a_inOutLights[i].data[PLIGHT_GROUP_ID]);

    float pp = 1.0f;

    if (groupId == -1)
      pp = pickGroupprob;
    else
      pp = pickGroupprob / float(groups[groupId]);

    if (lightFlags(&a_inOutLights[i]) & LIGHT_DO_NOT_SAMPLE_ME)
      pp = 0.0f;

    if (lightType(&a_inOutLights[i]) == PLAIN_LIGHT_TYPE_SKY_DOME)
    {
      if(disableSky.find(int(i)) != disableSky.end() || a_fwd)
        pp = 0.0f;
    }

    const float3 color = lightBaseColor(&a_inOutLights[i]);
    if (length(color) < 0.01f)
      pp = 0.0f;

    if(a_inOutLights[i].data[PLIGHT_PROB_MULT] > 0.0f)
      pp *= a_inOutLights[i].data[PLIGHT_PROB_MULT];

    if(a_fwd)
      a_inOutLights[i].data[PLIGHT_PICK_PROB_FWD] = pp; // override light pick probability here! Store it in the light.
    else
      a_inOutLights[i].data[PLIGHT_PICK_PROB_REV] = pp; // override light pick probability here! Store it in the light.
    
    pickProb[i] = pp;
  }

  return pickProb;
}


std::vector<float> CalcTrianglePickProbTable(const PlainMesh* pLMesh, double* a_pOutSurfaceAreaTotal)
{
  const float4* vpos     = meshVerts(pLMesh);
  const int32_t* indices = meshTriIndices(pLMesh);

  std::vector<float> triangleSurfaceArea(pLMesh->tIndicesNum/3);

  (*a_pOutSurfaceAreaTotal) = 0.0;
  for (int i = 0; i < pLMesh->tIndicesNum; i += 3)
  {
    const int iA = indices[i + 0];
    const int iB = indices[i + 1];
    const int iC = indices[i + 2];

    const float3 A = to_float3(vpos[iA]);
    const float3 B = to_float3(vpos[iB]);
    const float3 C = to_float3(vpos[iC]);

    const float triSA = 0.5f*length(cross(B - A, C - A));
    triangleSurfaceArea[i/3] = triSA;
    (*a_pOutSurfaceAreaTotal) += double(triSA);
  }

  return PrefixSumm(triangleSurfaceArea);
}





#include "RenderDriverRTE.h"

#include <iostream>
#include <string>

int RenderDriverRTE::CountMaterialsWithAlphaTest()
{
  int numWithAlpha = 0;

  for (auto p = m_materialUpdated.begin(); p != m_materialUpdated.end(); ++p)
  {
    if(p->second == nullptr)
      numWithAlpha++;
    else
    {
      int texId = as_int(p->second->m_plain.data[OPACITY_TEX_OFFSET]);
      if (texId != INVALID_TEXTURE || p->second->skipShadow)
        numWithAlpha++;
    }
  }

  return numWithAlpha;
}

static inline unsigned int CompressTexCoord16(const float2 a_txCoord)
{
  const float fx = WrapVal(a_txCoord.x);
  const float fy = WrapVal(a_txCoord.y);

  const float tx = clamp(0.5f*fx + 0.5f, 0.0f, 1.0f);
  const float ty = clamp(0.5f*fy + 0.5f, 0.0f, 1.0f);

  const unsigned int txi  = (unsigned int)(tx*65535.0f);
  const unsigned int tyi  = (unsigned int)(ty*65535.0f);

  const unsigned int res  = (tyi << 16) | txi;
  return res;
}

bool RenderDriverRTE::MeshHaveOpacity(const PlainMesh* pHeader) const
{
  const int triNum    = pHeader->tIndicesNum/3;
  const int* mindices = meshMatIndices(pHeader);

  bool meshHaveOpacity = false;
  for (int i = 0; i < triNum; i++)
  {
    const int mid = mindices[i];

    auto p = m_materialUpdated.find(mid);
    if (p != m_materialUpdated.end())
    {
      int texId = as_int(p->second->m_plain.data[OPACITY_TEX_OFFSET]);
      if (texId != INVALID_TEXTURE)
      {
        meshHaveOpacity = true;
        break;
      }
    }
  }

  return meshHaveOpacity;
}

void RenderDriverRTE::CreateAlphaTestTable(ConvertionResult& a_cnvRes, AlphaBuffers& a_outBuffers, bool& a_smoothOpacity)
{
  const int maxSamplers = CountMaterialsWithAlphaTest();
  const int auxSize     = maxSamplers*(sizeof(SWTexSampler) / sizeof(int2)); // put opacity samplers at the end of aux buffer

  if (maxSamplers == 0)
    return;

  cvex::vector<SWTexSampler> samplers; 
  samplers.reserve(maxSamplers);

  std::unordered_map<int, int> samplersOffsets;

  const float4* geomStorage = (const float4*)m_pGeomStorage->GetBegin();
  auto geomTable = m_pGeomStorage->GetTable();

  bool haveAtLeastOneSmoothOpacity = false;

  for (int treeId = 0; treeId < a_cnvRes.treesNum; treeId++)
  {
    std::vector<uint2>& a_otrData = a_outBuffers.buf[treeId];

    const int numPrims = a_cnvRes.trif4Num[treeId];
    a_otrData.resize(numPrims + auxSize); 

    bool haveAtLeastOneOpacityMesh = false;
    const int4* i4data = (const int4*)a_cnvRes.pTriangleData[treeId];

    for (int triOffset = 0; triOffset < a_cnvRes.trif4Num[treeId];)
    {
      //const int4 test = i4data[triOffset];
      const int4 test = LiteMath::load_u( (const int*) (i4data + triOffset + 0) );
      if (test.z == -1 && test.w == -1)    // skip object list header
      {
        a_otrData[triOffset] = uint2(-1,-1);
        triOffset++;
        continue;
      }

      const int4 data1 = test;
      const int4 data2 = LiteMath::load_u( (const int*) (i4data + triOffset + 1) );

      const int primId = data1.w; // i4data[triOffset + 0].w;
      const int geomId = data2.w; // i4data[triOffset + 1].w;
    
      const PlainMesh* mesh = (const PlainMesh*)(geomStorage + geomTable[geomId]);

      const int*    vertIndices  = meshTriIndices(mesh);

      const float4* vertPos      = meshVerts(mesh);
      const float4* vertNorm     = meshNorms(mesh);
      //const float2* vertTexCoord = meshTexCoords(mesh);
      const int*    matIndices   = meshMatIndices(mesh);

      const int mId = matIndices[primId];

      auto p = m_materialUpdated.find(mId); // #TODO: change m_materialUpdated from unordered_map to vector to go faster ?
      if (p != m_materialUpdated.end())
      {
        if (p->second->smoothOpacity)
          haveAtLeastOneSmoothOpacity = true;

        // make sampler .... 
        //
        int texId = as_int(p->second->m_plain.data[OPACITY_TEX_OFFSET]);
        if (texId != INVALID_TEXTURE || p->second->skipShadow)
        {
          haveAtLeastOneOpacityMesh = true;
          SWTexSampler* pSamplerInMaterial = (SWTexSampler*)(p->second->m_plain.data + OPACITY_SAMPLER_OFFSET);

          SWTexSampler dummy = DummySampler();
          if (texId == INVALID_TEXTURE)
            pSamplerInMaterial = &dummy;

          size_t relativeOffset = samplers.size();
          
          auto q = samplersOffsets.find(mId);
          if (q == samplersOffsets.end())
          {
            SWTexSampler copy;
            memcpy(&copy, pSamplerInMaterial, sizeof(SWTexSampler)); // unaligned data access alert! must use memcpy.
            samplersOffsets[mId] = int(relativeOffset);
            samplers.push_back(copy);
          }
          else
            relativeOffset = samplersOffsets[mId];

          const int mult = sizeof(SWTexSampler) / sizeof(int2);
          const int offs = numPrims + int(relativeOffset) * mult;

          a_otrData[triOffset + 0].x = (texId != INVALID_TEXTURE) ? offs : INVALID_TEXTURE;
          a_otrData[triOffset + 1].x = p->second->smoothOpacity   ? 1 : 0;  // (!) <== look here please.
          a_otrData[triOffset + 2].x = p->second->skipShadow      ? 1 : 0;  // (!) <== look here please.

          {
            const int offset = primId * 3;

            const int offs_A = vertIndices[offset + 0];
            const int offs_B = vertIndices[offset + 1];
            const int offs_C = vertIndices[offset + 2];

            const float4 vPosA  = LiteMath::load_u((const float*)(vertPos + offs_A)); // unaligned access to vertPos (!!!)
            const float4 vPosB  = LiteMath::load_u((const float*)(vertPos + offs_B));
            const float4 vPosC  = LiteMath::load_u((const float*)(vertPos + offs_C));

            const float4 vNormA = LiteMath::load_u((const float*)(vertNorm + offs_A)); // unaligned access to vertNorm (!!!)
            const float4 vNormB = LiteMath::load_u((const float*)(vertNorm + offs_B));
            const float4 vNormC = LiteMath::load_u((const float*)(vertNorm + offs_C));

            const float2 A_tex = float2(vPosA.w, vNormA.w); //vertTexCoord[offs_A];
            const float2 B_tex = float2(vPosB.w, vNormB.w); //vertTexCoord[offs_B];
            const float2 C_tex = float2(vPosC.w, vNormC.w); //vertTexCoord[offs_C];

            a_otrData[triOffset + 0].y = CompressTexCoord16(A_tex);
            a_otrData[triOffset + 1].y = CompressTexCoord16(B_tex);
            a_otrData[triOffset + 2].y = CompressTexCoord16(C_tex);
          }
        }
        else
        {
          a_otrData[triOffset + 0] = uint2(INVALID_TEXTURE, -1);
          a_otrData[triOffset + 1] = uint2(INVALID_TEXTURE, -1);
          a_otrData[triOffset + 2] = uint2(INVALID_TEXTURE, -1);
        }
      }
      else
      {
        a_otrData[triOffset+0] = uint2(INVALID_TEXTURE, -1);
        a_otrData[triOffset+1] = uint2(INVALID_TEXTURE, -1);
        a_otrData[triOffset+2] = uint2(INVALID_TEXTURE, -1);
      }

      triOffset+=3;
    }


    // copy samplers to the end of a_otrData buffer ... 
    //
    const int totalSizeInInts = numPrims + int(samplers.size()) * sizeof(SWTexSampler) / sizeof(int2);
    assert(totalSizeInInts <= a_otrData.size());

    if (samplers.size() > 0 && a_otrData.size() > 0)
      memcpy(&a_otrData[0] + numPrims, &samplers[0], samplers.size() * sizeof(SWTexSampler));

    if (haveAtLeastOneOpacityMesh)
    {
      a_cnvRes.pTriangleAlpha[treeId] = &a_otrData[0];    
      a_cnvRes.triAfNum      [treeId] = int(a_otrData.size());
    }
    else
    {
      a_cnvRes.pTriangleAlpha[treeId] = nullptr;
      a_cnvRes.triAfNum      [treeId] = 0;
    }
  }

  a_smoothOpacity = haveAtLeastOneSmoothOpacity;

}


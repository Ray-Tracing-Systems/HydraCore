#define ABSTRACT_MATERIAL_GUARDIAN

#ifndef RTGLOBALS
  #include "globals.h"
#endif

#ifndef RTCMATERIAL
  #ifndef RAND_MLT_CPU
    #define RAND_MLT_CPU
  #endif
  #include "cmaterial.h"
#endif

#ifndef RTCLIGHT
  #include "clight.h"
#endif

#include <vector>

SWTexSampler DummySampler();

namespace RAYTR
{

  class IMaterial
  {
  public:

    IMaterial() : smoothOpacity(false), skipShadow(false)
    { 
      memset(m_plain.data, 0, sizeof(PlainMaterial));

      ((int*)(m_plain.data))[EMISSIVE_TEXID_OFFSET] = INVALID_TEXTURE;
      ((int*)(m_plain.data))[EMISSIVE_TEXMATRIXID_OFFSET] = INVALID_TEXTURE;
      ((int*)(m_plain.data))[EMISSIVE_LIGHTID_OFFSET] = -1;

      ((int*)(m_plain.data))[OPACITY_TEX_OFFSET] = INVALID_TEXTURE;
      ((int*)(m_plain.data))[OPACITY_TEX_MATRIX] = INVALID_TEXTURE;
      ((int*)(m_plain.data))[NORMAL_TEX_OFFSET]  = INVALID_TEXTURE;
      ((int*)(m_plain.data))[NORMAL_TEX_MATRIX]  = INVALID_TEXTURE;

      ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = 0;
      SWTexSampler sampler = DummySampler();
      SetNormalSampler(INVALID_TEXTURE, sampler);
      SetOpacitySampler(INVALID_TEXTURE, sampler);
      SetEmissiveSampler(INVALID_TEXTURE, sampler);
    }

    virtual ~IMaterial() {}

    virtual std::vector<PlainMaterial> ConvertToPlainMaterial() const = 0;

    virtual void SetOpacityTex(int a_texId, int a_texMatrixId) 
    {
      ((int*)(m_plain.data))[OPACITY_TEX_OFFSET] = a_texId;
      ((int*)(m_plain.data))[OPACITY_TEX_MATRIX] = a_texMatrixId;
    }

    virtual void SetNormalTex(int a_texId, int a_texMatrixId)
    {
      ((int*)(m_plain.data))[NORMAL_TEX_OFFSET] = a_texId;
      ((int*)(m_plain.data))[NORMAL_TEX_MATRIX] = a_texMatrixId;
    }

    virtual void PutSamplerAt(int a_texId, SWTexSampler a_sampler, const int a_texSlotName, const int a_slotName, const int a_offset)
    {
      const int samplerOffset = (a_texId == INVALID_TEXTURE) ? INVALID_TEXTURE : (a_offset / 4); // calc offset in "float4/int4"
      ((int*)(m_plain.data))[a_texSlotName] = a_texId;
      ((int*)(m_plain.data))[a_slotName]    = samplerOffset;
      SWTexSampler* pSampler  = (SWTexSampler*)(m_plain.data + a_offset);
      (*pSampler) = a_sampler;
    }

    virtual void SetNormalSampler(int a_texId, SWTexSampler a_sampler)   { this->PutSamplerAt(a_texId, a_sampler, NORMAL_TEX_OFFSET, NORMAL_TEX_MATRIX, NORMAL_SAMPLER_OFFSET); }
    virtual void SetOpacitySampler(int a_texId, SWTexSampler a_sampler)  { this->PutSamplerAt(a_texId, a_sampler, OPACITY_TEX_OFFSET, OPACITY_TEX_MATRIX, OPACITY_SAMPLER_OFFSET); }
    virtual void SetEmissiveSampler(int a_texId, SWTexSampler a_sampler) { this->PutSamplerAt(a_texId, a_sampler, EMISSIVE_TEXID_OFFSET, EMISSIVE_TEXMATRIXID_OFFSET, EMISSIVE_SAMPLER_OFFSET); }

    virtual void AddFlags(int a_flags) { (((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET]) |= a_flags; }
    virtual int  GetFlags() const { return ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET]; }
    virtual void SetFlags(int a_flags) { ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = a_flags; }
    virtual void SetBlendSubMaterial(int a_lvl, std::shared_ptr<IMaterial> a_layer) {}

  //protected:

    PlainMaterial m_plain;

    bool smoothOpacity;
    bool skipShadow;
  };


  class ILight
  {
  public:


    ILight(){}
    virtual ~ILight(){}

    virtual PlainLight ConvertToPlainLight() const  { return m_plain; }
    virtual PlainLight Transform(const float4x4 a_matrix) const = 0;

    virtual int32_t    RelatedTextureIds(int32_t* a_outIds, const int a_maxNumber) const { return 0; }

    virtual int32_t    GetPdfTableId(int32_t a_tableOrder) const { return 0; }
    virtual void       SetPdfTableId(int32_t a_tableOrder, int32_t a_id) {  }

    virtual void PutSamplerAt(int a_texId, SWTexSampler a_sampler, const int a_texSlotName, const int a_slotName, const int a_offset)
    {
      const int samplerOffset = (a_texId == INVALID_TEXTURE) ? INVALID_TEXTURE : (a_offset / 4); // calc offset in "float4/int4"
      ((int*)(m_plain.data))[a_texSlotName] = a_texId;
      ((int*)(m_plain.data))[a_slotName]    = samplerOffset;
      SWTexSampler* pSampler                = (SWTexSampler*)(m_plain.data + a_offset);
      (*pSampler) = a_sampler;
    }

    inline int32_t GetType () const { return as_int(m_plain.data[PLIGHT_TYPE]); }
    inline int32_t GetFlags() const { return as_int(m_plain.data[PLIGHT_FLAGS]); }

  protected:

		void  TransformIESMatrix(const float4x4& a_matrix, PlainLight& copy);

    PlainLight m_plain;

  };

  using BumpParameters = float2;

}


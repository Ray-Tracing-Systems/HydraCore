// � Copyright 2017 Vladimir Frolov, MSU Grapics & Media Lab
//

#include "AbstractMaterial.h"
#include "../../HydraAPI/hydra_api/pugixml.hpp"
#include "../../HydraAPI/hydra_api/HydraXMLHelpers.h"

#include "RenderDriverRTE.h"

#include <unordered_map>
#include <algorithm>

using RAYTR::IMaterial;

using HydraLiteMath::float2;
using HydraLiteMath::float3;
using HydraLiteMath::float4;


SWTexSampler DummySampler()
{
  SWTexSampler dummy;
  dummy.texId = INVALID_TEXTURE;
  dummy.flags = 0;
  dummy.gamma = 2.2f;
  dummy.row0  = float4(1, 0, 0, 0);
  dummy.row1  = float4(0, 1, 0, 0);
  return dummy;
}


class EmissiveMaterial : public IMaterial
{

public:

  EmissiveMaterial() {  }
  EmissiveMaterial(float3 color, int texId, SWTexSampler a_sampler, int a_lightId)
  {
    m_plain.data[EMISSIVE_COLORX_OFFSET] = color.x;
    m_plain.data[EMISSIVE_COLORY_OFFSET] = color.y;
    m_plain.data[EMISSIVE_COLORZ_OFFSET] = color.z;

    this->SetEmissiveSampler(texId, a_sampler);
    ((int*)(m_plain.data))[EMISSIVE_LIGHTID_OFFSET] = a_lightId;
    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]   = PLAIN_MAT_CLASS_EMISSIVE;
  }

  void CopyEmissiveHeaderTo(std::shared_ptr<IMaterial> a_materialToCopy) const
  {
    float* fdata = a_materialToCopy->m_plain.data;
    int* idata   = ((int*)(a_materialToCopy->m_plain.data));

    fdata[EMISSIVE_COLORX_OFFSET]      = m_plain.data[EMISSIVE_COLORX_OFFSET];
    fdata[EMISSIVE_COLORY_OFFSET]      = m_plain.data[EMISSIVE_COLORY_OFFSET];
    fdata[EMISSIVE_COLORZ_OFFSET]      = m_plain.data[EMISSIVE_COLORZ_OFFSET];

    idata[EMISSIVE_TEXID_OFFSET]       = ((int*)(m_plain.data))[EMISSIVE_TEXID_OFFSET];
    idata[EMISSIVE_TEXMATRIXID_OFFSET] = ((int*)(m_plain.data))[EMISSIVE_TEXMATRIXID_OFFSET];
    idata[EMISSIVE_LIGHTID_OFFSET]     = ((int*)(m_plain.data))[EMISSIVE_LIGHTID_OFFSET];

    memcpy(idata + EMISSIVE_SAMPLER_OFFSET, m_plain.data + EMISSIVE_SAMPLER_OFFSET, sizeof(SWTexSampler));
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:

};



class ShadowMatteMaterial : public IMaterial
{

public:

  ShadowMatteMaterial()
  {
    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET] = PLAIN_MAT_CLASS_SHADOW_MATTE;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:

};


class LambertMaterial : public IMaterial
{

public:

  LambertMaterial() {  }
  LambertMaterial(float3 color, int texId, SWTexSampler a_sampler)
  {
    m_plain.data[LAMBERT_COLORX_OFFSET] = color.x;
    m_plain.data[LAMBERT_COLORY_OFFSET] = color.y;
    m_plain.data[LAMBERT_COLORZ_OFFSET] = color.z;

    const int samplerOffset = (texId == INVALID_TEXTURE) ? INVALID_TEXTURE : (LAMBERT_SAMPLER0 / 4); // calc offset in "float4/int4"

    ((int*)(m_plain.data))[LAMBERT_TEXID_OFFSET]       = texId;
    ((int*)(m_plain.data))[LAMBERT_TEXMATRIXID_OFFSET] = samplerOffset; 

    SWTexSampler* pSampler = (SWTexSampler*)(m_plain.data + LAMBERT_SAMPLER0);
    (*pSampler) = a_sampler;

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_LAMBERT;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_HAS_DIFFUSE;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:



};


class OrenNayarMaterial : public IMaterial
{

public:

  OrenNayarMaterial() {  }
  OrenNayarMaterial(float3 color, float a_roughness, int texId, SWTexSampler a_sampler)
  {
    m_plain.data[ORENNAYAR_COLORX_OFFSET] = color.x;
    m_plain.data[ORENNAYAR_COLORY_OFFSET] = color.y;
    m_plain.data[ORENNAYAR_COLORZ_OFFSET] = color.z;

    float sigma  = (a_roughness)*(M_PI/2.0f); //Radians(sig);
    float sigma2 = sigma*sigma;
    float A      = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
    float B      = 0.45f * sigma2 / (sigma2 + 0.09f);

    m_plain.data[ORENNAYAR_ROUGHNESS]     = a_roughness;
    m_plain.data[ORENNAYAR_A]             = A;
    m_plain.data[ORENNAYAR_B]             = B;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// put sampler directly inside material
    const int samplerOffset = (texId == INVALID_TEXTURE) ? INVALID_TEXTURE : (ORENNAYAR_SAMPLER0 / 4); // calc offset in "float4/int4"
    ((int*)(m_plain.data))[ORENNAYAR_TEXID_OFFSET]       = texId;
    ((int*)(m_plain.data))[ORENNAYAR_TEXMATRIXID_OFFSET] = samplerOffset;
    SWTexSampler* pSampler = (SWTexSampler*)(m_plain.data + ORENNAYAR_SAMPLER0);
    (*pSampler) = a_sampler;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// put sampler directly inside material

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_OREN_NAYAR;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_HAS_DIFFUSE;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};

class TranslucentMaterial : public IMaterial
{

public:

  TranslucentMaterial() {  }
  TranslucentMaterial(float3 color, int texId, SWTexSampler a_sampler)
  {
    m_plain.data[TRANS_COLORX_OFFSET] = color.x;
    m_plain.data[TRANS_COLORY_OFFSET] = color.y;
    m_plain.data[TRANS_COLORZ_OFFSET] = color.z;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// put sampler directly inside material
    const int samplerOffset = (texId == INVALID_TEXTURE) ? INVALID_TEXTURE : (TRANS_SAMPLER0_OFFSET / 4); // calc offset in "float4/int4"
    ((int*)(m_plain.data))[TRANS_TEXID_OFFSET]       = texId;
    ((int*)(m_plain.data))[TRANS_TEXMATRIXID_OFFSET] = samplerOffset;
    SWTexSampler* pSampler = (SWTexSampler*)(m_plain.data + TRANS_SAMPLER0_OFFSET);
    (*pSampler) = a_sampler;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// put sampler directly inside material

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_TRANSLUCENT;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_HAS_DIFFUSE;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};


class MirrorMaterial : public IMaterial
{

public:

  MirrorMaterial() { }
  MirrorMaterial(float3 color, int texId, const SWTexSampler a_sampler)
  {
    m_plain.data[MIRROR_COLORX_OFFSET] = color.x;
    m_plain.data[MIRROR_COLORY_OFFSET] = color.y;
    m_plain.data[MIRROR_COLORZ_OFFSET] = color.z;

    ((int*)(m_plain.data))[MIRROR_TEXID_OFFSET] = texId;

    const int texMatrixId = (texId == INVALID_TEXTURE) ? INVALID_TEXTURE : (MIRROR_SAMPLER0_OFFSET / 4); // #TODO: refactor this
    ((int*)(m_plain.data))[MIRROR_TEXMATRIXID_OFFSET] = texMatrixId;                                     // #TODO: refactor this
    SWTexSampler* pSampler1 = (SWTexSampler*)(m_plain.data + MIRROR_SAMPLER0_OFFSET);                    // #TODO: refactor this
    (*pSampler1) = a_sampler;                                                                            // #TODO: refactor this

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_PERFECT_MIRROR;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_CAST_CAUSTICS;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};



class ThinGlassMaterial : public IMaterial
{

public:

  ThinGlassMaterial() { }
  ThinGlassMaterial(float3 color, int texId, SWTexSampler a_samplerColor, float a_cosPower, float a_glosiness, int a_glossTexId, SWTexSampler a_samplerGloss)
  {
    m_plain.data[THINGLASS_COLORX_OFFSET] = color.x;
    m_plain.data[THINGLASS_COLORY_OFFSET] = color.y;
    m_plain.data[THINGLASS_COLORZ_OFFSET] = color.z;

    m_plain.data[THINGLASS_COS_POWER]     = a_cosPower;
    m_plain.data[THINGLASS_GLOSINESS]     = a_glosiness;

    ((int*)(m_plain.data))[THINGLASS_TEXID_OFFSET]           = texId;
    ((int*)(m_plain.data))[THINGLASS_GLOSINESS_TEXID_OFFSET] = a_glossTexId;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here
    const int texMatrixId      = (texId        == INVALID_TEXTURE) ? INVALID_TEXTURE : (THINGLASS_SAMPLER0_OFFSET / 4);
    const int glossTexMatrixId = (a_glossTexId == INVALID_TEXTURE) ? INVALID_TEXTURE : (THINGLASS_SAMPLER1_OFFSET / 4);

    ((int*)(m_plain.data))[THINGLASS_TEXMATRIXID_OFFSET]           = texMatrixId;
    ((int*)(m_plain.data))[THINGLASS_GLOSINESS_TEXMATRIXID_OFFSET] = glossTexMatrixId;

    SWTexSampler* pSampler1 = (SWTexSampler*)(m_plain.data + THINGLASS_SAMPLER0_OFFSET);
    SWTexSampler* pSampler2 = (SWTexSampler*)(m_plain.data + THINGLASS_SAMPLER1_OFFSET);
    (*pSampler1) = a_samplerColor;
    (*pSampler2) = a_samplerGloss;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_THIN_GLASS;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_CAST_CAUSTICS | PLAIN_MATERIAL_HAS_TRANSPARENCY;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};


class SkyPortalMaterial : public IMaterial
{

public:

  SkyPortalMaterial() { }
  SkyPortalMaterial(float3 color, int texId, SWTexSampler a_samplerColor, bool a_visible)
  {
    m_plain.data[THINGLASS_COLORX_OFFSET] = color.x;
    m_plain.data[THINGLASS_COLORY_OFFSET] = color.y;
    m_plain.data[THINGLASS_COLORZ_OFFSET] = color.z;

    m_plain.data[THINGLASS_COS_POWER]     = 1000000.0f;
    m_plain.data[THINGLASS_GLOSINESS]     = 1.0f;

    ((int*)(m_plain.data))[THINGLASS_TEXID_OFFSET]           = INVALID_TEXTURE;
    ((int*)(m_plain.data))[THINGLASS_GLOSINESS_TEXID_OFFSET] = INVALID_TEXTURE;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here
    const int texMatrixId  = (texId == INVALID_TEXTURE) ? INVALID_TEXTURE : (THINGLASS_SAMPLER0_OFFSET / 4);

    ((int*)(m_plain.data))[THINGLASS_TEXMATRIXID_OFFSET]           = texMatrixId;
    ((int*)(m_plain.data))[THINGLASS_GLOSINESS_TEXMATRIXID_OFFSET] = INVALID_TEXTURE;

    SWTexSampler* pSampler1 = (SWTexSampler*)(m_plain.data + THINGLASS_SAMPLER0_OFFSET);
    SWTexSampler* pSampler2 = (SWTexSampler*)(m_plain.data + THINGLASS_SAMPLER1_OFFSET);
    (*pSampler1) = a_samplerColor;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_THIN_GLASS;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_HAS_TRANSPARENCY | PLAIN_MATERIAL_SKIP_SHADOW | PLAIN_MATERIAL_SKIP_SKY_PORTAL;

    if(!a_visible)
     ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] |= PLAIN_MATERIAL_INVIS_LIGHT;

    this->skipShadow    = true;
    this->smoothOpacity = false;
 
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};




class GlassMaterial : public IMaterial
{

public:

  GlassMaterial() {  }
  GlassMaterial(float3 color, int texId, SWTexSampler a_samplerColor, float a_IOR, float3 a_fogColor, float a_fogMult, float a_cosPower, float a_glosiness, int a_glossTexId, SWTexSampler a_samplerGloss)
  {
    m_plain.data[GLASS_COLORX_OFFSET] = color.x;
    m_plain.data[GLASS_COLORY_OFFSET] = color.y;
    m_plain.data[GLASS_COLORZ_OFFSET] = color.z;

    m_plain.data[GLASS_FOG_COLORX_OFFSET] = a_fogColor.x;
    m_plain.data[GLASS_FOG_COLORY_OFFSET] = a_fogColor.y;
    m_plain.data[GLASS_FOG_COLORZ_OFFSET] = a_fogColor.z;
    m_plain.data[GLASS_FOG_MULT_OFFSET]   = a_fogMult;

    m_plain.data[GLASS_IOR_OFFSET] = a_IOR;
    m_plain.data[GLASS_COS_POWER]  = a_cosPower;
    m_plain.data[GLASS_GLOSINESS]  = a_glosiness;

    ((int*)(m_plain.data))[GLASS_TEXID_OFFSET]           = texId;
    ((int*)(m_plain.data))[GLASS_GLOSINESS_TEXID_OFFSET] = a_glossTexId;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here
    const int texMatrixId      = (texId        == INVALID_TEXTURE) ? INVALID_TEXTURE : (GLASS_SAMPLER0_OFFSET / 4);
    const int glossTexMatrixId = (a_glossTexId == INVALID_TEXTURE) ? INVALID_TEXTURE : (GLASS_SAMPLER1_OFFSET / 4);

    ((int*)(m_plain.data))[GLASS_TEXMATRIXID_OFFSET]           = texMatrixId;
    ((int*)(m_plain.data))[GLASS_GLOSINESS_TEXMATRIXID_OFFSET] = glossTexMatrixId;

    SWTexSampler* pSampler1 = (SWTexSampler*)(m_plain.data + GLASS_SAMPLER0_OFFSET);
    SWTexSampler* pSampler2 = (SWTexSampler*)(m_plain.data + GLASS_SAMPLER1_OFFSET);
    (*pSampler1) = a_samplerColor;
    (*pSampler2) = a_samplerGloss;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_GLASS;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_CAST_CAUSTICS | PLAIN_MATERIAL_HAS_TRANSPARENCY;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};



class PhongMaterial : public IMaterial
{

public:

  PhongMaterial() {  }
  PhongMaterial(float3 color, int texId, SWTexSampler a_samplerColor, float cosPower, int glossTexId, SWTexSampler a_samplerGloss, float a_glosiness)
  {
    m_plain.data[PHONG_COLORX_OFFSET] = color.x;
    m_plain.data[PHONG_COLORY_OFFSET] = color.y;
    m_plain.data[PHONG_COLORZ_OFFSET] = color.z;

    m_plain.data[PHONG_COSPOWER_OFFSET]  = cosPower;
    m_plain.data[PHONG_GLOSINESS_OFFSET] = a_glosiness;

    ((int*)(m_plain.data))[PHONG_TEXID_OFFSET]           = texId;
    ((int*)(m_plain.data))[PHONG_GLOSINESS_TEXID_OFFSET] = glossTexId;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here
    const int texMatrixId      = (texId == INVALID_TEXTURE)      ? INVALID_TEXTURE : (PHONG_SAMPLER0_OFFSET / 4);
    const int glossTexMatrixId = (glossTexId == INVALID_TEXTURE) ? INVALID_TEXTURE : (PHONG_SAMPLER1_OFFSET / 4);

    ((int*)(m_plain.data))[PHONG_TEXMATRIXID_OFFSET]           = texMatrixId;
    ((int*)(m_plain.data))[PHONG_GLOSINESS_TEXMATRIXID_OFFSET] = glossTexMatrixId;

    SWTexSampler* pSampler1 = (SWTexSampler*)(m_plain.data + PHONG_SAMPLER0_OFFSET);
    SWTexSampler* pSampler2 = (SWTexSampler*)(m_plain.data + PHONG_SAMPLER1_OFFSET);
    (*pSampler1) = a_samplerColor;
    (*pSampler2) = a_samplerGloss;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////// put samplers right here

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_PHONG_SPECULAR;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_CAST_CAUSTICS;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};


class BlinnTorranceSrappowMaterial : public IMaterial
{

public:

  BlinnTorranceSrappowMaterial() {}
  BlinnTorranceSrappowMaterial(float3 color, int texId, SWTexSampler a_samplerColor, float cosPower, int glossTexId, SWTexSampler a_samplerGloss, float a_glosiness)
  {
    m_plain.data[BLINN_COLORX_OFFSET] = color.x;
    m_plain.data[BLINN_COLORY_OFFSET] = color.y;
    m_plain.data[BLINN_COLORZ_OFFSET] = color.z;

    m_plain.data[BLINN_COSPOWER_OFFSET]  = cosPower;
    m_plain.data[BLINN_GLOSINESS_OFFSET] = a_glosiness; 

    this->PutSamplerAt(texId,      a_samplerColor, BLINN_TEXID_OFFSET,           BLINN_TEXMATRIXID_OFFSET,           BLINN_SAMPLER0_OFFSET);
    this->PutSamplerAt(glossTexId, a_samplerGloss, BLINN_GLOSINESS_TEXID_OFFSET, BLINN_GLOSINESS_TEXMATRIXID_OFFSET, BLINN_SAMPLER1_OFFSET);

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]  = PLAIN_MAT_CLASS_BLINN_SPECULAR;
    ((int*)(m_plain.data))[PLAIN_MAT_FLAGS_OFFSET] = PLAIN_MATERIAL_CAST_CAUSTICS;
  }


  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }

protected:


};



class VolumePerlinMaterial : public IMaterial
{
public:

  VolumePerlinMaterial() { }
  VolumePerlinMaterial(float3 color, int octavesNum, float persistence)
  {

    m_plain.data[VOLUME_PERLIN_COLORX_OFFSET] = color.x;
    m_plain.data[VOLUME_PERLIN_COLORY_OFFSET] = color.y;
    m_plain.data[VOLUME_PERLIN_COLORZ_OFFSET] = color.z;

    ((int*)(m_plain.data))[VOLUME_PERLIN_OCTAVES] = octavesNum;
    m_plain.data[VOLUME_PERLIN_PERSISTENCE] = persistence;


    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET] = PLAIN_MAT_CLASS_VOLUME_PERLIN;

  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1);
    res[0] = m_plain;
    return res;
  }
};

class SSSMaterial : public IMaterial
{
public:

	SSSMaterial() { }
	SSSMaterial(float3 absorption, float phase, float scattering, float transmission, float3 diffuse_color, float density)
	{
		m_plain.data[SSS_ABSORPTIONX_OFFSET] = absorption.x;
		m_plain.data[SSS_ABSORPTIONY_OFFSET] = absorption.y;
		m_plain.data[SSS_ABSORPTIONZ_OFFSET] = absorption.z;

		m_plain.data[SSS_PHASE] = phase;
		m_plain.data[SSS_SCATTERING] = scattering;
		m_plain.data[SSS_TRANSMISSION] = transmission;

		m_plain.data[SSS_DIFFUSEX_OFFSET] = diffuse_color.x;
		m_plain.data[SSS_DIFFUSEY_OFFSET] = diffuse_color.y;
		m_plain.data[SSS_DIFFUSEZ_OFFSET] = diffuse_color.z;

		m_plain.data[SSS_DENSITY] = density;
		

		((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET] = PLAIN_MAT_CLASS_SSS;

	}

	std::vector<PlainMaterial> ConvertToPlainMaterial() const
	{
		std::vector<PlainMaterial> res(1);
		res[0] = m_plain;
		return res;
	}
};

class BlendMaskMaterial : public IMaterial // simple mix of n components
{

public:

  BlendMaskMaterial() { }
  BlendMaskMaterial(std::shared_ptr<IMaterial> a_pMaterial1, std::shared_ptr<IMaterial> a_pMaterial2, float3 alpha_color, 
                    int alpha_texID, SWTexSampler a_samplerMask, bool isFresnel, bool VRayLike, int a_extrusion, float a_fresnelIOR,
                    int faloffOffset = -1, int faloffSize = 0)
  {
    m_pComponent1 = a_pMaterial1;
    m_pComponent2 = a_pMaterial2;

    m_plain.data[BLEND_MASK_COLORX_OFFSET] = alpha_color.x;
    m_plain.data[BLEND_MASK_COLORY_OFFSET] = alpha_color.y;
    m_plain.data[BLEND_MASK_COLORZ_OFFSET] = alpha_color.z;

    if (isFresnel)
      m_plain.data[BLEND_MASK_FRESNEL_IOR] = a_fresnelIOR;
    else
      m_plain.data[BLEND_MASK_FRESNEL_IOR] = 1.0f;

    if (isFresnel)
      ((int*)(m_plain.data))[BLEND_TYPE] = BLEND_FRESNEL;
    else if (faloffOffset >= 0)
      ((int*)(m_plain.data))[BLEND_TYPE] = BLEND_FALOFF;
    else
      ((int*)(m_plain.data))[BLEND_TYPE] = BLEND_SIMPLE;

    this->PutSamplerAt(alpha_texID, a_samplerMask, BLEND_MASK_TEXID_OFFSET, BLEND_MASK_TEXMATRIXID_OFFSET, BLEND_MASK_SAMPLER_OFFSET);

    ((int*)(m_plain.data))[PLAIN_MAT_TYPE_OFFSET]    = PLAIN_MAT_CLASS_BLEND_MASK;

    ((int*)(m_plain.data))[BLEND_MASK_FALOFF_OFFSET] = faloffOffset;
    ((int*)(m_plain.data))[BLEND_MASK_FALOFF_SIZE]   = faloffSize;

    ((int*)(m_plain.data))[BLEND_MASK_FLAGS_OFFSET]  = (isFresnel ? BLEND_MASK_FRESNEL : 0);
    if (VRayLike && !isFresnel)
      ((int*)(m_plain.data))[BLEND_MASK_FLAGS_OFFSET] |= BLEND_MASK_REFLECTION_WEIGHT_IS_ONE;

    ((int*)(m_plain.data))[BLEND_MASK_FLAGS_OFFSET] |= a_extrusion;
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(1); res.reserve(10);

    if (m_pComponent1 == nullptr || m_pComponent2 == nullptr)
      return res;

    std::vector<PlainMaterial> dump1 = m_pComponent1->ConvertToPlainMaterial();
    std::vector<PlainMaterial> dump2 = m_pComponent2->ConvertToPlainMaterial();

    ((int*)(m_plain.data))[BLEND_MASK_MATERIAL1_OFFSET] = int(res.size());
    res.insert(res.end(), dump1.begin(), dump1.end());

    ((int*)(m_plain.data))[BLEND_MASK_MATERIAL2_OFFSET] = int(res.size());
    res.insert(res.end(), dump2.begin(), dump2.end());

    res[0] = m_plain;
    return res;
  }

  IMaterial* getComponent1() { return m_pComponent1.get(); }
  IMaterial* getComponent2() { return m_pComponent2.get(); }

  void SetBlendSubMaterial(int a_lvl, std::shared_ptr<IMaterial> a_layer)
  {
    if (a_lvl == 0) m_pComponent1 = a_layer;
    if (a_lvl == 1) m_pComponent2 = a_layer;
  }

protected:

  std::shared_ptr<IMaterial> m_pComponent1;
  std::shared_ptr<IMaterial> m_pComponent2;

};



class EmissiveFaloff : public IMaterial // simple mix of n components
{

public:

  EmissiveFaloff() { }
  EmissiveFaloff(std::shared_ptr<IMaterial>& a_pEmissionBlend, std::shared_ptr<IMaterial>& a_pBRDF)
  {
    m_pEmissBlend = std::move(a_pEmissionBlend);
    m_pBRDFBlend  = std::move(a_pBRDF);
  }

  std::vector<PlainMaterial> ConvertToPlainMaterial() const
  {
    std::vector<PlainMaterial> res(0); // yes we do need zero initial size here
    res.reserve(10);

    std::vector<PlainMaterial> dump1 = m_pBRDFBlend->ConvertToPlainMaterial();
    std::vector<PlainMaterial> dump2 = m_pEmissBlend->ConvertToPlainMaterial();

    res.insert(res.end(), dump1.begin(), dump1.end());
    int offsEmissiveBlend = int(res.size());
    res.insert(res.end(), dump2.begin(), dump2.end());

    ((int*)(res[0].data))[EMISSIVE_BLEND_OFFSET] = offsEmissiveBlend;
    ((int*)(res[0].data))[PLAIN_MAT_FLAGS_OFFSET] |= PLAIN_MATERIAL_EMISSION_FALOFF;

    res[0].data[EMISSIVE_COLORX_OFFSET] = 1.0f;
    res[0].data[EMISSIVE_COLORY_OFFSET] = 1.0f;
    res[0].data[EMISSIVE_COLORZ_OFFSET] = 1.0f;

    ((int*)(res[0].data))[EMISSIVE_TEXID_OFFSET]       = INVALID_TEXTURE;
    ((int*)(res[0].data))[EMISSIVE_TEXMATRIXID_OFFSET] = INVALID_TEXTURE;
    ((int*)(res[0].data))[EMISSIVE_LIGHTID_OFFSET]     = -1;

    ((int*)(res[0].data))[OPACITY_TEX_OFFSET] = INVALID_TEXTURE;
    ((int*)(res[0].data))[OPACITY_TEX_MATRIX] = 0;
    ((int*)(res[0].data))[NORMAL_TEX_OFFSET]  = INVALID_TEXTURE;
    ((int*)(res[0].data))[NORMAL_TEX_MATRIX]  = 0;

    return res;
  }

protected:

  std::shared_ptr<IMaterial> m_pEmissBlend;
  std::shared_ptr<IMaterial> m_pBRDFBlend;

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const pugi::xml_node SamplerNode(const pugi::xml_node a_node)
{
  const pugi::xml_node nodeInsideColor = a_node.child(L"color").child(L"texture");
  if (nodeInsideColor != nullptr)
    return nodeInsideColor;
  else
    return a_node.child(L"texture");
}

SWTexSampler SamplerFromTexref(const pugi::xml_node a_node, bool allowAlphaToRGB = false)
{
  SWTexSampler res = DummySampler();

  res.texId = a_node.attribute(L"id").as_int();
  res.flags = 0;

  if (a_node.attribute(L"input_gamma") != nullptr)
    res.gamma = a_node.attribute(L"input_gamma").as_float();
  else
    res.gamma = 2.2f;

  float4x4 samplerMatrix;

  if (a_node.attribute(L"matrix") != nullptr)
    HydraXMLHelpers::ReadMatrix4x4(a_node, L"matrix", samplerMatrix.L());
  else
    samplerMatrix.identity();

  res.row0 = samplerMatrix.row[0];
  res.row1 = samplerMatrix.row[1];

  const std::wstring modeU    = a_node.attribute(L"addressing_mode_u").as_string();
  const std::wstring modeV    = a_node.attribute(L"addressing_mode_v").as_string();
  const std::wstring modeS    = a_node.attribute(L"filter").as_string();
  const std::wstring alphaSrc = a_node.attribute(L"input_alpha").as_string(); 

  if (modeU == L"clamp")
  	res.flags |= TEX_CLAMP_U;
  
  if (modeV == L"clamp")
  	res.flags |= TEX_CLAMP_V;

  if(modeS == L"point" || modeS == L"nearest")
    res.flags |= TEX_POINT_SAM;

  if (allowAlphaToRGB && alphaSrc == L"alpha")
    res.flags |= TEX_ALPHASRC_W;

  return res;
}

std::shared_ptr<EmissiveMaterial> EmissiveMaterialFromHydraMtl(const pugi::xml_node a_node)
{
  const pugi::xml_node emission = a_node.child(L"emission");
  const float3 colorE = HydraXMLHelpers::ReadValue3f(emission.child(L"color"));

  float mult = 1.0f;
  if (emission.child(L"multiplier").attribute(L"val") != nullptr)
    mult = emission.child(L"multiplier").attribute(L"val").as_float();

  int32_t lightId   = a_node.attribute(L"light_id").as_int();
  int32_t texId     = INVALID_TEXTURE;

  SWTexSampler sampler = DummySampler();
  if (SamplerNode(emission) != nullptr)
  {
    sampler = SamplerFromTexref(SamplerNode(emission));
    texId   = sampler.texId;
  }

  std::shared_ptr<EmissiveMaterial> pResult = std::make_shared<EmissiveMaterial>(colorE*mult, texId, sampler, lightId);
  
  if(emission.child(L"cast_gi") != nullptr && emission.child(L"cast_gi").attribute(L"val").as_int() == 0)
    pResult->AddFlags(PLAIN_MATERIAL_FORBID_EMISSIVE_GI);

  return pResult;
}

std::shared_ptr<IMaterial> DiffuseMaterialFromHydraMtl(const pugi::xml_node a_node)
{
  pugi::xml_node diffuse = a_node.child(L"diffuse");
  const float3 colorE    = HydraXMLHelpers::ReadValue3f(diffuse.child(L"color"));
  const float roughness  = HydraXMLHelpers::ReadValue1f(diffuse.child(L"roughness"));

  SWTexSampler sampler = DummySampler();
  int32_t texId = INVALID_TEXTURE;

  if (SamplerNode(diffuse) != nullptr)
  {
    sampler = SamplerFromTexref(SamplerNode(diffuse));
    texId   = sampler.texId;
  }

  std::wstring brdfType = diffuse.attribute(L"brdf_type").as_string();
  
  if(brdfType == L"orennayar")
  	return std::make_shared<OrenNayarMaterial>(colorE, roughness, texId, sampler);
  else
   return std::make_shared<LambertMaterial>(colorE, texId, sampler);
}

std::shared_ptr<IMaterial> TranslucentMaterialFromHydraMtl(const pugi::xml_node a_node, int& a_texId, SWTexSampler& a_texSampler)
{
  pugi::xml_node diffuse = a_node.child(L"translucency");
  const float3 colorE    = 0.5f*HydraXMLHelpers::ReadValue3f(diffuse.child(L"color")); //#TODO: fix this coeff ??? !!!!
  const float roughness  = HydraXMLHelpers::ReadValue1f(diffuse.child(L"roughness"));

  int32_t texId = INVALID_TEXTURE;

  SWTexSampler sampler = DummySampler();
  if (SamplerNode(diffuse) != nullptr)
  {
    sampler = SamplerFromTexref(SamplerNode(diffuse));
    texId   = sampler.texId;
  }

  a_texId      = texId;
  a_texSampler = sampler;

  return std::make_shared<TranslucentMaterial>(colorE, texId, sampler);
}

std::shared_ptr<IMaterial> DiffuseAndTranslucentBlendMaterialFromHydraMtl(const pugi::xml_node a_node)
{
	const float3 colorD = HydraXMLHelpers::ReadValue3f(a_node.child(L"diffuse").child(L"color"));
	const float3 colorT = HydraXMLHelpers::ReadValue3f(a_node.child(L"translucency").child(L"color"));

	int32_t ttexId = INVALID_TEXTURE;

  SWTexSampler sampler = DummySampler();

	if (length(colorD) > 1e-5f && length(colorT) > 1e-5f)
	{

    std::shared_ptr<IMaterial> pDiff  = DiffuseMaterialFromHydraMtl(a_node);
    std::shared_ptr<IMaterial> pTrans = TranslucentMaterialFromHydraMtl(a_node, ttexId, sampler);
    
    const bool  fresnelBlend  = false;
    const float fresnelIOR    = 1.5f;
    const int   reflExtrusion = BLEND_MASK_EXTRUSION_STRONG;
    
    SWTexSampler sampler2 = sampler;
    sampler2.texId        = INVALID_TEXTURE;

    // std::shared_ptr<BlendMaskMaterial> pResult = std::make_shared<BlendMaskMaterial>(pTrans, pDiff, make_float3(0.5f, 0.5f, 0.5f), INVALID_TEXTURE, sampler,
    //                                                                                  fresnelBlend, true, reflExtrusion, fresnelIOR);
    

    std::shared_ptr<BlendMaskMaterial> pResult = std::make_shared<BlendMaskMaterial>(pTrans, pDiff, colorT, ttexId, sampler,
                                                                                     fresnelBlend, true, reflExtrusion, fresnelIOR);

    return pResult;
	}
	else if (length(colorT) > 1e-5f)
		return TranslucentMaterialFromHydraMtl(a_node, ttexId, sampler);
	else
	  return DiffuseMaterialFromHydraMtl(a_node);
}

std::shared_ptr<IMaterial> ReflectiveMaterialFromHydraMtl(const pugi::xml_node a_node, int& a_texId, SWTexSampler& a_texSampler)
{
  pugi::xml_node reflect = a_node.child(L"reflectivity");
  pugi::xml_node gloss   = reflect.child(L"glossiness");

  const float3 colorS   = HydraXMLHelpers::ReadValue3f(reflect.child(L"color"));
  const float  glossVal = HydraXMLHelpers::ReadValue1f(gloss);

  int32_t texId = INVALID_TEXTURE;

  SWTexSampler sampler      = DummySampler();
  SWTexSampler samplerGloss = DummySampler();
  if (SamplerNode(reflect) != nullptr)
  {
    sampler = SamplerFromTexref(SamplerNode(reflect));
    texId   = sampler.texId;
  }

	a_texId      = texId;
	a_texSampler = sampler;

  int32_t texIdGloss = INVALID_TEXTURE;

  if (SamplerNode(gloss) != nullptr)
  {
    samplerGloss = SamplerFromTexref(SamplerNode(gloss));
    texIdGloss   = samplerGloss.texId;
  }


	const std::wstring brfdType = reflect.attribute(L"brdf_type").as_string();

  if (texIdGloss == INVALID_TEXTURE && glossVal >= 0.995f)
    return std::make_shared<MirrorMaterial>(colorS, texId, sampler);
  else if (brfdType == L"torranse_sparrow")
    return std::make_shared<BlinnTorranceSrappowMaterial>(colorS, texId, sampler, 0.0f, texIdGloss, samplerGloss, glossVal);
  else
    return std::make_shared<PhongMaterial>               (colorS, texId, sampler, 0.0f, texIdGloss, samplerGloss, glossVal);
}

std::shared_ptr<IMaterial> TransparentMaterialFromHydraMtl(const pugi::xml_node a_node, int& a_texId, SWTexSampler& a_texSampler)
{
	pugi::xml_node transp = a_node.child(L"transparency");
	pugi::xml_node gloss  = transp.child(L"glossiness");

	const float3 colorTF  = HydraXMLHelpers::ReadValue3f(transp.child(L"fog_color"));
	const float  colorMF  = HydraXMLHelpers::ReadValue1f(transp.child(L"fog_multiplier"));

	const float3 colorT   = HydraXMLHelpers::ReadValue3f(transp.child(L"color"));
	const float  glossVal = HydraXMLHelpers::ReadValue1f(transp.child(L"glossiness"));
	const float  iorVal   = HydraXMLHelpers::ReadValue1f(transp.child(L"ior"));

	const bool   thinWall = (transp.child(L"thin_walled").attribute(L"val").as_int() == 1);


	int32_t texId = INVALID_TEXTURE;

  SWTexSampler sampler = DummySampler();
  if (SamplerNode(transp) != nullptr)
  {
    sampler = SamplerFromTexref(SamplerNode(transp));
    texId   = sampler.texId;
  }

	a_texId      = texId;
	a_texSampler = sampler;

	int32_t texIdGloss = INVALID_TEXTURE;

  SWTexSampler samplerGloss = DummySampler();
  if (SamplerNode(gloss) != nullptr)
  {
    samplerGloss = SamplerFromTexref(SamplerNode(gloss));
    texIdGloss   = samplerGloss.texId;
  }

	const float cosPower = 0.0f;

	std::shared_ptr<IMaterial> pGlass = nullptr;

	if (fabs(iorVal) < 1e-4f  || thinWall)
		pGlass = std::make_shared<ThinGlassMaterial>(colorT, texId, sampler, cosPower, glossVal, texIdGloss, samplerGloss);
	else
		pGlass = std::make_shared<GlassMaterial>(colorT, texId, sampler, iorVal, colorTF, colorMF, cosPower, glossVal, texIdGloss, samplerGloss);

	pGlass->AddFlags(PLAIN_MATERIAL_HAS_TRANSPARENCY);

	return pGlass;
}

static const int ReadExtrusionType(pugi::xml_node a_node)
{
  if(a_node == nullptr)
    return BLEND_MASK_EXTRUSION_STRONG;
  
	const std::wstring extrusion = a_node.child(L"extrusion").attribute(L"val").as_string();

	if (extrusion == L"maxcolor")
		return BLEND_MASK_EXTRUSION_STRONG;
	else if (extrusion == L"luminance")
		return BLEND_MASK_EXTRUSION_LUMINANCE;
	else if (extrusion == L"colored")
		return 0;
	else
		return BLEND_MASK_EXTRUSION_STRONG;
}

static const float ReadFresnelIOR(pugi::xml_node a_node)
{
  if(a_node == nullptr)
    return 1.5f;
	else if (a_node.child(L"fresnel_ior") != nullptr)
		return a_node.child(L"fresnel_ior").attribute(L"val").as_float();
	else 
	  return a_node.child(L"fresnel_IOR").attribute(L"val").as_float();
}

static int2 GetNormalMapId(IMaterial* pNode)
{
  int2 normId(INVALID_TEXTURE, INVALID_TEXTURE);
  if (pNode == nullptr)
    return normId;

  normId.x = ((int*)(pNode->m_plain.data))[NORMAL_TEX_OFFSET];
  normId.y = ((int*)(pNode->m_plain.data))[NORMAL_TEX_MATRIX];

  return normId;
}

void PushDownNormalMaps(IMaterial* pNode, int2 a_texId, int a_flags, SWTexSampler a_sampler)
{
  BlendMaskMaterial* pBlend = dynamic_cast<BlendMaskMaterial*>(pNode);

  if (pBlend != nullptr)
  {
    int2 normId1 = GetNormalMapId(pBlend->getComponent1());
    int2 normId2 = GetNormalMapId(pBlend->getComponent2());

    if (pBlend->getComponent1() == nullptr || pBlend->getComponent2() == nullptr)
      return;

    int flags1 = (pBlend->getComponent1()->GetFlags()) & (PLAIN_MATERIAL_INVERT_NMAP_X | PLAIN_MATERIAL_INVERT_NMAP_Y | PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY | PLAIN_MATERIAL_INVERT_HEIGHT);
    int flags2 = (pBlend->getComponent2()->GetFlags()) & (PLAIN_MATERIAL_INVERT_NMAP_X | PLAIN_MATERIAL_INVERT_NMAP_Y | PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY | PLAIN_MATERIAL_INVERT_HEIGHT);

    if (normId1.x != INVALID_TEXTURE)
      PushDownNormalMaps(pBlend->getComponent1(), normId1, flags1, a_sampler);
    else
      PushDownNormalMaps(pBlend->getComponent1(), a_texId, a_flags, a_sampler);

    if (normId2.x != INVALID_TEXTURE)
      PushDownNormalMaps(pBlend->getComponent2(), normId2, flags2, a_sampler);
    else
      PushDownNormalMaps(pBlend->getComponent2(), a_texId, a_flags, a_sampler);
  }
  else
  {
    int2 normId = GetNormalMapId(pNode);

    if (a_texId.x != INVALID_TEXTURE)
    {
      ((int*)(pNode->m_plain.data))[NORMAL_TEX_OFFSET] = a_texId.x;
      ((int*)(pNode->m_plain.data))[NORMAL_TEX_MATRIX] = a_texId.y;

      int flagsWithoutNormal = pNode->GetFlags() & ~(PLAIN_MATERIAL_INVERT_NMAP_X | PLAIN_MATERIAL_INVERT_NMAP_Y | PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY | PLAIN_MATERIAL_INVERT_HEIGHT);
      pNode->SetFlags(flagsWithoutNormal | a_flags);
    }

  }

  pNode->SetNormalSampler(a_texId.x, a_sampler);

}

static int PopUpTransparencyAndCaustics(IMaterial* a_pNode)
{
  constexpr int mask = (PLAIN_MATERIAL_CAST_CAUSTICS | PLAIN_MATERIAL_HAS_TRANSPARENCY);

  BlendMaskMaterial* pBlend = dynamic_cast<BlendMaskMaterial*>(a_pNode);
  if (pBlend == nullptr)
    return (a_pNode->GetFlags() & mask);

  auto pMaterial1 = pBlend->getComponent1();
  auto pMaterial2 = pBlend->getComponent2();

  if (pMaterial1 == nullptr || pMaterial2 == nullptr)
    return 0;

  const int flags1 = (pMaterial1->GetFlags() & mask);
  const int flags2 = (pMaterial2->GetFlags() & mask);

  a_pNode->AddFlags(flags1 | flags2);

  return (flags1 | flags2);
}

static bool NodeIsBTDF(IMaterial* a_pNode)
{
  return (dynamic_cast<TranslucentMaterial*>(a_pNode) != nullptr);
}

static bool HaveAnyNodeWithBTDF(IMaterial* a_pNode)
{
  if (a_pNode == nullptr)
    return false;

  BlendMaskMaterial* pBlend = dynamic_cast<BlendMaskMaterial*>(a_pNode);
  if (pBlend == nullptr)
    return NodeIsBTDF(a_pNode);
  else
  {
    auto pMaterial1 = pBlend->getComponent1();
    auto pMaterial2 = pBlend->getComponent2();

    return HaveAnyNodeWithBTDF(pMaterial1) || HaveAnyNodeWithBTDF(pMaterial2);
  }
}


pugi::xml_attribute SmoothLvlAttr(const pugi::xml_node a_heightNode)
{
  pugi::xml_attribute attr1 = a_heightNode.attribute(L"smooth");
  if (attr1 != nullptr)
    return attr1;
  else
    return a_heightNode.attribute(L"smooth_lvl");
}

RAYTR::BumpParameters BumpAmtAndLvl(const pugi::xml_node a_materialNode);

void RenderDriverRTE::ReadBumpAndOpacity(std::shared_ptr<IMaterial> pResult, pugi::xml_node a_node)
{
  int32_t materialId       = a_node.attribute(L"id").as_int();
  const std::wstring mtype = a_node.attribute(L"type").as_string();

  // (2) add normalmap
  //
  pugi::xml_node displ  = a_node.child(L"displacement");
  pugi::xml_node invert = displ.child(L"normal_map").child(L"invert");

  int32_t texIdNM = INVALID_TEXTURE;
  SWTexSampler samplerNM = DummySampler();

  if (displ != nullptr)
  {
    const std::wstring btype = displ.attribute(L"type").as_string();

    if (btype == L"height_bump")
    {
      pugi::xml_node texNode = a_node.child(L"displacement").child(L"height_map").child(L"texture");

      if (texNode != nullptr)
      {
        samplerNM       = SamplerFromTexref(texNode);
        texIdNM         = samplerNM.texId;
        samplerNM.gamma = 1.0f; // well yep, this is really important thing !!!
      }

      // don't be foolish by this simple code, normalmaps will be generated in RenderDriverRTE::UpdateMaterial afterwards.
      // the auxilarry textures that contain real normalmap data will be place in auxilary texture storage with the same (!!!) texture id
      //
      pResult->SetNormalSampler(texIdNM, samplerNM);
      pResult->AddFlags(PLAIN_MATERIAL_INVERT_HEIGHT);

      auto params = BumpAmtAndLvl(a_node); 

      const PlainMaterial& mat = pResult->m_plain;
      const int32_t auxTexId   = this->GetCachedAuxNormalMatId(materialId, mat, texIdNM, a_node);
      ((int*)mat.data)[NORMAL_TEX_OFFSET] = auxTexId;
    }
    else if (btype == L"normal_bump")
    {
      pugi::xml_node texNode = a_node.child(L"displacement").child(L"normal_map").child(L"texture");

      if (texNode != nullptr)
      {
        samplerNM = SamplerFromTexref(texNode);
        texIdNM   = samplerNM.texId;
        if (texNode.attribute(L"input_gamma") == nullptr)
          samplerNM.gamma = 1.0f;
      }
  
      pResult->SetNormalSampler(texIdNM, samplerNM);
      const int32_t texId = texNode.attribute(L"id").as_int();
      PlainMaterial& mat  = pResult->m_plain;
      
      if(m_procTextures.find(texId) != m_procTextures.end())
      {
        ((int*)mat.data)[NORMAL_TEX_OFFSET] = texId;
      }
      else
      {
        const int32_t auxTexId = this->GetCachedAuxNormalMatId(materialId, mat, texIdNM, a_node);
        ((int*)mat.data)[NORMAL_TEX_OFFSET] = auxTexId;
      }
    }
    else if (btype == L"parallax")
    {
      pugi::xml_node texNode1 = a_node.child(L"displacement").child(L"height_map").child(L"texture");
      pugi::xml_node texNode2 = a_node.child(L"displacement").child(L"normal_map").child(L"texture");

    }

    if (invert == nullptr)
      invert = a_node.child(L"displacement").child(L"height_map").child(L"invert");

    if (invert.attribute(L"x").as_int() == 1)
      pResult->AddFlags(PLAIN_MATERIAL_INVERT_NMAP_X);

    if (invert.attribute(L"y").as_int() == 1)
      pResult->AddFlags(PLAIN_MATERIAL_INVERT_NMAP_Y);

    if (invert.attribute(L"swap_xy").as_int() == 1)
      pResult->AddFlags(PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY);

  }
  else
    pResult->SetNormalTex(INVALID_TEXTURE, INVALID_TEXTURE);

  // (3) add opacity map
  //

  if (a_node.child(L"opacity") != nullptr)
  {
    pugi::xml_node opacityTex = a_node.child(L"opacity").child(L"texture");
    SWTexSampler sampler      = SamplerFromTexref(opacityTex, true);
    int32_t texId             = sampler.texId;
    pResult->SetOpacitySampler(texId, sampler);

    pugi::xml_node opacitiNode = a_node.child(L"opacity");

    pResult->smoothOpacity = (opacitiNode.attribute(L"smooth").as_int() == 1);
    pResult->skipShadow    = (opacitiNode.child(L"skip_shadow").attribute(L"val").as_int() == 1) || (opacitiNode.attribute(L"skip_shadow").as_int() == 1);
  }
  else
    pResult->SetOpacityTex(INVALID_TEXTURE, INVALID_TEXTURE);

  // (4) push down normalmaps for all material nodes that needs them
  //
  if (mtype != L"hydra_blend" && mtype != L"blend")
  {
    int flags = pResult->GetFlags() & (PLAIN_MATERIAL_INVERT_NMAP_X | PLAIN_MATERIAL_INVERT_NMAP_Y | PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY | PLAIN_MATERIAL_INVERT_HEIGHT);
    PushDownNormalMaps(pResult.get(), GetNormalMapId(pResult.get()), flags, samplerNM);
  }
  
}

std::shared_ptr<IMaterial> CreateBlendDefferedProxyFromXmlNode(pugi::xml_node a_node)
{
  const std::wstring blendType = a_node.child(L"blend").attribute(L"type").as_string();

  bool fresnelBlend = false;
  bool extrusive = false;

  if (blendType == L"mask_blend")
  {
  
  }
  else if (blendType == L"fresnel_blend")
  {
    fresnelBlend = true;
  }
  else if (blendType == L"faloff_blend")
  {
  
  }
  
  const float fresnelIOR    = ReadFresnelIOR(a_node.child(L"blend"));
  const int   reflExtrusion = ReadExtrusionType(a_node);

  // TODO: if sigmoid   ...
  // TODO: if extrusive ... 
  
  // read mask texture
  //
  int32_t texId = INVALID_TEXTURE;
  SWTexSampler sampler = DummySampler();
  if (a_node.child(L"blend").child(L"mask").child(L"texture") != nullptr)
  {
    sampler = SamplerFromTexref(a_node.child(L"blend").child(L"mask").child(L"texture"));
    texId   = sampler.texId;
  }

  std::shared_ptr<IMaterial> m1 = nullptr;  // will replace them later
  std::shared_ptr<IMaterial> m2 = nullptr;  // will replace them later

  std::shared_ptr<BlendMaskMaterial> pResult2 = std::make_shared<BlendMaskMaterial>(m1, m2, float3(1,1,1), texId, sampler, 
                                                                                    fresnelBlend, extrusive, reflExtrusion, fresnelIOR);

  return pResult2;
}

std::shared_ptr<IMaterial> CreateFromHydraMaterialXmlNode(pugi::xml_node a_node)
{
  int32_t matId = a_node.attribute(L"id").as_int();

  pugi::xml_node emission = a_node.child(L"emission");
  pugi::xml_node diffuse  = a_node.child(L"diffuse");
  pugi::xml_node reflect  = a_node.child(L"reflectivity");
  pugi::xml_node transpar = a_node.child(L"transparency");
  pugi::xml_node sss      = a_node.child(L"translucency");

  float3 colorE   = HydraXMLHelpers::ReadValue3f(emission.child(L"color"));
  float3 colorD   = HydraXMLHelpers::ReadValue3f(diffuse.child(L"color"));
  float3 colorS   = HydraXMLHelpers::ReadValue3f(reflect.child(L"color"));
  float3 colorT   = HydraXMLHelpers::ReadValue3f(transpar.child(L"color"));
  float3 colorSSS = HydraXMLHelpers::ReadValue3f(sss.child(L"color"));

  if (length(colorD) <= 1e-5f) // if don't have diffuse, check translucency
    colorD = colorSSS;

  const bool haveFresnelRefl = (reflect.child(L"fresnel").attribute(L"val").as_int() == 1);

  int texReflId   = INVALID_TEXTURE;
  int texTranspId = INVALID_TEXTURE;
  SWTexSampler samplReflId   = DummySampler();
  SWTexSampler samplTranspId = DummySampler();

  auto pMaterialD = DiffuseAndTranslucentBlendMaterialFromHydraMtl(a_node);
  auto pMaterialS = ReflectiveMaterialFromHydraMtl(a_node, texReflId, samplReflId);
  auto pMaterialT = TransparentMaterialFromHydraMtl(a_node, texTranspId, samplTranspId);
  auto pMaterialE = EmissiveMaterialFromHydraMtl(a_node);

  std::shared_ptr<IMaterial> pResult = nullptr;

  if (length(colorT) > 1e-5f && length(colorS) > 1e-5f && length(colorD) > 1e-5f)
  {
    const bool  fresnelBlend  = haveFresnelRefl;
    const float fresnelIOR    = ReadFresnelIOR(reflect);
    const int   reflExtrusion = ReadExtrusionType(reflect);

    std::shared_ptr<BlendMaskMaterial> pST = std::make_shared<BlendMaskMaterial>(pMaterialS, pMaterialT, colorS, texReflId, samplReflId, fresnelBlend, true, reflExtrusion, fresnelIOR);
    std::shared_ptr<IMaterial> pMaterialST = pST;

    std::shared_ptr<BlendMaskMaterial> pSTD = std::make_shared<BlendMaskMaterial>(pMaterialST, pMaterialD, colorT, texTranspId, samplTranspId, false, true, reflExtrusion, fresnelIOR);

    pST->AddFlags(PLAIN_MATERIAL_HAS_TRANSPARENCY);
    pST->AddFlags(PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY);
    pSTD->AddFlags(PLAIN_MATERIAL_HAS_TRANSPARENCY);

    pResult = pSTD;
  }
  else if (length(colorT) > 1e-5f && length(colorS) > 1e-5f)
  {
    const bool  fresnelBlend  = haveFresnelRefl;
    const float fresnelIOR    = ReadFresnelIOR(reflect);
    const int   reflExtrusion = ReadExtrusionType(reflect);

    std::shared_ptr<BlendMaskMaterial> pResult2 = std::make_shared<BlendMaskMaterial>(pMaterialS, pMaterialT, colorS, texReflId, samplReflId, fresnelBlend, true, reflExtrusion, fresnelIOR);

    pResult2->AddFlags(PLAIN_MATERIAL_HAS_TRANSPARENCY);
    pResult2->AddFlags(PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY);
    //pResult2->AddFlags(PLAIN_MATERIAL_CAST_CAUSTICS); // ???

    pResult = pResult2;
  }
  else if ((length(colorD) > 1e-5f && length(colorS) > 1e-5f) || (length(colorS) > 1e-5f && haveFresnelRefl))
  {
    const bool  fresnelBlend  = haveFresnelRefl;
    const float fresnelIOR    = ReadFresnelIOR(reflect);
    const int   reflExtrusion = ReadExtrusionType(reflect);

    std::shared_ptr<BlendMaskMaterial> pResult2 = std::make_shared<BlendMaskMaterial>(pMaterialS, pMaterialD, colorS, texReflId, samplReflId, fresnelBlend, true, reflExtrusion, fresnelIOR);

    pResult = pResult2;
  }
  else if (length(colorD) > 1e-5f && length(colorT) > 1e-5f)
  {
    const bool  fresnelBlend  = false;
    const float fresnelIOR    = HydraXMLHelpers::ReadValue1f(transpar.child(L"ior"));
    const int   reflExtrusion = BLEND_MASK_EXTRUSION_STRONG;

    std::shared_ptr<BlendMaskMaterial> pResult2 = std::make_shared<BlendMaskMaterial>(pMaterialT, pMaterialD, colorT, texTranspId, samplTranspId, fresnelBlend, true, reflExtrusion, fresnelIOR);

    pResult2->AddFlags(PLAIN_MATERIAL_HAS_TRANSPARENCY);
    //pResult2->AddFlags(PLAIN_MATERIAL_CAST_CAUSTICS); // ???

    pResult = pResult2;
  }
  else if (length(colorT) > 1e-5f)
  {
    pResult = pMaterialT;
  }
  else if (length(colorS) > 1e-5f)
  {
    pResult = pMaterialS;
  }
  else if (length(colorD) > 1e-5f)
  {
    pResult = pMaterialD;
  }
  else if (length(colorE) > 1e-5f)
  {
    pResult = pMaterialE;
  }
  else
  {
    pResult = pMaterialD;
  }

  return pResult;
}

std::shared_ptr<IMaterial> CreateSkyPortalMaterial(pugi::xml_node a_node)
{
  const bool visible   = (a_node.attribute(L"visible").as_int() == 1);
  const auto colorNode = a_node.child(L"emission").child(L"color");
  const auto multNode  = a_node.child(L"emission").child(L"multiplier");

  float mult   = (multNode == nullptr) ? 1.0f : multNode.attribute(L"val").as_float();
  float3 color = HydraXMLHelpers::ReadFloat3(colorNode.attribute(L"val"));

  return std::make_shared<SkyPortalMaterial>(mult*color , INVALID_TEXTURE, SWTexSampler(), visible);
}


std::shared_ptr<IMaterial> CreateMaterialFromXmlNode(pugi::xml_node a_node, RenderDriverRTE* a_pRTE)
{
	const std::wstring mtype = a_node.attribute(L"type").as_string();
  const int32_t      mid   = a_node.attribute(L"id").as_int();

  pugi::xml_node emission = a_node.child(L"emission");
  pugi::xml_node diffuse  = a_node.child(L"diffuse");
  pugi::xml_node reflect  = a_node.child(L"reflectivity");
  pugi::xml_node transpar = a_node.child(L"transparency");
  pugi::xml_node sss      = a_node.child(L"translucency");

  float3 colorE   = HydraXMLHelpers::ReadValue3f(emission.child(L"color"));
 
	const bool haveFresnelRefl = (reflect.child(L"fresnel").attribute(L"val").as_int() == 1);

  auto pMaterialE = EmissiveMaterialFromHydraMtl(a_node);

  std::shared_ptr<IMaterial> pResult = nullptr;

  if (mtype == L"hydra_blend")
    pResult = CreateBlendDefferedProxyFromXmlNode(a_node);
  else if (mtype == L"shadow_catcher")
  {
    pResult = std::make_shared<ShadowMatteMaterial>();

    pugi::xml_node back = a_node.child(L"back");
    if (back != nullptr)
    {
      a_pRTE->m_shadowMatteBackTexId = back.child(L"texture").attribute(L"id").as_int();
      if (a_pRTE->m_shadowMatteBackTexId == 0)
        a_pRTE->m_shadowMatteBackTexId = INVALID_TEXTURE;

      if (back.child(L"texture").attribute(L"input_gamma") != nullptr)
        a_pRTE->m_shadowMatteBackGamma = back.child(L"texture").attribute(L"input_gamma").as_float();

      int enableRefl = back.attribute(L"reflection").as_int();
      if (enableRefl == 1)
        pResult->AddFlags(PLAIN_MATERIAL_CAMERA_MAPPED_REFL);
    }
  }
  else if(mtype == L"sky_portal_mtl")
    pResult = CreateSkyPortalMaterial(a_node);
  else
    pResult = CreateFromHydraMaterialXmlNode(a_node);

  // now add components that can be inside any material node
  // (1) add emission
  //
  if (mtype != L"shadow_catcher")
  {
    a_pRTE->ReadBumpAndOpacity(pResult, a_node);

    // Read Emission
    // 
    if (length(colorE) > 1e-5f && pResult != pMaterialE && ((pResult->GetFlags() & PLAIN_MATERIAL_SKIP_SKY_PORTAL) == 0) )
    {
      const bool visible = (a_node.attribute(L"visible").as_int() == 1); // spetial flag for invisiable lights
      if (!visible && a_node.attribute(L"light_id") != nullptr)
      {
        pResult = std::make_shared<ThinGlassMaterial>(float3(1, 1, 1), INVALID_TEXTURE, DummySampler(), 1e6f, 1.0f, INVALID_TEXTURE, DummySampler());
        pResult->AddFlags(PLAIN_MATERIAL_INVIS_LIGHT);
        pResult->skipShadow = true;
      }

      pMaterialE->CopyEmissiveHeaderTo(pResult);
    }

    if(HaveAnyNodeWithBTDF(pResult.get()))
      pResult->AddFlags(PLAIN_MATERIAL_HAVE_BTDF);
  }
  else
    a_pRTE->ReadBumpAndOpacity(pResult, a_node);

  PopUpTransparencyAndCaustics(pResult.get());

  return pResult;
}

std::shared_ptr<IMaterial> CreateDiffuseWhiteMaterial()
{
  return  std::make_shared<LambertMaterial>(float3(1, 1, 1), INVALID_TEXTURE, DummySampler());
}


#include "RenderDriverRTE.h"

void RenderDriverRTE::BeginMaterialUpdate()
{
  // m_materialUpdated.clear(); //#NOTE: can't actually clear it because we can change blend parameters in last state, and leafs parameters at some previous state 
  m_blendsToUpdate.clear(); // this is ok
}

bool RenderDriverRTE::MaterialDependsOfMaterial(pugi::xml_node a, pugi::xml_node b)
{
  const std::wstring mtype1 = a.attribute(L"type").as_string();
  if (mtype1 != L"hydra_blend")
    return false;

  const std::wstring mtype2 = b.attribute(L"type").as_string();

  const int32_t id1 = a.attribute(L"id").as_int();
  const int32_t id2 = b.attribute(L"id").as_int();
 
  if (mtype2 != L"hydra_blend")
  {
    const int32_t aid1 = a.attribute(L"node_top").as_int();
    const int32_t aid2 = a.attribute(L"node_bottom").as_int();
    return (aid1 == id2) || (aid2 == id2);
  }
  else
  {
    const int32_t bid1 = b.attribute(L"node_top").as_int();
    const int32_t bid2 = b.attribute(L"node_bottom").as_int();

    pugi::xml_node nodeTop    = std::get<1>(m_blendsToUpdate[bid1]);
    pugi::xml_node nodeBottom = std::get<1>(m_blendsToUpdate[bid2]);

    return MaterialDependsOfMaterial(a, nodeTop) || MaterialDependsOfMaterial(a, nodeBottom);
  }
  
}

void RenderDriverRTE::EndMaterialUpdate()
{
  std::vector<DefferedMaterialDataTuple> blendsSorted;
  for (auto blendDataTuple : m_blendsToUpdate)
    blendsSorted.push_back(blendDataTuple.second);

  using DMDT = DefferedMaterialDataTuple&;
  std::sort(blendsSorted.begin(), blendsSorted.end(), [this](DMDT a, DMDT b) {

    auto nodeA = std::get<1>(a);
    auto nodeB = std::get<1>(b);

    // if (MaterialDependsOfMaterial(nodeA, nodeB))  // #TODO: check this by test with forward ref
    //   return true;                                // #TODO: check this by test with forward ref

    const int32_t id1 = nodeA.attribute(L"id").as_int();
    const int32_t id2 = nodeB.attribute(L"id").as_int();

    return (id1 < id2);
  });

  //
  for (auto blendDataTuple : blendsSorted)
  {
    const int32_t idThis = std::get<1>(blendDataTuple).attribute(L"id").as_int();
    std::cout << "blend update id = " << idThis << std::endl;

    std::shared_ptr<RAYTR::IMaterial> pBlend;
    pugi::xml_node                    xmlNode;

    std::tie(pBlend, xmlNode) = blendDataTuple;
    int32_t matId             = xmlNode.attribute(L"id").as_int();

    int32_t id1 = 0, id2 = 0;

    if (xmlNode.attribute(L"node_top") == nullptr || xmlNode.attribute(L"node_bottom") == nullptr)
    {
      std::wcerr << L"[CRITICAL ERROR]: incorrect node_top/node_bottom for blend id = " << xmlNode.attribute(L"id").as_string() << std::endl;
    }
    else
    {
      id1 = xmlNode.attribute(L"node_top").as_int();
      id2 = xmlNode.attribute(L"node_bottom").as_int();
    }

    const auto mat1 = m_materialUpdated[id1];
    const auto mat2 = m_materialUpdated[id2];

    pBlend->SetBlendSubMaterial(0, mat1);
    pBlend->SetBlendSubMaterial(1, mat2);

    // (4) push down normalmaps for all material nodes if we have any material node
    //   
    const SWTexSampler* pSampler = (const SWTexSampler*)(pBlend->m_plain.data + BLEND_MASK_SAMPLER_OFFSET);
    int flags = pBlend->GetFlags() & (PLAIN_MATERIAL_INVERT_NMAP_X | PLAIN_MATERIAL_INVERT_NMAP_Y | PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY | PLAIN_MATERIAL_INVERT_HEIGHT);

    //PushDownNormalMaps(pBlend.get(), GetNormalMapId(pBlend.get()), flags, (*pSampler));

    PutAbstractMaterialToStorage(matId, pBlend, xmlNode, true);
  }

}


bool RenderDriverRTE::PutAbstractMaterialToStorage(const int32_t a_matId, std::shared_ptr<RAYTR::IMaterial> pMaterial, pugi::xml_node a_materialNode, bool processingBlend)
{
  // (1) get plain materials
  //
  std::vector<PlainMaterial> mdata = pMaterial->ConvertToPlainMaterial();

  const int32_t align = int32_t(m_pMaterialStorage->GetAlignSizeInBytes());
  assert(align >= sizeof(int4));

  // (2) extend mdata with proc tex data;
  //
  if (mdata.size() == 0)
  {
    std::cerr << "RenderDriverRTE::PutAbstractMaterialToStorage: empty material" << std::endl;
    return false;
  }

  if (MaterialHaveAtLeastOneProcTex(&mdata[0]))
  {
    int oldSize = int(mdata.size());
    mdata.push_back(pMaterial->prtexDataTail.offsetTable);
    for (const auto& argd : pMaterial->prtexDataTail.data)
      mdata.push_back(argd);

    int* pTableOffset = (int*)(&mdata[0].data[PROC_TEX_TABLE_OFFSET]);
    (*pTableOffset)   = oldSize * PLAIN_MATERIAL_DATA_SIZE;
  }

  // (3) send plain data to device 
  //
  m_pMaterialStorage->Update(a_matId, &mdata[0], mdata.size() * sizeof(PlainMaterial));

  return true;
}

int32_t RenderDriverRTE::AuxNormalTexPerMaterial(const int32_t matId, const int32_t texId)
{
  const int64_t key = (int64_t(matId) << 32) | int64_t(texId);

  int32_t auxTexId = INVALID_TEXTURE;

  auto found = m_auxTexNormalsPerMat.find(key);
  if (found == m_auxTexNormalsPerMat.end())
  {
    auxTexId = m_auxImageNumber;
    m_auxTexNormalsPerMat[key] = auxTexId;
    m_auxImageNumber++;
  }
  else
  {
    auxTexId = found->second;
  }

  return auxTexId;
}
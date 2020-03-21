// � Copyright 2017 Vladimir Frolov, MSU Grapics & Media Lab
//
#include "AbstractMaterial.h"
#include "IMemoryStorage.h"

#include "hydra_api/pugixml.hpp"
#include "hydra_api/HydraXMLHelpers.h"
#include "hydra_api/aligned_alloc.h"

#include "HDRImageLite.h"

using RAYTR::ILight;

static const bool  ROTATE_IES_90_DEG      = true;
static const float OLD_PHOTOMETRIC_SCALE  = 1.0f; // M_PI;

SWTexSampler DummySampler();

const pugi::xml_node SamplerNode(const pugi::xml_node a_node);
SWTexSampler SamplerFromTexref(const pugi::xml_node a_node, bool aAllowAlphaToRGB = false);

std::string ws2s(const std::wstring& s);

static bool UpHemisphereIsBlack(const std::vector<float>& sphericalTexture, int w, int h)
{
  const int angle180 = (h / 2);

  bool topHemisphereIsBlack = true;

  for (int y = angle180; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      const float val = sphericalTexture[y*w + x];
      if (val > 1e-6f)
      {
        topHemisphereIsBlack = false;
        break;
      }
    }

    if (!topHemisphereIsBlack)
      break;
  }

  return topHemisphereIsBlack;
}

static std::vector<float> ReadArrayFromString(const wchar_t* a_str)
{
  std::vector<float> data; data.reserve(100);
  std::wistringstream fin(a_str);

  while (!fin.eof())
  {
    float val = 0;
    fin >> val;
    data.push_back(val);
  }

  return data;
}


int2 AddIesTexTableToStorage(const std::wstring pathW, IMemoryStorage* a_storage,
                             std::unordered_map<std::wstring, int2>& a_iesCache, const std::wstring& a_libPath);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float4x4 GetSubMatrix3x3(PlainLight& a_lightData, int a_slotOffset)
{
  float4x4 matrix;
  matrix(0, 0) = a_lightData.data[a_slotOffset + 0];
  matrix(0, 1) = a_lightData.data[a_slotOffset + 1];
  matrix(0, 2) = a_lightData.data[a_slotOffset + 2];
  matrix(1, 0) = a_lightData.data[a_slotOffset + 3];
  matrix(1, 1) = a_lightData.data[a_slotOffset + 4];
  matrix(1, 2) = a_lightData.data[a_slotOffset + 5];
  matrix(2, 0) = a_lightData.data[a_slotOffset + 6];
  matrix(2, 1) = a_lightData.data[a_slotOffset + 7];
  matrix(2, 2) = a_lightData.data[a_slotOffset + 8];
  return matrix;
}

void PutSubMatrix3x3(PlainLight& a_lightData, int a_slotOffset, const float4x4& a_matrix)
{
  a_lightData.data[a_slotOffset + 0] = a_matrix(0, 0);
  a_lightData.data[a_slotOffset + 1] = a_matrix(0, 1);
  a_lightData.data[a_slotOffset + 2] = a_matrix(0, 2);
  a_lightData.data[a_slotOffset + 3] = a_matrix(1, 0);
  a_lightData.data[a_slotOffset + 4] = a_matrix(1, 1);
  a_lightData.data[a_slotOffset + 5] = a_matrix(1, 2);
  a_lightData.data[a_slotOffset + 6] = a_matrix(2, 0);
  a_lightData.data[a_slotOffset + 7] = a_matrix(2, 1);
  a_lightData.data[a_slotOffset + 8] = a_matrix(2, 2);
}

void PutSubMatrix3x3Transp(PlainLight& a_lightData, int a_slotOffset, const float4x4& a_matrix)
{
  a_lightData.data[a_slotOffset + 0] = a_matrix(0, 0);
  a_lightData.data[a_slotOffset + 1] = a_matrix(1, 0);
  a_lightData.data[a_slotOffset + 2] = a_matrix(2, 0);
  a_lightData.data[a_slotOffset + 3] = a_matrix(0, 1);
  a_lightData.data[a_slotOffset + 4] = a_matrix(1, 1);
  a_lightData.data[a_slotOffset + 5] = a_matrix(2, 1);
  a_lightData.data[a_slotOffset + 6] = a_matrix(0, 2);
  a_lightData.data[a_slotOffset + 7] = a_matrix(1, 2);
  a_lightData.data[a_slotOffset + 8] = a_matrix(2, 2);
}

void ILight::TransformIESMatrix(const float4x4& a_matrix, PlainLight& copy)
{
	float4x4 mrot = a_matrix;
	mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);

  float4x4 iesMatrix = GetSubMatrix3x3(m_plain, IES_LIGHT_MATRIX_E00);

	iesMatrix = mul(mrot, iesMatrix); // #TODO: remember C^(-1)*A*C rule ??? !!!

  ::PutSubMatrix3x3(copy, IES_LIGHT_MATRIX_E00, iesMatrix);
  ::PutSubMatrix3x3(copy, IES_INV_MATRIX_E00,   inverse4x4(iesMatrix));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class AreaDiffuseLight : public ILight
{
public:

  AreaDiffuseLight(pugi::xml_node a_node, IMemoryStorage* a_storage, std::unordered_map<std::wstring, int2>& a_iesCache, const std::wstring& a_libPath)
  {
    const std::wstring lshape = a_node.attribute(L"shape").as_string();
    const std::wstring distr  = a_node.attribute(L"distribution").as_string();

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const bool isSpot = (distr  == L"spot");
    const bool isDisk = (lshape == L"disk");

    float3   a_pos    = float3(0, 0, 0);
    float3   a_norm   = float3(0, -1, 0);
    float2   a_size   = HydraXMLHelpers::ReadRectLightSize(a_node);

    if (isDisk)
      a_size.x = HydraXMLHelpers::ReadSphereOrDiskLightRadius(a_node);

    float3   a_intensity        = OLD_PHOTOMETRIC_SCALE*HydraXMLHelpers::ReadLightIntensity(a_node); 
    float    a_lightSurfaceArea = 4.0f*a_size.x*a_size.y; 

    if (isDisk)
      a_lightSurfaceArea = M_PI*a_size.x*a_size.x;

    float4x4 iesMatrix;
    bool hasIESSPhere  = false;
    int32_t iesTexId = -1;
    int32_t iesPdfId = -1;

    if (a_node.child(L"ies").attribute(L"matrix") != nullptr)
      HydraXMLHelpers::ReadMatrix4x4(a_node.child(L"ies"), L"matrix", iesMatrix.L());

    const int iesPointArea = a_node.child(L"ies").attribute(L"point_area").as_int();
    
		if (ROTATE_IES_90_DEG)
		{
			float4x4 mrot = LiteMath::rotate4x4Y(DEG_TO_RAD*90.0f);
			iesMatrix     = mul(mrot, iesMatrix);
		}

    if (a_node.child(L"ies").attribute(L"loc") != nullptr && distr == L"ies")
    {
      int2 texids = AddIesTexTableToStorage(a_node.child(L"ies").attribute(L"loc").as_string(), a_storage, 
                                            a_iesCache, a_libPath);

      if(texids.x >= 0 && texids.y >= 0)
      {
        iesTexId     = texids.x;
        iesPdfId     = texids.y;
        hasIESSPhere = true;
      }
    }
    else if (a_node.child(L"ies").attribute(L"loc") != nullptr && distr != L"ies")
    {
      const std::string ldistr2 = ws2s(distr); 
      std::cerr << "[WARNING]: AreaDiffuseLight have 'ies' node, but light distribution is '" << ldistr2.c_str() << "'" << std::endl;
      std::cout << "[WARNING]: AreaDiffuseLight have 'ies' node, but light distribution is '" << ldistr2.c_str() << "'" << std::endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool isSkyPortal     = (a_node.child(L"sky_portal").attribute(L"val").as_int() == 1);     
    int  skyPortalSource = 0;

    if (isSkyPortal) // #TODO: add support for different sky portal sources
    {
      skyPortalSource = a_node.child(L"sky_portal").attribute(L"source_id").as_int();
			a_intensity *= (1.0f / OLD_PHOTOMETRIC_SCALE);
    }

    int a_texId            = INVALID_TEXTURE; //#TODO: a_legacy.emissiveTexId;
    int a_texMatrixId      = INVALID_TEXTURE; //#TODO: a_legacy.emissiveTexMatrixId; put 0 here if have some texture and put texture sampler data in AREA_LIGHT_SAMPLER0
    int a_skyPortalSource  = skyPortalSource;
 
    const float angle1 = HydraXMLHelpers::ReadValue1f(a_node.child(L"falloff_angle"));
    const float angle2 = HydraXMLHelpers::ReadValue1f(a_node.child(L"falloff_angle2"));
  
    const float a_spotCos2 = cosf(0.5f*DEG_TO_RAD*angle1); 
    const float a_spotCos1 = cosf(0.5f*DEG_TO_RAD*angle2); 

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    m_plain.data[PLIGHT_POS_X] = a_pos.x;
    m_plain.data[PLIGHT_POS_Y] = a_pos.y;
    m_plain.data[PLIGHT_POS_Z] = a_pos.z;

    m_plain.data[PLIGHT_NORM_X] = a_norm.x;
    m_plain.data[PLIGHT_NORM_Y] = a_norm.y;
    m_plain.data[PLIGHT_NORM_Z] = a_norm.z;

    m_plain.data[PLIGHT_COLOR_X] = a_intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = a_intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = a_intensity.z;

    ((int*)m_plain.data)[PLIGHT_COLOR_TEX]        = a_texId;
    ((int*)m_plain.data)[PLIGHT_COLOR_TEX_MATRIX] = a_texMatrixId;

    m_plain.data[PLIGHT_SURFACE_AREA]  = a_lightSurfaceArea;

    m_plain.data[AREA_LIGHT_SIZE_X] = a_size.x;
    m_plain.data[AREA_LIGHT_SIZE_Y] = a_size.y;

    float4x4 matrix;
    ::PutSubMatrix3x3Transp(m_plain, AREA_LIGHT_MATRIX_E00, matrix);

    m_plain.data[AREA_LIGHT_SPOT_COS1] = a_spotCos1;
    m_plain.data[AREA_LIGHT_SPOT_COS2] = a_spotCos2;

    ((int*)m_plain.data)[AREA_LIGHT_IS_DISK]    = int(isDisk);
    ((int*)m_plain.data)[AREA_LIGHT_SPOT_DISTR] = int(isSpot);
    ((int*)m_plain.data)[AREA_LIGHT_SKY_SOURCE] = a_skyPortalSource;

    ((int*)m_plain.data)[PLIGHT_TYPE]  = PLAIN_LIGHT_TYPE_AREA;
    ((int*)m_plain.data)[PLIGHT_FLAGS] = 0;

    if (isSkyPortal)
    {
      ((int*)m_plain.data)[PLIGHT_FLAGS] |= AREA_LIGHT_SKY_PORTAL;
      ((int*)m_plain.data)[AREA_LIGHT_SKYPORTAL_BTEX]        = INVALID_TEXTURE; //#TODO: fix a_legacy.emissiveTexBlurredId;
      ((int*)m_plain.data)[AREA_LIGHT_SKYPORTAL_BTEX_MATRIX] = INVALID_TEXTURE; //#TODO: fix a_legacy.emissiveTexBlurredMatrixId;
    }

    if (hasIESSPhere)
    {
      ((int*)m_plain.data)[PLIGHT_FLAGS] |= LIGHT_HAS_IES;
      if (iesPointArea == 1)
        ((int*)m_plain.data)[PLIGHT_FLAGS] |= LIGHT_IES_POINT_AREA;

      ((int*)m_plain.data)[IES_SPHERE_TEX_ID] = iesTexId;
      ((int*)m_plain.data)[IES_SPHERE_PDF_ID] = iesPdfId;          // #TODO: add pdf table here
    }
    else
    {
      ((int*)m_plain.data)[IES_SPHERE_TEX_ID] = INVALID_TEXTURE;
      ((int*)m_plain.data)[IES_SPHERE_PDF_ID] = INVALID_TEXTURE;
    }

    ::PutSubMatrix3x3(m_plain, IES_LIGHT_MATRIX_E00, iesMatrix);
  }

  PlainLight Transform(const float4x4 a_matrix) const
  {
    PlainLight copy = m_plain; // apply matrix to copy
    
    // (1) calc position 
    //
    float3 lpos = float3(copy.data[PLIGHT_POS_X], copy.data[PLIGHT_POS_Y], copy.data[PLIGHT_POS_Z]);

    lpos = mul(a_matrix, lpos);

    copy.data[PLIGHT_POS_X] = lpos.x;
    copy.data[PLIGHT_POS_Y] = lpos.y;
    copy.data[PLIGHT_POS_Z] = lpos.z;

    // (2) calc normal 
    //
    float3 lnorm  = float3(copy.data[PLIGHT_NORM_X], copy.data[PLIGHT_NORM_Y], copy.data[PLIGHT_NORM_Z]);

    float4x4 mrot = a_matrix;
    mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    
    lnorm = mul(mrot, lnorm);

    copy.data[PLIGHT_NORM_X] = lnorm.x;
    copy.data[PLIGHT_NORM_Y] = lnorm.y;
    copy.data[PLIGHT_NORM_Z] = lnorm.z;

    ::PutSubMatrix3x3Transp(copy, AREA_LIGHT_MATRIX_E00, mrot);

		const_cast<AreaDiffuseLight*>(this)->TransformIESMatrix(a_matrix, copy);

    // (4) calc surface area 
    //

    if (((int*)m_plain.data)[AREA_LIGHT_IS_DISK] != 0)
    {
      // (4) calc surface area 
      //
      float4x4 mrot = a_matrix;
      mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);

      const float3 vert  = mul(mrot, normalize(float3(1, 1, 1)));
      const float mult   = length(vert);
      const float radius = m_plain.data[AREA_LIGHT_SIZE_X] * mult;

      //copy.data[AREA_LIGHT_SIZE_X]   = radius;
      copy.data[PLIGHT_SURFACE_AREA] = 3.1415926535f*radius*radius;
    }
    else
    {
      float3 vert[4];
      float2 size = float2(m_plain.data[AREA_LIGHT_SIZE_X], m_plain.data[AREA_LIGHT_SIZE_Y]);

      vert[0] = float3(-size.x, 0, -size.y);
      vert[1] = float3(-size.x, 0, size.y);
      vert[2] = float3(size.x, 0, size.y);
      vert[3] = float3(size.x, 0, -size.y);

      for (int i = 0; i < 4; i++)
        vert[i] = mul(a_matrix, vert[i]);

      const float sizeX = length(vert[1] - vert[0]);
      const float sizeY = length(vert[1] - vert[2]);

      const float surfaceArea = sizeX*sizeY;

      copy.data[PLIGHT_SURFACE_AREA] = surfaceArea;
    }

    return copy;
  }

protected:
  

};

class CylinderLight : public ILight
{
public:

  CylinderLight(float zMin, float zMax, float radius, float phiMax, float3 a_intensity, SWTexSampler samplerColor, int32_t texIdColor)
  {
    m_plain.data[PLIGHT_POS_X]   = 0.0f;
    m_plain.data[PLIGHT_POS_Y]   = 0.0f;
    m_plain.data[PLIGHT_POS_Z]   = 0.0f;

    m_plain.data[PLIGHT_COLOR_X] = a_intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = a_intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = a_intensity.z;

    m_plain.data[CYLINDER_LIGHT_RADIUS] = radius;
    m_plain.data[CYLINDER_LIGHT_ZMIN]   = zMin;
    m_plain.data[CYLINDER_LIGHT_ZMAX]   = zMax;
    m_plain.data[CYLINDER_LIGHT_PHIMAX] = phiMax;

    m_plain.data[PLIGHT_SURFACE_AREA]   = (zMax - zMin) * radius * phiMax;

    this->PutSamplerAt(texIdColor, samplerColor, CYLINDER_TEX_ID, CYLINDER_TEXMATRIX_ID, CYLINDER_TEX_SAMPLER);

    ((int*)m_plain.data)[PLIGHT_TYPE]  = PLAIN_LIGHT_TYPE_CYLINDER;
    ((int*)m_plain.data)[PLIGHT_FLAGS] = 0;

    ((int*)m_plain.data)[CYLINDER_PDF_TABLE_ID] = -1;
  }

  PlainLight Transform(const float4x4 a_matrix) const
  {
    PlainLight copy = m_plain; // apply matrix to copy

    // (1) calc position 
    //
    float3 lpos = mul(a_matrix, float3(m_plain.data[PLIGHT_POS_X], m_plain.data[PLIGHT_POS_Y], m_plain.data[PLIGHT_POS_Z]));

    copy.data[PLIGHT_POS_X] = lpos.x;
    copy.data[PLIGHT_POS_Y] = lpos.y;
    copy.data[PLIGHT_POS_Z] = lpos.z;

    // (4) calc surface area 
    //
    float4x4 mrot = a_matrix;
    mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);

    const float zMin   = m_plain.data[CYLINDER_LIGHT_ZMIN];
    const float zMax   = m_plain.data[CYLINDER_LIGHT_ZMAX];
    const float phiMax = m_plain.data[CYLINDER_LIGHT_PHIMAX];

    const float center  = 0.5f*(zMax + zMin);
    const float3 vert   = mul(mrot, normalize(float3(1, 1, 1)));
    const float mult    = length(vert);
    
    const float newRadius = m_plain.data[CYLINDER_LIGHT_RADIUS] * mult;
    copy.data[PLIGHT_SURFACE_AREA] = (zMax - zMin)*mult*newRadius*phiMax;

    ::PutSubMatrix3x3Transp(copy, CYLINDER_LIGHT_MATRIX_E00, a_matrix);
  
    return copy;
  }

  int32_t RelatedTextureIds(int32_t* a_outIds, const int a_maxNumber) const override 
  {
    if (a_maxNumber < 2)
      return 0;

    const int* idata = (const int*)m_plain.data;
    a_outIds[0] = idata[CYLINDER_TEX_ID];
    return 1; 
  }

  void SetPdfTableId(int32_t a_tableOrder, int32_t a_id) override
  {
    if(a_tableOrder == 0)
      ((int*)m_plain.data)[CYLINDER_PDF_TABLE_ID] = a_id;
  }

  int32_t GetPdfTableId(int32_t a_tableOrder) const override
  { 
    if (a_tableOrder != 0)
      return -1;
    else
      return ((int*)m_plain.data)[CYLINDER_PDF_TABLE_ID];
  }

protected:


};


class SphereLight : public ILight
{
public:

  SphereLight(float3 a_pos, float a_radius, float3 a_intensity)
  {
    m_plain.data[PLIGHT_POS_X]   = a_pos.x;
    m_plain.data[PLIGHT_POS_Y]   = a_pos.y;
    m_plain.data[PLIGHT_POS_Z]   = a_pos.z;

    m_plain.data[PLIGHT_COLOR_X] = a_intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = a_intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = a_intensity.z;

    m_plain.data[SPHERE_LIGHT_RADIUS]  = a_radius;
    m_plain.data[PLIGHT_SURFACE_AREA]  = 4.0f*3.1415926535f*a_radius*a_radius;

    ((int*)m_plain.data)[PLIGHT_TYPE]  = PLAIN_LIGHT_TYPE_SPHERE;
    ((int*)m_plain.data)[PLIGHT_FLAGS] = 0;
  }

  PlainLight Transform(const float4x4 a_matrix) const
  {
    PlainLight copy = m_plain; // apply matrix to copy

    // (1) calc position 
    //
    float3 lpos = mul(a_matrix, float3(m_plain.data[PLIGHT_POS_X], m_plain.data[PLIGHT_POS_Y], m_plain.data[PLIGHT_POS_Z]));

    copy.data[PLIGHT_POS_X] = lpos.x;
    copy.data[PLIGHT_POS_Y] = lpos.y;
    copy.data[PLIGHT_POS_Z] = lpos.z;

    // (4) calc surface area 
    //
    float4x4 mrot = a_matrix;
    mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);

    const float3 vert  = mul(mrot, normalize(float3(1, 1, 1)));
    const float mult   = length(vert);
    const float radius = m_plain.data[SPHERE_LIGHT_RADIUS] * mult;

    copy.data[SPHERE_LIGHT_RADIUS] = radius;
    copy.data[PLIGHT_SURFACE_AREA] = 4.0f*3.1415926535f*radius*radius;

    return copy;
  }

protected:


};



class DirectLight : public ILight
{
public:

  DirectLight(float3 a_pos, float3 a_norm, float3 a_intensity, float a_radius1, float a_radius2, float a_shadowSoftAngleRadius)
  {
    m_plain.data[PLIGHT_POS_X] = a_pos.x;
    m_plain.data[PLIGHT_POS_Y] = a_pos.y;
    m_plain.data[PLIGHT_POS_Z] = a_pos.z;

    m_plain.data[PLIGHT_NORM_X] = a_norm.x;
    m_plain.data[PLIGHT_NORM_Y] = a_norm.y;
    m_plain.data[PLIGHT_NORM_Z] = a_norm.z;

    m_plain.data[PLIGHT_COLOR_X] = a_intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = a_intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = a_intensity.z;

    m_plain.data[DIRECT_LIGHT_RADIUS1] = a_radius1;
    m_plain.data[DIRECT_LIGHT_RADIUS2] = a_radius2;

    const float alpha    = DEG_TO_RAD*a_shadowSoftAngleRadius;
    const float tanAlpha = tan(alpha);
    const float cosAlpha = cos(alpha);

    m_plain.data[DIRECT_LIGHT_SSOFTNESS] = a_shadowSoftAngleRadius/0.25f;
    m_plain.data[DIRECT_LIGHT_ALPHA_TAN] = tanAlpha;
    m_plain.data[DIRECT_LIGHT_ALPHA_COS] = cosAlpha;

    m_plain.data[PLIGHT_SURFACE_AREA]    = M_PI*a_radius2*a_radius2;
    ((int*)m_plain.data)[PLIGHT_TYPE]    = PLAIN_LIGHT_TYPE_DIRECT;
    ((int*)m_plain.data)[PLIGHT_FLAGS]   = 0;
  }

  PlainLight Transform(const float4x4 a_matrix) const
  {
    PlainLight copy = m_plain; // apply matrix to copy
    
    // (1) calc position 
    //
    float3 lpos = float3(copy.data[PLIGHT_POS_X], copy.data[PLIGHT_POS_Y], copy.data[PLIGHT_POS_Z]);

    lpos = mul(a_matrix, lpos);

    copy.data[PLIGHT_POS_X] = lpos.x;
    copy.data[PLIGHT_POS_Y] = lpos.y;
    copy.data[PLIGHT_POS_Z] = lpos.z;

    // (2) calc normal 
    //
    float3 lnorm  = float3(copy.data[PLIGHT_NORM_X], copy.data[PLIGHT_NORM_Y], copy.data[PLIGHT_NORM_Z]);

    float4x4 mrot = a_matrix;
    mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);

    lnorm = mul(mrot, lnorm);

    copy.data[PLIGHT_NORM_X] = lnorm.x;
    copy.data[PLIGHT_NORM_Y] = lnorm.y;
    copy.data[PLIGHT_NORM_Z] = lnorm.z;

    return copy;
  }

protected:

};

class SpotLight : public ILight
{
public:

  SpotLight(float3 a_pos, float3 a_norm, float3 a_intensity, float a_cos1, float a_cos2)
  {
    m_plain.data[PLIGHT_POS_X]  = a_pos.x;
    m_plain.data[PLIGHT_POS_Y]  = a_pos.y;
    m_plain.data[PLIGHT_POS_Z]  = a_pos.z;

    m_plain.data[PLIGHT_NORM_X] = a_norm.x;
    m_plain.data[PLIGHT_NORM_Y] = a_norm.y;
    m_plain.data[PLIGHT_NORM_Z] = a_norm.z;

    m_plain.data[PLIGHT_COLOR_X] = a_intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = a_intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = a_intensity.z;

    m_plain.data[POINT_LIGHT_SPOT_COS1] = a_cos1;
    m_plain.data[POINT_LIGHT_SPOT_COS2] = a_cos2;

    m_plain.data[PLIGHT_SURFACE_AREA] = 1e-10f;
    ((int*)m_plain.data)[PLIGHT_TYPE] = PLAIN_LIGHT_TYPE_POINT_SPOT;
    ((int*)m_plain.data)[PLIGHT_FLAGS] = 0;
  }

  PlainLight Transform(const float4x4 a_matrix) const
  {
    PlainLight copy = m_plain; // apply matrix to copy

    // (1) calc position 
    //
    float3 lpos = float3(copy.data[PLIGHT_POS_X], copy.data[PLIGHT_POS_Y], copy.data[PLIGHT_POS_Z]);

    lpos = mul(a_matrix, lpos);

    copy.data[PLIGHT_POS_X] = lpos.x;
    copy.data[PLIGHT_POS_Y] = lpos.y;
    copy.data[PLIGHT_POS_Z] = lpos.z;

    // (2) calc normal 
    //
    float3 lnorm = float3(copy.data[PLIGHT_NORM_X], copy.data[PLIGHT_NORM_Y], copy.data[PLIGHT_NORM_Z]);

    float4x4 mrot = a_matrix;
    mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);

    lnorm = mul(mrot, lnorm);

    copy.data[PLIGHT_NORM_X] = lnorm.x;
    copy.data[PLIGHT_NORM_Y] = lnorm.y;
    copy.data[PLIGHT_NORM_Z] = lnorm.z;

    return copy;
  }

protected:

};

class PointLight : public ILight
{
public:

  PointLight(pugi::xml_node a_node, IMemoryStorage* a_storage, std::unordered_map<std::wstring, int2>& a_iesCache, const std::wstring& a_libPath)
  {
    float3 pos(0, 0, 0);
    float3 intensity = OLD_PHOTOMETRIC_SCALE*HydraXMLHelpers::ReadLightIntensity(a_node);
    float4x4 iesMatrix;
    
    const std::wstring ldistr = a_node.attribute(L"distribution").as_string();

    if (a_node.child(L"ies").attribute(L"matrix") != nullptr && ldistr == L"ies")
    {
      HydraXMLHelpers::ReadMatrix4x4(a_node.child(L"ies"), L"matrix", iesMatrix.L());
      if (ROTATE_IES_90_DEG)
      {
        float4x4 mrot = LiteMath::rotate4x4Y(DEG_TO_RAD*90.0f);
        iesMatrix = mul(mrot, iesMatrix);
      }
    }
    else if(a_node.child(L"honio").attribute(L"matrix") != nullptr)                        // dont know wtf
      HydraXMLHelpers::ReadMatrix4x4(a_node.child(L"honio"), L"matrix", iesMatrix.L());
    else if(a_node.child(L"ies").attribute(L"matrix") != nullptr && ldistr != L"ies")
    {
      const std::string ldistr2 = ws2s(ldistr); 
      std::cerr << "[WARNING]: PointLight have 'ies' node, but light distribution is '" << ldistr2.c_str() << "'" << std::endl;
      std::cout << "[WARNING]: PointLight have 'ies' node, but light distribution is '" << ldistr2.c_str() << "'" << std::endl;
    }

    m_plain.data[PLIGHT_POS_X]   = pos.x;
    m_plain.data[PLIGHT_POS_Y]   = pos.y;
    m_plain.data[PLIGHT_POS_Z]   = pos.z;

    m_plain.data[PLIGHT_COLOR_X] = intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = intensity.z;

    m_plain.data[PLIGHT_SURFACE_AREA]  = 1e-10f;
    ((int*)m_plain.data)[PLIGHT_TYPE]  = PLAIN_LIGHT_TYPE_POINT_OMNI;
    ((int*)m_plain.data)[PLIGHT_FLAGS] = 0;

    if (a_node.child(L"ies").attribute(L"loc") != nullptr && ldistr == L"ies")
    {
      int2 texids  = AddIesTexTableToStorage(a_node.child(L"ies").attribute(L"loc").as_string(), a_storage,
                                             a_iesCache, a_libPath);
      
      if(texids.x >= 0 && texids.y >= 0)
      {
        ((int*)m_plain.data)[IES_SPHERE_TEX_ID] = texids.x;
        ((int*)m_plain.data)[IES_SPHERE_PDF_ID] = texids.y;
        
        ((int*)m_plain.data)[PLIGHT_FLAGS] |= LIGHT_HAS_IES;
        
        ::PutSubMatrix3x3(m_plain, IES_LIGHT_MATRIX_E00, iesMatrix);
      }
    }
    else
    {
      ((int*)m_plain.data)[IES_SPHERE_TEX_ID] = INVALID_TEXTURE;
      ((int*)m_plain.data)[IES_SPHERE_PDF_ID] = INVALID_TEXTURE;
    }


  }


  PlainLight Transform(const float4x4 a_matrix) const
  {
    PlainLight copy = m_plain; // apply matrix to copy

                               // (1) calc position 
                               //
    float3 lpos = mul(a_matrix, float3(m_plain.data[PLIGHT_POS_X], m_plain.data[PLIGHT_POS_Y], m_plain.data[PLIGHT_POS_Z]));

    copy.data[PLIGHT_POS_X] = lpos.x;
    copy.data[PLIGHT_POS_Y] = lpos.y;
    copy.data[PLIGHT_POS_Z] = lpos.z;

    const_cast<PointLight*>(this)->TransformIESMatrix(a_matrix, copy);
    
		return copy;
  }

protected:

};

std::vector<float> CalcTrianglePickProbTable(const PlainMesh* pLMesh, double* a_pOutSurfaceAreaTotal);

class MeshLight : public ILight
{
public:

  MeshLight(pugi::xml_node a_node, IMemoryStorage* a_storage, std::unordered_map<std::wstring, int2>& a_iesCache, const std::wstring& a_libPath, const PlainMesh* pLMesh)
  {
    float3 pos(0, 0, 0);
    float3 intensity = OLD_PHOTOMETRIC_SCALE*HydraXMLHelpers::ReadLightIntensity(a_node);

    m_plain.data[PLIGHT_POS_X]   = pos.x;
    m_plain.data[PLIGHT_POS_Y]   = pos.y;
    m_plain.data[PLIGHT_POS_Z]   = pos.z;

    m_plain.data[PLIGHT_COLOR_X] = intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = intensity.z;

    const int32_t meshVerId  = a_storage->GetMaxObjectId() + 1;
    const int32_t meshPdfId  = a_storage->GetMaxObjectId() + 2;
 
    double surfaceAreaTotal = 0.0;
    std::vector<float> table = CalcTrianglePickProbTable(pLMesh, &surfaceAreaTotal);

    a_storage->Update(meshVerId, pLMesh, pLMesh->totalBytesNum);
    a_storage->Update(meshPdfId, &table[0], table.size()*sizeof(float));

    ((int*)m_plain.data)[MESH_LIGHT_MESH_OFFSET_ID]  = meshVerId;
    ((int*)m_plain.data)[MESH_LIGHT_TABLE_OFFSET_ID] = meshPdfId;
    ((int*)m_plain.data)[MESH_LIGHT_TRI_NUM]         = pLMesh->tIndicesNum / 3;

    m_plain.data[PLIGHT_SURFACE_AREA]  = float(surfaceAreaTotal);
    ((int*)m_plain.data)[PLIGHT_TYPE]  = PLAIN_LIGHT_TYPE_MESH;
    ((int*)m_plain.data)[PLIGHT_FLAGS] = 0;
    ((int*)m_plain.data)[IES_SPHERE_TEX_ID] = INVALID_TEXTURE;
    ((int*)m_plain.data)[IES_SPHERE_PDF_ID] = INVALID_TEXTURE;

    // save auxilarry light mesh data to emp variable to we can recalculate surface area further
    //
    const float4* vpos     = meshVerts(pLMesh);
    const int32_t* indices = meshTriIndices(pLMesh);

    tempPos.assign(vpos,    vpos + pLMesh->vPosNum);
    tempInd.assign(indices, indices + pLMesh->tIndicesNum);

    float4x4 idenity;
    ::PutSubMatrix3x3Transp(m_plain, MESH_LIGHT_MATRIX_E00, idenity);

    auto colorNode = a_node.child(L"intensity").child(L"color");

    SWTexSampler sampler    = DummySampler();
    int32_t      texIdColor = INVALID_TEXTURE;

    if (SamplerNode(colorNode) != nullptr)
    {
      sampler    = SamplerFromTexref(SamplerNode(colorNode));
      texIdColor = sampler.texId;
    }

    this->PutSamplerAt(texIdColor, sampler, MESH_LIGHT_TEX_ID, MESH_LIGHT_TEXMATRIX_ID, MESH_LIGHT_TEX_SAMPLER);
  }

  int32_t RelatedTextureIds(int32_t* a_outIds, const int a_maxNumber) const override
  {
    if (a_maxNumber < 2)
      return 0;
  
    const int* idata = (const int*)m_plain.data;
    a_outIds[0] = idata[MESH_LIGHT_TEX_ID];
    return 1;
  }

  PlainLight Transform(const float4x4 a_matrix) const
  {
    PlainLight copy = m_plain; // apply matrix to copy

    // (1) calc position 
    //
    float3 lpos = mul(a_matrix, float3(m_plain.data[PLIGHT_POS_X], m_plain.data[PLIGHT_POS_Y], m_plain.data[PLIGHT_POS_Z]));

    copy.data[PLIGHT_POS_X] = lpos.x;
    copy.data[PLIGHT_POS_Y] = lpos.y;
    copy.data[PLIGHT_POS_Z] = lpos.z;
  
    float4x4 mrot = a_matrix;
    mrot.m_col[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);

    ::PutSubMatrix3x3Transp(copy, MESH_LIGHT_MATRIX_E00, a_matrix);

    double totalSA = 0.0;
    for (int i = 0; i < tempInd.size(); i += 3)
    {
      const int iA = tempInd[i + 0];
      const int iB = tempInd[i + 1];
      const int iC = tempInd[i + 2];

      const float3 A = mul(a_matrix, to_float3(tempPos[iA]));
      const float3 B = mul(a_matrix, to_float3(tempPos[iB]));
      const float3 C = mul(a_matrix, to_float3(tempPos[iC]));

      const float triSA = 0.5f*length(cross(B - A, C - A));
      totalSA += double(triSA);
    }

    copy.data[PLIGHT_SURFACE_AREA] = float(totalSA);

		return copy;
  }

  cvex::vector<float4> tempPos;
  std::vector<int>     tempInd;

protected:

};


std::shared_ptr<ILight> CreateDirectLightFromXmlNode(pugi::xml_node a_node)
{
  float3 pos(0,0,0);
  float3 norm(0,-1,0);

  float3 intensity = OLD_PHOTOMETRIC_SCALE*HydraXMLHelpers::ReadLightIntensity(a_node);
  float  radius1   = HydraXMLHelpers::ReadNamedValue1f(a_node.child(L"size"), L"inner_radius");
  float  radius2   = HydraXMLHelpers::ReadNamedValue1f(a_node.child(L"size"), L"outer_radius");
  float  soft      = HydraXMLHelpers::ReadValue1f(a_node.child(L"shadow_softness"));
  float  angle     = HydraXMLHelpers::ReadValue1f(a_node.child(L"angle_radius"));
  
  if (a_node.child(L"angle_radius") == nullptr)
    angle = 0.25f*soft; // our sun is about 0.5f grad of agular size; so angle_radius is 0.25f deg.

  return std::make_shared<DirectLight>(pos, norm, intensity, radius1, radius2, angle);
}


std::shared_ptr<ILight> CreateSphereLightFromXmlNode(pugi::xml_node a_node)
{
  float3 pos(0, 0, 0);
  float3 intensity = OLD_PHOTOMETRIC_SCALE*HydraXMLHelpers::ReadLightIntensity(a_node);
  float  radius    = HydraXMLHelpers::ReadNamedValue1f(a_node.child(L"size"), L"radius");

  return std::make_shared<SphereLight>(pos, radius, intensity);
}

std::shared_ptr<ILight> CreateCylinderLightFromXmlNode(pugi::xml_node a_node)
{
  const float3 intensity = OLD_PHOTOMETRIC_SCALE*HydraXMLHelpers::ReadLightIntensity(a_node);

  const float  radius = HydraXMLHelpers::ReadNamedValue1f(a_node.child(L"size"), L"radius");
  const float  height = HydraXMLHelpers::ReadNamedValue1f(a_node.child(L"size"), L"height");
  const float  angle  = HydraXMLHelpers::ReadNamedValue1f(a_node.child(L"size"), L"angle");

  const float zMin = -0.5f*height;
  const float zMax = +0.5f*height;

  auto colorNode = a_node.child(L"intensity").child(L"color");

  SWTexSampler samplerColor = DummySampler();
  int32_t      texIdColor   = INVALID_TEXTURE;

  if (SamplerNode(colorNode) != nullptr)
  {
    samplerColor = SamplerFromTexref(SamplerNode(colorNode));
    texIdColor   = samplerColor.texId;
  }

  samplerColor.row0 = float4(1, 0, 0, 0);  //#NOTE: we do not support texture matrices for textures on cylindrical lights.
  samplerColor.row1 = float4(0, 1, 0, 0);  //#NOTE: we do not support texture matrices for textures on cylindrical lights.

  return std::make_shared<CylinderLight>(zMin, zMax, radius, DEG_TO_RAD*angle, intensity, samplerColor, texIdColor);
}

std::shared_ptr<ILight> CreatePointSpotLightFromXmlNode(pugi::xml_node a_node)
{
  float3 pos(0, 0, 0);
  float3 norm(0, -1, 0);

  float3 intensity   = OLD_PHOTOMETRIC_SCALE*HydraXMLHelpers::ReadLightIntensity(a_node);

  const float angle2 = HydraXMLHelpers::ReadValue1f(a_node.child(L"falloff_angle"));
  const float angle1 = HydraXMLHelpers::ReadValue1f(a_node.child(L"falloff_angle2"));

  return std::make_shared<SpotLight>(pos, norm, intensity, cos(0.5f*DEG_TO_RAD*angle1), cos(0.5f*DEG_TO_RAD*angle2));
}


class SkyDomeLight : public ILight
{
public:

  SkyDomeLight(pugi::xml_node a_node)
  {
    float3 a_intensity  = HydraXMLHelpers::ReadLightIntensity(a_node);
    float3 a_intensity2 = a_intensity; // a_light.color2*a_light.intensity;

    float3 sunColor    (1, 1, 1);              // overwrite this later
    float3 sunDirection(0, -1, 0);             // overwrite this later
    float  turbidity = a_node.child(L"perez").attribute(L"turbidity").as_float();

		if(a_node.child(L"perez") != nullptr)
			((int*)m_plain.data)[PLIGHT_FLAGS] |= SKY_LIGHT_USE_PEREZ_ENVIRONMENT;

    float  lightBoundingSphereRadius = 100.0f; // owevrwite this later
    
    // environment
    //
    m_plain.data[PLIGHT_COLOR_X] = a_intensity.x;
    m_plain.data[PLIGHT_COLOR_Y] = a_intensity.y;
    m_plain.data[PLIGHT_COLOR_Z] = a_intensity.z;

    float4x4 samplerMat0, samplerMat1;

    // if (!(a_light.flags & RAYTR::Light::LIGHT_SEPARATE_SKY))
    // {
    //   a_light.sphTexIndex[0]    = a_light.sphTexIndex[1];
    //   a_light.sphTexMatrices[0] = a_light.sphTexMatrices[1];
    //   a_intensity2              = a_light.color2*a_light.intensity;
    //   a_intensity               = a_intensity2;
    // }

    SWTexSampler* pSampler0 = (SWTexSampler*)(m_plain.data + SKY_DOME_SAMPLER0);
    SWTexSampler* pSampler1 = (SWTexSampler*)(m_plain.data + SKY_DOME_SAMPLER1);

    float4x4* pInvMatrix0 = (float4x4*)(m_plain.data + SKY_DOME_INV_MATRIX0);
    float4x4* pInvMatrix1 = (float4x4*)(m_plain.data + SKY_DOME_INV_MATRIX0);

    pSampler0->gamma  = 1.0f;
    pSampler0->texId  = INVALID_TEXTURE;
    pSampler0->flags  = 0;
    pSampler0->dummy2 = 0;

    auto texNode = a_node.child(L"intensity").child(L"color").child(L"texture");

    if (texNode != nullptr)
    {
      if(texNode.attribute(L"matrix") != nullptr)
        HydraXMLHelpers::ReadMatrix4x4(texNode, L"matrix", samplerMat0.L());

      if (texNode.attribute(L"input_gamma") != nullptr)
        pSampler0->gamma = texNode.attribute(L"input_gamma").as_float();
      else
        pSampler0->gamma = 1.0f;

      if (texNode.attribute(L"id") != nullptr)
        pSampler0->texId = texNode.attribute(L"id").as_int();
      else
        pSampler0->texId = INVALID_TEXTURE;

      pSampler0->flags  = 0;
      pSampler0->dummy2 = 0;

      ((int*)m_plain.data)[PLIGHT_COLOR_TEX]        = pSampler0->texId; // used by CPU code to construct pdf table
      ((int*)m_plain.data)[PLIGHT_COLOR_TEX_MATRIX] = 0;                // have sampler and texture
    }
    else
    {
      pSampler0->gamma  = 1.0f;
      pSampler0->texId  = INVALID_TEXTURE;
      pSampler0->flags  = 0;
      pSampler0->dummy2 = 0;

      ((int*)m_plain.data)[PLIGHT_COLOR_TEX]        = INVALID_TEXTURE; // no texture
      ((int*)m_plain.data)[PLIGHT_COLOR_TEX_MATRIX] = INVALID_TEXTURE; // don't have sa,pler and texture 
    }

    samplerMat1  = samplerMat0;  // TODO: implement secondary sampler later
    (*pSampler1) = (*pSampler0); // TODO: implement secondary sampler later

    pSampler0->row0 = samplerMat0.get_row(0);
    pSampler0->row1 = samplerMat0.get_row(1);
    pSampler1->row0 = samplerMat1.get_row(0);
    pSampler1->row1 = samplerMat1.get_row(1);

    (*pInvMatrix0) = inverse4x4(samplerMat0);
    (*pInvMatrix1) = inverse4x4(samplerMat1);


    // separated sky light we have such
    //
    m_plain.data[SKY_DOME_COLOR_AUX_X] = a_intensity2.x;
    m_plain.data[SKY_DOME_COLOR_AUX_Y] = a_intensity2.y;
    m_plain.data[SKY_DOME_COLOR_AUX_Z] = a_intensity2.z;

    ((int*)m_plain.data)[SKY_DOME_COLOR_TEX_AUX]        = ((int*)m_plain.data)[PLIGHT_COLOR_TEX];
    ((int*)m_plain.data)[SKY_DOME_COLOR_TEX_MATRIX_AUX] = ((int*)m_plain.data)[PLIGHT_COLOR_TEX_MATRIX];
    ((int*)m_plain.data)[SKY_DOME_AUX_TEX_MATRIX_INV]   = INVALID_TEXTURE; // for sampling sky light with apply texture matrix

    // sun if we have it
    //
    m_plain.data[SKY_DOME_SUN_DIR_X] = sunDirection.x;
    m_plain.data[SKY_DOME_SUN_DIR_Y] = sunDirection.y;
    m_plain.data[SKY_DOME_SUN_DIR_Z] = sunDirection.z;

    m_plain.data[SKY_DOME_TURBIDITY] = turbidity;

    m_plain.data[SKY_SUN_COLOR_X]    = sunColor.x;
    m_plain.data[SKY_SUN_COLOR_Y]    = sunColor.y;
    m_plain.data[SKY_SUN_COLOR_Z]    = sunColor.z;

    if (a_node.child(L"perez").attribute(L"sun_id") != nullptr)
      ((int*)m_plain.data)[SKY_DOME_SUN_DIR_ID] = a_node.child(L"perez").attribute(L"sun_id").as_int();
    else
      ((int*)m_plain.data)[SKY_DOME_SUN_DIR_ID] = -1;

    ((int*)m_plain.data)[PLIGHT_TYPE]  = PLAIN_LIGHT_TYPE_SKY_DOME;

  }

  PlainLight Transform(const float4x4 a_matrix) const
  {
    return m_plain;
  }

  int32_t RelatedTextureIds(int32_t* a_outIds, const int a_maxNumber) const
  { 
    if (a_maxNumber < 2)
      return 0;

    const int* idata = (const int*)m_plain.data;
    a_outIds[0]      = idata[PLIGHT_COLOR_TEX];
    a_outIds[1]      = idata[SKY_DOME_COLOR_TEX_AUX];
    return 2; 
  }

  int32_t GetPdfTableId(int32_t a_tableOrder) const 
  {
    return ((int*)m_plain.data)[SKY_DOME_PDF_TABLE0 + a_tableOrder];
  }

  void SetPdfTableId(int32_t a_tableOrder, int32_t a_id) 
  {
    ((int*)m_plain.data)[SKY_DOME_PDF_TABLE0 + a_tableOrder] = a_id;
  }

protected:

};


std::shared_ptr<RAYTR::ILight> CreateLightFromXmlNode(pugi::xml_node a_node, IMemoryStorage* a_storage, std::unordered_map<std::wstring, int2>& a_iesCache, const std::wstring& a_libPath, const PlainMesh* pLightMeshHeader)
{
  const std::wstring ltype  = a_node.attribute(L"type").as_string();
  const std::wstring lshape = a_node.attribute(L"shape").as_string();
  const std::wstring ldistr = a_node.attribute(L"distribution").as_string();

  if (ltype == L"directional" || ldistr == L"directional")
  {
    return CreateDirectLightFromXmlNode(a_node);
  }
  else if (ltype == L"sky")
  {
    return std::make_shared<SkyDomeLight>(a_node);
  }
  else
  {
    if (lshape == L"rect" || lshape == L"disk")
    {
      return std::make_shared<AreaDiffuseLight>(a_node, a_storage, a_iesCache, a_libPath);
    }
    else if (lshape == L"cylinder")
    {
      return CreateCylinderLightFromXmlNode(a_node);
    }
    else if (lshape == L"sphere")
    {
      return CreateSphereLightFromXmlNode(a_node);
    }
    else if (lshape == L"point")
    {
      if (ldistr == L"spot")
        return CreatePointSpotLightFromXmlNode(a_node);
      else
        return std::make_shared<PointLight>(a_node, a_storage, a_iesCache, a_libPath);
    }
    else if (lshape == L"mesh")
    {
      return std::make_shared<MeshLight>(a_node, a_storage, a_iesCache, a_libPath, pLightMeshHeader);
    }
    else
      return nullptr;
  }

}


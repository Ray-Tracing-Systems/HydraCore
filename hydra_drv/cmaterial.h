#ifndef RTCMATERIAL
#define RTCMATERIAL

#include "cglobals.h"
#include "cfetch.h"
#include "cmatpbrt.h"

static inline float3 materialGetEmission(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[EMISSIVE_COLORX_OFFSET], a_pMat->data[EMISSIVE_COLORY_OFFSET], a_pMat->data[EMISSIVE_COLORZ_OFFSET]); }
static inline  int2  materialGetEmissionTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[EMISSIVE_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[EMISSIVE_TEXMATRIXID_OFFSET]);
  return res;
}

static inline float3 materialLeafEvalEmission(__global const PlainMaterial* a_pMat, const float2 a_texCoord, 
                                              __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 texColor = sample2DExt(materialGetEmissionTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  return materialGetEmission(a_pMat)*texColor;
}

static inline int materialIsLight(__global const PlainMaterial* a_pMat) { return length(materialGetEmission(a_pMat)) > 1e-4f ? 1 : 0; }

static inline bool materialIsLeafBRDF(__global const PlainMaterial* a_pMat)
{
  int type = materialGetType(a_pMat);
  return (type != PLAIN_MAT_CLASS_BLEND_MASK);
}


static inline  int2 materialGetNormalTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[NORMAL_TEX_OFFSET]);
  res.y = as_int(a_pMat->data[NORMAL_TEX_MATRIX]);
  return res;
}

static inline int2 materialGetOpacitytex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[OPACITY_TEX_OFFSET]);
  res.y = as_int(a_pMat->data[OPACITY_TEX_MATRIX]);
  return res;
}

// this is needed for high gloss value when it is near mirrors
//

static inline float sigmoid(float x)
{
  return 1.0f / (1.0f + exp(-1.0f*x));
}

static inline float3 MultiScatterBRDF(const float gloss, const float dotNV, const float dotNL, const float G, const float Pss, const float3 reflColor)
{
  // A simple heuristic compensation brightness (imitation microfaceting multiscattering).

  // lerp() { return u + t * (v - u); }

  //const float t = tanh(pow(gloss, 1.0f) * 3.0f);           // mask for blend main brightness correction.
  //const float u = 1.0f / G * 4.0f * dotNV * dotNL;         // inverse G factor.
  //const float lerp1 = u + t * (1.0f - u);                      // delete G factor for low glossiness.
  //const float t2 = pow(sin(pow(t, 1.8f) * M_PI), 0.7f);     // mask for about 0.7 glossiness.
  //const float v = 0.9f / (0.63f + dotNV * dotNV * 0.37f);  // falloff color.
  //const float lerp2 = 1.0f + t2 * (v - 1.0f);                  // delete falloff with mask.

  // White reflColor - full multi-scattering reflection in microfaceting in any glossiness.
  // Black reflColor is a single scattering, it is the basic formula of microfacet, without this compensation function.
  // There is also a slight saturation of color as the roughness increases, due to the multiplication of color in the re-reflections
  // of the micro-facets.
  //const float3 white = make_float3(1.0f, 1.0f, 1.0f);
  //const float3 lerp3 = white + reflColor * (lerp1 * lerp2 - white);
  //return lerp3;

  // From paper "Practical multiple scattering compensation for microfacet models". Emmanuel Turquin. Industrial Light & Magic.
  const float EssWo = Pss;// *dotNL;
  const float EssWoComp = EssWo / (1.0f + EssWo); 
  const float F0        = 1.0f;
  const float3 white    = make_float3(1.0f, 1.0f, 1.0f);

  const float3 result = 1.0f + F0 * (1.0f - EssWoComp) / EssWoComp * INV_TWOPI * 0.5f * white;
  
  const float invEss  = 1.0f / Pss * INV_PI;
  const float t       = EssWoComp;
  //const float3 result = invEss + t * (1.0f - invEss) * white; // lerp
  
  return white;

  //return result;
}

//////////////////////////////////////////////////////////////// all other components may overlay their offsets

// lambert material
//
#define LAMBERT_COLORX_OFFSET       10
#define LAMBERT_COLORY_OFFSET       11
#define LAMBERT_COLORZ_OFFSET       12

#define LAMBERT_TEXID_OFFSET        13
#define LAMBERT_TEXMATRIXID_OFFSET  14

#define LAMBERT_SAMPLER0            20 // float4 sampler header + float2x4
#define LAMBERT_SMATRIX0            24 // float2x4

static inline float3 lambertGetDiffuseColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[LAMBERT_COLORX_OFFSET], a_pMat->data[LAMBERT_COLORY_OFFSET], a_pMat->data[LAMBERT_COLORZ_OFFSET]); }
static inline  int2   lambertGetDiffuseTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[LAMBERT_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[LAMBERT_TEXMATRIXID_OFFSET]);
  return res;
}


static inline float  lambertEvalPDF (__global const PlainMaterial* a_pMat, const float3 l, const float3 n) { return fabs(dot(l, n))*INV_PI; }

static inline float3 lambertEvalBxDF(__global const PlainMaterial* a_pMat, const float2 a_texCoord, const float2 a_texCoordProj,
                                     __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const int samplerOffset    = lambertGetDiffuseTex(a_pMat).y;
  const SWTexSampler sampler = ReadSampler((__global const int4*)a_pMat, samplerOffset);                // TODO: read sampler flags only
  const float2 texCoord2     = (sampler.flags & TEX_COORD_CAM_PROJ) != 0 ? a_texCoordProj : a_texCoord; // read cam projected texCoord for "ShadowCatcher2"

  const float3 texColor = sample2DExt(samplerOffset, texCoord2, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  return clamp(texColor*lambertGetDiffuseColor(a_pMat), 0.0f, 1.0f)*INV_PI;
}

static inline void LambertSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, const float3 a_normal, const float2 a_texCoord, const float2 a_texCoordProj,
                                            __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                            __private MatSample* a_out)
{ 
  const int samplerOffset    = lambertGetDiffuseTex(a_pMat).y;
  const SWTexSampler sampler = ReadSampler((__global const int4*)a_pMat, samplerOffset);                // TODO: read sampler flags only
  const float2 texCoord2     = (sampler.flags & TEX_COORD_CAM_PROJ) != 0 ? a_texCoordProj : a_texCoord; // read cam projected texCoord for "ShadowCatcher2"

  const float3 texColor   = sample2DExt(samplerOffset, texCoord2, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 kd         = clamp(texColor*lambertGetDiffuseColor(a_pMat), 0.0f, 1.0f);

  const float3 newDir     = MapSampleToCosineDistribution(a_r1, a_r2, a_normal, a_normal, 1.0f);
  const float  cosTheta   = dot(newDir, a_normal);

  a_out->direction = newDir;
  a_out->pdf       = cosTheta*INV_PI;
  a_out->color     = kd*INV_PI;
  if (cosTheta <= DEPSILON)
    a_out->color = make_float3(0, 0, 0);

  a_out->flags = RAY_EVENT_D;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// lambert material
//
#define ORENNAYAR_COLORX_OFFSET       10
#define ORENNAYAR_COLORY_OFFSET       11
#define ORENNAYAR_COLORZ_OFFSET       12

#define ORENNAYAR_TEXID_OFFSET        13
#define ORENNAYAR_TEXMATRIXID_OFFSET  14

#define ORENNAYAR_ROUGHNESS           15
#define ORENNAYAR_A                   16
#define ORENNAYAR_B                   17

#define ORENNAYAR_SAMPLER0            20
#define ORENNAYAR_SMATRIX0            24

static inline float3 orennayarGetDiffuseColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[ORENNAYAR_COLORX_OFFSET], a_pMat->data[ORENNAYAR_COLORY_OFFSET], a_pMat->data[ORENNAYAR_COLORZ_OFFSET]); }
static inline int2   orennayarGetDiffuseTex  (__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[ORENNAYAR_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[ORENNAYAR_TEXMATRIXID_OFFSET]);
  return res;
}


static inline float orennayarFunc(const float3 l, const float3 v, const float3 n, const float A, const float B)
{
  float cosTheta_wi = dot(l, n);
  float cosTheta_wo = dot(v, n);

  float sinTheta_wi = sqrt(fmax(0.0f, 1.0f - cosTheta_wi*cosTheta_wi));
  float sinTheta_wo = sqrt(fmax(0.0f, 1.0f - cosTheta_wo*cosTheta_wo));

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = -ray_dir
  // wi = l = newDir
  //
  float3 nx, ny, nz = n;
  CoordinateSystem(nz, &nx, &ny);

  float3 wo = make_float3(-dot(v, nx), -dot(v, ny), -dot(v, nz));
  float3 wi = make_float3(-dot(l, nx), -dot(l, ny), -dot(l, nz));
  //
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  // Compute cosine term of Oren-Nayar model
  float maxcos = 0.f;

  if (sinTheta_wi > 1e-4 && sinTheta_wo > 1e-4)
  {
    float sinphii = SinPhiPBRT1(wi, sinTheta_wi), cosphii = CosPhiPBRT1(wi, sinTheta_wi);
    float sinphio = SinPhiPBRT1(wo, sinTheta_wo), cosphio = CosPhiPBRT1(wo, sinTheta_wo);
    float dcos    = cosphii * cosphio + sinphii * sinphio;
    maxcos = fmax(0.f, dcos);
  }

  // Compute sine and tangent terms of Oren-Nayar model
  float sinalpha = 0.0f, tanbeta = 0.0f;

  if (fabs(cosTheta_wi) > fabs(cosTheta_wo))
  {
    sinalpha = sinTheta_wo;
    tanbeta  = sinTheta_wi / fmax(fabs(cosTheta_wi), DEPSILON);
  }
  else
  {
    sinalpha = sinTheta_wi;
    tanbeta  = sinTheta_wo / fmax(fabs(cosTheta_wo), DEPSILON);
  }

  return (A + B * maxcos * sinalpha * tanbeta);
}


static inline float3  orennayarEvalBxDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float2 a_texCoord,
                                        __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 texColor = sample2DExt(orennayarGetDiffuseTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 kd       = clamp(texColor*orennayarGetDiffuseColor(a_pMat), 0.0f, 1.0f);

  return kd*INV_PI*orennayarFunc(l, v, n, a_pMat->data[ORENNAYAR_A], a_pMat->data[ORENNAYAR_B]);
}

static inline float  orennayarEvalPDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n)
{ 
  return fabs(dot(l, n))*INV_PI; 
}

static inline void OrennayarSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, const float3 ray_dir, const float3 a_normal, const float2 a_texCoord,
                                              __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                              __private MatSample* a_out)
{
  const float3 texColor   = sample2DExt(orennayarGetDiffuseTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 kd         = clamp(texColor*orennayarGetDiffuseColor(a_pMat), 0.0f, 1.0f);

  const float3 newDir     = MapSampleToCosineDistribution(a_r1, a_r2, a_normal, a_normal, 1.0f);
  const float  cosTheta   = dot(newDir, a_normal);

  a_out->direction = newDir;
  a_out->pdf       = cosTheta*INV_PI;
  a_out->color     = kd*INV_PI*orennayarFunc(newDir, (-1.0f)*ray_dir, a_normal, a_pMat->data[ORENNAYAR_A], a_pMat->data[ORENNAYAR_B]);

  if (cosTheta <= DEPSILON)
    a_out->color = make_float3(0, 0, 0);

  a_out->flags = RAY_EVENT_D;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// perfect mirror material
//
#define MIRROR_COLORX_OFFSET       10
#define MIRROR_COLORY_OFFSET       11
#define MIRROR_COLORZ_OFFSET       12

#define MIRROR_TEXID_OFFSET        13
#define MIRROR_TEXMATRIXID_OFFSET  14

#define MIRROR_SAMPLER0_OFFSET     16

static inline float3 mirrorGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[MIRROR_COLORX_OFFSET], a_pMat->data[MIRROR_COLORY_OFFSET], a_pMat->data[MIRROR_COLORZ_OFFSET]); }
static inline int2   mirrorGetTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[MIRROR_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[MIRROR_TEXMATRIXID_OFFSET]);
  return res;
}


static inline float mirrorEvalPDF(__global const PlainMaterial* a_pMat, float3 l, float3 v, float3 n)
{
  return 0.0f; // because we don't want to sample this material with shadow rays
}

static inline float3 mirrorEvalBxDF(__global const PlainMaterial* a_pMat, float3 l, float3 v, float3 n)
{
  return make_float3(0,0,0);  // because we don't want to sample this material with shadow rays
}

static inline void MirrorSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, const float3 ray_dir, const float3 a_normal, const float2 a_texCoord,
                                           __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                           __private MatSample* a_out)
{
  const float3 texColor = sample2DExt(mirrorGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  
  float3 newDir = reflect(ray_dir, a_normal);
  if (dot(ray_dir, a_normal) > 0.0f)
    newDir = ray_dir;

  const float cosThetaOut = dot(newDir, a_normal);

  // BSDF is multiplied (outside) by cosThetaOut.
  // For mirrors this shouldn't be done, so we pre-divide here instead.
  //
  a_out->direction    = newDir;
  a_out->pdf          = 1.0f;
  a_out->color        = mirrorGetColor(a_pMat)*texColor*(1.0f/fmax(cosThetaOut, 1e-6f));
  if (cosThetaOut <= 1e-6f)
    a_out->color = make_float3(0, 0, 0);

  a_out->flags = RAY_EVENT_S;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef OCL_COMPILER
  #define __my_const_data_t __constant
#else
  #define __my_const_data_t static 
#endif

__my_const_data_t float glosscoeff[10][4] = {
    { 8.88178419700125e-14f, -1.77635683940025e-14f, 5.0f, 1.0f },        // 0-0.1
    { 357.142857142857f, -35.7142857142857f, 5.0f, 1.5f },                // 0.1-0.2
    { -2142.85714285714f, 428.571428571429f, 8.57142857142857f, 2.0f },   // 0.2-0.3
    { 428.571428571431f, -42.8571428571432f, 30.0f, 5.0f },               // 0.3-0.4
    { 2095.23809523810f, -152.380952380952f, 34.2857142857143f, 8.0f },   // 0.4-0.5
    { -4761.90476190476f, 1809.52380952381f, 66.6666666666667f, 12.0f },  // 0.5-0.6
    { 9914.71215351811f, 1151.38592750533f, 285.714285714286f, 32.0f },   // 0.6-0.7
    { 45037.7068059246f, 9161.90096119855f, 813.432835820895f, 82.0f },   // 0.7-0.8
    { 167903.678757035f, 183240.189801913f, 3996.94423223835f, 300.0f },  // 0.8-0.9
    { -20281790.7444668f, 6301358.14889336f, 45682.0925553320f, 2700.0f } // 0.9-1.0
};


static inline float cosPowerFromGlosiness(float glosiness)
{
  //const float cMin = 1.0f;
  const float cMax = 1000000.0f;

  float x = glosiness;

  int k = (fabs(x - 1.0f) < 1e-5f) ? 10 : (int)(x*10.0f);

  const float x1 = (x - (float)(k)*0.1f);

  if (k == 10 || x >= 0.99f)
    return cMax;
  else
    return glosscoeff[k][3] + glosscoeff[k][2] * x1 + glosscoeff[k][1] * x1*x1 + glosscoeff[k][0] * x1*x1*x1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// thin glass surface
//
#define THINGLASS_COLORX_OFFSET       10
#define THINGLASS_COLORY_OFFSET       11
#define THINGLASS_COLORZ_OFFSET       12

#define THINGLASS_TEXID_OFFSET        13
#define THINGLASS_TEXMATRIXID_OFFSET  14

#define THINGLASS_COS_POWER           15
#define THINGLASS_GLOSINESS           16

#define THINGLASS_GLOSINESS_TEXID_OFFSET        17
#define THINGLASS_GLOSINESS_TEXMATRIXID_OFFSET  18

#define THINGLASS_SAMPLER0_OFFSET               20
#define THINGLASS_SMATRIX0_OFFSET               24
#define THINGLASS_SAMPLER1_OFFSET               32
#define THINGLASS_SMATRIX1_OFFSET               36

static inline float3 thinglassGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[THINGLASS_COLORX_OFFSET], a_pMat->data[THINGLASS_COLORY_OFFSET], a_pMat->data[THINGLASS_COLORZ_OFFSET]); }
static inline  int2   thinglassGetTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[THINGLASS_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[THINGLASS_TEXMATRIXID_OFFSET]);
  return res;
}

static inline float thinglassCosPower(__global const PlainMaterial* a_pMat, const float2 a_texCoord,
                                      __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const int2   texId      = make_int2(as_int(a_pMat->data[THINGLASS_GLOSINESS_TEXID_OFFSET]), as_int(a_pMat->data[THINGLASS_GLOSINESS_TEXMATRIXID_OFFSET]));
  const float3 glossColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float  glossMult  = a_pMat->data[THINGLASS_GLOSINESS];
  const float  glosiness  = clamp(glossMult*maxcomp(glossColor), 0.0f, 1.0f);
  return cosPowerFromGlosiness(glosiness);
}



static inline float thinglassEvalPDF(__global const PlainMaterial* a_pMat, float3 l, float3 v, float3 n)
{
  return 0.0f;  // because we don't want to sample thsi material with shadow rays
}

static inline float3 thinglassEvalBxDF(__global const PlainMaterial* a_pMat, float3 l, float3 v, float3 n)
{
  return make_float3(0, 0, 0);  // because we don't want to sample thsi material with shadow rays
}

static inline void ThinglassSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, float3 ray_dir, const float3 a_normal, const float2 a_texCoord,
                                              __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                              __private MatSample* a_out)
{
  const float3 texColor = sample2DExt(thinglassGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float cosPower  = thinglassCosPower(a_pMat, a_texCoord, a_globals, a_tex, a_ptList); // a_pMat->data[THINGLASS_COS_POWER];

  float pdf  = 1.0f;
  float fVal = 1.0f;

  if (cosPower < 1e6f)
  { 
    bool underSurface = false;
    float3 oldDir     = ray_dir;
    ray_dir = MapSampleToModifiedCosineDistribution(a_r1, a_r2, ray_dir, (-1.0f)*a_normal, cosPower, &underSurface);

    float cosTheta = clamp(dot(oldDir, ray_dir), 0.0, M_PI*0.499995f);

    fVal = (cosPower + 2.0f) * INV_TWOPI * pow(cosTheta, cosPower);
    if(underSurface)
      fVal = 0.0f;
    pdf  = pow(cosTheta, cosPower) * (cosPower + 1.0f) * (0.5f * INV_PI);
  }

  const float cosThetaOut = dot(ray_dir, a_normal);
  const float cosMult     = 1.0f / fmax(fabs(cosThetaOut), 1e-6f);

  a_out->direction    = ray_dir;
  a_out->pdf          = pdf;
  a_out->color        = fVal*thinglassGetColor(a_pMat)*texColor*cosMult;
  if (cosThetaOut >= -1e-6f) // refraction/transparency must be under surface!
    a_out->color = make_float3(0, 0, 0);

  a_out->flags = (RAY_EVENT_S | RAY_EVENT_T | RAY_EVENT_TNINGLASS);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// simple glass surface
//

#define GLASS_COLORX_OFFSET       10
#define GLASS_COLORY_OFFSET       11
#define GLASS_COLORZ_OFFSET       12

#define GLASS_TEXID_OFFSET        13
#define GLASS_TEXMATRIXID_OFFSET  14

#define GLASS_IOR_OFFSET          15

#define GLASS_FOG_COLORX_OFFSET   16
#define GLASS_FOG_COLORY_OFFSET   17
#define GLASS_FOG_COLORZ_OFFSET   18
#define GLASS_FOG_MULT_OFFSET     19

#define GLASS_COS_POWER           20
#define GLASS_GLOSINESS           21

#define GLASS_GLOSINESS_TEXID_OFFSET        22
#define GLASS_GLOSINESS_TEXMATRIXID_OFFSET  23

#define GLASS_SAMPLER0_OFFSET               24
#define GLASS_SMATRIX0_OFFSET               28
#define GLASS_SAMPLER1_OFFSET               36
#define GLASS_SMATRIX1_OFFSET               40


static inline float3 glassGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[GLASS_COLORX_OFFSET], a_pMat->data[GLASS_COLORY_OFFSET], a_pMat->data[GLASS_COLORZ_OFFSET]); }
static inline  int2   glassGetTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[GLASS_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[GLASS_TEXMATRIXID_OFFSET]);
  return res;
}

static inline float glassCosPower(__global const PlainMaterial* a_pMat, const float2 a_texCoord, 
                                  __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const int2   texId      = make_int2(as_int(a_pMat->data[GLASS_GLOSINESS_TEXID_OFFSET]), as_int(a_pMat->data[GLASS_GLOSINESS_TEXMATRIXID_OFFSET]));
  const float3 glossColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float  glossMult  = a_pMat->data[GLASS_GLOSINESS];
  const float  glosiness  = clamp(glossMult*maxcomp(glossColor), 0.0f, 1.0f);
  return cosPowerFromGlosiness(glosiness);
}

static inline float4 glassGetFog(__global const PlainMaterial* a_pMat) { return make_float4(a_pMat->data[GLASS_FOG_COLORX_OFFSET], a_pMat->data[GLASS_FOG_COLORY_OFFSET], a_pMat->data[GLASS_FOG_COLORZ_OFFSET], a_pMat->data[GLASS_FOG_MULT_OFFSET]); }

static inline float glassEvalPDF(__global const PlainMaterial* a_pMat, float3 l, float3 v, float3 n)
{
  return 0.0f;  // because we don't want to sample this material with shadow rays
}

static inline float3 glassEvalBxDF(__global const PlainMaterial* a_pMat, float3 l, float3 v, float3 n)
{
  return make_float3(0, 0, 0);  // because we don't want to sample thsi material with shadow rays
}


typedef struct RefractResultT
{

  float3 ray_dir;
  bool   success;
  float  eta;

}RefractResult;


static inline RefractResult myrefract(float3 ray_dir, float3 a_normal, float a_matIOR, float a_outsideIOR, float a_rand)
{
  float eta = a_outsideIOR / a_matIOR; // from air to our material
  float cos_theta = -dot(a_normal, ray_dir);

  if (cos_theta < 0)
  {
    cos_theta *= -1.0f;
    a_normal *= -1.0f;
    eta = 1.0f / eta;
  }

  const float k = 1.0f - eta*eta*(1.0f - cos_theta*cos_theta);

  // to make borders soft
  //
  const float threshold = 0.0025f;
  const bool stochasticRefl = (k < threshold) && (sqrt(a_rand)*threshold >= k);
  const bool refrSuccess = ((k >= 0.0f) && !stochasticRefl);

  if (refrSuccess)
    ray_dir = normalize(eta*ray_dir + (eta*cos_theta - sqrt(k))*a_normal);
  else
    ray_dir = reflect(ray_dir, a_normal);

  RefractResult res;
  res.ray_dir = ray_dir;
  res.success = refrSuccess;
  if (refrSuccess)
    res.eta = eta;
  else
    res.eta = 1.0f;
  return res;
}

static inline void GlassSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float3 rands, const float3 ray_dir, float3 a_normal, const float2 a_texCoord, const bool a_hitFromInside, 
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList, const bool a_isFwdDir,
                                          __private MatSample* a_out)
{
  const float3 normal2 = a_hitFromInside ? (-1.0f)*a_normal : a_normal;
  const float IOR      = a_pMat->data[GLASS_IOR_OFFSET];                // #TODO: add IOR change based on current wave length if spectral trace is used

  RefractResult refractData = myrefract(ray_dir, normal2, IOR, 1.0f, rands.z);

  const float3 texColor = sample2DExt(glassGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float  cosPower = glassCosPower(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  float pdf  = 1.0f;
  float fVal = 1.0f;
  bool  spec = true;

  if (cosPower < 1e6f && refractData.success)
  { 
    bool underSurface   = false;
    float3 oldDir       = refractData.ray_dir;
    refractData.ray_dir = MapSampleToModifiedCosineDistribution(rands.x, rands.y, refractData.ray_dir, (-1.0f)*a_normal, cosPower, &underSurface);

    float cosTheta = clamp(dot(oldDir, refractData.ray_dir), 0.0, M_PI*0.499995f);

    fVal = (cosPower + 2.0f) * INV_TWOPI * pow(cosTheta, cosPower);
    pdf  = pow(cosTheta, cosPower) * (cosPower + 1.0f) * (0.5f * INV_PI);
    spec = false;

    if(underSurface)
      fVal = 0.0f;
  } 

  const float cosThetaOut = dot(refractData.ray_dir, a_normal);
  const float cosMult     = 1.0f / fmax(fabs(cosThetaOut), 1e-6f);
                                                                                     // only camera paths are multiplied by this factor, and etas
                                                                                     // are swapped because radiance flows in the opposite direction
  const float adjointBsdfMult = a_isFwdDir ? 1.0f : refractData.eta*refractData.eta; // see SmallVCM and or Veach adjoint bsdf

  a_out->direction    = refractData.ray_dir;
  a_out->pdf          = pdf;
  a_out->color        = refractData.success ? fVal*clamp(glassGetColor(a_pMat)*texColor, 0.0f, 1.0f)*cosMult*adjointBsdfMult : make_float3(1.0f, 1.0f, 1.0f)*cosMult;

  if(spec)
    a_out->flags = (RAY_EVENT_S | RAY_EVENT_T);
  else
    a_out->flags = (RAY_EVENT_G | RAY_EVENT_T);

  if (refractData.success && cosThetaOut >= -1e-6f)
    a_out->color = make_float3(0, 0, 0);                 // refraction/transparency must be under surface!
  else if (!refractData.success && cosThetaOut < 1e-6f)  // reflection happened in wrong way
    a_out->color = make_float3(0, 0, 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Using the Modified Phong Reflectance Model for Physically ... 
//
#define PHONG_COLORX_OFFSET       10
#define PHONG_COLORY_OFFSET       11
#define PHONG_COLORZ_OFFSET       12

#define PHONG_TEXID_OFFSET        13
#define PHONG_TEXMATRIXID_OFFSET  14

#define PHONG_COSPOWER_OFFSET     15
#define PHONG_GLOSINESS_OFFSET    16

#define PHONG_GLOSINESS_TEXID_OFFSET        17
#define PHONG_GLOSINESS_TEXMATRIXID_OFFSET  18

#define PHONG_SAMPLER0_OFFSET               20
#define PHONG_SMATRIX0_OFFSET               24

#define PHONG_SAMPLER1_OFFSET               32
#define PHONG_SMATRIX1_OFFSET               36

static inline float3 phongGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[PHONG_COLORX_OFFSET], a_pMat->data[PHONG_COLORY_OFFSET], a_pMat->data[PHONG_COLORZ_OFFSET]); }
static inline int2   phongGetTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[PHONG_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[PHONG_TEXMATRIXID_OFFSET]);
  return res;
}


static inline float phongGlosiness(__global const PlainMaterial* a_pMat, const float2 a_texCoord,
                                   __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  if (as_int(a_pMat->data[PHONG_GLOSINESS_TEXID_OFFSET]) != INVALID_TEXTURE)
  {
    const int2   texId      = make_int2(as_int(a_pMat->data[PHONG_GLOSINESS_TEXID_OFFSET]), as_int(a_pMat->data[PHONG_GLOSINESS_TEXMATRIXID_OFFSET]));
    const float3 glossColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
    const float  glossMult  = a_pMat->data[PHONG_GLOSINESS_OFFSET];
    const float  glosiness  = clamp(glossMult*maxcomp(glossColor), 0.0f, 1.0f);
    return glosiness; 
  }
  else
    return a_pMat->data[PHONG_GLOSINESS_OFFSET];
}

static inline float phongEvalPDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float2 a_texCoord,
                                 __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 r        = reflect((-1.0)*v, n);
  const float  cosTheta = clamp(fabs(dot(l, r)), DEPSILON2, M_PI*0.499995f); 

  const float  gloss    = phongGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
  const float  cosPower = cosPowerFromGlosiness(gloss);

  return pow(cosTheta, cosPower) * (cosPower + 1.0f) * INV_TWOPI; // please see "Using the Modified Phong Reflectance Model for Physically ... 
}


static inline float PhongEnergyFix(const float dotRL, const float3 l, const float3 n, const float3 v)
{ 
  const float dotNL         = dot(n, l);
  const float dotNV         = dot(n, (-1.0f) * v);
  
  const float geomCosFix    = 1.0f / dotNL * dotRL; // the transfer of geometric cosine in the space of reflection.
  
  const float volUnderSurf      = 2.0f / 3.0f * M_PI * (1.0f - dotRL); // the volume of the spherical segment 
  const float halfVolUnderSurf  = 1.0f + volUnderSurf;// *0.5f;
  const float power             = 1;// 0.0f + halfVolUnderSurf * (1.0f - 0.0f);
  const float underSurfFix      = halfVolUnderSurf + dotNV * (1.0f - halfVolUnderSurf); // fix half ray under surface

  return geomCosFix;// *underSurfFix;
}

static inline float3 phongEvalBxDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float2 a_texCoord, const int a_evalFlags,
                                   __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 texColor = sample2DExt(phongGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color    = clamp(phongGetColor(a_pMat)*texColor, 0.0f, 1.0f);
  const float  gloss    = phongGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList); 
  const float  cosPower = cosPowerFromGlosiness(gloss);

  const float3 r        = reflect((-1.0)*v, n);
  const float  cosAlpha = clamp(dot(l, r), 0.0f, 1.0f);

  const float energyFix = PhongEnergyFix(cosAlpha, l, n, v);
  
  return color*(cosPower + 2.0f)*INV_TWOPI*pow(cosAlpha, cosPower)*energyFix; // please see "Using the Modified Phong Reflectance Model for Physically ... 
}

static inline void PhongSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, const float3 ray_dir, const float3 a_normal, const float2 a_texCoord,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                          __private MatSample* a_out)
{
  const float3 texColor = sample2DExt(phongGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color    = clamp(phongGetColor(a_pMat)*texColor, 0.0f, 1.0f);
  const float  gloss    = phongGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
  const float  cosPower = cosPowerFromGlosiness(gloss);
  
  bool underSurface     = false;
  const float3 r        = reflect(ray_dir, a_normal);
  const float3 newDir   = MapSampleToModifiedCosineDistribution(a_r1, a_r2, r, a_normal, cosPower, &underSurface);

  const float cosAlpha  = clamp(dot(newDir, r), 0.0, 1.0f);
  const float dotNL     = dot(a_normal, newDir);
  const float energyFix = PhongEnergyFix(cosAlpha, newDir, a_normal, ray_dir);

  const float powCosAlphaCosPowInvTwoPi = pow(cosAlpha, cosPower) * INV_TWOPI;

  a_out->direction  = newDir;
  a_out->pdf        = powCosAlphaCosPowInvTwoPi * (cosPower + 1.0f);
  a_out->color      = powCosAlphaCosPowInvTwoPi * (cosPower + 2.0f) * color * energyFix; // please see "Using the Modified Phong Reflectance Model for Physically ... 
    
  if (dotNL <= 1e-6f || underSurface)                                                // reflection under surface must be zerowed!
    a_out->color = make_float3(0, 0, 0);

  a_out->flags = (gloss == 1.0f) ? RAY_EVENT_S : RAY_EVENT_G;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// microfacet mode, blinn distribution and torrance-sparrow brdf in analogue to pbrt
//
#define BLINN_COLORX_OFFSET       10
#define BLINN_COLORY_OFFSET       11
#define BLINN_COLORZ_OFFSET       12

#define BLINN_TEXID_OFFSET        13
#define BLINN_TEXMATRIXID_OFFSET  14

#define BLINN_COSPOWER_OFFSET     15
#define BLINN_GLOSINESS_OFFSET    16

#define BLINN_GLOSINESS_TEXID_OFFSET        17
#define BLINN_GLOSINESS_TEXMATRIXID_OFFSET  18
#define BLINN_ANISOTROPY_OFFSET             19 

#define BLINN_SAMPLER0_OFFSET               20 
#define BLINN_SAMPLER1_OFFSET               32

static inline float3 blinnGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[BLINN_COLORX_OFFSET], a_pMat->data[BLINN_COLORY_OFFSET], a_pMat->data[BLINN_COLORZ_OFFSET]); }
static inline int2   blinnGetTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[BLINN_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[BLINN_TEXMATRIXID_OFFSET]);
  return res;
}

static inline float blinnGlosiness(__global const PlainMaterial* a_pMat, const float2 a_texCoord, 
                                   __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  if (as_int(a_pMat->data[BLINN_GLOSINESS_TEXID_OFFSET]) != INVALID_TEXTURE)
  {
    const int2   texId      = make_int2(as_int(a_pMat->data[BLINN_GLOSINESS_TEXID_OFFSET]), as_int(a_pMat->data[BLINN_GLOSINESS_TEXMATRIXID_OFFSET]));
    const float3 glossColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
    const float  glossMult  = a_pMat->data[BLINN_GLOSINESS_OFFSET];
    const float  glosiness  = clamp(glossMult*maxcomp(glossColor), 0.0f, 1.0f);
    return glosiness;
  }
  else
    return a_pMat->data[BLINN_GLOSINESS_OFFSET];
}

static inline float blinnCosPower(__global const PlainMaterial* a_pMat, const float2 a_texCoord, 
                                  __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  return cosPowerFromGlosiness(blinnGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList));
}


static inline float blinnEvalPDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, 
                                 const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float  exponent = blinnCosPower(a_pMat, a_texCoord, a_globals, a_tex, a_ptList); // a_pMat->data[BLINN_COSPOWER_OFFSET];

  const float3 wh      = normalize(l + v);
  const float costheta = fabs(dot(wh,n));

  float blinn_pdf = ((exponent + 1.0f) * pow(costheta, exponent)) / (M_TWOPI * 4.0f * dot(l, wh));

  // if (dot(l, wh) <= 0.0f) // #TODO: this may cause problems when under-surface hit during PT; light strategy weight becobes zero. Or not? see costheta = fabs(dot(wh,n))
  //   blinn_pdf = 0.0f;     // #TODO: this may cause problems when under-surface hit during PT; light strategy weight becobes zero. Or not? see costheta = fabs(dot(wh,n))

  return blinn_pdf;
}

static inline float3 blinnEvalBxDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, 
                                   const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 texColor = sample2DExt(blinnGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);

  const float3 color    = clamp(blinnGetColor(a_pMat)*texColor, 0.0f, 1.0f);
  const float  gloss    = blinnGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
  const float  exponent = cosPowerFromGlosiness(gloss); 

  const float3 wh       = normalize(l + v);

  const float cosThetaH = fabs(dot(wh,n));
  const float D         = (exponent + 2.0f) * INV_TWOPI * pow(cosThetaH, exponent);
  const float dotNL     = fmax(dot(l, n), 0.0f);

  const float Pss       = D * TorranceSparrowGF2(l, v, n);               // Energy single-scattering
  const float Pms       = 1.0f;// MultiScatterBRDF(0, 0, dotNL, 0, Pss, color);  // Energy multiple-scattering

  return color * Pss * Pms;
}

static inline void BlinnSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, const float3 ray_dir, const float3 a_normal, const float2 a_texCoord, const float3 a_tan, const float3 a_bitan,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                          __private MatSample* a_out)
{
  const float3 texColor = sample2DExt(blinnGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color    = clamp(blinnGetColor(a_pMat)*texColor, 0.0f, 1.0f);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
  //
  float3 nx, ny, nz     = a_normal;
  const float aniso     = a_pMat->data[BLINN_ANISOTROPY_OFFSET];
  const float glossOrig = blinnGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  float phi, gloss;
  CoordinateSystem(nz, &nx, &ny);
  phi   = a_r2 * M_TWOPI;
  gloss = glossOrig;
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo        = make_float3(-dot(ray_dir, nx), -dot(ray_dir, ny), -dot(ray_dir, nz));
  const float  exponent  = cosPowerFromGlosiness(gloss);                                                //

  // Compute sampled half-angle vector $\wh$ for Blinn distribution
  //
  const float costheta = pow(a_r1, 1.0f / (exponent + 1.0f));
  const float sintheta = sqrt(fmax(0.0f, 1.0f - costheta*costheta));
  
  float3 wh = SphericalDirectionPBRT(sintheta, costheta, phi);

  bool underSurface = false;
  if (!SameHemispherePBRT(wo, wh))
    underSurface = true;

  const float3 wi = (2.0f * dot(wo, wh) * wh) - wo; // Compute incident direction by reflecting about $\wh$

  const float blinn_pdf = ((exponent + 1.0f) * pow(costheta, exponent)) / fmax(M_TWOPI * 4.0f * dot(wo, wh), DEPSILON);
  const float D         = ((exponent + 2.0f) * INV_TWOPI * pow(costheta, exponent));

  const float3 newDir   = wi.x*nx + wi.y*ny + wi.z*nz; // back to normal coordinate system
  const float  dotNL    = dot(newDir, a_normal);
  const float  GF1      = TorranceSparrowGF1(wo, wi);

  //const float estimatedThoroughput = fabs(dotNL)*(D * GF1) / fmax(blinn_pdf, DEPSILON2);
  //const float brightBordersMult    = fmin(estimatedThoroughput, 1.0f) / fmax(estimatedThoroughput, DEPSILON2);
  
  const float Pss       = D * GF1;                                       // Pass single-scattering
  const float Pms       = 1.0f;// MultiScatterBRDF(0, 0, dotNL, 0, Pss, color);  // Pass multiple-scattering

  a_out->direction      = newDir; 
  a_out->pdf            = blinn_pdf;
  a_out->color          = color * Pss *Pms; // *brightBordersMult;

  if(dotNL < 1e-6f || underSurface)
    a_out->color   = make_float3(0,0,0);

  a_out->flags     = RAY_EVENT_G;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GGX_COLORX_OFFSET       10
#define GGX_COLORY_OFFSET       11
#define GGX_COLORZ_OFFSET       12

#define GGX_TEXID_OFFSET        13
#define GGX_TEXMATRIXID_OFFSET  14

#define GGX_COSPOWER_OFFSET     15
#define GGX_GLOSINESS_OFFSET    16

#define GGX_GLOSINESS_TEXID_OFFSET        17
#define GGX_GLOSINESS_TEXMATRIXID_OFFSET  18

#define GGX_SAMPLER0_OFFSET               20
#define GGX_SMATRIX0_OFFSET               24

#define GGX_SAMPLER1_OFFSET               32
#define GGX_SMATRIX1_OFFSET               36

static inline float3 ggxGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[GGX_COLORX_OFFSET], a_pMat->data[GGX_COLORY_OFFSET], a_pMat->data[GGX_COLORZ_OFFSET]); }
static inline int2   ggxGetTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[GGX_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[GGX_TEXMATRIXID_OFFSET]);
  return res;
}


static inline float ggxGlosiness(__global const PlainMaterial* a_pMat, const float2 a_texCoord,
                                 __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  if (as_int(a_pMat->data[GGX_GLOSINESS_TEXID_OFFSET]) != INVALID_TEXTURE)
  {
    const int2   texId = make_int2(as_int(a_pMat->data[GGX_GLOSINESS_TEXID_OFFSET]), as_int(a_pMat->data[GGX_GLOSINESS_TEXMATRIXID_OFFSET]));
    const float3 glossColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
    const float  glossMult = a_pMat->data[GGX_GLOSINESS_OFFSET];
    const float  glosiness = clamp(glossMult*maxcomp(glossColor), 0.0f, 0.999f);
    return glosiness;
  }
  else
    return a_pMat->data[GGX_GLOSINESS_OFFSET];
}

static inline float Luminance(const float3 color) { return (color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f); } //https://en.wikipedia.org/wiki/Relative_luminance



// GGX
static inline float GGX_HeightCorrelatedGeomShadMask(const float dotNV, const float dotNL, const float alpha)
{
  // Height-Correlated Masking and Shadowing function
  // Masking
  const float tanWo = sqrt(1.0f - dotNV * dotNV) / dotNV;
  const float aWo = 1.0f / (alpha * tanWo);
  const float lambdaWo = (-1.0f + sqrt(1.0f + 1.0f / (aWo*aWo))) / 2.0f;
  // Shadowing
  const float tanWi = sqrt(1.0f - dotNL * dotNL) / dotNL;
  const float aWi = 1.0f / (alpha * tanWi);
  const float lambdaWi = (-1.0f + sqrt(1.0f + 1.0f / (aWi*aWi))) / 2.0f;
  // Full
  const float G = 1.0f / (1.0f + lambdaWo + lambdaWi);
  return G;
}

static inline float GGX_GeomShadMask(const float cosThetaN, const float alpha)
{
  // Height - Correlated G.
  //const float tanNV      = sqrt(1.0f - cosThetaN * cosThetaN) / cosThetaN;
  //const float a          = 1.0f / (alpha * tanNV);
  //const float lambda     = (-1.0f + sqrt(1.0f + 1.0f / (a*a))) / 2.0f;
  //const float G          = 1.0f / (1.0f + lambda);

  // Optimized and equal to the commented-out formulas on top.
  const float cosTheta_sqr = clamp(cosThetaN*cosThetaN, 0.0f, 1.0f);
  const float tan2         = (1.0f - cosTheta_sqr) / cosTheta_sqr;
  const float GP           = 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * tan2));
  return GP;
}

static inline float GGX_Distribution(const float cosThetaNH, const float alpha)
{
  const float alpha2  = alpha * alpha;
  const float NH_sqr  = clamp(cosThetaNH * cosThetaNH, 0.0f, 1.0f);
  const float den     = NH_sqr * alpha2 + (1.0f - NH_sqr);
  return alpha2 / (M_PI * den * den);
}

static inline float ggxEvalPDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float2 a_texCoord,
                               __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float dotNL = dot(n, l);
  const float dotNV = dot(n, v);

  if (dotNV <= 0.0f || dotNL <= 1e-6f)
    return 0.0f;

  const float3 texColor = sample2DExt(ggxGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color    = clamp(ggxGetColor(a_pMat)*texColor, 0.0f, 1.0f);
  const float maxColor  = fmax(color.x, fmax(color.y, color.z));
  const float  gloss    = ggxGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
  const float roughness = 1.0f - gloss;
  const float roughSqr  = roughness * roughness;

  const float3 h    = normalize(v + l); // half vector.
  const float dotNH = dot(n, h);
  const float dotHV = dot(h, v);

  const float D = GGX_Distribution(dotNH, roughSqr);

  return  D * dotNH / (4.0f * dotHV);
}

static inline float3 ggxEvalBxDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float2 a_texCoord, const int a_evalFlags,
                                 __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  //GGX for light (explicit strategy).
  const float dotNL = dot(n, l);
  const float dotNV = dot(n, v);

  if (dotNV <= 0.0f || dotNL <= 1e-6f)
    return make_float3(0.0f, 0.0f, 0.0f);

  const float3 texColor = sample2DExt(ggxGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color    = clamp(ggxGetColor(a_pMat)*texColor, 0.0f, 1.0f);
  const float maxColor  = fmax(color.x, fmax(color.y, color.z));
  const float  gloss    = ggxGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
  const float roughness = 1.0f - gloss;
  const float roughSqr  = roughness * roughness;

  const float3 h    = normalize(v + l); // half vector.
  const float dotNH = dot(n, h);

  const float D     = GGX_Distribution(dotNH, roughSqr);
  //const float G     = GGX_GeomShadMask(dotNV, roughSqr)*GGX_GeomShadMask(dotNL, roughSqr); more correctly than a simple multiplication of masks.
  const float G     = GGX_HeightCorrelatedGeomShadMask(dotNV, dotNL, roughSqr);
  //const float F   = 1.0f;                                                       // Fresnel is not needed here, because it is used for the blend with diffusion.

  const float Pss   = D /** F*/ * G / (4.0f * dotNV * dotNL);                     // Pass single-scattering

  return color * Pss;
}

static inline void GGXSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, const float3 ray_dir, const float3 a_normal, const float2 a_texCoord,
                                        __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                        __private MatSample* a_out)
{
  // GGX for implicit strategy
  const float3 texColor   = sample2DExt(ggxGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color      = clamp(ggxGetColor(a_pMat)*texColor, 0.0f, 1.0f);
  const float  gloss      = ggxGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
  const float  roughness  = 1.0f - gloss;
  const float  roughSqr   = roughness * roughness;

  float3 nx, ny, nz       = a_normal;
  CoordinateSystem(nz, &nx, &ny);

  // to PBRT coordinate system
  const float3 wo       = make_float3(-dot(ray_dir, nx), -dot(ray_dir, ny), -dot(ray_dir, nz));   // wo (output) = v = ray_dir

  // Compute sampled half-angle vector wh
  const float phi       = a_r1 * M_TWOPI;
  const float cosTheta  = clamp(sqrt((1.0f - a_r2) / (1.0f + roughSqr * roughSqr * a_r2 - a_r2)), 0.0f, 1.0f);
  const float sinTheta  = sqrt(1.0f - cosTheta * cosTheta);
  float3 wh             = SphericalDirectionPBRT(sinTheta, cosTheta, phi);

  bool underSurface = false;
  if (!SameHemispherePBRT(wo, wh))
    underSurface = true;

  const float3 wi       = (2.0f * dot(wo, wh) * wh) - wo; // Compute incident direction by reflecting about wh. wi (input) = light
  const float3 newDir   = wi.x*nx + wi.y*ny + wi.z*nz;    // back to normal coordinate system

  const float3 v    = (-1.0f) * ray_dir;
  const float3 l    = newDir;
  const float3 h    = normalize(v + l);                   // half vector.
  const float dotNL = clamp(dot(a_normal, l), 0.0f, 1.0f);
  const float dotNV = clamp(dot(a_normal, v), 0.0f, 1.0f);
  const float dotNH = clamp(dot(a_normal, h), 0.0f, 1.0f);
  const float dotHV = clamp(dot(h, v), 0.0f, 1.0f);
  const float dotHL = clamp(dot(h, l), 0.0f, 1.0f);
    
  const float D     = GGX_Distribution(dotNH, roughSqr);
  //const float G     = GGX_GeomShadMask(dotNV, roughSqr) * GGX_GeomShadMask(dotNL, roughSqr); 
  const float G     = GGX_HeightCorrelatedGeomShadMask(dotNV, dotNL, roughSqr); // more correctly than a simple multiplication of masks.
  //const float F   = 1.0f;                                                     // Fresnel is not needed here, because it is used for the blend with diffusion.

  const float Pss   = D /** F*/ * G / (4.0f * dotNV * dotNL);                   // Pass single-scattering
  
  a_out->direction  = newDir;
  a_out->pdf        = D * dotNH / (4.0f * dotHV);    
  a_out->color      = color * Pss;  

  if (dotNV <= 1e-6f || dotNL <= 1e-6f || underSurface) // reflection under surface must be zerowed!
    a_out->color = make_float3(0, 0, 0);

  a_out->flags = (gloss == 1.0f) ? RAY_EVENT_S : RAY_EVENT_G;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BECKMANN_COLORX_OFFSET       10
#define BECKMANN_COLORY_OFFSET       11
#define BECKMANN_COLORZ_OFFSET       12

#define BECKMANN_TEXID_OFFSET        13
#define BECKMANN_TEXMATRIXID_OFFSET  14

#define BECKMANN_COSPOWER_OFFSET     15
#define BECKMANN_GLOSINESS_OFFSET    16

#define BECKMANN_GLOSINESS_TEXID_OFFSET        17
#define BECKMANN_GLOSINESS_TEXMATRIXID_OFFSET  18
#define BECKMANN_ANISOTROPY_OFFSET             19

#define BECKMANN_SAMPLER0_OFFSET               20 
#define BECKMANN_SAMPLER1_OFFSET               32
#define BECKMANN_SAMPLER2_OFFSET               44
#define BECKMANN_SAMPLER3_OFFSET               56

#define BECKMANN_ANISO_ROT_OFFSET              68

#define BECKMANN_ANISO_TEXID_OFFSET            69
#define BECKMANN_ANISO_TEXMATRIXID_OFFSET      70

#define BECKMANN_ROT_TEXID_OFFSET              71
#define BECKMANN_ROT_TEXMATRIXID_OFFSET        72

static inline float3 beckmannGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[BECKMANN_COLORX_OFFSET], a_pMat->data[BECKMANN_COLORY_OFFSET], a_pMat->data[BECKMANN_COLORZ_OFFSET]); }
static inline int2   beckmannGetTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[BECKMANN_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[BECKMANN_TEXMATRIXID_OFFSET]);
  return res;
}


static inline float beckmannGlosiness(__global const PlainMaterial* a_pMat, const float2 a_texCoord, 
                                      __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  if (as_int(a_pMat->data[BECKMANN_GLOSINESS_TEXID_OFFSET]) != INVALID_TEXTURE)
  {
    const int2   texId      = make_int2(as_int(a_pMat->data[BECKMANN_GLOSINESS_TEXID_OFFSET]), as_int(a_pMat->data[BECKMANN_GLOSINESS_TEXMATRIXID_OFFSET]));
    const float3 glossColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
    const float  glossMult  = a_pMat->data[BECKMANN_GLOSINESS_OFFSET];
    const float  glosiness  = clamp(glossMult*maxcomp(glossColor), 0.0f, 0.999f);
    return glosiness;
  }
  else
    return a_pMat->data[BECKMANN_GLOSINESS_OFFSET];
}

static inline float beckmannAnisotropy(__global const PlainMaterial* a_pMat, const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList) 
{ 
  const int2   texId      = make_int2(as_int(a_pMat->data[BECKMANN_ANISO_TEXID_OFFSET]), as_int(a_pMat->data[BECKMANN_ANISO_TEXMATRIXID_OFFSET]));
  const float3 anisoColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float texMult     = maxcomp(anisoColor); // (texId.y == INVALID_TEXTURE) ? 1.0f : 1.0f - maxcomp(anisoColor);

  return clamp(a_pMat->data[BECKMANN_ANISOTROPY_OFFSET]*texMult, 0.0f, 1.0f); 
}

static inline float beckmannAnisoRot(__global const PlainMaterial* a_pMat, const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList) 
{ 
  const int2   texId    = make_int2(as_int(a_pMat->data[BECKMANN_ROT_TEXID_OFFSET]), as_int(a_pMat->data[BECKMANN_ROT_TEXMATRIXID_OFFSET]));
  const float3 rotColor = sample2DExt(texId.y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float texMult   = maxcomp(rotColor); //(texId.y == INVALID_TEXTURE) ? 1.0f : 1.0f - maxcomp(rotColor);

  return clamp(a_pMat->data[BECKMANN_ANISO_ROT_OFFSET]*texMult, 0.0f, 1.0f); 
}

static inline float2 beckmannAlphaXY(__global const PlainMaterial* a_pMat, const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float roughness = 0.5f - 0.5f*beckmannGlosiness(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
  const float anisoMult = 1.0f - beckmannAnisotropy(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  const float alphax = BeckmannRoughnessToAlpha(roughness*roughness);  
  const float alphay = BeckmannRoughnessToAlpha(roughness*roughness*anisoMult*anisoMult);  

  return make_float2(alphax, alphay);
}

static inline void BeckmanTangentSpace(__global const PlainMaterial* a_pMat, float2 alpha, float3 nz, float3 a_tan, float3 a_bitan, float2 a_texCoord,
                                       __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                       float3* pNx, float3* pNy)
{
  if(fabs(alpha.x - alpha.y) > 1.0e-5f)
  {
    (*pNx) = a_bitan;
    (*pNy) = a_tan;

    const float rotVal  = beckmannAnisoRot(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);
    const float4x4 mRot = RotateAroundVector4x4(nz, rotVal*M_PI*2.0f);
    (*pNx) = mul3x3(mRot, (*pNx));
    (*pNy) = mul3x3(mRot, (*pNy));
  }
  else
  {
    CoordinateSystem(nz, pNx, pNy);
  }

  if((materialGetFlags(a_pMat) & PLAIN_MATERIAL_FLIP_TANGENT) != 0)
  {
    float3 temp = (*pNx);
    (*pNx) = (*pNy);
    (*pNy) = temp; 
  }

}

static inline float beckmannEvalPDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float3 a_tan, const float3 a_bitan,
                                    const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float2 alpha = beckmannAlphaXY(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
  //
  float3 nx, ny, nz = n;
  BeckmanTangentSpace(a_pMat, alpha, n, a_tan, a_bitan, a_texCoord, a_globals, a_tex, a_ptList, 
                      &nx, &ny);
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo  = make_float3(-dot(v, nx), -dot(v, ny), -dot(v, nz));
  const float3 wh  = normalize(l + v);
  
  return BeckmannDistributionPdf(wo, wh, alpha.x, alpha.y);
}

static inline float3 beckmannEvalBxDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float3 a_tan, const float3 a_bitan,
                                      const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 texColor = sample2DExt(beckmannGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color    = clamp(texColor*beckmannGetColor(a_pMat), 0.0f, 1.0f);

  const float2 alpha = beckmannAlphaXY(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
  //
  float3 nx, ny, nz = n;
  BeckmanTangentSpace(a_pMat, alpha, nz, a_tan, a_bitan, a_texCoord, a_globals, a_tex, a_ptList, 
                      &nx, &ny);
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo  = make_float3(-dot(v, nx), -dot(v, ny), -dot(v, nz));
  const float3 wi  = make_float3(-dot(l, nx), -dot(l, ny), -dot(l, nz));
  
  return color*fmin(BeckmannBRDF_PBRT(wo, wi, alpha.x, alpha.y), 500.0f);
}

static inline void BeckmannSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, 
                                             const float3 ray_dir, const float3 a_normal, const float2 a_texCoord, const float3 a_tan, const float3 a_bitan,
                                             __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                             __private MatSample* a_out)
{
  const float2 alpha = beckmannAlphaXY(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
  //
  float3 nx, ny, nz = a_normal;
  BeckmanTangentSpace(a_pMat, alpha, nz, a_tan, a_bitan, a_texCoord, a_globals, a_tex, a_ptList, 
                      &nx, &ny);
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo  = make_float3(-dot(ray_dir, nx), -dot(ray_dir, ny), -dot(ray_dir, nz));
  const float3 wh  = BeckmannDistributionSampleWH(wo, make_float2(a_r1, a_r2), alpha.x, alpha.y);
  const float3 wi  = (2.0f * dot(wo, wh) * wh) - wo; // Compute incident direction by reflecting about wh
  
  const float3 texColor    = sample2DExt(beckmannGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 kd          = clamp(texColor*beckmannGetColor(a_pMat), 0.0f, 1.0f);
  
  const float3 newDir      = wi.x*nx + wi.y*ny + wi.z*nz; // back to normal coordinate system
  const float  cosThetaOut = dot(newDir, a_normal);

  a_out->direction = newDir;
  a_out->pdf       = BeckmannDistributionPdf(wo, wh, alpha.x, alpha.y);
  if (cosThetaOut <= DEPSILON)
    a_out->color = make_float3(0, 0, 0);
  else  
    a_out->color = kd*fmin(BeckmannBRDF_PBRT(wo, wi, alpha.x, alpha.y), 500.0f);
  a_out->flags = RAY_EVENT_G;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float trggxEvalPDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float3 a_tan, const float3 a_bitan,
                                 const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float2 alpha = beckmannAlphaXY(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
  //
  float3 nx, ny, nz = n;
  BeckmanTangentSpace(a_pMat, alpha, nz, a_tan, a_bitan, a_texCoord, a_globals, a_tex, a_ptList, 
                      &nx, &ny);
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo = make_float3(-dot(v, nx), -dot(v, ny), -dot(v, nz));
  const float3 wh = normalize(l + v);

  return TrowbridgeReitzDistributionPdf(wo, wh, alpha.x, alpha.y);
}

static inline float3 trggxEvalBxDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float3 a_tan, const float3 a_bitan,
                                   const float2 a_texCoord, __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 texColor = sample2DExt(beckmannGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 color    = clamp(texColor*beckmannGetColor(a_pMat), 0.0f, 1.0f);

  const float2 alpha    = beckmannAlphaXY(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
  //
  float3 nx, ny, nz = n;
  BeckmanTangentSpace(a_pMat, alpha, nz, a_tan, a_bitan, a_texCoord, a_globals, a_tex, a_ptList, 
                      &nx, &ny);
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo = make_float3(-dot(v, nx), -dot(v, ny), -dot(v, nz));
  const float3 wi = make_float3(-dot(l, nx), -dot(l, ny), -dot(l, nz));

  return color*TrowbridgeReitzBRDF_PBRT(wo, wi, alpha.x, alpha.y);
}

static inline void TRGGXSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2,
                                          const float3 ray_dir, const float3 a_normal, const float2 a_texCoord, const float3 a_tan, const float3 a_bitan,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                          __private MatSample* a_out)
{
  const float2 alpha = beckmannAlphaXY(a_pMat, a_texCoord, a_globals, a_tex, a_ptList);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
  //
  float3 nx, ny, nz = a_normal;
  BeckmanTangentSpace(a_pMat, alpha, nz, a_tan, a_bitan, a_texCoord, a_globals, a_tex, a_ptList, 
                      &nx, &ny);
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo = make_float3(-dot(ray_dir, nx), -dot(ray_dir, ny), -dot(ray_dir, nz));
  const float3 wh = TrowbridgeReitzDistributionSampleWH(wo, make_float2(a_r1, a_r2), alpha.x, alpha.y);
  const float3 wi = (2.0f * dot(wo, wh) * wh) - wo; // Compute incident direction by reflecting about wh

  const float3 texColor = sample2DExt(beckmannGetTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 kd       = clamp(texColor*beckmannGetColor(a_pMat), 0.0f, 1.0f);

  const float3 newDir      = wi.x*nx + wi.y*ny + wi.z*nz; // back to normal coordinate system
  const float  cosThetaOut = dot(newDir, a_normal);

  a_out->direction = newDir;
  a_out->pdf       = TrowbridgeReitzDistributionPdf(wo, wh, alpha.x, alpha.y);
  if (cosThetaOut <= DEPSILON)
    a_out->color = make_float3(0, 0, 0);
  else
    a_out->color = kd*TrowbridgeReitzBRDF_PBRT(wo, wi, alpha.x, alpha.y);
  a_out->flags = RAY_EVENT_G;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// translucent lambert material
//
#define TRANS_COLORX_OFFSET       10
#define TRANS_COLORY_OFFSET       11
#define TRANS_COLORZ_OFFSET       12

#define TRANS_TEXID_OFFSET        13
#define TRANS_TEXMATRIXID_OFFSET  14

#define TRANS_SAMPLER0_OFFSET     20
#define TRANS_SMATRIX0_OFFSET     24

static inline float3 translucentGetDiffuseColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[TRANS_COLORX_OFFSET], a_pMat->data[TRANS_COLORY_OFFSET], a_pMat->data[TRANS_COLORZ_OFFSET]); }
static inline int2   translucentGetDiffuseTex(__global const PlainMaterial* a_pMat)
{
  int2 res;
  res.x = as_int(a_pMat->data[TRANS_TEXID_OFFSET]);
  res.y = as_int(a_pMat->data[TRANS_TEXMATRIXID_OFFSET]);
  return res;
}


static inline float  translucentEvalPDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n)
{
  float sign1 = dot(l, n) > 0 ? 1.0f : -1.0f;
  float sign2 = dot(v, n) > 0 ? 1.0f : -1.0f;

  float coeff = (sign1*sign2 < 0.0f) ? 1.0f : 0.0f;
  return fabs(dot(l, n))*INV_PI*coeff;
}

static inline float3 translucentEvalBxDF(__global const PlainMaterial* a_pMat, const float3 l, const float3 v, const float3 n, const float2 a_texCoord,
                                         __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const float3 texColor = sample2DExt(translucentGetDiffuseTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);

  const float sign1 = dot(l,n) > 0 ? 1.0f : -1.0f;
  const float sign2 = dot(v,n) > 0 ? 1.0f : -1.0f;
  const float coeff = (sign1*sign2 < 0.0f) ? 1.0f : 0.0f;

  return clamp(texColor*translucentGetDiffuseColor(a_pMat), 0.0f, 1.0f)*coeff*INV_PI;
}

static inline void TranslucentSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, const float a_r1, const float a_r2, const float3 ray_dir, const float3 a_normal, const float2 a_texCoord,
                                                __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList,
                                                __private MatSample* a_out)
{
  const float3 texColor = sample2DExt(translucentGetDiffuseTex(a_pMat).y, a_texCoord, (__global const int4*)a_pMat, a_tex, a_globals, a_ptList);
  const float3 kd       = clamp(texColor*translucentGetDiffuseColor(a_pMat), 0.0f, 1.0f);
  const float3 newDir   = MapSampleToCosineDistribution(a_r1, a_r2, (-1.0f)*a_normal, (-1.0f)*a_normal, 1.0f);
  const float  cosTheta = dot(newDir, (-1.0f)*a_normal);

  a_out->direction    = newDir;
  a_out->pdf          = cosTheta*INV_PI;
  a_out->color        = kd*INV_PI;
  if (cosTheta <= 1e-6f)
    a_out->color = make_float3(0, 0, 0);

  a_out->flags = (RAY_EVENT_D | RAY_EVENT_T);
}


static inline bool materialHasDiffuse(__global const PlainMaterial* a_pMat) // # NOT STRICLY CORRECT lambert/orennayer; works just because lambertGetDiffuseColor and orennayarGetDiffuseColor use same offsets!!!!!
{
  return (
           ( (materialGetType(a_pMat) == PLAIN_MAT_CLASS_LAMBERT || materialGetType(a_pMat) == PLAIN_MAT_CLASS_OREN_NAYAR) && (length(lambertGetDiffuseColor(a_pMat)) > 1e-3f) ) ||
           ( (materialGetType(a_pMat) == PLAIN_MAT_CLASS_TRANSLUCENT) && (length(translucentGetDiffuseColor(a_pMat)) > 1e-3f) )
         );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float  shadowmatteEvalPDF(__global const PlainMaterial* a_pMat, float3 l, float3 v, float3 n) { return 0.0f; }

static inline float3 shadowmatteEvalBxDF(__global const PlainMaterial* a_pMat, float3 l, float3 n, float2 a_texCoord)
{
  return make_float3(0,0,0);
}

static inline void ShadowmatteSampleAndEvalBRDF(__global const PlainMaterial* a_pMat, float3 ray_dir, float3 a_normal, float3 a_shadowVal, 
                                                __private MatSample* a_out)
{
  const float cosThetaOut = dot(ray_dir, a_normal);

  // const float t           = fmin(a_shadowVal.x, fmin(a_shadowVal.y, a_shadowVal.x));
  // const float3 colorBlack = shadowMatteColor(a_pMat);
  // const float3 colorRes   = (t < 0.5f) ? colorBlack : make_float3(1,1,1);

  a_out->direction    = ray_dir;
  a_out->pdf          = 1.0f;
  a_out->color        = a_shadowVal/fmax(fabs(cosThetaOut), 1e-5f);
  a_out->flags        = RAY_EVENT_S | RAY_EVENT_T;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MMIX_NUMC_OFFSET    10
#define MMIX_DUMMY_OFFSET   11

#define MMIX_START          12
#define MMIX_MAX_COMPONENTS ((PLAIN_MATERIAL_DATA_SIZE - MMIX_START)/2)

typedef struct BRDFSelectorT
{
  int   localOffs;
  float w;

} BRDFSelector;



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#define BLEND_MASK_COLORX_OFFSET       10
#define BLEND_MASK_COLORY_OFFSET       11
#define BLEND_MASK_COLORZ_OFFSET       12

#define BLEND_MASK_TEXID_OFFSET        13
#define BLEND_MASK_TEXMATRIXID_OFFSET  14
#define BLEND_MASK_FLAGS_OFFSET        15

#define BLEND_MASK_SAMPLER_OFFSET      20

enum BLEND_MASK_FLAGS { BLEND_MASK_FRESNEL = 1,
                        BLEND_MASK_FALOFF  = 2,
                        BLEND_MASK_REFLECTION_WEIGHT_IS_ONE = 4,
                        BLEND_MASK_EXTRUSION_STRONG         = 8,
                        BLEND_MASK_EXTRUSION_LUMINANCE      = 16 };

enum BLEND_TYPES_ENUM {
    BLEND_FRESNEL = 1,
    BLEND_FALOFF  = 2,
    BLEND_SIMPLE  = 3,
    BLEND_SIGMOID = 4,
  };

enum BLEND_FLAGS_ENUM {
  BLEND_INVERT_FALOFF = 1,

};

#define BLEND_MASK_MATERIAL1_OFFSET    16 // alpha = luminance(BLEND_MASK_COLOR);
#define BLEND_MASK_MATERIAL2_OFFSET    17 // res = BLEND_MASK_MATERIAL1*alpha + BLEND_MASK_MATERIAL2_OFFSET*(1.0f-alpha)

#define BLEND_MASK_FRESNEL_IOR         18

#define BLEND_MASK_FALOFF_OFFSET       19
#define BLEND_MASK_FALOFF_SIZE         20

#define BLEND_TYPE                     21
#define BLEND_SIGMOID_EXP              22

#define BLEND_FLAGS                    23


static inline float hermiteSplineEval(const float s, const float2 start, const float2 end, const float2 tangent1, const float2 tangent2)
{
  const float s2 = s*s;
  const float s3 = s2*s;

  const float h1 = 2.0f*s3 - 3.0f*s2 + 1.0f; // calculate basis function 1
  const float h2 = -2.0f*s3 + 3.0f*s2;       // calculate basis function 2
  const float h3 = s3 - 2.0f*s2 + s;         // calculate basis function 3
  const float h4 = s3 - s2;                  // calculate basis function 4

  //float2 res;
  //res.y = h1*start.y + h2*end.y + h3*tangent1.y + h4*tangent2.y;
  //res.x = h1*start.x + h2*end.x + h3*tangent1.x + h4*tangent2.x;

  return clamp(h1*start.y + h2*end.y + h3*tangent1.y + h4*tangent2.y, 0.0f, 1.0f);
}

static inline float hermiteSplineEvalT(float t, __global const float2* a_points, __global const float2* a_tangents, int a_numPoints)
{
  int pointStart = (int)(t*(float)(a_numPoints - 1)); 
  if (pointStart == a_numPoints - 1)
    pointStart--;
  const int pointEnd = pointStart + 1;

  const float tStart = (float)(pointStart) / (float)(a_numPoints - 1);
  const float tEnd   = (float)(pointEnd) / (float)(a_numPoints - 1);
  
  const float s = fabs(t - tStart) / (tEnd - tStart);

  return 1.0f - hermiteSplineEval(s, a_points[pointStart], a_points[pointEnd], a_tangents[pointStart], a_tangents[pointEnd]);
}

static inline float maxSigmoid(const float x, const float gamma)
{
  const float x2 = -5.0f + 10.0f*x;
  return 1.04f / (1.0f + exp(-gamma*x2)) - 0.02f;
}

static inline float myluminance(const float3 a_lum) { return dot(make_float3(0.35f, 0.51f, 0.14f), a_lum); }

static inline float blendMaskAlpha2(__global const PlainMaterial* pMat, 
                                    const float3 v, const float3 n, const float2 hitTexCoord, 
                                    __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  const int2   texId    = make_int2(as_int(pMat->data[BLEND_MASK_TEXID_OFFSET]), as_int(pMat->data[BLEND_MASK_TEXMATRIXID_OFFSET]));
  const float3 texColor = sample2DExt(texId.y, hitTexCoord, (__global const int4*)pMat, a_tex, a_globals, a_ptList);
  
  const float3 lum1     = clamp(texColor*make_float3(pMat->data[BLEND_MASK_COLORX_OFFSET], pMat->data[BLEND_MASK_COLORY_OFFSET], pMat->data[BLEND_MASK_COLORZ_OFFSET]), 0.0f, 1.0f);

  float lum;
  if ((as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_EXTRUSION_LUMINANCE) != 0)
    lum = myluminance(lum1); // dot(lum1, make_float3(0.3333f, 0.3334f, 0.3333f)); // myluminance(lum1);
  else
    lum = fmax(lum1.x, fmax(lum1.y, lum1.z));

  const float normAngle = fabs(dot(v, n));
  float faloff = 0.0f;

  if (as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_FALOFF)
  {
    const int faloffStart = as_int(pMat->data[BLEND_MASK_FALOFF_OFFSET]);
    const int faloffSize  = as_int(pMat->data[BLEND_MASK_FALOFF_SIZE]);
    const int numPoints   = faloffSize / 4;

    __global const float* points   = floatArraysPtr(a_globals) + faloffStart;
    __global const float* tangents = points + faloffSize / 2;

    float faloffParam = (as_int(pMat->data[BLEND_FLAGS]) & BLEND_INVERT_FALOFF) ? normAngle : 1.0f - normAngle;

    faloff = hermiteSplineEvalT(faloffParam, (__global const float2*)points, (__global const float2*)tangents, numPoints);
  }

  if (as_int(pMat->data[BLEND_TYPE]) == BLEND_SIGMOID)
    lum = maxSigmoid(lum, pMat->data[BLEND_SIGMOID_EXP]);

  if (as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_FALOFF) // --> faloff by normal angle
    return clamp(faloff, 0.0f, 1.0f);
  else if (as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_FRESNEL)
    return clamp(lum*fresnelReflectionCoeffMentalLike(normAngle, pMat->data[BLEND_MASK_FRESNEL_IOR]), 0.0f, 1.0f);
  else
    return clamp(lum, 0.0f, 1.0f);
}

static inline BRDFSelector blendSelectBRDF(__global const PlainMaterial* pMat, const float a_r3, 
                                           const float3 rayDir, const float3 hitNorm, const float2 hitTexCoord, const bool a_sampleReflectionOnly,
                                           __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  float alpha = blendMaskAlpha2(pMat, rayDir, hitNorm, hitTexCoord, a_globals, a_tex, a_ptList);

  BRDFSelector mat1, mat2;
  
  mat1.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL1_OFFSET]);
  mat2.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL2_OFFSET]);

  __global const PlainMaterial* pComponent1 = pMat + mat1.localOffs;

  // #TODO: try it without importance sampling! figure out BLEND_MASK_REFLECTION_WEIGHT_IS_ONE issue.
  //

  // mat2.w = 2.0f*(alpha);
  // mat1.w = 2.0f*(1.0f-alpha);
  // 
  // if ((as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_REFLECTION_WEIGHT_IS_ONE) && materialIsLeafBRDF(pComponent1))
  //   mat2.w *= alpha;
  // 
  // if (a_r3 <= 0.5f)
  //   return mat1;
  // else
  //   return mat2;

  mat1.w = 1.0f;
  mat2.w = 1.0f;
  
  if ((as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_REFLECTION_WEIGHT_IS_ONE) && materialIsLeafBRDF(pComponent1))
  {
    mat1.w = alpha;
    mat2.w = 1.0f;
  }
  
  if ((as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_FRESNEL) != 0 && a_sampleReflectionOnly) // effective sampling of highlights on glass when evaluate Direct Light
  {                                                                                                      // deterministic select
    mat1.w = alpha;                                                                                      // deterministic select, must use weight
    alpha  = 1.0f;                                                                                       // select reflection; never select refraction;
  }
  
  if (a_r3 <= alpha)
    return mat1;
  else
    return mat2;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define VOLUME_PERLIN_COLORX_OFFSET           10
#define VOLUME_PERLIN_COLORY_OFFSET           11
#define VOLUME_PERLIN_COLORZ_OFFSET           12

#define VOLUME_PERLIN_OCTAVES                 13
#define VOLUME_PERLIN_PERSISTENCE             14


static inline float3 volumePerlinGetColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[VOLUME_PERLIN_COLORX_OFFSET], a_pMat->data[VOLUME_PERLIN_COLORY_OFFSET], a_pMat->data[VOLUME_PERLIN_COLORZ_OFFSET]); }
static inline int    volumePerlinGetOctavesNum(__global const PlainMaterial* a_pMat) { return as_int(a_pMat->data[VOLUME_PERLIN_OCTAVES]);}
static inline float  volumePerlinGetPersistence(__global const PlainMaterial* a_pMat) { return a_pMat->data[VOLUME_PERLIN_PERSISTENCE];}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define SSS_ABSORPTIONX_OFFSET            10
#define SSS_ABSORPTIONY_OFFSET            11
#define SSS_ABSORPTIONZ_OFFSET            12

#define SSS_PHASE                         13
#define SSS_SCATTERING                    14
#define SSS_TRANSMISSION                  15

#define SSS_DIFFUSEX_OFFSET               16
#define SSS_DIFFUSEY_OFFSET               17
#define SSS_DIFFUSEZ_OFFSET               18
#define SSS_DENSITY                       19


static inline float3 sssGetAbsorption(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[SSS_ABSORPTIONX_OFFSET], a_pMat->data[SSS_ABSORPTIONY_OFFSET], a_pMat->data[SSS_ABSORPTIONZ_OFFSET]); }
static inline float3 sssGetDiffuseColor(__global const PlainMaterial* a_pMat) { return make_float3(a_pMat->data[SSS_DIFFUSEX_OFFSET], a_pMat->data[SSS_DIFFUSEY_OFFSET], a_pMat->data[SSS_DIFFUSEZ_OFFSET]); }
static inline float  sssGetPhase(__global const PlainMaterial* a_pMat) { return a_pMat->data[SSS_PHASE]; }
static inline float  sssGetDensity(__global const PlainMaterial* a_pMat) { return a_pMat->data[SSS_DENSITY]; }
static inline float  sssGetScattering(__global const PlainMaterial* a_pMat) { return a_pMat->data[SSS_SCATTERING]; }
static inline float  sssGetTransmission(__global const PlainMaterial* a_pMat) { return a_pMat->data[SSS_TRANSMISSION]; }



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline BRDFSelector materialRandomWalkBRDF(__global const PlainMaterial* a_pMat, __private float* a_rands, const float3 rayDir, const float3 hitNorm, const float2 hitTexCoord,
                                                  __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList, const bool a_reflOnly)
{
  BRDFSelector res, sel;

  res.localOffs = 0;
  res.w         = 1.0f;

  sel.localOffs = 0;
  sel.w         = 1.0f;

  __global const PlainMaterial* node = a_pMat;
  int i = 0;
  while (!materialIsLeafBRDF(node) && i < MMLT_FLOATS_PER_MLAYER)
  {
    const float rndVal = a_rands[MMLT_FLOATS_PER_SAMPLE + i];

    if (materialGetType(node) == PLAIN_MAT_CLASS_BLEND_MASK)
      sel = blendSelectBRDF(node, rndVal, rayDir, hitNorm, hitTexCoord, (a_reflOnly && (i==0)), a_globals, a_tex, a_ptList);

    res.w         = res.w*sel.w;
    res.localOffs = res.localOffs + sel.localOffs;
    node          = node          + sel.localOffs;
    i++;
  }

  return res;
}

static inline float3 materialNormalMapFetch(__global const PlainMaterial* pHitMaterial, const float2 a_texCoord, texture2d_t a_tex, 
                                            __global const EngineGlobals* a_globals, __private const ProcTextureList* a_ptList)
{
  const int materialFlags    = materialGetFlags(pHitMaterial);
  const int2 texIds          = materialGetNormalTex(pHitMaterial);
  const float3 normalFromTex = sample2DAuxExt(texIds, a_texCoord, (__global const int4*)pHitMaterial, a_tex, a_globals, a_ptList);

  float3 normalTS = make_float3(2.0f * normalFromTex.x - 1.0f, 2.0f * normalFromTex.y - 1.0f, normalFromTex.z);
  //float3 normalTS = make_float3(0, 0, 1);

  if (materialFlags & PLAIN_MATERIAL_INVERT_NMAP_Y)
    normalTS.y *= (-1.0f);

  if (materialFlags & PLAIN_MATERIAL_INVERT_NMAP_X)
    normalTS.x *= (-1.0f);

  if (materialFlags & PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY)
  {
    float temp = normalTS.x;
    normalTS.x = normalTS.y;
    normalTS.y = temp;
  }

  return normalize(normalTS);
}

static inline float3 BumpMapping(const float3 tangent, const float3 bitangent, const float3 normal, const float2 a_texCoord,
                                 __global const PlainMaterial* pHitMaterial, __global const EngineGlobals* a_globals, texture2d_t a_texNormal, 
                                 __private const ProcTextureList* a_ptList)
{
  const float3   normalTS         = materialNormalMapFetch(pHitMaterial, a_texCoord, a_texNormal, a_globals, a_ptList);
  const float3x3 tangentTransform = make_float3x3(tangent, bitangent, normal);

  return  normalize(mul3x3x3(inverse(tangentTransform), normalTS));
}

static inline void MaterialLeafSampleAndEvalBRDF(__global const PlainMaterial* pMat, __private const SurfaceHit* pSurfHit, const float3 ray_dir, const float3 rands, const float3 a_shadow, const bool a_isFwdDir, 
                                                 __global const EngineGlobals* a_globals, texture2d_t a_tex, texture2d_t a_texNormal, __private const ProcTextureList* a_ptList,
                                                 __private MatSample* a_out)
{
  float3 hitNorm = pSurfHit->normal;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  const bool hasNormalMap = (materialGetNormalTex(pMat).x != INVALID_TEXTURE);

  if (hasNormalMap)
  {
    const bool isGlass    = (materialGetType(pMat) == PLAIN_MAT_CLASS_GLASS);                                    // dirty hack for glass; #TODO: fix that more accurate
    const float3 flatNorm = pSurfHit->hfi && !isGlass ? (-1.0f)*pSurfHit->flatNormal : pSurfHit->flatNormal;     // dirty hack for glass; #TODO: fix that more accurate
    hitNorm               = BumpMapping(pSurfHit->tangent, pSurfHit->biTangent, flatNorm, pSurfHit->texCoord, pMat, a_globals, a_texNormal, a_ptList);
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  a_out->color        = make_float3(0.0f, 0.0f, 0.0f);
  a_out->direction    = make_float3(0.0f, 1.0f, 0.0f);
  a_out->pdf          = 1.0f;
  a_out->flags        = 0;

  switch (materialGetType(pMat))
  {
  case PLAIN_MAT_CLASS_PHONG_SPECULAR: 
    PhongSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, a_globals, a_tex, a_ptList,
                           a_out);
    break;

  case PLAIN_MAT_CLASS_BLINN_SPECULAR: 
    BlinnSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, pSurfHit->tangent, pSurfHit->biTangent, a_globals, a_tex, a_ptList,
                           a_out);
    break;
  
  case PLAIN_MAT_CLASS_BECKMANN: 
    BeckmannSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, pSurfHit->tangent, pSurfHit->biTangent, a_globals, a_tex, a_ptList,
                              a_out);
    break;
  
  case PLAIN_MAT_CLASS_TRGGX:
    TRGGXSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, pSurfHit->tangent, pSurfHit->biTangent, a_globals, a_tex, a_ptList,
                           a_out);
    break;

  case PLAIN_MAT_CLASS_GGX: 
    GGXSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, a_globals, a_tex, a_ptList,
                         a_out);
    break;

  case PLAIN_MAT_CLASS_PERFECT_MIRROR: 
    MirrorSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, a_globals, a_tex, a_ptList,
                            a_out);
    break;

  case PLAIN_MAT_CLASS_THIN_GLASS: 
    ThinglassSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, a_globals, a_tex, a_ptList,
                               a_out);
    break;
  case PLAIN_MAT_CLASS_GLASS: 
    GlassSampleAndEvalBRDF(pMat, rands, ray_dir, hitNorm, pSurfHit->texCoord, pSurfHit->hfi, a_globals, a_tex, a_ptList, a_isFwdDir,
                           a_out);
    break;

  case PLAIN_MAT_CLASS_TRANSLUCENT   : 
    TranslucentSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, a_globals, a_tex, a_ptList,
                                 a_out);
    break;

  case PLAIN_MAT_CLASS_OREN_NAYAR    : 
    OrennayarSampleAndEvalBRDF(pMat, rands.x, rands.y, ray_dir, hitNorm, pSurfHit->texCoord, a_globals, a_tex, a_ptList,
                               a_out);
    break;

  case PLAIN_MAT_CLASS_LAMBERT       : 
    LambertSampleAndEvalBRDF(pMat, rands.x, rands.y, hitNorm, pSurfHit->texCoord, pSurfHit->texCoordCamProj, a_globals, a_tex, a_ptList,
                             a_out);
    break;

  case PLAIN_MAT_CLASS_SHADOW_MATTE  : 
    ShadowmatteSampleAndEvalBRDF(pMat, ray_dir, hitNorm, a_shadow,
                                 a_out);
    break;
  };

  // BSDF is multiplied (outside) by cosThetaOut1.
  // When normal map is enables this becames wrong because normal is changed;
  // First : return cosThetaOut in sam;
  // Second: apply cos(theta2)/cos(theta1) to cos(theta1) to get cos(theta2)
  //
  if (hasNormalMap)
  {
    const float cosThetaOut1 = fabs(dot(a_out->direction, pSurfHit->normal));
    const float cosThetaOut2 = fabs(dot(a_out->direction, hitNorm));
    a_out->color *= (cosThetaOut2 / fmax(cosThetaOut1, DEPSILON2));
  }

  if (a_out->pdf <= 0.0f)                // can happen because of shitty geometry and material models
    a_out->color = make_float3(0, 0, 0); //

}

/**
\brief Sample and eval layered material (with blends). Store result in a_out.
\param  
\param  

Sample and eval layered material (with blends). Store result in a_out.

*/
static inline void MaterialSampleAndEvalBxDF(__global const PlainMaterial* pMat, __private float a_rands[MMLT_FLOATS_PER_BOUNCE], 
                                             __private const SurfaceHit* pSurfHit, const float3 a_rayDir, const float3 a_shadow, const uint rayFlags, const bool a_isFwdDir,
                                             __global const EngineGlobals* a_globals, texture2d_t a_tex, texture2d_t a_texNormal, __private const ProcTextureList* a_ptList,
                                             __private MatSample* a_out, __private int* pLocalOffset)
{
  const uint otherRayFlags        = unpackRayFlags(rayFlags);
  const bool canSampleReflOnly    = (materialGetFlags(pMat) & PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY) != 0;
  const bool sampleReflectionOnly = ((otherRayFlags & RAY_GRAMMAR_DIRECT_LIGHT) != 0) && canSampleReflOnly; 
 
  const BRDFSelector mixSelector  = materialRandomWalkBRDF(pMat, a_rands, a_rayDir, pSurfHit->normal, pSurfHit->texCoord, 
                                                           a_globals, a_tex, a_ptList, sampleReflectionOnly); 

  __global const PlainMaterial* pMatLeaf = pMat + mixSelector.localOffs;
  (*pLocalOffset)                        = mixSelector.localOffs;
  
  MaterialLeafSampleAndEvalBRDF(pMatLeaf, pSurfHit, a_rayDir, make_float3(a_rands[0], a_rands[1], a_rands[2]), a_shadow, a_isFwdDir,
                                a_globals, a_tex, a_texNormal, a_ptList,
                                a_out);

  a_out->color *= 1.0f/fmax(mixSelector.w, 0.015625f); // it is essential to do this right here. Don't put this in PDF!

  if (materialIsSkyPortal(pMatLeaf) && isEyeRay(rayFlags))
  {
    a_out->color = make_float3(1, 1, 1);
    a_out->pdf   = 1.0f;
  }
}



typedef struct BxDFResultT
{
  float3 brdf;
  float  pdfFwd;

  float3 btdf;
  float  pdfRev;

  bool   diffuse;

} BxDFResult;


/**
\brief  Compute fix coeff to make Light transport symmetric with LT and smooth shading normal. See Veach thesis "5.3 Non-symmetry due to shading normals" 
\param  toLightWo - direction to light  (where light comes from)
\param  toCamWi   - direction to camera (where light gpes to)
\param  shadeNorm - smooth shade normal
\param  geomNorm  - flat geometry normal
\param  maxVal    - maximum coeff value. 
\return shade normal coeffitien fix for LT.

*/
static inline float adjointBsdfShadeNormalFix(const float3 toLightWo, const float3 toCamWi, const float3 shadeNorm, float3 geomNorm, const float maxVal)
{
  if (dot(shadeNorm, geomNorm) < 0)
    geomNorm = (-1.0f)*geomNorm;

  if (1.0f - fabs(dot(shadeNorm, geomNorm)) <= 1e-6f)
    return 1.0f;
  else if (dot(toCamWi, geomNorm)*dot(toCamWi, shadeNorm) <= 0 || dot(toLightWo, geomNorm)*dot(toLightWo, shadeNorm) <= 0)
    return 1.0f;

  const float k1 = dot(toLightWo, shadeNorm);
  const float k2 = dot(toCamWi, geomNorm);
  const float k3 = dot(toLightWo, geomNorm);
  const float k4 = dot(toCamWi, shadeNorm);

  const float res = (k1*k2)/fmax(k3*k4, DEPSILON2);
  return fmin(fmax(res, 0.1f), maxVal);
}

/**
\brief  Evaluate BRDF and BTDF values and also fwd. and rev. pdfs
\param  pMat          - surface material
\param  sc            - all other surface info
\param  a_fwdDir      - a flag if we use light tracing 
\return RDF and BTDF values and also fwd. and rev. pdfs

*/
static inline BxDFResult materialLeafEval(__global const PlainMaterial* pMat, __private const ShadeContext* sc, const int a_evalFlags,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, texture2d_t a_texNormal, __private const ProcTextureList* a_ptList)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// cosThetaFix for normalmap
  const bool a_fwdDir = ((a_evalFlags & EVAL_FLAG_FWD_DIR) != 0);

  float3 n       = sc->n;
  float cosMult  = 1.0f;
  float cosMult2 = 1.0f;
  if (materialGetNormalTex(pMat).x != INVALID_TEXTURE)
  {
    n = BumpMapping(sc->tg, sc->bn, sc->fn, sc->tc, pMat, a_globals, a_texNormal, a_ptList);
   
    const float3 lDir     = a_fwdDir ? sc->v : sc->l; // magic swap for normal mapping works with LT; don't ask me why this works, i don't know!
    const float  clampVal = a_fwdDir ? 0.15f : 1e-6f; // fuck reversibility
    const float  ltFuckUp = dot(sc->l, sc->fn);       // fuck reversibility

    const float cosThetaOut1 = fmax(dot(lDir, sc->n), 0.0f);
    const float cosThetaOut2 = fmax(dot(lDir, n),     0.0f);
    cosMult                  = (cosThetaOut2 / fmax(cosThetaOut1, clampVal));
    if (cosThetaOut1 <= 0.0f)
      cosMult = 0.0f;

    const float cosThetaOut3 = fmax(-dot(lDir, sc->n), 0.0f);
    const float cosThetaOut4 = fmax(-dot(lDir, n), 0.0f);
    cosMult2 = (cosThetaOut4 / fmax(cosThetaOut3, clampVal));
    if (cosThetaOut3 <= 0.0f)
      cosMult2 = 0.0f;

    if (a_fwdDir && ltFuckUp <= 0.0f)
    {
      cosMult  = 0.0f;
      cosMult2 = 0.0f;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// cosThetaFix for normalmap

  BxDFResult res;

  res.brdf    = make_float3(0, 0, 0);
  res.btdf    = make_float3(0, 0, 0);
  res.pdfFwd  = 0.0f;
  res.pdfRev  = 0.0f;
  res.diffuse = false;

  switch (materialGetType(pMat))
  {
  case PLAIN_MAT_CLASS_PHONG_SPECULAR: 
    res.brdf    = phongEvalBxDF(pMat, sc->l, sc->v, n, sc->tc, a_evalFlags, a_globals, a_tex, a_ptList)*cosMult;
    res.pdfFwd  = phongEvalPDF (pMat, sc->l, sc->v, n, sc->tc,              a_globals, a_tex, a_ptList);
    res.pdfRev  = phongEvalPDF (pMat, sc->v, sc->l, n, sc->tc,              a_globals, a_tex, a_ptList);
    break;
  case PLAIN_MAT_CLASS_BLINN_SPECULAR: 
    res.brdf    = blinnEvalBxDF(pMat, sc->l, sc->v, n, sc->tc, a_globals, a_tex, a_ptList)*cosMult;
    res.pdfFwd  = blinnEvalPDF (pMat, sc->l, sc->v, n, sc->tc, a_globals, a_tex, a_ptList);
    res.pdfRev  = blinnEvalPDF (pMat, sc->v, sc->l, n, sc->tc, a_globals, a_tex, a_ptList);
    break;
  case PLAIN_MAT_CLASS_BECKMANN:
    res.brdf    = beckmannEvalBxDF(pMat, sc->l, sc->v, n, sc->tg, sc->bn, sc->tc, a_globals, a_tex, a_ptList)*cosMult;
    res.pdfFwd  = beckmannEvalPDF (pMat, sc->l, sc->v, n, sc->tg, sc->bn, sc->tc, a_globals, a_tex, a_ptList);
    res.pdfRev  = beckmannEvalPDF (pMat, sc->v, sc->l, n, sc->tg, sc->bn, sc->tc, a_globals, a_tex, a_ptList);
    break;
  case PLAIN_MAT_CLASS_TRGGX:
    res.brdf   = trggxEvalBxDF(pMat, sc->l, sc->v, n, sc->tg, sc->bn, sc->tc, a_globals, a_tex, a_ptList)*cosMult;
    res.pdfFwd = trggxEvalPDF (pMat, sc->l, sc->v, n, sc->tg, sc->bn, sc->tc, a_globals, a_tex, a_ptList);
    res.pdfRev = trggxEvalPDF (pMat, sc->v, sc->l, n, sc->tg, sc->bn, sc->tc, a_globals, a_tex, a_ptList);
    break;
  case PLAIN_MAT_CLASS_GGX:  
    res.brdf    = ggxEvalBxDF(pMat, sc->l, sc->v, n, sc->tc, a_evalFlags, a_globals, a_tex, a_ptList)*cosMult;
    res.pdfFwd  = ggxEvalPDF (pMat, sc->l, sc->v, n, sc->tc,              a_globals, a_tex, a_ptList);
    res.pdfRev  = ggxEvalPDF (pMat, sc->v, sc->l, n, sc->tc,              a_globals, a_tex, a_ptList);
    break;   
  case PLAIN_MAT_CLASS_PERFECT_MIRROR: 
    res.brdf   = mirrorEvalBxDF(pMat, sc->l, sc->v, n)*cosMult;
    res.pdfFwd = mirrorEvalPDF (pMat, sc->l, sc->v, n);
    res.pdfRev = mirrorEvalPDF (pMat, sc->v, sc->l, n);
    break;
  case PLAIN_MAT_CLASS_THIN_GLASS: 
    res.brdf   = thinglassEvalBxDF(pMat, sc->l, sc->v, n)*cosMult;
    res.pdfFwd = thinglassEvalPDF (pMat, sc->l, sc->v, n);
    res.pdfRev = thinglassEvalPDF (pMat, sc->v, sc->l, n);
    break;
  case PLAIN_MAT_CLASS_GLASS:  
    res.brdf   = glassEvalBxDF(pMat, sc->l, sc->v, n)*cosMult;
    res.pdfFwd = glassEvalPDF (pMat, sc->l, sc->v, n);
    res.pdfRev = glassEvalPDF (pMat, sc->v, sc->l, n);
    break;
  case PLAIN_MAT_CLASS_TRANSLUCENT:
    res.btdf    = translucentEvalBxDF(pMat, sc->l, sc->v, n, sc->tc, a_globals, a_tex, a_ptList)*cosMult2;
    res.pdfFwd  = translucentEvalPDF (pMat, sc->l, sc->v, n);
    res.pdfRev  = translucentEvalPDF (pMat, sc->v, sc->l, n);
    res.diffuse = true;
    break;
  case PLAIN_MAT_CLASS_SHADOW_MATTE: 
    res.brdf   = shadowmatteEvalBxDF(pMat, sc->l, n, sc->tc)*cosMult;
    res.pdfFwd = shadowmatteEvalPDF (pMat, sc->l, sc->v, n);
    res.pdfRev = shadowmatteEvalPDF (pMat, sc->v, sc->l, n);
    break;
  case PLAIN_MAT_CLASS_OREN_NAYAR: 
    res.brdf    = orennayarEvalBxDF(pMat, sc->l, sc->v, n, sc->tc, a_globals, a_tex, a_ptList)*cosMult;
    res.pdfFwd  = orennayarEvalPDF (pMat, sc->l, sc->v, n);
    res.pdfRev  = orennayarEvalPDF (pMat, sc->v, sc->l, n);
    res.diffuse = true;
    break;
  case PLAIN_MAT_CLASS_LAMBERT:  
    res.brdf    = lambertEvalBxDF(pMat, sc->tc, sc->tccp, a_globals, a_tex, a_ptList)*cosMult;
    res.pdfFwd  = lambertEvalPDF (pMat, sc->l, n);
    res.pdfRev  = lambertEvalPDF (pMat, sc->v, n);
    res.diffuse = true;
    break;
  };
  
  // Veach phd thesis 5.3.2. The adjoint BSDF for shading normals.
  //
  if (a_fwdDir) 
  {
    const float maxVal2   = res.diffuse ? 20.0f : 2.0f;
    const float smoothFix = adjointBsdfShadeNormalFix(sc->v, sc->l, sc->n, sc->fn, maxVal2); // use old sc->n instead of n here to exclude normal map influence
    res.brdf *= smoothFix;
    //res.btdf *= smoothFix;
  }

  return res;
}


static inline BxDFResult materialEval(__global const PlainMaterial* a_pMat, __private const ShadeContext* sc, const int a_evalFlags,
                                      __global const EngineGlobals* a_globals, texture2d_t a_tex, texture2d_t a_texNormal, __private const ProcTextureList* a_ptList)
{
  BxDFResult val;
  val.brdf    = make_float3(0, 0, 0);
  val.btdf    = make_float3(0, 0, 0);
  val.pdfFwd  = 0.0f;
  val.pdfRev  = 0.0f;
  val.diffuse = true;

  const bool disableCaustics = ((a_evalFlags & EVAL_FLAG_DISABLE_CAUSTICS) != 0);

  float2 stack[MIX_TREE_MAX_DEEP];
  int top = 0;

  int   currGlobalOffset = 0;
  float currW            = 1.0f;

  do
  {
    if (top > 0)
    {
      top--;
      float2 poped     = stack[top];
      currGlobalOffset = as_int(poped.y);
      currW            = poped.x;
    }

    __global const PlainMaterial* pMat = a_pMat + currGlobalOffset;

    if (materialGetType(pMat) == PLAIN_MAT_CLASS_BLEND_MASK)
    {
      BRDFSelector mat1, mat2;

      const float alpha = blendMaskAlpha2(pMat, sc->v, sc->n, sc->tc, a_globals, a_tex, a_ptList);

      mat1.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL1_OFFSET]);
      mat1.w         = alpha;

      mat2.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL2_OFFSET]);
      mat2.w         = 1.0f - alpha;

      __global const PlainMaterial* pComponent1 = pMat + mat1.localOffs;
      //__global const PlainMaterial* pComponent2 = pMat + mat2.localOffs;

      if ((as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_REFLECTION_WEIGHT_IS_ONE) && materialIsLeafBRDF(pComponent1))
        mat1.w = 1.0f;

      if (top < MIX_TREE_MAX_DEEP)
      {
        stack[top] = make_float2(currW*mat1.w, as_float(currGlobalOffset + mat1.localOffs));
        top++;
      }

      if (top < MIX_TREE_MAX_DEEP)
      {
        stack[top] = make_float2(currW*mat2.w, as_float(currGlobalOffset + mat2.localOffs));
        top++;
      }

    }
    else  if (!(disableCaustics && materialCastCaustics(pMat)))
    {
      const BxDFResult bxdfAndPdf = materialLeafEval(pMat, sc, a_evalFlags, a_globals, a_tex, a_texNormal, a_ptList);
      val.brdf   += currW*bxdfAndPdf.brdf;
      val.btdf   += currW*bxdfAndPdf.btdf;
      val.pdfFwd += currW*bxdfAndPdf.pdfFwd;
      val.pdfRev += currW*bxdfAndPdf.pdfRev;
      val.diffuse = val.diffuse && bxdfAndPdf.diffuse;
    }

  } while (top > 0);

  return val;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///

static inline float3 materialLeafEvalTransparency(__global const PlainMaterial* pMat, 
                                                  const float3 l, const float3 n, const float2 hitTexCoord,
                                                  __global const EngineGlobals* a_globals, texture2d_t a_tex)
{
  int2 texIds;
  texIds.x = texIds.y = 0;

  if (materialGetType(pMat) == PLAIN_MAT_CLASS_THIN_GLASS)
    texIds = thinglassGetTex(pMat);
  else if (materialGetType(pMat) == PLAIN_MAT_CLASS_GLASS)
    texIds = glassGetTex(pMat);

  const float3 texColor = sample2D(texIds.y, hitTexCoord, (__global const int4*)pMat, a_tex, a_globals);
  
  float3 transparency = make_float3(0.0f, 0.0f, 0.0f);

  if (materialGetType(pMat) == PLAIN_MAT_CLASS_THIN_GLASS)
    transparency = thinglassGetColor(pMat)*texColor;
  else if (materialGetType(pMat) == PLAIN_MAT_CLASS_GLASS)
    transparency = glassGetColor(pMat)*texColor;
  else if (materialGetType(pMat) == PLAIN_MAT_CLASS_SHADOW_MATTE)
    transparency = make_float3(1, 1, 1);

  return clamp(transparency, 0.0f, 1.0f);
}


static inline bool materialIsTransparent(__global const PlainMaterial* a_pMat)
{
  return (materialGetType(a_pMat) == PLAIN_MAT_CLASS_GLASS) || (materialGetType(a_pMat) == PLAIN_MAT_CLASS_THIN_GLASS) || (materialGetType(a_pMat) == PLAIN_MAT_CLASS_SHADOW_MATTE);
}

static inline bool materialIsThinGlass(__global const PlainMaterial* a_pMat)
{
  return (materialGetType(a_pMat) == PLAIN_MAT_CLASS_SHADOW_MATTE) || (materialGetType(a_pMat) == PLAIN_MAT_CLASS_THIN_GLASS);
}


static inline bool materialIsReflective(__global const PlainMaterial* a_pMat)
{
  return (materialGetType(a_pMat) != PLAIN_MAT_CLASS_GLASS) && (materialGetType(a_pMat) != PLAIN_MAT_CLASS_THIN_GLASS) && (materialGetType(a_pMat) != PLAIN_MAT_CLASS_TRANSLUCENT);
}


static inline bool materialLeafHasBeerAttenuation(__global const PlainMaterial* a_pMat)
{
  return (materialGetType(a_pMat) == PLAIN_MAT_CLASS_GLASS);
}

static inline float4 materialLeafGetFog(__global const PlainMaterial* a_pMat)
{
  if (materialLeafHasBeerAttenuation(a_pMat))
    return glassGetFog(a_pMat);
  else
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

typedef struct TransAndFogT2
{
  float4 wfogData;
  float3 wtrnData;
  int    offset;

}TransparencyAndFog2;


typedef struct TransAndFogT
{
  float4 fog;
  float3 transparency;

}TransparencyAndFog;


static inline TransparencyAndFog materialEvalTransparencyAndFog(__global const PlainMaterial* a_pMat, 
                                                                const float3 l, const float3 n, const float2 hitTexCoord,
                                                                __global const EngineGlobals* a_globals, texture2d_t a_tex, 
                                                                __private const ProcTextureList* a_ptList)
{
  float4 val  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float3 valT = make_float3(0.0f, 0.0f, 0.0f);

  TransparencyAndFog2 tstack[MIX_TREE_MAX_DEEP];
  int top = 0;

  int currGlobalOffset = 0;
  float4 currW = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  float3 currT = make_float3(1.0f, 1.0f, 1.0f);

  do
  {
    if (top > 0)
    {
      top--;
      currGlobalOffset = tstack[top].offset;
      currW            = tstack[top].wfogData;
      currT            = tstack[top].wtrnData;
    }

    __global const PlainMaterial* pMat = a_pMat + currGlobalOffset;

    if (materialGetType(pMat) == PLAIN_MAT_CLASS_BLEND_MASK)
    {
      BRDFSelector mat1, mat2;

      const float alpha = blendMaskAlpha2(pMat, l, n, hitTexCoord, a_globals, a_tex, a_ptList);

      mat1.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL1_OFFSET]);
      mat1.w = alpha;

      mat2.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL2_OFFSET]);
      mat2.w = 1.0f - alpha;

      __global const PlainMaterial* pComponent1 = pMat + mat1.localOffs;
      //__global const PlainMaterial* pComponent2 = pMat + mat2.localOffs;

      if ((as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_REFLECTION_WEIGHT_IS_ONE) && materialIsLeafBRDF(pComponent1))
        mat1.w = 1.0f;

      if (top < MIX_TREE_MAX_DEEP)
      {
        tstack[top].offset   = currGlobalOffset + mat1.localOffs;
        tstack[top].wfogData = currW*make_float4(mat1.w, mat1.w, mat1.w, alpha);
        tstack[top].wtrnData = currT*mat1.w;
        top++;
      }

      if (top < MIX_TREE_MAX_DEEP)
      {
        tstack[top].offset   = currGlobalOffset + mat2.localOffs;
        tstack[top].wfogData = currW*make_float4(mat2.w, mat2.w, mat2.w, 1.0f - alpha);
        tstack[top].wtrnData = currT*mat2.w;
        top++;
      }

    }
    else
    {
      val  += currW*materialLeafGetFog(pMat);
      valT += currT*materialLeafEvalTransparency(pMat, l, n, hitTexCoord, a_globals, a_tex);
    }

  } while (top > 0);


  TransparencyAndFog res;
  res.transparency = valT;
  res.fog          = val;

  return res;
}


static inline float3 transparencyAttenuation(const float3 fogColor, const float fogMult, const float t)
{
  float3 transparency;

  transparency.x = (1.0f / exp(fmax(1.0f - fogColor.x, 0.0f)*fogMult*t));
  transparency.y = (1.0f / exp(fmax(1.0f - fogColor.y, 0.0f)*fogMult*t));
  transparency.z = (1.0f / exp(fmax(1.0f - fogColor.z, 0.0f)*fogMult*t));

  transparency.x = fmax(fmin(transparency.x, 1.0f), 0.0f);
  transparency.y = fmax(fmin(transparency.y, 1.0f), 0.0f);
  transparency.z = fmax(fmin(transparency.z, 1.0f), 0.0f);

  return transparency;
}


static inline float3 attenuationStep(__global const PlainMaterial* pHitMaterial, const float dist, const bool gotOutside, __global float4* a_state)
{
	float3 fogAtten = make_float3(1.0f, 1.0f, 1.0f);
	{
		const float4 fogDataU = (*a_state);
		const bool   accountAttenuation = (fogDataU.w != 0.0f);

		if (accountAttenuation)
		{
			float3 fogColor = to_float3(fogDataU);
			float  fogMult = fogDataU.w;

			fogAtten = transparencyAttenuation(fogColor, fogMult, dist);
		}

		if (materialLeafHasBeerAttenuation(pHitMaterial))
		{
			if (accountAttenuation && gotOutside)
				(*a_state) = make_float4(0, 0, 0, 0);
			else
				(*a_state) = materialLeafGetFog(pHitMaterial);
		}
	}

	return fogAtten;
}

// extract diffuse component for ML filter, IC and photons
//

static inline float3 materialLeafEvalDiffuse(__global const PlainMaterial* a_pMat, const float3 l, const float3 n, const float2 hitTexCoord,
                                             __global const EngineGlobals* a_globals, __global const int4* in_texStorage, __private const ProcTextureList* a_ptList)
{
  if (materialGetType(a_pMat) == PLAIN_MAT_CLASS_LAMBERT)
  {
    const int2   texIds   = lambertGetDiffuseTex(a_pMat);
    const float3 texColor = sample2DExt(texIds.y, hitTexCoord, (__global const int4*)a_pMat, in_texStorage, a_globals, a_ptList);
    return texColor*lambertGetDiffuseColor(a_pMat);
  }
  else if (materialGetType(a_pMat) == PLAIN_MAT_CLASS_OREN_NAYAR)
  {
    const int2   texIds   = orennayarGetDiffuseTex(a_pMat);
    const float3 texColor = sample2DExt(texIds.y, hitTexCoord, (__global const int4*)a_pMat, in_texStorage, a_globals, a_ptList);
    return texColor*orennayarGetDiffuseColor(a_pMat);
  }
  else
    return make_float3(0, 0, 0);
}


static inline float3 materialEvalDiffuse(__global const PlainMaterial* a_pMat,
                                         const float3 l, const float3 n, const float2 hitTexCoord,
                                         __global const EngineGlobals* a_globals, __global const int4* in_texStorage, __private const ProcTextureList* a_ptList)
{
  float3 val = make_float3(0.0f, 0.0f, 0.0f);

  float2 stack[MIX_TREE_MAX_DEEP];
  int top = 0;

  int currGlobalOffset = 0;
  float currW          = 1.0f;

  do
  {
    if (top > 0)
    {
      top--;
      float2 poped     = stack[top];
      currGlobalOffset = as_int(poped.y);
      currW            = poped.x;
    }

    __global const PlainMaterial* pMat = a_pMat + currGlobalOffset;

    if (materialGetType(pMat) == PLAIN_MAT_CLASS_BLEND_MASK)
    {
      BRDFSelector mat1, mat2;

      const float alpha = blendMaskAlpha2(pMat, l, n, hitTexCoord, a_globals, in_texStorage, a_ptList);

      mat1.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL1_OFFSET]);
      mat1.w         = alpha;

      mat2.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL2_OFFSET]);
      mat2.w         = 1.0f - alpha;

      __global const PlainMaterial* pComponent1 = pMat + mat1.localOffs;
      //__global const PlainMaterial* pComponent2 = pMat + mat2.localOffs;

      if ((as_int(pMat->data[BLEND_MASK_FLAGS_OFFSET]) & BLEND_MASK_REFLECTION_WEIGHT_IS_ONE) && materialIsLeafBRDF(pComponent1))
        mat1.w = 1.0f;

      if (top < MIX_TREE_MAX_DEEP)
      {
        stack[top] = make_float2(currW*mat1.w, as_float(currGlobalOffset + mat1.localOffs));
        top++;
      }

      if (top < MIX_TREE_MAX_DEEP)
      {
        stack[top] = make_float2(currW*mat2.w, as_float(currGlobalOffset + mat2.localOffs));
        top++;
      }

    }
    else
      val += currW*materialLeafEvalDiffuse(pMat, l, n, hitTexCoord, a_globals, in_texStorage, a_ptList);


  } while (top > 0);


  return val;
}

static inline float3 materialEvalEmission(__global const PlainMaterial* a_pMat, const float3 v, const float3 n, const float2 a_texCoord,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __private const ProcTextureList* a_ptList)
{
  
  float3 val = make_float3(0.0f, 0.0f, 0.0f);

  float2 stack[MIX_TREE_MAX_DEEP];
  int top = 0;

  int currGlobalOffset = 0;
  float currW = 1.0f;

  do
  {
    if (top > 0)
    {
      top--;
      float2 poped     = stack[top];
      currGlobalOffset = as_int(poped.y);
      currW            = poped.x;
    }

    __global const PlainMaterial* pMat = a_pMat + currGlobalOffset;

    if (materialGetType(pMat) == PLAIN_MAT_CLASS_BLEND_MASK) // && (materialGetFlags(pMat) & PLAIN_MATERIAL_SURFACE_BLEND)
    {
      BRDFSelector mat1, mat2;

      const float alpha = blendMaskAlpha2(pMat, v, n, a_texCoord, a_globals, a_tex, a_ptList);

      mat1.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL1_OFFSET]);
      mat1.w         = alpha;

      mat2.localOffs = as_int(pMat->data[BLEND_MASK_MATERIAL2_OFFSET]);
      mat2.w         = 1.0f - alpha;

      //__global const PlainMaterial* pComponent1 = pMat + mat1.localOffs;
      //__global const PlainMaterial* pComponent2 = pMat + mat2.localOffs;

      if (top < MIX_TREE_MAX_DEEP)
      {
        stack[top] = make_float2(currW*mat1.w, as_float(currGlobalOffset + mat1.localOffs));
        top++;
      }

      if (top < MIX_TREE_MAX_DEEP)
      {
        stack[top] = make_float2(currW*mat2.w, as_float(currGlobalOffset + mat2.localOffs));
        top++;
      }

    }
    //else //#NOTE: BLEND NODES CAN HAVE EMISSIVE HEADERS!, so we can not skip blend itself!!! 
    val += currW*materialLeafEvalEmission(pMat, a_texCoord, a_globals, a_tex, a_ptList);

  } while (top > 0);


  return val;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// static inline float3 LinearSearch(float3 startPoint, float3 t, float4 texTransform, int texMatrixId, bool invertHeight, texture2d_t a_tex, __global const float4* a_matrices)
// {
//   float3 currPoint = startPoint;
//   float texHeight  = 0.0f;
// 
//   int i = 0;
// 
//   while (currPoint.z > texHeight && i <= 127)
//   {
//     float2 trnWithM  = texCoordMatrixTransform(make_float2(currPoint.x, currPoint.y), texMatrixId, a_matrices);
//     float2 trnWithMT = texCoordTableTransform(trnWithM, texTransform);
//     
//     float4 texColor4 = make_float4(0, 0, 0, 0);
// 
//     #ifdef OCL_COMPILER
//    
//       const sampler_t SMPL_PREF = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
//       texColor4 = read_imagef(a_tex, SMPL_PREF, trnWithMT);
//     
//     #else
//       #ifdef __CUDACC__ 
//         texColor4 = tex2D(normalmapTexture, trnWithMT.x, trnWithMT.y);
//       #endif
//     #endif
// 
//     texHeight = texColor4.w;
//     if(invertHeight)
//       texHeight = 1.0f - texHeight;
// 
//     currPoint += t; 
//     i++;
//   }
// 
//   return currPoint;
// }



// static inline float3 SimpleBinarySearch(float3 startPoint, float3 t, float4 texTransform, int texMatrixId, bool invertHeight, texture2d_t a_tex, __global const float4* a_matrices)
// {
//   float3 currPoint = startPoint + t;
//   float3 prevPoint = startPoint - 2 * t;
// 
//   float  texHeight = 0.0f;
//   float  delta = 1.0f;
// 
//   int i = 0;
// 
//   while (i < 16 && fabs(delta) > 1e-4f)
//   {
//     float3 newPoint  = 0.5f*(currPoint + prevPoint);
//     float2 trnWithM  = texCoordMatrixTransform(make_float2(newPoint.x, newPoint.y), texMatrixId, a_matrices);
//     float2 trnWithMT = texCoordTableTransform(trnWithM, texTransform);
// 
//     float4 texColor4;
// 
//     #ifdef OCL_COMPILER
//    
//       const sampler_t SMPL_PREF = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
//       texColor4 = read_imagef(a_tex, SMPL_PREF, trnWithMT);
//       
//     #else
//       #ifdef __CUDACC__ 
//         texColor4 = tex2D(normalmapTexture, trnWithMT.x, trnWithMT.y);
//       #endif
//     #endif
// 
//     texHeight = texColor4.w;
//     if (invertHeight)
//       texHeight = 1.0f - texHeight;
// 
//     delta = newPoint.z - texHeight;
//     prevPoint = (delta >= 0.0f) ? newPoint : prevPoint;
//     currPoint = (delta >= 0.0f) ? currPoint : newPoint;
// 
//     i++;
//   }
// 
//   if (i >= 16)
//     return (startPoint - 0.02f*t);
//   else
//     return (delta >= 0.0f) ? prevPoint : currPoint;
// }
// 
// 
// static inline void PalallaxOcclusionMapping(float3 ViewWS, float3 tangent, float3 bitangent, float3 a_flatNorm, __global const PlainMaterial* pHitMaterial, 
//                                       __global const EngineGlobals* a_globals, texture2d_t a_tex,
//                                       __private float3* a_pNormalWS, __private float2* a_pTexCoord) // float3& a_hitPos, int materialFlags
// {
//   float3x3 tangentTransform = make_float3x3(tangent, bitangent, a_flatNorm);
//   float3 viewTS             = normalize(mul3x3x3(tangentTransform, (-1.0f)*ViewWS));
// 
//   float3 p1 = make_float3((*a_pTexCoord).x, (*a_pTexCoord).y, 1.0f);
// 
//   float2 parallaxDirection = normalize(make_float2(viewTS.x, viewTS.y));
//   float  parallaxScale     = pHitMaterial->data[PARALLAX_HEIGHT]; 
//   float  parallaxLength    = parallaxScale*length(make_float2(viewTS.x, viewTS.y) / (fabs(viewTS.z) + 0.0001f));
// 
//   float3 p2 = make_float3(p1.x + parallaxDirection.x*parallaxLength, 
//                           p1.y + parallaxDirection.y*parallaxLength, 
//                           0.0f);
// 
//   // now search between p1 and p2
//   //
//   float minSteps = 8.0f;
//   float maxSteps = 128.0f;
//   float ts       = fmax(1.0f - fabs(viewTS.z), 0.0f);
//   float steps    = clamp(minSteps + ts*(maxSteps - minSteps), minSteps, maxSteps);
//   float3 t       = (p2 - p1)*(1.0f / steps);
// 
//   int2 texIds     = materialGetNormalTex(pHitMaterial);
//   int texId       = texIds.x;
//   int texMatrixId = texIds.y;
// 
//   const float4 texTransform = make_float4(0, 0, 0, 0); // tableNormalsLDRPtr(a_globals)[safeTexId(texId)];    // #TODO: fix
//   bool invertHeight         = false; // !(materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_INVERT_HEIGHT);     // #TODO: fix
// 
//   float3 currPoint = make_float3(0,0,0); 
//   //currPoint = LinearSearch(p1, t, texTransform, texMatrixId, invertHeight, a_tex, texMatricesPtr(a_globals));                // #TODO: fix
//   //currPoint = SimpleBinarySearch(currPoint, t, texTransform, texMatrixId, invertHeight, a_tex, texMatricesPtr(a_globals));   // #TODO: fix
// 
//   // one
//   //
//   (*a_pTexCoord).x = currPoint.x;
//   (*a_pTexCoord).y = currPoint.y;
// 
//   // #TODO: REFACROR THIS
//   //
//   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   const float3 normalFromTex = sample2D(texMatrixId, (*a_pTexCoord), (__global const int4*)pHitMaterial, a_tex, a_globals);
//   
//   float3 normalTS = make_float3(2.0f * normalFromTex.x - 1.0f, 2.0f * normalFromTex.y - 1.0f, normalFromTex.z);
// 
//   if (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_INVERT_NMAP_Y)
//     normalTS.y *= (-1.0f);
// 
//   if (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_INVERT_NMAP_X)
//     normalTS.x *= (-1.0f);
// 
//   if (materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY)
//   {
//     float temp = normalTS.x;
//     normalTS.x = normalTS.y;
//     normalTS.y = temp;
//   }
// 
//   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 
//   float3x3 inverseTangentTransform = inverse(tangentTransform);
// 
//   float3 oldNormal = (*a_pNormalWS);
//   (*a_pNormalWS)   = normalize(mul3x3x3(inverseTangentTransform, normalTS));
// 
//   // three
//   //float3 para
//   //llaxDirectionWS = normalize(mul3x3x3(inverseTangentTransform, make_float3(parallaxDirection.x, parallaxDirection.y, 0.0f)));
//   //float  heightFinal = abs(1.0f - currPoint.z)*parallaxScale;
// 
//   //float sinAlpha = abs(viewTS.z);
//   //float cosAlpha = sqrtf(1.0f - sinAlpha*sinAlpha);
//   //float tgAlpha = sinAlpha / fmaxf(cosAlpha, 0.0001f);
//   //float tAlongSurface = 2.0f*heightFinal / tgAlpha;
// 
//   //a_hitPos += parallaxDirectionWS*tAlongSurface;
//   //a_hitPos += RayEpsilon(a_hitPos)*oldNormal;
// }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct TransparencyShadowStepDataT
{
  float3 currFogColor;
  float  currFogMult;
  float  currHitDist;

} TransparencyShadowStepData;

static inline float3 transparencyStep(__private TransparencyShadowStepData* pData, float3 shadow, 
                                      const int alphaMatId, const float hit_t, const float3 a_rdir, float3 shadowHitNorm, const float2 texCoordS,
                                      __global const EngineGlobals* a_globals, __global const float4* a_mltStorage, texture2d_t a_shadingTexture)
{

  if (alphaMatId == ALPHA_MATERIAL_MASK)
    return shadow;

  bool disableCaustics = !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS);

  if (dot(shadowHitNorm, a_rdir) > 0.0f)
    shadowHitNorm *= (-1.0f);

  __global const PlainMaterial* pShadowHitMaterial = materialAt(a_globals, a_mltStorage, alphaMatId);

  ProcTextureList ptl;       // #TODO: read this from outside!!!
  InitProcTextureList(&ptl); // #TODO: read this from outside!!!

  TransparencyAndFog matFogAndTransp = materialEvalTransparencyAndFog(pShadowHitMaterial, a_rdir, shadowHitNorm, texCoordS, a_globals, a_shadingTexture, &ptl);

  if (disableCaustics)
    shadow *= matFogAndTransp.transparency;


  if ((materialGetFlags(pShadowHitMaterial) & PLAIN_MATERIAL_SKIP_SHADOW) || (materialGetFlags(pShadowHitMaterial) & PLAIN_MATERIAL_SKIP_SKY_PORTAL))
    shadow *= make_float3(1.0f, 1.0f, 1.0f);
  else if (pData->currFogMult != 0.0f && disableCaustics)
    shadow *= transparencyAttenuation(pData->currFogColor, pData->currFogMult, fabs(hit_t - pData->currHitDist));

  float4 matFogData = matFogAndTransp.fog;

  if (matFogData.w != 0.0f)
  {
    if (pData->currFogMult == 0.0f)
    {
      pData->currFogColor = to_float3(matFogData);
      pData->currFogMult  = matFogData.w;
    }
    else
    {
      pData->currFogMult  = 0.0f;
      pData->currFogColor = make_float3(0.0f, 0.0f, 0.0f);
    }

    pData->currHitDist = hit_t;
  }

  return shadow;
}


static inline unsigned int flagsNextBounce(unsigned int flags, const MatSample a_matSample, __global const EngineGlobals* a_globals)
{

  const bool thisBounceIsDiffuse   = (a_matSample.flags & RAY_EVENT_D) != 0;

  const unsigned int rayBounceNum  = unpackBounceNum(flags);
  const unsigned int diffBounceNum = unpackBounceNumDiff(flags);
  unsigned int       otherFlags    = unpackRayFlags(flags);

  flags = packBounceNum(flags, rayBounceNum + 1);

  if (thisBounceIsDiffuse)
    flags = packBounceNumDiff(flags, diffBounceNum + 1);

  const unsigned int rayBounceNum2  = rayBounceNum + 1;
  const unsigned int diffBounceNum2 = unpackBounceNumDiff(flags);

  if ((rayBounceNum2 >= (unsigned int)a_globals->varsI[HRT_TRACE_DEPTH]) || (diffBounceNum2 >= (unsigned int)a_globals->varsI[HRT_DIFFUSE_TRACE_DEPTH] + 1))
    otherFlags |= RAY_IS_DEAD;

  if (isGlossy(a_matSample))
    otherFlags |= RAY_EVENT_G;

  if (isPureSpecular(a_matSample))
    otherFlags |= RAY_EVENT_S;

  if (isDiffuse(a_matSample))
    otherFlags |= RAY_EVENT_D;

  if(isTransparent(a_matSample))
    otherFlags |= RAY_EVENT_T;

  // specific render layer output

  const bool stopP = (a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_PRIMARY) && (rayBounceNum >= 1);
  const bool stopR = (a_globals->varsI[HRT_RENDER_LAYER] == LAYER_COLOR_THE_REST || a_globals->varsI[HRT_RENDER_LAYER] == LAYER_COLOR_PRIMARY_AND_REST) && (diffBounceNum2 >= 1) && (rayBounceNum == 0);

  if (stopP || stopR)
    otherFlags |= RAY_IS_DEAD;

  return packRayFlags(flags, otherFlags);
}


static inline unsigned int flagsNextBounceLite(unsigned int flags, const MatSample a_matSample, __global const EngineGlobals* a_globals)
{
  const bool thisBounceIsDiffuse   = (a_matSample.flags & RAY_EVENT_D) != 0;
  const unsigned int rayBounceNum  = unpackBounceNum(flags);
  const unsigned int diffBounceNum = unpackBounceNumDiff(flags);
  unsigned int       otherFlags    = unpackRayFlags(flags);

  flags = packBounceNum(flags, rayBounceNum + 1);
  if (thisBounceIsDiffuse)
    flags = packBounceNumDiff(flags, diffBounceNum + 1);

  const unsigned int rayBounceNum2  = rayBounceNum + 1;
  const unsigned int diffBounceNum2 = unpackBounceNumDiff(flags);

  if ((rayBounceNum2 >= (unsigned int)a_globals->varsI[HRT_TRACE_DEPTH]) ||
      (diffBounceNum2 >= (unsigned int)a_globals->varsI[HRT_DIFFUSE_TRACE_DEPTH] + 1) ) // terminate ray
    otherFlags |= RAY_IS_DEAD;

  if (isGlossy(a_matSample))
    otherFlags |= RAY_EVENT_G;

  if (isPureSpecular(a_matSample))
    otherFlags |= RAY_EVENT_S;

  if (isDiffuse(a_matSample))
    otherFlags |= RAY_EVENT_D;

  if (isTransparent(a_matSample))
    otherFlags |= RAY_EVENT_T;

  return packRayFlags(flags, otherFlags);
}

static inline bool flagsHaveOnlySpecular(const unsigned int flags)
{
  const unsigned int otherFlags = unpackRayFlags(flags);

  return ( (otherFlags & RAY_EVENT_G) == 0) && 
         ( (otherFlags & RAY_EVENT_D) == 0) ;
}


#define TRANSPARENCY_LIST_SIZE 16


#endif

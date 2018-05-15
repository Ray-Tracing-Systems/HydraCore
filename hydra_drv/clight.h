#ifndef RTCLIGHT
#define RTCLIGHT

#include "cglobals.h"
#include "cfetch.h"

static inline float mylocalsmoothstep(float edge0, float edge1, float x)
{
  float  tVal = (x - edge0) / (edge1 - edge0);
  float  t    = fmin(fmax(tVal, 0.0f), 1.0f); 
  return t * t * (3.0f - 2.0f * t);
}

#define PLIGHT_TYPE  0
#define PLIGHT_FLAGS 1

#define PLIGHT_POS_X 2
#define PLIGHT_POS_Y 3
#define PLIGHT_POS_Z 4

#define PLIGHT_NORM_X 5
#define PLIGHT_NORM_Y 6
#define PLIGHT_NORM_Z 7

#define PLIGHT_COLOR_X 8
#define PLIGHT_COLOR_Y 9
#define PLIGHT_COLOR_Z 10

#define PLIGHT_COLOR_TEX        11
#define PLIGHT_COLOR_TEX_MATRIX 12

#define PLIGHT_SURFACE_AREA     13
#define SPHERE_LIGHT_RADIUS     14


#define PLIGHT_PROB_MULT       (LIGHT_DATA_SIZE-24)
#define PLIGHT_GROUP_ID        (LIGHT_DATA_SIZE-23)
#define PLIGHT_PICK_PROB_FWD   (LIGHT_DATA_SIZE-22)
#define PLIGHT_PICK_PROB_REV   (LIGHT_DATA_SIZE-21)

#define IES_INV_MATRIX_E00 (LIGHT_DATA_SIZE-20)
#define IES_INV_MATRIX_E01 (LIGHT_DATA_SIZE-19)
#define IES_INV_MATRIX_E02 (LIGHT_DATA_SIZE-18)
#define IES_INV_MATRIX_E10 (LIGHT_DATA_SIZE-17)
#define IES_INV_MATRIX_E11 (LIGHT_DATA_SIZE-16)
#define IES_INV_MATRIX_E12 (LIGHT_DATA_SIZE-15)
#define IES_INV_MATRIX_E20 (LIGHT_DATA_SIZE-14)
#define IES_INV_MATRIX_E21 (LIGHT_DATA_SIZE-13)
#define IES_INV_MATRIX_E22 (LIGHT_DATA_SIZE-12)

#define IES_LIGHT_MATRIX_E00 (LIGHT_DATA_SIZE-11)
#define IES_LIGHT_MATRIX_E01 (LIGHT_DATA_SIZE-10)
#define IES_LIGHT_MATRIX_E02 (LIGHT_DATA_SIZE-9)
#define IES_LIGHT_MATRIX_E10 (LIGHT_DATA_SIZE-8)
#define IES_LIGHT_MATRIX_E11 (LIGHT_DATA_SIZE-7)
#define IES_LIGHT_MATRIX_E12 (LIGHT_DATA_SIZE-6)
#define IES_LIGHT_MATRIX_E20 (LIGHT_DATA_SIZE-5)
#define IES_LIGHT_MATRIX_E21 (LIGHT_DATA_SIZE-4)
#define IES_LIGHT_MATRIX_E22 (LIGHT_DATA_SIZE-3)

#define IES_SPHERE_PDF_ID (LIGHT_DATA_SIZE-2)
#define IES_SPHERE_TEX_ID (LIGHT_DATA_SIZE-1)


static inline int lightType (__global const PlainLight* pLight) { return as_int(pLight->data[PLIGHT_TYPE]); }
static inline int lightFlags(__global const PlainLight* pLight) { return as_int(pLight->data[PLIGHT_FLAGS]); }

static inline float3 lightPos (__global const PlainLight* pLight) { return make_float3(pLight->data[PLIGHT_POS_X], pLight->data[PLIGHT_POS_Y], pLight->data[PLIGHT_POS_Z]); }
static inline float3 lightNorm(__global const PlainLight* pLight) { return make_float3(pLight->data[PLIGHT_NORM_X], pLight->data[PLIGHT_NORM_Y], pLight->data[PLIGHT_NORM_Z]); }
static inline float3 lightBaseColor(__global const PlainLight* pLight) { return make_float3(pLight->data[PLIGHT_COLOR_X], pLight->data[PLIGHT_COLOR_Y], pLight->data[PLIGHT_COLOR_Z]); }

static inline __global const float* lightIESPdfTable(__global const PlainLight* pLight, __global const EngineGlobals* a_globals, __global const float4* a_tableStorage,
                                                     __private int* pW, __private int* pH)
{
  const int texId     = as_int(pLight->data[IES_SPHERE_PDF_ID]);
  const int texOffset = pdfTableHeaderOffset(texId, a_globals);
  __global const float* pTexHeader = (__global const float*)(a_tableStorage + texOffset);

  (*pW) = as_int(pTexHeader[0]);
  (*pH) = as_int(pTexHeader[1]);

  return pTexHeader + 4;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#define CYLINDER_LIGHT_MATRIX_E00   16
#define CYLINDER_LIGHT_MATRIX_E01   17
#define CYLINDER_LIGHT_MATRIX_E02   18
#define CYLINDER_LIGHT_MATRIX_E10   19
#define CYLINDER_LIGHT_MATRIX_E11   20
#define CYLINDER_LIGHT_MATRIX_E12   21
#define CYLINDER_LIGHT_MATRIX_E20   22
#define CYLINDER_LIGHT_MATRIX_E21   23
#define CYLINDER_LIGHT_MATRIX_E22   24

#define CYLINDER_LIGHT_RADIUS       25
#define CYLINDER_LIGHT_ZMIN         26
#define CYLINDER_LIGHT_ZMAX         27
#define CYLINDER_LIGHT_PHIMAX       28

#define CYLINDER_TEX_ID             29
#define CYLINDER_TEXMATRIX_ID       30
#define CYLINDER_PDF_TABLE_ID       31
#define CYLINDER_TEX_SAMPLER        32

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define POINT_LIGHT_SPOT_COS1    14
#define POINT_LIGHT_SPOT_COS2    15

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DIRECT_LIGHT_RADIUS1   14
#define DIRECT_LIGHT_RADIUS2   15
#define DIRECT_LIGHT_SSOFTNESS 16
#define DIRECT_LIGHT_ALPHA_TAN 17
#define DIRECT_LIGHT_ALPHA_COS 18

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SKY_DOME_PDF_IMAGE_SIZE_X  14
#define SKY_DOME_PDF_IMAGE_SIZE_Y  15

#define SKY_DOME_COLOR_AUX_X 17
#define SKY_DOME_COLOR_AUX_Y 18
#define SKY_DOME_COLOR_AUX_Z 19

#define SKY_DOME_COLOR_TEX_AUX         20
#define SKY_DOME_COLOR_TEX_MATRIX_AUX  21

#define SKY_DOME_AUX_TEX_MATRIX_INV    22

#define SKY_DOME_SUN_DIR_X 23
#define SKY_DOME_SUN_DIR_Y 24
#define SKY_DOME_SUN_DIR_Z 25

#define SKY_DOME_TURBIDITY 26

#define SKY_SUN_COLOR_X    27
#define SKY_SUN_COLOR_Y    28
#define SKY_SUN_COLOR_Z    29

#define SKY_DOME_PDF_TABLE0 30
#define SKY_DOME_PDF_TABLE1 31

/////
#define SKY_DOME_SAMPLER0   32  // sampler take 3xfloat4: sampler data - float4 + float2x4;
#define SKY_DOME_MATRIX0    36  // 
#define SKY_DOME_SAMPLER1   44  // sampler take 3xfloat4: sampler data - float4 + float2x4;
#define SKY_DOME_MATRIX1    48 

#define SKY_DOME_INV_MATRIX0 56 // takes float4x4
#define SKY_DOME_INV_MATRIX1 72 // takes float4x4

#define SKY_DOME_SUN_DIR_ID  88

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MESH_LIGHT_MESH_OFFSET_ID  14
#define MESH_LIGHT_TABLE_OFFSET_ID 15
#define MESH_LIGHT_TRI_NUM         16
#define MESH_LIGHT_MATRIX_E00      20

#define MESH_LIGHT_TEX_ID          30
#define MESH_LIGHT_TEXMATRIX_ID    31
#define MESH_LIGHT_TEX_SAMPLER     32 // must divided by 4

static inline float3 perezZenith(float t, float thetaSun)
{
  const float	  pi = 3.1415926f;
  const float4	cx1 = make_float4(0.0f, 0.00209f, -0.00375f, 0.00165f);
  const float4	cx2 = make_float4(0.00394f, -0.03202f, 0.06377f, -0.02903f);
  const float4	cx3 = make_float4(0.25886f, 0.06052f, -0.21196f, 0.11693f);
  const float4	cy1 = make_float4(0.0f, 0.00317f, -0.00610f, 0.00275f);
  const float4	cy2 = make_float4(0.00516f, -0.04153f, 0.08970f, -0.04214f);
  const float4	cy3 = make_float4(0.26688f, 0.06670f, -0.26756f, 0.15346f);

  float	  t2 = t*t;							// turbidity squared
  float	  chi = (4.0f / 9.0f - t / 120.0f) * (pi - 2.0f * thetaSun);
  float4	theta = make_float4(1.0f, thetaSun, thetaSun*thetaSun, thetaSun*thetaSun*thetaSun);

  float	Y = (4.0453f * t - 4.9710f) * tan(chi) - 0.2155f * t + 2.4192f;
  float	x = t2 * dot(cx1, theta) + t * dot(cx2, theta) + dot(cx3, theta);
  float	y = t2 * dot(cy1, theta) + t * dot(cy2, theta) + dot(cy3, theta);

  return make_float3(Y, x, y);
}

//
// Perez allweather func (turbidity, cosTheta, cosGamma)
//

static inline float3	perezFunc(float t, float cosTheta, float cosGamma)
{
  float	gamma = acos(cosGamma);
  float	cosGammaSq = cosGamma * cosGamma;

  float	aY = 0.17872f * t - 1.46303f;
  float	bY = -0.35540f * t + 0.42749f;
  float	cY = -0.02266f * t + 5.32505f;
  float	dY = 0.12064f * t - 2.57705f;
  float	eY = -0.06696f * t + 0.37027f;
  float	ax = -0.01925f * t - 0.25922f;
  float	bx = -0.06651f * t + 0.00081f;
  float	cx = -0.00041f * t + 0.21247f;
  float	dx = -0.06409f * t - 0.89887f;
  float	ex = -0.00325f * t + 0.04517f;
  float	ay = -0.01669f * t - 0.26078f;
  float	by = -0.09495f * t + 0.00921f;
  float	cy = -0.00792f * t + 0.21023f;
  float	dy = -0.04405f * t - 1.65369f;
  float	ey = -0.01092f * t + 0.05291f;

  return make_float3((1.0f + aY * exp(bY / cosTheta)) * (1.0f + cY * exp(dY * gamma) + eY*cosGammaSq),
                     (1.0f + ax * exp(bx / cosTheta)) * (1.0f + cx * exp(dx * gamma) + ex*cosGammaSq),
                     (1.0f + ay * exp(by / cosTheta)) * (1.0f + cy * exp(dy * gamma) + ey*cosGammaSq));
}

static inline float3 perezSky(float turbidity, float cosTheta, float cosGamma, float cosThetaSun)
{
  return ( perezZenith(turbidity, acos(cosThetaSun)) * perezFunc(turbidity, cosTheta, cosGamma) ) / perezFunc(turbidity, 1.0f, cosThetaSun);
}


static inline float3	convertColor(float3 clrYxy)
{
  // now rescale Y component
  clrYxy.x = 1.0f - exp(-clrYxy.x / 20.0f);

  float	ratio = clrYxy.x / fmax(clrYxy.z, 1e-10f);		// Y / y = X + Y + Z
  float3	XYZ;

  XYZ.x = clrYxy.y * ratio;		 // X = x * ratio
  XYZ.y = clrYxy.x;						 // Y = Y
  XYZ.z = ratio - XYZ.x - XYZ.y; // Z = ratio - X - Y

  const	float3	rCoeffs = make_float3(3.240479f, -1.53715f, -0.49853f);
  const	float3	gCoeffs = make_float3(-0.969256f, 1.875991f, 0.041556f);
  const	float3	bCoeffs = make_float3(0.055684f, -0.204043f, 1.057311f);

  return clamp(make_float3(dot(rCoeffs, XYZ), dot(gCoeffs, XYZ), dot(bCoeffs, XYZ)), 0.0f, 1.0f);
}


static inline float3 skyLightPerezColor(__global const PlainLight* pLight, float3 ray_dir)
{
  //return make_float3(0.25, 0.5f, 0.65);

  float3 sunDir   = make_float3(pLight->data[SKY_DOME_SUN_DIR_X], pLight->data[SKY_DOME_SUN_DIR_Y], pLight->data[SKY_DOME_SUN_DIR_Z]);
  float turbidity = pLight->data[SKY_DOME_TURBIDITY];

  float3 colorYxy = perezSky(turbidity, fmax(ray_dir.y, 0.0f) + 0.05f, fmax(dot(sunDir, (-1.0f)*ray_dir), 0.0f), fmax(-sunDir.y, 0.0f));
  float3 rgb      = convertColor(colorYxy);
    
  rgb.x = pow(rgb.x, 2.2f);
  rgb.y = pow(rgb.y, 2.2f);
  rgb.z = pow(rgb.z, 2.2f);

  float tSunAngle = fmax(-sunDir.y, 0.0f);

  float threshold = 0.9985f + tSunAngle*(0.9995f - 0.9985f);
  float tSun = dot(sunDir, (-1.0f)*ray_dir);

  if (tSun >= threshold)
  {
    float3 sunColor = (2.0f + 2.0f*tSunAngle)*make_float3(pLight->data[SKY_SUN_COLOR_X], pLight->data[SKY_SUN_COLOR_Y], pLight->data[SKY_SUN_COLOR_Z]);
    float tSun2 = (tSun - threshold) / (1.0f - threshold);
    tSun2 = tSun2*tSun2;
    rgb = sunColor*tSun2 + (1.0f - tSun2)*rgb;
  }

  return rgb;
}


static inline float3 skyLightGetIntensityEnv(__global const PlainLight* pLight, float3 a_direction)
{
  return lightBaseColor(pLight);
}


static inline float3 skyLightGetIntensityTexturedENV(__global const PlainLight* pLight, float3 a_direction,
                                                     __global const EngineGlobals* a_globals, 
                                                     __global const float4* a_pdfStorage, texture2d_t a_tex)
{
  const int sOffs       = as_int(pLight->data[PLIGHT_COLOR_TEX_MATRIX]);

  float sintheta = 0.0f;
  const float2 texCoord = sphereMapTo2DTexCoord(a_direction, &sintheta);
  const float3 texColor = sample2D(sOffs, texCoord, (__global const int4*)(pLight->data + SKY_DOME_SAMPLER0), a_tex, a_globals);
  const float3 envColor = skyLightGetIntensityEnv(pLight, a_direction);

  if (lightFlags(pLight) & SKY_LIGHT_USE_PEREZ_ENVIRONMENT)
    return envColor*skyLightPerezColor(pLight, a_direction);
  else
    return envColor*texColor;
}

static inline float evalMap2DPdf(const float2 texCoordT, __global const float* intervals, const int sizeX, const int sizeY)
{
  const float fw = (float)sizeX;
  const float fh = (float)sizeY;

  int pixelX = (int)(fw*texCoordT.x - 0.5f);
  int pixelY = (int)(fh*texCoordT.y - 0.5f);

  if (pixelX >= sizeX) pixelX = sizeX - 1;
  if (pixelY >= sizeY) pixelY = sizeY - 1;

  if (pixelX < 0) pixelX += sizeX;
  if (pixelY < 0) pixelY += sizeY;

  const int pixelOffset = pixelY*sizeX + pixelX;
  const int maxSize = sizeX*sizeY;
  const int offset0 = (pixelOffset + 0 < maxSize) ? pixelOffset + 0 : maxSize - 1;
  const int offset1 = (pixelOffset + 1 < maxSize) ? pixelOffset + 1 : maxSize - 1;

  const float2 interval = make_float2(intervals[offset0], intervals[offset1]);

  return (interval.y - interval.x)*(fw*fh)/intervals[sizeX*sizeY];
}

static inline float skyLightEvalPDF(__global const PlainLight* pLight, float3 illuminatingPoint, float3 rayDir, __global const EngineGlobals* a_globals, __global const float4* a_pdfStorage)
{
  __global const float* pdfHeader = pdfTableHeader(as_int(pLight->data[SKY_DOME_PDF_TABLE0]), a_pdfStorage, a_globals);
  __global const float* intervals = pdfHeader + 4;

  const int sizeX = as_int(pdfHeader[0]);
  const int sizeY = as_int(pdfHeader[1]);
  const float fw  = (float)sizeX;
  const float fh  = (float)sizeY;
  const float fN  = fw*fh;

  // get tex coords and transform them with tex matrix, move to texture space
  //
  float sintheta = 0.0f;
  const float2 texCoord = sphereMapTo2DTexCoord(rayDir, &sintheta);
  if (sintheta == 0.f)
    return 0.f;

  // apply inverse texcoord transform to get phi and theta and than get correct pdf from table 
  //
  __global const float4x4* pMatrix = (__global const float4x4*)(pLight->data + SKY_DOME_MATRIX0);

  const float2 texCoordT = mul2x4(pMatrix->row[0], pMatrix->row[1], texCoord);
  const float mapPdf     = evalMap2DPdf(texCoordT, intervals, sizeX, sizeY);
  return (mapPdf * 1.0f) / (2.f * M_PI * M_PI * fmax(sintheta, DEPSILON));
}


static inline int mylocalimax(int a, int b) { return (a > b) ? a : b; }
static inline int mylocalimin(int a, int b) { return (a < b) ? a : b; }

typedef struct Map2DPiecewiseSampleT
{
  float2 texCoord;
  float  mapPdf;
} Map2DPiecewiseSample;

static inline Map2DPiecewiseSample sampleMap2D(float3 rands, __global const float* intervals, const int sizeX, const int sizeY)
{
  const float fw = (float)sizeX;
  const float fh = (float)sizeY;
  const float fN = fw*fh;

  float pdf = 1.0f;
  int pixelOffset = SelectIndexPropToOpt(rands.z, intervals, sizeX*sizeY+1, &pdf);

  if (pixelOffset >= sizeX*sizeY)
    pixelOffset = sizeX*sizeY - 1;

  const int yPos = pixelOffset / sizeX;
  const int xPos = pixelOffset - yPos*sizeX;

  const float texX = (1.0f / fw)*(((float)(xPos) + 0.5f) + (rands.x*2.0f - 1.0f)*0.5f);
  const float texY = (1.0f / fh)*(((float)(yPos) + 0.5f) + (rands.y*2.0f - 1.0f)*0.5f);

  Map2DPiecewiseSample result;
  result.mapPdf   = pdf*fN; 
  result.texCoord = make_float2(texX, texY);
  return result;
}

/**
\brief  Sample sphere around light according to IES table thats is stored as spheremap.

\param  pLight         - input light
\param  rands          - input 3 randoms in [0,1]
\param  a_globals      - input engine globals
\param  a_tableStorage - input texture and pdf storage. Note that for IES texture and pdftable storage is the same storage.
\param  a_outDir       - output direction in world space. I.e. all matrices are already applied.
\param  a_outPdfW      - output pdf in solid angle (pdfW)

*/

static inline void LightSampleIESSphere(__global const PlainLight* pLight, float3 rands, __global const EngineGlobals* a_globals, __global const float4* a_tableStorage,
                                        __private float3* a_outDir, __private float* a_outPdfW)
{
  int w, h;
  __global const float* table = lightIESPdfTable(pLight, a_globals, a_tableStorage,
                                                 &w, &h);
  
  const Map2DPiecewiseSample sample = sampleMap2D(rands, table, w, h);
  __global const float* pMatrix     = pLight->data + IES_INV_MATRIX_E00;

  float sinTheta = 0.0f;
  float3 lsDir   = texCoord2DToSphereMap(sample.texCoord, &sinTheta);
  (*a_outDir)    = normalize(matrix3x3f_mult_float3(pMatrix, lsDir));
  (*a_outPdfW)   = INV_PI*INV_PI*0.5f*(sample.mapPdf/fmax(sinTheta, DEPSILON2));
}

inline static void SkyLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint,
                                     __global const EngineGlobals* a_globals, __global const float4* a_pdfStorage, texture2d_t a_tex,
                                     __private ShadowSample* a_out)
{
  __global const float* pdfHeader = pdfTableHeader(as_int(pLight->data[SKY_DOME_PDF_TABLE0]), a_pdfStorage, a_globals);
  __global const float* intervals = pdfHeader + 4;

  const int sizeX = as_int(pdfHeader[0]);
  const int sizeY = as_int(pdfHeader[1]);

  const Map2DPiecewiseSample sample = sampleMap2D(rands, intervals, sizeX, sizeY);

  // apply inverse texcoord transform to get phi and theta
  //
  __global const float4x4* pMatrix = (__global const float4x4*)(pLight->data + SKY_DOME_INV_MATRIX0);
  const float3 texCoordT = mul((*pMatrix), make_float3(sample.texCoord.x, sample.texCoord.y, 0.0f));

  //
  //
  float sintheta = 0.0f;
  const float3 sampleDir = texCoord2DToSphereMap(make_float2(texCoordT.x, texCoordT.y), &sintheta);
  const float3 samplePos = illuminatingPoint + sampleDir*a_globals->varsF[HRT_BSPHERE_RADIUS]; 

  const int    sOffs  = as_int(pLight->data[PLIGHT_COLOR_TEX_MATRIX]);
  const float3 txClr  = sample2D(sOffs, make_float2(texCoordT.x, texCoordT.y), (__global const int4*)(pLight->data + SKY_DOME_SAMPLER0), a_tex, a_globals);
  const float3 color  = skyLightGetIntensityEnv(pLight, sampleDir)*txClr; 

  const float hitDist = length(illuminatingPoint - samplePos);
  const float pdf     = (sample.mapPdf * 1.0f) / (2.f * M_PI * M_PI * fmax(sintheta, DEPSILON));

  a_out->isPoint = false;
  a_out->pos     = samplePos;
  a_out->color   = color;
  a_out->pdf     = pdf;
  a_out->maxDist = hitDist;
}


static inline float3 lightDistributionMask(__global const PlainLight* pLight, float3 a_rayDir, __global const EngineGlobals* a_globals, __global const float4* a_texStorage)
{
  float3 color = lightBaseColor(pLight);
  
  __global const float* pMatrix = pLight->data + IES_LIGHT_MATRIX_E00;
  a_rayDir = normalize(matrix3x3f_mult_float3(pMatrix, a_rayDir));
  
  if (as_int(pLight->data[PLIGHT_FLAGS]) & LIGHT_HAS_IES)
  {
    float sintheta = 0.0f;
    const float2 texCoord = sphereMapTo2DTexCoord((-1.0f)*a_rayDir, &sintheta);
    const int texId     = as_int(pLight->data[IES_SPHERE_TEX_ID]);
    const int texOffset = pdfTableHeaderOffset(texId, a_globals);
    const float val     = read_imagef_sw1((__global const int4*)(a_texStorage + texOffset), texCoord, (TEX_CLAMP_U | TEX_CLAMP_V));
    return make_float3(val, val, val);
  }
  else
    return make_float3(1.0f, 1.0f, 1.0f);
}

static inline float3 pointLightGetIntensity(__global const PlainLight* pLight, float3 a_rayDir, __global const EngineGlobals* a_globals, __global const float4* a_tableStorage)
{
  float3 maskColor = lightDistributionMask(pLight, a_rayDir, a_globals, a_tableStorage);
  return maskColor*lightBaseColor(pLight);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define AREA_LIGHT_SIZE_X       14
#define AREA_LIGHT_SIZE_Y       15

#define AREA_LIGHT_MATRIX_E00   16
#define AREA_LIGHT_MATRIX_E01   17
#define AREA_LIGHT_MATRIX_E02   18
#define AREA_LIGHT_MATRIX_E10   19
#define AREA_LIGHT_MATRIX_E11   20
#define AREA_LIGHT_MATRIX_E12   21
#define AREA_LIGHT_MATRIX_E20   22
#define AREA_LIGHT_MATRIX_E21   23
#define AREA_LIGHT_MATRIX_E22   24

#define AREA_LIGHT_IS_DISK      25
#define AREA_LIGHT_SPOT_DISTR   26

#define AREA_LIGHT_SPOT_COS1    27
#define AREA_LIGHT_SPOT_COS2    28

#define AREA_LIGHT_SKY_OFFSET   29
#define AREA_LIGHT_SKY_SOURCE   30

#define AREA_LIGHT_SKYPORTAL_BTEX        31
#define AREA_LIGHT_SKYPORTAL_BTEX_MATRIX 32

#define AREA_LIGHT_SAMPLER0   40  // sampler take 3xfloat4: sampler data - float4 + float2x4;
#define AREA_LIGHT_MATRIX0    44  // float2x4
#define AREA_LIGHT_SAMPLER1   52  // sampler take 3xfloat4: sampler data - float4 + float2x4;
#define AREA_LIGHT_MATRIX1    56  // float2x4


static inline float areaDiffuseLightEvalPDF(__global const PlainLight* pLight, float3 rayDir, float hitDist)
{
  const float3 lnorm = lightNorm(pLight);
  const float pdfA   = 1.0f / fmax(pLight->data[PLIGHT_SURFACE_AREA], DEPSILON);
  const float cosVal = (as_int(pLight->data[PLIGHT_FLAGS]) & LIGHT_HAS_IES) ? fabs(dot(rayDir, -1.0f*lnorm)) : fmax(dot(rayDir, -1.0f*lnorm), 0.0f);
  return PdfAtoW(pdfA, hitDist, cosVal);
}

static inline float areaSpotLightAttenuation(__global const PlainLight* pLight, float3 in_shadowRayDir)
{
  float cos1      = pLight->data[AREA_LIGHT_SPOT_COS1];
  float cos2      = pLight->data[AREA_LIGHT_SPOT_COS2];
  float3 norm     = lightNorm(pLight);
  float cos_theta = fmax(dot(in_shadowRayDir, norm), 0.0f);
  return mylocalsmoothstep(cos2, cos1, cos_theta);
}


static inline float3 areaDiffuseLightGetIntensity(__global const PlainLight* pLight, float3 a_rayDir, float2 a_texCoord, bool eyeRay, 
                                                  __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage)
{
  float3 color = lightBaseColor(pLight);

  if (as_int(pLight->data[PLIGHT_COLOR_TEX]) != INVALID_TEXTURE)
  {
    //int texId       = as_int(pLight->data[PLIGHT_COLOR_TEX]);
    int texMatrixId = as_int(pLight->data[PLIGHT_COLOR_TEX_MATRIX]);

    if (lightFlags(pLight) & AREA_LIGHT_SKY_PORTAL)
    {
      float sintheta = 0.0f;
      a_texCoord = sphereMapTo2DTexCoord(a_rayDir, &sintheta);
    }
    color *= sample2D(texMatrixId, a_texCoord, (__global const int4*)(pLight->data + AREA_LIGHT_SAMPLER0), 
			                                         (__global const int4*)a_tableStorage, a_globals);                //#CHECK_THIS
  }
  else if (as_int(pLight->data[PLIGHT_FLAGS]) & LIGHT_HAS_IES)
  {
    float3 atten = lightDistributionMask(pLight, a_rayDir, a_globals, a_tableStorage);

    // crap to make area spot white   
    if (!eyeRay)
      color *= atten;
    else
    {
      const float maxColor = fmax(color.x, fmax(color.y, color.z));
      const float outsideSpotVal = (1.0f / maxColor);
      color *= outsideSpotVal;
    }
  }
  else
  {
    if (as_int(pLight->data[AREA_LIGHT_SPOT_DISTR]) != 0)
    {
      float atten = areaSpotLightAttenuation(pLight, (-1.0f)*a_rayDir);

      // crap to make area spot white   
      if (!eyeRay)
        color *= clamp(atten, 0.0, 1.0f);
      else
      {
        const float maxColor       = fmax(color.x, fmax(color.y, color.z));
        const float outsideSpotVal = (1.0f / maxColor);
        color *= outsideSpotVal;
      }

    }

    // get color from sky
    //
    if (lightFlags(pLight) & AREA_LIGHT_SKY_PORTAL)                      //#TODO: fix
    {
      int skyLightOffset = as_int(pLight->data[AREA_LIGHT_SKY_OFFSET]);  //#TODO: fix
      __global const PlainLight* pSkyLight = pLight + skyLightOffset;    //#TODO: fix, put sky light parameters to area light ? 

      float3 sunColor = make_float3(1.0f, 1.0f, 1.0f);
      
      if (lightFlags(pSkyLight) & SKY_LIGHT_USE_PEREZ_ENVIRONMENT)
        sunColor = 0.5f*skyLightPerezColor(pSkyLight, a_rayDir);
      else
        sunColor = skyLightGetIntensityTexturedENV(pSkyLight, a_rayDir, a_globals, a_tableStorage, a_tex);
     
      color = color*sunColor; 
    }
  }

  return color;
}


static inline float3 areaLightSkyPortalCustomColor(__global const PlainLight* pLight, float3 rayDir,
                                                   __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_storagePdf)
{
  //return make_float3(0, 1, 0);
  int skyLightOffset = as_int(pLight->data[AREA_LIGHT_SKY_OFFSET]); //#TODO: fix
  __global const PlainLight* pSkyLight = pLight + skyLightOffset;   //#TODO: fix, put sky light parameters to area light ? 

  float3 skyColor;
  if (lightFlags(pSkyLight) & SKY_LIGHT_USE_PEREZ_ENVIRONMENT)
    skyColor = skyLightPerezColor(pSkyLight, rayDir)*0.5f;
  else
    skyColor = skyLightGetIntensityTexturedENV(pSkyLight, rayDir, a_globals, a_storagePdf, a_tex);

  return lightBaseColor(pLight)*skyColor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct LightSampleT2
{
  float3 pos;
  float3 dir;
  float3 norm;
  float3 color;
  float  pdfA;
  float  pdfW;
  float  cosTheta;
  bool   isPoint;

} LightSampleFwd;


typedef struct LightPDFT2
{
  float  pdfA;
  float  pdfW;
  float  pickProb;
} LightPdfFwd;

static inline void AreaLightSampleForward(__global const PlainLight* pLight, float4 rands, float2 rands2,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                          __private LightSampleFwd* a_outRes)
{
  float offsetX = rands.x * 2.0f - 1.0f;
  float offsetY = rands.y * 2.0f - 1.0f;

  float3 samplePos = make_float3(offsetX*pLight->data[AREA_LIGHT_SIZE_X], 0.0f, offsetY*pLight->data[AREA_LIGHT_SIZE_Y]);  

  if (as_int(pLight->data[AREA_LIGHT_IS_DISK]) != 0)
  {
    float2 xz = MapSamplesToDisc(make_float2(offsetX, offsetY))*pLight->data[AREA_LIGHT_SIZE_X]; // disk radius
    samplePos = make_float3(xz.x, 0.0f, xz.y);
  }

  __global const float* pMatrix = pLight->data + AREA_LIGHT_MATRIX_E00;
  samplePos = matrix3x3f_mult_float3(pMatrix, samplePos);                                                                     // transform with rotation matrix
  samplePos = samplePos + lightPos(pLight);    // translate to world light position

  if (lightFlags(pLight) & LIGHT_IES_POINT_AREA)
    samplePos = lightPos(pLight);

  float3 lnorm          = lightNorm(pLight);
  float3 sampleDir      = MapSampleToCosineDistribution(rands.z, rands.w, lnorm, lnorm, 1.0f);
  float cosTheta        = fmax(dot(sampleDir, lnorm), 0.0f);

  float pdfW  = cosTheta*INV_PI;
  if (lightFlags(pLight) & LIGHT_HAS_IES)
  {
    LightSampleIESSphere(pLight, make_float3(rands.z, rands.w, rands2.x), a_globals, a_tableStorage,
                         &sampleDir, &pdfW);

    lnorm    = dot(lnorm, sampleDir) > 0.0f ? lnorm : -1.0f*lnorm;
  }
  else if (as_int(pLight->data[AREA_LIGHT_SPOT_DISTR]) != 0)
  {
    //const float cos1 = pLight->data[AREA_LIGHT_SPOT_COS1];
    const float cos2 = pLight->data[AREA_LIGHT_SPOT_COS2];

    sampleDir = MapSamplesToCone(cos2, make_float2(rands.z, rands.w), lnorm); 
    pdfW      = 1.0f / (2.0f * M_PI * (1.0f - cos2));
  }

  cosTheta = fmax(dot(sampleDir, lnorm), 0.0f);

  float3 color;
  if (as_int(pLight->data[PLIGHT_FLAGS]) & AREA_LIGHT_SKY_PORTAL)
    color = areaLightSkyPortalCustomColor(pLight, (-1.0f)*sampleDir, a_globals, a_tex, a_tableStorage);
  else
  {
    float3 customRayDir = (-1.0f)*sampleDir;
    //if (lightFlags(pLight) & LIGHT_IES_POINT_AREA)   // 
    //  customRayDir = normalize(lightPos(pLight) - illuminatingPoint);

    color = areaDiffuseLightGetIntensity(pLight, customRayDir, make_float2(rands.x, rands.y), false, a_globals, a_tex, a_tableStorage);
  }

  a_outRes->isPoint  = false;
  a_outRes->pos      = samplePos + epsilonOfPos(samplePos)*lnorm;
  a_outRes->dir      = sampleDir;
  a_outRes->color    = color*cosTheta;
  a_outRes->pdfA     = 1.0f / pLight->data[PLIGHT_SURFACE_AREA]; 
  a_outRes->pdfW     = pdfW;
  a_outRes->cosTheta = cosTheta;
  a_outRes->norm     = lnorm;
}


static inline void SphereLightSampleForward(__global const PlainLight* pLight, float4 rands,
                                            __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                            __private LightSampleFwd* a_outRes)
{
  // try MapSamplesToSphere ?

  const float theta = 2.0f * M_PI * rands.x;
  const float phi   = acos(1.0f - 2.0f * rands.y);
  const float x     = sin(phi) * cos(theta);
  const float y     = sin(phi) * sin(theta);
  const float z     = cos(phi);

  const float3 lcenter = lightPos(pLight);
  const float  lradius = pLight->data[SPHERE_LIGHT_RADIUS];
  const float3 color   = lightBaseColor(pLight);

  const float3 samplePos = lcenter + lradius*make_float3(x, y, z);
  const float3 lightNorm = normalize(samplePos - lcenter);
  const float3 sampleDir = MapSampleToCosineDistribution(rands.z, rands.w, lightNorm, lightNorm, 1.0f);
  const float  cosTheta  = fmax(dot(sampleDir, lightNorm), 0.0f);

  a_outRes->isPoint  = false;
  a_outRes->pos      = samplePos + epsilonOfPos(samplePos)*lightNorm;
  a_outRes->dir      = sampleDir;
  a_outRes->color    = color*cosTheta;
  a_outRes->pdfA     = 1.0f / pLight->data[PLIGHT_SURFACE_AREA]; 
  a_outRes->pdfW     = (cosTheta*INV_PI);   
  a_outRes->cosTheta = cosTheta;
  a_outRes->norm     = lightNorm;
}

static inline float3 cylinderLightGetIntensity(__global const PlainLight* pLight, float2 texCoord, 
                                               __global const EngineGlobals* a_globals, texture2d_t a_texStorage)
{
  const int samplerOffset = as_int(pLight->data[CYLINDER_TEXMATRIX_ID]);
  const float3 texColor   = sample2D(samplerOffset, texCoord, (__global const int4*)pLight, a_texStorage, a_globals);
  return texColor*lightBaseColor(pLight);
}

static inline void CylinderLightSamplePos(__global const PlainLight* pLight, float3 rands,
                                          __global const EngineGlobals* a_globals, texture2d_t a_texStorage, __global const float4* a_tableStorage,
                                          __private float3* a_pPos, __private float3* a_pNormal, __private float2* a_pTexCoord, __private float* pPdfA)

{
  Map2DPiecewiseSample sample;
  sample.texCoord.x = rands.x;
  sample.texCoord.y = rands.y;
  sample.mapPdf     = 1.0f;

  const int texId = as_int(pLight->data[CYLINDER_PDF_TABLE_ID]);
  if (texId > 0)
  {
    __global const float* pdfHeader = pdfTableHeader(texId, a_tableStorage, a_globals);
    __global const float* intervals = pdfHeader + 4;
    const int sizeX = as_int(pdfHeader[0]);
    const int sizeY = as_int(pdfHeader[1]);
    sample = sampleMap2D(rands, intervals, sizeX, sizeY);
  }

  (*pPdfA) = sample.mapPdf / pLight->data[PLIGHT_SURFACE_AREA];

  /////////////////////////////////////////////////////////////////////////////// sample cylinder uniformly in local space

  const float zMin   = pLight->data[CYLINDER_LIGHT_ZMIN];
  const float zMax   = pLight->data[CYLINDER_LIGHT_ZMAX];
  const float radius = pLight->data[CYLINDER_LIGHT_RADIUS];
  const float phiMax = pLight->data[CYLINDER_LIGHT_PHIMAX];

  const float z   = zMin + sample.texCoord.x * (zMax - zMin);
  const float phi = sample.texCoord.y * phiMax;

  const float2 scv  = sincos2f(phi);
  float3 pObj       = make_float3(radius * scv.y, radius * scv.x, z);
  float3 n          = normalize(make_float3(pObj.x, pObj.y, 0));

  const float hitRad = sqrt(pObj.x * pObj.x + pObj.y * pObj.y);
  pObj.x *= radius / hitRad;
  pObj.y *= radius / hitRad;

  /////////////////////////////////////////////////////////////////////////////// sample cylinder uniformly in local space

  __global const float* pMatrix = pLight->data + CYLINDER_LIGHT_MATRIX_E00;

  n = normalize(matrix3x3f_mult_float3(pMatrix, n));

  const float3 center    = lightPos(pLight);
  const float3 samplePos = center + matrix3x3f_mult_float3(pMatrix, pObj) + epsilonOfPos(center)*n;

  (*a_pPos)      = samplePos;
  (*a_pNormal)   = n;
  (*a_pTexCoord) = sample.texCoord;
}


static inline void CylinderLightSampleForward(__global const PlainLight* pLight, const float4 rands, const float2 rands2,
                                              __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                              __private LightSampleFwd* a_outRes)
{

  float3 samplePos, lightNorm; float2 texCoord; float pdfA;
  CylinderLightSamplePos(pLight, make_float3(rands.x, rands.y, rands2.x), a_globals, a_tex, a_tableStorage,
                         &samplePos, &lightNorm, &texCoord, &pdfA);

  const float3 sampleDir = MapSampleToCosineDistribution(rands.z, rands.w, lightNorm, lightNorm, 1.0f);
  const float  cosTheta  = fmax(dot(sampleDir, lightNorm), 0.0f);

  a_outRes->isPoint  = false;
  a_outRes->pos      = samplePos + epsilonOfPos(samplePos)*lightNorm;
  a_outRes->dir      = sampleDir;
  a_outRes->color    = cylinderLightGetIntensity(pLight, texCoord, a_globals, a_tex)*cosTheta;
  a_outRes->pdfA     = pdfA;
  a_outRes->pdfW     = (cosTheta*INV_PI);   
  a_outRes->cosTheta = cosTheta;
  a_outRes->norm     = lightNorm;
}

static inline void PointLightSampleForward(__global const PlainLight* pLight, float4 rands,
                                           __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                           __private LightSampleFwd* a_outRes)
{
  
  float pdfW       = INV_PI*0.25f;
  float3 sampleDir = UniformSampleSphere(rands.x, rands.y);

  if (lightFlags(pLight) & LIGHT_HAS_IES)
  {
    LightSampleIESSphere(pLight, to_float3(rands), a_globals, a_tableStorage,
                         &sampleDir, &pdfW);
  }
  
  const float3 samplePos = lightPos(pLight);
  const float3 color     = pointLightGetIntensity(pLight, (-1.0f)*sampleDir, a_globals, a_tableStorage);

  a_outRes->isPoint  = true;
  a_outRes->pos      = samplePos + epsilonOfPos(samplePos)*sampleDir;
  a_outRes->dir      = sampleDir;
  a_outRes->color    = color*(1.0f / pLight->data[PLIGHT_SURFACE_AREA]);
  a_outRes->pdfA     = 1.0f / pLight->data[PLIGHT_SURFACE_AREA];
  a_outRes->pdfW     = pdfW;
  a_outRes->cosTheta = 1.0f;
  a_outRes->norm     = sampleDir;
}

static inline void PointSpotSampleForward(__global const PlainLight* pLight, float4 rands,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                          __private LightSampleFwd* a_outRes)
{ 
  const float3 lnorm   = lightNorm(pLight);
  const float3 lcenter = lightPos(pLight);
  const float3 color   = lightBaseColor(pLight);

  const float cos1     = pLight->data[POINT_LIGHT_SPOT_COS1];
  const float cos2     = pLight->data[POINT_LIGHT_SPOT_COS2];

  const float3 samplePos = lcenter;
  const float3 sampleDir = MapSamplesToCone(cos2, make_float2(rands.x, rands.y), lnorm); 

  const float cosThetaOut = fmax(dot(sampleDir, lnorm), 0.0f);
  const float k1          = mylocalsmoothstep(cos2, cos1, cosThetaOut);

  a_outRes->isPoint  = true;
  a_outRes->pos      = samplePos + epsilonOfPos(samplePos)*sampleDir;
  a_outRes->dir      = sampleDir;
  a_outRes->color    = k1*color*(1.0f / pLight->data[PLIGHT_SURFACE_AREA]);
  a_outRes->pdfA     = 1.0f / pLight->data[PLIGHT_SURFACE_AREA];
  a_outRes->pdfW     = 1.0f / (2.0f * M_PI * (1.0f - cos2));         // UniformConePdf
  a_outRes->cosTheta = cosThetaOut;
  a_outRes->norm     = sampleDir;
}

static inline float directLightAttenuation(__global const PlainLight* pLight, float3 illuminatingPoint)
{
  const float  radius1 = pLight->data[DIRECT_LIGHT_RADIUS1];
  const float  radius2 = pLight->data[DIRECT_LIGHT_RADIUS2];
  
  const float3 lpos = lightPos(pLight);
  const float3 norm = lightNorm(pLight);
  
  const float cos_alpha = dot(normalize(illuminatingPoint - lpos), norm);

  if (cos_alpha > 0.0f)
  {
    const float sinAlpha = sqrt(1.0f - cos_alpha*cos_alpha);
    const float d        = length(illuminatingPoint - lpos)*sinAlpha;
    const float maxd     = fmax(radius2, radius1);
    const float mind     = fmin(radius2, radius1);
    return mylocalsmoothstep(maxd, mind, d);
  }
  else
    return 0.0f;
}


static inline void DirectLightSampleForward(__global const PlainLight* pLight, float4 rands,
                                            __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                            __private LightSampleFwd* a_outRes)
{ 
  const float3 lnorm   = lightNorm(pLight);
  const float3 lcenter = lightPos(pLight);

  //const float bsphereR = a_globals->varsF[HRT_BSPHERE_RADIUS];
  const float  radius1 = pLight->data[DIRECT_LIGHT_RADIUS1];
  const float  radius2 = pLight->data[DIRECT_LIGHT_RADIUS2];

  const float2 diskSam = radius2*MapSamplesToDisc(2.0f*make_float2(rands.x - 0.5f, rands.y - 0.5f));
  const float d        = length(diskSam);
  const float maxd     = fmax(radius2, radius1);
  const float mind     = fmin(radius2, radius1);
  const float atten    = mylocalsmoothstep(maxd, mind, d);

  float3 nx, nz;
  CoordinateSystem(lnorm, &nx, &nz);

  float3 samplePos = lcenter + nx*diskSam.x + nz*diskSam.y;
  float3 sampleDir = lnorm; 
  float3 color     = lightBaseColor(pLight)*atten;

  float pdfW = 1.0f;
  if (pLight->data[DIRECT_LIGHT_SSOFTNESS] > 1e-5f)
  {
    const float cosAlpha = pLight->data[DIRECT_LIGHT_ALPHA_COS];
    sampleDir = MapSamplesToCone(cosAlpha, make_float2(rands.z, rands.w), lnorm);
    //pdfW = 1.0f / (2.0f * M_PI * (1.0f - cosAlpha));
  }

  a_outRes->isPoint  = true;
  a_outRes->pos      = samplePos + sampleDir*epsilonOfPos(samplePos);
  a_outRes->dir      = sampleDir;
  a_outRes->color    = color*pdfW; // *(M_PI*0.25f);
  a_outRes->pdfA     = 1.0f / pLight->data[PLIGHT_SURFACE_AREA];
  a_outRes->pdfW     = pdfW;
  a_outRes->cosTheta = 1.0f;
  a_outRes->norm     = sampleDir;
}

static inline float3 meshLightGetIntensity(__global const PlainLight* pLight, float2 texCoord, 
                                           __global const EngineGlobals* a_globals, texture2d_t a_texStorage)
{
  const int samplerOffset = as_int(pLight->data[MESH_LIGHT_TEXMATRIX_ID]);
  const float3 texColor   = sample2D(samplerOffset, texCoord, (__global const int4*)pLight, a_texStorage, a_globals);
  return texColor*lightBaseColor(pLight);
}


static inline void MeshLightSamplePos(__global const PlainLight* pLight, float3 rands, __global const float4* a_tableStorage, __global const EngineGlobals* a_globals,
                                      __private float3* pPos, __private float3* pNorm, __private float2* pTexCoord, __private float* pdfA)
{
  // extract mesh and table
  //
  const int meshId = as_int(pLight->data[MESH_LIGHT_MESH_OFFSET_ID]);
  const int pdftId = as_int(pLight->data[MESH_LIGHT_TABLE_OFFSET_ID]);
  const int triNum = as_int(pLight->data[MESH_LIGHT_TRI_NUM]);

  const int meshOffset = pdfTableHeaderOffset(meshId, a_globals);
  const int pdftOffset = pdfTableHeaderOffset(pdftId, a_globals);

  __global const PlainMesh* pMesh = (__global const PlainMesh*)(a_tableStorage + meshOffset);
  __global const float*     table = (__global const float*)    (a_tableStorage + pdftOffset);

  __global const float4* vpos  = meshVerts(pMesh);
  __global const float4* vnorm = meshNorms(pMesh);
  //__global const float2* texc  = meshTexCoords(pMesh);
  __global const int* indices  = meshTriIndices(pMesh);

  float pickProb = 1.0f;
  const int triangleId = SelectIndexPropToOpt(rands.z, table, triNum + 1, &pickProb);

  const int iA = indices[triangleId * 3 + 0];
  const int iB = indices[triangleId * 3 + 1];
  const int iC = indices[triangleId * 3 + 2];

  const float4 dataA  = vpos[iA];
  const float4 dataB  = vpos[iB];
  const float4 dataC  = vpos[iC];

  const float4 datanA = vnorm[iA];
  const float4 datanB = vnorm[iB];
  const float4 datanC = vnorm[iC];

  const float3 A  = to_float3(dataA);
  const float3 B  = to_float3(dataB);
  const float3 C  = to_float3(dataC);

  const float3 nA = to_float3(datanA);
  const float3 nB = to_float3(datanB);
  const float3 nC = to_float3(datanC);

  const float2 tA = make_float2(dataA.w, datanA.w); // texc[iA];
  const float2 tB = make_float2(dataB.w, datanB.w); // texc[iB];
  const float2 tC = make_float2(dataC.w, datanC.w); // texc[iC];
  
  // uniform barycentrics
  //
  float u = rands.x;
  float v = rands.y;
  if (u + v > 1.0f)
  {
    u = 1.0f - u;
    v = 1.0f - v;
  }
  const float w = 1.0f - u - v;

  (*pPos)      = (A*u  + B*v  + C*w);
  (*pNorm)     = (nA*u + nB*v + nC*w);
  (*pTexCoord) = (tA*u + tB*v + tC*w);
  (*pdfA)      = 1.0f/pLight->data[PLIGHT_SURFACE_AREA];
}


static inline void MeshLightSampleForward(__global const PlainLight* pLight, float4 rands, float2 rands2,
                                          __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                          __private LightSampleFwd* a_outRes)
{
  float3 samplePos, sampleNorm;
  float2 sampleTexCoord;
  float pdfA;

  MeshLightSamplePos(pLight, make_float3(rands.x, rands.y, rands2.x), a_tableStorage, a_globals,
                     &samplePos, &sampleNorm, &sampleTexCoord, &pdfA);

  __global const float* pMatrix = pLight->data + MESH_LIGHT_MATRIX_E00;
  samplePos  = matrix3x3f_mult_float3(pMatrix, samplePos);             // transform with rotation matrix
  sampleNorm = normalize(matrix3x3f_mult_float3(pMatrix, sampleNorm)); // transform with rotation matrix
  samplePos  = samplePos + lightPos(pLight);                           // translate to world light position

  const float3 sampleDir = MapSampleToCosineDistribution(rands.z, rands.w, sampleNorm, sampleNorm, 1.0f);
  const float  cosTheta  = fmax(dot(sampleDir, sampleNorm), 0.0f);
  const float  pdfW      = cosTheta*INV_PI;

  const float3 color = meshLightGetIntensity(pLight, sampleTexCoord, a_globals, a_tex); // lightBaseColor(pLight);

  a_outRes->isPoint  = false;
  a_outRes->pos      = samplePos + epsilonOfPos(samplePos)*sampleNorm;
  a_outRes->dir      = sampleDir;
  a_outRes->color    = color*cosTheta;
  a_outRes->pdfA     = 1.0f / pLight->data[PLIGHT_SURFACE_AREA]; 
  a_outRes->pdfW     = pdfW;
  a_outRes->cosTheta = cosTheta;
  a_outRes->norm     = sampleNorm;
}


static inline void LightSampleForward(__global const PlainLight* pLight, const float4 rands, const float2 rands2,
                                      __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                      __private LightSampleFwd* pOut)
{
  const int ltype = lightType(pLight);

  switch (ltype)
  {
    //case PLAIN_LIGHT_TYPE_SKY_DOME: 
    case PLAIN_LIGHT_TYPE_DIRECT:       
      DirectLightSampleForward(pLight, rands, a_globals, a_tex, a_tableStorage, 
                               pOut);
      break;

    case PLAIN_LIGHT_TYPE_POINT_SPOT:   
      PointSpotSampleForward(pLight, rands, a_globals, a_tex, a_tableStorage, 
                             pOut);
      break;

    case PLAIN_LIGHT_TYPE_POINT_OMNI:
      PointLightSampleForward(pLight, rands, a_globals, a_tex, a_tableStorage, 
                              pOut);
      break;

    case PLAIN_LIGHT_TYPE_SPHERE: 
      SphereLightSampleForward(pLight, rands, a_globals, a_tex, a_tableStorage, 
                               pOut);
      break;

    case PLAIN_LIGHT_TYPE_CYLINDER:
      CylinderLightSampleForward(pLight, rands, rands2, a_globals, a_tex, a_tableStorage, 
                                 pOut);
      break;

    case PLAIN_LIGHT_TYPE_MESH:
      MeshLightSampleForward(pLight, rands, rands2, a_globals, a_tex, a_tableStorage,
                             pOut);
      break;

    case PLAIN_LIGHT_TYPE_AREA:
    default: 
      AreaLightSampleForward(pLight, rands, rands2, a_globals, a_tex, a_tableStorage, 
                             pOut);
      break;
  }

}

static inline float lightPdfSelectFwd(__global const PlainLight* pLight)
{
  return pLight->data[PLIGHT_PICK_PROB_FWD];
}

static inline LightPdfFwd lightPdfFwd(__global const PlainLight* pLight, const float3 ray_dir, const float a_cosTheta,
                                      __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage)
{
  LightPdfFwd res;
  res.pdfA     = 1.0f / pLight->data[PLIGHT_SURFACE_AREA];
  res.pdfW     = fmax(a_cosTheta*INV_PI, 0.0f);
  res.pickProb = lightPdfSelectFwd(pLight);

  const int ltype = as_int(pLight->data[PLIGHT_TYPE]);

  if (ltype == PLAIN_LIGHT_TYPE_POINT_OMNI)
    res.pdfW = INV_PI*0.25f;
  else if (ltype == PLAIN_LIGHT_TYPE_POINT_SPOT)
  {
    const float cos2 = pLight->data[POINT_LIGHT_SPOT_COS2];
    res.pdfW = 1.0f / (2.0f * M_PI * (1.0f - cos2));        // UniformConePdf
    if (a_cosTheta < cos2)                                  // we hit light from the direction that is out of cone
      res.pdfW = 0.0f;
  }
  else if (ltype == PLAIN_LIGHT_TYPE_DIRECT)
  {
    const float  radius2 = pLight->data[DIRECT_LIGHT_RADIUS2];
    res.pdfA = 1.0f / (M_PI*radius2*radius2);
    res.pdfW = 0.0f;

    // if (pLight->data[DIRECT_LIGHT_SSOFTNESS] > 1e-5f)
    // {
    //   const float cosAlpha = pLight->data[DIRECT_LIGHT_ALPHA_COS];
    //   res.pdfW = 1.0f / (2.0f * M_PI * (1.0f - cosAlpha));
    //   if (a_cosTheta < cosAlpha)                                  // we hit light from the direction that is out of cone
    //     res.pdfW = 0.0f;
    // }
  }

  if (lightFlags(pLight) & LIGHT_HAS_IES)
  {
    __global const float* pMatrix = pLight->data + IES_LIGHT_MATRIX_E00;
    const float3 rayDir = matrix3x3f_mult_float3(pMatrix, ray_dir);

    int w, h;
    __global const float* table = lightIESPdfTable(pLight, a_globals, a_tableStorage,
                                                   &w, &h);
    float sintheta = 0.0f;
    const float2 texCoord = sphereMapTo2DTexCoord((-1.0f)*rayDir, &sintheta);
    const float mapPdf    = evalMap2DPdf(texCoord, table, w, h);
    res.pdfW              = mapPdf / (2.f * M_PI * M_PI * fmax(sintheta, DEPSILON2));
  }
  else if (ltype == PLAIN_LIGHT_TYPE_AREA && as_int(pLight->data[AREA_LIGHT_SPOT_DISTR]) != 0)
  {
    //const float cos1 = pLight->data[AREA_LIGHT_SPOT_COS1];
    const float cos2 = pLight->data[AREA_LIGHT_SPOT_COS2];

    res.pdfW = 1.0f / (2.0f * M_PI * (1.0f - cos2));        // UniformConePdf
    if (a_cosTheta < cos2)                                  // we hit light from the direction that is out of cone
      res.pdfW = 0.0f;
  }

  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void AreaLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint,
                                      __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                      __private ShadowSample* a_out)
{
  float offsetX = rands.x * 2.0f - 1.0f;
  float offsetY = rands.y * 2.0f - 1.0f;

  float3 samplePos;

  samplePos.x = offsetX*pLight->data[AREA_LIGHT_SIZE_X];
  samplePos.y = 0.0f;
  samplePos.z = offsetY*pLight->data[AREA_LIGHT_SIZE_Y];
  

  if (as_int(pLight->data[AREA_LIGHT_IS_DISK]) != 0)
  {
    float2 xz = MapSamplesToDisc(make_float2(offsetX, offsetY))*pLight->data[AREA_LIGHT_SIZE_X]; // disk radius
    samplePos.x = xz.x;
    samplePos.y = 0;
    samplePos.z = xz.y;
  }

  __global const float* pMatrix = pLight->data + AREA_LIGHT_MATRIX_E00;
  samplePos = matrix3x3f_mult_float3(pMatrix, samplePos); // transform with rotation matrix
  samplePos = samplePos + lightPos(pLight);               // translate to world light position

  float3 rayDir = normalize(samplePos - illuminatingPoint);
  float hitDist = length(samplePos - illuminatingPoint);

  float3 color;
  
  if (as_int(pLight->data[PLIGHT_FLAGS]) & AREA_LIGHT_SKY_PORTAL)
    color = areaLightSkyPortalCustomColor(pLight, rayDir, a_globals, a_tex, a_tableStorage);
  else
  {
    float3 customRayDir = rayDir;                                     // this is for point-area approximation
    if(lightFlags(pLight) & LIGHT_IES_POINT_AREA)                     // 
      customRayDir = normalize(lightPos(pLight) - illuminatingPoint); // 
    color = areaDiffuseLightGetIntensity(pLight, customRayDir, make_float2(rands.x, rands.y), false, a_globals, a_tex, a_tableStorage);
  }

  const float3 lnorm = lightNorm(pLight);

  a_out->isPoint     = false;
  a_out->pos         = samplePos + epsilonOfPos(samplePos)*lnorm;
  a_out->color       = color;
  a_out->pdf         = areaDiffuseLightEvalPDF(pLight, rayDir, hitDist);
  a_out->maxDist     = hitDist;
  a_out->cosAtLight  = -dot(rayDir, lnorm);
}


static inline float areaSpotAttenuationPredict(__global const PlainLight* pLight, float3 hpos)
{
  float3 samplePos = lightPos(pLight);
  float3 norm      = lightNorm(pLight);

  // plane equation Ax + By + Cz + D = 0;  => D = dot(samplePos, norm);
  //
  const float A = norm.x;
  const float B = norm.y;
  const float C = norm.z;
  const float D = -dot(samplePos, norm);

  // line equation; P(x,y,z) = O + D1*t;  
  //
  const float3 O  = hpos;
  const float3 D1 = (-1.0f)*norm;

  if (fabs(A*D1.x + B*D1.y + C*D1.z) < DEPSILON) //  
    return 0.0f;

  const float t = -(A*O.x + B*O.y + C*O.z + D) / (A*D1.x + B*D1.y + C*D1.z); // A*(O.x + t*D1.x) + B*(O.y + t*D1.y) + C*(O.z + t*D1.z) + D = 0

  const float3 projP = O + t*D1;
  const float3 projD = normalize(projP - samplePos);
  const float  projL = length(projP - samplePos);
  
  const float maxLightSize = fmax(pLight->data[AREA_LIGHT_SIZE_X], pLight->data[AREA_LIGHT_SIZE_Y]);
  const float3 lpos2       = samplePos + projD*fmin(projL, maxLightSize);

  const float cos1     = pLight->data[AREA_LIGHT_SPOT_COS1];
  const float cos2     = pLight->data[AREA_LIGHT_SPOT_COS2];
  const float cosTheta = fmax(dot(normalize(hpos - lpos2), norm), 0.0f);

  return mylocalsmoothstep(cos2, cos1, cosTheta);
}


static inline float areaPredictVisibility(__global const PlainLight* pLight, float3 illuminatingPoint)
{
  const float3 samplePos = lightPos(pLight);
  const float3 lnorm     = lightNorm(pLight);
  const float3 rayDir    = normalize(samplePos - illuminatingPoint);

  float cosVal = fmax(dot(rayDir, -1.0f*lnorm), 0.0f);
  if (as_int(pLight->data[AREA_LIGHT_SPOT_DISTR]) != 0)
    cosVal *= areaSpotAttenuationPredict(pLight, illuminatingPoint);

  if (as_int(pLight->data[PLIGHT_FLAGS]) & LIGHT_HAS_IES)
    return 1.0f;
  else
    return cosVal;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float sphereLightEvalPDF(__global const PlainLight* pLight, float3 illuminatingPoint, float3 lpos, float3 lnorm)
{
  float  lradius = pLight->data[SPHERE_LIGHT_RADIUS];
  float3 lcenter = lightPos(pLight);

  if (DistanceSquared(illuminatingPoint, lcenter) - lradius*lradius <= 0.0f)
    return 1.0f; // 

  const float  pdfA   = 1.0f / pLight->data[PLIGHT_SURFACE_AREA];
  const float  dist   = length(lpos - illuminatingPoint);

  const float3 dirToV    = normalize(lpos - illuminatingPoint);
  const float cosAtLight = fabs(dot(dirToV, lnorm));

  return PdfAtoW(pdfA, dist, cosAtLight);
}

static inline float3 sphereLightGetIntensity(__global const PlainLight* pLight)
{
  return lightBaseColor(pLight);
}

static inline void SphereLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint,
                                        __private ShadowSample* a_out)
{
  const float theta = 2.0f * M_PI * rands.x;
  const float phi   = acos(1.0f - 2.0f * rands.y);
  const float x     = sin(phi) * cos(theta);
  const float y     = sin(phi) * sin(theta);
  const float z     = cos(phi);

  const float3 lcenter = lightPos(pLight);
  const float  lradius = pLight->data[SPHERE_LIGHT_RADIUS];

  const float3 samplePos = lcenter + lradius*make_float3(x, y, z);
  const float3 lightNorm = normalize(samplePos - lcenter);
  const float3 dirToV    = normalize(samplePos - illuminatingPoint);

  a_out->isPoint    = false;
  a_out->pos        = samplePos;
  a_out->color      = sphereLightGetIntensity(pLight);
  a_out->pdf        = sphereLightEvalPDF(pLight, illuminatingPoint, samplePos, lightNorm);
  a_out->maxDist    = length(samplePos - illuminatingPoint);
  a_out->cosAtLight = fabs(dot(lightNorm, dirToV));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float cylinderLightEvalPDF(__global const PlainLight* pLight, float3 illuminatingPoint, float3 lpos, float3 lnorm, float2 texCoord, 
                                         __global const float4* a_tableStorage, __global const EngineGlobals* a_globals)
{
  float mapPdf = 1.0f;
  const int texId = as_int(pLight->data[CYLINDER_PDF_TABLE_ID]);
  if (texId)
  {
    __global const float* pdfHeader = pdfTableHeader(texId, a_tableStorage, a_globals);
    __global const float* intervals = pdfHeader + 4;
    const int sizeX = as_int(pdfHeader[0]);
    const int sizeY = as_int(pdfHeader[1]);
    mapPdf = evalMap2DPdf(texCoord, intervals, sizeX, sizeY);
  }

  const float hitDist = length(lpos - illuminatingPoint);
  const float3 rayDir = normalize(lpos - illuminatingPoint);
  const float pdfA    = mapPdf / fmax(pLight->data[PLIGHT_SURFACE_AREA], DEPSILON);
  const float cosVal  = fmax(dot(rayDir, -1.0f*lnorm), 0.0f);
  return PdfAtoW(pdfA, hitDist, cosVal);
}



static inline void CylinderLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint, 
                                          __global const EngineGlobals* a_globals, texture2d_t a_texStorage, __global const float4* a_tableStorage,
                                          __private ShadowSample* a_out)
{

  float3 samplePos, n; float2 texCoord; float pdfA;
  CylinderLightSamplePos(pLight, rands, a_globals, a_texStorage, a_tableStorage,
                         &samplePos, &n, &texCoord, &pdfA);

  const float hitDist = length(samplePos - illuminatingPoint);
  const float3 rayDir = normalize(samplePos - illuminatingPoint);
  const float  cosVal = fmax(dot(rayDir, -1.0f*n), 0.0f);

  a_out->isPoint    = false;
  a_out->pos        = samplePos;
  a_out->color      = cylinderLightGetIntensity(pLight, texCoord, a_globals, a_texStorage);
  a_out->pdf        = PdfAtoW(pdfA, hitDist, cosVal);
  a_out->maxDist    = hitDist;
  a_out->cosAtLight = cosVal;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float pointLightEvalPDF(__global const PlainLight* pLight, float3 illuminatingPoint)
{
  float3 samplePos = lightPos(pLight);
  float  hitDist   = length(samplePos - illuminatingPoint);
  return PdfAtoW(1.0f, hitDist, 1.0f);
}

static inline void PointLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint, __global const EngineGlobals* a_globals, __global const float4* a_tableStorage,
                                       __private ShadowSample* a_out)
{
  const float3 samplePos = lightPos(pLight);
  const float3 rayDir    = normalize(samplePos - illuminatingPoint);
  const float hitDist    = length(samplePos - illuminatingPoint);

  a_out->isPoint    = true;
  a_out->pos        = samplePos;
  a_out->color      = pointLightGetIntensity(pLight, rayDir, a_globals, a_tableStorage);
  a_out->pdf        = PdfAtoW(1.0f, hitDist, 1.0f);
  a_out->maxDist    = hitDist;
  a_out->cosAtLight = 1.0f;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



static inline float pointSpotLightAttenuation(__global const PlainLight* pLight, float3 in_shadowRayDir)
{
  const float  cos1 = pLight->data[POINT_LIGHT_SPOT_COS1];
  const float  cos2 = pLight->data[POINT_LIGHT_SPOT_COS2];
  const float3 norm = lightNorm(pLight);
  const float  cos_theta = fmax(dot(in_shadowRayDir, norm), 0.0f);
  return mylocalsmoothstep(cos2, cos1, cos_theta);
}

static inline float spotLightEvalPDF(__global const PlainLight* pLight, float3 illuminatingPoint)
{
  float3 samplePos = lightPos(pLight);
  float  hitDist = length(samplePos - illuminatingPoint);
  return PdfAtoW(1.0f, hitDist, 1.0f);
}

static inline void SpotLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint,
                                      __private ShadowSample* a_out)
{
  const float3 samplePos = lightPos(pLight);
  const float3 norm      = lightNorm(pLight);
  
  const float hitDist = length(samplePos - illuminatingPoint);
  const float3 rayDir = normalize(samplePos - illuminatingPoint);

  const float3 color    = lightBaseColor(pLight)*pointSpotLightAttenuation(pLight, (-1.0f)*rayDir);
  const float  cosTheta = fmax(-dot(rayDir, norm), 0.0f);

  a_out->isPoint    = true;
  a_out->pos        = samplePos;
  a_out->color      = color;
  a_out->pdf        = PdfAtoW(1.0f, hitDist, 1.0f);
  a_out->maxDist    = hitDist;
  a_out->cosAtLight = cosTheta;
}


static inline float spotPredictVisibility(__global const PlainLight* pLight, float3 illuminatingPoint)
{
  float3 samplePos = lightPos(pLight);
  float3 rayDir = normalize(samplePos - illuminatingPoint);
  return pointSpotLightAttenuation(pLight, (-1.0f)*rayDir);
}



static inline float directLightEvalPDF(__global const PlainLight* pLight, float3 ray_dir)
{
  if (pLight->data[DIRECT_LIGHT_SSOFTNESS] > 1e-5f)
  {
    const float3 norm    = lightNorm(pLight);

    //const float cosAlpha = pLight->data[DIRECT_LIGHT_ALPHA_COS];
    const float tanAlpha = pLight->data[DIRECT_LIGHT_ALPHA_TAN];
    const float cosTheta = -dot(ray_dir, norm);

    return M_PI*(tanAlpha*tanAlpha)*(cosTheta*cosTheta*cosTheta);
  }
  else
    return 1.0f;
}

static inline void DirectLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint,
                                        __private ShadowSample* a_out)
{
  float3 lpos = lightPos(pLight);
  float3 norm = lightNorm(pLight);

  float pdfW = 1.0f;
  if (pLight->data[DIRECT_LIGHT_SSOFTNESS] > 1e-5f)
  {
    const float cosAlpha = pLight->data[DIRECT_LIGHT_ALPHA_COS];
    norm = MapSamplesToCone(cosAlpha, make_float2(rands.x, rands.y), norm);
    //pdfW = 1.0f / (2.0f * M_PI * (1.0f - cosAlpha));
  }

  float3 AC    = illuminatingPoint - lpos;
  float  CBLen = dot(normalize(AC), norm)*length(AC);

  float3 samplePos = illuminatingPoint - norm*CBLen;
  float hitDist    = CBLen; // length(samplePos - illuminatingPoint);

  const float3 color = lightBaseColor(pLight)*directLightAttenuation(pLight, illuminatingPoint);

  a_out->isPoint    = true; // (pdfW == 1.0f);
  a_out->pos        = samplePos;
  a_out->color      = color*pdfW;
  a_out->pdf        = pdfW;
  a_out->maxDist    = hitDist;
  a_out->cosAtLight = 1.0f;
}

static inline float directPredictVisibility(__global const PlainLight* pLight, float3 illuminatingPoint)
{
  return directLightAttenuation(pLight, illuminatingPoint);
}

static inline void MeshLightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint,
                                      __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_tableStorage,
                                      __private ShadowSample* a_out)
{
  float3 samplePos, sampleNorm;
  float2 sampleTexCoord;
  float pdfA;

  MeshLightSamplePos(pLight, rands, a_tableStorage, a_globals,
                     &samplePos, &sampleNorm, &sampleTexCoord, &pdfA); 

  __global const float* pMatrix = pLight->data + MESH_LIGHT_MATRIX_E00;
  samplePos  = matrix3x3f_mult_float3(pMatrix, samplePos);             // transform with rotation matrix
  sampleNorm = normalize(matrix3x3f_mult_float3(pMatrix, sampleNorm)); // transform with rotation matrix
  samplePos  = samplePos + lightPos(pLight);                           // translate to world light position

  const float3 rayDir = normalize(samplePos - illuminatingPoint);
  const float hitDist = length(samplePos - illuminatingPoint);
  const float cosVal  = fmax(-dot(rayDir, sampleNorm), 0.0f);

  a_out->isPoint    = false;
  a_out->pos        = samplePos + epsilonOfPos(samplePos)*sampleNorm;
  a_out->color      = meshLightGetIntensity(pLight, sampleTexCoord, a_globals, a_tex); //lightBaseColor(pLight);
  a_out->pdf        = PdfAtoW(pdfA, hitDist, cosVal);
  a_out->maxDist    = hitDist;
  a_out->cosAtLight = cosVal;
}

static inline float meshLightEvalPDF(__global const PlainLight* pLight, float3 rayDir, const float3 lnorm, float hitDist)
{
  const float pdfA   = 1.0f / fmax(pLight->data[PLIGHT_SURFACE_AREA], DEPSILON);
  const float cosVal = fmax(dot(rayDir, -1.0f*lnorm), 0.0f);
  return PdfAtoW(pdfA, hitDist, cosVal);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
\brief  Sample light for reverse direction (i.e. from eye to light).
\param  pLight            - pointer to light
\param  rands             - random numbers in range [0,1]
\param  illuminatingPoint - a point on a surface that we want to illuminate with our sample
\param  a_globals         - engine globals
\param  a_pdfStorage      - float1 storage for tex and pdf tables
\param  a_tex             - HDR texture for cylinders, sky lights and e.t.c.
\param  a_out             - resulting sample

*/
static inline void LightSampleRev(__global const PlainLight* pLight, float3 rands, float3 illuminatingPoint, 
                                  __global const EngineGlobals* a_globals, __global const float4* a_pdfStorage, texture2d_t a_tex,
                                  __private ShadowSample* a_out)
{
  int lightType = as_int(pLight->data[PLIGHT_TYPE]);

  switch (lightType)
  {
  case PLAIN_LIGHT_TYPE_SKY_DOME:
    SkyLightSampleRev(pLight, rands, illuminatingPoint, a_globals, a_pdfStorage, a_tex,
                      a_out);
    break;

  case PLAIN_LIGHT_TYPE_DIRECT: 
    DirectLightSampleRev(pLight, rands, illuminatingPoint,
                         a_out);
    break;

  case PLAIN_LIGHT_TYPE_POINT_SPOT:
    SpotLightSampleRev(pLight, rands, illuminatingPoint,
                       a_out);
    break;

  case PLAIN_LIGHT_TYPE_POINT_OMNI:
    PointLightSampleRev(pLight, rands, illuminatingPoint, a_globals, a_pdfStorage,
                        a_out);
    break;

  case PLAIN_LIGHT_TYPE_SPHERE:       
    SphereLightSampleRev(pLight, rands, illuminatingPoint, 
                         a_out);
    break;

  case PLAIN_LIGHT_TYPE_CYLINDER: 
    CylinderLightSampleRev(pLight, rands, illuminatingPoint, a_globals, a_tex, a_pdfStorage,
                           a_out);
    break;

  case PLAIN_LIGHT_TYPE_MESH:
    MeshLightSampleRev(pLight, rands, illuminatingPoint, a_globals, a_tex, a_pdfStorage,
                       a_out);
    break;

  case PLAIN_LIGHT_TYPE_AREA:         
  default:                            
    AreaLightSampleRev(pLight, rands, illuminatingPoint, a_globals, a_tex, a_pdfStorage,
                       a_out);
    break;
  }
}


static inline float lightEvalPDF(__global const PlainLight* pLight, float3 illuminatingPoint, float3 rayDir, float3 lpos, float3 lnorm, float2 texCoord, 
                                 __global const float4* a_pdfTable, __global const EngineGlobals* a_globals)
{
  const float hitDist = length(illuminatingPoint - lpos);
  const int lightType = as_int(pLight->data[PLIGHT_TYPE]);
  
  switch (lightType)
  {
  //case PLAIN_LIGHT_TYPE_SKY_DOME:  return skyLightEvalPDF(pLight, rayDir, hitDist); // this happends in a different place of code, so do't have to put sky light pdf here
  case PLAIN_LIGHT_TYPE_DIRECT:       return directLightEvalPDF(pLight, rayDir);            // #TODO: cosAtLight
  case PLAIN_LIGHT_TYPE_POINT_SPOT:   return spotLightEvalPDF(pLight, illuminatingPoint);   // #TODO: cosAtLight
  case PLAIN_LIGHT_TYPE_POINT_OMNI:   return pointLightEvalPDF(pLight, illuminatingPoint);  // #TODO: cosAtLight
  case PLAIN_LIGHT_TYPE_SPHERE:       return sphereLightEvalPDF(pLight, illuminatingPoint, lpos, lnorm); // #TODO: cosAtLight
  case PLAIN_LIGHT_TYPE_CYLINDER:     return cylinderLightEvalPDF(pLight, illuminatingPoint, lpos, lnorm, texCoord, a_pdfTable, a_globals); // #TODO: cosAtLight
  case PLAIN_LIGHT_TYPE_MESH:         return meshLightEvalPDF(pLight, rayDir, lnorm, hitDist);
  case PLAIN_LIGHT_TYPE_AREA:        
  default:                            
                                      return areaDiffuseLightEvalPDF(pLight, rayDir, hitDist);
  }

}

static inline bool mltStrageCondition(const int flags, const int a_gflags, const MisData misPrev)
{
  return false;
}


static inline int hitDirectLight(const float3 ray_dir, __global const EngineGlobals* a_globals)
{
  int hitId = -1;

  for (int sunId = 0; sunId < a_globals->sunNumber; sunId++)
  {
    __global const PlainLight* pLight = a_globals->suns + sunId;
    float3 norm = lightNorm(pLight);

    const float cosAlpha = pLight->data[DIRECT_LIGHT_ALPHA_COS];
    const float cosTheta = -dot(ray_dir, norm);

    if (cosTheta > cosAlpha)
    {
      hitId = sunId;
      break;
    }

  }

  return hitId;
}



static inline float3 lightGetIntensity(__global const PlainLight* pLight, float3 ray_pos, float3 a_rayDir, float3 hitNorm, float2 a_texCoord, unsigned int flags, const MisData misPrev,
                                       __global const EngineGlobals* a_globals, texture2d_t a_tex, __global const float4* a_storagePdf) // float3 rayDir
{
  const bool eyeRay   = (unpackBounceNumDiff(flags) == 0);
  const int lightType = as_int(pLight->data[PLIGHT_TYPE]);

  //if (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE)
    //hitNorm = (-1.0f)*hitNorm;

  if ((as_int(pLight->data[PLIGHT_FLAGS]) & AREA_LIGHT_SKY_PORTAL) && unpackBounceNumDiff(flags) > 0)
  {
    const int hitId                 = hitDirectLight(a_rayDir, a_globals);
    const bool makeZeroBecauseOfMLT = mltStrageCondition(flags, a_globals->g_flags, misPrev);

    if (hitId >= 0) // hit any sun light
    {
      __global const PlainLight* pLight = a_globals->suns + hitId;
      float3 lightColor = lightBaseColor(pLight)*directLightAttenuation(pLight, ray_pos);

      const float pdfW = directLightEvalPDF(pLight, a_rayDir);

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////
      if (makeZeroBecauseOfMLT || (unpackBounceNum(flags) > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0))) 
        lightColor = make_float3(0, 0, 0);
      else if (((misPrev.isSpecular == 1) && (a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS)) || (a_globals->g_flags & HRT_STUPID_PT_MODE))
        lightColor *= (1.0f / pdfW);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////

      return lightColor;
    }
    else
      return areaLightSkyPortalCustomColor(pLight, a_rayDir, a_globals, a_tex, a_storagePdf);
  }
  else if (lightType == PLAIN_LIGHT_TYPE_AREA)
  {
    float3 customDir = a_rayDir;
    if (lightFlags(pLight) & LIGHT_IES_POINT_AREA)
      customDir = normalize(lightPos(pLight) - ray_pos); // 
    return areaDiffuseLightGetIntensity(pLight, customDir, a_texCoord, eyeRay, a_globals, a_tex, a_storagePdf);
  }
  else if (lightType == PLAIN_LIGHT_TYPE_CYLINDER)
    return cylinderLightGetIntensity(pLight, a_texCoord, a_globals, a_tex);
  else if (lightType == PLAIN_LIGHT_TYPE_MESH)
    return meshLightGetIntensity(pLight, a_texCoord, a_globals, a_tex);
  else
    return lightBaseColor(pLight);
}


static inline float lightPredictVisibility(__global const PlainLight* pLight, float3 illuminatingPoint)
{
  int lightType = as_int(pLight->data[PLIGHT_TYPE]);
  int flags     = as_int(pLight->data[PLIGHT_FLAGS]);

  if (flags & DISABLE_SAMPLING)
    return 0.0f;

  if (flags & LIGHT_HAS_IES)
    return 1.0f;

  float3 rayDir = normalize(lightPos(pLight) - illuminatingPoint);

  switch (lightType)
  {
  case PLAIN_LIGHT_TYPE_DIRECT:       return directPredictVisibility(pLight, illuminatingPoint);
  case PLAIN_LIGHT_TYPE_POINT_SPOT:   return spotPredictVisibility(pLight, illuminatingPoint);
  case PLAIN_LIGHT_TYPE_AREA:         return areaPredictVisibility(pLight, illuminatingPoint);
  //case PLAIN_LIGHT_TYPE_POINT_OMNI:   return length(lightDistributionMask(pLight, rayDir, a_globals, a_tex, a_texHdr));
  default:                            return 1.0f;
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __global const PlainLight* lightAt(__global const EngineGlobals* a_pGlobals, int lightId)
{
  __global const int* pBegin  = (__global const int*)a_pGlobals;
  __global const int* pTarget = pBegin + a_pGlobals->lightsOffset;
  return (((__global const PlainLight*)pTarget) + lightId);
}

static inline unsigned int rndIntFromFloatLocal(float r, unsigned int a, unsigned int b)
{
  const float fa = (float)a;
  const float fb = (float)b;
  const float fR = fa + r * (fb - fa);

  const unsigned int res = (unsigned int)(fR);

  if (res > b - 1)
    return b - 1;
  else
    return res;
}

/**
\brief select random visiable light.
\param a_r       - random in range [0,1]
\param hitPos    - position on surface which we are going to lit
\param a_globals - engine globals
\param pickProb  - out light pick probability
\return selected light offset in global instanced lights array

*/
static inline int SelectRandomLightRev(float2 a_r, float3 hitPos, __global const EngineGlobals* a_globals, 
                                       __private float* pickProb)
{
  const int tableSize = lightSelPdfTableSizeRev(a_globals);
  if (tableSize == 0)
  {
    (*pickProb) = 1.0f;
    return -1;
  }
  else if (tableSize <= 2)
  {
    (*pickProb) = 1.0f;
    return 0;
  }
  else
  {
    __global const float* table = lightSelPdfTableRev(a_globals);
    return SelectIndexPropToOpt(a_r.x, table, tableSize, pickProb);
  }
}

static inline float lightPdfSelectRev(__global const PlainLight* pLight)
{
  return pLight->data[PLIGHT_PICK_PROB_REV];
}

/**
\brief select random light
\param a_r       - random in range [0,1]
\param a_globals - engine globals
\param pickProb  - out light pick probability
\return selected light offset in global instanced lights array

*/
static inline int SelectRandomLightFwd(float2 a_r, __global const EngineGlobals* a_globals, 
                                       __private float* pickProb)
{
  const int tableSize = lightSelPdfTableSizeFwd(a_globals);
  if (tableSize <= 2)
  {
    (*pickProb) = 1.0f;
    return 0;
  }
  else
  {
    __global const float* table = lightSelPdfTableFwd(a_globals);
    return SelectIndexPropToOpt(a_r.x, table, tableSize, pickProb);
  }
}




#endif


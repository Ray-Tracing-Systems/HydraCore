#ifndef CTFETCH
#define CTFETCH

#include "cglobals.h"

#define LIGHT_DATA_SIZE 128

struct PlainLightT
{
  float data[LIGHT_DATA_SIZE];
};

typedef struct PlainLightT PlainLight;

/////////////////////////////////////////////////////////////////

#define MAX_SKY_PDFS 32
#define MAX_SUN_NUM  8

typedef struct GlobalRenderDataT
{
  //////////////////////////////////////////////////////////////////////////////////////////////

  float mProj[16];
  float mWorldView[16];

  float mProjInverse[16];
  float mWorldViewInverse[16];

  int   varsI[GMAXVARS];
  float varsF[GMAXVARS];

  float camForward[3];  ///< needed for light tracing
  float imagePlaneDist; ///< needed for light tracing

  ///////////////////////////////////////////////////////////////////////////////////////////////

  int texturesTableOffset;
  int materialsTableOffset;
  int pdfTableTableOffset;     
  int geometryTableOffset;
  int texturesAuxTableOffset;

  int texturesTableSize;
  int materialsTableSize;
  int pdfTableTableSize;       
  int geometryTableSize;
  int texturesAuxTableSize;

  int floatArraysOffset;    // ??
  int floatsArraysSize;     // ??

  int lightSelectorTableOffsetRev;
  int lightSelectorTableSizeRev;
  int lightSelectorTableOffsetFwd;
  int lightSelectorTableSizeFwd;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  int g_flags;
  int skyLightId;         
  int lightsOffset;       
  int lightsSize;         
                          
  int lightsNum;
  int dummy1;
  int dummy2;
  int dummy3;


  int        sunNumber;           // #change this?
  PlainLight suns[MAX_SUN_NUM];   // #change this?


} EngineGlobals;


typedef struct SWTextureHeaderT
{
  int width;
  int height;
  int depth;
  int bpp;

  // int mips;

} SWTextureHeader;


typedef struct SWTexSamplerT
{
  int      flags;
  float    gamma;
  int      texId;
  int      dummy2;

  float4 row0;
  float4 row1;

} SWTexSampler;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int meshHeaderOffset(const Lite_Hit a_liteHit, __global const EngineGlobals* a_pGlobals)
{
  __global const int* pBegin = (__global const int*)a_pGlobals;
  return pBegin[a_pGlobals->geometryTableOffset + a_liteHit.geomId];
}

static inline int textureHeaderOffset(__global const EngineGlobals* a_pGlobals, const int a_index)
{
  __global const int* pBegin = (__global const int*)a_pGlobals;
  return pBegin[a_pGlobals->texturesTableOffset + a_index];
}

static inline int textureAuxHeaderOffset(__global const EngineGlobals* a_pGlobals, const int a_index)
{
  __global const int* pBegin = (__global const int*)a_pGlobals;
  return pBegin[a_pGlobals->texturesAuxTableOffset + a_index];
}

static inline int pdfTableHeaderOffset(const int a_tableId, __global const EngineGlobals* a_pGlobals)
{
  __global const int* pBegin = (__global const int*)a_pGlobals;
  return pBegin[a_pGlobals->pdfTableTableOffset + a_tableId];
}

static inline __global const float* pdfTableHeader(const int a_tableId, __global const float4* a_pdfTableStorage, __global const EngineGlobals* a_pGlobals)
{
  const int offset = pdfTableHeaderOffset(a_tableId, a_pGlobals);
  return (__global const float*)(a_pdfTableStorage + offset);
}


static inline __global const float* floatArraysPtr(__global const EngineGlobals* a_pGlobals)
{
  __global const int* pBegin = (__global const int*)a_pGlobals;
  __global const int* pTarget = pBegin + a_pGlobals->floatArraysOffset;
  return (__global const float*)pTarget;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __global const float* lightSelPdfTableRev(__global const EngineGlobals* a_pGlobals)
{
  __global const int* pBegin = (__global const int*)a_pGlobals;
  __global const int* pTarget = pBegin + a_pGlobals->lightSelectorTableOffsetRev;
  return (__global const float*)pTarget;
}

static inline __global const float* lightSelPdfTableFwd(__global const EngineGlobals* a_pGlobals)
{
  __global const int* pBegin = (__global const int*)a_pGlobals;
  __global const int* pTarget = pBegin + a_pGlobals->lightSelectorTableOffsetFwd;
  return (__global const float*)pTarget;
}

static inline int lightSelPdfTableSizeFwd(__global const EngineGlobals* a_pGlobals) { return a_pGlobals->lightSelectorTableSizeFwd; }
static inline int lightSelPdfTableSizeRev(__global const EngineGlobals* a_pGlobals) { return a_pGlobals->lightSelectorTableSizeRev; }

static inline int materialOffset(__global const EngineGlobals* a_pGlobals, const int matId)
{
  __global const int*    pBegin = (__global const int*)a_pGlobals;
  __global const int*    ids = pBegin + a_pGlobals->materialsTableOffset;
  return ids[matId];
}

static inline __global const PlainMaterial* materialAtOffset(__global const float4* a_mltStorage, const int matOffset)
{
  return (__global const PlainMaterial*)(a_mltStorage + matOffset);
}

static inline __global const PlainMaterial* materialAt(__global const EngineGlobals* a_pGlobals, __global const float4* a_mltStorage, const int matId)
{
  if (matId == -1)
    return 0;
  else
  {
    const int matOffset = materialOffset(a_pGlobals, matId);
    return materialAtOffset(a_mltStorage, matOffset);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float2 projectedPixelSize2(float dist, __global const EngineGlobals* a_globals)
{
  const float size = projectedPixelSize(dist, a_globals->varsF[HRT_FOV_X], a_globals->varsF[HRT_WIDTH_F], a_globals->varsF[HRT_HEIGHT_F]);
  return make_float2(size, size);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float2 sphereMapToPhiTheta(float3 ray_dir)
{
  const float x = ray_dir.z;
  const float y = ray_dir.x;
  const float z = -ray_dir.y;
                             // r == 1.0f
  float theta = acos(z);     // [0,pi] 
  float phi   = atan2(y, x); // [-pi,pi]
  if (phi < 0.0f)
    phi += 2.0f*M_PI;        // [-pi,pi] --> [0, 2*pi];  see PBRT.

  return make_float2(phi, theta);
}

static inline float2 sphereMapTo2DTexCoord(float3 ray_dir, __private float* pSinTheta) // should be consistent with sphereMapToPhiTheta
{
  const float2 angles = sphereMapToPhiTheta(ray_dir);

  const float texX = clamp(angles.x*0.5f*INV_PI, 0.0f, 1.0f);
  const float texY = clamp(angles.y*INV_PI, 0.0f, 1.0f);

  (*pSinTheta) = sin(angles.y);
  return make_float2(texX, texY);
}

static inline float3 texCoord2DToSphereMap(float2 a_texCoord, __private float* pSinTheta) // reverse to sphereMapTo2DTexCoord 
{
  const float phi   = a_texCoord.x * 2.f * M_PI; // see PBRT coords:  Float phi = uv[0] * 2.f * Pi;
  const float theta = a_texCoord.y * M_PI;       // see PBRT coords:  Float theta = uv[1] * Pi

  const float sinTheta = sin(theta);

  const float x = sinTheta*cos(phi);           // see PBRT coords: (Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta)
  const float y = sinTheta*sin(phi);
  const float z = cos(theta);

  (*pSinTheta)  = sinTheta;
  return make_float3(y, -z, x);
}

static inline float4 read_array_uchar4(__global const uchar4* a_data, int offset)
{
  const float mult = 0.003921568f; // (1.0f/255.0f);
  const uchar4 c0  = a_data[offset];
  return mult*make_float4((float)c0.x, (float)c0.y, (float)c0.z, (float)c0.w);
}

static inline int4 bilinearOffsets(const float ffx, const float ffy, const int a_flags, const int w, const int h)
{
	const int sx = (ffx > 0.0f) ? 1 : -1;
	const int sy = (ffy > 0.0f) ? 1 : -1;

	const int px = (int)(ffx);
	const int py = (int)(ffy);

	int px_w0, px_w1, py_w0, py_w1;

	if (a_flags & TEX_CLAMP_U)
	{
		px_w0 = (px     >= w) ? w - 1 : px;
		px_w1 = (px + 1 >= w) ? w - 1 : px + 1;

		px_w0 = (px_w0 < 0) ? 0 : px_w0;
		px_w1 = (px_w1 < 0) ? 0 : px_w1;
	}
	else
	{
		px_w0 = px        % w;
		px_w1 = (px + sx) % w;

		px_w0 = (px_w0 < 0) ? px_w0 + w : px_w0;
		px_w1 = (px_w1 < 0) ? px_w1 + w : px_w1;
	}

	if (a_flags & TEX_CLAMP_V)
	{
		py_w0 = (py     >= h) ? h - 1 : py;
		py_w1 = (py + 1 >= h) ? h - 1 : py + 1;

		py_w0 = (py_w0 < 0) ? 0 : py_w0;
		py_w1 = (py_w1 < 0) ? 0 : py_w1;
	}
	else
	{
		py_w0 = py        % h;
		py_w1 = (py + sy) % h;

		py_w0 = (py_w0 < 0) ? py_w0 + h : py_w0;
		py_w1 = (py_w1 < 0) ? py_w1 + h : py_w1;
	}

	const int offset0 = py_w0*w + px_w0;
	const int offset1 = py_w0*w + px_w1;
	const int offset2 = py_w1*w + px_w0;
	const int offset3 = py_w1*w + px_w1;

	return make_int4(offset0, offset1, offset2, offset3);
}


static inline float4 read_imagef_sw4(texture2d_t a_tex, const float2 a_texCoord, const int a_flags)
{
  const int4 header = (*a_tex);

  const int w   = header.x;
  const int h   = header.y;
  //const int d   = header.z;
  const int bpp = header.w;

  const float fw  = (float)(w);
  const float fh  = (float)(h);

  float ffx = a_texCoord.x*fw - 0.5f;
  float ffy = a_texCoord.y*fh - 0.5f;

  if ((a_flags & TEX_CLAMP_U) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_flags & TEX_CLAMP_V) != 0 && ffy < 0) ffy = 0.0f;

  float4 res;

  if (a_flags & TEX_POINT_SAM)
  {
    int px = (int)(ffx + 0.5f);
    int py = (int)(ffy + 0.5f);

    if (a_flags & TEX_CLAMP_U)
    {
      px = (px >= w) ? w - 1 : px;
      px = (px < 0) ? 0 : px;
    }
    else
    {
      px = px % w;
      px = (px < 0) ? px + w : px;
    }

    if (a_flags & TEX_CLAMP_V)
    {
      py = (py >= h) ? h - 1 : py;
      py = (py < 0) ? 0 : py;
    }
    else
    {
      py = py % h;
      py = (py < 0) ? py + h : py;
    }

    const int offset = py*w + px;

    if (bpp == 4)
      res = read_array_uchar4((__global const uchar4*)(a_tex + 1), offset);
    else if (bpp == 16)
    {
      __global const float4* fdata = (__global const float4*)(a_tex + 1);
      res = fdata[offset];
    }
  }
  else
  {
    // Calculate the weights for each pixel
    //
    const int   px = (int)(ffx);
    const int   py = (int)(ffy);

    const float fx = fabs(ffx - (float)px);
    const float fy = fabs(ffy - (float)py);
    const float fx1 = 1.0f - fx;
    const float fy1 = 1.0f - fy;

    const float w1 = fx1 * fy1;
    const float w2 = fx  * fy1;
    const float w3 = fx1 * fy;
    const float w4 = fx  * fy;

    const int4 offsets = bilinearOffsets(ffx, ffy, a_flags, w, h);

    // fetch pixels
    //
    float4 f1, f2, f3, f4;

    if (bpp == 4)
    {
      f1 = read_array_uchar4((__global const uchar4*)(a_tex + 1), offsets.x);
      f2 = read_array_uchar4((__global const uchar4*)(a_tex + 1), offsets.y);
      f3 = read_array_uchar4((__global const uchar4*)(a_tex + 1), offsets.z);
      f4 = read_array_uchar4((__global const uchar4*)(a_tex + 1), offsets.w);
    }
    else if (bpp == 16)
    {
      __global const float4* fdata = (__global const float4*)(a_tex + 1);

      f1 = fdata[offsets.x];
      f2 = fdata[offsets.y];
      f3 = fdata[offsets.z];
      f4 = fdata[offsets.w];
    }

    // Calculate the weighted sum of pixels (for each color channel)
    //
    const float outr = f1.x * w1 + f2.x * w2 + f3.x * w3 + f4.x * w4;
    const float outg = f1.y * w1 + f2.y * w2 + f3.y * w3 + f4.y * w4;
    const float outb = f1.z * w1 + f2.z * w2 + f3.z * w3 + f4.z * w4;
    const float outa = f1.w * w1 + f2.w * w2 + f3.w * w3 + f4.w * w4;

    res = make_float4(outr, outg, outb, outa);
  }

  return res;
}

static inline float read_imagef_sw1(texture2d_t a_tex, const float2 a_texCoord, const int a_flags)
{
  const int4 header = (*a_tex);

  const int w   = header.x;
  const int h   = header.y;

  const float fw  = (float)(w);
  const float fh  = (float)(h);

  float ffx = a_texCoord.x*fw - 0.5f;
  float ffy = a_texCoord.y*fh - 0.5f;

  if ((a_flags & TEX_CLAMP_U) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_flags & TEX_CLAMP_V) != 0 && ffy < 0) ffy = 0.0f;

  // fetch pixels
  //  
  __global const float* fdata = (__global const float*)(a_tex + 1);

  float res;
  if(a_flags & TEX_POINT_SAM)
  {
    int px = (int)(ffx + 0.5f);
    int py = (int)(ffy + 0.5f);

    if (a_flags & TEX_CLAMP_U)
    {
      px = (px >= w) ? w - 1 : px;
      px = (px < 0)  ? 0     : px;
    }
    else
    {
      px = px % w;
      px = (px < 0) ? px + w : px;
    }

    if (a_flags & TEX_CLAMP_V)
    {
      py = (py >= h) ? h - 1 : py;
      py = (py < 0)  ? 0     : py;
    }
    else
    {
      py = py % h;
      py = (py < 0) ? py + h : py;
    }
    
    res = fdata[py*w + px];
  }
  else
  {
    const int px = (int)(ffx);
    const int py = (int)(ffy);

    // Calculate the weights for each pixel
    //
    const float fx = fabs(ffx - (float)px);
    const float fy = fabs(ffy - (float)py);
    const float fx1 = 1.0f - fx;
    const float fy1 = 1.0f - fy;

    const float w1 = fx1 * fy1;
    const float w2 = fx  * fy1;
    const float w3 = fx1 * fy;
    const float w4 = fx  * fy;

    const int4 offsets = bilinearOffsets(ffx, ffy, a_flags, w, h);

    const float f1 = fdata[offsets.x];
    const float f2 = fdata[offsets.y];
    const float f3 = fdata[offsets.z];
    const float f4 = fdata[offsets.w];

    res = f1 * w1 + f2 * w2 + f3 * w3 + f4 * w4;
  }

  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline SWTexSampler ReadSampler(__global const int4* a_samStorage, int a_samplerOffset)
{
  SWTexSampler res;

  __global const float4* a_samStoragef = (__global const float4*)a_samStorage;
  
  const int4 header = a_samStorage[a_samplerOffset + 0];
  
  res.flags  = header.x;
  res.gamma  = as_float(header.y);
  res.texId  = header.z;
  res.dummy2 = header.w;
  res.row0   = a_samStoragef[a_samplerOffset + 1];
  res.row1   = a_samStoragef[a_samplerOffset + 2];
 
  //__global const float* dataF = (__global const float*)(a_samStorage + a_samplerOffset);
  //
  //res.flags  = as_int(dataF[0]);
  //res.gamma  = dataF[1];
  //res.texId  = as_int(dataF[2]);
  //res.dummy2 = as_int(dataF[3]);
  //
  //res.row0   = make_float4(dataF[4 + 0], dataF[4 + 1], dataF[4 + 2], dataF[4 + 3]);
  //res.row1   = make_float4(dataF[8 + 0], dataF[8 + 1], dataF[8 + 2], dataF[8 + 3]);

  return res;
}

static inline SWTexSampler ReadSamplerUI2(__global const uint2* a_samStorage, int a_samplerOffset)
{
  uint2 header1 = a_samStorage[a_samplerOffset + 0];
  uint2 header2 = a_samStorage[a_samplerOffset + 1];

  uint2 header3 = a_samStorage[a_samplerOffset + 2];
  uint2 header4 = a_samStorage[a_samplerOffset + 3];

  uint2 header5 = a_samStorage[a_samplerOffset + 4];
  uint2 header6 = a_samStorage[a_samplerOffset + 5];

  SWTexSampler res;

  res.flags  = header1.x;
  res.gamma  = as_float(header1.y);
  res.texId  = header2.x;
  res.dummy2 = header2.y;
  res.row0   = make_float4(as_float(header3.x), as_float(header3.y), as_float(header4.x), as_float(header4.y));
  res.row1   = make_float4(as_float(header5.x), as_float(header5.y), as_float(header6.x), as_float(header6.y));

  return res;
}




static inline float2 mul2x4(const float4 row0, const float4 row1, float2 v)
{
  float2 res;
  res.x = row0.x*v.x + row0.y*v.y + row0.w;
  res.y = row1.x*v.x + row1.y*v.y + row1.w;
  return res;
}

static inline float3 sample2D(int a_samplerOffset, float2 texCoord, __global const int4* a_samStorage, __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  if(a_samplerOffset == INVALID_TEXTURE || a_samplerOffset < 0)
    return make_float3(1, 1, 1);

  const SWTexSampler sampler = ReadSampler(a_samStorage, a_samplerOffset); 

  if (sampler.texId == 0)
    return make_float3(1, 1, 1);

  const float2 texCoordT = mul2x4(sampler.row0, sampler.row1, texCoord);

  const int offset = textureHeaderOffset(a_globals, sampler.texId);
  if(offset < 0)
    return make_float3(1, 1, 1);

  float4 texColor4 = read_imagef_sw4(a_texStorage + offset, texCoordT, sampler.flags); 

  texColor4.x = pow(texColor4.x, sampler.gamma);
  texColor4.y = pow(texColor4.y, sampler.gamma);
  texColor4.z = pow(texColor4.z, sampler.gamma);

  if (sampler.flags & TEX_ALPHASRC_W)
  {
    texColor4.x = texColor4.w;
    texColor4.y = texColor4.w;
    texColor4.z = texColor4.w;
  }

  return to_float3(texColor4);
}

static inline float3 sample2DExt(int a_samplerOffset, float2 texCoord, 
                                 __global const int4* a_samStorage, __global const int4* a_texStorage, 
                                 __global const EngineGlobals* a_globals, __private const ProcTextureList* a_ptList)
{
  if (a_samplerOffset == INVALID_TEXTURE || a_samplerOffset < 0)
    return make_float3(1, 1, 1);

  const SWTexSampler sampler = ReadSampler(a_samStorage, a_samplerOffset);

  if (sampler.texId <= 0)
    return make_float3(1, 1, 1);

  const float2 texCoordT = mul2x4(sampler.row0, sampler.row1, texCoord);

  int offset = textureHeaderOffset(a_globals, sampler.texId);
  float4 texColor2;
  if (offset >= 0)
    texColor2 = read_imagef_sw4(a_texStorage + offset, texCoordT, sampler.flags);
  else
    texColor2 = make_float4(1, 1, 1, 1);

  //float4 texColor4 = make_float4(1, 1, 1, -1.0f);
  float4 texColor1 = readProcTex(sampler.texId, a_ptList);

  float4 texColor4 = (fabs(texColor1.w + 1.0f) < 1e-5f) ? texColor2 : texColor1;
  
  texColor4.x = pow(texColor4.x, sampler.gamma);
  texColor4.y = pow(texColor4.y, sampler.gamma);
  texColor4.z = pow(texColor4.z, sampler.gamma);

  if (sampler.flags & TEX_ALPHASRC_W)
  {
    texColor4.x = texColor4.w;
    texColor4.y = texColor4.w;
    texColor4.z = texColor4.w;
  }

  return to_float3(texColor4);
}

// we do need two different functions -- sample2D and sample2DUI2 to call ReadSamplerUI2 inside
// this is due to pointer cast from uint2 to uint4 or float4 break aligned read on Nvidia;

static inline float3 sample2DUI2(int a_samplerOffset, float2 texCoord, __global const uint2* a_samStorage, __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  if (a_samplerOffset == INVALID_TEXTURE)
    return make_float3(1, 1, 1);

  const SWTexSampler sampler = ReadSamplerUI2(a_samStorage, a_samplerOffset);
  const float2 texCoordT     = mul2x4(sampler.row0, sampler.row1, texCoord);
  
  if (sampler.texId == 0)
    return make_float3(1, 1, 1);

  const int offset = textureHeaderOffset(a_globals, sampler.texId);

  float4 texColor4 = read_imagef_sw4(a_texStorage + offset, texCoordT, sampler.flags);

  texColor4.x = pow(texColor4.x, sampler.gamma);
  texColor4.y = pow(texColor4.y, sampler.gamma);
  texColor4.z = pow(texColor4.z, sampler.gamma);

  if (sampler.flags & TEX_ALPHASRC_W)
  {
    texColor4.x = texColor4.w;
    texColor4.y = texColor4.w;
    texColor4.z = texColor4.w;
  }

  return to_float3(texColor4);
}

static inline float3 sample2DLite(int a_samplerOffset, float2 texCoord, __global const uint2* a_samStorage, __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  if (a_samplerOffset == INVALID_TEXTURE || a_samplerOffset < 0)
    return make_float3(1, 1, 1);

  SWTexSampler sampler   = ReadSamplerUI2(a_samStorage, a_samplerOffset);
  const float2 texCoordT = mul2x4(sampler.row0, sampler.row1, texCoord);

  if(sampler.texId == 0)
    return make_float3(1, 1, 1);

  const int offset = textureHeaderOffset(a_globals, sampler.texId);

  float4 texColor4 = read_imagef_sw4(a_texStorage + offset, texCoordT, sampler.flags);

  if (sampler.flags & TEX_ALPHASRC_W)
  {
    texColor4.x = texColor4.w;
    texColor4.y = texColor4.w;
    texColor4.z = texColor4.w;
  }

  return to_float3(texColor4);
}

static inline float3 sample2DAux(int2 a_samplerOffset, float2 texCoord, __global const int4* a_samStorage, __global const int4* a_texStorage, __global const EngineGlobals* a_globals)
{
  if (a_samplerOffset.y == INVALID_TEXTURE)
    return make_float3(1, 1, 1);

  const SWTexSampler sampler = ReadSampler(a_samStorage, a_samplerOffset.y);
  const float2 texCoordT     = mul2x4(sampler.row0, sampler.row1, texCoord);

  if (sampler.texId == 0)
    return make_float3(1, 1, 1);

  const int offset = textureAuxHeaderOffset(a_globals, a_samplerOffset.x);
  //const int offset = textureAuxHeaderOffset(a_globals, sampler.texId);

  float4 texColor4 = read_imagef_sw4(a_texStorage + offset, texCoordT, sampler.flags);

  texColor4.x = pow(texColor4.x, sampler.gamma);
  texColor4.y = pow(texColor4.y, sampler.gamma);
  texColor4.z = pow(texColor4.z, sampler.gamma);

  if (sampler.flags & TEX_ALPHASRC_W)
  {
    texColor4.x = texColor4.w;
    texColor4.y = texColor4.w;
    texColor4.z = texColor4.w;
  }

  return to_float3(texColor4);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Basically the sensor plane is perpendicular to the viewing
// direction, the optical axis which intersects its center.
// Focusing can be done by moving the sensor back and forth
// in the optical axis.The sensor is called shifted in case it
// is moved laterally (within the sensor plane) and tilted if it
// is not oriented perpendicularly to the optical axis.
//

ID_CALL float3 tiltCorrection(float3 ray_pos, float3 ray_dir, __global const EngineGlobals* a_globals)
{
  float tiltX = a_globals->varsF[HRT_TILT_ROT_X];
  float tiltY = a_globals->varsF[HRT_TILT_ROT_Y];

  if ((fabs(tiltX) > 0.0f || fabs(tiltY) > 0.0) && fabs(ray_dir.z) > 0.0f) // tilt shift is enabled
  {
    // (1) intersect rays with plane at (0,0,-1)

    float  t = (-1.0f - ray_pos.z) / ray_dir.z;
    float3 p = ray_pos + t*ray_dir;

    // (2) rotate intersection point around (0,0,-1) for known tilt angles { +(0,0,1), rotate, -(0,0,1) }

    p.z += 1.0f;

    // rotate

    if (fabs(tiltY) > 0.0)
      p = mul(make_matrix_rotationY(-tiltY), p);

    if (fabs(tiltX) > 0.0f)
      p = mul(make_matrix_rotationX(tiltX), p);

    p.z -= 1.0f;

    // (3) calc new normalized ray_dirs
    ray_dir = normalize(p - ray_pos);
  }

  return ray_dir;
}


/**
\brief  Generate random ray through pixel.
\param  x         - pixel x coordinate. Should be in interval [0, w-1].
\param  y         - pixel y coordinate. Should be in interval [0, h-1].
\param  offsets   - input  randoms in range [-1,1] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Please Note this!!!
\param  a_globals - engine globals.
\param  pRayPos   - out ray position.
\param  pRayDir   - out ray direction.

*/

static inline void MakeRandEyeRay(int x, int y, int w, int h, float4 offsets, __global const EngineGlobals* a_globals, 
                                  __private float3* pRayPos, __private float3* pRayDir)
{
 
  const float4x4 a_mViewProjInv  = make_float4x4(a_globals->mProjInverse);
  const float4x4 a_mWorldViewInv = make_float4x4(a_globals->mWorldViewInverse);

  float2 screenPosF = make_float2((float)x, (float)y);
  // area to render -> blow up
  //{
  //  float fw = ((float)w);
  //  float fh = ((float)h);
  //
  //  float offsScaledX = a_globals->varsF[HRT_ABLOW_OFFSET_X] * fw * (+1.0f);
  //  float offsScaledY = a_globals->varsF[HRT_ABLOW_OFFSET_Y] * fh * (-1.0f);
  //
  //  screenPosF.x = (screenPosF.x - 0.5f*fw) * a_globals->varsF[HRT_ABLOW_SCALE_X] + offsScaledX + 0.5f*fw;
  //  screenPosF.y = (screenPosF.y - 0.5f*fh) * a_globals->varsF[HRT_ABLOW_SCALE_Y] + offsScaledY + 0.5f*fh;
  //}

  float3 ray_pos = make_float3(0.0f, 0.0f, 0.0f);
  float3 ray_dir = EyeRayDir(screenPosF.x, screenPosF.y, (float)w, (float)h, a_mViewProjInv);

  // simple AA for PT
  //
  if (true)
  {
    const float sinFov  = sin(0.5f*a_globals->varsF[HRT_CAM_FOV]);
    const float pxSizeX = sinFov*(1.0f / (float)w);
    const float pxSizeY = sinFov*(1.0f / (float)h);

    ray_dir.x += pxSizeX*offsets.x;
    ray_dir.y += pxSizeY*offsets.y;
    ray_dir.z = -sqrt(1.0f - (ray_dir.x*ray_dir.x + ray_dir.y*ray_dir.y));
  }

  ray_dir = tiltCorrection(ray_pos, ray_dir, a_globals);

  if (a_globals->varsI[HRT_ENABLE_DOF] == 1)
  {
    const float  tFocus        = a_globals->varsF[HRT_DOF_FOCAL_PLANE_DIST] / (-ray_dir.z);
    const float3 focusPosition = ray_pos + ray_dir*tFocus;
    const float2 xy            = a_globals->varsF[HRT_DOF_LENS_RADIUS] * MapSamplesToDisc(1.0f*make_float2(offsets.z, offsets.w));
    ray_pos.x += xy.x;
    ray_pos.y += xy.y;

    ray_dir = normalize(focusPosition - ray_pos);
  }

  matrix4x4f_mult_ray3(a_mWorldViewInv, &ray_pos, &ray_dir);

  (*pRayPos) = ray_pos;
  (*pRayDir) = ray_dir;
}


static inline void MakeEyeRayFromF4Rnd(float4 lensOffs, __global const EngineGlobals* a_globals,
                                      __private float3* pRayPos, __private float3* pRayDir, __private float* pX, __private float* pY)
{
  const float fwidth  = a_globals->varsF[HRT_WIDTH_F];
  const float fheight = a_globals->varsF[HRT_HEIGHT_F];

  const float xPosPs = lensOffs.x;
  const float yPosPs = lensOffs.y;
  const float x      = fwidth*xPosPs;
  const float y      = fheight*yPosPs;

  const float4x4 a_mViewProjInv  = make_float4x4(a_globals->mProjInverse);
  const float4x4 a_mWorldViewInv = make_float4x4(a_globals->mWorldViewInverse);

  float3 ray_pos = make_float3(0.0f, 0.0f, 0.0f);
  float3 ray_dir = EyeRayDir(x, y, fwidth, fheight, a_mViewProjInv);
  ray_dir        = tiltCorrection(ray_pos, ray_dir, a_globals);

  if (a_globals->varsI[HRT_ENABLE_DOF] == 1)
  {
    const float tFocus         = a_globals->varsF[HRT_DOF_FOCAL_PLANE_DIST] / (-ray_dir.z);
    const float3 focusPosition = ray_pos + ray_dir*tFocus;
    const float2 xy            = a_globals->varsF[HRT_DOF_LENS_RADIUS]*2.0f*MapSamplesToDisc(make_float2(lensOffs.z - 0.5f, lensOffs.w - 0.5f));
    ray_pos.x += xy.x;
    ray_pos.y += xy.y;

    ray_dir = normalize(focusPosition - ray_pos);
  }

  matrix4x4f_mult_ray3(a_mWorldViewInv, &ray_pos, &ray_dir);
  (*pX)      = lensOffs.x*fwidth;
  (*pY)      = lensOffs.y*fheight;

  (*pRayPos) = ray_pos;
  (*pRayDir) = ray_dir;
}

#ifdef USE_1D_TEXTURES

IDH_CALL int2 getObjectList(unsigned int offset, __read_only image1d_buffer_t objListTex)
{
  int2 res;
  float4 tmp = read_imagef(objListTex, offset); //objListTex[offset]; 
  res.x = as_int(tmp.x);
  res.y = as_int(tmp.y);
  return res;
}

IDH_CALL BVHNode GetBVHNode(unsigned int offset, __read_only image1d_buffer_t bvhTex)
{
  float4 nodeHalf1 = read_imagef(bvhTex, (int)(2 * offset + 0));
  float4 nodeHalf2 = read_imagef(bvhTex, (int)(2 * offset + 1));

  BVHNode node;
  node.m_boxMin.x = nodeHalf1.x;
  node.m_boxMin.y = nodeHalf1.y;
  node.m_boxMin.z = nodeHalf1.z;
  node.m_leftOffsetAndLeaf = as_int(nodeHalf1.w);

  node.m_boxMax.x = nodeHalf2.x;
  node.m_boxMax.y = nodeHalf2.y;
  node.m_boxMax.z = nodeHalf2.z;
  node.m_escapeIndex = as_int(nodeHalf2.w);

  return node;
}

#else

IDH_CALL int2 getObjectList(unsigned int offset, __global const float4* objListTex)
{
  int2 res;
  float4 tmp = objListTex[offset]; 
  res.x = as_int(tmp.x);
  res.y = as_int(tmp.y);
  return res;
}


IDH_CALL BVHNode GetBVHNode(int offset, __global const float4* bvhTex)
{
  const int    offset2   = (offset >= 0) ? offset : 0;

  const float4 nodeHalf1 = bvhTex[2*offset2 + 0]; // = read_1Dimagef(bvhTex, samplerReadElement, 2 * offset);
  const float4 nodeHalf2 = bvhTex[2*offset2 + 1]; // = read_1Dimagef(bvhTex, samplerReadElement, 2 * offset + 1);

  BVHNode node;
  node.m_boxMin.x = nodeHalf1.x;
  node.m_boxMin.y = nodeHalf1.y;
  node.m_boxMin.z = nodeHalf1.z;
  node.m_leftOffsetAndLeaf = as_int(nodeHalf1.w);

  node.m_boxMax.x = nodeHalf2.x;
  node.m_boxMax.y = nodeHalf2.y;
  node.m_boxMax.z = nodeHalf2.z;
  node.m_escapeIndex = as_int(nodeHalf2.w);

  return node;
}

#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct ALIGN_S(16) PlainMeshT
{
  int vPosOffset;
  int vNormOffset;
  int vTexCoordOffset;
  int vIndicesOffset;

  int vPosNum;
  int vNormNum;
  int vTexCoordNum;
  int tIndicesNum;

  int mIndicesOffset;
  int mIndicesNum;
  int vTangentOffset;
  int vTangentNum;

  unsigned int totalBytesNum;
  int polyShadowOffset;
  int dummy2;

} PlainMesh;

static inline __global const PlainMesh* fetchMeshHeader(const Lite_Hit a_liteHit, __global const float4* a_geomStorage, __global const EngineGlobals* a_globals)
{
  const int meshOffset = meshHeaderOffset(a_liteHit, a_globals);
  return (__global const PlainMesh*)(a_geomStorage + meshOffset);
}

static inline __global const float4* meshVerts(__global const PlainMesh* a_pMesh)
{
  __global const float4* pheader = (__global const float4*)a_pMesh;
  return pheader + a_pMesh->vPosOffset;
}

static inline __global const float4* meshNorms(__global const PlainMesh* a_pMesh)
{
  __global const float4* pheader = (__global const float4*)a_pMesh;
  return pheader + a_pMesh->vNormOffset;
}

static inline __global const uint* meshTangentsCompressed(__global const PlainMesh* a_pMesh)
{
  __global const float4* pheader = (__global const float4*)a_pMesh;
  return (__global const uint*)(pheader + a_pMesh->vTangentOffset);
}

//vTangentOffset

static inline __global const float2* meshTexCoords(__global const PlainMesh* a_pMesh)
{
  __global const float4* pheader    = (__global const float4*)a_pMesh;
  __global const float4* ptexcoords = pheader + a_pMesh->vTexCoordOffset;
  return (__global const float2*)ptexcoords;
}

static inline __global const int* meshTriIndices(__global const PlainMesh* a_pMesh)
{
  __global const float4* pheader  = (__global const float4*)a_pMesh;
  __global const float4* pdata    = pheader + a_pMesh->vIndicesOffset;
  return (__global const int*)pdata;
}

static inline __global const int* meshMatIndices(__global const PlainMesh* a_pMesh)
{
  __global const float4* pheader = (__global const float4*)a_pMesh;
  __global const float4* pdata   = pheader + a_pMesh->mIndicesOffset;
  return (__global const int*)pdata;
}

static inline __global const float* meshShadowRayOff(__global const PlainMesh* a_pMesh)
{
  __global const float4* pheader = (__global const float4*)a_pMesh;
  __global const float4* pdata   = pheader + a_pMesh->polyShadowOffset;
  return (__global const float*)pdata;
}


#endif

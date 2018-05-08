#ifndef RTGLOBALS
#define RTGLOBALS

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WARP_SIZE 32

#define Z_ORDER_BLOCK_SIZE 16
#define CMP_RESULTS_BLOCK_SIZE 256

#define HRT_RAY_MISS 0xFFFFFFFE
#define HRT_RAY_HIT 0xFFFFFFFF

#define GMAXVARS 64

#define INVALID_TEXTURE  0xFFFFFFFE 

#define TEX_POINT_SAM    0x10000000
#define TEX_ALPHASRC_W   0x20000000
#define TEX_CLAMP_U      0x40000000
#define TEX_CLAMP_V      0x80000000

// they are related because data are storen in one int32 variable triAlphaTest
//
#define ALPHA_MATERIAL_MASK   0x00FFFFFF
#define ALPHA_LIGHTMESH_MASK  0xFF000000
#define ALPHA_LIGHTMESH_SHIFT 24

#define ALPHA_OPACITY_TEX_HAPPEND  0x80000000
#define ALPHA_TRANSPARENCY_HAPPEND 0x40000000


#define TEXMATRIX_ID_MASK     0x00FFFFFF // for texture slots - 'color_texMatrixId' and e.t.c
#define TEXSAMPLER_TYPE_MASK  0xFF000000 // for texture slots - 'color_texMatrixId' and e.t.c

#ifndef M_PI
#define M_PI          3.14159265358979323846f
#endif

#ifndef INV_PI
#define INV_PI        0.31830988618379067154f
#endif

#ifndef INV_TWOPI
#define INV_TWOPI     0.15915494309189533577f
#endif

#ifndef INV_FOURPI
#define INV_FOURPI    0.07957747154594766788f
#endif

#ifndef DEG_TO_RAD
#define DEG_TO_RAD (M_PI / 180.f)
#endif

#define GEPSILON      5e-6f
#define DEPSILON      1e-20f
#define DEPSILON2     1e-30f
#define PEPSILON      0.025f
#define PG_SCALE      1000.0f


enum MEGATEX_USAGE{ MEGATEX_SHADING      = 1, 
                    MEGATEX_SHADING_HDR  = 2, 
                    MEGATEX_NORMAL       = 3, 
                    MEGATEX_OPACITY      = 4,
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__ 


#else

  #ifdef OCL_COMPILER                // OpenCL
   
  #define ALIGN_S(x) __attribute__ ((aligned (x)))

   #define __device__
   #define IDH_CALL inline
   #define ID_CALL  inline

   IDH_CALL ushort2 make_ushort2(ushort x, ushort y) { ushort2 res; res.x = x; res.y = y; return res; }
   IDH_CALL int2    make_int2(int a, int b)          { int2 res; res.x = a; res.y = b; return res; }
   IDH_CALL int4    make_int4(int a, int b, int c, int d) { int4 res; res.x = a; res.y = b; res.z = c; res.w = d; return res; }

   #define GLOBAL_ID_X get_global_id(0)
   #define GLOBAL_ID_Y get_global_id(1)

   #define LOCAL_ID_X  get_local_id(0)
   #define LOCAL_ID_Y  get_local_id(1)

   #define _PACKED __attribute__ ((packed))
   #define __device__

   //#define SYNCTHREADS        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
   #define SYNCTHREADS_LOCAL  barrier(CLK_LOCAL_MEM_FENCE)
   #define SYNCTHREADS_GLOBAL barrier(CLK_GLOBAL_MEM_FENCE)

   IDH_CALL float maxcomp(float3 v) { return fmax(v.x, fmax(v.y, v.z)); }

   #define NULL 0

   IDH_CALL ushort4 make_ushort4(ushort a, ushort b, ushort c, ushort d)
   {
     ushort4 res;
     res.x = a;
     res.y = b;
     res.z = c;
     res.w = d;
     return res;
   }

   ID_CALL void atomic_addf(volatile __global float *source, const float operand)
   {
     union {
       unsigned int intVal;
       float floatVal;
     } newVal;

     union {
       unsigned int intVal;
       float floatVal;
     } prevVal;

     do {
       prevVal.floatVal = *source;
       newVal.floatVal = prevVal.floatVal + operand;
     } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
   }

   IDH_CALL float dot3 (const float4 u, const float4 v) { return (u.x*v.x + u.y*v.y + u.z*v.z); }

   typedef struct float4x3T
   {
     float4 row[3];
   } float4x3;

   typedef struct float4x4T
   {
     float4 row[4];
   } float4x4;

   typedef struct float3x3T
   {
     float3 row[3];
   } float3x3;

   IDH_CALL float2 make_float2(float a, float b)
   {
     float2 res;
     res.x = a;
     res.y = b;
     return res;
   }

   IDH_CALL float3 make_float3(float a, float b, float c)
   {
     float3 res;
     res.x = a;
     res.y = b;
     res.z = c;
     return res;
   }

   IDH_CALL float4 make_float4(float a, float b, float c, float d)
   {
     float4 res;
     res.x = a;
     res.y = b;
     res.z = c;
     res.w = d;
     return res;
   }

   IDH_CALL float2 to_float2(float4 f4)
   {
     float2 res;
     res.x = f4.x;
     res.y = f4.y;
     return res;
   }

   IDH_CALL float3 to_float3(float4 f4)
   {
     float3 res;
     res.x = f4.x;
     res.y = f4.y;
     res.z = f4.z;
     return res;
   }

   IDH_CALL float4 to_float4(float3 v, float w)
   {
     float4 res;
     res.x = v.x;
     res.y = v.y;
     res.z = v.z;
     res.w = w;
     return res;
   }

   static inline float3 mul4x3(float4x4 m, float3 v)
   {
     float3 res;
     res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z + m.row[0].w;
     res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z + m.row[1].w;
     res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z + m.row[2].w;
     return res;
   }

   static inline float3 mul3x3(float4x4 m, float3 v)
   {
     float3 res;
     res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z;
     res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z;
     res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z;
     return res;
   }

   static inline float2 sincos2f(float a_value)
   {
     float cosVal;
     float sinVal = sincos(a_value, &cosVal);
     return make_float2(sinVal, cosVal);
   }

  #else                              // Common C++
    
    #ifdef WIN32
      #define ALIGN_S(x) __declspec(align(x))
    #else
      #define ALIGN_S(x) __attribute__ ((aligned (x)))
    #endif

    #undef  M_PI
    #include <math.h>
    #undef  M_PI
    #define M_PI 3.14159265358979323846f
    
    #include "../../HydraAPI/hydra_api/LiteMath.h"  
    using namespace HydraLiteMath;

    #include "../../HydraAPI/hydra_api/HR_HDRImage.h"  
    typedef HydraRender::HDRImage4f HDRImage4f;

    typedef unsigned int   uint;
    typedef unsigned short ushort;

    typedef struct float4x3T
    {
      float4 row[3];
    } float4x3;

    typedef struct float3x3T
    {
      float3 row[3];
    } float3x3;
    
    static inline float2 sincos2f(float a_value) 
    {
      return make_float2(sin(a_value), cos(a_value));
    }

    #define IDH_CALL static inline
    #define ID_CALL  static inline

    #define __global
    #define __constant const

    #define __private
    #define __read_only 

    typedef int image1d_t;
    typedef int image1d_buffer_t;
    typedef int image2d_t;
    typedef int sampler_t;

    const int CLK_NORMALIZED_COORDS_TRUE  = 1;
    const int CLK_NORMALIZED_COORDS_FALSE = 2;
    const int CLK_ADDRESS_CLAMP           = 4;
    const int CLK_FILTER_NEAREST          = 8;
    const int CLK_FILTER_LINEAR           = 16;
    const int CLK_ADDRESS_REPEAT          = 32;

    #define COMMON_CPLUS_PLUS_CODE 1

    ID_CALL int   __float_as_int(float x) { return *( (int*)&x ); }
    ID_CALL float __int_as_float(int x) { return *( (float*)&x ); }
   
    ID_CALL int   as_int(float x) { return __float_as_int(x); }
    ID_CALL float as_float(int x) { return __int_as_float(x); }

    #define _PACKED

    typedef unsigned short half;
    ID_CALL void vstore_half(float data, size_t offset, __global half *p) { p[offset] = 0; }

    IDH_CALL float sign(float a) { return (a > 0.0f) ? 1.0f : -1.0f; }

    IDH_CALL int2 make_int2(int a, int b) { int2 res; res.x = a; res.y = b; return res; }

    #define ENABLE_OPACITY_TEX 1
    #define SHADOW_TRACE_COLORED_SHADOWS 1
    #define ENABLE_BLINN 1

    #include "globals_sys.h"
    
#endif

#endif

typedef __global const int4* texture2d_t;

#ifndef INFINITY
  #define INFINITY (1e38f)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct MatSampleT
{
  float3 color;
  float3 direction;
  float  pdf; 
  int    flags;

} MatSample;

enum FLAG_BITS{HRT_COMPUTE_SHADOWS                 = 1,
               HRT_DISABLE_SHADING                 = 2, 
               HRT_DIFFUSE_REFLECTION              = 4,
               HRT_UNIFIED_IMAGE_SAMPLING          = 8,

               HRT_DUMMY1                          = 16,  
               HRT_USE_MIS                         = 32,
               HRT_DUMMY2                          = 64, 
               HRT_STORE_SUBPIXELS                 = 128,
               HRT_FORWARD_TRACING                 = 256, /// tracing from light to eye; otherwise from eye to light.
               HRT_DRAW_LIGHT_LT                   = 512,
               HRT_3WAY_MIS_WEIGHTS                = 1024,
    
               HRT_STORE_RAY_SAMPLES               = 8192,
               HRT_ENABLE_MMLT                     = 16384,
               HRT_DUMMY3                          = 65536*2,
               HRT_DUMMY4                          = 65536*4,
               HRT_STUPID_PT_MODE                  = 65536*8,
               HRT_NO_RANDOM_LIGHTS_SELECT         = 65536*16,
               HRT_MARK_SURFACES_FG                = 65536*32, // 
               HRT_DUMMY5                          = 65536*64, // tracing photons to form spetial photonmap to speed-up direct light sampling
               HRT_DEBUG_DRAW_LIGHT_PHOTONS        = 65536*128,
               HRT_PHOTONS_STORE_MULTIPLY_COLORS   = 65536*256,
             
               HRT_ENABLE_PT_CAUSTICS              = 65536*2048,
               HRT_USE_BOTH_PHOTON_MAPS            = 65536*4096,
               HRT_ENABLE_QMC_ONE_SEED             = 65536*8192, // !!!!!!!! DONT MOVE THIS FLAG !!!! See random generator implementation
               HRT_ENABLE_COHERENT_PT              = 65536*16384,
              };

enum VARIABLE_NAMES { // int vars
                      //
                      HRT_ENABLE_DOF               = 0,
                      HRT_DEBUG_DRAW_LAYER         = 1,
                      HRT_FIRST_BOUNCE_STORE_CACHE = 2,
                      HRT_ENABLE_MRAYS_COUNTERS    = 3,
                      HRT_DEBUG_OUTPUT             = 4,
                      HRT_MEASURE_RAYS_TYPE        = 5,
                      HRT_BLACK_DIFFUSE_OFFSET     = 6,
                      HRT_STORE_SHADOW_COLOR_W     = 7,
                      HRT_WHITE_DIFFUSE_OFFSET     = 8,
                      HRT_TRACE_DEPTH              = 9,
                      HRT_PHOTONS_STORE_BOUNCE     = 10,
                      HRT_PHOTONS_GARTHER_BOUNCE   = 11,
                      HRT_RAYS_APPENDBUFFER_SIZE   = 12,
                      HRT_DIFFUSE_TRACE_DEPTH      = 13,
                      HRT_DISPLAY_IC_INTERMEDIATE  = 14,
                      HRT_PT_FILTER_TYPE           = 15,
                      HRT_ENABLE_BAKE              = 16,
                      HRT_SILENT_MODE              = 17,
                      HRT_VAR_ENABLE_RR            = 18,
                      HRT_RENDER_LAYER             = 19,
                      HRT_RENDER_LAYER_DEPTH       = 20,
                      HRT_IC_ENABLED               = 21,
                      HRT_IMAP_ENABLED             = 22,
                      HRT_SPHEREMAP_TEXID0         = 23,
                      HRT_SPHEREMAP_TEXID1         = 24,
                      HRT_USE_GAMMA_FOR_ENV        = 25,
                      HRT_HRT_SCENE_HAVE_PORTALS   = 26,
                      HRT_SPHEREMAP_TEXMATRIXID0   = 27,
                      HRT_SPHEREMAP_TEXMATRIXID1   = 28,
                      HRT_ENABLE_PATH_REGENERATE   = 29,
                      HRT_ENV_PDF_TABLE_ID         = 30,
                      HRT_MLT_MAX_NUMBERS          = 31,
                      HRT_MLT_ITERS_MULT           = 32,
                      HRT_MLT_BURN_ITERS           = 33,
                      HRT_MMLT_FIRST_BOUNCE        = 34,
};

enum VARIABLE_FLOAT_NAMES{ // float vars
                           //
                           HRT_DOF_LENS_RADIUS                     = 0,
                           HRT_DOF_FOCAL_PLANE_DIST                = 1,
                           
                           HRT_TILT_ROT_X                        = 2,
                           HRT_TRACE_PROCEEDINGS_TRESHOLD          = 3, 
                           HRT_TILT_ROT_Y                        = 4,
                           HRT_CAUSTIC_POWER_MULT                  = 5,
                           
                           HRT_IMAGE_GAMMA                         = 6,
                           HRT_TEXINPUT_GAMMA                      = 7,
                           
                           HRT_ENV_COLOR_X                         = 8,
                           HRT_ENV_COLOR_Y                         = 9,
                           HRT_ENV_COLOR_Z                         = 10,
                           HRT_ENV_COLOR2_X                        = 11,
                           HRT_ENV_COLOR2_Y                        = 12,
                           HRT_ENV_COLOR2_Z                        = 13,
                           
                           HRT_CAM_FOV                             = 14,
                           HRT_PATH_TRACE_ERROR                    = 15,  
                           HRT_ENV_CLAMPING                        = 16,
                           HRT_BSDF_CLAMPING                       = 17, 

                           HRT_BSPHERE_CENTER_X                    = 18,
                           HRT_BSPHERE_CENTER_Y                    = 19,
                           HRT_BSPHERE_CENTER_Z                    = 20,
                           HRT_BSPHERE_RADIUS                      = 21,
                           HRT_GVOXEL_SIZE                         = 22,

                           HRT_FOV_X                               = 23, // viewport parameters
                           HRT_FOV_Y                               = 24,
                           HRT_WIDTH_F                             = 25,
                           HRT_HEIGHT_F                            = 26,

                           HRT_ABLOW_OFFSET_X                      = 27,
                           HRT_ABLOW_OFFSET_Y                      = 28,
                           HRT_ABLOW_SCALE_X                       = 29,
                           HRT_ABLOW_SCALE_Y                       = 30,

                           HRT_IMG_AVG_LUM                         = 31,
                           HRT_MLT_PLARGE                          = 32,
                           HRT_MLT_BKELEMEN                        = 33,
                           HRT_MLT_SCREEN_SCALE_X                  = 34,
                           HRT_MLT_SCREEN_SCALE_Y                  = 35,
};


enum RENDER_LAYER {
  LAYER_COLOR                  = 0,
  LAYER_POSITIONS              = 1,
  LAYER_NORMALS                = 2,
  LAYER_TEXCOORD               = 3,
  LAYER_TEXCOLOR_AND_MATERIAL  = 4,   // material mask
  LAYER_INCOMING_PRIMARY       = 5,   // incoming primary
  LAYER_INCOMING_RADIANCE      = 6,   // incoming secondary
  LAYER_COLOR_PRIMARY_AND_REST = 7,   // primary + refractions and other bounces
  LAYER_COLOR_THE_REST         = 8,
  LAYER_PRIMARY                = 9,
  LAYER_SECONDARY              = 10
}; // refractions, and other bounces

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


IDH_CALL uint ZIndex(ushort x, ushort y, __constant ushort* a_mortonTable256)
{
  return	(a_mortonTable256[y >> 8]   << 17) |
          (a_mortonTable256[x >> 8]   << 16) |
          (a_mortonTable256[y & 0xFF] << 1 ) |
          (a_mortonTable256[x & 0xFF]      );
}


IDH_CALL ushort ExtractFromZIndex3D(uint zIndex, int stride)
{
  uint result = 0;
  for (int i = 0; i < 10; i++)
  {
    int bitBask = 1 << (3 * i + stride);
    int bit = (bitBask & zIndex) ? 1 : 0;
    result |= (bit << i);
  }
  return (ushort)result;
}

IDH_CALL ushort ExtractFromZIndex2D(uint zIndex, int stride)
{
  uint result = 0;
  for (int i = 0; i < 16; i++)
  {
    int bitBask = 1 << (2 * i + stride);
    int bit = (bitBask & zIndex) ? 1 : 0;
    result |= (bit << i);
  }
  return (ushort)result;
}

IDH_CALL ushort ExtractXFromZIndex(uint zIndex)
{
  uint result = 0;
  for (int i = 0; i<16; i++)
    result |= ((1 << (2 * i)) & zIndex) >> i;
  return (ushort)result;
}


IDH_CALL ushort ExtractYFromZIndex(uint zIndex)
{
  uint result = 0;
  for (int i = 0; i<16; i++)
    result |= ((1 << (2 * i + 1)) & zIndex) >> i;
  return (ushort)(result >> 1);
}

IDH_CALL int blocks(int elems, int threadsPerBlock)
{
  if (elems % threadsPerBlock == 0 && elems >= threadsPerBlock)
    return elems / threadsPerBlock;
  else
    return (elems / threadsPerBlock) + 1;
}

IDH_CALL size_t blocksST(size_t elems, int threadsPerBlock)
{
  if (elems % threadsPerBlock == 0 && elems >= threadsPerBlock)
    return elems / threadsPerBlock;
  else
    return (elems / threadsPerBlock) + 1;
}

IDH_CALL size_t roundBlocks(size_t elems, int threadsPerBlock)
{
  if (elems < threadsPerBlock)
    return (size_t)threadsPerBlock;
  else
    return blocksST(elems, threadsPerBlock) * threadsPerBlock;
}

IDH_CALL uint Index2D(uint x, uint y, int pitch) { return y*pitch + x; }

IDH_CALL uint IndexZBlock2D(int x, int y, int pitch, __constant ushort* a_mortonTable) // window_size[0]
{
  uint zOrderX = x % Z_ORDER_BLOCK_SIZE;
  uint zOrderY = y % Z_ORDER_BLOCK_SIZE;

  uint zIndex  = ZIndex(zOrderX, zOrderY, a_mortonTable);

  uint wBlocks = pitch / Z_ORDER_BLOCK_SIZE;
  uint blockX  = x / Z_ORDER_BLOCK_SIZE;
  uint blockY  = y / Z_ORDER_BLOCK_SIZE;

  return (blockX + (blockY)*(wBlocks))*Z_ORDER_BLOCK_SIZE*Z_ORDER_BLOCK_SIZE + zIndex;
}


IDH_CALL ushort2 GetXYFromZBlockIndex(uint a_offset, int w, int h)
{
  int blocksSizeX = w / Z_ORDER_BLOCK_SIZE;
  //int blocksSizeY = h / Z_ORDER_BLOCK_SIZE;

  int blockId    = a_offset / (Z_ORDER_BLOCK_SIZE*Z_ORDER_BLOCK_SIZE);
  int zIdInBlock = a_offset % (Z_ORDER_BLOCK_SIZE*Z_ORDER_BLOCK_SIZE);

  int blockY = blockId / blocksSizeX;
  int blockX = blockId - blockY*blocksSizeX;

  int localX = (int)ExtractXFromZIndex(zIdInBlock);
  int localY = (int)ExtractYFromZIndex(zIdInBlock);

  ushort2 res;
  res.x = (ushort)(blockX*Z_ORDER_BLOCK_SIZE + localX);
  res.y = (ushort)(blockY*Z_ORDER_BLOCK_SIZE + localY);
  return res;
}


IDH_CALL uint SpreadBits(int x, int offset)
{
  x = (x | (x << 10)) & 0x000F801F;
  x = (x | (x << 4)) & 0x00E181C3;
  x = (x | (x << 2)) & 0x03248649;
  x = (x | (x << 2)) & 0x09249249;

  return (uint)(x) << offset;
}

IDH_CALL uint GetMortonNumber(int x, int y, int z)
{
  return SpreadBits(x, 0) | SpreadBits(y, 1) | SpreadBits(z, 2);
}


IDH_CALL float3 reflect(float3 dir, float3 normal) { return normalize((normal * dot(dir, normal) * (-2.0f)) + dir); }


///////////////////////////////////////////////////////////////////////////////////////////////////////////
///// a simple tone mapping
IDH_CALL float3 ToneMapping(float3 color)  { return make_float3(fmin(color.x, 1.0f), fmin(color.y, 1.0f), fmin(color.z, 1.0f)); }
IDH_CALL float4 ToneMapping4(float4 color) { return make_float4(fmin(color.x, 1.0f), fmin(color.y, 1.0f), fmin(color.z, 1.0f), fmin(color.w, 1.0f)); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
////
IDH_CALL uint RealColorToUint32_f3(float3 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  unsigned char red = (unsigned char)r, green = (unsigned char)g, blue = (unsigned char)b;
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}


IDH_CALL uint RealColorToUint32(float4 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  float  a = real_color.w*255.0f;

  unsigned char red   = (unsigned char)r;
  unsigned char green = (unsigned char)g;
  unsigned char blue  = (unsigned char)b;
  unsigned char alpha = (unsigned char)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

static inline float3 SafeInverse(float3 d)
{
  const float ooeps = 1.0e-36f; // Avoid div by zero.

  float3 res;
  res.x = 1.0f / (fabs(d.x) > ooeps ? d.x : copysign(ooeps, d.x));
  res.y = 1.0f / (fabs(d.y) > ooeps ? d.y : copysign(ooeps, d.y));
  res.z = 1.0f / (fabs(d.z) > ooeps ? d.z : copysign(ooeps, d.z));
  return res;
}

static inline float epsilonOfPos(float3 hitPos) { return fmax(fmax(fabs(hitPos.x), fmax(fabs(hitPos.y), fabs(hitPos.z))), 2.0f*GEPSILON)*GEPSILON; }
static inline float misHeuristicPower1(float p) { return isfinite(p) ? fabs(p) : 0.0f; }
static inline float misHeuristicPower2(float p) { return isfinite(p) ? p*p     : 0.0f; }

static inline float misWeightHeuristic(float a, float b)
{
  const float w = misHeuristicPower1(a) / fmax(misHeuristicPower1(a) + misHeuristicPower1(b), DEPSILON2);
  return isfinite(w) ? w : 0.0f;
}

static inline float misWeightHeuristic3(float a, float b, float c)
{
  const float w = misHeuristicPower2(a) / fmax(misHeuristicPower2(a) + misHeuristicPower2(b) + misHeuristicPower2(c), DEPSILON2);
  return isfinite(w) ? w : 0.0f;
}

/**
\brief offset reflected ray position by epsilon;
\param  a_hitPos      - world space position on surface
\param  a_surfaceNorm - surface normal at a_hitPos
\param  a_sampleDir   - ray direction in which we are going to trace reflected ray
\return offseted ray position
*/
static inline float3 OffsRayPos(const float3 a_hitPos, const float3 a_surfaceNorm, const float3 a_sampleDir)
{
  const float signOfNormal2 = dot(a_sampleDir, a_surfaceNorm) < 0.0f ? -1.0f : 1.0f;
  const float offsetEps     = epsilonOfPos(a_hitPos);
  return a_hitPos + signOfNormal2*offsetEps*a_surfaceNorm;
}

/**
\brief offset reflected ray position by epsilon;
\param  a_hitPos        - world space position on surface
\param  a_surfaceNorm   - surface normal at a_hitPos
\param  a_sampleDir     - ray direction in which we are going to trace reflected ray
\param  a_shadowOffsAux - per poly auxilarry shadow offset. 
\return offseted ray position
*/
static inline float3 OffsShadowRayPos(const float3 a_hitPos, const float3 a_surfaceNorm, const float3 a_sampleDir, const float a_shadowOffsAux)
{
  const float signOfNormal2 = dot(a_sampleDir, a_surfaceNorm) < 0.0f ? -1.0f : 1.0f;
  const float offsetEps     = epsilonOfPos(a_hitPos);
  return a_hitPos + signOfNormal2*(offsetEps + a_shadowOffsAux)*a_surfaceNorm;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline float4x4 make_float4x4(__global const float* a_data)
{
  float4x4 matrix;
  matrix.row[0] = make_float4(a_data[0], a_data[1], a_data[2], a_data[3]);
  matrix.row[1] = make_float4(a_data[4], a_data[5], a_data[6], a_data[7]);
  matrix.row[2] = make_float4(a_data[8], a_data[9], a_data[10], a_data[11]);
  matrix.row[3] = make_float4(a_data[12], a_data[13], a_data[14], a_data[15]);
  return matrix;
}

IDH_CALL float4x4 make_matrix_rotationX(float a_angle)
{
  float sinx = sin(a_angle);
  float cosx = cos(a_angle);

  float4x4 matrix;
  matrix.row[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  matrix.row[1] = make_float4(0.0f, cosx, sinx, 0.0f);
  matrix.row[2] = make_float4(0.0f, -sinx, cosx, 0.0f);
  matrix.row[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  return matrix;
}

IDH_CALL float4x4 make_matrix_rotationY(float a_angle)
{
  float siny = sin(a_angle);
  float cosy = cos(a_angle);

  float4x4 matrix;
  matrix.row[0] = make_float4(cosy, 0.0f, -siny, 0.0f);
  matrix.row[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  matrix.row[2] = make_float4(siny, 0.0f, cosy, 0.0f);
  matrix.row[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  return matrix;
}

IDH_CALL float3 mul4x3x3(float4x3 m, float3 v)
{
  float3 res;
  res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z + m.row[0].w;
  res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z + m.row[1].w;
  res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z + m.row[2].w;
  return res;
}

IDH_CALL float4 mul4x4x4(float4x4 m, float4 v)
{
  float4 res;
  res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z + m.row[0].w*v.w;
  res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z + m.row[1].w*v.w;
  res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z + m.row[2].w*v.w;
  res.w = m.row[3].x*v.x + m.row[3].y*v.y + m.row[3].z*v.z + m.row[3].w*v.w;
  return res;
}

#ifndef COMMON_CPLUS_PLUS_CODE

IDH_CALL float3 mul(float4x4 m, float3 v)
{
  float3 res;
  res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z + m.row[0].w;
  res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z + m.row[1].w;
  res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z + m.row[2].w;
  return res;
}

#endif

IDH_CALL float3 mul3x4(float4x3 m, float3 v)
{
  float3 res;
  res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z + m.row[0].w;
  res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z + m.row[1].w;
  res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z + m.row[2].w;
  return res;
}


IDH_CALL float3x3 make_float3x3(float3 a, float3 b, float3 c)
{
  float3x3 m;
  m.row[0] = a;
  m.row[1] = b;
  m.row[2] = c;
  return m;
}

IDH_CALL float3x3 make_float3x3_by_columns(float3 a, float3 b, float3 c)
{
  float3x3 m;
  m.row[0].x = a.x;
  m.row[1].x = a.y;
  m.row[2].x = a.z;

  m.row[0].y = b.x;
  m.row[1].y = b.y;
  m.row[2].y = b.z;

  m.row[0].z = c.x;
  m.row[1].z = c.y;
  m.row[2].z = c.z;
  return m;
}

IDH_CALL float3 mul3x3x3(float3x3 m, const float3 v)
{
  float3 res;
  res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z;
  res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z;
  res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z;
  return res;
}

IDH_CALL float3x3 mul3x3x3x3(float3x3 m1, float3x3 m2)
{
  float3 column1 = mul3x3x3(m1, make_float3(m2.row[0].x, m2.row[1].x, m2.row[2].x));
  float3 column2 = mul3x3x3(m1, make_float3(m2.row[0].y, m2.row[1].y, m2.row[2].y));
  float3 column3 = mul3x3x3(m1, make_float3(m2.row[0].z, m2.row[1].z, m2.row[2].z));

  return make_float3x3_by_columns(column1, column2, column3);
}

IDH_CALL float3x3 inverse(float3x3 a)
{
  float det = a.row[0].x * (a.row[1].y * a.row[2].z - a.row[1].z * a.row[2].y) -
              a.row[0].y * (a.row[1].x * a.row[2].z - a.row[1].z * a.row[2].x) +
              a.row[0].z * (a.row[1].x * a.row[2].y - a.row[1].y * a.row[2].x);

  float3x3 b;
  b.row[0].x = (a.row[1].y * a.row[2].z - a.row[1].z * a.row[2].y);
  b.row[0].y = -(a.row[0].y * a.row[2].z - a.row[0].z * a.row[2].y);
  b.row[0].z = (a.row[0].y * a.row[1].z - a.row[0].z * a.row[1].y);
  b.row[1].x = -(a.row[1].x * a.row[2].z - a.row[1].z * a.row[2].x);
  b.row[1].y = (a.row[0].x * a.row[2].z - a.row[0].z * a.row[2].x);
  b.row[1].z = -(a.row[0].x * a.row[1].z - a.row[0].z * a.row[1].x);
  b.row[2].x = (a.row[1].x * a.row[2].y - a.row[1].y * a.row[2].x);
  b.row[2].y = -(a.row[0].x * a.row[2].y - a.row[0].y * a.row[2].x);
  b.row[2].z = (a.row[0].x * a.row[1].y - a.row[0].y * a.row[1].x);

  float s = 1.0f / det;
  b.row[0] *= s;
  b.row[1] *= s;
  b.row[2] *= s;

  return b;
}

#ifndef COMMON_CPLUS_PLUS_CODE

static inline float4x4 inverse4x4(float4x4 m1)
{
  float tmp[12]; // temp array for pairs
  float4x4 m;

  // calculate pairs for first 8 elements (cofactors)
  //
  tmp[0] = m1.row[2].z * m1.row[3].w;
  tmp[1] = m1.row[2].w * m1.row[3].z;
  tmp[2] = m1.row[2].y * m1.row[3].w;
  tmp[3] = m1.row[2].w * m1.row[3].y;
  tmp[4] = m1.row[2].y * m1.row[3].z;
  tmp[5] = m1.row[2].z * m1.row[3].y;
  tmp[6] = m1.row[2].x * m1.row[3].w;
  tmp[7] = m1.row[2].w * m1.row[3].x;
  tmp[8] = m1.row[2].x * m1.row[3].z;
  tmp[9] = m1.row[2].z * m1.row[3].x;
  tmp[10] = m1.row[2].x * m1.row[3].y;
  tmp[11] = m1.row[2].y * m1.row[3].x;

  // calculate first 8 m1.rowents (cofactors)
  //
  m.row[0].x = tmp[0] * m1.row[1].y + tmp[3] * m1.row[1].z + tmp[4] * m1.row[1].w;
  m.row[0].x -= tmp[1] * m1.row[1].y + tmp[2] * m1.row[1].z + tmp[5] * m1.row[1].w;
  m.row[1].x = tmp[1] * m1.row[1].x + tmp[6] * m1.row[1].z + tmp[9] * m1.row[1].w;
  m.row[1].x -= tmp[0] * m1.row[1].x + tmp[7] * m1.row[1].z + tmp[8] * m1.row[1].w;
  m.row[2].x = tmp[2] * m1.row[1].x + tmp[7] * m1.row[1].y + tmp[10] * m1.row[1].w;
  m.row[2].x -= tmp[3] * m1.row[1].x + tmp[6] * m1.row[1].y + tmp[11] * m1.row[1].w;
  m.row[3].x = tmp[5] * m1.row[1].x + tmp[8] * m1.row[1].y + tmp[11] * m1.row[1].z;
  m.row[3].x -= tmp[4] * m1.row[1].x + tmp[9] * m1.row[1].y + tmp[10] * m1.row[1].z;
  m.row[0].y = tmp[1] * m1.row[0].y + tmp[2] * m1.row[0].z + tmp[5] * m1.row[0].w;
  m.row[0].y -= tmp[0] * m1.row[0].y + tmp[3] * m1.row[0].z + tmp[4] * m1.row[0].w;
  m.row[1].y = tmp[0] * m1.row[0].x + tmp[7] * m1.row[0].z + tmp[8] * m1.row[0].w;
  m.row[1].y -= tmp[1] * m1.row[0].x + tmp[6] * m1.row[0].z + tmp[9] * m1.row[0].w;
  m.row[2].y = tmp[3] * m1.row[0].x + tmp[6] * m1.row[0].y + tmp[11] * m1.row[0].w;
  m.row[2].y -= tmp[2] * m1.row[0].x + tmp[7] * m1.row[0].y + tmp[10] * m1.row[0].w;
  m.row[3].y = tmp[4] * m1.row[0].x + tmp[9] * m1.row[0].y + tmp[10] * m1.row[0].z;
  m.row[3].y -= tmp[5] * m1.row[0].x + tmp[8] * m1.row[0].y + tmp[11] * m1.row[0].z;

  // calculate pairs for second 8 m1.rowents (cofactors)
  //
  tmp[0] = m1.row[0].z * m1.row[1].w;
  tmp[1] = m1.row[0].w * m1.row[1].z;
  tmp[2] = m1.row[0].y * m1.row[1].w;
  tmp[3] = m1.row[0].w * m1.row[1].y;
  tmp[4] = m1.row[0].y * m1.row[1].z;
  tmp[5] = m1.row[0].z * m1.row[1].y;
  tmp[6] = m1.row[0].x * m1.row[1].w;
  tmp[7] = m1.row[0].w * m1.row[1].x;
  tmp[8] = m1.row[0].x * m1.row[1].z;
  tmp[9] = m1.row[0].z * m1.row[1].x;
  tmp[10] = m1.row[0].x * m1.row[1].y;
  tmp[11] = m1.row[0].y * m1.row[1].x;

  // calculate second 8 m1 (cofactors)
  //
  m.row[0].z = tmp[0] * m1.row[3].y + tmp[3] * m1.row[3].z + tmp[4] * m1.row[3].w;
  m.row[0].z -= tmp[1] * m1.row[3].y + tmp[2] * m1.row[3].z + tmp[5] * m1.row[3].w;
  m.row[1].z = tmp[1] * m1.row[3].x + tmp[6] * m1.row[3].z + tmp[9] * m1.row[3].w;
  m.row[1].z -= tmp[0] * m1.row[3].x + tmp[7] * m1.row[3].z + tmp[8] * m1.row[3].w;
  m.row[2].z = tmp[2] * m1.row[3].x + tmp[7] * m1.row[3].y + tmp[10] * m1.row[3].w;
  m.row[2].z -= tmp[3] * m1.row[3].x + tmp[6] * m1.row[3].y + tmp[11] * m1.row[3].w;
  m.row[3].z = tmp[5] * m1.row[3].x + tmp[8] * m1.row[3].y + tmp[11] * m1.row[3].z;
  m.row[3].z -= tmp[4] * m1.row[3].x + tmp[9] * m1.row[3].y + tmp[10] * m1.row[3].z;
  m.row[0].w = tmp[2] * m1.row[2].z + tmp[5] * m1.row[2].w + tmp[1] * m1.row[2].y;
  m.row[0].w -= tmp[4] * m1.row[2].w + tmp[0] * m1.row[2].y + tmp[3] * m1.row[2].z;
  m.row[1].w = tmp[8] * m1.row[2].w + tmp[0] * m1.row[2].x + tmp[7] * m1.row[2].z;
  m.row[1].w -= tmp[6] * m1.row[2].z + tmp[9] * m1.row[2].w + tmp[1] * m1.row[2].x;
  m.row[2].w = tmp[6] * m1.row[2].y + tmp[11] * m1.row[2].w + tmp[3] * m1.row[2].x;
  m.row[2].w -= tmp[10] * m1.row[2].w + tmp[2] * m1.row[2].x + tmp[7] * m1.row[2].y;
  m.row[3].w = tmp[10] * m1.row[2].z + tmp[4] * m1.row[2].x + tmp[9] * m1.row[2].y;
  m.row[3].w -= tmp[8] * m1.row[2].y + tmp[11] * m1.row[2].z + tmp[5] * m1.row[2].x;

  // calculate matrix inverse
  //
  float k = 1.0f / (m1.row[0].x * m.row[0].x + m1.row[0].y * m.row[1].x + m1.row[0].z * m.row[2].x + m1.row[0].w * m.row[3].x);

  for (int i = 0; i<4; i++)
  {
    m.row[i].x *= k;
    m.row[i].y *= k;
    m.row[i].z *= k;
    m.row[i].w *= k;
  }

  return m;
}


// Look At matrix creation
// return the inverse view matrix
//

IDH_CALL float4x4 lookAt(float3 eye, float3 center, float3 up)
{
  float3 x, y, z; // basis; will make a rotation matrix

  z.x = eye.x - center.x;
  z.y = eye.y - center.y;
  z.z = eye.z - center.z;
  z = normalize(z);

  y.x = up.x;
  y.y = up.y;
  y.z = up.z;

  x = cross(y, z); // X vector = Y cross Z
  y = cross(z, x); // Recompute Y = Z cross X

  // cross product gives area of parallelogram, which is < 1.0 for
  // non-perpendicular unit-length vectors; so normalize x, y here
  x = normalize(x);
  y = normalize(y);

  float4x4 M;
  M.row[0].x = x.x; M.row[1].x = x.y; M.row[2].x = x.z; M.row[3].x = -x.x * eye.x - x.y * eye.y - x.z*eye.z;
  M.row[0].y = y.x; M.row[1].y = y.y; M.row[2].y = y.z; M.row[3].y = -y.x * eye.x - y.y * eye.y - y.z*eye.z;
  M.row[0].z = z.x; M.row[1].z = z.y; M.row[2].z = z.z; M.row[3].z = -z.x * eye.x - z.y * eye.y - z.z*eye.z;
  M.row[0].w = 0.0; M.row[1].w = 0.0; M.row[2].w = 0.0; M.row[3].w = 1.0;
  return M;
}

static inline float4x4 transpose(const float4x4 a_mat)
{
  float4x4 res;
  res.row[0].x = a_mat.row[0].x;
  res.row[0].y = a_mat.row[1].x;
  res.row[0].z = a_mat.row[2].x;
  res.row[0].w = a_mat.row[3].x;
  res.row[1].x = a_mat.row[0].y;
  res.row[1].y = a_mat.row[1].y;
  res.row[1].z = a_mat.row[2].y;
  res.row[1].w = a_mat.row[3].y;
  res.row[2].x = a_mat.row[0].z;
  res.row[2].y = a_mat.row[1].z;
  res.row[2].z = a_mat.row[2].z;
  res.row[2].w = a_mat.row[3].z;
  res.row[3].x = a_mat.row[0].w;
  res.row[3].y = a_mat.row[1].w;
  res.row[3].z = a_mat.row[2].w;
  res.row[3].w = a_mat.row[3].w;
  return res;
}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IDH_CALL float3 EyeRayDir(float x, float y, float w, float h, float4x4 a_mViewProjInv) // g_mViewProjInv
{
  float4 pos = make_float4( 2.0f * (x + 0.5f) / w - 1.0f, 
                           -2.0f * (y + 0.5f) / h + 1.0f, 
                            0.0f, 
                            1.0f );

  pos = mul4x4x4(a_mViewProjInv, pos);
  pos /= pos.w;

  pos.y *= (-1.0f);

  return normalize(to_float3(pos));
}


IDH_CALL void matrix4x4f_mult_ray3(float4x4 a_mWorldViewInv, __private float3* ray_pos, __private float3* ray_dir) // g_mWorldViewInv
{
  float3 pos  = mul(a_mWorldViewInv, (*ray_pos));
  float3 pos2 = mul(a_mWorldViewInv, ((*ray_pos) + 100.0f*(*ray_dir)));

  float3 diff = pos2 - pos;

  (*ray_pos)  = pos;
  (*ray_dir)  = normalize(diff);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
////
IDH_CALL float3 matrix3x3f_mult_float3(__global const float* M, float3 v)
{
  float3 res;
  res.x = M[0 * 3 + 0] * v.x + M[0 * 3 + 1] * v.y + M[0 * 3 + 2] * v.z;
  res.y = M[1 * 3 + 0] * v.x + M[1 * 3 + 1] * v.y + M[1 * 3 + 2] * v.z;
  res.z = M[2 * 3 + 0] * v.x + M[2 * 3 + 1] * v.y + M[2 * 3 + 2] * v.z;
  return res;
}


IDH_CALL float DistanceSquared(float3 a, float3 b)
{
  float3 diff = b - a;
  return dot(diff, diff);
}

IDH_CALL float UniformConePdf(float cosThetaMax) { return 1.0f / (2.0f * M_PI * (1.0f - cosThetaMax)); }

IDH_CALL float3 UniformSampleSphere(float u1, float u2)
{
  float z = 1.0f - 2.0f * u1;
  float r = sqrt(fmax(0.0f, 1.0f - z*z));
  float phi = 2.0f * M_PI * u2;
  float x = r * cos(phi);
  float y = r * sin(phi);
  return make_float3(x, y, z);
}

IDH_CALL float lerp2(float t, float a, float b)
{
  return (1.0f - t) * a + t * b;
}

IDH_CALL float3 UniformSampleCone(float u1, float u2, float costhetamax, float3 x, float3 y, float3 z)
{
  float costheta = lerp2(u1, costhetamax, 1.0f);
  float sintheta = sqrt(1.0f - costheta*costheta);
  float phi = u2 * 2.0f * M_PI;
  return cos(phi) * sintheta * x + sin(phi) * sintheta * y + costheta * z;
}

IDH_CALL float2 RaySphereIntersect(float3 rayPos, float3 rayDir, float3 sphPos, float radius)
{
  float3 k = rayPos - sphPos;
  float  b = dot(k, rayDir);
  float  c = dot(k, k) - radius*radius;
  float  d = b * b - c;

  float2 res;

  if (d >= 0.0f)
  {
    float sqrtd = sqrt(d);
    float t1 = -b - sqrtd;
    float t2 = -b + sqrtd;

    res.x = fmin(t1, t2);
    res.y = fmax(t1, t2);
  }
  else
  {
    res.x = -1e28f;
    res.y = -1e28f;
  }

  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct ObjectListTriangle
{
  float4 v1;
  float4 v2;
  float4 v3;
};

struct ObjectListSphere
{
  float3 pos;
  float r;
};

struct ObjectList
{

#ifndef OCL_COMPILER
#ifndef __CUDACC__

  ObjectList() { m_triangleCount = m_offset = dummy1 = dummy2 = 0; }

  inline ObjectListTriangle* GetTriangles() const { return  (ObjectListTriangle*)(((char*)this) + sizeof(ObjectList)); }
  inline const ObjectListSphere* GetSpheres() const { return  (const ObjectListSphere*)((char*)this + sizeof(ObjectList) + m_triangleCount*sizeof(ObjectListTriangle)); }

#endif
#endif

  int m_offset;
  int m_triangleCount;
  int dummy1;
  int dummy2;
};

IDH_CALL int GetNumTriangles(struct ObjectList ol)  { return ol.m_triangleCount; }
IDH_CALL int GetOffset(struct ObjectList ol)        { return ol.m_offset; }
IDH_CALL int GetNumPrimitives(struct ObjectList ol) { return GetNumTriangles(ol); }


struct ALIGN_S(16) Lite_HitT
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
};

typedef struct Lite_HitT Lite_Hit;

IDH_CALL Lite_Hit Make_Lite_Hit(float t, int a_treeId)
{
  int a_geomId = 0;

  Lite_Hit hit;
  hit.t      = t;
  hit.primId = -1;
  hit.instId = -1;
  hit.geomId = (a_geomId & 0x3FFFFFFF) | ((a_treeId << 30) & 0xC0000000);
  return hit;
}

static inline bool HitNone(const Lite_Hit a_hit) { return (a_hit.primId == -1) || !isfinite(a_hit.t); }
static inline bool HitSome(const Lite_Hit a_hit) { return (a_hit.primId != -1) && isfinite(a_hit.t); }

IDH_CALL int IS_LEAF(int a_leftOffsetAndLeaf)                 { return a_leftOffsetAndLeaf & 0x80000000; }
IDH_CALL int PACK_LEAF_AND_OFFSET(int a_leftOffset, int leaf) { return (a_leftOffset & 0x7fffffff) | (leaf & 0x80000000); }
IDH_CALL int EXTRACT_OFFSET(int a_leftOffsetAndLeaf)          { return a_leftOffsetAndLeaf & 0x7fffffff; }


// a know about bit fields, but in CUDA they didn't work
//
struct BVHNodeT
{

#ifndef __CUDACC__
#ifndef OCL_COMPILER

  BVHNodeT() 
  { 
    m_leftOffsetAndLeaf = 0xffffffff; 
    m_escapeIndex       = 0xffffffff;
    m_boxMin            = float3(INFINITY, INFINITY, INFINITY);
    m_boxMax            = float3(-INFINITY, -INFINITY, -INFINITY);
  }


  inline unsigned int Leaf() const { return (m_leftOffsetAndLeaf & 0x80000000) >> 31; }

  inline void SetLeaf(unsigned int a_Leaf)          { m_leftOffsetAndLeaf = (m_leftOffsetAndLeaf & 0x7fffffff) | ((a_Leaf) << 31); }
  inline void SetLeftOffset(unsigned int in_offset) { m_leftOffsetAndLeaf = (m_leftOffsetAndLeaf & 0x80000000) | (in_offset & 0x7fffffff); }

  inline void SetObjectListOffset(unsigned int in_offset)
  {
    if (Leaf())
      SetLeftOffset(in_offset);
  }

  inline unsigned int GetLeftOffset()       const { return m_leftOffsetAndLeaf & 0x7fffffff; }
  inline unsigned int GetRightOffset()      const { return GetLeftOffset() + 1; }
  inline unsigned int GetObjectListOffset() const { return GetLeftOffset(); }

  inline void SetInstance(unsigned int a_Leaf) { m_escapeIndex = a_Leaf; }
  inline unsigned int Instance() const { return (m_escapeIndex == 1); }

#endif
#endif

  float3 m_boxMin;
  unsigned int m_leftOffsetAndLeaf;

  float3 m_boxMax;
  unsigned int m_escapeIndex;

};

typedef struct BVHNodeT BVHNode;

IDH_CALL bool IsValidNode(const BVHNode a_node) { return !((a_node.m_leftOffsetAndLeaf == 0xffffffff) && (a_node.m_escapeIndex == 0xffffffff)); }

struct _PACKED RayFlagsT
{
  unsigned char  diffuseBounceNum;
  unsigned char  bounceNum;
  unsigned short otherFlags;
};

typedef struct RayFlagsT RayFlags;

enum MATERIAL_EVENT {
  RAY_EVENT_S    = 1,  ///< Indicates Specular reflection or refraction (check for RAY_EVENT_T)
  RAY_EVENT_D    = 2,  ///< Indicates Diffuse  reflection or translucent (check for RAY_EVENT_T)
  RAY_EVENT_G    = 4,  ///< Indicates GLossy   reflection or refraction (check for RAY_EVENT_T)
  RAY_EVENT_T    = 8,  ///< Indicates Transparensy or reftacrion. 
  RAY_EVENT_V    = 16, ///< Indicates Volume scattering, not used for a while
  RAY_EVENT_TOUT = 32, ///< Indicates Transparensy Outside of water or glass or e.t.c. (old RAY_IS_INSIDE_TRANSPARENT_OBJECT = 128)
};

static inline bool isPureSpecular(const MatSample a_sample) { return (a_sample.flags & RAY_EVENT_S) != 0; }
static inline bool isDiffuse     (const MatSample a_sample) { return (a_sample.flags & RAY_EVENT_D) != 0; }
static inline bool isGlossy      (const MatSample a_sample) { return (a_sample.flags & RAY_EVENT_G) != 0; }

enum {
  RAY_GRAMMAR_DIRECT_LIGHT         = 64,
  RAY_GRAMMAR_OUT_OF_SCENE         = 128,
  RAY_DUMMY_FLAG_NOT_USED          = 256,
  RAY_HIT_SURFACE_FROM_OTHER_SIDE  = 2048,
  RAY_IS_DEAD                      = 4096,  // set when ray had account environment or died on the surface
  RAY_SHADE_FROM_OTHER_SIDE        = 8192,
  RAY_SHADE_FROM_SKY_LIGHT         = 16384,
};

static inline uint unpackRayFlags(uint a_flags)                       { return ((a_flags & 0xFFFF0000) >> 16); } 
static inline uint packRayFlags(uint a_oldData, uint a_flags)         { return (a_oldData & 0x0000FFFF) | (a_flags << 16); } 

static inline uint unpackBounceNum(uint a_flags)     { return ((a_flags & 0x0000FF00) >> 8); }          
static inline uint unpackBounceNumDiff(uint a_flags) { return (a_flags & 0x000000FF); }                 

static inline uint packBounceNum    (uint a_oldData, uint a_bounceNum)  { return (a_oldData & 0xFFFF00FF) | (a_bounceNum << 8); } 
static inline uint packBounceNumDiff(uint a_oldData, uint a_bounceNum)  { return (a_oldData & 0xFFFFFF00) | (a_bounceNum); } 

static inline bool rayIsActiveS(RayFlags a_flags) { return (a_flags.otherFlags & (RAY_GRAMMAR_OUT_OF_SCENE | RAY_IS_DEAD)) == 0; }
static inline bool rayIsActiveU(uint     a_flags) { return ( ((a_flags & 0xFFFF0000) >> 16) & (RAY_GRAMMAR_OUT_OF_SCENE | RAY_IS_DEAD)) == 0; }

static inline bool isEyeRay(uint a_flags)
{
  const uint otherFlags = unpackRayFlags(a_flags);
  const bool haveSomeNonSpecularReflections = (otherFlags & RAY_EVENT_D) || (otherFlags & RAY_EVENT_G);
  return (unpackBounceNum(a_flags) == 0) || !haveSomeNonSpecularReflections;
  // return (unpackBounceNum(a_flags) == 0);
}

typedef struct MisDataT
{
  float matSamplePdf;
  float cosThetaPrev;
  int   prevMaterialOffset;
  int   isSpecular;

} MisData;

static inline MisData makeInitialMisData()
{
  MisData data;
  data.matSamplePdf = 1.0f;
  data.cosThetaPrev = 1.0f;
  data.prevMaterialOffset = -1;
  data.isSpecular         = 1;
  return data;
}

static inline uint encodeNormal(float3 n)
{
  short x = (short)(n.x*32767.0f);
  short y = (short)(n.y*32767.0f);

  ushort sign = (n.z >= 0) ? 0 : 1;

  int sx = ((int)(x & 0xfffe) | sign);
  int sy = ((int)(y & 0xfffe) << 16);

  return (sx | sy);
}

static inline float3 decodeNormal(uint a_data)
{
  const float divInv = 1.0f / 32767.0f;

  short a_enc_x, a_enc_y;

  a_enc_x = (short)(a_data & 0x0000FFFF);
  a_enc_y = (short)((int)(a_data & 0xFFFF0000) >> 16);

  float sign = (a_enc_x & 0x0001) ? -1.0f : 1.0f;

  float x = (short)(a_enc_x & 0xfffe)*divInv;
  float y = (short)(a_enc_y & 0xfffe)*divInv;
  float z = sign*sqrt(fmax(1.0f - x*x - y*y, 0.0f));

  return make_float3(x, y, z);
}

struct ALIGN_S(16) HitPosNormT
{
  float  pos_x;
  float  pos_y;
  float  pos_z;
  uint   norm_xy;

#ifdef __CUDACC__

  __device__ float3 GetNormal() const { return decodeNormal(norm_xy); }
  __device__ void SetNormal(float3 a_norm) { norm_xy = encodeNormal(normalize(a_norm)); }

#endif

};

typedef struct HitPosNormT HitPosNorm;

ID_CALL HitPosNorm make_HitPosNorm(float4 a_data)
{
  HitPosNorm res;
  res.pos_x   = a_data.x;
  res.pos_y   = a_data.y;
  res.pos_z   = a_data.z;
  res.norm_xy = (uint)(as_int(a_data.w));
  return res;
}

IDH_CALL float3 GetPos(HitPosNorm a_data) { return make_float3(a_data.pos_x, a_data.pos_y, a_data.pos_z); }
IDH_CALL void   SetPos(__private HitPosNorm* a_pData, float3 a_pos) { a_pData->pos_x = a_pos.x; a_pData->pos_y = a_pos.y; a_pData->pos_z = a_pos.z; }

struct ALIGN_S(8) HitTexCoordT
{
  float  tex_u;
  float  tex_v;
};

typedef struct HitTexCoordT HitTexCoord;


struct ALIGN_S(8) HitMatRefT
{
  int   m_data;
  float accumDist;
};

typedef struct HitMatRefT HitMatRef;

IDH_CALL int GetMaterialId(HitMatRef a_hitMat) { return a_hitMat.m_data; }

IDH_CALL void SetHitType(__private HitMatRef* a_pHitMat, int a_id)
{
  int mask = a_id << 28;
  int m_data2 = a_pHitMat->m_data & 0x0FFFFFFF;
  a_pHitMat->m_data = m_data2 | mask;
}


IDH_CALL void SetMaterialId(__private HitMatRef* a_pHitMat, int a_mat_id)
{
  int mask = a_mat_id & 0x0FFFFFFF;
  int m_data2 = a_pHitMat->m_data & 0xF0000000;
  a_pHitMat->m_data = m_data2 | mask;
}


struct ALIGN_S(8) Hit_Part4T
{
  uint tangentCompressed;
  uint bitangentCompressed;
};

typedef struct Hit_Part4T Hit_Part4;

static inline void CoordinateSystem(float3 v1, __private float3* v2, __private float3* v3)
{
  float invLen = 1.0f;

  if (fabs(v1.x) > fabs(v1.y))
  {
    invLen = 1.0f / sqrt(v1.x*v1.x + v1.z*v1.z);
    (*v2) = make_float3(-v1.z * invLen, 0.0f, v1.x * invLen);
  }
  else
  {
    invLen = 1.0f / sqrt(v1.y*v1.y + v1.z*v1.z);
    (*v2) = make_float3(0.0f, v1.z * invLen, -v1.y * invLen);
  }

  (*v3) = cross(v1, (*v2));
}


IDH_CALL float3 MapSampleToCosineDistribution(float r1, float r2, float3 direction, float3 hit_norm, float power)
{
  if(power >= 1e6f)
    return direction;

  float sin_phi = sin(2.0f*r1*3.141592654f);
  float cos_phi = cos(2.0f*r1*3.141592654f);

  //sincos(2.0f*r1*3.141592654f, &sin_phi, &cos_phi);

  float cos_theta = pow(1.0f - r2, 1.0f / (power + 1.0f));
  float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

  float3 deviation;
  deviation.x = sin_theta*cos_phi;
  deviation.y = sin_theta*sin_phi;
  deviation.z = cos_theta;

  float3 ny = direction, nx, nz;
  CoordinateSystem(ny, &nx, &nz);

  {
    float3 temp = ny;
    ny = nz;
    nz = temp;
  }

  float3 res = nx*deviation.x + ny*deviation.y + nz*deviation.z;

  float invSign = dot(direction, hit_norm) > 0.0f ? 1.0f : -1.0f;

  if (invSign*dot(res, hit_norm) < 0.0f) // reflected ray is below surface #CHECK_THIS
  {
    res = (-1.0f)*nx*deviation.x + ny*deviation.y - nz*deviation.z;
    //belowSurface = true;
  }

  return res;
}


// Using the modified Phong reflectance model for physically based rendering
//
IDH_CALL float3 MapSampleToModifiedCosineDistribution(float r1, float r2, float3 direction, float3 hit_norm, float power)
{
  if (power >= 1e6f)
    return direction;

  // float sin_phi, cos_phi;
  // sincosf(2 * r1*3.141592654f, &sin_phi, &cos_phi);
  float sin_phi = sin(2.0f*r1*3.141592654f);
  float cos_phi = cos(2.0f*r1*3.141592654f);

  float sin_theta = sqrt(1.0f - pow(r2, 2.0f / (power + 1.0f)));

  float3 deviation;
  deviation.x = sin_theta*cos_phi;
  deviation.y = sin_theta*sin_phi;
  deviation.z = pow(r2, 1.0f / (power + 1.0f));

  float3 ny = direction, nx, nz;
  CoordinateSystem(ny, &nx, &nz);

  {
    float3 temp = ny;
    ny = nz;
    nz = temp;
  }

  float3 res = nx*deviation.x + ny*deviation.y + nz*deviation.z;

  float invSign = dot(direction, hit_norm) >= 0.0f ? 1.0f : -1.0f;
  if (invSign*dot(res, hit_norm) < 0.0f)                                  // reflected ray is below surface #CHECK_THIS
    res = (-1.0f)*nx*deviation.x - ny*deviation.y + nz*deviation.z;

  return res;
}


/**
\brief  transform float2 sample in rect [-1,1]x[-1,1] to disc centered at (0,0) with radius == 1. 
\param  xy - input sample in rect [-1,1]x[-1,1]
\return position in disc

*/
static inline float2 MapSamplesToDisc(float2 xy)
{
  float x = xy.x;
  float y = xy.y;

  float r = 0;
  float phi = 0;

  float2 res = xy;

  if (x>y && x>-y)
  {
    r = x;
    phi = 0.25f*3.141592654f*(y / x);
  }

  if (x < y && x > -y)
  {
    r = y;
    phi = 0.25f*3.141592654f*(2.0f - x / y);
  }

  if (x < y && x < -y)
  {
    r = -x;
    phi = 0.25f*3.141592654f*(4.0f + y / x);
  }

  if (x >y && x<-y)
  {
    r = -y;
    phi = 0.25f*3.141592654f*(6 - x / y);
  }

  //float sin_phi, cos_phi;
  //sincosf(phi, &sin_phi, &cos_phi);
  float sin_phi = sin(phi);
  float cos_phi = cos(phi);

  res.x = r*sin_phi;
  res.y = r*cos_phi;

  return res;
}


static inline float3 MapSamplesToCone(float cosCutoff, float2 sample, float3 direction)
{
  float cosTheta = (1.0f - sample.x) + sample.x * cosCutoff;
  float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

  //float sinPhi, cosPhi;
  //sincosf(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);
  float sinPhi = sin(2.0f * M_PI * sample.y);
  float cosPhi = cos(2.0f * M_PI * sample.y);

  float3 deviation = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);

  // transform to different basis
  //
  float3 ny = direction;
  float3 nx, nz;
  CoordinateSystem(ny, &nx, &nz);

  //swap(ny, nz);
  {
    float3 temp = ny;
    ny = nz;
    nz = temp;
  }

  return nx*deviation.x + ny*deviation.y + nz*deviation.z;
}

IDH_CALL float3 MapSamplesToSphere(float r1, float r2) // [-1..1]
{
  float phi = r1*3.141592654f * 2.0f; // [0  .. 2PI]
  float h   = r2*2.0f - 1.0f;         // [-1 .. 1]

  float sin_phi = sin(phi);
  float cos_phi = cos(phi);

  float x = sin_phi*sqrt(1 - h*h);
  float y = cos_phi*sqrt(1 - h*h);
  float z = h;

  return make_float3(x, y, z);
}



struct ALIGN_S(16) ZBlockT
{

#ifndef OCL_COMPILER  
#ifndef __CUDACC__

  ZBlockT() { index = 0; diff = 100; counter = 0; index2 = 0; }
  
  ZBlockT(int a_index, float a_diff)
  {
    index   = a_index;
    index2  = 0;
    diff    = a_diff;
    counter = 0;
  }

#endif

  inline static int GetSize() { return Z_ORDER_BLOCK_SIZE*Z_ORDER_BLOCK_SIZE; }
  inline int GetOffset() const { return index*GetSize(); }
  inline bool operator<(const ZBlockT& rhs) const { return (diff < rhs.diff); }
#endif

  int index;   // just block offset if global screen buffer
  int index2;  // index in other buffer + avg trace depth
  int counter; // how many times this block was traced?
  float diff;  // error in some units. stop criterion if fact
};

typedef struct ZBlockT ZBlock;

//IDH_CALL uint unpackAvgTraceDepth(uint a_flags)                       { return ((a_flags  & 0xFF000000) >> 24); }
//IDH_CALL uint packAvgTraceDepth(uint a_oldData, uint a_flags)         { return (a_oldData & 0x00FFFFFF) | (a_flags << 24); }

//IDH_CALL uint unpackIndex2(uint a_flags)                       { return (a_flags   & 0x00FFFFFF); }
//IDH_CALL uint packIndex2(uint a_oldData, uint a_flags)         { return (a_oldData & 0xFF000000) | a_flags; }


static bool BlockFinished(ZBlock block, int a_minRaysPerPixel, int a_maxRaysPerPixel, float* a_outDiff) // for use on the cpu side ... for current
{
  int samplesPerPixel = block.counter; // was *2 due to odd and even staff

  if(a_outDiff!=NULL)
    *a_outDiff = block.diff;

  float acceptedBadPixels = 8.0f; // sqrt((float)(CMP_RESULTS_BLOCK_SIZE));
  int minRaysPerPixel     = a_minRaysPerPixel;

  bool summErrorOk = (block.diff <= acceptedBadPixels);
  bool maxErrorOk = false;

  return ((summErrorOk || maxErrorOk) && samplesPerPixel >= minRaysPerPixel) || (samplesPerPixel >= a_maxRaysPerPixel);
}

IDH_CALL uint ThreadSwizzle1D(uint pixelId, uint zBlockIndex)
{
  uint indexInsideZBlock = pixelId%CMP_RESULTS_BLOCK_SIZE;
  return zBlockIndex*CMP_RESULTS_BLOCK_SIZE + indexInsideZBlock;
}


static inline float PdfAtoW(const float aPdfA, const float aDist, const float aCosThere)
{
  return (aPdfA*aDist*aDist) / fmax(aCosThere, DEPSILON2);
}

static inline float PdfWtoA(const float aPdfW, const float aDist, const float aCosThere)
{
  return aPdfW * fabs(aCosThere) / fmax(aDist*aDist, DEPSILON2);
}

struct MRaysStat
{
#ifndef OCL_COMPILER
  MRaysStat() { memset(this, 0, sizeof(MRaysStat)); }
#endif

  int   traceTimePerCent;
  float raysPerSec;
  float samplesPerSec;

  float reorderTimeMs;

  float traversalTimeMs;
  float samLightTimeMs;
  float shadowTimeMs;
  float shadeTimeMs;
  float bounceTimeMS;
  float evalHitMs;
  float nextBounceMs;

  float sampleTimeMS;
};

IDH_CALL float probabilityAbsorbRR(uint a_flags, uint a_globalFlags)
{
  if (a_globalFlags & HRT_ENABLE_MMLT) // metropolis don't use roultte
    return 0.0f;

  const uint diffBounceNum = unpackBounceNumDiff(a_flags);
  const uint otherFlags    = unpackRayFlags(a_flags);

  float pabsorb = 0.0f;

  if (diffBounceNum >= 4)
    pabsorb = 0.50f;
  else if (diffBounceNum >= 3)
    pabsorb = 0.25f;
  else
    pabsorb = 0.0f;

  return pabsorb;
}

static inline float MonteCarloVariance(float3 avgColor, float sqrColor, int nSamples)
{
  const float  maxColor  = fmax(avgColor.x, fmax(avgColor.y, avgColor.z));
  const float fnSamples  = ((float)(nSamples));
  const float nSampleInv = 1.0f / fnSamples;

  return fabs(sqrColor*nSampleInv - (maxColor*maxColor*nSampleInv*nSampleInv));
}

static inline float MonteCarloRelErr(float maxColor, float sqrColor, int nSamples)
{
  const float fnSamples  = ((float)(nSamples));
  const float nSampleInv = 1.0f / fnSamples;
  
  const float variance   = fabs(sqrColor*nSampleInv - (maxColor*maxColor*nSampleInv*nSampleInv));
  const float stdError   = sqrt(variance);

  return stdError / (fmax(maxColor, 0.00001f));
}

static inline float MonteCarloRelErr2(float3 avgColor, float sqrColor, int nSamples)
{
  const float maxColor = fmax(avgColor.x, fmax(avgColor.y, avgColor.z));
  return MonteCarloRelErr(maxColor, sqrColor, nSamples);
}


static inline float colorSquareMax3(float3 calcColor)
{
  float3 calcColorSqr;
  calcColorSqr.x = calcColor.x*calcColor.x;
  calcColorSqr.y = calcColor.y*calcColor.y;
  calcColorSqr.z = calcColor.z*calcColor.z;

  return fmax(calcColorSqr.x, fmax(calcColorSqr.y, calcColorSqr.z));
}

static inline float colorSquareMax4(float4 calcColor)
{
  float4 calcColorSqr;

  calcColorSqr.x = calcColor.x*calcColor.x;
  calcColorSqr.y = calcColor.y*calcColor.y;
  calcColorSqr.z = calcColor.z*calcColor.z;
  calcColorSqr.w = calcColor.w*calcColor.w;

  return fmax(calcColorSqr.x, fmax(calcColorSqr.y, calcColorSqr.z));
}


// Unpolarized fresnel reflection term for dielectric materials
// this formula is simplified and should be checked 
//
static inline float fresnelCoeffSimple(float cosThetaI, float a_eta)
{
  float g = sqrt(a_eta*a_eta - 1.0f + cosThetaI * cosThetaI);
  float t1 = (g - cosThetaI) / (g + cosThetaI);
  float t2 = (cosThetaI * (g + cosThetaI) - 1) / (cosThetaI * (g - cosThetaI) + 1.0f);
  return 0.5f * t1 * t1 * (1.0f + t2 * t2);
}

//  The following functions calculate the reflected and refracted 
//	directions in addition to the fresnel coefficients. Based on PBRT
//	and the paper "Derivation of Refraction Formulas" by Paul S. Heckbert.
//

static inline float fresnelDielectric(float cosTheta1, float cosTheta2, float etaExt, float etaInt)
{
  float Rs = (etaExt * cosTheta1 - etaInt * cosTheta2) / (etaExt * cosTheta1 + etaInt * cosTheta2);
  float Rp = (etaInt * cosTheta1 - etaExt * cosTheta2) / (etaInt * cosTheta1 + etaExt * cosTheta2);

  return (Rs * Rs + Rp * Rp) / 2.0f;
}

static inline float fresnelConductor(float cosTheta, float eta, float roughness)
{
  float tmp = (eta*eta + roughness*roughness) * (cosTheta * cosTheta);
  float rParl2 = (tmp - (eta * (2.0f * cosTheta)) + 1.0f) / (tmp + (eta * (2.0f * cosTheta)) + 1.0f);
  float tmpF = eta*eta + roughness*roughness;
  float rPerp2 = (tmpF - (eta * (2.0f * cosTheta)) + (cosTheta*cosTheta)) / (tmpF + (eta * (2.0f * cosTheta)) + (cosTheta*cosTheta));
  return (rParl2 + rPerp2) / 2.0f;
}


static inline float fresnelReflectionCoeff(float cosTheta1, float etaExt, float etaInt)
{
  // Swap the indices of refraction if the interaction starts
  // at the inside of the object
  //
  if (cosTheta1 < 0.0f)
  {
    float temp = etaInt;
    etaInt = etaExt;
    etaExt = temp;
  }

  // Using Snell's law, calculate the sine of the angle
  // between the transmitted ray and the surface normal 
  //
  float sinTheta2 = etaExt / etaInt * sqrt(fmax(0.0f, 1.0f - cosTheta1*cosTheta1));

  if (sinTheta2 > 1.0f)
    return 1.0f;  // Total internal reflection!

  // Use the sin^2+cos^2=1 identity - max() guards against
  //	numerical imprecision
  //
  float cosTheta2 = sqrt(fmax(0.0f, 1.0f - sinTheta2*sinTheta2));

  // Finally compute the reflection coefficient
  //
  return fresnelDielectric(fabs(cosTheta1), cosTheta2, etaInt, etaExt);
}

static inline float fresnelReflectionCoeffMentalLike(float cosTheta, float refractIOR)
{
  return fresnelReflectionCoeff(fabs(cosTheta), 1.0f, refractIOR);
}


static inline float contribFunc(float3 color)
{
  return fmax(0.33334f*(color.x + color.y + color.z), 0.0f);
}

static inline int packXY1616(int x, int y) { return (y << 16) | (x & 0x0000FFFF); }

// CPU and CUDA only code 
//

#ifndef OCL_COMPILER 

IDH_CALL float3 clamp3(float3 x, float a, float b) { return make_float3(fmin(fmax(x.x, a), b), fmin(fmax(x.y, a), b), fmin(fmax(x.z, a), b)); }

static unsigned short MortonTable256Host[] =
{
  0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015,
  0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055,
  0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115,
  0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155,
  0x0400, 0x0401, 0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415,
  0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455,
  0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515,
  0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555,
  0x1000, 0x1001, 0x1004, 0x1005, 0x1010, 0x1011, 0x1014, 0x1015,
  0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055,
  0x1100, 0x1101, 0x1104, 0x1105, 0x1110, 0x1111, 0x1114, 0x1115,
  0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155,
  0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411, 0x1414, 0x1415,
  0x1440, 0x1441, 0x1444, 0x1445, 0x1450, 0x1451, 0x1454, 0x1455,
  0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514, 0x1515,
  0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555,
  0x4000, 0x4001, 0x4004, 0x4005, 0x4010, 0x4011, 0x4014, 0x4015,
  0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055,
  0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115,
  0x4140, 0x4141, 0x4144, 0x4145, 0x4150, 0x4151, 0x4154, 0x4155,
  0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415,
  0x4440, 0x4441, 0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455,
  0x4500, 0x4501, 0x4504, 0x4505, 0x4510, 0x4511, 0x4514, 0x4515,
  0x4540, 0x4541, 0x4544, 0x4545, 0x4550, 0x4551, 0x4554, 0x4555,
  0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015,
  0x5040, 0x5041, 0x5044, 0x5045, 0x5050, 0x5051, 0x5054, 0x5055,
  0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115,
  0x5140, 0x5141, 0x5144, 0x5145, 0x5150, 0x5151, 0x5154, 0x5155,
  0x5400, 0x5401, 0x5404, 0x5405, 0x5410, 0x5411, 0x5414, 0x5415,
  0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451, 0x5454, 0x5455,
  0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515,
  0x5540, 0x5541, 0x5544, 0x5545, 0x5550, 0x5551, 0x5554, 0x5555
};

static inline uint ZIndexHost(ushort x, ushort y)
{
  return	MortonTable256Host[y >> 8] << 17  |
          MortonTable256Host[x >> 8] << 16  |
          MortonTable256Host[y & 0xFF] << 1 |
          MortonTable256Host[x & 0xFF];
}

static inline uint HostIndexZBlock2D(int x, int y, int pitch)
{
  uint zOrderX = x % Z_ORDER_BLOCK_SIZE;
  uint zOrderY = y % Z_ORDER_BLOCK_SIZE;

  uint zIndex = ZIndexHost(zOrderX, zOrderY);

  uint wBlocks = pitch / Z_ORDER_BLOCK_SIZE;
  uint blockX = x / Z_ORDER_BLOCK_SIZE;
  uint blockY = y / Z_ORDER_BLOCK_SIZE;

  return (blockX + (blockY)*(wBlocks))*Z_ORDER_BLOCK_SIZE*Z_ORDER_BLOCK_SIZE + zIndex;
}


static void ImageZBlockMemToRowPitch(const float4* inData, float4* outData, int w, int h)
{
  #pragma omp parallel for
  for (int y = 0; y<h; y++)
  {
    for (int x = 0; x<w; x++)
    {
      int indexSrc = HostIndexZBlock2D(x, y, w);
      int indexDst = Index2D(x, y, w);
      outData[indexDst] = inData[indexSrc];
    }
  }


}

#endif


static inline int calcMegaBlockSize(int w, int h, size_t memAmount)
{ 
  int MEGA_BLOCK_SIZE = 512*512;
  if (w*h <= 512 * 512)
    MEGA_BLOCK_SIZE = 256 * 256;
  else if (w*h <= 1024 * 768)
    MEGA_BLOCK_SIZE = 512 * 512;
  else if (w*h < 1920 * 1200)
    MEGA_BLOCK_SIZE = (1024*1024/2); // #TODO: check this !!!
  else 
    MEGA_BLOCK_SIZE = 1024 * 1024;
  return MEGA_BLOCK_SIZE;
}


#ifdef __CUDACC__ 
  #undef ushort
  #undef uint
#endif


#define PREFIX_SUMM_MACRO(idata,odata,l_Data,_bsize)       \
{                                                          \
  uint pos = 2 * LOCAL_ID_X - (LOCAL_ID_X & (_bsize - 1)); \
  l_Data[pos] = 0;                                         \
  pos += _bsize;                                           \
  l_Data[pos] = idata;                                     \
                                                           \
  for (uint offset = 1; offset < _bsize; offset <<= 1)     \
  {                                                        \
    SYNCTHREADS_LOCAL;                                     \
    uint t = l_Data[pos] + l_Data[pos - offset];           \
    SYNCTHREADS_LOCAL;                                     \
    l_Data[pos] = t;                                       \
  }                                                        \
                                                           \
  odata = l_Data[pos];                                     \
}                                                          \



enum CLEAR_FLAGS{ CLEAR_MATERIALS   = 1, 
                  CLEAR_GEOMETRY    = 2, 
                  CLEAR_LIGHTS      = 4,
                  CLEAR_TEXTURES    = 8,
                  CLEAR_CUSTOM_DATA = 16,
                  CLEAR_ALL         = CLEAR_MATERIALS | CLEAR_GEOMETRY | CLEAR_LIGHTS | CLEAR_TEXTURES | CLEAR_CUSTOM_DATA };


enum BVH_FLAGS { BVH_ENABLE_SMOOTH_OPACITY = 1};


typedef struct GBuffer1T
{
  float  depth;
  float3 norm;
  float4 rgba;
  int    matId;
  float  coverage;

} GBuffer1;

typedef struct GBuffer2T
{
  float2 texCoord;
  int    objId;
  int    instId;
} GBuffer2;


typedef struct GBufferAll
{
  GBuffer1 data1;
  GBuffer2 data2;

} GBufferAll;

static inline void initGBufferAll(__private GBufferAll* a_pElem)
{
  a_pElem->data1.depth    = 1e+6f;
  a_pElem->data1.norm     = make_float3(0, 0, 0);
  a_pElem->data1.rgba     = make_float4(0, 0, 0, 1);
  a_pElem->data1.matId    = -1;
  a_pElem->data1.coverage = 0.0f;

  a_pElem->data2.texCoord = make_float2(0, 0);
  a_pElem->data2.objId    = -1;
  a_pElem->data2.instId   = -1;
}

#define GBUFFER_SAMPLES 16

static inline float4 packGBuffer1(GBuffer1 a_input)
{
  float4 resColor;

  unsigned int packedRGBX = RealColorToUint32(a_input.rgba);

  const float clampedCoverage  = fmin(fmax(a_input.coverage*255.0f, 0.0f), 255.0f);
  const int compressedCoverage = ((int)(clampedCoverage)) << 24;
  const int packedMIdAncCov    = (a_input.matId & 0x00FFFFFF) | (compressedCoverage & 0xFF000000);

  resColor.x = a_input.depth;
  resColor.y = as_float(encodeNormal(a_input.norm));
  resColor.z = as_float(packedMIdAncCov);
  resColor.w = as_float(packedRGBX);

  return resColor;
}


static inline GBuffer1 unpackGBuffer1(float4 a_input)
{
  GBuffer1 res;

  res.depth = a_input.x;
  res.norm  = decodeNormal(as_int(a_input.y));
  res.matId = as_int(a_input.z) & 0x00FFFFFF;

  const int compressedCoverage = (as_int(a_input.z) & 0xFF000000) >> 24;
  res.coverage = ((float)compressedCoverage)*(1.0f / 255.0f);

  unsigned int rgba = as_int(a_input.w);
  res.rgba.x = (rgba & 0x000000FF)*(1.0f / 255.0f);
  res.rgba.y = ((rgba & 0x0000FF00) >> 8)*(1.0f / 255.0f);
  res.rgba.z = ((rgba & 0x00FF0000) >> 16)*(1.0f / 255.0f);
  res.rgba.w = ((rgba & 0xFF000000) >> 24)*(1.0f / 255.0f);

  return res;
}

static inline float4 packGBuffer2(GBuffer2 a_input)
{
  float4 res;
  res.x = a_input.texCoord.x;
  res.y = a_input.texCoord.y;
  res.z = as_float(a_input.objId);
  res.w = as_float(a_input.instId);
  return res;
}

static inline GBuffer2 unpackGBuffer2(float4 a_input)
{
  GBuffer2 res;
  res.texCoord.x = a_input.x;
  res.texCoord.y = a_input.y;
  res.objId      = as_int(a_input.z);
  res.instId     = as_int(a_input.w);
  return res;
}



static inline float projectedPixelSize(float dist, float FOV, float w, float h)
{
  float ppx = (FOV / w)*dist;
  float ppy = (FOV / h)*dist;

  if (dist > 0.0f)
    return 2.0f*fmax(ppx, ppy);
  else
    return 1000.0f;
}

static inline float surfaceSimilarity(float4 data1, float4 data2, const float MADXDIFF)
{
  const float MANXDIFF = 0.15f;

  float3 n1 = to_float3(data1);
  float3 n2 = to_float3(data2);

  float dist = length(n1 - n2);
  if (dist >= MANXDIFF)
    return 0.0f;

  float d1 = data1.w;
  float d2 = data2.w;

  if (fabs(d1 - d2) >= MADXDIFF)
    return 0.0f;

  float normalSimilar = sqrt(1.0f - (dist / MANXDIFF));
  float depthSimilar  = sqrt(1.0f - fabs(d1 - d2) / MADXDIFF);

  return normalSimilar * depthSimilar;
}

static inline float gbuffDiff(GBufferAll s1, GBufferAll s2, const float a_fov, float w, float h)
{
  const float ppSize         = projectedPixelSize(s1.data1.depth, a_fov, w, h);
  const float surfaceSimilar = surfaceSimilarity(to_float4(s1.data1.norm, s1.data1.depth),
                                                 to_float4(s2.data1.norm, s2.data1.depth), ppSize*2.0f);

  const float surfaceDiff    = 1.0f - surfaceSimilar;
  const float objDiff        = (s1.data2.instId == s2.data2.instId && s1.data2.objId == s2.data2.objId) ? 0.0f : 1.0f;
  const float matDiff        = (s1.data1.matId  == s2.data1.matId) ? 0.0f : 1.0f;
  const float alphaDiff      = fabs(s1.data1.rgba.w - s2.data1.rgba.w);

  return surfaceDiff + objDiff + matDiff + alphaDiff;
}

static inline float gbuffDiffObj(GBufferAll s1, GBufferAll s2, const float a_fov, int w, int h)
{
  const float objDiff = (s1.data2.instId == s2.data2.instId && s1.data2.objId == s2.data2.objId) ? 0.0f : 1.0f;
  const float matDiff = (s1.data1.matId  == s2.data1.matId) ? 0.0f : 1.0f;

  return objDiff + matDiff;
}


enum PLAIN_LIGHT_TYPES {
  PLAIN_LIGHT_TYPE_POINT_OMNI   = 0,
  PLAIN_LIGHT_TYPE_POINT_SPOT   = 1,
  PLAIN_LIGHT_TYPE_DIRECT       = 2,
  PLAIN_LIGHT_TYPE_SKY_DOME     = 3,
  PLAIN_LIGHT_TYPE_AREA         = 4,
  PLAIN_LIGHT_TYPE_SPHERE       = 5,
  PLAIN_LIGHT_TYPE_CYLINDER     = 6,
  PLAIN_LIGHT_TYPE_MESH         = 7,
};

enum PLAIN_LIGHT_FLAGS{
  DISABLE_SAMPLING                = 1,
  SEPARATE_SKY_LIGHT_ENVIRONMENT  = 2,
  SKY_LIGHT_USE_PEREZ_ENVIRONMENT = 4,
  AREA_LIGHT_SKY_PORTAL           = 8,
  LIGHT_HAS_IES                   = 16,  ///< have spherical distribution mask around light
  LIGHT_IES_POINT_AREA            = 32,  ///< apply IES honio from the center of light always.
  LIGHT_DO_NOT_SAMPLE_ME          = 64,  ///< zero selection probability. never sample it.
};


enum SKY_PORTAL_COLOR_SOURCE { SKY_PORTAL_SOURCE_ENVIRONMENT = 1, 
                               SKY_PORTAL_SOURCE_SKYLIGHT    = 2,
                               SKY_PORTAL_SOURCE_CUSTOM      = 3
};


static inline float3 triBaricentrics3(float3 ray_pos, float3 ray_dir, float3 A_pos, float3 B_pos, float3 C_pos)
{
  const float3 edge1 = B_pos - A_pos;
  const float3 edge2 = C_pos - A_pos;
  const float3 pvec  = cross(ray_dir, edge2);
  const float  det   = dot(edge1, pvec);

  const float  inv_det = 1.0f / det;
  const float3 tvec    = ray_pos - A_pos;
  const float  v       = dot(tvec, pvec)*inv_det;

  const float3 qvec = cross(tvec, edge1);
  const float  u    = dot(ray_dir, qvec)*inv_det;
  const float  t    = dot(edge2, qvec)*inv_det;

  return make_float3(u, v, t);
}


typedef struct ShadeContextT
{
  float3 wp;    ///< world pos
                //float3 lp;    ///< local pos
  float3 l;     ///< direction to light
  float3 v;     ///< view vector
  float3 n;     ///< smooth normal (for shading and new rays offsets)

  float3 fn;    ///< flat normal (for bump mapping and tangent space transform)
  float3 tg;    ///< tangent     (for bump mapping and tangent space transform)
  float3 bn;    ///< binormal    (for bump mapping and tangent space transform)

  float2 tc;    ///< tex coord (0);
                //float2 tc1; ///< tex coord (1);

  bool   hfi;   ///< Hit.From.Inside. if hit surface from the inside of the object that have glass or SSS material 

} ShadeContext;


/**
\brief this structure will store results of procedural texture kernel execution.

*/

#define MAXPROCTEX 5

typedef struct ProcTextureListT
{
  float3  fdata4[MAXPROCTEX];  
  int     id_f4 [MAXPROCTEX];

} ProcTextureList;


static inline void InitProcTextureList(__private ProcTextureList* a_pList)
{
  a_pList->id_f4[0] = INVALID_TEXTURE;
  a_pList->id_f4[1] = INVALID_TEXTURE;
  a_pList->id_f4[2] = INVALID_TEXTURE;
  a_pList->id_f4[3] = INVALID_TEXTURE;
  a_pList->id_f4[4] = INVALID_TEXTURE;
}

static inline void ReadProcTextureList(__global float4* fdata, int tid, int size,
                                       __private ProcTextureList* a_pRes)
{
  if (fdata == 0)
    return;

  const float4 f3 = fdata[tid + size * 3];
  const float4 f4 = fdata[tid + size * 4];

  a_pRes->id_f4[0] = as_int(f4.x);
  a_pRes->id_f4[1] = as_int(f4.y);
  a_pRes->id_f4[2] = as_int(f4.z);
  a_pRes->id_f4[3] = as_int(f4.w);
  a_pRes->id_f4[4] = as_int(f3.w);

  a_pRes->fdata4[3] = to_float3(f3);

  if (a_pRes->id_f4[0] != INVALID_TEXTURE || a_pRes->id_f4[4] != INVALID_TEXTURE)
  {
    const float4 f0     = fdata[tid + size * 0];
    a_pRes->fdata4[0]   = to_float3(f0);
    a_pRes->fdata4[4].x = f0.w;
  }

  if (a_pRes->id_f4[1] != INVALID_TEXTURE || a_pRes->id_f4[4] != INVALID_TEXTURE)
  {
    const float4 f1     = fdata[tid + size * 1];
    a_pRes->fdata4[1]   = to_float3(f1);
    a_pRes->fdata4[4].y = f1.w;
  }
  
  if (a_pRes->id_f4[2] != INVALID_TEXTURE || a_pRes->id_f4[4] != INVALID_TEXTURE)
  {
    const float4 f2     = fdata[tid + size * 2];
    a_pRes->fdata4[2]   = to_float3(f2);
    a_pRes->fdata4[4].z = f2.w;
  }
 
}


static inline void WriteProcTextureList(__global float4* fdata, int tid, int size, __private const ProcTextureList* a_pRes)
{
  const float4 f0 = make_float4(a_pRes->fdata4[0].x, a_pRes->fdata4[0].y, a_pRes->fdata4[0].z, a_pRes->fdata4[4].x);
  const float4 f1 = make_float4(a_pRes->fdata4[1].x, a_pRes->fdata4[1].y, a_pRes->fdata4[1].z, a_pRes->fdata4[4].y);
  const float4 f2 = make_float4(a_pRes->fdata4[2].x, a_pRes->fdata4[2].y, a_pRes->fdata4[2].z, a_pRes->fdata4[4].z);
  const float4 f3 = make_float4(a_pRes->fdata4[3].x, a_pRes->fdata4[3].y, a_pRes->fdata4[3].z, as_float(a_pRes->id_f4[4]));

  if (a_pRes->id_f4[0] != INVALID_TEXTURE || a_pRes->id_f4[4] != INVALID_TEXTURE)
    fdata[tid + size * 0] = f0;
  
  if (a_pRes->id_f4[1] != INVALID_TEXTURE || a_pRes->id_f4[4] != INVALID_TEXTURE)
    fdata[tid + size * 1] = f1;

  if (a_pRes->id_f4[2] != INVALID_TEXTURE || a_pRes->id_f4[4] != INVALID_TEXTURE)
    fdata[tid + size * 2] = f2;

  if (a_pRes->id_f4[3] != INVALID_TEXTURE || a_pRes->id_f4[4] != INVALID_TEXTURE)
    fdata[tid + size * 3] = f3;

  fdata[tid + size * 4] = make_float4( as_float(a_pRes->id_f4[0]), as_float(a_pRes->id_f4[1]), as_float(a_pRes->id_f4[2]), as_float(a_pRes->id_f4[3]));
}


static inline bool isProcTexId(int a_texId, const __private ProcTextureList* a_pList)
{
  return (a_pList->id_f4[0] != INVALID_TEXTURE);
}

/**
\brief get color for precomputed procedural texture
\param a_texId       - input tex id
\param a_pList       - input ptl

\return texture color; 
*/

static inline float4 readProcTex(int a_texId, const __private ProcTextureList* a_pList)
{
  float4 res = make_float4(1, 1, 1, -1.0f);

  res = (a_texId == a_pList->id_f4[0]) ? to_float4(a_pList->fdata4[0], 0.0f) : res;
  res = (a_texId == a_pList->id_f4[1]) ? to_float4(a_pList->fdata4[1], 0.0f) : res;
  res = (a_texId == a_pList->id_f4[2]) ? to_float4(a_pList->fdata4[2], 0.0f) : res;
  res = (a_texId == a_pList->id_f4[3]) ? to_float4(a_pList->fdata4[3], 0.0f) : res;
  res = (a_texId == a_pList->id_f4[4]) ? to_float4(a_pList->fdata4[4], 0.0f) : res;

  //if (a_texId == a_pList->id_f4[0])
  //  res = to_float4(a_pList->fdata4[0], 0.0f);
  //if (a_texId == a_pList->id_f4[1])
  //  res = to_float4(a_pList->fdata4[1], 0.0f);
  //if (a_texId == a_pList->id_f4[2])
  //  res = to_float4(a_pList->fdata4[2], 0.0f);
  //if (a_texId == a_pList->id_f4[3])
  //  res = to_float4(a_pList->fdata4[3], 0.0f);
  //if (a_texId == a_pList->id_f4[4])
  //  res = to_float4(a_pList->fdata4[4], 0.0f);

  return res;
}

typedef struct ShadowSampleT
{

  float3 pos;
  float3 color;
  float  pdf;
  float  maxDist;
  float  cosAtLight;
  bool   isPoint;

} ShadowSample;

/**
\brief Per ray accumulated (for all bounces) data. 

*/
typedef struct ALIGN_S(16) PerRayAccT
{
  float  pdfGTerm;    ///< accumulated G term equal to product of G(x1,x2,x3) for all bounces
  float  pdfLightWP;  ///< accumulated probability per projected solid angle for light path
  float  pdfCameraWP; ///< accumulated probability per projected solid angle for camera path
  float  pdfCamA0;    ///< equal to pdfWP[0]*G[0] (if [0] means light)

} PerRayAcc;

typedef struct SurfaceHitT
{
  float3 pos;
  float3 normal;
  float3 flatNormal;
  float3 tangent;
  float3 biTangent;
  float2 texCoord;
  int    matId;
  float  t;
  float  sRayOff;
  bool   hfi;
} SurfaceHit;

static inline PerRayAcc InitialPerParAcc()
{
  PerRayAcc res;
  res.pdfGTerm     = 1.0f;
  res.pdfLightWP   = 1.0f;
  res.pdfCameraWP  = 1.0f;
  res.pdfCamA0     = 1.0f;
  return res;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


enum PLAIN_MAT_TYPES {

  PLAIN_MAT_CLASS_PHONG_SPECULAR = 0,
  PLAIN_MAT_CLASS_BLINN_SPECULAR = 1, // Micro Facet Torrance Sparrow model with Blinn distribution
  PLAIN_MAT_CLASS_PERFECT_MIRROR = 2,
  PLAIN_MAT_CLASS_THIN_GLASS     = 3,
  PLAIN_MAT_CLASS_GLASS          = 4,
  PLAIN_MAT_CLASS_TRANSLUCENT    = 5,
  PLAIN_MAT_CLASS_SHADOW_MATTE   = 6,
  PLAIN_MAT_CLASS_LAMBERT        = 7,
  PLAIN_MAT_CLASS_OREN_NAYAR     = 8,
  PLAIN_MAT_CLASS_BLEND_MASK     = 9,
  PLAIN_MAT_CLASS_EMISSIVE       = 10,
  PLAIN_MAT_CLASS_VOLUME_PERLIN  = 11,
	PLAIN_MAT_CLASS_SSS            = 12
};


enum PLAIN_MAT_FLAGS{

  PLAIN_MATERIAL_IS_LIGHT         = 1,
  PLAIN_MATERIAL_CAST_CAUSTICS    = 2,
  PLAIN_MATERIAL_HAS_DIFFUSE      = 4,
  PLAIN_MATERIAL_HAS_TRANSPARENCY = 8,

  PLAIN_MATERIAL_INVERT_NMAP_X       = 16,
  PLAIN_MATERIAL_INVERT_NMAP_Y       = 32,
  PLAIN_MATERIAL_INVERT_SWAP_NMAP_XY = 64,
  PLAIN_MATERIAL_INVERT_HEIGHT       = 128,
  PLAIN_MATERIAL_SKIP_SHADOW         = 256,
  PLAIN_MATERIAL_FORBID_EMISSIVE_GI  = 512,
  PLAIN_MATERIAL_SKIP_SKY_PORTAL     = 1024,
  PLAIN_MATERIAL_EMISSION_FALOFF     = 2048,

  // This flag marks node as a real blend of different materials.
  // It used for blending emissive properties and normal maps.
  //
  PLAIN_MATERIAL_SURFACE_BLEND        = 4096,
  PLAIN_MATERIAL_HAVE_BTDF            = 8192,
  PLAIN_MATERIAL_INVIS_LIGHT          = 16384,
  PLAIN_MATERIAL_CAN_SAMPLE_REFL_ONLY = 32768,
  PLAIN_MATERIAL_HAVE_PROC_TEXTURES   = 32768*2,
  PLAIN_MATERIAL_LOCAL_AO1            = 32768*4,
  PLAIN_MATERIAL_LOCAL_AO2            = 32768*8,
};

#define PLAIN_MATERIAL_DATA_SIZE        144
#define PLAIN_MATERIAL_CUSTOM_DATA_SIZE 64
#define MIX_TREE_MAX_DEEP               16

struct PlainMaterialT
{
  float data[PLAIN_MATERIAL_DATA_SIZE];
};

typedef struct PlainMaterialT PlainMaterial;

// emissive component, always present in material to speed-up code
//
#define EMISSIVE_COLORX_OFFSET       4
#define EMISSIVE_COLORY_OFFSET       5
#define EMISSIVE_COLORZ_OFFSET       6

#define EMISSIVE_TEXID_OFFSET        7
#define EMISSIVE_TEXMATRIXID_OFFSET  8
#define EMISSIVE_LIGHTID_OFFSET      9

#define OPACITY_TEX_OFFSET           (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+1)
#define OPACITY_TEX_MATRIX           (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+2)

#define NORMAL_TEX_OFFSET            (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+3)
#define NORMAL_TEX_MATRIX            (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+4)

#define EMISSIVE_BLEND_OFFSET        (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+5)
#define PARALLAX_HEIGHT              (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+6)

#define EMISSIVE_SAMPLER_OFFSET      (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+8)
#define NORMAL_SAMPLER_OFFSET        (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+20)
#define OPACITY_SAMPLER_OFFSET       (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+32)

#define PROC_TEX1_F4_HEAD_OFFSET     (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+33)
#define PROC_TEX2_F4_HEAD_OFFSET     (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+34)
#define PROC_TEX3_F4_HEAD_OFFSET     (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+35)
#define PROC_TEX4_F4_HEAD_OFFSET     (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+36)
#define PROC_TEX5_F4_HEAD_OFFSET     (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+37)

#define PROC_TEX_TABLE_OFFSET        (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+38)

#define PROC_TEX_AO_TYPE             (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+39)
#define PROC_TEX_AO_SAMPLER          (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+40)
#define PROC_TEX_TEX_ID              (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+52)
#define PROC_TEXMATRIX_ID            (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+53)
#define PROC_TEX_AO_LENGTH           (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+54)

#define PROC_TEX_AO_TYPE2            (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+55)
#define PROC_TEX_AO_SAMPLER2         (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+56)
#define PROC_TEX_TEX_ID2             (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+68)
#define PROC_TEXMATRIX_ID2           (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+69)
#define PROC_TEX_AO_LENGTH2          (PLAIN_MATERIAL_CUSTOM_DATA_SIZE+70)

enum AO_TYPES { AO_TYPE_NONE = 0, AO_TYPE_UP = 1, AO_TYPE_DOWN = 2, AO_TYPE_BOTH = 4 };


#define PLAIN_MAT_TYPE_OFFSET        0
#define PLAIN_MAT_FLAGS_OFFSET       1
#define PLAIN_MAT_COMPONENTS_OFFSET  2

static inline int  materialGetType         (__global const PlainMaterial* a_pMat) { return as_int(a_pMat->data[PLAIN_MAT_TYPE_OFFSET]); }
static inline int  materialGetFlags        (__global const PlainMaterial* a_pMat) { return as_int(a_pMat->data[PLAIN_MAT_FLAGS_OFFSET]); }

static inline bool materialCastCaustics    (__global const PlainMaterial* a_pMat) { return (as_int(a_pMat->data[PLAIN_MAT_FLAGS_OFFSET]) & PLAIN_MATERIAL_CAST_CAUSTICS) != 0; }
static inline bool materialHasTransparency (__global const PlainMaterial* a_pMat) { return (as_int(a_pMat->data[PLAIN_MAT_FLAGS_OFFSET]) & PLAIN_MATERIAL_HAS_TRANSPARENCY) != 0; }
static inline bool materialIsSkyPortal     (__global const PlainMaterial* a_pMat) { return (as_int(a_pMat->data[PLAIN_MAT_FLAGS_OFFSET]) & PLAIN_MATERIAL_SKIP_SKY_PORTAL) != 0; }
static inline bool materialIsInvisLight    (__global const PlainMaterial* a_pMat) { return (as_int(a_pMat->data[PLAIN_MAT_FLAGS_OFFSET]) & PLAIN_MATERIAL_INVIS_LIGHT) != 0; }


static inline void PutProcTexturesIdListToMaterialHead(const ProcTextureList* a_pData, PlainMaterial* a_pMat)
{
  ((int*)(a_pMat->data))[PROC_TEX1_F4_HEAD_OFFSET] = a_pData->id_f4[0];
  ((int*)(a_pMat->data))[PROC_TEX2_F4_HEAD_OFFSET] = a_pData->id_f4[1];
  ((int*)(a_pMat->data))[PROC_TEX3_F4_HEAD_OFFSET] = a_pData->id_f4[2];
  ((int*)(a_pMat->data))[PROC_TEX4_F4_HEAD_OFFSET] = a_pData->id_f4[3];
  ((int*)(a_pMat->data))[PROC_TEX5_F4_HEAD_OFFSET] = a_pData->id_f4[4];
}

static inline void GetProcTexturesIdListFromMaterialHead(__global const PlainMaterial* a_pMat, __private ProcTextureList* a_pData)
{
  a_pData->id_f4[0] = as_int(a_pMat->data[PROC_TEX1_F4_HEAD_OFFSET]);
  a_pData->id_f4[1] = as_int(a_pMat->data[PROC_TEX2_F4_HEAD_OFFSET]);
  a_pData->id_f4[2] = as_int(a_pMat->data[PROC_TEX3_F4_HEAD_OFFSET]);
  a_pData->id_f4[3] = as_int(a_pMat->data[PROC_TEX4_F4_HEAD_OFFSET]);
  a_pData->id_f4[4] = as_int(a_pMat->data[PROC_TEX5_F4_HEAD_OFFSET]);
}

static inline bool materialHeadHaveTargetProcTex(__global const PlainMaterial* a_pMat, int a_texId)
{
  return (as_int(a_pMat->data[PROC_TEX1_F4_HEAD_OFFSET]) == a_texId || 
          as_int(a_pMat->data[PROC_TEX2_F4_HEAD_OFFSET]) == a_texId ||
          as_int(a_pMat->data[PROC_TEX3_F4_HEAD_OFFSET]) == a_texId ||
          as_int(a_pMat->data[PROC_TEX4_F4_HEAD_OFFSET]) == a_texId ||
          as_int(a_pMat->data[PROC_TEX5_F4_HEAD_OFFSET]) == a_texId);
}

static inline bool MaterialHaveAtLeastOneProcTex(__global const PlainMaterial* a_pMat)
{
  return as_int(a_pMat->data[PROC_TEX1_F4_HEAD_OFFSET]) != INVALID_TEXTURE;
}

static inline bool MaterialHaveAO(__global const PlainMaterial* a_pMat)
{
  return as_int(a_pMat->data[PROC_TEX_AO_TYPE]) != AO_TYPE_NONE;
}

static inline bool MaterialHaveAO2(__global const PlainMaterial* a_pMat)
{
  return as_int(a_pMat->data[PROC_TEX_AO_TYPE]) != AO_TYPE_NONE && as_int(a_pMat->data[PROC_TEX_AO_TYPE2]) != AO_TYPE_NONE;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
\brief  Select index proportional to piecewise constant function that is stored in a_accum[0 .. N-2]; Binary search version.
\param  a_r     - input random variable in rage [0, 1]
\param  a_accum - input float array. it must be a result of prefix summ - i.e. it must be sorted.
\param  N       - size of extended array - i.e. a_accum[N-1] == summ(a_accum[0 .. N-2]).
\param  pPDF    - out parameter. probability of picking up found value.
\return found index

*/
static int SelectIndexPropToOpt(const float a_r, __global const float* a_accum, const int N, 
                                __private float* pPDF) 
{
  int leftBound  = 0;
  int rightBound = N - 2; // because a_accum[N-1] == summ(a_accum[0 .. N-2]).
  int counter    = 0;
  int currPos    = -1;

  const int maxStep = 50;
  const float x = a_r*a_accum[N - 1];

  while (rightBound - leftBound > 1 && counter < maxStep)
  {
    const int currSize = rightBound + leftBound;
    const int currPos1 = (currSize % 2 == 0) ? (currSize + 1) / 2 : (currSize + 0) / 2;

    const float a = a_accum[currPos1 + 0];
    const float b = a_accum[currPos1 + 1];

    if (a < x && x <= b)
    {
      currPos = currPos1;
      break;
    }
    else if (x <= a)
      rightBound = currPos1;
    else if (x > b)
      leftBound = currPos1;

    counter++;
  }

  if (currPos < 0) // check the rest intervals
  {
    const float a1 = a_accum[leftBound + 0];
    const float b1 = a_accum[leftBound + 1];
    const float a2 = a_accum[rightBound + 0];
    const float b2 = a_accum[rightBound + 1];
    if (a1 < x && x <= b1)
      currPos = leftBound;
    if (a2 < x && x <= b2)
      currPos = rightBound;
  }

  if (x == 0.0f)
    currPos = 0;
  else if (currPos < 0)
    currPos = (rightBound + leftBound + 1) / 2;

  (*pPDF) = (a_accum[currPos + 1] - a_accum[currPos]) / a_accum[N - 1];
  return currPos;
}

/**
\brief search for for the lower bound (left range)
\param a          - array
\param length     - array size
\param left_range - value to search for

*/
static inline int binarySearchForLeftRange(__global const int2* a, int length, int left_range)
{
  if (a[length - 1].x < left_range)
    return -1;

  int low = 0;
  int high = length - 1;

  while (low <= high)
  {
    int mid = low + ((high - low) / 2);

    if (a[mid].x >= left_range)
      high = mid - 1;
    else //if(a[mid]<i)
      low = mid + 1;
  }

  return high + 1;
}


/**
\brief search for for the upper bound (right range)
\param a          - array
\param length     - array size
\param left_range - value to search for

*/
static inline int binarySearchForRightRange(__global const int2* a, int length, int right_range)
{
  if (a[0].x > right_range)
    return -1;

  int low = 0;
  int high = length - 1;

  while (low <= high)
  {
    int mid = low + ((high - low) / 2);

    if (a[mid].x > right_range)
      high = mid - 1;
    else //if(a[mid]<i)
      low = mid + 1;
  }

  return low - 1;
}

/**
\brief perform material id remap for instanced objects; 
\param a_mId         - input old material id
\param a_instId      - input instance id 
\param in_remapInst  - array/table that maps instance id to remap list id
\param a_instTabSize - max instance id / size of 'in_remapInst' array
\param in_allMatRemapLists - all remap listss packed in to single array
\param in_remapTable    - array/table that store offset inside 'in_allMatRemapLists' for each remap list which id we got from 'in_remapInst'
\papam a_remapTableSize - size of 'in_remapTable' array

\return new material id

*/
static inline int remapMaterialId(int a_mId, int a_instId, 
                                  __global const int*  in_remapInst, int a_instTabSize, 
                                  __global const int*  in_allMatRemapLists, 
                                  __global const int2* in_remapTable, int a_remapTableSize)
{
  
  if (a_mId < 0 || a_instId < 0 || a_instId >= a_instTabSize || in_remapInst == 0 || in_allMatRemapLists == 0 || in_remapTable == 0)
    return a_mId;

  const int remapListId = in_remapInst[a_instId];
  if(remapListId < 0 || remapListId >= a_remapTableSize) // || remapListId >= some size
    return a_mId;

  const int2 offsAndSize = in_remapTable[remapListId];

  // int res = a_mId;
  // for (int i = 0; i < offsAndSize.y; i++) // #TODO: change to binery search
  // {
  //   int idRemapFrom = in_allMatRemapLists[offsAndSize.x + i * 2 + 0];
  //   int idRemapTo   = in_allMatRemapLists[offsAndSize.x + i * 2 + 1];
  // 
  //   if (idRemapFrom == a_mId)
  //   {
  //     res = idRemapTo;
  //     break;
  //   }
  // }

  int low  = 0;
  int high = offsAndSize.y - 1;
  
  while (low <= high)
  {
    const int mid = low + ((high - low) / 2);
  
    const int idRemapFrom = in_allMatRemapLists[offsAndSize.x + mid * 2 + 0];
  
    if (idRemapFrom >= a_mId)
      high = mid - 1;
    else //if(a[mid]<i)
      low = mid + 1;
  }

  if (high+1 < offsAndSize.y)
  {
    const int idRemapFrom = in_allMatRemapLists[offsAndSize.x + (high + 1) * 2 + 0];
    const int idRemapTo   = in_allMatRemapLists[offsAndSize.x + (high + 1) * 2 + 1];
    const int res         = (idRemapFrom == a_mId) ? idRemapTo : a_mId;
    return res;
  }
  else
    return a_mId;
}


#define AO_RAYS_PACKED 4


#endif

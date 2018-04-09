#ifndef RTCRANDOM
#define RTCRANDOM

#include "globals.h"

#define SIMPLE_RANDOM_GEN

#ifdef SIMPLE_RANDOM_GEN


typedef struct RandomGenT
{
  uint2          state;
  unsigned int   maxNumbers;
  unsigned int   lazy; // or dummy to have uint4 generator data.
  float3         sobol;

#ifdef RAND_MLT_CPU
  __global const float* rptr;
  __global const float* rptr2;

#endif

} RandomGen;


static inline unsigned int NextState(RandomGen* gen)
{
  const unsigned int x = (gen->state).x * 17 + (gen->state).y * 13123;
  (gen->state).x = (x << 13) ^ x;
  (gen->state).y ^= (x << 7);
  return x;
}


static inline RandomGen RandomGenInit(const int a_seed)
{
  RandomGen gen;

  gen.state.x  = (a_seed * (a_seed * a_seed * 15731 + 74323) + 871483);
  gen.state.y  = (a_seed * (a_seed * a_seed * 13734 + 37828) + 234234);
  gen.lazy     = 0;

  for(int i=0;i<(a_seed%7);i++)
    NextState(&gen);

#ifdef RAND_MLT_CPU
  gen.rptr = 0;
#endif

  return gen;
}


static inline unsigned int rndInt_Pseudo(RandomGen* gen)
{
  return NextState(gen);
}

static inline float4 rndFloat4_Pseudo(RandomGen* gen)
{
  unsigned int x = NextState(gen);

  const unsigned int x1 = (x * (x * x * 15731 + 74323) + 871483);
  const unsigned int y1 = (x * (x * x * 13734 + 37828) + 234234);
  const unsigned int z1 = (x * (x * x * 11687 + 26461) + 137589);
  const unsigned int w1 = (x * (x * x * 15707 + 789221) + 1376312589);

  const float scale = (1.0f / 4294967296.0f);

  return make_float4((float)(x1), (float)(y1), (float)(z1), (float)(w1))*scale;
}

static inline float2 rndFloat2_Pseudo(RandomGen* gen)
{
  unsigned int x = NextState(gen);

  const unsigned int x1 = (x * (x * x * 15731 + 74323) + 871483);
  const unsigned int y1 = (x * (x * x * 13734 + 37828) + 234234);

  const float scale     = (1.0f / 4294967296.0f);

  return make_float2((float)(x1), (float)(y1))*scale;
}

static inline float rndFloat1_Pseudo(RandomGen* gen)
{
  const unsigned int x   = NextState(gen);
  const unsigned int tmp = (x * (x * x * 15731 + 74323) + 871483);
  const float scale      = (1.0f / 4294967296.0f);
  return ((float)(tmp))*scale;
}

// static inline float rndFloat1_Pseudo16(RandomGen* gen)
// {
//   const unsigned int x   = NextState(gen);
//   const unsigned int tmp = (x * (x * x * 15731 + 74323) + 871483);
//   const float scale      = (1.0f / 65535.0f);
//   return ((float)(tmp & 0x0000FFFF))*scale;
// }
// 
// static inline float rndFloat1_Pseudo8(RandomGen* gen)
// {
//   const unsigned int x   = NextState(gen);
//   const unsigned int tmp = (x * (x * x * 15731 + 74323) + 871483);
//   const float scale      = (1.0f / 255.0f);
//   return ((float)(tmp & 0x000000FF))*scale;
// }

#else


typedef struct RandomGenT
{
  unsigned int x[5];

#ifdef RAND_MLT_CPU

  __global const float* rptr;
  uint rtop;

#endif

} RandomGen;


static inline unsigned int NextState(RandomGen* gen)
{
  typedef unsigned long long int _my_uint64_t;

  _my_uint64_t sum = (_my_uint64_t)2111111111UL * (_my_uint64_t)(gen->x[3]) +
                     (_my_uint64_t)1492         * (_my_uint64_t)(gen->x[2]) +
                     (_my_uint64_t)1776         * (_my_uint64_t)(gen->x[1]) +
                     (_my_uint64_t)5115         * (_my_uint64_t)(gen->x[0]) +
                     (_my_uint64_t)gen->x[4];
        
  gen->x[3] = gen->x[2];  
  gen->x[2] = gen->x[1];  
  gen->x[1] = gen->x[0];
  gen->x[4] = (unsigned int)(sum >> 32); // Carry
  gen->x[0] = (unsigned int)sum;         // Low 32 bits of sum

  return gen->x[0];
}

static inline RandomGen RandomGenInit(unsigned int a_seed)
{
  RandomGen gen;
  unsigned int s = a_seed;

  // make random numbers and put them into the buffer
  for (int i = 0; i < 5; i++) 
  {
    s = s * 29943829 - 1;
    gen.x[i] = s;
  }

  // randomize some more
  for (int i=0; i<19; i++) 
    NextState(&gen);

#ifdef RAND_MLT_CPU
  gen.rptr = 0;
  gen.rtop = 0;
#endif

  return gen;
}

static inline unsigned int rndInt_Pseudo(RandomGen* gen)
{
  return NextState(gen);
}

static inline float rndFloat1_Pseudo(RandomGen* gen)
{
  return ((float)NextState(gen)) * (1.0f/(65536.0f*65536.0f));
}


static inline float4 rndFloat4_Pseudo(RandomGen* gen)
{
  float4 res;
  res.x = rndFloat1_Pseudo(gen);
  res.y = rndFloat1_Pseudo(gen);
  res.z = rndFloat1_Pseudo(gen);
  res.w = rndFloat1_Pseudo(gen);
  return res;
}


#endif // SIMPLE_RANDOM_GEN or COMPLEX


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float MutateKelemen(float valueX, __private RandomGen* pGen, const float p2)   // mutate in primary space
{
  const float2 rands = rndFloat2_Pseudo(pGen);

  const float s1 = 1.0f / 1024.0f, s2 = 1.0f / p2;
  const float dv = fmax(s2*( exp(-log(s2 / s1)*sqrt(rands.x)) - exp(-log(s2 / s1)) ), 0.0f);

  if (rands.y < 0.5f)
  {
    valueX += dv;
    if (valueX > 1.0f)
      valueX -= 1.0f;
  }
  else
  {
    valueX -= dv;
    if (valueX < 0.0f)
      valueX += 1.0f;
  }

  return valueX;
}


//#define MCMC_LAZY
//#define COMPRESS_RAND

#define MUTATE_LAZY_YES   1
#define MUTATE_LAZY_NO    0
#define MUTATE_LAZY_LARGE 2

#define MUTATE_COEFF_SCREEN 128.0f
#define MUTATE_COEFF_BSDF   64.0f

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define QRNG_DIMENSIONS 4 
#define QRNG_DIMENSIONS 7 // for test 
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)

static inline float rndQmcSobolN(unsigned int pos, int dim, __constant unsigned int *c_Table)
{
  unsigned int result = 0;
  unsigned int data   = pos;

  for (int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
    if (data & 1) result ^= c_Table[bit + dim*QRNG_RESOLUTION];

  return (float)(result + 1) * INT_SCALE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef struct LightSamplingRandsT // sampling from spetial formatted PSS vector
{
  float3 pos;
  float  numberAsFloat;

} LSRands;


typedef struct float6GroupT
{
  float4  group24;
  ushort2 group16;

} float6_gr;


#define MANTISSA24 8388608.0f // 16777216.0f actually you have to use only 23 bit because sign is not used, randoms from [0..1] are always positive ... 
#define MANTISSA24_INV (1.0f / MANTISSA24)

static inline uint4 packBounceGroup(float6_gr data)
{
  const unsigned int ix0 = (unsigned int)(MANTISSA24*data.group24.x);
  const unsigned int ix1 = (unsigned int)(MANTISSA24*data.group24.y);
  const unsigned int ix2 = (unsigned int)(MANTISSA24*data.group24.z);
  const unsigned int ix3 = (unsigned int)(MANTISSA24*data.group24.w);
  const unsigned int iy0 = (unsigned int)(data.group16.x);
  const unsigned int iy1 = (unsigned int)(data.group16.y);

  uint4 res;
  res.x = (ix0 & 0x00FFFFFF) | ((ix1 & 0x00FF0000) << 8);
  res.y = (ix1 & 0x0000FFFF) | ((iy0 & 0x0000FFFF) << 16);
  res.z = (ix2 & 0x00FFFFFF) | ((ix3 & 0x00FF0000) << 8);
  res.w = (ix3 & 0x0000FFFF) | ((iy1 & 0x0000FFFF) << 16);
  return res;
}

static inline float6_gr unpackBounceGroup(uint4 data)
{
  const unsigned int y0i = data.y >> 16;
  const unsigned int y1i = data.w >> 16;

  const unsigned int x0i = data.x & 0x00FFFFFF;
  const unsigned int x1i = ((data.x & 0xFF000000) >> 8) | (data.y & 0x0000FFFF);

  const unsigned int x2i = data.z & 0x00FFFFFF;
  const unsigned int x3i = ((data.z & 0xFF000000) >> 8) | (data.w & 0x0000FFFF);

  float6_gr res;
  res.group24.x = (float)(x0i)*MANTISSA24_INV;
  res.group24.y = (float)(x1i)*MANTISSA24_INV;
  res.group24.z = (float)(x2i)*MANTISSA24_INV;
  res.group24.w = (float)(x3i)*MANTISSA24_INV;
  res.group16.x = (ushort)(y0i);
  res.group16.y = (ushort)(y1i);
  return res;
}


#ifdef RAND_MLT_CPU
  #undef COMPRESS_RAND
#endif

#ifdef COMPRESS_RAND

#define MLT_INT4_PER_BOUNCE 2

static inline int rndLightOffset(const int a_bounceId) { return a_bounceId*MLT_INT4_PER_BOUNCE + 0; }
static inline int rndMatOffset  (const int a_bounceId) { return a_bounceId*MLT_INT4_PER_BOUNCE + 1; }
static inline int rndMaxBounce(const RandomGen* gen)   { return (gen->maxNumbers / (MLT_INT4_PER_BOUNCE*4)); }

static inline float4 rndLensOld(__global const float* rptr)
{
  __global const uint4* rptr2 = (__global const uint4*)rptr;
  return unpackBounceGroup(rptr2[0]).group24;
}

static inline float4 rndLens(RandomGen* gen, __global const float* rptr, const float2 screenScale)
{
  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE))
  {
    __global const uint4* rptr2 = (__global const uint4*)rptr;

    float4 rands = unpackBounceGroup(rptr2[0]).group24;

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      rands.x = MutateKelemen(rands.x, gen, MUTATE_COEFF_SCREEN*screenScale.x);
      rands.y = MutateKelemen(rands.y, gen, MUTATE_COEFF_SCREEN*screenScale.y);
      rands.z = MutateKelemen(rands.z, gen, MUTATE_COEFF_BSDF);
      rands.w = MutateKelemen(rands.w, gen, MUTATE_COEFF_BSDF);
    }

    return rands;
  }
  else
    return rndFloat4_Pseudo(gen);
}

static inline float4 rndLight(RandomGen* gen, __global const float* rptr, const int bounceId)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE) && bounceId < MLT_MAX_BOUNCE)
  {
    __global const uint4* rptr2 = (__global const uint4*)rptr;

    const int offset   = rndLightOffset(bounceId);
    float4 rands = unpackBounceGroup(rptr2[offset]).group24;

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      rands.x = MutateKelemen(rands.x, gen, MUTATE_COEFF_BSDF);
      rands.y = MutateKelemen(rands.y, gen, MUTATE_COEFF_BSDF);
      rands.z = MutateKelemen(rands.z, gen, MUTATE_COEFF_BSDF);
      rands.w = rands.w;  // don't mutate light number !!!
    }

    return rands;
  }
  else
    return rndFloat4_Pseudo(gen);
}


static inline float3 rndMat(RandomGen* gen, __global const float* rptr, const int bounceId)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE) && bounceId < MLT_MAX_BOUNCE)
  {
    __global const uint4* rptr2 = (__global const uint4*)rptr;

    const int offset   = rndMatOffset(bounceId);
    float4 rands = unpackBounceGroup(rptr2[offset]).group24;

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      rands.x = MutateKelemen(rands.x, gen, MUTATE_COEFF_BSDF);
      rands.y = MutateKelemen(rands.y, gen, MUTATE_COEFF_BSDF);
      rands.z = MutateKelemen(rands.z, gen, MUTATE_COEFF_BSDF);
    }

    return to_float3(rands);
  }
  else
    return to_float3(rndFloat4_Pseudo(gen));
}

static inline float rndMatLayer(RandomGen* gen, __global const float* rptr, const int bounceId, const int layerId)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE) && bounceId < MLT_MAX_BOUNCE)
  {
    __global const uint4* rptr2 = (__global const uint4*)rptr;

    if(layerId <= 2)
    {
      const float6_gr group2 = unpackBounceGroup(rptr2[rndMatOffset(bounceId)]);
      if(layerId == 0)
        return group2.group24.w;
      else if(layerId == 1)
        return (float)(group2.group16.x)*(1.0f / 65535.0f);
      else
        return (float)(group2.group16.y)*(1.0f / 65535.0f);
    }
    else if(layerId <= 4)
    {
      const ushort2 group1 = unpackBounceGroup(rptr2[rndLightOffset(bounceId)]).group16;
      if(layerId == 3)
        return (float)(group1.x)*(1.0f / 65535.0f);
      else 
        return (float)(group1.y)*(1.0f / 65535.0f);
    }
    else
      return rndFloat1_Pseudo(gen); //rptr[rndMatLOffset(bounceId) + layerId];
  }
  else
  {
    return rndFloat1_Pseudo(gen);
  }
}

#define MLT_FLOATS_PER_MLAYER 5

#else // not compressed layout

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MLT_FLOATS_PER_SAMPLE (4+3)
#define MLT_FLOATS_PER_MLAYER 5

#define MLT_FLOATS_PER_BOUNCE (MLT_FLOATS_PER_SAMPLE + MLT_FLOATS_PER_MLAYER)

#define MLT_LIGHT_GROUP_LOCAL_OFFS 0
#define MLT_MAT_GROUP_LOCAL_OFFS   4

static inline int rndLightOffset(const int a_bounceId) { return a_bounceId*MLT_FLOATS_PER_BOUNCE + MLT_LIGHT_GROUP_LOCAL_OFFS; }
static inline int rndMatOffset  (const int a_bounceId) { return a_bounceId*MLT_FLOATS_PER_BOUNCE + MLT_MAT_GROUP_LOCAL_OFFS; }
static inline int rndMatLOffset (const int a_bounceId) { return a_bounceId*MLT_FLOATS_PER_BOUNCE + MLT_FLOATS_PER_SAMPLE; }
static inline int rndMaxBounce  (const RandomGen* gen) { return (gen->maxNumbers / MLT_FLOATS_PER_BOUNCE); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float4 rndLensOld(__global const float* rptr)
{
  return make_float4(rptr[0], rptr[1], rptr[2], rptr[3]);
}

static inline float4 rndLens(RandomGen* gen, __global const float* rptr, const float2 screenScale, __constant unsigned int* a_qmcTable, const unsigned int qmcPos)
{
  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE))
  {
    float x = rptr[0];
    float y = rptr[1];
    float z = rptr[2];
    float w = rptr[3];

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      x = MutateKelemen(x, gen, MUTATE_COEFF_SCREEN*screenScale.x);
      y = MutateKelemen(y, gen, MUTATE_COEFF_SCREEN*screenScale.y);
      z = MutateKelemen(z, gen, MUTATE_COEFF_BSDF);
      w = MutateKelemen(w, gen, MUTATE_COEFF_BSDF);
    }

    return make_float4(x, y, z, w);
  }
  else
  {
    if (a_qmcTable != 0)
    {
      float4 lensOffs;
      lensOffs.x = rndQmcSobolN(qmcPos, 0, a_qmcTable);
      lensOffs.y = rndQmcSobolN(qmcPos, 1, a_qmcTable);
      lensOffs.z = rndQmcSobolN(qmcPos, 2, a_qmcTable);
      lensOffs.w = rndQmcSobolN(qmcPos, 3, a_qmcTable);
      return lensOffs;
    }
    else
      return rndFloat4_Pseudo(gen);
  }
}

static inline float4 rndLight(RandomGen* gen, __global const float* rptr, const int bounceId)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE) && bounceId < MLT_MAX_BOUNCE)
  {
    const int offset = rndLightOffset(bounceId);

    float x = rptr[offset + 0];
    float y = rptr[offset + 1];
    float z = rptr[offset + 2];
    float w = rptr[offset + 3];

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      x = MutateKelemen(x, gen, MUTATE_COEFF_BSDF);
      y = MutateKelemen(y, gen, MUTATE_COEFF_BSDF);
      z = MutateKelemen(z, gen, MUTATE_COEFF_BSDF);
      //w = MutateKelemen(w, gen, MUTATE_COEFF_BSDF);  // don't mutate light number
    }

    return make_float4(x, y, z, w);
  }
  else
    return rndFloat4_Pseudo(gen);
}

static inline float3 rndMat(RandomGen* gen, __global const float* rptr, const int bounceId)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE) && bounceId < MLT_MAX_BOUNCE)
  {
    const int offset = rndMatOffset(bounceId);

    float x = rptr[offset + 0];
    float y = rptr[offset + 1];
    float z = rptr[offset + 2];

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      x = MutateKelemen(x, gen, MUTATE_COEFF_BSDF);
      y = MutateKelemen(y, gen, MUTATE_COEFF_BSDF);
      z = MutateKelemen(z, gen, MUTATE_COEFF_BSDF);
    }

    return make_float3(x, y, z);
  }
  else
    return to_float3(rndFloat4_Pseudo(gen));
}

static inline float rndMatLayer(RandomGen* gen, __global const float* rptr, const int bounceId, const int layerId)
{
  const int MLT_MAX_BOUNCE = rndMaxBounce(gen);

  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE) && bounceId < MLT_MAX_BOUNCE)
    return rptr[rndMatLOffset(bounceId) + layerId]; 
  else
    return rndFloat1_Pseudo(gen);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif

static inline unsigned int mapRndFloatToUInt(float a_val, unsigned int a, unsigned int b)
{
  const float fa = (float)a;
  const float fb = (float)b;
  const float fR = fa + a_val * (fb - fa);

  const unsigned int res = (unsigned int)(fR);

  if (res > b - 1)
    return b - 1;
  else
    return res;
}

// (3+5)*(d-1) for materials

#define MMLT_FLOATS_PER_SAMPLE 3
#define MMLT_FLOATS_PER_BOUNCE (MMLT_FLOATS_PER_SAMPLE + MLT_FLOATS_PER_MLAYER)
#define MMLT_HEAD_TOTAL_SIZE 16  // (4+8+1) used + 3 unused

// [0-3]:   LENS;  4 in total
// [4-11]:  LIGHT; 5 in total
// [12  ]:  SPLIT; 1 in total
// [13-15]: TAIL;  3 in total, not used

static inline int rndMatOffsetMMLT (const int a_bounceId) { return a_bounceId*MMLT_FLOATS_PER_BOUNCE; }                          // relative offset, dont add MMLT_HEAD_TOTAL_SIZE!
static inline int rndMatLOffsetMMLT(const int a_bounceId) { return a_bounceId*MMLT_FLOATS_PER_BOUNCE + MMLT_FLOATS_PER_SAMPLE; } // relative offset, dont add MMLT_HEAD_TOTAL_SIZE!

static inline float3 rndMatMMLT(RandomGen* gen, __global const float* rptr, const int bounceId)
{
  if (rptr != 0 && gen->lazy != MUTATE_LAZY_LARGE)
  {
    const int offset = rndMatOffsetMMLT(bounceId);

    float x = rptr[offset + 0];
    float y = rptr[offset + 1];
    float z = rptr[offset + 2];

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      x = MutateKelemen(x, gen, MUTATE_COEFF_BSDF);
      y = MutateKelemen(y, gen, MUTATE_COEFF_BSDF);
      z = MutateKelemen(z, gen, MUTATE_COEFF_BSDF);
    }

    return make_float3(x, y, z);
  }
  else
    return to_float3(rndFloat4_Pseudo(gen));
}

static inline float rndMatLayerMMLT(RandomGen* gen, __global const float* rptr, const int bounceId, const int layerId)
{
  if (rptr != 0 && gen->lazy != MUTATE_LAZY_LARGE)
    return rptr[rndMatLOffsetMMLT(bounceId) + layerId];
  else
    return rndFloat1_Pseudo(gen);
}

typedef struct LightGroup2T
{
  float4 group1;
  float4 group2;

} LightGroup2;


static inline void RndLightMMLT(RandomGen* gen, __global const float* rptr,
                                __private LightGroup2* pOut)
{
  if (rptr != 0 && gen->lazy != MUTATE_LAZY_LARGE)
  {
    float x  = rptr[4 + 0]; //#TODO: opt read as 2xfloat4
    float y  = rptr[4 + 1];
    float z  = rptr[4 + 2];
    float w  = rptr[4 + 3];
  
    float x1 = rptr[4 + 4];
    float y1 = rptr[4 + 5];
    float n  = rptr[4 + 6];
    float n1 = rptr[4 + 7];
     
    if (gen->lazy == MUTATE_LAZY_YES)
    {
      x  = MutateKelemen(x,  gen, MUTATE_COEFF_BSDF);
      y  = MutateKelemen(y,  gen, MUTATE_COEFF_BSDF);
      z  = MutateKelemen(z,  gen, MUTATE_COEFF_BSDF);
      w  = MutateKelemen(w,  gen, MUTATE_COEFF_BSDF); 
      x1 = MutateKelemen(x1, gen, MUTATE_COEFF_BSDF);
      y1 = MutateKelemen(y1, gen, MUTATE_COEFF_BSDF);
      //n = n, n1=n1; // don't mutate light numbers
    }

    pOut->group1 = make_float4(x, y, z, w);
    pOut->group2 = make_float4(x1,y1,n, n1);
  }
  else
  {
    pOut->group1 = rndFloat4_Pseudo(gen);
    pOut->group2 = rndFloat4_Pseudo(gen);
  }
}


static inline int rndSplitMMLT(RandomGen* gen, __global const float* rptr, const int d)
{
  const int offset = 12; 
  
  float x;
  if (rptr != 0 && gen->lazy != MUTATE_LAZY_LARGE)
  {
    x = rptr[offset];
    if (gen->lazy == MUTATE_LAZY_YES)
      x = MutateKelemen(x, gen, MUTATE_COEFF_BSDF);
  }
  else
    x = rndFloat1_Pseudo(gen);

  return mapRndFloatToUInt(x, 0, d+1);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifdef RAND_MLT_CPU // FOR CPU 

static inline float4 rndFloat4(RandomGen* gen)
{
  return rndFloat4_Pseudo(gen);
}

static inline float3 rndFloat3(RandomGen* gen)
{
  return to_float3(rndFloat4_Pseudo(gen));
}

static inline float2 rndFloat2(RandomGen* gen)
{
  const float4 xyzw = rndFloat4_Pseudo(gen);
  return make_float2(xyzw.x, xyzw.y);
}

static inline float rndFloat1(RandomGen* gen)
{
  return rndFloat1_Pseudo(gen);
}

static inline unsigned int rndIntFromFloat(float r, unsigned int a, unsigned int b)
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

static inline unsigned int rndInt(RandomGen* gen, unsigned int a, unsigned int b)
{
  return rndIntFromFloat(rndFloat1_Pseudo(gen), a, b);
}

#endif // RAND_MLT_CPU

static inline float4 rndUniform(RandomGen* gen, float a, float b)
{
  return make_float4(a, a, a, a) + (b - a)*rndFloat4_Pseudo(gen);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static inline int randArraySizeOfDepthMMLT(int d)
{
  return MMLT_HEAD_TOTAL_SIZE + MMLT_FLOATS_PER_BOUNCE * d;
}

static inline int camOffsetInRandArrayMMLT(int s)
{
  return  MMLT_HEAD_TOTAL_SIZE + MMLT_FLOATS_PER_BOUNCE * s;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif


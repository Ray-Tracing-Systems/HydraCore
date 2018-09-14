#ifndef RTCRANDOM
#define RTCRANDOM

#include "cglobals.h"

#define SIMPLE_RANDOM_GEN

#ifdef SIMPLE_RANDOM_GEN

typedef struct RandomGenT
{
  uint2          state;
  unsigned int   maxNumbers;
  unsigned int   lazy; // or dummy to have uint4 generator data.

#ifdef RAND_MLT_CPU
  __global const float* rptr;
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

#else

typedef struct RandomGenT
{
  unsigned int x[5];
  unsigned int maxNumbers;
  unsigned int lazy; 
  unsigned int dummy;  // to have uint4 generator data.

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

static inline float2 rndFloat2_Pseudo(RandomGen* gen)
{
  float2 res;
  res.x = rndFloat1_Pseudo(gen);
  res.y = rndFloat1_Pseudo(gen);
  return res;
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

#define QRNG_DIMENSIONS 11
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



/**
\brief get qmc number for target qmc var (see defines up)
\param pGen      - inout pseudo random generator
\param a_tab     - remap table that store numbers for target defines; i.e. a_tab[QMC_VAR_SCREEN_X]
\param pos       - id of qmc number
\param pickProb  - a_varName - name of qmc number
\param c_Table   - qmc table for sobol-neideriter
\return quasi random float in range [0,1]

*/
static inline float rndQmcTab(__private RandomGen* pGen, __global const int* a_tab,
                              unsigned int pos, int a_varName, __constant unsigned int *c_Table) // pre (a_tab != nullptr && c_Table != nullptr)
{
  const int dim = a_tab[a_varName];
  
  if(dim < 0)
    return rndFloat1_Pseudo(pGen);
  else
    return rndQmcSobolN(pos, dim, c_Table);
}


static inline int rndMatOffsetMMLT(const int a_bounceId) { return a_bounceId*MMLT_FLOATS_PER_BOUNCE; }                          // relative offset, dont add MMLT_HEAD_TOTAL_SIZE!

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct float6GroupT
{
  float4 group24;
  float2 group16;
} float6_gr;


#define MANTISSA24 16777215.0f // 16777215.0f 
#define MANTISSA24_INV (1.0f / MANTISSA24)

#define MANTISSA16 65535.0f
#define MANTISSA16_INV (1.0f/65535.0f)

static inline uint4 packBounceGroup(float6_gr data)
{
  const unsigned int ix0 = (unsigned int)(MANTISSA24*data.group24.x);
  const unsigned int ix1 = (unsigned int)(MANTISSA24*data.group24.y);
  const unsigned int ix2 = (unsigned int)(MANTISSA24*data.group24.z);
  const unsigned int ix3 = (unsigned int)(MANTISSA24*data.group24.w);
  const unsigned int iy0 = (unsigned int)(MANTISSA16*data.group16.x);
  const unsigned int iy1 = (unsigned int)(MANTISSA16*data.group16.y);

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
  res.group16.x = (float)(y0i)*MANTISSA16_INV;
  res.group16.y = (float)(y1i)*MANTISSA16_INV;
  return res;
}

static inline uint2 packBounceGroup2(float4 data)
{
  const unsigned int iy0 = (unsigned int)(MANTISSA16*data.x);
  const unsigned int iy1 = (unsigned int)(MANTISSA16*data.y);
  const unsigned int iy2 = (unsigned int)(MANTISSA16*data.z);
  const unsigned int iy3 = (unsigned int)(MANTISSA16*data.w);

  uint2 res;
  res.x = iy0 | (iy1 << 16);
  res.y = iy2 | (iy3 << 16);
  return res;
}

static inline float4 unpackBounceGroup2(uint2 data)
{
  const unsigned int y0i = (data.x & 0x0000FFFF);
  const unsigned int y1i = (data.x & 0xFFFF0000) >> 16;
  const unsigned int y2i = (data.y & 0x0000FFFF);
  const unsigned int y3i = (data.y & 0xFFFF0000) >> 16;

  float4 res;
  res.x = (float)(y0i)*MANTISSA16_INV;
  res.y = (float)(y1i)*MANTISSA16_INV;
  res.z = (float)(y2i)*MANTISSA16_INV;
  res.w = (float)(y3i)*MANTISSA16_INV;
  return res;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
\brief obtain old (i.e. not mutated random numbers) to get old contribution sample (x,y)
\param rptr - in random number storage pointer
 
\return 4 old random numbers for using them in camera\lens sampler
*/
static inline float4 rndLensOld(__global const float* rptr)
{
  return make_float4(rptr[MMLT_DIM_SCR_X], rptr[MMLT_DIM_SCR_Y], rptr[MMLT_DIM_DOF_X], rptr[MMLT_DIM_DOF_Y]);
}

/**
\brief obtain 4 random numbers for using them in camera\lens sampler
\param gen         - in out random generator.
\param rptr        - in random number storage pointer
\param screenScale - in scale for screen space muations (used to ajust mutation size according to specific resolution)
\param a_qmcTable  - in qmc sobol (or else) table for permutations or sms like that;
\param a_qmcPos    - in qmc poition index (i-th qmc number from generator)
\param a_tab       - in qmc remap table; 

\return 4 random numbers for using them in camera\lens sampler
*/
static inline float4 rndLens(RandomGen* gen, __global const float* rptr, const float2 screenScale,
                             __global const int* a_tab, const unsigned int a_qmcPos, __constant unsigned int* a_qmcTable)
{
  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE))
  {
    float x = rptr[MMLT_DIM_SCR_X];
    float y = rptr[MMLT_DIM_SCR_Y];
    float z = rptr[MMLT_DIM_DOF_X];
    float w = rptr[MMLT_DIM_DOF_Y];

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      x = MutateKelemen(x, gen, MUTATE_COEFF_SCREEN*screenScale.x);
      y = MutateKelemen(y, gen, MUTATE_COEFF_SCREEN*screenScale.y);
      z = MutateKelemen(z, gen, MUTATE_COEFF_BSDF);
      w = MutateKelemen(w, gen, MUTATE_COEFF_BSDF);
    }

    return make_float4(x, y, z, w);
  }
  else if (a_qmcTable != 0)
  {
    float4 lensOffs;
    lensOffs.y = rndQmcTab(gen, a_tab, a_qmcPos, QMC_VAR_SCR_Y, a_qmcTable);
    lensOffs.z = rndQmcTab(gen, a_tab, a_qmcPos, QMC_VAR_DOF_X, a_qmcTable);
    lensOffs.w = rndQmcTab(gen, a_tab, a_qmcPos, QMC_VAR_DOF_Y, a_qmcTable);
    lensOffs.x = rndQmcTab(gen, a_tab, a_qmcPos, QMC_VAR_SCR_X, a_qmcTable);
    return lensOffs;
  }
  else
    return rndFloat4_Pseudo(gen);
}

/**
\brief obtain 4 random numbers for light sampling on the target bounce during to Path Tracing only (reverse light path)
\param gen         - in out random generator.
\param bounceId    - in target bhounce number (0 is the primary visiable surface)
\param a_qmcTable  - in qmc sobol (or else) table for permutations or sms like that;
\param a_qmcPos    - in qmc poition index (i-th qmc number from generator)
\param a_tab       - in qmc remap table; 

\return 4 random numbers for using them in light sampling
*/
static inline float4 rndLight(RandomGen* gen, const int bounceId,
                              __global const int* a_tab, const unsigned int qmcPos, __constant unsigned int* a_qmcTable)
{
  if(bounceId == 0 && a_tab != 0 && a_qmcTable != 0)
  {
    float4 res;
    res.x = rndQmcTab(gen, a_tab, qmcPos, QMC_VAR_LGT_0, a_qmcTable);
    res.y = rndQmcTab(gen, a_tab, qmcPos, QMC_VAR_LGT_1, a_qmcTable);
    res.z = rndQmcTab(gen, a_tab, qmcPos, QMC_VAR_LGT_2, a_qmcTable);
    res.w = rndFloat1_Pseudo(gen);
    return res;
  }
  else
    return rndFloat4_Pseudo(gen);
}

/**
\brief obtain 3 random numbers for material sampling on the target bounce (assume reverse order, i.e. 0 is the first bounce from camera)
\param gen         - in out random generator.
\param rptr        - in random number storage pointer
\param bounceId    - in target bhounce number (0 is the primary visiable surface)
\param a_qmcTable  - in qmc sobol (or else) table for permutations or sms like that;
\param a_qmcPos    - in qmc poition index (i-th qmc number from generator)
\param a_tab       - in qmc remap table; 

\return 3 random numbers for using them in material sampling
*/
static inline float3 rndMat(RandomGen* gen, __global const float* rptr, const int bounceId,
                            __global const int* a_tab, const unsigned int qmcPos, __constant unsigned int* a_qmcTable)
{
  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE))
  {
    float x = rptr[0];
    float y = rptr[1];
    float z = rptr[2];

    if (gen->lazy == MUTATE_LAZY_YES)
    {
      x = MutateKelemen(x, gen, MUTATE_COEFF_BSDF);
      y = MutateKelemen(y, gen, MUTATE_COEFF_BSDF);
      z = MutateKelemen(z, gen, MUTATE_COEFF_BSDF);
    }

    return make_float3(x, y, z);
  }
  else if(bounceId == 0 && a_tab != 0 && a_qmcTable != 0)
  {
    float3 res;
    res.x = rndQmcTab(gen, a_tab, qmcPos, QMC_VAR_MAT_0, a_qmcTable);
    res.y = rndQmcTab(gen, a_tab, qmcPos, QMC_VAR_MAT_1, a_qmcTable);
    res.z = rndFloat1_Pseudo(gen);
    return res;
  }
  else
    return to_float3(rndFloat4_Pseudo(gen));
}

/**
\brief obtain 1 random numbers for material layer selection on the target bounce and target three depth (assume reverse order, i.e. 0 is the first bounce from camera)
\param gen         - in out random generator.
\param rptr        - in random number storage pointer
\param bounceId    - in target bhounce number (0 is the primary visiable surface)
\param layerId     - in material tree depth

\param a_qmcTable  - in qmc sobol (or else) table for permutations or sms like that;
\param a_qmcPos    - in qmc poition index (i-th qmc number from generator)
\param a_tab       - in qmc remap table; 

\return 3 random numbers for using them in material sampling
*/
static inline float rndMatLayer(RandomGen* gen, __global const float* rptr, const int bounceId, const int layerId,
                                __global const int* a_tab, const unsigned int a_qmcPos, __constant unsigned int* a_qmcTable)
{
  if (rptr != 0 && (gen->lazy != MUTATE_LAZY_LARGE))                   // MCMC way; #NOTE: Lazy mutations is not needed due to small step
    return rptr[layerId];                                              // must never change material layer, no mutations is allowed!
  else if(bounceId == 0 && a_tab != 0 && a_qmcTable != 0)              // QMC way;
    return rndQmcTab(gen, a_tab, a_qmcPos, QMC_VAR_MAT_L, a_qmcTable);
  else                                                                 // OMC way;
    return rndFloat1_Pseudo(gen);                                      
}


static inline void RndMatAll(RandomGen* gen, __global const float* rptr, const int bounceId,
                             __global const int* a_tab, const unsigned int a_qmcPos, __constant unsigned int* a_qmcTable, 
                             __private float a_rands[MMLT_FLOATS_PER_BOUNCE])
{

  const float3 directionRands = rndMat(gen, rptr, bounceId,  a_tab, a_qmcPos, a_qmcTable);
  
  a_rands[0] = directionRands.x;
  a_rands[1] = directionRands.y;
  a_rands[2] = directionRands.z;

  __global const float* rptrLayer = (rptr == 0) ? 0 : rptr + MMLT_FLOATS_PER_SAMPLE;

  #pragma unroll MMLT_FLOATS_PER_MLAYER
  for(int layerId=0;layerId<MMLT_FLOATS_PER_MLAYER;layerId++)
    a_rands[MMLT_FLOATS_PER_SAMPLE + layerId] = rndMatLayer(gen, rptrLayer, bounceId, layerId, a_tab, a_qmcPos, a_qmcTable);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



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

typedef struct LightGroup2T
{
  float4 group1;
  float3 group2;

} LightGroup2;

static inline void RndLightMMLT(RandomGen* gen, __global const float* rptr,
                                __private LightGroup2* pOut)
{
  if (rptr != 0 && gen->lazy != MUTATE_LAZY_LARGE)
  {
    float x  = rptr[MMLT_DIM_LGT_X]; //#TODO: opt read as 2xfloat4
    float y  = rptr[MMLT_DIM_LGT_Y];
    float z  = rptr[MMLT_DIM_LGT_Z];
    float w  = rptr[MMLT_DIM_LGT_W];
  
    float x1 = rptr[MMLT_DIM_LGT_X1];
    float y1 = rptr[MMLT_DIM_LGT_Y1];
    float n  = rptr[MMLT_DIM_LGT_N];
     
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
    pOut->group2 = make_float3(x1, y1, n);
  }
  else
  {
    pOut->group1 = rndFloat4_Pseudo(gen);
    pOut->group2 = to_float3(rndFloat4_Pseudo(gen));
  }
}

static inline int rndSplitMMLT(RandomGen* gen, __global const float* rptr, const int d)
{
  float x;
  if (rptr != 0 && gen->lazy != MUTATE_LAZY_LARGE)
  {
    x = rptr[MMLT_DIM_SPLIT];
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

/**
\brief get total size (in floats) of the array
*/
static inline int randArraySizeOfDepthMMLT(int d)
{
  return MMLT_HEAD_TOTAL_SIZE + MMLT_FLOATS_PER_BOUNCE * d;
}

/**
\brief get offset (in floats) to camera-path part of the array
*/
static inline int camOffsetInRandArrayMMLT(int s)
{
  return  MMLT_HEAD_TOTAL_SIZE + MMLT_FLOATS_PER_BOUNCE * s;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif


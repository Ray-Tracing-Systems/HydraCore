#ifndef RTCMPBRT
#define RTCMPBRT

#include "cglobals.h"
#include "cfetch.h"

static inline float CosPhiPBRT(const float3 w, const float sintheta)
{
  if (sintheta == 0.f) 
    return 1.f;
  else
    return clamp(w.x / sintheta, -1.f, 1.f);
}

static inline float SinPhiPBRT(const float3 w, const float sintheta)
{
  if (sintheta == 0.f)
    return 0.f;
  else
    return clamp(w.y / sintheta, -1.f, 1.f);
}

static inline float3 SphericalDirectionPBRT(const float sintheta, const float costheta, const float phi) 
{ 
  return make_float3(sintheta * cos(phi), sintheta * sin(phi), costheta); 
}

static inline bool SameHemispherePBRT(float3 w, float3 wp) { return w.z * wp.z > 0.f; }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline float CosThetaPBRT   (float3 w) { return w.z; }
inline float Cos2ThetaPBRT  (float3 w) { return w.z * w.z; }
inline float AbsCosThetaPBRT(float3 w) { return fabs(w.z); }
inline float Sin2ThetaPBRT  (float3 w) { return fmax(0.0f, 1.0f - Cos2ThetaPBRT(w)); }

inline float SinThetaPBRT (float3 w) { return sqrt(Sin2ThetaPBRT(w)); }
inline float TanThetaPBRT (float3 w) { return SinThetaPBRT(w)  / CosThetaPBRT(w); }
inline float Tan2ThetaPBRT(float3 w) { return Sin2ThetaPBRT(w) / Cos2ThetaPBRT(w); }

inline float CosPhiPBRT(float3 w) 
{
    const float sinTheta = SinThetaPBRT(w);
    return (sinTheta == 0.0f) ? 1.0f : clamp(w.x / sinTheta, -1.0f, 1.0f);
}

inline float SinPhiPBRT(float3 w) 
{
    const float sinTheta = SinThetaPBRT(w);
    return (sinTheta == 0.0f) ? 0.0f : clamp(w.y / sinTheta, -1.0f, 1.0f);
}

inline float Cos2PhiPBRT(float3 w) { return CosPhiPBRT(w) * CosPhiPBRT(w); }
inline float Sin2PhiPBRT(float3 w) { return SinPhiPBRT(w) * SinPhiPBRT(w); }

static inline float BeckmannDistributionD(const float3 wh, float alphax, float alphay)
{
  float tan2Theta = Tan2ThetaPBRT(wh);
  if (!isfinite(tan2Theta)) 
    return 0.0f;
  float cos4Theta = Cos2ThetaPBRT(wh) * Cos2ThetaPBRT(wh);
  return std::exp(-tan2Theta * (Cos2PhiPBRT(wh) / (alphax * alphax) +
                                Sin2PhiPBRT(wh) / (alphay * alphay))) / (M_PI * alphax * alphay * cos4Theta);
}

static inline float BeckmannDistributionLambda(const float3 w, float alphax, float alphay)
{
  const float absTanTheta = fabs(TanThetaPBRT(w));
  if (!isfinite(absTanTheta)) 
    return 0.0f;
  // Compute _alpha_ for direction _w_
  const float alpha = sqrt(Cos2PhiPBRT(w) * alphax * alphax + Sin2PhiPBRT(w) * alphay * alphay);
  const float a = 1.0f / (alpha * absTanTheta);
  if (a >= 1.6f) 
    return 0.0f;
  return (1.0f - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float TrowbridgeReitzDistributionD(const float3 wh, float alphax, float alphay) 
{
  float tan2Theta = Tan2ThetaPBRT(wh);
  if (!isfinite(tan2Theta)) 
    return 0.0f;
  const float cos4Theta = Cos2ThetaPBRT(wh) * Cos2ThetaPBRT(wh);
  const float e = (Cos2PhiPBRT(wh) / (alphax * alphax) + Sin2PhiPBRT(wh) / (alphay * alphay)) * tan2Theta;
  return 1.0f / (M_PI * alphax * alphay * cos4Theta * (1.0f + e) * (1.0f + e));
}


static inline float TrowbridgeReitzDistributionLambda(const float3 w, float alphax, float alphay)
{
  const float absTanTheta = fabs(TanThetaPBRT(w));
  if (!isfinite(absTanTheta)) 
    return 0.0f;
  // Compute _alpha_ for direction _w_
  const float alpha           = sqrt(Cos2PhiPBRT(w) * alphax * alphax + Sin2PhiPBRT(w) * alphay * alphay);
  const float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
  return (-1.0f + sqrt(1.f + alpha2Tan2Theta)) / 2.0f;
}

#endif

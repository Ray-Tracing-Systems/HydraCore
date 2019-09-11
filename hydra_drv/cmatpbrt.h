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

#endif

#ifndef RTCMPBRT
#define RTCMPBRT

#include "cglobals.h"
#include "cfetch.h"

static inline float FrCond(const float cosi, const float eta, const float k)
{
  const float tmp    = (eta*eta + k*k) * cosi*cosi;
  const float Rparl2 = (tmp - (2.f * eta * cosi) + 1.0f) / (tmp + (2.f * eta * cosi) + 1.0f);
  const float tmp_f  = eta*eta + k*k;
  const float Rperp2 = (tmp_f - (2.f * eta * cosi) + cosi*cosi) / (tmp_f + (2.f * eta * cosi) + cosi*cosi);
  return fabs(Rparl2 + Rperp2) / 2.f;
}

static inline float CosPhiPBRT1(const float3 w, const float sintheta)
{
  if (sintheta == 0.f) 
    return 1.f;
  else
    return clamp(w.x / sintheta, -1.f, 1.f);
}

static inline float SinPhiPBRT1(const float3 w, const float sintheta)
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

inline float ErfPBRT(float x) 
{
  // constants
  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027f;
  const float a5 = 1.061405429f;
  const float p = 0.3275911f;
  // Save the sign of x
  int sign = 1;
  if (x < 0) 
    sign = -1;
  x = fabs(x);
  // A&S formula 7.1.26
  const float t = 1 / (1 + p * x);
  const float y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  return sign * y;
}

static inline float ErfInvPBRT(float x) 
{
  float w, p;
  x = clamp(x, -0.99999f, 0.99999f);
  w = -log((1.0f - x) * (1.0f + x));
  if (w < 5.0f) 
  {
    w = w - 2.5f;
    p = 2.81022636e-08f;
    p = 3.43273939e-07f + p * w;
    p = -3.5233877e-06f + p * w;
    p = -4.39150654e-06f + p * w;
    p = 0.00021858087f + p * w;
    p = -0.00125372503f + p * w;
    p = -0.00417768164f + p * w;
    p = 0.246640727f + p * w;
    p = 1.50140941f + p * w;
  } 
  else 
  {
    w = sqrt(w) - 3;
    p = -0.000200214257f;
    p = 0.000100950558f + p * w;
    p = 0.00134934322f + p * w;
    p = -0.00367342844f + p * w;
    p = 0.00573950773f + p * w;
    p = -0.0076224613f + p * w;
    p = 0.00943887047f + p * w;
    p = 1.00167406f + p * w;
    p = 2.83297682f + p * w;
  }
  return p * x;
}

static inline float BeckmannDistributionD(const float3 wh, float alphax, float alphay)
{
  float tan2Theta = Tan2ThetaPBRT(wh);
  if (!isfinite(tan2Theta)) 
    return 0.0f;
  float cos4Theta = Cos2ThetaPBRT(wh) * Cos2ThetaPBRT(wh);
  return exp(-tan2Theta * (Cos2PhiPBRT(wh) / (alphax * alphax) + Sin2PhiPBRT(wh) / (alphay * alphay))) / (M_PI * alphax * alphay * cos4Theta);
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

static inline void BeckmannSample11(float cosThetaI, float U1, float U2,
                                    float *slope_x, float *slope_y) 
{
  /* Special case (normal incidence) */
  if (cosThetaI > 0.9999f) 
  {
    const float r      = sqrt(-log(1.0f - U1));
    const float sinPhi = sin(2.0f * M_PI * U2);
    const float cosPhi = cos(2.0f * M_PI * U2);
    *slope_x = r * cosPhi;
    *slope_y = r * sinPhi;
    return;
  }
   
  // The original inversion routine from the paper contained
  // discontinuities, which causes issues for QMC integration
  // and techniques like Kelemen-style MLT. The following code
  // performs a numerical inversion with better behavior 
  //
  const float sinThetaI = sqrt(fmax(0.0f, 1.0f - cosThetaI * cosThetaI));
  const float tanThetaI = sinThetaI / cosThetaI;
  const float cotThetaI = 1.0f / tanThetaI;
  
  // Search interval -- everything is parameterized
  // in the Erf() domain 
  //
  float a = -1, c = ErfPBRT(cotThetaI);
  const float sample_x = max(U1, 1e-6f);

  // Start with a good initial guess 
  // Float b = (1-sample_x) * a + sample_x * c;
  // We can do better (inverse of an approximation computed in Mathematica)
  // 
  const float thetaI = acos(cosThetaI);
  const float fit    = 1 + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
  float b            = c - (1 + c) * pow(1 - sample_x, fit);

  // Normalization factor for the CDF 
  //
  const float SQRT_PI_INV = 1.f / sqrt(M_PI);
  float normalization = 1.0f / (1.0f + c + SQRT_PI_INV * tanThetaI * exp(-cotThetaI * cotThetaI));

  int it = 0;
  while (++it < 10) 
  {
    // Bisection criterion -- the oddly-looking
    // Boolean expression are intentional to check
    // for NaNs at little additional cost 
    //
    if (!(b >= a && b <= c)) 
      b = 0.5f * (a + c);
    // Evaluate the CDF and its derivative
    //   (i.e. the density function) 
    const float invErf     = ErfInvPBRT(b);
    const float value      = normalization * (1.0f + b + SQRT_PI_INV * tanThetaI * exp(-invErf * invErf)) - sample_x;
    const float derivative = normalization * (1 - invErf * tanThetaI);
    if (fabs(value) < 1e-5f) 
      break;
    /* Update bisection intervals */
    if (value > 0)
      c = b;
    else
      a = b;
    b -= value / derivative;
  }

  // Now convert back into a slope value 
  *slope_x = ErfInvPBRT(b);
  // Simulate Y component 
  *slope_y = ErfInvPBRT(2.0f * fmax(U2, 1e-6f) - 1.0f);
  
  //CHECK(!std::isinf(*slope_x));
  //CHECK(!std::isnan(*slope_x));
  //CHECK(!std::isinf(*slope_y));
  //CHECK(!std::isnan(*slope_y));
}

static inline float3 BeckmannSample(const float3 wi, float alpha_x, float alpha_y, float U1, float U2) 
{
  // 1. stretch wi
  const float3 wiStretched = normalize(make_float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

  // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
  float slope_x, slope_y;
  BeckmannSample11(CosThetaPBRT(wiStretched), U1, U2, 
                   &slope_x, &slope_y);

  // 3. rotate
  float tmp = CosPhiPBRT(wiStretched) * slope_x - SinPhiPBRT(wiStretched) * slope_y;
  slope_y   = SinPhiPBRT(wiStretched) * slope_x + CosPhiPBRT(wiStretched) * slope_y;
  slope_x   = tmp;

  // 4. unstretch
  slope_x = alpha_x * slope_x;
  slope_y = alpha_y * slope_y;
  
  // 5. compute normal
  return normalize(make_float3(-slope_x, -slope_y, 1.f));
}

static inline float3 BeckmannDistributionSampleWH(const float3 wo, const float2 u, float alphax, float alphay)
{
  float3 wh;
  const bool flip = (wo.z < 0.0f);
  wh = BeckmannSample(flip ? (-1.0f)*wo : wo, alphax, alphay, u.x, u.y);
  if (flip) 
    wh = (-1.0f)*wh;
  return wh;
}

static inline float BeckmannG1(float3 w, float alphax, float alphay) 
{
  //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
  return 1.0f / (1.0f + BeckmannDistributionLambda(w, alphax, alphay));
}

static inline float BeckmannDistributionPdf(float3 wo, float3 wh, float alphax, float alphay) 
{
  return BeckmannDistributionD(wh, alphax, alphay) * BeckmannG1(wo, alphax, alphay) * fabs(dot(wo, wh)) / AbsCosThetaPBRT(wo);
}

static inline float BeckmannRoughnessToAlpha(float roughness) 
{
  const float x = log(fmax(roughness, 1.0e-3f));
  return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

static inline float BeckmannG(const float3 wo, const float3 wi, float alphax, float alphay)
{
  return 1.0f / (1.0f + BeckmannDistributionLambda(wo, alphax, alphay) + BeckmannDistributionLambda(wi, alphax, alphay));
}

static inline float BeckmannBRDF_PBRT(const float3 wo, const float3 wi, float alphax, float alphay)
{
  const float cosThetaO = AbsCosThetaPBRT(wo); 
  const float cosThetaI = AbsCosThetaPBRT(wi);

  float3 wh = wi + wo;

  // Handle degenerate cases for microfacet reflection
  if (cosThetaI <= 1e-6f || cosThetaO <= 1e-6f) 
    return 0.0f;
    
  if (fabs(wh.x) <= 1e-6f && fabs(wh.y) <= 1e-6f && fabs(wh.z) <= 1e-6f) 
    return 0.0f;
  
  wh = normalize(wh);
  const float F = 1.0f; // FrCond(dot(wi, wh), 5.0f, 1.25f);

  return BeckmannDistributionD(wh, alphax, alphay) * BeckmannG(wo, wi, alphax, alphay) * F / fmax(4.0f * cosThetaI * cosThetaO, DEPSILON);
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


static inline void TrowbridgeReitzSample11(float cosTheta, float U1, float U2,
                                           float *slope_x, float *slope_y)
{
  // special case (normal incidence)
  if (cosTheta > 0.9999f) 
  {
    const float r   = sqrt(U1 / (1.0f - U1));
    const float phi = 6.28318530718f * U2;
    *slope_x = r * cos(phi);
    *slope_y = r * sin(phi);
    return;
  }

  const float sinTheta = sqrt(fmax(0.0f, 1.0f - cosTheta * cosTheta));
  const float tanTheta = sinTheta / cosTheta;
  const float a        = 1.0f / tanTheta;
  const float G1       = 2.0f / (1.0f + sqrt(1.f + 1.f / (a * a)));

  // sample slope_x
  const float A  = 2.0f * U1 / G1 - 1.0f;
  float tmp      = 1.f / (A * A - 1.f);
  if (tmp > 1e10f) 
    tmp = 1e10f;

  const float B         = tanTheta;
  const float D         = sqrt(fmax(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.0f));
  const float slope_x_1 = B * tmp - D;
  const float slope_x_2 = B * tmp + D;

  *slope_x = (A < 0.0f || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

  // sample slope_y
  float S;
  if (U2 > 0.5f) 
  {
    S  = 1.f;
    U2 = 2.f * (U2 - .5f);
  } 
  else 
  {
    S = -1.f;
    U2 = 2.f * (.5f - U2);
  }
  const float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
                  (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);

  *slope_y = S * z * sqrt(1.f + *slope_x * *slope_x);
  
  //CHECK(!std::isinf(*slope_y));
  //CHECK(!std::isnan(*slope_y));
}

static inline float3 TrowbridgeReitzSample(const float3 wi, float alpha_x, float alpha_y, float U1, float U2) 
{
  // 1. stretch wi
  const float3 wiStretched = normalize(make_float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

  // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
  float slope_x, slope_y;
  TrowbridgeReitzSample11(CosThetaPBRT(wiStretched), U1, U2, 
                          &slope_x, &slope_y);

  // 3. rotate
  float tmp = CosPhiPBRT(wiStretched) * slope_x - SinPhiPBRT(wiStretched) * slope_y;
  slope_y   = SinPhiPBRT(wiStretched) * slope_x + CosPhiPBRT(wiStretched) * slope_y;
  slope_x   = tmp;

  // 4. unstretch
  slope_x = alpha_x * slope_x;
  slope_y = alpha_y * slope_y;

  // 5. compute normal
  return normalize(make_float3(-slope_x, -slope_y, 1.));
}

static inline float3 TrowbridgeReitzDistributionSampleWH(const float3 wo, const float2 u, float alphax, float alphay)
{
  float3 wh;
  bool flip = wo.z < 0;
  wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u.x, u.y);
  if (flip) 
    wh = -wh;
  return wh;
}

static inline float TrowbridgeReitzG1(float3 w, float alphax, float alphay) 
{
  //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
  return 1.0f / (1.0f + TrowbridgeReitzDistributionLambda(w, alphax, alphay));
}

static inline float TrowbridgeReitzDistributionPdf(float3 wo, float3 wh, float alphax, float alphay) 
{
  return TrowbridgeReitzDistributionD(wh, alphax, alphay) * TrowbridgeReitzG1(wo, alphax, alphay) * fabs(dot(wo, wh)) / AbsCosThetaPBRT(wo);
}

inline float TrowbridgeReitzRoughnessToAlpha(float roughness) 
{
  const float x = log(fmax(roughness, 1e-3f));
  return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

#endif

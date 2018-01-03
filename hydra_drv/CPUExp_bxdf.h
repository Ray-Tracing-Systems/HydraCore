#ifndef CPUEXP_BXDF
#define CPUEXP_BXDF

#define RAND_MLT_CPU

#include "globals.h"
#include "crandom.h"
#include "cfetch.h"
#include "ctrace.h"

//#include "cmaterial.h"
//#include "clight.h"

#ifndef ABSTRACT_MATERIAL_GUARDIAN
#include "AbstractMaterial.h"
#endif

#include <vector>
#include <tuple>
#include "CPUExp_Integrators.h"

//svorovano from tungsten render
class Microfacet
{
public:


	static float roughnessToAlpha(float roughness)
	{
		float MinAlpha = 1e-3f;
		roughness = fmax(roughness, MinAlpha);
		
		return roughness;
	}

	static float D(float alpha, const float3 &m)
	{
		if (m.z <= 0.0f)
			return 0.0f;


		float alphaSq = alpha*alpha;
		float cosThetaSq = m.z*m.z;
		float tanThetaSq = fmax(1.0f - cosThetaSq, 0.0f) / cosThetaSq;
		float cosThetaQu = cosThetaSq*cosThetaSq;

		return alphaSq*INV_PI / (cosThetaQu*(alphaSq + tanThetaSq)*(alphaSq + tanThetaSq));

	}

	static float G1(float alpha, const float3 &v, const float3 &m)
	{
		if (dot(v, m)*v.z <= 0.0f)
			return 0.0f;


		float alphaSq = alpha*alpha;
		float cosThetaSq = v.z*v.z;
		float tanThetaSq = fmax(1.0f - cosThetaSq, 0.0f) / cosThetaSq;

		return 2.0f / (1.0f + sqrtf(1.0f + alphaSq*tanThetaSq));
	}

	static float G(float alpha, const float3 &i, const float3 &o, const float3 &m)
	{
		return G1(alpha, i, m)*G1(alpha, o, m);
	}

	static float pdf(float alpha, const float3 &m)
	{
		return D(alpha, m)*m.z;
	}

	static float3 sample(float alpha, float2 xi)
	{
		float phi = xi.y*M_PI*2;
		float cosTheta = 0.0f;


		float tanThetaSq = alpha*alpha*xi.x / (1.0f - xi.x);

		cosTheta = 1.0f / sqrtf(1.0f + tanThetaSq);

		float r = sqrtf(fmax(1.0f - cosTheta*cosTheta, 0.0f));


		return float3(cosf(phi)*r, sinf(phi)*r, cosTheta);
	}
};

float dielectricReflectance(float eta, float cosThetaI, float &cosThetaT)
{
	if (cosThetaI < 0.0f) 
	{
		eta = 1.0f / eta;
		cosThetaI = -cosThetaI;
	}

	float sinThetaTSq = eta*eta*(1.0f - cosThetaI*cosThetaI);

	if (sinThetaTSq > 1.0f) 
	{
		cosThetaT = 0.0f;
		return 1.0f;
	}

	cosThetaT = sqrtf(fmax(1.0f - sinThetaTSq, 0.0f));

	float Rs = (eta*cosThetaI - cosThetaT) / (eta*cosThetaI + cosThetaT);
	float Rp = (eta*cosThetaT - cosThetaI) / (eta*cosThetaT + cosThetaI);

	return (Rs*Rs + Rp*Rp)*0.5f;
}

float dielectricReflectance(float eta, float cosThetaI)
{
	float cosThetaT;
	return dielectricReflectance(eta, cosThetaI, cosThetaT);
}

float3 evalRoughDielectricBsdf(float3 wi, float3 wo, float3 N, bool sampleR, bool sampleT, float roughness, float ior)
{
	float wiDotN = dot(wi, N);
	float woDotN = dot(wo, N);

	bool reflect = wiDotN*woDotN >= 0.0f;

	if ((reflect && !sampleR) || (!reflect && !sampleT))
		return float3(0.0f, 0.0f, 0.0f);

	float alpha = Microfacet::roughnessToAlpha(roughness);

	float eta = wiDotN < 0.0f ? ior : 1.0f / ior;

	float3 m;

	if (reflect)
		m = normalize(sign(wiDotN)*(wi + wo));
	else
		m = -1.*normalize((wi*eta + wo));

	float wiDotM = dot(wi, m);
	float woDotM = dot(wo, m);
	float F = dielectricReflectance(1.0f / ior, wiDotM);
	float G = Microfacet::G(alpha, wi, wo, m);
	float D = Microfacet::D(alpha, m);

	if (reflect) 
	{
		float fr = (F*G*D*0.25f) / fabs(wiDotN);
		return float3(fr, fr, fr);
	}
	else 
	{
		float fs = fabs(wiDotM*woDotM)*(1.0f - F)*G*D / ((eta*wiDotM + woDotM)*(eta*wiDotM + woDotM)*fabs(wiDotN));
		return float3(fs, fs, fs);
	}
}

float pdfRoughDielectricBsdf(float3 wi, float3 wo, float3 N, bool sampleR, bool sampleT, float roughness, float ior)
{
	float wiDotN = dot(wi, N);
	float woDotN = dot(wo, N);

	bool reflect = wiDotN*woDotN >= 0.0f;
	if ((reflect && !sampleR) || (!reflect && !sampleT))
		return 0.0f;

	float sampleRoughness = (1.2f - 0.2f*sqrtf(fabs(wiDotN)))*roughness;
	float sampleAlpha = Microfacet::roughnessToAlpha(sampleRoughness);

	float eta = wiDotN < 0.0f ? ior : 1.0f / ior;
	float3 m;

	if (reflect)
		m = normalize(sign(wiDotN)*(wi + wo));
	else
		m = -1.*normalize((wi*eta + wo));

	float wiDotM = dot(wi, m);
	float woDotM = dot(wo, m);
	float F = dielectricReflectance(1.0f / ior, wiDotM);
	float pm = Microfacet::pdf(sampleAlpha, m);

	float pdf;

	if (reflect)
		pdf = pm*0.25f / fabs(wiDotM);
	else
		pdf = pm*fabs(woDotM) / ((eta*wiDotM + woDotM)*(eta*wiDotM + woDotM));

	if (sampleR && sampleT) 
	{
		if (reflect)
			pdf *= F;
		else
			pdf *= 1.0f - F;
	}

	return pdf;
}


bool sampleBase(bool sampleR, bool sampleT,
	float roughness, float ior)
{
	

	return true;
}


std::tuple<MatSample, int, float3> IntegratorShadowPTSSS::sampleAndEvalGGXBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float &sigmaS, float3 &sigmaA, float3 &nonAbsorbed)
{

	const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

	int matType = as_int(pHitMaterial->data[PLAIN_MAT_TYPE_OFFSET]);
	auto& gen   = randomGen();

	float ior       = 1.486f;
	float roughness = 0.1f;

	if (matType == PLAIN_MAT_CLASS_SSS)
	{
		MatSample res;
		
		float3 wi = ray_dir;
		float3 wo;

		float wiDotN = dot(ray_dir, surfElem.normal);

		float eta = wiDotN < 0.0f ? ior : 1.0f / ior;

		float sampleRoughness = (1.2f - 0.2f*sqrtf(fabs(wiDotN)))*roughness;

		float alpha = Microfacet::roughnessToAlpha(roughness);
		float sampleAlpha = Microfacet::roughnessToAlpha(sampleRoughness);

		float2 rand2 = rndFloat2(&gen);
		
		const bool hitFromBack = (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE) != 0;

		float3 m = Microfacet::sample(sampleAlpha, rand2);
		float pm = Microfacet::pdf(sampleAlpha, m);

		if (pm < 1e-10f)
		{
			res.color = float3(0, 0, 0);
			res.pdf   = 0;
      res.flags = RAY_EVENT_G; //#YEP?
			return std::make_tuple(res, 0, float3(1, 1, 1));
		}

		float wiDotM = dot(wi, m);
		float cosThetaT = 0.0f;
		float F = dielectricReflectance(1.0f / ior, wiDotM, cosThetaT);
		float etaM = wiDotM < 0.0f ? ior : 1.0f / ior;

		bool reflect;

		float rand1 = rndFloat1(&gen);
		float transmissionP = sssGetTransmission(pHitMaterial);

		float density = sssGetDensity(pHitMaterial);

		if (rand1 < transmissionP)
		{
			reflect = false;

			if (hitFromBack)
			{
				sigmaS = 0.0;
				sigmaA = float3(0.0, 0.0, 0.0);
				//res.direction = transmissionDirection(surfElem.normal, -1.*ray_dir, ior, 1.0);
			}
			else
			{
				sigmaA = density*sssGetAbsorption(pHitMaterial);
				sigmaS = density*sssGetScattering(pHitMaterial);
				//res.direction = transmissionDirection(surfElem.normal, -1.*ray_dir, 1.0, ior);
			}

		}
		else
		{
			reflect = true;

		}

		if (reflect)
			wo = 2.0f*wiDotM*m - wi;
		else
			wo = (etaM*wiDotM - sign(wiDotM)*cosThetaT)*m - etaM*wi;

		res.direction = wo;

		float woDotN = dot(wo, surfElem.normal);

		/*bool reflected = wiDotN*woDotN > 0.0f;
		if (reflected != reflect)
			return false;*/

		float woDotM = dot(wo, m);
		float G = Microfacet::G(alpha, wi, wo, m);
		float D = Microfacet::D(alpha, m);
		float color = (fabs(wiDotM)*G*D / (fabs(wiDotN)*pm));

		const float3 kd = sssGetDiffuseColor(pHitMaterial);

		res.color = float3(kd.x*color, kd.y*color, kd.z*color);

		if (reflect)
			res.pdf = pm*0.25f / fabs(wiDotM);
		else
			res.pdf = pm*fabs(woDotM) / ((eta*wiDotM + woDotM)*(eta*wiDotM + woDotM));

		if (reflect)
			res.pdf *= F;
		else
			res.pdf *= 1.0f - F;

		return std::make_tuple(res, 0, float3(1, 1, 1));
	}
	else
		return IntegratorCommon::sampleAndEvalBxDF(ray_dir, surfElem, flags, shadow);
}









#endif

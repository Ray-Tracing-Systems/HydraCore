#include <omp.h>
#include "CPUExp_Integrators.h"
#include "CPUExp_bxdf.h"


#define E                     2.7182818284590452353602874713526624977572


float3 transmissionDirection(const float3 &normal, const float3 &incident, float iorIncident, float iorTransmitted) 
{
	// Snell's Law

	float cosTheta1 = dot(normal, incident);

	float n1_n2 = iorIncident / iorTransmitted;

	float radicand = 1 - pow(n1_n2, 2) * (1 - pow(cosTheta1, 2));

	if (radicand < 0)
		return make_float3(0, 0, 0);

	float cosTheta2 = sqrt(radicand);

	if (cosTheta1 > 0) 
	{ // normal and incident are on same side of the surface.
		return n1_n2 * (-1 * incident) + (n1_n2 * cosTheta1 - cosTheta2) * normal;
	}
	else 
	{ // normal and incident are on opposite sides of the surface.
		return n1_n2 * (-1 * incident) + (n1_n2 * cosTheta1 + cosTheta2) * normal;
	}

}


std::tuple<MatSample, int, float3> IntegratorStupidPTSSS::sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float &sigmaS, float3 &sigmaA, float3 &nonAbsorbed)
{

	const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

	int matType = as_int(pHitMaterial->data[PLAIN_MAT_TYPE_OFFSET]);
	auto& gen = randomGen();

	if (matType == PLAIN_MAT_CLASS_SSS)
	{

		float3 r1 = rndFloat3(&gen);

		//bool inside = (dot(surfElem.normal, ray_dir) < 0.f);
		const bool hitFromBack = (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE) != 0;

		float ior = 1.486f;
		//const float3 kd = float3(1.0, 0.0, 0.0);
		const float3 kd = sssGetDiffuseColor(pHitMaterial);
		
		//float3 newDir = transmissionDirection(surfElem.normal, -1.*ray_dir, 1.0, ior);

		MatSample res;

		const float3 newDir = MapSampleToCosineDistribution(r1.x, r1.y, surfElem.normal, surfElem.normal, 1.0f);
		
		res.direction = newDir;
		float  cosTheta = fmax(dot(res.direction, surfElem.normal), 0.0f);

		float transmissionP = sssGetTransmission(pHitMaterial);

		float density = sssGetDensity(pHitMaterial);

		if (r1.z < transmissionP)
		{
			float3 col_diff;
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

			
			res.direction.x *= -1.;
			res.direction.y *= -1.;
			res.direction.z *= -1.;
			//res.color = nonAbsorbed*float3(1, 1, 1)*cosTheta*INV_PI;
			col_diff = float3(1 - kd.x, 1 - kd.y, 1 - kd.z);
			res.color = nonAbsorbed*col_diff*cosTheta*INV_PI;
		}
		else
		{
			res.color = nonAbsorbed*kd*cosTheta*INV_PI;
		}

		res.pdf   = cosTheta*INV_PI;
    res.flags = RAY_EVENT_D;

		//new_ray_pos += sssGetRadius(pHitMaterial);

		return std::make_tuple(res, 0, float3(1, 1, 1));
	}
	else
		return IntegratorCommon::sampleAndEvalBxDF(ray_dir, surfElem, flags, shadow);
}

float3 SampleHenyeyGreenstein(const float g, const float e1, const float e2)
{
	const float s = 1.f - 2.f * e1;
	const float cost = (s + 2.f * g * g * g * (-1.f + e1) * e1 + g * g * s + 2.f * g * (1.f - e1 + e1 * e1)) / ((1.f + g * s)*(1.f + g * s));
	const float sint = sqrt(fmax(1.f - cost * cost,1e-6f));

	float3 res;

	res.x = cos(2.f * 3.14159265358979323846f * e2) * sint;
	res.y = sin(2.f * 3.14159265358979323846f * e2) * sint;
	res.z = cost;

	return res;
}

void myCoordinateSystem(const float3 v1, float3 &v2, float3 &v3) 
{
	if (fabs(v1.x) > fabs(v1.y)) 
	{
		float invLen = 1.f / sqrt(v1.x * v1.x + v1.z * v1.z);
		v2.x = -v1.z * invLen;
		v2.y = 0.f;
		v2.z = v1.x * invLen;
	}
	else 
	{
		float invLen = 1.f / sqrt(v1.y * v1.y + v1.z * v1.z);
		v2.x = 0.f;
		v2.y = v1.z * invLen;
		v2.z = -v1.y * invLen;
	}

	v3 = cross(v1, v2);
}

float3 IntegratorStupidPTSSS::PathTraceVol(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags, float sigmaS, float3 sigmaA)
{

	float currentSigmaS = sigmaS;
	float3 currentSigmaA = sigmaA;
	//float currentSigmaT = currentSigmaS + currentSigmaA;
	auto& gen = randomGen();

	float3 nonAbsorbed(1.f, 1.f, 1.f);

	if (a_currDepth >= m_maxDepth)
		return float3(0, 0, 0);

	Lite_Hit hit = rayTrace(ray_pos, ray_dir);

	if (HitNone(hit))
		return float3(0, 0, 0);

	SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);
	const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);


	if (currentSigmaS > 0.f) 
	{
		float3 r1 = rndFloat3(&gen);
		float scatterDistance = -log(r1.x) / (currentSigmaS);

		if (scatterDistance < hit.t) 
		{
			float3 nextRay_pos = ray_pos + ray_dir*scatterDistance;
			//float3 nextRay_dir = MapSamplesToSphere(r1.y, r1.z); //isotropic
			float3 scatter_dir = normalize(SampleHenyeyGreenstein(sssGetPhase(pHitMaterial), r1.y, r1.z)); //phase function

			float3 u, v;
			CoordinateSystem(ray_dir, &u, &v);

			float3 nextRay_dir;
			nextRay_dir.x = u.x * scatter_dir.x + v.x * scatter_dir.y + ray_dir.x * scatter_dir.z;
			nextRay_dir.y = u.y * scatter_dir.x + v.y * scatter_dir.y + ray_dir.y * scatter_dir.z;
			nextRay_dir.z = u.z * scatter_dir.x + v.z * scatter_dir.y + ray_dir.z * scatter_dir.z;

			// Compute how much light was absorbed along the ray before it was scattered:
			nonAbsorbed.x = /*(currentSigmaS / (currentSigmaS + currentSigmaA.x))*/pow((float)E, (float)(-1 * (/*currentSigmaS +*/ currentSigmaA.x) * scatterDistance));
			nonAbsorbed.y = /*(currentSigmaS / (currentSigmaS + currentSigmaA.y))*/pow((float)E, (float)(-1 * (/*currentSigmaS +*/ currentSigmaA.y) * scatterDistance));
			nonAbsorbed.z = /*(currentSigmaS / (currentSigmaS + currentSigmaA.z))*/pow((float)E, (float)(-1 * (/*currentSigmaS +*/ currentSigmaA.z) * scatterDistance));
			
			return nonAbsorbed*PathTraceVol(nextRay_pos, nextRay_dir, MisData(), a_currDepth + 1, flags, currentSigmaS, currentSigmaA);
		}
		else
		{
			nonAbsorbed.x = pow((float)E, (float)(-1 * (/*currentSigmaS +*/ currentSigmaA.x) * hit.t));
			nonAbsorbed.y = pow((float)E, (float)(-1 * (/*currentSigmaS +*/ currentSigmaA.y) * hit.t));
			nonAbsorbed.z = pow((float)E, (float)(-1 * (/*currentSigmaS +*/ currentSigmaA.z) * hit.t));
		}
	}

	const bool hitFromBack = (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE) != 0;

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
	if (dot(emission, emission) > 1e-3f)
	{
		if (hitFromBack)
			return float3(0, 0, 0);
		else
			return nonAbsorbed*emission;
	}
		
	
	MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem, flags, float3(0, 0, 0), currentSigmaS, currentSigmaA, nonAbsorbed));
	float3 bxdfVal = matSam.color*(1.0f / fmaxf(matSam.pdf, 1e-20f));

	float3 nextRay_dir = matSam.direction;
	float3 nextRay_pos = ray_pos + ray_dir*surfElem.t;
	nextRay_pos = nextRay_pos + sign(dot(nextRay_dir, surfElem.normal))*surfElem.normal*1e-5f; // add small offset to ray position

	
	// change flags if needed

	return bxdfVal*PathTraceVol(nextRay_pos, nextRay_dir, MisData(), a_currDepth + 1, flags, currentSigmaS, currentSigmaA);  // --*(1.0 / (1.0 - pabsorb));
}

float3 IntegratorStupidPTSSS::PathTrace(float3 a_rpos, float3 a_rdir, MisData misPrev, int a_currDepth, uint flags)
{
	return PathTraceVol(a_rpos, a_rdir, misPrev, a_currDepth, flags);  // --*(1.0 / (1.0 - pabsorb));
}


//****SHADOW*****************************************************************************************************************************************
//***************************************************************************************************************************************************
//***************************************************************************************************************************************************


std::tuple<MatSample, int, float3> IntegratorShadowPTSSS::sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float &sigmaS, float3 &sigmaA, float3 &nonAbsorbed)
{

	const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

	int matType = as_int(pHitMaterial->data[PLAIN_MAT_TYPE_OFFSET]);
	auto& gen = randomGen();

	if (matType == PLAIN_MAT_CLASS_SSS)
	{

		float3 r1 = rndFloat3(&gen);

		//bool inside = (dot(surfElem.normal, ray_dir) < 0.f);
		const bool hitFromBack = (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE) != 0;

		float ior = 1.486f;
		//const float3 kd = float3(1.0, 0.0, 0.0);
		const float3 kd = sssGetDiffuseColor(pHitMaterial);

		//float3 newDir = transmissionDirection(surfElem.normal, -1.*ray_dir, 1.0, ior);

		MatSample res;

		const float3 newDir = MapSampleToCosineDistribution(r1.x, r1.y, surfElem.normal, surfElem.normal, 1.0f);

		res.direction = newDir;
		float  cosTheta = fmax(dot(res.direction, surfElem.normal), 0.0f);

		float transmissionP = sssGetTransmission(pHitMaterial);

		float density = sssGetDensity(pHitMaterial);

		if (r1.z < transmissionP)
		{
			float3 col_diff;
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


			res.direction.x *= -1.;
			res.direction.y *= -1.;
			res.direction.z *= -1.;
			//res.color = nonAbsorbed*float3(1, 1, 1)*cosTheta*INV_PI;
			col_diff = float3(1 - kd.x, 1 - kd.y, 1 - kd.z);
			res.color = /*nonAbsorbed*/col_diff*cosTheta*INV_PI;
		}
		else
		{
			res.color = /*nonAbsorbed*/kd*cosTheta*INV_PI;
		}

		res.pdf = cosTheta*INV_PI;
    res.flags = RAY_EVENT_D;

		//new_ray_pos += sssGetRadius(pHitMaterial);

		return std::make_tuple(res, 0, float3(1, 1, 1));
	}
	else
		return IntegratorCommon::sampleAndEvalBxDF(ray_dir, surfElem, flags, shadow);
}


float3 IntegratorShadowPTSSS::directLightSample(const SurfaceHit surfElem, const PlainMaterial* pHitMaterial, float3 ray_dir, uint flags)
{
	float3 explicitColor(0, 0, 0);

  auto& gen = randomGen();
  float lightPickProb = 1.0f;
  int lightOffset = SelectRandomLightRev(rndFloat2_Pseudo(&gen), surfElem.pos, m_pGlobals,
                                         &lightPickProb);

	if (lightOffset >= 0)
	{
		__global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    ShadowSample explicitSam;
    LightSampleRev(pLight, rndFloat3(&gen), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   &explicitSam); 

		float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos); // explicitSam.direction;
		float3 shadowRayPos = surfElem.pos + shadowRayDir*fmax(maxcomp(surfElem.pos), 1.0f)*GEPSILON;

		float3 shadow = shadowTrace(shadowRayPos, shadowRayDir, explicitSam.maxDist*0.9995f);


		const bool hitFromBack = (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE) != 0;

		ShadeContext sc;
		sc.wp = surfElem.pos;
		sc.l = shadowRayDir;
		sc.v = (-1.0f)*ray_dir;
		sc.n = surfElem.normal;
		sc.fn = surfElem.flatNormal;
		sc.tg = surfElem.tangent;
		sc.bn = surfElem.biTangent;
		sc.tc = surfElem.texCoord;

		float3 brdfVal = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, nullptr, nullptr).brdf; // a_shadingTexture

		explicitColor = (explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON)))*brdfVal*shadow; // #ALERT!!! MUST MUL BY cosThetaOut !!!!!
	}

	return explicitColor;
}

float3 IntegratorShadowPTSSS::directLightSampleScatter(const SurfaceHit surfElem, const float3 pos, const PlainMaterial* pHitMaterial, float3 ray_dir, uint flags, float3 sigmaA, float sigmaS)
{
	float3 explicitColor(0, 0, 0);
	auto& gen = randomGen();

  float lightPickProb = 1.0f;
  int lightOffset = SelectRandomLightRev(rndFloat2_Pseudo(&gen), surfElem.pos, m_pGlobals,
                                         &lightPickProb);

	int matType = as_int(pHitMaterial->data[PLAIN_MAT_TYPE_OFFSET]);

	if (lightOffset >= 0)
	{
		__global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    ShadowSample explicitSam;
    LightSampleRev(pLight, rndFloat3(&gen), pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   &explicitSam);

		float3 shadowRayDir = normalize(explicitSam.pos - pos); // explicitSam.direction;
		float3 shadowRayPos = pos; //+ shadowRayDir*fmax(maxcomp(pos), 1.0f)*GEPSILON;

		Lite_Hit hit = rayTrace(shadowRayPos, shadowRayDir);

		if (HitNone(hit))
			return float3(0, 0, 0);

		SurfaceHit surfElem2 = surfaceEval(shadowRayPos, shadowRayDir, hit);

		float3 shadow = shadowTrace(shadowRayPos + shadowRayDir*(hit.t + 1e-5f), shadowRayDir, explicitSam.maxDist*0.9995f);

	/*	if (matType == PLAIN_MAT_CLASS_SSS && shadow.x > 0)
		{
			int a = 1;
			explicitColor = explicitColor*a;
		}*/

		float3 nonAbsorbed(1.0, 1.0, 1.0);

		nonAbsorbed.x = nonAbsorbed.x * pow((float)E, (float)(-1 * (sigmaA.x) * hit.t));
		nonAbsorbed.y = nonAbsorbed.y * pow((float)E, (float)(-1 * (sigmaA.y) * hit.t));
		nonAbsorbed.z = nonAbsorbed.z * pow((float)E, (float)(-1 * (sigmaA.z) * hit.t));


		float  cosTheta = fmax(dot(shadowRayDir, -1.*surfElem2.normal), 0.0f);

		//MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem, flags, shadow, sigmaS, sigmaA, nonAbsorbed));
		//if (!hitFromBack)
		explicitColor = (explicitSam.color * nonAbsorbed*(1.0f / fmax(explicitSam.pdf, DEPSILON)))*shadow/*sssGetTransmission(pHitMaterial)**cosTheta*INV_PI*/;
		/*if (dot(explicitColor, explicitColor) > 0)
		{
			int a = 1;
			explicitColor = explicitColor*a;
		}*/
	}

	return explicitColor;
}


float3  IntegratorShadowPTSSS::PathTraceVol(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags, float sigmaS, float3 sigmaA)
{

	float currentSigmaS = sigmaS;
	float3 currentSigmaA = sigmaA;
	//float currentSigmaT = currentSigmaS + currentSigmaA;

	float3 nonAbsorbed(1.f, 1.f, 1.f);

	if (a_currDepth >= m_maxDepth)
		return float3(0, 0, 0);

	Lite_Hit hit = rayTrace(ray_pos, ray_dir);

	if (HitNone(hit))
		return float3(0, 0, 0);

	SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
	if (dot(emission, emission) > 1e-3f)
	{
		return float3(0, 0, 0);
	}

	const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
	auto& gen = randomGen();

	if (currentSigmaS > 0.f)
	{
		float3 r1 = rndFloat3(&gen);
		float scatterDistance = -log(r1.x) / currentSigmaS;

		if (scatterDistance < hit.t)
		{
			float3 nextRay_pos = ray_pos + ray_dir*scatterDistance;
			//float3 nextRay_dir = MapSamplesToSphere(r1.y, r1.z); //isotropic

			float3 scatter_dir = SampleHenyeyGreenstein(sssGetPhase(pHitMaterial), r1.y, r1.z); //phase function

			float3 u, v;
			CoordinateSystem(ray_dir, &u, &v);

			float3 nextRay_dir;
			nextRay_dir.x = u.x * scatter_dir.x + v.x * scatter_dir.y + ray_dir.x * scatter_dir.z;
			nextRay_dir.y = u.y * scatter_dir.x + v.y * scatter_dir.y + ray_dir.y * scatter_dir.z;
			nextRay_dir.z = u.z * scatter_dir.x + v.z * scatter_dir.y + ray_dir.z * scatter_dir.z;

			// Compute how much light was absorbed along the ray before it was scattered:
			nonAbsorbed.x = nonAbsorbed.x * pow((float)E, (float)(-1 * (currentSigmaA.x) * scatterDistance));
			nonAbsorbed.y = nonAbsorbed.y * pow((float)E, (float)(-1 * (currentSigmaA.y) * scatterDistance));
			nonAbsorbed.z = nonAbsorbed.z * pow((float)E, (float)(-1 * (currentSigmaA.z) * scatterDistance));

			//direct sample lights
			float3 explicit_color = directLightSampleScatter(surfElem, nextRay_pos, pHitMaterial, ray_dir, flags, currentSigmaA, currentSigmaS);


			return nonAbsorbed*explicit_color + nonAbsorbed*PathTraceVol(nextRay_pos, nextRay_dir, MisData(), a_currDepth + 1, flags, currentSigmaS, currentSigmaA);
		}
		else
		{
			nonAbsorbed.x = nonAbsorbed.x * pow((float)E, (float)(-1 * (currentSigmaA.x) * hit.t));
			nonAbsorbed.y = nonAbsorbed.y * pow((float)E, (float)(-1 * (currentSigmaA.y) * hit.t));
			nonAbsorbed.z = nonAbsorbed.z * pow((float)E, (float)(-1 * (currentSigmaA.z) * hit.t));
		}
	}
	

	float3 explicitColor(0, 0, 0);

	explicitColor = directLightSample(surfElem, pHitMaterial, ray_dir, flags);

	/*if (explicitColor.x > 0 && nonAbsorbed.y < 1)
	{
		int a = 1;
		explicitColor = 1*explicitColor;
	}*/

	//explicitColor = nonAbsorbed*explicitColor;

	//float3 nextRay_pos(0, 0, 0);

	MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem, flags, float3(0, 0, 0), currentSigmaS, currentSigmaA, nonAbsorbed));
	float3 bxdfVal = matSam.color * (1.0f / fmaxf(matSam.pdf, 1e-20f));

	float3 nextRay_pos = ray_pos + ray_dir*surfElem.t;
	float3 nextRay_dir = matSam.direction;
	nextRay_pos = nextRay_pos + sign(dot(nextRay_dir, surfElem.normal))*surfElem.normal*1e-5f; // add small offset to ray position

	// change flags if needed

	return nonAbsorbed*explicitColor + nonAbsorbed*bxdfVal*PathTraceVol(nextRay_pos, nextRay_dir, MisData(), a_currDepth + 1, flags, currentSigmaS, currentSigmaA);  // --*(1.0 / (1.0 - pabsorb));
}




std::tuple<MatSample, int, float3> IntegratorShadowPTSSS::sampleAndEvalBxDF(float3 ray_dir, const SurfaceHit& surfElem, uint flags, float3 shadow, float3 &new_ray_pos)
{

	const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);

	int matType = as_int(pHitMaterial->data[PLAIN_MAT_TYPE_OFFSET]);
	auto& gen = randomGen();

	new_ray_pos = surfElem.pos;

	if (matType == PLAIN_MAT_CLASS_SSS)
	{

		float3 r1 = rndFloat3(&gen);

		const float3 kd = clamp(sssGetAbsorption(pHitMaterial), 0.0f, 1.0f);
		const float3 newDir = MapSampleToCosineDistribution(r1.x, r1.y, surfElem.normal, surfElem.normal, 1.0f);
		const float cosTheta = fmax(dot(newDir, surfElem.normal), 0.0f);

		MatSample res;
		res.direction = newDir;
		res.pdf = cosTheta*INV_PI;
		res.color = kd*cosTheta*INV_PI;
    res.flags = RAY_EVENT_D;

		//new_ray_pos += sssGetRadius(pHitMaterial);

		return std::make_tuple(res, 0, float3(1, 1, 1));
	}
	else
		return IntegratorCommon::sampleAndEvalBxDF(ray_dir, surfElem, flags, shadow);
}

float3  IntegratorShadowPTSSS::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, uint flags)
{
	return PathTraceVol(ray_pos, ray_dir, misPrev, a_currDepth, flags);  // --*(1.0 / (1.0 - pabsorb));
}
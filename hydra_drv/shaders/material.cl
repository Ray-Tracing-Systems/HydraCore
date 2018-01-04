#include "globals.h"
#include "cfetch.h"
#include "crandom.h"
#include "cmaterial.h"
#include "clight.h"

static inline ushort4 compressShadow(float3 shadow)
{
  ushort4 shadowCompressed;

  shadowCompressed.x = (ushort)(65535.0f * shadow.x);
  shadowCompressed.y = (ushort)(65535.0f * shadow.y);
  shadowCompressed.z = (ushort)(65535.0f * shadow.z);
  shadowCompressed.w = 0;

  return shadowCompressed;
}

static inline float3 decompressShadow(ushort4 shadowCompressed)
{
  const float invNormCoeff = 1.0f / 65535.0f;
  return invNormCoeff*make_float3((float)shadowCompressed.x, (float)shadowCompressed.y, (float)shadowCompressed.z);
}


__kernel void Shade(__global const float4*    restrict a_rpos,
                    __global const float4*    restrict a_rdir,
                    __global const uint*      restrict a_flags,
                    
                    __global const float4*    restrict in_hitPosNorm,
                    __global const float2*    restrict in_hitTexCoord,
                    __global const uint*      restrict in_flatNorm,
                    __global const HitMatRef* restrict in_matData,
                    __global const Hit_Part4* restrict in_hitTangent,
                    
                    __global const float4*    restrict in_data1,
                    __global const float4*    restrict in_data2,
                    __global const ushort4*   restrict in_shadow,
                    __global const float*     restrict in_lightPickProb,
                    __global const float4*    restrict in_normalsFull,

                    __global float4*          restrict out_color,
                    
                    __global const float4*    restrict a_texStorage1,
                    __global const float4*    restrict a_texStorage2,
                    __global const float4*    restrict a_mtlStorage,

                    int iNumElements,
                    __global const EngineGlobals* restrict a_globals
                    )
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  if (a_globals->lightsNum == 0)
  {
    out_color[tid] = make_float4(0,0,0,0);
    return;
  }

  const bool wasGlossyOrDiffuse = (unpackRayFlags(flags) & RAY_GRAMMAR_GLOSSY_REFLECTION) || (unpackRayFlags(flags) & RAY_GRAMMAR_DIFFUSE_REFLECTION);
  const uint rayBounceNum       = unpackBounceNum(flags);

  if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
  {
    if (wasGlossyOrDiffuse && rayBounceNum >= 1)
      return;
  }

  if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
  {
    if (!wasGlossyOrDiffuse)
      return;
  }


  Hit_Part4 btanAndN = in_hitTangent[tid];

  //float4 data        = in_hitPosNorm[tid];
  float3 hitPos      = to_float3(in_hitPosNorm[tid]);
  float3 hitNorm     = to_float3(in_normalsFull[tid]); // normalize(decodeNormal(as_int(data.w)));
  float2 hitTexCoord = in_hitTexCoord[tid];

  float3 hitBiTang = decodeNormal(btanAndN.tangentCompressed);
  float3 hitBiNorm = decodeNormal(btanAndN.bitangentCompressed);

  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mtlStorage, GetMaterialId(in_matData[tid]));

  float3 ray_pos = to_float3(a_rpos[tid]);
  float3 ray_dir = to_float3(a_rdir[tid]);

  float4 data1   = in_data1[tid];
  float4 data2   = in_data2[tid];
  float3 flatN   = decodeNormal(in_flatNorm[tid]);

  ShadowSample explicitSam;

  explicitSam.pos     = to_float3(data1);
  explicitSam.color   = to_float3(data2);
  explicitSam.pdf     = fabs(data1.w);
  explicitSam.maxDist = data2.w;
  explicitSam.isPoint = (data1.w <= 0);

  //float3 shadowRayPos = hitPos + hitNorm*maxcomp(hitPos)*GEPSILON;
  float3 shadowRayDir = normalize(explicitSam.pos - hitPos); 

  const bool currLightCastCaustics = (a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS);
  const bool disableCaustics       = (unpackBounceNumDiff(flags) > 0) && !currLightCastCaustics;
  
  if ((a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_PRIMARY) && (a_globals->varsI[HRT_RENDER_LAYER_DEPTH] == unpackBounceNum(flags))) //////////////////////////////////////////////////////////////////
    pHitMaterial = materialAtOffset(a_mtlStorage, a_globals->varsI[HRT_WHITE_DIFFUSE_OFFSET]);

  // bool hitFromBack = (bool)(unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE);

  ShadeContext sc;
  sc.wp = hitPos;
  sc.l  = shadowRayDir;
  sc.v  = (-1.0f)*ray_dir;
  sc.n  = hitNorm;
  sc.fn = flatN;
  sc.tg = hitBiTang;
  sc.bn = hitBiNorm;
  sc.tc = hitTexCoord;

  const BxDFResult brdfAndPdf = materialEval(pHitMaterial, &sc, disableCaustics, false, /* global data --> */ a_globals, a_texStorage1, a_texStorage2);

  const float3 shadow        = decompressShadow(in_shadow[tid]);
  const float  lightPickProb = in_lightPickProb[tid];

  float  lgtPdf   = explicitSam.pdf*lightPickProb;
  float misWeight = misWeightHeuristic(lgtPdf, brdfAndPdf.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
  
  if (explicitSam.isPoint)
    misWeight = 1.0f;

  const float cosThetaOut1 = fmax(+dot(shadowRayDir, hitNorm), 0.0f);
  const float cosThetaOut2 = fmax(-dot(shadowRayDir, hitNorm), 0.0f);

  const float3 bxdfVal = (brdfAndPdf.brdf*cosThetaOut1 + brdfAndPdf.btdf*cosThetaOut2);

  float3 shadeColor = (explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON)))*bxdfVal*misWeight*shadow; 

  if (unpackBounceNum(flags) > 0)
    shadeColor = clamp(shadeColor, 0.0f, a_globals->varsF[HRT_BSDF_CLAMPING]);

  float maxColorLight = fmax(explicitSam.color.x, fmax(explicitSam.color.y, explicitSam.color.z));
  float maxColorShade = fmax(shadeColor.x, fmax(shadeColor.y, shadeColor.z));

  float fVisiableLightsNum = 1.0f / lightPickProb;
  float fShadowSamples = 1.0f;
  shadeColor *= (fVisiableLightsNum / fShadowSamples);

  out_color[tid] = to_float4(shadeColor, lightPickProb);
} 



__kernel void NextBounce(__global   float4*        restrict a_rpos,
                         __global   float4*        restrict a_rdir,
                         __global   uint*          restrict a_flags,
                         __global RandomGen*       restrict out_gens,
                         
                         __global const float4*    restrict in_hitPosNorm,
                         __global const float2*    restrict in_hitTexCoord,
                         __global const uint*      restrict in_flatNorm,
                         __global const HitMatRef* restrict in_matData,
                         __global const Hit_Part4* restrict in_hitTangent,
                         __global const float4*    restrict in_hitNormFull,

                         __global float4*          restrict a_color,
                         __global float4*          restrict a_thoroughput,
                         __global MisData*         restrict a_misDataPrev,
                         __global ushort4*         restrict a_shadow,
                         __global float4*          restrict a_fog,
                         __global const float4*    restrict in_shadeColor,

                         __global const float4*    restrict a_texStorage1,    
                         __global const float4*    restrict a_texStorage2,
                         __global const float4*    restrict a_mtlStorage,
                         __global const float4*    restrict a_pdfStorage,   //

                         __global const int*       restrict a_instLightInstId,
                         __global const Lite_Hit*  restrict in_liteHit,
                         
                         int iNumElements,
                         __global const EngineGlobals*  restrict a_globals,
                         __global const float*          restrict a_qmcVec,
                         __global const int2*           restrict a_qmcSorted)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];

  float3 ray_pos = to_float3(a_rpos[tid]);
  float3 ray_dir = to_float3(a_rdir[tid]);

#ifdef MLT_MULTY_PROPOSAL
  __global const float* qmcVec = (a_qmcVec == 0) ? 0 : a_qmcVec + (tid/MLT_PROPOSALS)*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
#else
  
  #ifndef MLT_SORT_RAYS
  __global const float* qmcVec = (a_qmcVec == 0) ? 0 : a_qmcVec + tid*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #else
  __global const float* qmcVec = (a_qmcVec == 0) ? 0 : a_qmcVec + a_qmcSorted[tid].y*a_globals->varsI[HRT_MLT_MAX_NUMBERS];
  #endif

#endif

  // if hit environment
  //
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE)
  {
    const int  hitId         = hitDirectLight(ray_dir, a_globals);
    const bool makeZeroOfMLT = mltStrageCondition(flags, a_globals->g_flags, a_misDataPrev[tid]);
    float3     nextPathColor = make_float3(0, 0, 0);

    if (hitId >= 0) // hit any sun light
    {
      __global const PlainLight* pLight = a_globals->suns + hitId;  
      float3 lightColor = lightBaseColor(pLight)*directLightAttenuation(pLight, ray_pos);

      const float pdfW = directLightEvalPDF(pLight, ray_dir);

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////
      MisData misPrev = a_misDataPrev[tid];

      if (makeZeroOfMLT || (unpackBounceNum(flags) > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0))) //#TODO: check this for bug with 2 hdr env light (test 335)
      {
        lightColor = make_float3(0, 0, 0);
      }
      else if (((misPrev.isSpecular == 1) && (a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS)) || (a_globals->g_flags & HRT_STUPID_PT_MODE))
        lightColor *= (1.0f/pdfW);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////

      const float3 pathThroughput = to_float3(a_thoroughput[tid]);
      nextPathColor = to_float3(a_color[tid]) + pathThroughput*lightColor;
    }
    else
    {
      int renderLayer = a_globals->varsI[HRT_RENDER_LAYER];
      
      float3 envColor       = environmentColor(to_float3(a_rdir[tid]), a_misDataPrev[tid], flags, a_globals, a_mtlStorage, a_pdfStorage, a_texStorage1);
      float3 pathThroughput = to_float3(a_thoroughput[tid]);
      
      if (makeZeroOfMLT || ((renderLayer == LAYER_SECONDARY) && (unpackBounceNum(flags) <= 1)))
        envColor = make_float3(0,0,0);
      
      nextPathColor = to_float3(a_color[tid]) + pathThroughput*envColor;
    }

    const uint otherFlags = unpackRayFlags(flags);
    a_flags      [tid] = packRayFlags(flags, RAY_IS_DEAD | (otherFlags & (~RAY_GRAMMAR_OUT_OF_SCENE)));
    a_color      [tid] = to_float4(nextPathColor, 0.0f);
    a_thoroughput[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }


  float3 outPathColor      = make_float3(0,0,0);
  float3 outPathThroughput = make_float3(0,0,0);

  float3 hitPos;
  float3 hitNorm;
  float3 flatNorm;
  float2 hitTexCoord;
  float3 hitBiTang;
  float3 hitBiNorm;

  bool thisBounceIsDiffuse = false;
  bool hitLightSource = false;

  __global const PlainMaterial* pHitMaterial = 0;

  if (rayIsActiveU(flags)) 
  {
    float4 data  = in_hitPosNorm[tid];
  
    hitPos       = to_float3(data);
    hitNorm      = to_float3(in_hitNormFull[tid]); //normalize(decodeNormal(as_int(data.w)));
    flatNorm     = normalize(decodeNormal(in_flatNorm[tid]));
    hitTexCoord  = in_hitTexCoord[tid];
  
    Hit_Part4 btanAndN = in_hitTangent[tid];
    hitBiTang    = decodeNormal(btanAndN.tangentCompressed);
    hitBiNorm    = decodeNormal(btanAndN.bitangentCompressed);
  
    pHitMaterial = materialAt(a_globals, a_mtlStorage, GetMaterialId(in_matData[tid]));
  
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
    //if hit ligh ?
    //

    bool hitEmissiveMaterialMLT = false;
    const bool skipPieceOfShit  = materialIsInvisLight(pHitMaterial) && isEyeRay(flags);
    const float3 emissionVal    = (pHitMaterial == 0) ? make_float3(0,0,0) : materialEvalEmission(pHitMaterial, ray_dir, hitNorm, hitTexCoord, a_globals, a_texStorage1, a_texStorage2);
    
    if (dot(emissionVal, emissionVal) > 1e-6f && !skipPieceOfShit)
    {
      if (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE)
        hitNorm = hitNorm*(-1.0f);
    
      const Lite_Hit liteHit = in_liteHit[tid];
    
      if (dot(ray_dir, hitNorm) < 0.0f)
      {
        outPathColor = emissionVal;
    
        const int lightOffset = (a_globals->lightsNum == 0) ? -1 : a_instLightInstId[liteHit.instId];
        if (lightOffset >= 0)
        {
          MisData misPrev = a_misDataPrev[tid];
    
          __global const PlainLight* pLight = lightAt(a_globals, lightOffset);  
          outPathColor = lightGetIntensity(pLight, ray_pos, ray_dir, hitNorm, hitTexCoord, flags, misPrev, a_globals, a_texStorage1, a_texStorage2);
    
          if (unpackBounceNum(flags) > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0))
          {
            const float lgtPdf    = lightPdfSelectRev(pLight)*lightEvalPDF(pLight, ray_pos, ray_dir, hitPos, hitNorm, hitTexCoord, a_pdfStorage, a_globals);
            const float bsdfPdf   = misPrev.matSamplePdf;
            const float misWeight = misWeightHeuristic(bsdfPdf, lgtPdf); // (bsdfPdf*bsdfPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    
            outPathColor *= misWeight;
          }
    
          if (unpackBounceNum(flags) > 0)
            outPathColor = clamp(outPathColor, 0.0f, a_globals->varsF[HRT_BSDF_CLAMPING]);
    
          __global const PlainMaterial* pPrevMaterial = materialAtOffset(a_mtlStorage, misPrev.prevMaterialOffset); 
    
          bool disableCaustics = (unpackBounceNumDiff(flags) > 0) && !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS) && materialCastCaustics(pPrevMaterial); // and prev material cast caustics
          if (disableCaustics)
            outPathColor = make_float3(0, 0, 0);
    
          hitLightSource = true; // kill thread next if it hit real light source
        }
        else // hit emissive material, not a light 
        {
          const uint rayBounceNum = unpackBounceNum(flags);
          
          const MisData misPrev         = a_misDataPrev[tid];
          const uint otherRayFlags      = unpackRayFlags(flags);
          const bool wasGlossyOrDiffuse = (otherRayFlags & RAY_GRAMMAR_DIFFUSE_REFLECTION) || (otherRayFlags & RAY_GRAMMAR_GLOSSY_REFLECTION);
          
          if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
          {
            if (wasGlossyOrDiffuse)
              outPathColor = make_float3(0, 0, 0);
          }
          else if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
          {
            if (wasGlossyOrDiffuse)
              hitEmissiveMaterialMLT = true;
          }
        }
      }
      
    
      // make lights black for 'LAYER_INCOMING_RADIANCE'
      const uint rayBounceNum     = unpackBounceNum(flags);
      const uint rayBounceNumDiff = unpackBounceNumDiff(flags);
    
      if ((a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_RADIANCE || (a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_PRIMARY)) && (a_globals->varsI[HRT_RENDER_LAYER_DEPTH] == rayBounceNum))
        outPathColor = make_float3(0, 0, 0);
    
      if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_FORBID_EMISSIVE_GI) && rayBounceNumDiff > 0)
        outPathColor = make_float3(0, 0, 0);
    
      if (a_globals->varsI[HRT_RENDER_LAYER] == LAYER_SECONDARY && rayBounceNum <= 1)
        outPathColor = make_float3(0, 0, 0);
    
      if ((a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY) && !hitEmissiveMaterialMLT)
      {
        const MisData misPrev         = a_misDataPrev[tid];
        const uint otherRayFlags      = unpackRayFlags(flags);
        const bool wasGlossyOrDiffuse = (otherRayFlags & RAY_GRAMMAR_DIFFUSE_REFLECTION) || (otherRayFlags & RAY_GRAMMAR_GLOSSY_REFLECTION);
    
        if (((misPrev.isSpecular == 0) && wasGlossyOrDiffuse) || rayBounceNum <= 1 || !wasGlossyOrDiffuse)
          outPathColor = make_float3(0, 0, 0);
      }
    
    }
  
  }
  else if (pHitMaterial != 0)
  {
     // don't put any shit here !!!
     //
  
  } 
  
  MatSample brdfSample;
  int    matOffset   = 0;
  bool   isThinGlass = false;

  RandomGen gen  = out_gens[tid];
  gen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  if (rayIsActiveU(flags))
  {
    matOffset    = materialOffset(a_globals, GetMaterialId(in_matData[tid]));
  
    BRDFSelector mixSelector = materialRandomWalkBRDF(pHitMaterial, &gen, qmcVec, ray_dir, hitNorm, hitTexCoord, a_globals, a_texStorage1, unpackBounceNum(flags), false, false);
  
    uint rayBounceNum = unpackBounceNum(flags);
    if ((a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_RADIANCE) && (a_globals->varsI[HRT_RENDER_LAYER_DEPTH] == rayBounceNum)) // piss way (render incoming radiance)
    {
      pHitMaterial        = materialAtOffset(a_mtlStorage, a_globals->varsI[HRT_WHITE_DIFFUSE_OFFSET]);
      mixSelector.w       = 1.0f;
      thisBounceIsDiffuse = true;
    }
    else                                               // normal execution way
    {
			// this is needed component for mis further (misNext.prevMaterialOffset = matOffset;)
      matOffset           = matOffset    + mixSelector.localOffs*(sizeof(PlainMaterial)/sizeof(float4));
      pHitMaterial        = pHitMaterial + mixSelector.localOffs;
      thisBounceIsDiffuse = materialHasDiffuse(pHitMaterial);
    }
  
    const float3 shadow = decompressShadow(a_shadow[tid]);
  
    // sample material
    {
      ShadeContext sc;
  
      sc.wp  = hitPos;
      sc.l   = ray_dir; 
      sc.v   = ray_dir;
      sc.n   = hitNorm;
      sc.fn  = flatNorm;
      sc.tg  = hitBiTang;
      sc.bn  = hitBiNorm;
      sc.tc  = hitTexCoord;
      sc.hfi = (materialGetType(pHitMaterial) == PLAIN_MAT_CLASS_GLASS) && (bool)(unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE);  //hit glass from other side
  
      const float3 randsm = rndMat(&gen, qmcVec, unpackBounceNum(flags));
  
      MaterialLeafSampleAndEvalBRDF(pHitMaterial, randsm, &sc, shadow, a_globals, a_texStorage1, a_texStorage2,
                                    &brdfSample);

      isThinGlass = isPureSpecular(brdfSample) && (rayBounceNum > 0) && !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS) && (materialGetType(pHitMaterial) == PLAIN_MAT_CLASS_THIN_GLASS); // materialIsTransparent(pHitMaterial);
    }
  
    const float selectorPdf = mixSelector.w;
    const float invPdf      = 1.0f / fmax(brdfSample.pdf*selectorPdf, DEPSILON);
    const float cosTheta    = fabs(dot(brdfSample.direction, hitNorm));

    outPathThroughput = clamp(cosTheta*brdfSample.color*invPdf, 0.0f, 1.0f);
   
    if (!isfinite(outPathThroughput.x)) outPathThroughput.x = 0.0f;
    if (!isfinite(outPathThroughput.y)) outPathThroughput.y = 0.0f;
    if (!isfinite(outPathThroughput.z)) outPathThroughput.z = 0.0f;
  
    // calc new ray
    //    
    ray_dir = brdfSample.direction;
    ray_pos = OffsRayPos(hitPos, hitNorm, brdfSample.direction); 
  }
  

  const float pabsorb = probabilityAbsorbRR(flags, a_globals->g_flags);
  
  float rrChoice = 0.0f;
  if(pabsorb > 0.0f)
    rrChoice = rndFloat1_Pseudo(&gen);
   
  out_gens[tid] = gen;
  
  if (pabsorb >= 0.1f)
  {
    if (rrChoice < pabsorb)
      outPathThroughput = make_float3(0.0f, 0.0f, 0.0f);
    else
      outPathThroughput = outPathThroughput*(1.0f / (1.0f - pabsorb));
  } 

  float3 oldPathThroughput = make_float3(1,1,1);
  float3 newPathThroughput = make_float3(1,1,1);

  if (rayIsActiveU(flags))
  { 
    // calc attenuation in thick glass
		//
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		const float dotNewOld  = dot(ray_dir, to_float3(a_rdir[tid]));
		const bool  gotOutside = (dotNewOld > 0.0f) && (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE);
		const float dist       = length(to_float3(in_hitPosNorm[tid]) - to_float3(a_rpos[tid]));

		const float3 fogAtten  = attenuationStep(pHitMaterial, dist, gotOutside, a_fog + tid);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    uint rayBounceNum  = unpackBounceNum(flags);
    uint diffBounceNum = unpackBounceNumDiff(flags);

    float4 shadeData  = in_shadeColor[tid];

    oldPathThroughput = to_float3(a_thoroughput[tid])*fogAtten;
    newPathThroughput = oldPathThroughput*outPathThroughput;

    if (a_globals->varsI[HRT_RENDER_LAYER] == LAYER_PRIMARY && rayBounceNum >= 1)
    {
      shadeData         = make_float4(0, 0, 0, 0);
      newPathThroughput = make_float3(0, 0, 0);
    }

    if (a_globals->varsI[HRT_RENDER_LAYER] == LAYER_SECONDARY && rayBounceNum == 0)
      shadeData = make_float4(0, 0, 0, 0);


    // split primary and secondary lighting
    //
    {
      const uint otherRayFlags      = unpackRayFlags(flags);
      const bool wasGlossyOrDiffuse = (otherRayFlags & RAY_GRAMMAR_GLOSSY_REFLECTION) || (otherRayFlags & RAY_GRAMMAR_DIFFUSE_REFLECTION);
      const bool isGlossyOrDiffuse  = (isDiffuse(brdfSample) || isGlossy(brdfSample));

      if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
      {
        if ((isGlossyOrDiffuse && rayBounceNum >= 1) || wasGlossyOrDiffuse)
          newPathThroughput = make_float3(0, 0, 0);

        if (wasGlossyOrDiffuse)
          shadeData = make_float4(0, 0, 0, 0);
      }
      else if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
      {
        uint otherRayFlags = unpackRayFlags(flags);
        if (!wasGlossyOrDiffuse)
          shadeData = make_float4(0, 0, 0, 0);
      }
    }


    if (!(a_globals->g_flags & (HRT_DIFFUSE_PHOTON_TRACING | HRT_CAUSTIC_PHOTON_TRACING)))
    {
      if (a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_RADIANCE && (a_globals->varsI[HRT_RENDER_LAYER_DEPTH] == rayBounceNum) &&
        !(a_globals->g_flags & (HRT_DIFFUSE_PHOTON_TRACING | HRT_CAUSTIC_PHOTON_TRACING))) // no shade need when compute irradiance, no shade when trace photons
        outPathColor += make_float3(0, 0, 0);
      else
        outPathColor += to_float3(shadeData);
    }

   
    if (!isThinGlass)                                    // THIS IS A VERY HUGE HACK  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    {
      MisData misNext            = a_misDataPrev[tid];
      
      misNext.matSamplePdf       = brdfSample.pdf;
      misNext.isSpecular         = (int)isPureSpecular(brdfSample);
      misNext.lightPickProb      = shadeData.w;
      misNext.prevMaterialOffset = matOffset;

      a_misDataPrev[tid]         = misNext;
    }

    flags = flagsNextBounce(flags, brdfSample, a_globals);

    float4 nextPathColor;

    if (a_globals->g_flags & (HRT_DIFFUSE_PHOTON_TRACING | HRT_CAUSTIC_PHOTON_TRACING)) // photon maps
    { 
      if (rayBounceNum == 0)
        nextPathColor = shadeData*to_float4(outPathThroughput*fogAtten, 1.0f);
      else
        nextPathColor = a_color[tid]*to_float4(outPathThroughput*fogAtten, 1.0f);
      nextPathColor.w = 1.0f;
    }
    else
    { 
      nextPathColor   = a_color[tid] + to_float4(oldPathThroughput*outPathColor, 0.0f);
      nextPathColor.w = 1.0f;
    }


    if (maxcomp(newPathThroughput) < 0.00001f || hitLightSource)
      flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);

    if (unpackRayFlags(flags) & RAY_IS_DEAD)
      newPathThroughput = make_float3(0, 0, 0);

    a_flags      [tid] = flags;
    a_rpos       [tid] = to_float4(ray_pos, 0.0f);
    a_rdir       [tid] = to_float4(ray_dir, 0.0f);
    a_color      [tid] = nextPathColor;
    a_thoroughput[tid] = to_float4(newPathThroughput, 0.0f);
  }


}


__kernel void NextTransparentBounce(__global   float4*    a_rpos,
                                    __global   float4*    a_rdir,
                                    __global   uint*      a_flags,
                                    
                                    __global float4*    in_hitPosNorm,
                                    __global float2*    in_hitTexCoord,
                                    __global HitMatRef* in_matData,
                                    
                                    __global float4*    a_color,
                                    __global float4*    a_thoroughput,
                                    __global float4*    a_fog,
                                 
                                    texture2d_t         a_shadingTexture,    
                                    
                                    int iNumElements,
                                    __global const EngineGlobals* a_globals)
{
  __global const float4* a_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];

  /*
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE) // if hit environment
  {
    a_color[tid]       = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    a_thoroughput[tid] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  }
  else
  {
    a_color[tid]       = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    a_thoroughput[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  */
  
  
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE) // if hit environment
  {
    uint otherFlags    = unpackRayFlags(flags);
    a_flags[tid]       = packRayFlags(flags, (otherFlags & (~RAY_GRAMMAR_OUT_OF_SCENE)) | RAY_IS_DEAD); // disable RAY_GRAMMAR_OUT_OF_SCENE, write flags;
    a_color[tid]       = a_thoroughput[tid];
    a_thoroughput[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  if (rayIsActiveU(flags))
  {
    float4 data = in_hitPosNorm[tid];

    float3 hitPos      = to_float3(data);
    float3 hitNorm     = normalize(decodeNormal(as_int(data.w)));
    float2 hitTexCoord = in_hitTexCoord[tid];

    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mtlStorage, GetMaterialId(in_matData[tid]));

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float3 ray_pos = to_float3(a_rpos[tid]);
    float3 ray_dir = to_float3(a_rdir[tid]);

    TransparencyAndFog matFogAndTransp = materialEvalTransparencyAndFog(pHitMaterial, ray_dir, hitNorm, hitTexCoord, a_globals, a_shadingTexture);

    float4 newPathThroughput = a_thoroughput[tid] * to_float4(matFogAndTransp.transparency, 1);

    float offsetLength = 10.0f*fmax(fmax(fabs(hitPos.x), fmax(fabs(hitPos.y), fabs(hitPos.z))), GEPSILON)*GEPSILON;
    ray_pos = hitPos + ray_dir*offsetLength;
    a_rpos[tid] = to_float4(ray_pos, 0.0f);

    if (maxcomp(to_float3(newPathThroughput)) < 0.00001f)
      flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);

    a_flags[tid]       = flags;
    a_thoroughput[tid] = newPathThroughput;

  } //  if (rayIsActiveU(flags))
  
  
}


__kernel void TransparentShadowKenrel(__global const uint*     in_flags,
                                      __global const float4*   in_hitPosNorm,
                                      __global const float4*   a_data1,
                                      __global const float4*   a_data2,
                                      __global       ushort4*  a_shadow,
                                      __global       int*      a_transpL,
                                      __global const float*    a_hitPolySize,
                                      int a_size,

                                      
                                     #ifdef USE_1D_TEXTURES
                                       __read_only image1d_buffer_t  ak_inputBVH,
                                       __read_only image1d_buffer_t  ak_inputObjList,
                                     #else
                                       __global const float4* ak_inputBVH,
                                       __global const float4* ak_inputObjList,
                                     #endif

                                       __global const int*    a_vertIndices,           
                                       __global const float2* a_vertTexCoord,
                                       __global const float4* a_vertNorm,
                                        
                                       texture2d_t a_shadingTexture,       //
                                       __global const EngineGlobals* a_globals
                                      )
{

  __global const float4* a_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  uint flags = in_flags[tid];
  if (!rayIsActiveU(flags))
   return;

  const bool wasGlossyOrDiffuse = (unpackRayFlags(flags) & RAY_GRAMMAR_GLOSSY_REFLECTION) || (unpackRayFlags(flags) & RAY_GRAMMAR_DIFFUSE_REFLECTION);
  const uint rayBounceNum       = unpackBounceNum(flags);

  if (a_globals->g_flags & HRT_PT_PRIMARY_AND_REFLECTIONS)
  {
    if (wasGlossyOrDiffuse && rayBounceNum >= 1)
      return;
  }

  if (a_globals->g_flags & HRT_PT_SECONDARY_AND_GLOSSY)
  {
    if (!wasGlossyOrDiffuse)
      return;
  }

  // read transparency list size first
  //
  
  int listSize = a_transpL[tid*TRANSPARENCY_LIST_SIZE + 0];
  if (listSize <= 1 || listSize > TRANSPARENCY_LIST_SIZE)
  {
    if (listSize > TRANSPARENCY_LIST_SIZE)
      a_shadow[tid] = make_ushort4(0,0,0,0);
    return;
  }
  
  float3 shadow = decompressShadow(a_shadow[tid]);
  if (shadow.x + shadow.y + shadow.z < 1e-6f) // shadow is exactly 0
    return;
 

  // construct shadow ray
  //
  float4 data1 = a_data1[tid];
  float4 data2 = a_data2[tid];

  ShadowSample explicitSam;

  explicitSam.pos     = to_float3(data1);
  explicitSam.color   = to_float3(data2);
  explicitSam.pdf     = data1.w > 0 ? data1.w : 1.0f;
  explicitSam.maxDist = data2.w;
  explicitSam.isPoint = (data1.w <= 0);

  const float4 data    = in_hitPosNorm[tid];
  const float3 hitPos  = to_float3(data);
  const float3 hitNorm = normalize(decodeNormal(as_int(data.w)));

  const float  epsilon = fmax(maxcomp(hitPos), 1.0f)*GEPSILON;
  const float  polyEps = fmin(fmax(PEPSILON*a_hitPolySize[tid], epsilon), PG_SCALE*epsilon);

  const float3 shadowRayDir = normalize(explicitSam.pos - hitPos); // explicitSam.direction;
  const float  offsetSign   = (dot(shadowRayDir, hitNorm) >= 0.0f) ? 1.0f : -1.0f;
  const float3 shadowRayPos = hitPos + epsilon*shadowRayDir + polyEps*hitNorm*offsetSign;

  //
  //

  TransparencyShadowStepData stepData;
  stepData.currFogColor = make_float3(1, 1, 1);
  stepData.currFogMult  = 0.0f;
  stepData.currHitDist  = 0.0f;

  //a_shadow[tid] = compressShadow(make_float3(1,1,1));
  //return;

  float currDist = 0.0f;

  for (int i = 1; i < listSize; i++)
  {
    int triAddress = a_transpL[tid*TRANSPARENCY_LIST_SIZE + i];  
    if(triAddress < 0)
      break;

    // // ray tri intersect
  #ifdef USE_1D_TEXTURES
    float4 data1 = read_imagef(ak_inputObjList, triAddress + 0);
    float4 data2 = read_imagef(ak_inputObjList, triAddress + 1);
    float4 data3 = read_imagef(ak_inputObjList, triAddress + 2);
  #else
    float4 data1 = ak_inputObjList[triAddress + 0];
    float4 data2 = ak_inputObjList[triAddress + 1];
    float4 data3 = ak_inputObjList[triAddress + 2];
  #endif
    
    int offset     =  as_int(data3.w) & 0x3FFFFFFF; // '0011' -> 3 
    int alphaMatId = (as_int(data1.w) & ALPHA_MATERIAL_MASK);
    
    int offs_A = a_vertIndices[offset + 0];
    int offs_B = a_vertIndices[offset + 1];
    int offs_C = a_vertIndices[offset + 2];
    
    float3 A_pos = to_float3(data1);
    float3 B_pos = to_float3(data2);
    float3 C_pos = to_float3(data3);
    
    float2 A_tex = a_vertTexCoord[offs_A];
    float2 B_tex = a_vertTexCoord[offs_B];
    float2 C_tex = a_vertTexCoord[offs_C];
    
    float3 A_norm = to_float3(a_vertNorm[offs_A]);
    float3 B_norm = to_float3(a_vertNorm[offs_B]);
    float3 C_norm = to_float3(a_vertNorm[offs_C]);
    
    float3 uv            = triBaricentrics3(shadowRayPos, shadowRayDir, A_pos, B_pos, C_pos);
    float  hit_t         = uv.z;
    float2 texCoordS     = (1.0f - uv.x - uv.y)*A_tex + uv.y*B_tex + uv.x*C_tex;
    float3 shadowHitNorm = (1.0f - uv.x - uv.y)*A_norm + uv.y*B_norm + uv.x*C_norm;
    // \\

    // if (uv.x + uv.y > 1.0f)
      // continue;

    if (fabs(currDist - hit_t) > GEPSILON*hit_t) // a double hit bug
    {
      currDist = hit_t;
      shadow   = transparencyStep(&stepData, shadow, alphaMatId, hit_t, shadowRayDir, shadowHitNorm, texCoordS, a_globals, a_mtlStorage, a_shadingTexture);
    }
    
    if (fmax(shadow.x, fmax(shadow.y, shadow.z)) < 1e-5f)
      break;

  }

  // shadow = make_float3(0, 0, 0);

  a_shadow[tid] = compressShadow(shadow);

}


__kernel void ReadDiffuseColor(__global const float4*    a_rdir, 
                               __global const Lite_Hit*  in_hits,
                               __global const float4*    in_posNorm,
                               __global const float2*    in_texCoord,
                               __global const HitMatRef* in_matData,

                               texture2d_t                   a_shadingTexture,
                               __global const EngineGlobals* a_globals, 
                               __global float4* a_color, int iNumElements)
{
  __global const float4* a_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float3 color = make_float3(0, 0, 0);
  Lite_Hit hit = in_hits[tid];

  if (HitSome(hit))
  {
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mtlStorage, GetMaterialId(in_matData[tid]));

    float2 txcrd = in_texCoord[tid];
    float3 norm  = normalize(decodeNormal(as_int(in_posNorm[tid].w)));

    color = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), norm, txcrd, a_globals, a_shadingTexture);
  }

  a_color[tid] = to_float4(color, 0.0f);
}


__kernel void GetGBufferFirstBounce(__global const uint*      a_flags,
                                    __global const float4*    a_rdir, 
                                    __global const Lite_Hit*  in_hits, 
                                    __global const float4*    in_posNorm,
                                    __global const float2*    in_texCoord,
                                    __global const HitMatRef* in_matData,

                                    texture2d_t                   a_shadingTexture,
                                    __global const EngineGlobals* a_globals,
                                    __global float4*              a_color, 
                                    int iNumElements)
{
  __global const float4* a_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint     flags     = a_flags[tid];
  Lite_Hit hit       = in_hits[tid];
  float3   norm      = normalize(decodeNormal(as_int(in_posNorm[tid].w)));
  float3   diffColor = make_float3(0,0,0);
  int      matIndex  = -1;

  if (HitNone(hit))
  {
    norm = make_float3(0, 0, 0);
    hit.t = 1000000000.0f;
  }
  else
  {
    matIndex = GetMaterialId(in_matData[tid]);
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mtlStorage, matIndex);

    float2 txcrd = in_texCoord[tid];

    diffColor = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), norm, txcrd, a_globals, a_shadingTexture);
  }

  float alpha = (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE) ? 0.0f : 1.0f;

  GBuffer1 buffData;
  buffData.depth = hit.t;
  buffData.norm  = norm;
  buffData.matId = matIndex;
  buffData.rgba  = to_float4(diffColor, alpha);
  a_color[tid]   = packGBuffer1(buffData);
}

// change 12.12.2017 16:04;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "globals.h"
#include "cfetch.h"
#include "crandom.h"
#include "cmaterial.h"
#include "clight.h"
#include "cbidir.h"

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

__kernel void MakeEyeShadowRays(__global const uint*          restrict a_flags,
                                __global const float4*        restrict in_hitPosNorm,
                                __global const float4*        restrict in_normalsFull,
                                __global const HitMatRef*     restrict in_matData,

                                __global const float4*        restrict in_mtlStorage,
                                __global const EngineGlobals* restrict a_globals,
  
                                __global float4* restrict out_sraypos,
                                __global float4* restrict out_sraydir,
                                 int iNumElements, int haveMaterials)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  const float3 hitPos  = to_float3(in_hitPosNorm[tid]);
  const float3 hitNorm = to_float3(in_normalsFull[tid]); 
  
  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(hitPos, hitNorm, a_globals,
                                                                &camDir, &zDepth);

  float signOfNormal = 1.0f;

  if (haveMaterials == 1)
  {
    const int matId = GetMaterialId(in_matData[tid]);
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, matId);

    if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, hitNorm) < -0.01f)
      signOfNormal = -1.0f;
  }

  out_sraypos[tid] = to_float4(hitPos + epsilonOfPos(hitPos)*signOfNormal*hitNorm, zDepth); // OffsRayPos(hitPos, hitNorm, camDir);
  out_sraydir[tid] = to_float4(camDir, imageToSurfaceFactor);
}



__kernel void UpdateForwardPdfFor3Way(__global const uint*          restrict a_flags,
                                      __global const float4*        restrict in_raydir,
                                      __global const float4*        restrict in_raydirNext,

                                      __global const float4*        restrict in_hitPosNorm,
                                      __global const float4*        restrict in_normalsFull,
                                      __global const float2*        restrict in_hitTexCoord,
                                      __global const uint*          restrict in_flatNorm,
                                      __global const HitMatRef*     restrict in_matData,
                                      __global const Hit_Part4*     restrict in_hitTangent,

                                      __global const MisData*       restrict in_misDataCurr,
                                      __global PerRayAcc*           restrict a_pdfAcc,
                                      
                                      __global const float4*        restrict in_texStorage1,
                                      __global const float4*        restrict in_texStorage2,
                                      __global const float4*        restrict in_mtlStorage,
                                      __global const EngineGlobals* restrict a_globals,
                                      
                                      int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  const int a_currDepth    = unpackBounceNum(flags) + 1;

  const MisData mSamData   = in_misDataCurr[tid];
  const bool  isSpecular   = (mSamData.isSpecular != 0);
  const float matSamplePdf = mSamData.matSamplePdf;
  const float cosNext      = mSamData.cosThetaPrev; // because we have already updated mSamData.cosThetaPrev inside 'NextBounce' kernel

  const float3 hitNorm     = to_float3(in_normalsFull[tid]);
  const float3 ray_dir     = to_float3(in_raydir[tid]);
  const float cosCurr      = fabs(-dot(ray_dir, hitNorm));

  PerRayAcc accData = a_pdfAcc[tid];

  // eval reverse pdf
  //
  if (!isSpecular)
  {
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));
    const Hit_Part4 btanAndN = in_hitTangent[tid];

    ShadeContext sc;
    sc.wp = to_float3(in_hitPosNorm[tid]);
    sc.l  = (-1.0f)*ray_dir;
    sc.v  = (-1.0f)*to_float3(in_raydirNext[tid]);
    sc.n  = hitNorm;
    sc.fn = decodeNormal(in_flatNorm[tid]);
    sc.tg = decodeNormal(btanAndN.tangentCompressed);
    sc.bn = decodeNormal(btanAndN.bitangentCompressed);
    sc.tc = in_hitTexCoord[tid];

    const float pdfW = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ a_globals, in_texStorage1, in_texStorage2).pdfFwd;

    accData.pdfCameraWP *= (pdfW / fmax(cosCurr, DEPSILON));
    accData.pdfLightWP  *= (matSamplePdf / fmax(cosNext, DEPSILON));

    if (a_currDepth == 1)
      accData.pdfCamA0 *= (pdfW / fmax(cosCurr, DEPSILON)); // now pdfRevA0 will do store correct product pdfWP[0]*G[0] (if [0] means light)
  }
  else
  {
    accData.pdfCameraWP *= 1.0f;
    accData.pdfLightWP  *= 1.0f;
    //if (a_currDepth == 1)
    //  accData.pdfCameraWP = 0.0f;
  }

  a_pdfAcc[tid] = accData;
}


__kernel void ConnectToEyeKernel(__global const uint*          restrict a_flags,
                                 __global const float4*        restrict in_oraydir,
                                 __global const float4*        restrict in_sraydir,
                                 __global const ushort4*       restrict in_shadow,
                                 
                                 __global const float4*        restrict in_hitPosNorm,
                                 __global const float2*        restrict in_hitTexCoord,
                                 __global const uint*          restrict in_flatNorm,
                                 __global const HitMatRef*     restrict in_matData,
                                 __global const Hit_Part4*     restrict in_hitTangent,
                                 __global const float4*        restrict in_normalsFull,
                                 __global const PerRayAcc*     restrict in_pdfAcc,
                                 __global const int*           restrict in_lightId,
                                 __global const float4*        restrict in_lsam2,
                                 
                                 __global const float4*        restrict in_mtlStorage,
                                 __global const EngineGlobals* restrict a_globals,
                                 __global const float4*        restrict in_texStorage1,
                                 __global const float4*        restrict in_texStorage2,
                                 __constant ushort*            restrict a_mortonTable256,
                                 
                                 __global const float4*        restrict a_colorIn,
                                 __global float4*              restrict a_colorOut,
                                 __global int2*                restrict out_zind,

                                 //__global float4*              restrict a_debugOut,
                                 
                                 float mLightSubPathCount,
                                 int   a_currBounce,
                                 int   iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  //if (a_debugOut != 0)
    //a_debugOut[tid] = make_float4(0, 0, 0, 0);

  uint flags = a_flags[tid];
  if (!rayIsActiveU(flags))
  {
    if (out_zind != 0)
      out_zind[tid] = make_int2(0xFFFFFFFF, tid);
    a_colorOut[tid] = make_float4(0,0,0,as_float(0xFFFFFFFF));
    return;
  }

  const float3 ray_dir = (in_oraydir == 0) ? make_float3(0,0,0) : to_float3(in_oraydir[tid]);
  const float3 hitPos  = to_float3(in_hitPosNorm[tid]); //  
  const float3 hitNorm = to_float3(in_normalsFull[tid]);
  const float4 data2   = in_sraydir[tid];

  const float3 camDir              = to_float3(data2); // compute it in MakeEyeShadowRays kernel
  const float imageToSurfaceFactor = data2.w;          // compute it in MakeEyeShadowRays kernel

  float  signOfNormal = 1.0f;
  float  pdfRevW      = 1.0f;
  float3 colorConnect = make_float3(1,1,1);
  if(a_currBounce > 0) // if 0, this is light surface
  {
    const int matId = GetMaterialId(in_matData[tid]);
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, matId);
    if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, hitNorm) < -0.01f)
      signOfNormal = -1.0f;
      
    const Hit_Part4 btanAndN = in_hitTangent[tid];
    const float2 hitTexCoord = in_hitTexCoord[tid];
    const float3 flatN       = decodeNormal(in_flatNorm[tid]);

    ShadeContext sc;
    sc.wp = hitPos;
    sc.l  = camDir;           
    sc.v  = (-1.0f)*ray_dir;  
    sc.n  = hitNorm;
    sc.fn = flatN; 
    sc.tg = decodeNormal(btanAndN.tangentCompressed);
    sc.bn = decodeNormal(btanAndN.bitangentCompressed);
    sc.tc = hitTexCoord;
   
    BxDFResult matRes = materialEval(pHitMaterial, &sc, false, true, /* global data --> */ a_globals, in_texStorage1, in_texStorage2);
    colorConnect      = matRes.brdf + matRes.btdf; 
    pdfRevW           = matRes.pdfRev;
  }

  float misWeight = 1.0f;
  if ((a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) && (a_currBounce > 0))
  {
    const PerRayAcc accData = in_pdfAcc[tid];

    const float cosCurr = fabs(dot(ray_dir, hitNorm));
    const float pdfRevWP = pdfRevW / fmax(cosCurr, DEPSILON); // pdfW po pdfWP

    float pdfCamA0 = accData.pdfCamA0;
    if (a_currBounce == 1)
      pdfCamA0 *= pdfRevWP; // see pdfRevWP? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.

    const float cancelImplicitLightHitPdf = (1.0f / fmax(pdfCamA0, DEPSILON2));

    __global const PlainLight* pLight = lightAt(a_globals, in_lightId[tid]);
    const float lightPickProbFwd = lightPdfSelectFwd(pLight);
    const float lightPickProbRev = lightPdfSelectRev(pLight);

    const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
    const float lightPdfA  = in_lsam2[tid].w; //PerThread().pdfLightA0; // remember that we packed it in lsam2 inside 'LightSampleForwardKernel'

    const float pdfAccFwdA = 1.0f*accData.pdfLightWP*accData.pdfGTerm*(lightPdfA*lightPickProbFwd);
    const float pdfAccRevA = cameraPdfA * (pdfRevWP*accData.pdfCameraWP)*accData.pdfGTerm; // see pdfRevWP? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.
                                                                                           // we didn't eval reverse pdf yet. Imagine light ray hit surface and we immediately connect.
    const float pdfAccExpA = cameraPdfA * (pdfRevWP*accData.pdfCameraWP)*accData.pdfGTerm*(cancelImplicitLightHitPdf*(lightPdfA*lightPickProbRev));

    misWeight = misWeightHeuristic3(pdfAccFwdA, pdfAccRevA, pdfAccExpA);
  }
  

  // We divide the contribution by surfaceToImageFactor to convert the (already
  // divided) pdf from surface area to image plane area, w.r.t. which the
  // pixel integral is actually defined. We also divide by the number of samples
  // this technique makes, which is equal to the number of light sub-paths
  //
  const float3 a_accColor  = to_float3(a_colorIn[tid]);
  const float3 shadowColor = decompressShadow(in_shadow[tid]);
  float3 sampleColor       = misWeight*shadowColor*(a_accColor*colorConnect) * (imageToSurfaceFactor / mLightSubPathCount);
  
  if (!isfinite(sampleColor.x) || !isfinite(sampleColor.y) || !isfinite(sampleColor.z) || imageToSurfaceFactor <= 0.0f)
    sampleColor = make_float3(0, 0, 0);

  if(a_currBounce <= 0)
  {
    const int lightType = as_int(in_hitPosNorm[tid].w);
    if (lightType == PLAIN_LIGHT_TYPE_DIRECT || lightType == PLAIN_LIGHT_TYPE_SKY_DOME)
      sampleColor = make_float3(0, 0, 0);
  }

  int x = 65535, y = 65535;
  if (dot(sampleColor, sampleColor) > 1e-12f) // add final result to image
  {
    const float2 posScreenSpace = worldPosToScreenSpace(hitPos, a_globals);

    x = (int)(posScreenSpace.x);
    y = (int)(posScreenSpace.y);
  }

  const int zid = (int)ZIndex(x, y, a_mortonTable256);
  if(out_zind != 0)
    out_zind[tid] = make_int2(zid, tid);
  
  a_colorOut[tid] = to_float4(sampleColor, as_float(packXY1616(x,y)));

}


__kernel void HitEnvOrLightKernel(__global const float4*    restrict in_rpos,
                                  __global const float4*    restrict in_rdir,
                                  __global uint*            restrict a_flags,
                                  
                                  __global const float4*    restrict in_hitPosNorm,
                                  __global const float2*    restrict in_hitTexCoord,
                                  __global const HitMatRef* restrict in_matData,
                                  __global const Hit_Part4* restrict in_hitTangent,
                                  __global const float4*    restrict in_hitNormFull,
                                  
                                  __global float4*          restrict a_color,
                                  __global float4*          restrict a_thoroughput,
                                  __global MisData*         restrict a_misDataPrev,
                                  __global float4*          restrict out_emission,
                                  
                                  __global const MisData*   restrict in_misDataPrev,
                                  __global PerRayAcc*       restrict a_pdfAcc,
                                  __global PerRayAcc*       restrict a_pdfAccCopy,
                                  __global float*           restrict a_pdfCamA,
                                  
                                  __global const float4*    restrict in_texStorage1,
                                  __global const float4*    restrict in_texStorage2,
                                  __global const float4*    restrict in_mtlStorage,
                                  __global const float4*    restrict in_pdfStorage,
                                  __global const EngineGlobals*  restrict a_globals,
                                  
                                  __global const int*       restrict in_instLightInstId,
                                  __global const Lite_Hit*  restrict in_liteHit,
                                  float a_mLightSubPathCount, int a_currDepth, int iNumElements)
                                  //__global float4*          restrict a_debugf4)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const uint flags = a_flags[tid];

  //a_debugf4[tid] = make_float4(-1, -1, -1, -1);

  // if hit environment
  //
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE)
  {
    const float3 ray_pos = to_float3(in_rpos[tid]);
    const float3 ray_dir = to_float3(in_rdir[tid]);

    const int  hitId = hitDirectLight(ray_dir, a_globals);
    float3     nextPathColor = make_float3(0, 0, 0);

    if (hitId >= 0) // hit any sun light
    {
      __global const PlainLight* pLight = a_globals->suns + hitId;
      float3 lightColor = lightBaseColor(pLight)*directLightAttenuation(pLight, ray_pos);

      const float pdfW = directLightEvalPDF(pLight, ray_dir);

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////
      MisData misPrev = a_misDataPrev[tid];

      if ((unpackBounceNum(flags) > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0))) //#TODO: check this for bug with 2 hdr env light (test 335)
      {
        lightColor = make_float3(0, 0, 0);
      }
      else if (((misPrev.isSpecular == 1) && (a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS)) || (a_globals->g_flags & HRT_STUPID_PT_MODE))
        lightColor *= (1.0f / pdfW);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////

      const float3 pathThroughput = to_float3(a_thoroughput[tid]);
      nextPathColor = to_float3(a_color[tid]) + pathThroughput * lightColor;
    }
    else
    {
      int renderLayer = a_globals->varsI[HRT_RENDER_LAYER];

      float3 envColor = environmentColor(ray_dir, a_misDataPrev[tid], flags, a_globals, in_mtlStorage, in_pdfStorage, in_texStorage1);
      float3 pathThroughput = to_float3(a_thoroughput[tid]);

      nextPathColor = to_float3(a_color[tid]) + pathThroughput * envColor;
    }

    uint otherFlags    = unpackRayFlags(flags);
    a_flags      [tid] = packRayFlags(flags, RAY_IS_DEAD | (otherFlags & (~RAY_GRAMMAR_OUT_OF_SCENE)));
    a_color      [tid] = to_float4(nextPathColor, 0.0f);
    a_thoroughput[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  else if (rayIsActiveU(flags)) // if thread is active
  {
    float3 emissColor = make_float3(0, 0, 0);
    bool hitLightSource = false;

    const float3 hitPos = to_float3(in_hitPosNorm[tid]);
    float3 hitNorm = to_float3(in_hitNormFull[tid]); // or normalize(decodeNormal(as_int(data.w))) where data is in_hitPosNorm[tid];

    const float2 hitTexCoord = in_hitTexCoord[tid];
    const float3 ray_pos = to_float3(in_rpos[tid]);
    const float3 ray_dir = to_float3(in_rdir[tid]);

    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));

    // eval PDFs for 3WAY 
    //
    PerRayAcc accData;
    float GTerm = 1.0f, cosHere = 1.0f, cosPrev = 1.0f;
    if (a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS)
    {
      const float dist = length(ray_pos - hitPos);
      cosPrev = in_misDataPrev[tid].cosThetaPrev;
      cosHere = fabs(dot(ray_dir, hitNorm));
      GTerm = cosHere * cosPrev / fmax(dist*dist, DEPSILON2);
      accData = a_pdfAcc[tid]; // for 3 bounce we need to store (p0*G0)*(p1*G1) and do not include (p2*G2) to we could replace it with explicit strategy pdf

      if (a_currDepth == 0)
      {
        float3 camDirDummy; float zDepthDummy;
        const float imageToSurfaceFactor = CameraImageToSurfaceFactor(hitPos, hitNorm, a_globals,
                                                                      &camDirDummy, &zDepthDummy);

        const float cameraPdfA = imageToSurfaceFactor / a_mLightSubPathCount;
        a_pdfCamA[tid]         = cameraPdfA;
        a_pdfAccCopy[tid]      = accData;
      }
      else
      {
        const PerRayAcc prevData = accData;
        accData.pdfGTerm *= GTerm;
        a_pdfAcc[tid] = accData;
        a_pdfAccCopy[tid] = prevData;
      }
    }

    // now check if we hit light
    //
    bool hitEmissiveMaterialMLT = false;
    const bool skipPieceOfShit  = materialIsInvisLight(pHitMaterial) && isEyeRay(flags);
    const float3 emissionVal    = (pHitMaterial == 0) ? make_float3(0, 0, 0) : materialEvalEmission(pHitMaterial, ray_dir, hitNorm, hitTexCoord, a_globals, in_texStorage1, in_texStorage2);

    if (dot(emissionVal, emissionVal) > 1e-6f && !skipPieceOfShit)
    {
      if (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE)
        hitNorm = hitNorm * (-1.0f);

      const Lite_Hit liteHit = in_liteHit[tid];

      if (dot(ray_dir, hitNorm) < 0.0f)
      {
        emissColor = emissionVal;

        const int lightOffset = (a_globals->lightsNum == 0) ? -1 : in_instLightInstId[liteHit.instId];
        if (lightOffset >= 0)
        {
          MisData misPrev = a_misDataPrev[tid];

          __global const PlainLight* pLight = lightAt(a_globals, lightOffset);
          emissColor = lightGetIntensity(pLight, ray_pos, ray_dir, hitNorm, hitTexCoord, flags, misPrev, a_globals, in_texStorage1, in_pdfStorage);

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// \\\\\\\\\\\\\\\\\\\\\\
          
          if (a_globals->g_flags & HRT_ENABLE_MMLT)
          {
            const int bounceNum = unpackBounceNum(flags);

            if(bounceNum == 2)
              emissColor *= 1.0f; // #TODO: check if this is right bounce; and if it is not, kill emissColor
            else
              emissColor *= 0.0f;
          }
          else if (a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS)
          {
            const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, a_globals, in_texStorage1, in_pdfStorage);

            accData.pdfLightWP *= (lPdfFwd.pdfW / fmax(cosHere, DEPSILON));

            const float lightPdfA  = lPdfFwd.pdfA;
            const float cancelPrev = (misPrev.matSamplePdf / fmax(cosPrev, DEPSILON))*GTerm; // calcel previous pdfA 
            const float cameraPdfA = a_pdfCamA[tid];

            float pdfAccFwdA = 1.0f       * (accData.pdfLightWP *accData.pdfGTerm) * lightPdfA*lPdfFwd.pickProb;
            float pdfAccRevA = cameraPdfA * (accData.pdfCameraWP*accData.pdfGTerm);
            float pdfAccExpA = cameraPdfA * (accData.pdfCameraWP*accData.pdfGTerm)*(lightPdfA*lightPdfSelectRev(pLight) / fmax(cancelPrev, DEPSILON));

            if (a_currDepth == 0)
            {
              pdfAccFwdA = 0.0f;
              pdfAccRevA = 1.0f;
              pdfAccExpA = 0.0f;
            }
            else if (misPrev.isSpecular)
            {
              pdfAccExpA = 0.0f; // comment this to kill SDS caustics.
            }

            //a_debugf4[tid] = make_float4(pdfAccFwdA, pdfAccRevA, pdfAccExpA, 0);

            const float misWeight = misWeightHeuristic3(pdfAccRevA, pdfAccFwdA, pdfAccExpA);
            emissColor *= misWeight;
          }
          else if (unpackBounceNum(flags) > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0)) // old MIS weights via pdfW
          {
            const float lgtPdf    = lightPdfSelectRev(pLight)*lightEvalPDF(pLight, ray_pos, ray_dir, hitPos, hitNorm, hitTexCoord, in_pdfStorage, a_globals);
            const float bsdfPdf   = misPrev.matSamplePdf;
            const float misWeight = misWeightHeuristic(bsdfPdf, lgtPdf); // (bsdfPdf*bsdfPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
            emissColor *= misWeight;
          }

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// \\\\\\\\\\\\\\\\\\\\\\
            
          if (unpackBounceNum(flags) > 0)
            emissColor = clamp(emissColor, 0.0f, a_globals->varsF[HRT_BSDF_CLAMPING]);

          if (misPrev.prevMaterialOffset >= 0)
          {
            __global const PlainMaterial* pPrevMaterial = materialAtOffset(in_mtlStorage, misPrev.prevMaterialOffset);

            bool disableCaustics = (unpackBounceNumDiff(flags) > 0) && !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS) && materialCastCaustics(pPrevMaterial); // and prev material cast caustics
            if (disableCaustics)
              emissColor = make_float3(0, 0, 0);
          }

          hitLightSource = true; // kill thread next if it hit real light source
        }
        else // hit emissive material, not a light 
        {
          
        }
      }

      // make lights black for 'LAYER_INCOMING_RADIANCE'
      const uint rayBounceNum = unpackBounceNum(flags);
      const uint rayBounceNumDiff = unpackBounceNumDiff(flags);

      if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_FORBID_EMISSIVE_GI) && rayBounceNumDiff > 0)
        emissColor = make_float3(0, 0, 0);

      const int packedIsLight = hitLightSource ? 1 : 0;
      out_emission[tid] = to_float4(emissColor, as_float(packedIsLight));

    } // \\ if light emission is not zero
    else
    {
      out_emission[tid] = make_float4(0, 0, 0, as_float(0));
    }


  } // \\ else if thread is active

}


__kernel void Shade(__global const float4*    restrict a_rpos,
                    __global const float4*    restrict a_rdir,
                    __global       uint*      restrict a_flags,
                    
                    __global const float4*    restrict in_hitPosNorm,
                    __global const float2*    restrict in_hitTexCoord,
                    __global const uint*      restrict in_flatNorm,
                    __global const HitMatRef* restrict in_matData,
                    __global const Hit_Part4* restrict in_hitTangent,
                    
                    __global const float4*    restrict in_data1,
                    __global const float4*    restrict in_data2,
                    __global const ushort4*   restrict in_shadow,
                    __global const float*     restrict in_lightPickProb,
                    __global const float*     restrict in_lcos,
  
                    __global const float4*    restrict in_normalsFull,

                    __global const PerRayAcc* restrict in_pdfAccPrev,
                    __global const float4*    restrict in_rayDirAndLightId,
                    __global const float*     restrict in_pdfCamA,

                    __global float4*          restrict out_color,
                    __global uchar*           restrict out_shadow,
                     
                    __global const float4*    restrict in_texStorage1,
                    __global const float4*    restrict in_texStorage2,
                    __global const float4*    restrict in_mtlStorage,
                    __global const float4*    restrict in_pdfStorage,
                    __global const EngineGlobals* restrict a_globals,
                    int iNumElements)
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const uint flags        = a_flags[tid];
  const uint rayBounceNum = unpackBounceNum(flags);

  if (!rayIsActiveU(flags))
  {
    if(out_shadow != 0 && rayBounceNum == 0)
      out_shadow[tid] = 0;
    return;
  }

  if (a_globals->lightsNum == 0)
  {
    out_color[tid] = make_float4(0,0,0,0);
    return;
  }


  // read surfaceHit
  //
  const Hit_Part4 btanAndN = in_hitTangent[tid];
  const float3 hitPos      = to_float3(in_hitPosNorm[tid]);
  const float3 hitNorm     = to_float3(in_normalsFull[tid]); // normalize(decodeNormal(as_int(data.w)));
  const float2 hitTexCoord = in_hitTexCoord[tid];
  const float3 hitBiTang   = decodeNormal(btanAndN.tangentCompressed);
  const float3 hitBiNorm   = decodeNormal(btanAndN.bitangentCompressed);

  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));

  float3 ray_pos = to_float3(a_rpos[tid]);
  float3 ray_dir = to_float3(a_rdir[tid]);

  float4 data1   = in_data1[tid];
  float4 data2   = in_data2[tid];
  float3 flatN   = decodeNormal(in_flatNorm[tid]);

  ShadowSample explicitSam;

  explicitSam.pos        = to_float3(data1);
  explicitSam.color      = to_float3(data2);
  explicitSam.pdf        = fabs(data1.w);
  explicitSam.maxDist    = data2.w;
  explicitSam.isPoint    = (data1.w <= 0);
  explicitSam.cosAtLight = in_lcos[tid];

  //float3 shadowRayPos = hitPos + hitNorm*maxcomp(hitPos)*GEPSILON;
  float3 shadowRayDir = normalize(explicitSam.pos - hitPos); 

  const bool currLightCastCaustics = (a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS);
  const bool disableCaustics       = (unpackBounceNumDiff(flags) > 0) && !currLightCastCaustics;
  
  if ((a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_PRIMARY) && (a_globals->varsI[HRT_RENDER_LAYER_DEPTH] == unpackBounceNum(flags))) //////////////////////////////////////////////////////////////////
    pHitMaterial = materialAtOffset(in_mtlStorage, a_globals->varsI[HRT_WHITE_DIFFUSE_OFFSET]);

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

  const BxDFResult evalData = materialEval(pHitMaterial, &sc, disableCaustics, false, /* global data --> */ a_globals, in_texStorage1, in_texStorage2);
  const float3 shadow       = decompressShadow(in_shadow[tid]);
  const float lightPickProb = in_lightPickProb[tid];

  const int lightOffset = as_int(in_rayDirAndLightId[tid].w);

  __global const PlainLight* pLight = lightAt(a_globals, lightOffset);

  float cosThetaOutAux = 1.0f;

  float misWeight = 1.0f;
  if ((a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) && (lightType(pLight) != PLAIN_LIGHT_TYPE_SKY_DOME))
  {
    const float cosHere      = fabs(dot(ray_dir, hitNorm));
    const int a_currDepth    = rayBounceNum;

    const float cosThetaOut1 = fmax(+dot(shadowRayDir, hitNorm), 0.0f);
    const float cosThetaOut2 = fmax(-dot(shadowRayDir, hitNorm), 0.0f);
    const bool  underSurface = (dot(evalData.btdf, evalData.btdf)*cosThetaOut2 > 0.0f && dot(evalData.brdf, evalData.brdf)*cosThetaOut1 <= 0.0f);
    const float cosThetaOut  = underSurface ? cosThetaOut2 : cosThetaOut1;
    const float cosAtLight   = fmax(explicitSam.cosAtLight, 0.0f);
    
    cosThetaOutAux = cosThetaOut1;

    const float bsdfRevWP    = (evalData.pdfFwd == 0.0f) ? 1.0f : evalData.pdfFwd / fmax(cosThetaOut, DEPSILON);
    const float bsdfFwdWP    = (evalData.pdfRev == 0.0f) ? 1.0f : evalData.pdfRev / fmax(cosHere, DEPSILON);
    const float shadowDist   = length(hitPos - explicitSam.pos);
    const float GTermShadow  = cosThetaOut * cosAtLight / fmax(shadowDist*shadowDist, DEPSILON);
    
    const PerRayAcc prevData = in_pdfAccPrev[tid];

    const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, shadowRayDir, cosAtLight, a_globals, in_texStorage1, in_pdfStorage);
     
    float pdfFwdWP1 = 1.0f;  // madness of IBPT. 
    if (a_currDepth > 0)     // Imagine ray that hit light source after (second?) bounce (or first ?). pdfAccFwdA = pdfLightA*PdfLightW*GTermShadow.
      pdfFwdWP1 = bsdfFwdWP; // 
     
    const float cameraPdfA = in_pdfCamA[tid];
     
    float pdfAccFwdA = pdfFwdWP1  * (prevData.pdfLightWP *prevData.pdfGTerm)*((lPdfFwd.pdfW / fmax(cosAtLight, DEPSILON))*GTermShadow)*(lPdfFwd.pdfA*lPdfFwd.pickProb);
    float pdfAccRevA = cameraPdfA * (prevData.pdfCameraWP*prevData.pdfGTerm)*bsdfRevWP*GTermShadow;
    float pdfAccExpA = cameraPdfA * (prevData.pdfCameraWP*prevData.pdfGTerm)*(lPdfFwd.pdfA*lightPickProb);
    if (explicitSam.isPoint)
      pdfAccRevA = 0.0f;

    misWeight = misWeightHeuristic3(pdfAccExpA, pdfAccRevA, pdfAccFwdA);
  }
  else
  {
    const float lgtPdf = explicitSam.pdf*lightPickProb;
    cosThetaOutAux     = fmax(+dot(shadowRayDir, hitNorm), 0.0f);

    misWeight = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0f;
  }

  if (out_shadow != 0 && rayBounceNum == 0)
  {
    if (materialGetType(pHitMaterial) == PLAIN_MAT_CLASS_SHADOW_MATTE && cosThetaOutAux > 1e-5f)
      cosThetaOutAux = 1.0f;
    const float shadow1 = cosThetaOutAux*256.0f*0.33333f*(shadow.x + shadow.y + shadow.z);
    out_shadow[tid]     = (uchar)(255.0f - clamp(shadow1, 0.0f, 255.0f));
  }

  const float cosThetaOut1 = fmax(+dot(shadowRayDir, hitNorm), 0.0f);
  const float cosThetaOut2 = fmax(-dot(shadowRayDir, hitNorm), 0.0f);

  const float3 bxdfVal = (evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2);

  float3 shadeColor = (explicitSam.color * (1.0f / fmax(explicitSam.pdf, DEPSILON)))*bxdfVal*misWeight*shadow; 

  if (unpackBounceNum(flags) > 0)
    shadeColor = clamp(shadeColor, 0.0f, a_globals->varsF[HRT_BSDF_CLAMPING]);

  float maxColorLight = fmax(explicitSam.color.x, fmax(explicitSam.color.y, explicitSam.color.z));
  float maxColorShade = fmax(shadeColor.x, fmax(shadeColor.y, shadeColor.z));

  float fVisiableLightsNum = 1.0f / lightPickProb;
  float fShadowSamples = 1.0f;
  shadeColor *= (fVisiableLightsNum / fShadowSamples);

  // (1) save shaded color 
  //
  out_color [tid] = to_float4(shadeColor, lightPickProb);

  // // (2) signal that we shade from the ground for shadow matte case
  // //
  // {
  //   int otherFlags = unpackRayFlags(flags);
  //   if (cosThetaOutAux <= 0.0f)
  //     otherFlags = otherFlags | RAY_SHADE_FROM_OTHER_SIDE;
  //   else
  //     otherFlags = otherFlags & (~RAY_SHADE_FROM_OTHER_SIDE);
  // 
  //   if (lightType(pLight) == PLAIN_LIGHT_TYPE_SKY_DOME)
  //     otherFlags = otherFlags | RAY_SHADE_FROM_SKY_LIGHT;
  //   else
  //     otherFlags = otherFlags & (~RAY_SHADE_FROM_SKY_LIGHT);
  // 
  //   a_flags[tid] = packRayFlags(flags, otherFlags);
  // }
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
                         __global const float4*    restrict in_emissionColor,
                         __global PerRayAcc*       restrict a_pdfAcc,        // used only by 3-Way PT/LT passes
                         __global float*           restrict a_camPdfA,       // used only by 3-Way PT/LT passes

                         __global const float4*    restrict in_texStorage1,    
                         __global const float4*    restrict in_texStorage2,
                         __global const float4*    restrict in_mtlStorage,
                         __global const float4*    restrict in_pdfStorage,   //
                         
                         int iNumElements,
                         __global const EngineGlobals*  restrict a_globals)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];

  float3 ray_pos = to_float3(a_rpos[tid]);
  float3 ray_dir = to_float3(a_rdir[tid]);

  __global const float* pssVec = 0;

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
  
    pHitMaterial = materialAt(a_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));
  
    float4 emissData = (in_emissionColor == 0) ? make_float4(0, 0, 0, 0) : in_emissionColor[tid];
    outPathColor     = to_float3(emissData);
    hitLightSource   = (emissData.w == 1);
  }
  

  MatSample brdfSample;
  int    matOffset   = 0;
  bool   isThinGlass = false;

  RandomGen gen  = out_gens[tid];
  gen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

  float GTerm = 1.0f;
  float3 oldRayDir;
 
  if (rayIsActiveU(flags))
  {
    matOffset = materialOffset(a_globals, GetMaterialId(in_matData[tid]));
  
    BRDFSelector mixSelector = materialRandomWalkBRDF(pHitMaterial, &gen, pssVec, ray_dir, hitNorm, hitTexCoord, a_globals, in_texStorage1, unpackBounceNum(flags), false, false);
  
    const  uint rayBounceNum = unpackBounceNum(flags);
   
    matOffset           = matOffset    + mixSelector.localOffs*(sizeof(PlainMaterial)/sizeof(float4));
    pHitMaterial        = pHitMaterial + mixSelector.localOffs;
    thisBounceIsDiffuse = materialHasDiffuse(pHitMaterial);

    const float3 shadow = decompressShadow(a_shadow[tid]); // fmax(-dot(ray_dir, hitNorm), 0.0f);
  
    /////////////////////////////////////////////////////////////////////////////// begin sample material
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
  
      const float3 randsm = rndMat(&gen, pssVec, unpackBounceNum(flags));
  
      MaterialLeafSampleAndEvalBRDF(pHitMaterial, randsm, &sc, shadow, a_globals, in_texStorage1, in_texStorage2,
                                    &brdfSample);

      isThinGlass = isPureSpecular(brdfSample) && (rayBounceNum > 0) && !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS) && materialIsThinGlass(pHitMaterial);
	  //isThinGlass = isPureSpecular(brdfSample) && (rayBounceNum > 0) && materialIsThinGlass(pHitMaterial);
    }

    /////////////////////////////////////////////////////////////////////////////// end   sample material
  
    const float selectorPdf = mixSelector.w;
    const float invPdf      = 1.0f / fmax(brdfSample.pdf*selectorPdf, DEPSILON);
    const float cosTheta    = fabs(dot(brdfSample.direction, hitNorm));

    outPathThroughput = clamp(cosTheta*brdfSample.color*invPdf, 0.0f, 1.0f); //#TODO: this is not correct actually !!!
   
    // const int otherFlagsSMSK = unpackRayFlags(flags);
    // if ((materialGetType(pHitMaterial) == PLAIN_MAT_CLASS_SHADOW_MATTE) && (otherFlagsSMSK & RAY_SHADE_FROM_SKY_LIGHT)) // shadow matte hack to sample only top hemisphere
    // {
    //   if (otherFlagsSMSK & RAY_SHADE_FROM_OTHER_SIDE)
    //     outPathThroughput *= 0.0f;
    //   else
    //     outPathThroughput *= 2.0f;
    // }

    if (!isfinite(outPathThroughput.x)) outPathThroughput.x = 0.0f;
    if (!isfinite(outPathThroughput.y)) outPathThroughput.y = 0.0f;
    if (!isfinite(outPathThroughput.z)) outPathThroughput.z = 0.0f;

    /////////////////////////////////////////////////////////////////////////////// finish with outPathThroughput
  
    const float3 nextRay_dir = brdfSample.direction;
    const float3 nextRay_pos = OffsRayPos(hitPos, hitNorm, brdfSample.direction);

    // values that bidirectional techniques needs
    //
    const float cosPrev = fabs(a_misDataPrev[tid].cosThetaPrev);
    const float cosCurr = fabs(-dot(ray_dir, hitNorm));
    //const float cosNext = fabs(+dot(nextRay_dir, hitNorm));
    const float dist    = length(hitPos - ray_pos);
    GTerm = (cosPrev*cosCurr / fmax(dist*dist, DEPSILON2));

    // calc new ray
    //    
    oldRayDir = ray_dir;
    ray_dir   = nextRay_dir;
    ray_pos   = nextRay_pos;
  }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////// begin russian roulette
  const float pabsorb = probabilityAbsorbRR(flags, a_globals->g_flags);

  float rrChoice = 0.0f;
  if (pabsorb > 0.0f)
    rrChoice = rndFloat1_Pseudo(&gen);

  out_gens[tid] = gen;

  if (pabsorb >= 0.1f)
  {
    if (rrChoice < pabsorb)
      outPathThroughput = make_float3(0.0f, 0.0f, 0.0f);
    else
      outPathThroughput = outPathThroughput * (1.0f / (1.0f - pabsorb));
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////// end   russian roulette

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

    const uint rayBounceNum  = unpackBounceNum(flags);
    const uint diffBounceNum = unpackBounceNumDiff(flags);

    float4 shadeData  = in_shadeColor[tid];

    oldPathThroughput = to_float3(a_thoroughput[tid])*fogAtten;
    newPathThroughput = oldPathThroughput*outPathThroughput;
   
    ///////////////////////////////////////////////// #NOTE: OK, THIS SEEMS TO WORK FINE; JUST CHECK IT WITH WINDOW GLASS WHEN IMPLEMENT TRANSPARENT SHADOWS;
    if (!isThinGlass)  
    {
      MisData misNext;  
      misNext.matSamplePdf       = brdfSample.pdf;
      misNext.isSpecular         = (int)isPureSpecular(brdfSample);
      misNext.prevMaterialOffset = matOffset;
      misNext.cosThetaPrev       = fabs(+dot(ray_dir, hitNorm)); // update it withCosNextActually ...
      a_misDataPrev[tid]         = misNext;
    }
    ///////////////////////////////////////////////// 

    flags = flagsNextBounce(flags, brdfSample, a_globals);

    float4 nextPathColor;

    if (a_globals->g_flags & HRT_ENABLE_MMLT)
    {
      nextPathColor   = a_color[tid] + to_float4(oldPathThroughput*outPathColor, 0.0f); //#TODO: this works only for stupid PT and emissive color !!!!
      nextPathColor.w = 1.0f;
    }
    else if (a_globals->g_flags & HRT_FORWARD_TRACING) // don't add color from shade pass, this is simple case
    { 
      nextPathColor   = a_color[tid]*to_float4(outPathThroughput*fogAtten, 1.0f);
      nextPathColor.w = 1.0f;

      if (a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS)
      {
        PerRayAcc accPdf = a_pdfAcc[tid];
        {
          const int currDepth = rayBounceNum + 1;
          accPdf.pdfGTerm *= GTerm;
          if (currDepth == 1)
            accPdf.pdfCamA0 = GTerm; // spetial case, multiply it by pdf later ... 
        }
        a_pdfAcc[tid] = accPdf;
      }
    }
    else                                                                                     // more complex case, need to add color from shade on each bounce
    { 
      outPathColor   += to_float3(shadeData);
      nextPathColor   = a_color[tid] + to_float4(oldPathThroughput*outPathColor, 0.0f);
      nextPathColor.w = 1.0f;

      if (a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) 
      {
        const int a_currDepth = rayBounceNum;
        PerRayAcc accPdf      = a_pdfAcc[tid];          // #TODO: refactor code inside brackets; try to make procedure call
        {
          if (!isPureSpecular(brdfSample))
          {
            const float cosHere = fabs(dot(oldRayDir, hitNorm));
            const float cosNext = fabs(dot(brdfSample.direction, hitNorm));

            accPdf.pdfCameraWP *= (brdfSample.pdf / fmax(cosNext, DEPSILON));

            if (a_currDepth > 0)
            {
              ShadeContext sc;
              sc.wp = hitPos;
              sc.l  = (-1.0f)*oldRayDir;      // fliped; if compare to normal PT
              sc.v  = brdfSample.direction; // fliped; if compare to normal PT
              sc.n  = hitNorm;
              sc.fn = flatNorm;
              sc.tg = hitBiTang;
              sc.bn = hitBiNorm;
              sc.tc = hitTexCoord;

              const float pdfFwdW = materialEval(pHitMaterial, &sc, false, false, /* global data --> */  a_globals, in_texStorage1, in_texStorage2).pdfFwd;
             
              accPdf.pdfLightWP *= (pdfFwdW / fmax(cosHere, DEPSILON));
            }
          }
          else
          {
            accPdf.pdfCameraWP *= 1.0f; // in the case of specular bounce pdfFwd = pdfRev = 1.0f;
            accPdf.pdfLightWP *= 1.0f;  //
            if (a_currDepth == 0)
              accPdf.pdfLightWP = 0.0f;
          }
        }
        a_pdfAcc[tid] = accPdf;
      }
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void NextTransparentBounce(__global   float4*    a_rpos,
                                    __global   float4*    a_rdir,
                                    __global   uint*      a_flags,
                                    
                                    __global float4*      in_hitPosNorm,
                                    __global float2*      in_hitTexCoord,
                                    __global HitMatRef*   in_matData,
                                    
                                    __global float4*              a_thoroughput,  
                                    __global const float4*        in_mtlStorage,
                                    texture2d_t                   in_shadingTexture,    
                                    __global const EngineGlobals* in_globals,
                                    int iNumElements)
{

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];
  
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE) // if hit environment
  {
    uint otherFlags    = unpackRayFlags(flags);
    a_flags[tid]       = packRayFlags(flags, (otherFlags & (~RAY_GRAMMAR_OUT_OF_SCENE)) | RAY_IS_DEAD); // disable RAY_GRAMMAR_OUT_OF_SCENE, write flags;
  }
  else if (rayIsActiveU(flags))
  {
    const float4 data = in_hitPosNorm[tid];
   
    const float3 hitPos      = to_float3(data);
    const float3 hitNorm     = normalize(decodeNormal(as_int(data.w)));
    const float2 hitTexCoord = in_hitTexCoord[tid];

    __global const PlainMaterial* pHitMaterial = materialAt(in_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (pHitMaterial != 0)
    {
      const float3 ray_pos = to_float3(a_rpos[tid]);
      const float3 ray_dir = to_float3(a_rdir[tid]);

      TransparencyAndFog matFogAndTransp = materialEvalTransparencyAndFog(pHitMaterial, ray_dir, hitNorm, hitTexCoord, in_globals, in_shadingTexture);

      float4 newPathThroughput = a_thoroughput[tid] * to_float4(matFogAndTransp.transparency, 1.0f);

      const float3 nextRay_pos = OffsRayPos(hitPos, hitNorm, ray_dir);

      if (maxcomp(to_float3(newPathThroughput)) < 0.00001f)
        flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);

      a_rpos [tid] = to_float4(nextRay_pos, 0.0f);
      a_flags[tid] = flags;
      a_thoroughput[tid] = newPathThroughput;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

  __global const float4* in_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  uint flags = in_flags[tid];
  if (!rayIsActiveU(flags))
   return;

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
      shadow   = transparencyStep(&stepData, shadow, alphaMatId, hit_t, shadowRayDir, shadowHitNorm, texCoordS, a_globals, in_mtlStorage, a_shadingTexture);
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
  __global const float4* in_mtlStorage = 0; // #TODO: fix

  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  float3 color = make_float3(0, 0, 0);
  Lite_Hit hit = in_hits[tid];

  if (HitSome(hit))
  {
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, GetMaterialId(in_matData[tid]));

    float2 txcrd = in_texCoord[tid];
    float3 norm  = normalize(decodeNormal(as_int(in_posNorm[tid].w)));

    color = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), norm, txcrd, a_globals, a_shadingTexture);
  }

  a_color[tid] = to_float4(color, 0.0f);
}


__kernel void GetGBufferSample(__global const float4*    a_rdir,
                               __global const Lite_Hit*  restrict in_hits,
                               __global const uint*      restrict in_flags,
                               __global const float4*    restrict in_hitPosNorm,
                               __global const float2*    restrict in_hitTexCoord,
                               __global const HitMatRef* restrict in_matData,
                               
                               __global float4*          restrict out_gbuff1,
                               __global float4*          restrict out_gbuff2,

                               __global const HitMatRef*     restrict in_mtlStorage,
                               texture2d_t                   restrict a_shadingTexture,
                               __global const EngineGlobals* restrict a_globals,
                               int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  __local GBufferAll samples[GBUFFER_SAMPLES];
 
  const float4 hitPosNorm = in_hitPosNorm[tid];
  const Lite_Hit liteHit  = in_hits[tid];
  const uint flags        = in_flags[tid];

  if (rayIsActiveU(flags))
  {
    const int materialId = GetMaterialId(in_matData[tid]);
    const float2 txcrd   = in_hitTexCoord[tid];
    const float3 normal  = normalize(decodeNormal(as_int(hitPosNorm.w)));

    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, materialId);
    const float3 diffColor = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), normal, txcrd, a_globals, a_shadingTexture);

    samples[LOCAL_ID_X].data1.depth    = liteHit.t;
    samples[LOCAL_ID_X].data1.norm     = normal;
    samples[LOCAL_ID_X].data1.rgba     = to_float4(diffColor, 0.0f);
    samples[LOCAL_ID_X].data1.matId    = materialId;
    samples[LOCAL_ID_X].data1.coverage = 0.0f;

    samples[LOCAL_ID_X].data2.texCoord = txcrd;
    samples[LOCAL_ID_X].data2.objId    = liteHit.geomId;
    samples[LOCAL_ID_X].data2.instId   = liteHit.instId;
  }
  else
  {
    samples[LOCAL_ID_X].data1.depth    = 1e+6f;
    samples[LOCAL_ID_X].data1.norm     = make_float3(0,0,0);
    samples[LOCAL_ID_X].data1.rgba     = make_float4(0, 0, 0, 1);
    samples[LOCAL_ID_X].data1.matId    = -1;
    samples[LOCAL_ID_X].data1.coverage = 0.0f;

    samples[LOCAL_ID_X].data2.texCoord = make_float2(0,0);
    samples[LOCAL_ID_X].data2.objId    = -1;
    samples[LOCAL_ID_X].data2.instId   = -1;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const float a_fov    = a_globals->varsF[HRT_FOV_X];
  const float a_width  = a_globals->varsF[HRT_WIDTH_F];
  const float a_height = a_globals->varsF[HRT_HEIGHT_F];

  // now find the biggest cluster and take it's sample as the result;
  //
  if (LOCAL_ID_X == 0)
  {
    float minDiff   = 100000000.0f;
    int   minDiffId = 0;

    for (int i = 0; i < GBUFFER_SAMPLES; i++)
    {
      float diff = 0.0f;
      float coverage = 0.0f;
      for (int j = 0; j < GBUFFER_SAMPLES; j++)
      {
        const float thisDiff = gbuffDiff(samples[i], samples[j], a_fov, a_width, a_height);
        diff += thisDiff;
        if (thisDiff < 1.0f)
          coverage += 1.0f;
      }

      coverage *= (1.0f / (float)GBUFFER_SAMPLES);
      samples[i].data1.coverage = coverage;

      if (diff < minDiff)
      {
        minDiff   = diff;
        minDiffId = i;
      }
    }

    out_gbuff1[tid / GBUFFER_SAMPLES] = packGBuffer1(samples[minDiffId].data1);
    out_gbuff2[tid / GBUFFER_SAMPLES] = packGBuffer2(samples[minDiffId].data2);
  }

}

__kernel void PutAlphaToGBuffer(__global const float4* restrict in_thoroughput,
                                __global float4*       restrict inout_gbuff1,
                                int a_size)
{
  int tid = GLOBAL_ID_X;
  if (tid >= a_size)
    return;

  const float4 thoroughput = in_thoroughput[tid];

  const float opacity = fmax(thoroughput.x, fmax(thoroughput.y, thoroughput.z));
  const float alpha   = 1.0f - clamp(opacity, 0.0f, 1.0f);

  ///////////////////////////////////////////////////////////////////////////////////////
  __local float sArray[GBUFFER_SAMPLES];
  sArray[LOCAL_ID_X] = alpha;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint c = GBUFFER_SAMPLES / 2; c>0; c /= 2)
  {
    if (LOCAL_ID_X < c)
      sArray[LOCAL_ID_X] += sArray[LOCAL_ID_X + c];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  ///////////////////////////////////////////////////////////////////////////////////////

  if (LOCAL_ID_X == 0)
  {
    const int bid     = tid / GBUFFER_SAMPLES;
    GBuffer1 data1    = unpackGBuffer1(inout_gbuff1[bid]);
    data1.rgba.w      = sArray[0]*(1.0f/ (float)GBUFFER_SAMPLES);
    inout_gbuff1[bid] = packGBuffer1(data1);
  }
}



// change 31.01.2018 15:20;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

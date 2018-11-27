#include "cglobals.h"
#include "cfetch.h"
#include "crandom.h"
#include "cmaterial.h"
#include "clight.h"
#include "cbidir.h"

__kernel void MakeEyeShadowRays(__global const uint*          restrict a_flags,
                                __global const float4*        restrict in_surfaceHit,

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

  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &sHit);
  
  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(sHit.pos, sHit.normal, a_globals,
                                                                &camDir, &zDepth);

  // const int s1 = dot(camDir, sHit.normal) < 0.0f ? -1 : 1; // note that both flatNorm and sHit.normal are already fliped if they needed; so dot(camDir, sHit.normal) < -0.01f is enough
  // const int s2 = dot(rayDir, sHit.normal) < 0.0f ? -1 : 1; // note that both flatNorm and sHit.normal are already fliped if they needed; so dot(camDir, sHit.normal) < -0.01f is enough

  float signOfNormal = 1.0f;

  if (haveMaterials == 1)
  {
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, sHit.matId);
    if(pHitMaterial != 0)
    {
      if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, sHit.normal) < -0.01f)
        signOfNormal *= -1.0f;
    }
  }

  out_sraypos[tid] = to_float4(sHit.pos + epsilonOfPos(sHit.pos)*signOfNormal*sHit.normal, zDepth); // OffsRayPos(sHit.pos, sHit.normal, camDir);
  out_sraydir[tid] = to_float4(camDir, as_float(-1));

  //#TODO: accelarate shadow trace by reverse shadow dir !!!

  //const float3 camPos = sHit.pos + epsilonOfPos(sHit.pos)*signOfNormal*sHit.normal*zDepth;
  //
  //out_sraypos[tid] = to_float4(camPos, zDepth); 
  //out_sraydir[tid] = to_float4(camDir*(-1.0f), as_float(-1));
}


#define BUGGY_AMD_IBPT_PROCTEX_FETCH

__kernel void UpdateForwardPdfFor3Way(__global const uint*          restrict a_flags,
                                      __global const float4*        restrict in_raydir,
                                      __global const float4*        restrict in_raydirNext,

                                      __global const float4*        restrict in_surfaceHit,
                                      __global const float4*        restrict in_procTexData,

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
  
  SurfaceHit sHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &sHit);

  const float3 ray_dir = to_float3(in_raydir[tid]);
  const float cosCurr  = fabs(-dot(ray_dir, sHit.normal));

  PerRayAcc accData = a_pdfAcc[tid];

  // eval reverse pdf
  //
  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, sHit.matId);
  if (!isSpecular && pHitMaterial != 0)
  {
    ShadeContext sc;
    sc.wp  = sHit.pos;
    sc.l   = (-1.0f)*ray_dir;
    sc.v   = (-1.0f)*to_float3(in_raydirNext[tid]);
    sc.n   = sHit.normal;
    sc.fn  = sHit.flatNormal;
    sc.tg  = sHit.tangent;
    sc.bn  = sHit.biTangent;
    sc.tc  = sHit.texCoord;
    sc.hfi = false;

    ProcTextureList ptl;        
    InitProcTextureList(&ptl);  

    #ifndef BUGGY_AMD_IBPT_PROCTEX_FETCH
    ReadProcTextureList(in_procTexData, tid, iNumElements, 
                        &ptl);
    #endif
    const float pdfW = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), /* global data --> */ a_globals, in_texStorage1, in_texStorage2, &ptl).pdfFwd;
    
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
                                 __global const float4*        restrict in_surfaceHit,

                                 __global const PerRayAcc*     restrict in_pdfAcc,
                                 __global const int*           restrict in_lightId,
                                 __global const float*         restrict in_lsam2,
                                 __global const float4*        restrict in_procTexData,
                                 
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
  
  SurfaceHit surfHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &surfHit);
  
  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(surfHit.pos, surfHit.normal, a_globals,
                                                                &camDir, &zDepth);

  float  signOfNormal = 1.0f;
  float  pdfRevW      = 1.0f;
  float3 colorConnect = make_float3(1,1,1);
  if(a_currBounce > 0) // if 0, this is light surface
  {
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfHit.matId);
    if (pHitMaterial != 0)
    {
      if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, surfHit.normal) < -0.01f)
        signOfNormal = -1.0f;

      ShadeContext sc;
      sc.wp = surfHit.pos;
      sc.l  = camDir;
      sc.v  = (-1.0f)*ray_dir;
      sc.n  = surfHit.normal;
      sc.fn = surfHit.flatNormal;
      sc.tg = surfHit.tangent;
      sc.bn = surfHit.biTangent;
      sc.tc = surfHit.texCoord;

      ProcTextureList ptl;
      InitProcTextureList(&ptl);
      ReadProcTextureList(in_procTexData, tid, iNumElements,
                          &ptl);

      BxDFResult matRes = materialEval(pHitMaterial, &sc, (EVAL_FLAG_FWD_DIR | EVAL_FLAG_APPLY_GLOSS_SIG), /* global data --> */ a_globals, in_texStorage1, in_texStorage2, &ptl);
      colorConnect      = matRes.brdf + matRes.btdf;
      pdfRevW           = matRes.pdfRev;
    }
  }

  float misWeight = 1.0f;
  if ((a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) && (a_currBounce > 0))
  {
    const PerRayAcc accData = in_pdfAcc[tid];

    const float cosCurr  = fabs(dot(ray_dir, surfHit.normal));
    const float pdfRevWP = pdfRevW / fmax(cosCurr, DEPSILON); // pdfW po pdfWP

    float pdfCamA0 = accData.pdfCamA0;
    if (a_currBounce == 1)
      pdfCamA0 *= pdfRevWP; // see pdfRevWP? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.

    const float cancelImplicitLightHitPdf = (1.0f / fmax(pdfCamA0, DEPSILON2));

    __global const PlainLight* pLight = lightAt(a_globals, in_lightId[tid]);
    const float lightPickProbFwd = lightPdfSelectFwd(pLight);
    const float lightPickProbRev = lightPdfSelectRev(pLight);

    const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
    const float lightPdfA  = in_lsam2[tid]; //PerThread().pdfLightA0; // remember that we packed it in lsam2 inside 'LightSampleForwardKernel'

    const float pdfAccFwdA = 1.0f*accData.pdfLightWP*accData.pdfGTerm*(lightPdfA*lightPickProbFwd);
    const float pdfAccRevA = cameraPdfA * (pdfRevWP*accData.pdfCameraWP)*accData.pdfGTerm; // see pdfRevWP? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.
                                                                                           // we didn't eval reverse pdf yet. Imagine light ray hit surface and we immediately connect.
    const float pdfAccExpA = cameraPdfA * (pdfRevWP*accData.pdfCameraWP)*accData.pdfGTerm*(cancelImplicitLightHitPdf*(lightPdfA*lightPickProbRev));

    misWeight = misWeightHeuristic3(pdfAccFwdA, pdfAccRevA, pdfAccExpA);
    if (!isfinite(misWeight))
      misWeight = 0.0f;
  }


  // We divide the contribution by surfaceToImageFactor to convert the (already
  // divided) pdf from surface area to image plane area, w.r.t. which the
  // pixel integral is actually defined. We also divide by the number of samples
  // this technique makes, which is equal to the number of light sub-paths
  //
  float3 a_accColor  = to_float3(a_colorIn[tid]);        // a_accColor = make_float3(10,10,10);
  float3 shadowColor = decompressShadow(in_shadow[tid]);
  float3 sampleColor = misWeight*shadowColor*(a_accColor*colorConnect) * (imageToSurfaceFactor / mLightSubPathCount);
  
  if (!isfinite(sampleColor.x) || !isfinite(sampleColor.y) || !isfinite(sampleColor.z) || imageToSurfaceFactor <= 0.0f)
    sampleColor = make_float3(0, 0, 0);

  int x = 65535, y = 65535;
  if (dot(sampleColor, sampleColor) > 1e-12f) // add final result to image
  {
    const float2 posScreenSpace = worldPosToScreenSpace(surfHit.pos, a_globals);

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
                                  __global const int*       restrict in_packXY,
                                  
                                  __global const float4*    restrict in_surfaceHit,
                                  __global const float4*    restrict in_procTexData,
                                  
                                  __global float4*          restrict a_color,
                                  __global float4*          restrict a_thoroughput,
                                  __global MisData*         restrict a_misDataPrev,
                                  __global float4*          restrict out_emission,
                                  __global uchar*           restrict out_shadow,
                                  
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

  // if hit environment
  //
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE)
  {
    const float3 ray_pos = to_float3(in_rpos[tid]);
    const float3 ray_dir = to_float3(in_rdir[tid]);

    const int  hitId         = hitDirectLight(ray_dir, a_globals);
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

      if ((a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) != 0) //#TODO: fix IBPT
        lightColor = make_float3(0, 0, 0); 

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////

      const float3 pathThroughput = to_float3(a_thoroughput[tid]);
      nextPathColor = to_float3(a_color[tid]) + pathThroughput * lightColor;
    }
    else
    {
      float3 envColor       = environmentColor(ray_dir, a_misDataPrev[tid], flags, a_globals, in_mtlStorage, in_pdfStorage, in_texStorage1);
      float3 pathThroughput = to_float3(a_thoroughput[tid]);

      const uint rayBounce     = unpackBounceNum(flags);
      unsigned int otherFlags  = unpackRayFlags(flags);
      const int backTextureId  = a_globals->varsI[HRT_SHADOW_MATTE_BACK];
      const bool transparent   = (rayBounce == 1 && (otherFlags & RAY_EVENT_T) != 0);
      
      if (backTextureId != INVALID_TEXTURE && (rayBounce == 0 || transparent))
      {
        const int packedXY = in_packXY[tid];
        const int screenX  = (packedXY & 0x0000FFFF);
        const int screenY  = (packedXY & 0xFFFF0000) >> 16;

        const float texCoordX = (float)screenX / a_globals->varsF[HRT_WIDTH_F];
        const float texCoordY = (float)screenY / a_globals->varsF[HRT_HEIGHT_F];

        const float gammaInv = a_globals->varsF[HRT_BACK_TEXINPUT_GAMMA];

        const int offset = textureHeaderOffset(a_globals, backTextureId);
        envColor = to_float3(read_imagef_sw4(in_texStorage1 + offset, make_float2(texCoordX, texCoordY), TEX_CLAMP_U | TEX_CLAMP_V));

        envColor.x = pow(envColor.x, gammaInv);
        envColor.y = pow(envColor.y, gammaInv);
        envColor.z = pow(envColor.z, gammaInv);
      }

      nextPathColor = to_float3(a_color[tid]) + pathThroughput * envColor;
    }

    uint otherFlags    = unpackRayFlags(flags);
    a_flags      [tid] = packRayFlags(flags, RAY_IS_DEAD | (otherFlags & (~RAY_GRAMMAR_OUT_OF_SCENE)));
    a_color      [tid] = to_float4(nextPathColor, 0.0f);
    a_thoroughput[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  else if (rayIsActiveU(flags)) // if thread is active
  {
    bool hitLightSource = false;

    SurfaceHit surfHit;
    ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                  &surfHit);

    const float3 ray_pos = to_float3(in_rpos[tid]);
    const float3 ray_dir = to_float3(in_rdir[tid]);

    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfHit.matId);
    if(pHitMaterial == 0)
      return;

    // eval PDFs for 3WAY 
    //
    PerRayAcc accData;
    float GTerm = 1.0f, cosHere = 1.0f, cosPrev = 1.0f;
    if (a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS)
    {
      const float dist = length(ray_pos - surfHit.pos);
      cosPrev = in_misDataPrev[tid].cosThetaPrev;
      cosHere = fabs(dot(ray_dir, surfHit.normal));
      GTerm   = cosHere * cosPrev / fmax(dist*dist, DEPSILON2);
      accData = a_pdfAcc[tid]; // for 3 bounce we need to store (p0*G0)*(p1*G1) and do not include (p2*G2) to we could replace it with explicit strategy pdf

      if (a_currDepth == 0)
      {
        float3 camDirDummy; float zDepthDummy;
        const float imageToSurfaceFactor = CameraImageToSurfaceFactor(surfHit.pos, surfHit.normal, a_globals,
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

    bool reflectProjectedBack = false;
    {
      const uint rayBounce    = unpackBounceNum(flags);
      unsigned int otherFlags = unpackRayFlags(flags);
      const int backTextureId = a_globals->varsI[HRT_SHADOW_MATTE_BACK];
      const bool reflected    = (rayBounce >= 1 && (otherFlags & RAY_EVENT_T) == 0 &&
                                 ((otherFlags & RAY_EVENT_S) != 0 || (otherFlags & RAY_EVENT_D) != 0 || (otherFlags & RAY_EVENT_G) != 0)
                                );

      reflectProjectedBack = (backTextureId != INVALID_TEXTURE) && reflected && (materialGetType(pHitMaterial) == PLAIN_MAT_CLASS_SHADOW_MATTE) &&
                             ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_CAMERA_MAPPED_REFL) != 0);
    }


    // now check if we hit light
    //
    ProcTextureList ptl;        
    InitProcTextureList(&ptl);  
    ReadProcTextureList(in_procTexData, tid, iNumElements, 
                        &ptl);

    const bool skipPieceOfShit  = materialIsInvisLight(pHitMaterial) && isEyeRay(flags);
    const Lite_Hit liteHit      = in_liteHit[tid];
    const MisData misPrev       = a_misDataPrev[tid];
    const int lightOffset       = (a_globals->lightsNum == 0 || liteHit.instId < 0) ? -1 : in_instLightInstId[liteHit.instId];
    __global const PlainLight* pLight = lightAt(a_globals, lightOffset);

    const float3 emissionVal = emissionEval(ray_pos, ray_dir, &surfHit, flags, (misPrev.isSpecular == 1), pLight,
                                            pHitMaterial, in_texStorage1, in_pdfStorage, a_globals, &ptl);

    if (dot(emissionVal, emissionVal) > 1e-6f && !skipPieceOfShit)
    {
      const float3 lightNorm = surfHit.hfi ? surfHit.normal  * (-1.0f) : surfHit.normal;
      float3 emissColor      = make_float3(0, 0, 0);

      if (dot(ray_dir, lightNorm) < 0.0f)
      {
        emissColor = emissionVal;

        if (lightOffset >= 0)
        {
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

            if (isfinite(misWeight))
              emissColor *= misWeight;
            else
              emissColor *= make_float3(0, 0, 0);
          }
          else if (unpackBounceNum(flags) > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0)) // old MIS weights via pdfW
          {
            const float lgtPdf    = lightPdfSelectRev(pLight)*lightEvalPDF(pLight, ray_pos, ray_dir, 
                                                                           surfHit.pos, surfHit.normal, surfHit.texCoord, in_pdfStorage, a_globals);
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
            if(pPrevMaterial != 0)
            {
              bool disableCaustics = (unpackBounceNumDiff(flags) > 0) && !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS) &&
                                     materialCastCaustics(pPrevMaterial); // and prev material cast caustics
              if (disableCaustics)
                emissColor = make_float3(0, 0, 0);
            }
          }

          hitLightSource = true; // kill thread next if it hit real light source
        }
        else // hit emissive material, not a light 
        {
          
        }
      }
      
      const int packedIsLight = hitLightSource ? 1 : 0;
      out_emission[tid]       = to_float4(emissColor, as_float(packedIsLight));

    } // \\ if light emission is not zero
    else if (reflectProjectedBack) // fucking shadow catcher (camera mapped reflections)
    {
      const float3 surfHitPos     = ReadSurfaceHitPos(in_surfaceHit, tid, iNumElements);
      const float2 posScreenSpace = worldPosToScreenSpace(surfHitPos, a_globals);
      const int    backTextureId  = a_globals->varsI[HRT_SHADOW_MATTE_BACK];

      const float x   = (posScreenSpace.x + 0.5f);
      const float y   = (posScreenSpace.y + 0.5f);
      float3 envColor = make_float3(0, 0, 0);

      if (x >= 0.0f && y >= 0.0f && x <= a_globals->varsF[HRT_WIDTH_F] && y <= a_globals->varsF[HRT_HEIGHT_F])
      {
        const float texCoordX = x / a_globals->varsF[HRT_WIDTH_F];
        const float texCoordY = y / a_globals->varsF[HRT_HEIGHT_F];
        const float gammaInv  = a_globals->varsF[HRT_BACK_TEXINPUT_GAMMA];

        const int offset = textureHeaderOffset(a_globals, backTextureId);
        envColor = to_float3(read_imagef_sw4(in_texStorage1 + offset, make_float2(texCoordX, texCoordY), TEX_CLAMP_U | TEX_CLAMP_V));

        envColor.x = pow(envColor.x, gammaInv);
        envColor.y = pow(envColor.y, gammaInv);
        envColor.z = pow(envColor.z, gammaInv);
      }

      out_emission[tid] = to_float4(envColor, 0.0f);
    }
    else
    {
      out_emission[tid] = make_float4(0, 0, 0, as_float(0));
    }

  } // \\ else if thread is active

  // account implicit shadows ... 
  //
  
}

__kernel void Shade(__global const float4*    restrict a_rpos,
                    __global const float4*    restrict a_rdir,
                    __global       uint*      restrict a_flags,
              
                    __global const float4*    restrict in_surfaceHit,
                  
                
                    __global const ushort4*   restrict in_shadow,

                    __global const float4*    restrict in_lrev,
                    __global const float4*    restrict in_procTexData,

                    __global const PerRayAcc* restrict in_pdfAccPrev,
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

  SurfaceHit surfHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &surfHit);

  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfHit.matId);
  if(pHitMaterial == 0)
    return;

  float3 ray_pos = to_float3(a_rpos[tid]);
  float3 ray_dir = to_float3(a_rdir[tid]);


  ShadowSample explicitSam; float lightPickProb; int lightOffset;
  ReadShadowSample(in_lrev, tid, iNumElements,
                   &explicitSam, &lightPickProb, &lightOffset);

  //float3 shadowRayPos = surfHit.pos + surfHit.normal*maxcomp(surfHit.pos)*GEPSILON;
  float3 shadowRayDir = normalize(explicitSam.pos - surfHit.pos); 

  const bool currLightCastCaustics = (a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS);
  const bool disableCaustics       = (unpackBounceNumDiff(flags) > 0) && !currLightCastCaustics;
  
  if ((a_globals->varsI[HRT_RENDER_LAYER] == LAYER_INCOMING_PRIMARY) && (a_globals->varsI[HRT_RENDER_LAYER_DEPTH] == unpackBounceNum(flags))) //////////////////////////////////////////////////////////////////
    pHitMaterial = materialAtOffset(in_mtlStorage, a_globals->varsI[HRT_WHITE_DIFFUSE_OFFSET]);

  // bool hitFromBack = (bool)(unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE);

  ShadeContext sc;
  sc.wp  = surfHit.pos;
  sc.l   = shadowRayDir;
  sc.v   = (-1.0f)*ray_dir;
  sc.n   = surfHit.normal;
  sc.fn  = surfHit.flatNormal;
  sc.tg  = surfHit.tangent;
  sc.bn  = surfHit.biTangent;
  sc.tc  = surfHit.texCoord;
  sc.hfi = surfHit.hfi;

  ProcTextureList ptl;
  InitProcTextureList(&ptl);
  ReadProcTextureList(in_procTexData, tid, iNumElements,
                      &ptl);

  const int evalFlags       = (disableCaustics ? EVAL_FLAG_DISABLE_CAUSTICS : EVAL_FLAG_DEFAULT);

  const BxDFResult evalData = materialEval(pHitMaterial, &sc, (evalFlags), /* global data --> */ a_globals, in_texStorage1, in_texStorage2, &ptl);

  __global const PlainLight* pLight = lightAt(a_globals, lightOffset);

  float cosThetaOutAux = 1.0f;

  float misWeight = 1.0f;
  if ((a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) && (lightType(pLight) != PLAIN_LIGHT_TYPE_SKY_DOME))
  {
    const float cosHere      = fabs(dot(ray_dir, surfHit.normal));
    const int a_currDepth    = rayBounceNum;

    const float cosThetaOut1 = fmax(+dot(shadowRayDir, surfHit.normal), 0.0f);
    const float cosThetaOut2 = fmax(-dot(shadowRayDir, surfHit.normal), 0.0f);
    const bool  underSurface = (dot(evalData.btdf, evalData.btdf)*cosThetaOut2 > 0.0f && dot(evalData.brdf, evalData.brdf)*cosThetaOut1 <= 0.0f);
    const float cosThetaOut  = underSurface ? cosThetaOut2 : cosThetaOut1;
    const float cosAtLight   = fmax(explicitSam.cosAtLight, 0.0f);
    
    cosThetaOutAux = dot(shadowRayDir, surfHit.normal);

    const float bsdfRevWP    = (evalData.pdfFwd == 0.0f) ? 1.0f : evalData.pdfFwd / fmax(cosThetaOut, DEPSILON);
    const float bsdfFwdWP    = (evalData.pdfRev == 0.0f) ? 1.0f : evalData.pdfRev / fmax(cosHere, DEPSILON);
    const float shadowDist   = length(surfHit.pos - explicitSam.pos);
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
    if (!isfinite(misWeight))
      misWeight = 0.0f;
  }
  else
  {
    const float lgtPdf = explicitSam.pdf*lightPickProb;
    cosThetaOutAux     = dot(shadowRayDir, surfHit.normal);

    misWeight = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
    if (explicitSam.isPoint)
      misWeight = 1.0f;
  }

  const float3 shadow = decompressShadow(in_shadow[tid]);

  if (out_shadow != 0 && rayBounceNum == 0)
  {
    float shadow1 = 255.0f*0.33333f*(shadow.x + shadow.y + shadow.z);
    if ( (cosThetaOutAux < 0.1f) || (explicitSam.cosAtLight < 0.1f && lightType(pLight) != PLAIN_LIGHT_TYPE_SKY_DOME))
      shadow1 = 255.0f;
   
    out_shadow[tid] = (uchar)(255.0f - clamp(shadow1, 0.0f, 255.0f));
  }

  const float cosThetaOut1 = fmax(+dot(shadowRayDir, surfHit.normal), 0.0f);
  const float cosThetaOut2 = fmax(-dot(shadowRayDir, surfHit.normal), 0.0f);

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
}

__kernel void NextBounce(__global   float4*        restrict a_rpos,
                         __global   float4*        restrict a_rdir,
                         __global   uint*          restrict a_flags,
                         __global RandomGen*       restrict out_gens,
                         
                         __global const float4*    restrict in_surfaceHit,
                         __global const float4*    restrict in_procTexData,

                         __global float4*          restrict a_color,
                         __global float4*          restrict a_thoroughput,
                         __global MisData*         restrict a_misDataPrev,
                         __global const ushort4*   restrict in_shadow,
                         __global float4*          restrict a_fog,
                         __global const float4*    restrict in_shadeColor,
                         __global const float4*    restrict in_emissionColor,
                         __global PerRayAcc*       restrict a_pdfAcc,        // used only by 3-Way PT/LT passes
                         __global float*           restrict a_camPdfA,       // used only by 3-Way PT/LT passes

                         __global const float4*    restrict in_texStorage1,    
                         __global const float4*    restrict in_texStorage2,
                         __global const float4*    restrict in_mtlStorage,
                         __global const float4*    restrict in_pdfStorage,   //


                         __constant unsigned int*  restrict a_qmcTable,
                         int a_passNumberForQmc,
                         
                         int iNumElements,
                         __global const EngineGlobals*  restrict a_globals)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const unsigned int qmcPos = reverseBits(tid, iNumElements) + a_passNumberForQmc * iNumElements;
  
  uint flags = a_flags[tid];
  if (!rayIsActiveU(flags))
    return;

  float3 ray_pos      = to_float3(a_rpos[tid]);
  float3 ray_dir      = to_float3(a_rdir[tid]);
  float3 outPathColor = make_float3(0,0,0);

  SurfaceHit surfHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &surfHit);

  __global const PlainMaterial* pHitMaterial     = materialAt(a_globals, in_mtlStorage, surfHit.matId);
  if(pHitMaterial == 0)
    return;

  float4 emissData    = (in_emissionColor == 0) ? make_float4(0, 0, 0, 0) : in_emissionColor[tid];
  outPathColor        = to_float3(emissData);
  int matOffset       = materialOffset(a_globals, surfHit.matId);
  bool hitLightSource = (emissData.w == 1);

  ProcTextureList ptl;
  InitProcTextureList(&ptl);
  ReadProcTextureList(in_procTexData, tid, iNumElements,
                      &ptl);
  
  const int rayBounceNum = unpackBounceNum(flags);


  float allRands[MMLT_FLOATS_PER_BOUNCE];
  float rrChoice = 0.0f, pabsorb  = 0.0f;
  {
    RandomGen gen  = out_gens[tid];
   
    RndMatAll(&gen, 0, rayBounceNum, a_globals->rmQMC, qmcPos, a_qmcTable,
              allRands);

    pabsorb = probabilityAbsorbRR(flags, a_globals->g_flags);
    if (pabsorb > 0.0f)
      rrChoice = rndFloat1_Pseudo(&gen);
    out_gens[tid] = gen;
  }
  
  const float3 shadowVal = decompressShadow(in_shadow[tid]);

  MatSample brdfSample; int localOffset = 0; 
  MaterialSampleAndEvalBxDF(pHitMaterial, allRands, &surfHit, ray_dir, shadowVal, flags,
                            a_globals, in_texStorage1, in_texStorage2, &ptl, 
                            &brdfSample, &localOffset);
                            
  matOffset    = matOffset    + localOffset*(sizeof(PlainMaterial)/sizeof(float4));
  pHitMaterial = pHitMaterial + localOffset;

  const float invPdf       = 1.0f / fmax(brdfSample.pdf, DEPSILON2);
  const float cosTheta     = fabs(dot(brdfSample.direction, surfHit.normal));
  float3 outPathThroughput = cosTheta*brdfSample.color*invPdf; 
  if (!isfinite(outPathThroughput.x)) outPathThroughput.x = 0.0f;
  if (!isfinite(outPathThroughput.y)) outPathThroughput.y = 0.0f;
  if (!isfinite(outPathThroughput.z)) outPathThroughput.z = 0.0f;
  
  const bool isThinGlass         = ((brdfSample.flags & RAY_EVENT_TNINGLASS) != 0) && (rayBounceNum > 0) && !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS);
  const bool thisBounceIsDiffuse = ((brdfSample.flags & RAY_EVENT_D)         != 0);

  /////////////////////////////////////////////////////////////////////////////// finish with outPathThroughput

  const float3 nextRay_dir = brdfSample.direction;
  const float3 nextRay_pos = OffsRayPos(surfHit.pos, surfHit.normal, brdfSample.direction);
  // values that bidirectional techniques needs
  //
  const float cosPrev = fabs(a_misDataPrev[tid].cosThetaPrev);
  const float cosCurr = fabs(-dot(ray_dir, surfHit.normal));
  const float dist    = length(surfHit.pos - ray_pos);
  const float GTerm   = (cosPrev*cosCurr / fmax(dist*dist, DEPSILON2));
  
  // calc new ray
  //
  float3 oldRayDir = ray_dir;
  ray_dir          = nextRay_dir;
  ray_pos          = nextRay_pos;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////// begin russian roulette
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

  // calc attenuation in thick glass
	//
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	const float dotNewOld  = dot(ray_dir, to_float3(a_rdir[tid]));
	const bool  gotOutside = (dotNewOld > 0.0f) && (unpackRayFlags(flags) & RAY_HIT_SURFACE_FROM_OTHER_SIDE);
	const float dist2      = length(surfHit.pos - to_float3(a_rpos[tid]));
	const float3 fogAtten  = attenuationStep(pHitMaterial, dist2, gotOutside, a_fog + tid);
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  const bool killShade    = (a_globals->g_flags & HRT_STUPID_PT_MODE) || ((a_globals->g_flags & HRT_DIRECT_LIGHT_MODE)!=0 && rayBounceNum > 0);  
  const float3 shadeColor = killShade ? make_float3(0,0,0) : to_float3(in_shadeColor[tid]);
  oldPathThroughput       = to_float3(a_thoroughput[tid])*fogAtten;
  newPathThroughput       = oldPathThroughput*outPathThroughput;
  
  ///////////////////////////////////////////////// #NOTE: OK, THIS SEEMS TO WORK FINE; JUST CHECK IT WITH WINDOW GLASS WHEN IMPLEMENT TRANSPARENT SHADOWS;
  if (!isThinGlass)  
  {
    MisData misNext;  
    misNext.matSamplePdf       = brdfSample.pdf;
    misNext.isSpecular         = (int)isPureSpecular(brdfSample);
    misNext.prevMaterialOffset = matOffset;
    misNext.cosThetaPrev       = fabs(+dot(ray_dir, surfHit.normal)); // update it withCosNextActually ...
    a_misDataPrev[tid]         = misNext;
  }
  ///////////////////////////////////////////////// 
  flags = flagsNextBounce(flags, brdfSample, a_globals);
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Specular DL for MMLT
  const bool evalSpecularOnly  = ((a_globals->g_flags & HRT_DIRECT_LIGHT_MODE) !=0);
  if(evalSpecularOnly && !flagsHaveOnlySpecular(flags) && rayBounceNum > 1)
  {
    outPathThroughput = make_float3(0,0,0);
    newPathThroughput = make_float3(0,0,0);
    oldPathThroughput = make_float3(0,0,0);
    flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
  }
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Specular DL for MMLT

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
    outPathColor   += shadeColor;
    nextPathColor   = a_color[tid] + to_float4(oldPathThroughput*outPathColor, 0.0f);
    nextPathColor.w = 1.0f;
    if (a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) 
    {
      const int a_currDepth = rayBounceNum;
      PerRayAcc accPdf      = a_pdfAcc[tid];          // #TODO: refactor code inside brackets; try to make procedure call
      {
        if (!isPureSpecular(brdfSample))
        {
          const float cosHere = fabs(dot(oldRayDir, surfHit.normal));
          const float cosNext = fabs(dot(brdfSample.direction, surfHit.normal));
          accPdf.pdfCameraWP *= (brdfSample.pdf / fmax(cosNext, DEPSILON));
          if (a_currDepth > 0)
          {
            ShadeContext sc;
            sc.wp = surfHit.pos;
            sc.l  = (-1.0f)*oldRayDir;    // fliped; if compare to normal PT
            sc.v  = brdfSample.direction; // fliped; if compare to normal PT
            sc.n  = surfHit.normal;
            sc.fn = surfHit.flatNormal;
            sc.tg = surfHit.tangent;
            sc.bn = surfHit.biTangent;
            sc.tc = surfHit.texCoord;
            ProcTextureList ptl;       
            InitProcTextureList(&ptl); 
            ReadProcTextureList(in_procTexData, tid, iNumElements,
                                &ptl);
            const float pdfFwdW = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), // global data on the second line -->                          
                                               a_globals, in_texStorage1, in_texStorage2, &ptl).pdfFwd; 
           
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void NextTransparentBounce(__global   float4*        restrict a_rpos,
                                    __global   float4*        restrict a_rdir,
                                    __global   uint*          restrict a_flags,
                                                             
                                    __global const float4*    restrict in_surfaceHit,
                                    __global const float4*    restrict in_procTexData,
                                    
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
    SurfaceHit surfHit;
    ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                   &surfHit);

    __global const PlainMaterial* pHitMaterial = materialAt(in_globals, in_mtlStorage, surfHit.matId);
    if(pHitMaterial == 0)
      return;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (pHitMaterial != 0)
    {
      const float3 ray_pos = to_float3(a_rpos[tid]);
      const float3 ray_dir = to_float3(a_rdir[tid]);

      ProcTextureList ptl;       
      InitProcTextureList(&ptl); 
      ReadProcTextureList(in_procTexData, tid, iNumElements,
                          &ptl);

      TransparencyAndFog matFogAndTransp = materialEvalTransparencyAndFog(pHitMaterial, ray_dir, surfHit.normal, surfHit.texCoord, 
                                                                          in_globals, in_shadingTexture, &ptl);

      float4 newPathThroughput = a_thoroughput[tid] * to_float4(matFogAndTransp.transparency, 1.0f);

      const float3 nextRay_pos = OffsRayPos(surfHit.pos, surfHit.normal, ray_dir);

      if (maxcomp(to_float3(newPathThroughput)) < 0.00001f)
        flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);

      a_rpos       [tid] = to_float4(nextRay_pos, 0.0f);
      a_flags      [tid] = flags;
      a_thoroughput[tid] = newPathThroughput;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  } //  if (rayIsActiveU(flags))
  
  
}


__kernel void TransparentShadowKenrel(__global const uint*     restrict in_flags,
                                      __global const float4*   restrict in_surfaceHit,
                                      __global const float4*   restrict a_data1,
                                      __global const float4*   restrict a_data2,
                                      __global       ushort4*  restrict a_shadow,
                                      __global       int*      restrict a_transpL,
                                      __global const float*    restrict a_hitPolySize,
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
                                       __global const EngineGlobals* a_globals,
                                      int iNumElements)
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

  SurfaceHit surfHit;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                   &surfHit);

  const float  epsilon = fmax(maxcomp(surfHit.pos), 1.0f)*GEPSILON;
  const float  polyEps = fmin(fmax(PEPSILON*a_hitPolySize[tid], epsilon), PG_SCALE*epsilon);

  const float3 shadowRayDir = normalize(explicitSam.pos - surfHit.pos); // explicitSam.direction;
  const float  offsetSign   = (dot(shadowRayDir, surfHit.normal) >= 0.0f) ? 1.0f : -1.0f;
  const float3 shadowRayPos = OffsRayPos(surfHit.pos, surfHit.normal, surfHit.normal*offsetSign); 

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


__kernel void ReadDiffuseColor(__global const float4*    restrict a_rdir, 
                               __global const Lite_Hit*  restrict in_hits,
                               __global const float4*    restrict in_surfaceHit,

                               __global const float4*    restrict in_procTexData,
                               __global const int4*      restrict in_texStorage1,

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
    SurfaceHit surfHit;
    ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                   &surfHit);

    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfHit.matId);
    if(pHitMaterial != 0)
    {
       ProcTextureList ptl;
       InitProcTextureList(&ptl);
       ReadProcTextureList(in_procTexData, tid, iNumElements,
                           &ptl);

       color = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), surfHit.normal, surfHit.texCoord, a_globals,
                                   in_texStorage1, &ptl);
    }
  }

  a_color[tid] = to_float4(color, 0.0f);
}


__kernel void GetGBufferSample(__global const float4*    a_rdir,
                               __global const Lite_Hit*  restrict in_hits,
                               __global const uint*      restrict in_flags,
                               __global const float4*    restrict in_surfaceHit,
                               __global const float4*    restrict in_procTexData,

                               __global float4*          restrict out_gbuff1,
                               __global float4*          restrict out_gbuff2,

                               __global const HitMatRef*     restrict in_mtlStorage,
                               texture2d_t                   restrict a_shadingTexture,
                               __global const EngineGlobals* restrict a_globals,
                               int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  __local GBufferAll samples[GBUFFER_SAMPLES];

  const Lite_Hit liteHit  = in_hits[tid];
  const uint flags        = in_flags[tid];

  if (rayIsActiveU(flags))
  {
    SurfaceHit surfHit;
    ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                   &surfHit);

    ProcTextureList ptl;        
    InitProcTextureList(&ptl);  
    ReadProcTextureList(in_procTexData, tid, iNumElements,
                        &ptl);

    float3 diffColor = make_float3(0,0,0);
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfHit.matId);
    if(pHitMaterial != 0)
      diffColor = materialEvalDiffuse(pHitMaterial, to_float3(a_rdir[tid]), surfHit.normal, surfHit.texCoord, a_globals, a_shadingTexture, &ptl);

    samples[LOCAL_ID_X].data1.depth    = liteHit.t;
    samples[LOCAL_ID_X].data1.norm     = surfHit.normal;
    samples[LOCAL_ID_X].data1.rgba     = to_float4(diffColor, 0.0f);
    samples[LOCAL_ID_X].data1.matId    = surfHit.matId;
    samples[LOCAL_ID_X].data1.coverage = 0.0f;

    samples[LOCAL_ID_X].data2.texCoord = surfHit.texCoord;
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



// change 31.08.2018 13:55;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

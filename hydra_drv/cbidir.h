#ifndef RTCBIDIR
#define RTCBIDIR

#include "cglobals.h"
#include "cfetch.h"
#include "cmaterial.h"
#include "clight.h"

typedef struct PathVertexT
{
  SurfaceHit hit;
  float3     ray_dir;
  float3     accColor;
  float      lastGTerm;
  bool       valid;
  bool       wasSpecOnly;  ///< Was Specular Only. Exclude Direct Light - ES*(D|G)L or ES*L.
  bool       hitLight   ;  ///< used only by GPU debug code; needed to disting CameraVertex that hit light and CameraVertex that have to be connected
} PathVertex;

typedef struct PdfVertexT
{
  float pdfFwd;
  float pdfRev;
} PdfVertex;

static inline void InitPathVertex(__private PathVertex* a_pVertex) 
{
  a_pVertex->lastGTerm   = 1.0f;
  a_pVertex->accColor    = make_float3(1, 1, 1);
  a_pVertex->valid       = false;
  a_pVertex->hitLight    = false; 
  a_pVertex->wasSpecOnly = false;
}

#define PATH_VERTEX_SUPPLEMENT_SIZE_IN_F4 2

static inline void WritePathVertexSupplement(const __private PathVertex* a_pVertex, int a_tid, int a_threadNum, 
                                             __global float4* a_out)
{
  const int bit1  = a_pVertex->valid        ? PV_PACK_VALID_FIELD : 0;
  const int bit2  = a_pVertex->wasSpecOnly  ? PV_PACK_WASSP_FIELD : 0;
  const int bit3  = a_pVertex->hitLight     ? PV_PACK_RCONN_FIELD : 0;

  const float4 f1 = to_float4(a_pVertex->ray_dir, a_pVertex->lastGTerm);
  const float4 f2 = to_float4(a_pVertex->accColor, as_float(bit1 | bit2 | bit3));

  a_out[a_tid + 0*a_threadNum] = f1;
  a_out[a_tid + 1*a_threadNum] = f2;
} 

static inline void ReadPathVertexSupplement(const __global float4* a_in, int a_tid, int a_threadNum, 
                                            __private PathVertex* a_pVertex)
{
  const float4 f1 = a_in[a_tid + 0*a_threadNum];
  const float4 f2 = a_in[a_tid + 1*a_threadNum];

  a_pVertex->ray_dir     = to_float3(f1); a_pVertex->lastGTerm = f1.w;
  a_pVertex->accColor    = to_float3(f2); 
  const int flags        = as_int(f2.w);

  a_pVertex->valid        = ((flags & PV_PACK_VALID_FIELD) != 0);
  a_pVertex->wasSpecOnly  = ((flags & PV_PACK_WASSP_FIELD) != 0);
  a_pVertex->hitLight     = ((flags & PV_PACK_RCONN_FIELD) != 0);
} 

/**
\brief  Compute pdf conversion factor from image plane area to surface area.
\param  a_hitPos  - world position point we are going to connect with camera in LT
\param  a_hitNorm - normal of the  point we are going to connect with camera in LT
\param  a_globals - engine globals
\param  a_diskOffs- offset of the camera in interval [-1,1] with disk distribution; (0,0) means no offset.
\param  pCamDir   - out parameter. Camera forward direction.
\param  pZDepth   - out parameter. Distance between a_hitPos and camera position.
\return pdf conversion factor from image plane area to surface area.

*/

static inline float CameraImageToSurfaceFactor(const float3 a_hitPos, const float3 a_hitNorm, __global const EngineGlobals* a_globals, float2 a_diskOffs,
                                               __private float3* pCamDir, __private float* pZDepth)
{
  const float4x4 mWorldViewInv  = make_float4x4(a_globals->mWorldViewInverse);

  const float3   camForward     = make_float3(a_globals->camForward[0],  a_globals->camForward[1],  a_globals->camForward[2]);
  const float3   camUp          = make_float3(a_globals->camUpVector[0], a_globals->camUpVector[1], a_globals->camUpVector[2]);
  const float3   camLeft        = normalize(cross(camForward, camUp)); 
  const float    imagePlaneDist = a_globals->imagePlaneDist;


  const float3 camPos = mul(mWorldViewInv, make_float3(0, 0, 0)) + camUp*a_diskOffs.y*a_globals->varsF[HRT_DOF_LENS_RADIUS] + camLeft*a_diskOffs.x*a_globals->varsF[HRT_DOF_LENS_RADIUS];

  const float  zDepth = length(camPos - a_hitPos);
  const float3 camDir = (1.0f / zDepth)*(camPos - a_hitPos); // normalize

  (*pCamDir) = camDir;
  (*pZDepth) = zDepth;

  // Compute pdf conversion factor from image plane area to surface area
  //
  const float cosToCamera = fabs(dot(a_hitNorm, camDir)); 
  const float cosAtCamera = dot(camForward, (-1.0f)*camDir);

  const float relation = a_globals->varsF[HRT_WIDTH_F] / a_globals->varsF[HRT_HEIGHT_F];
  const float fov      = relation*fmax(a_globals->varsF[HRT_FOV_X], a_globals->varsF[HRT_FOV_Y]);
  if (cosAtCamera <= cos(fov))
    return 0.0f;

  const float imagePointToCameraDist  = imagePlaneDist / cosAtCamera;
  const float imageToSolidAngleFactor = (imagePointToCameraDist*imagePointToCameraDist) / cosAtCamera; // PdfAtoW
  const float imageToSurfaceFactor    = imageToSolidAngleFactor * cosToCamera / (zDepth*zDepth);       // PdfWtoA

  if (isfinite(imageToSurfaceFactor))
    return imageToSurfaceFactor/(relation*relation);
  else
    return 0.0f;
}

static inline float2 clipSpaceToScreenSpace(float4 a_pos, const float fw, const float fh)
{
  const float x = a_pos.x*0.5f + 0.5f;
  const float y = a_pos.y*0.5f + 0.5f;
  return make_float2(x*fw, y*fh);
}

static inline float2 worldPosToScreenSpace(float3 a_wpos, __global const EngineGlobals* a_globals)
{
  const float4 posWorldSpace  = to_float4(a_wpos, 1.0f);
  const float4 posCamSpace    = mul4x4x4(make_float4x4(a_globals->mWorldView), posWorldSpace);
  const float4 posNDC         = mul4x4x4(make_float4x4(a_globals->mProj),      posCamSpace);
  const float4 posClipSpace   = posNDC*(1.0f / fmax(posNDC.w, DEPSILON));
  const float2 posScreenSpace = clipSpaceToScreenSpace(posClipSpace, a_globals->varsF[HRT_WIDTH_F], a_globals->varsF[HRT_HEIGHT_F]);
  return posScreenSpace;
}

static inline float2 worldPosToScreenSpaceNorm(float3 a_wpos, __global const EngineGlobals* a_globals)
{
  const float4 posWorldSpace  = to_float4(a_wpos, 1.0f);
  const float4 posCamSpace    = mul4x4x4(make_float4x4(a_globals->mWorldView), posWorldSpace);
  const float4 posNDC         = mul4x4x4(make_float4x4(a_globals->mProj),      posCamSpace);
  const float4 posClipSpace   = posNDC*(1.0f / fmax(posNDC.w, DEPSILON));
  return make_float2(posClipSpace.x*0.5f + 0.5f, posClipSpace.y*0.5f + 0.5f);
}

static inline float2 worldPosToScreenSpaceWithDOF(float3 a_wpos, __global const EngineGlobals* a_globals, float2 a_diskOffs)
{
  const float3   camForward   = make_float3(a_globals->camForward[0],  a_globals->camForward[1],  a_globals->camForward[2]);
  const float3   camUp        = make_float3(a_globals->camUpVector[0], a_globals->camUpVector[1], a_globals->camUpVector[2]);
  const float3   camLeft      = normalize(cross(camForward, camUp)); 
  
        float3   camPos       = mul(make_float4x4(a_globals->mWorldViewInverse), make_float3(0, 0, 0));

  const float F   = a_globals->imagePlaneDist; // ?
  const float P   = length(camPos - make_float3(a_globals->camLookAt[0], a_globals->camLookAt[1], a_globals->camLookAt[2])); // camera target distance
  const float D   = length(a_wpos - camPos);                                                                                 // posCamSpace.z

  const float PoH = (P <= D) ? 2.0f : 1.25f;
  const float LaS = 0.0f; // pow(fmin(fabs(P-D) / fmax(D, 1e-10f), 1.0f), PoH);

  const float3   camShift     = camUp*a_diskOffs.y*a_globals->varsF[HRT_DOF_LENS_RADIUS] + camLeft*a_diskOffs.x*a_globals->varsF[HRT_DOF_LENS_RADIUS];
  const float3   camLookaAt   = make_float3(a_globals->camLookAt[0],   a_globals->camLookAt[1],   a_globals->camLookAt[2]) + LaS*camShift;
                 camPos       = camPos + camShift;

  const float4x4 mWorldView   = lookAt(camPos, camLookaAt, camUp);

  const float4 posCamSpace    = mul4x4x4(mWorldView, to_float4(a_wpos, 1.0f));
  const float4 posNDC         = mul4x4x4(make_float4x4(a_globals->mProj), posCamSpace);
  const float4 posClipSpace   = posNDC*(1.0f / fmax(posNDC.w, DEPSILON));
  const float2 posScreenSpace = clipSpaceToScreenSpace(posClipSpace, a_globals->varsF[HRT_WIDTH_F], a_globals->varsF[HRT_HEIGHT_F]);
  return posScreenSpace;
}

/**
\brief  Light Tracing "connect vertex to eye" stage. Don't trace ray and don't compute shadow. You must compute shadow outside this procedure.
\param  a_lv                 - light path vertex
\param  a_mLightSubPathCount - number of total light samples
\param  a_shadowHit          - a hit of 'shadow' (i.e. surface to eye) ray from a_lv.hit.pos to camPos
\param  a_globals            - engine globals
\param  a_mltStorage         - material storage
\param  a_texStorage1        - main texture storage (color)
\param  a_texStorage2        - auxilarry texture storage (for normalmaps and other)
\param  a_ptList             - proc textures list data

\param  v0                   - pdfArea array for fwd and reverse pdf's
\param  v1                   - pdfArea array for fwd and reverse pdf's
\param  pX                   - resulting screen X position
\param  pY                   - resulting screen Y position

\return resulting color

*/

static float3 ConnectEyeP(const PathVertex* a_lv, float a_mLightSubPathCount, float3 camDir, const float imageToSurfaceFactor,
                        __global const EngineGlobals* a_globals, __global const float4* a_mltStorage, texture2d_t a_texStorage1, texture2d_t a_texStorage2, __private const ProcTextureList* a_ptList,
                        __private PdfVertex* v0, __private PdfVertex* v1, int* pX, int* pY)
{
  const float surfaceToImageFactor  = 1.f / imageToSurfaceFactor;

  float  pdfRevW      = 1.0f;
  float3 colorConnect = make_float3(1, 1, 1);
  {
    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mltStorage, a_lv->hit.matId);
    if(pHitMaterial != 0)
    {
      ShadeContext sc;
      sc.wp = a_lv->hit.pos;
      sc.l  = camDir;                // seems like that sc.l = camDir, see smallVCM
      sc.v  = (-1.0f)*a_lv->ray_dir; // seems like that sc.v = (-1.0f)*ray_dir, see smallVCM
      sc.n  = a_lv->hit.normal;
      sc.fn = a_lv->hit.flatNormal;
      sc.tg = a_lv->hit.tangent;
      sc.bn = a_lv->hit.biTangent;
      sc.tc = a_lv->hit.texCoord;

      BxDFResult colorAndPdf = materialEval(pHitMaterial, &sc, (EVAL_FLAG_FWD_DIR), a_globals, a_texStorage1, a_texStorage2, a_ptList);

      colorConnect     = colorAndPdf.brdf + colorAndPdf.btdf;
      pdfRevW          = colorAndPdf.pdfRev;
    }
  }

  // we didn't eval reverse pdf yet. Imagine light ray hit surface and we immediately connect.
  //
  const float cosCurr    = fabs(dot(a_lv->ray_dir, a_lv->hit.normal));
  const float pdfRevWP   = pdfRevW / fmax(cosCurr, DEPSILON2); // pdfW po pdfWP
  const float cameraPdfA = imageToSurfaceFactor / a_mLightSubPathCount;
  
  v0->pdfRev = (pdfRevW == 0.0f) ? -1.0f*a_lv->lastGTerm : pdfRevWP*a_lv->lastGTerm;  
  v1->pdfFwd = 1.0f;
  v1->pdfRev = cameraPdfA;

  ///////////////////////////////////////////////////////////////////////////////

  // We divide the contribution by surfaceToImageFactor to convert the(already
  // divided) pdf from surface area to image plane area, w.r.t. which the
  // pixel integral is actually defined. We also divide by the number of samples
  // this technique makes, which is equal to the number of light sub-paths
  //
  const float3 sampleColor = a_lv->accColor*(colorConnect / (a_mLightSubPathCount*surfaceToImageFactor));
  float3 resColor  = make_float3(0, 0, 0);

  const int width  = (int)(a_globals->varsF[HRT_WIDTH_F]);
  const int height = (int)(a_globals->varsF[HRT_HEIGHT_F]);

  if (dot(sampleColor, sampleColor) > 1e-12f) // add final result to image
  {
    const float2 posScreenSpace = worldPosToScreenSpace(a_lv->hit.pos, a_globals);
    
    (*pX)         = (int)(posScreenSpace.x);
    (*pY)         = (int)(posScreenSpace.y);
    resColor      = sampleColor;
  }

  return resColor;
}

static inline float3 bsdfClamping(float3 a_val)
{
  return a_val;
  //const float maxVal = 10.0f;
  //return make_float3(fmin(a_val.x, maxVal), fmin(a_val.y, maxVal), fmin(a_val.z, maxVal));
}

/**
\brief  Shadow ray connection (camera vertex to light) stage. Don't trace ray and don't compute shadow. You must compute shadow outside of this procedure.
\param  a_cv          - camera vertex we want to connect with a light 
\param  a_camDepth    - camera trace depth equal to t;
\param  a_pLight      - light that we want to connect
\param  a_explicitSam - light sample
\param  a_lightPickProb - inverse number of visiable lights
\param  a_globals     - engine globals
\param  a_mltStorage  - materials storage 
\param  a_texStorage1 - general texture storage
\param  a_texStorage2 - aux texture storage
\param  a_ptList      - proc texture list data

\param  a_tableStorage - pdf table storage
\param  v0             - out pdfArea for v0 (starting from light)
\param  v1             - out pdfArea for v1 (starting from light)
\param  v2             - out pdfArea for v2 (starting from light)
\return connection throughput color without shadow

#TODO: add spetial check for glossy material when connect?

*/

static float3 ConnectShadowP(__private const PathVertex* a_cv, const int a_camDepth, __global const PlainLight* a_pLight, const ShadowSample a_explicitSam, const float a_lightPickProb,
                             __global const EngineGlobals* a_globals, __global const float4* a_mltStorage, texture2d_t a_texStorage1, texture2d_t a_texStorage2, __global const float4* a_tableStorage, __private const ProcTextureList* a_ptList,
                             __private PdfVertex* v0, __private PdfVertex* v1, __private PdfVertex* v2)
{
  const float3 shadowRayDir = normalize(a_explicitSam.pos - a_cv->hit.pos); // explicitSam.direction;
  
  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mltStorage, a_cv->hit.matId);
  if(pHitMaterial == 0)
    return make_float3(0,0,0);

  ShadeContext sc;
  sc.wp = a_cv->hit.pos;
  sc.l  = shadowRayDir;
  sc.v  = (-1.0f)*a_cv->ray_dir;
  sc.n  = a_cv->hit.normal;
  sc.fn = a_cv->hit.flatNormal;
  sc.tg = a_cv->hit.tangent;
  sc.bn = a_cv->hit.biTangent;
  sc.tc = a_cv->hit.texCoord;
  
  const BxDFResult evalData = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), a_globals, a_texStorage1, a_texStorage2, a_ptList);
  const float pdfFwdAt1W    = evalData.pdfRev;
  
  const float cosThetaOut1  = fmax(+dot(shadowRayDir, a_cv->hit.normal), DEPSILON);
  const float cosThetaOut2  = fmax(-dot(shadowRayDir, a_cv->hit.normal), DEPSILON);
  const bool  inverseCos    = ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(shadowRayDir, a_cv->hit.normal) < -0.01f);

  const float cosThetaOut   = inverseCos ? cosThetaOut2 : cosThetaOut1;  
  const float cosAtLight    = fmax(a_explicitSam.cosAtLight, DEPSILON);
  const float cosThetaPrev  = fmax(-dot(a_cv->ray_dir, a_cv->hit.normal), DEPSILON);
  
  const float3 brdfVal      = evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2;
  const float  pdfRevWP     = evalData.pdfFwd / fmax(cosThetaOut, DEPSILON);
  
  const float shadowDist    = length(a_cv->hit.pos - a_explicitSam.pos);
  const float GTerm         = cosThetaOut*cosAtLight / fmax(shadowDist*shadowDist, DEPSILON2);
  
  const LightPdfFwd lPdfFwd = lightPdfFwd(a_pLight, shadowRayDir, cosAtLight, a_globals, a_texStorage1, a_tableStorage);

  v0->pdfFwd = lPdfFwd.pdfA*a_lightPickProb;
  v0->pdfRev = 1.0f; // a_explicitSam.isPoint ? 0.0f : 1.0f;
  
  v1->pdfFwd = (lPdfFwd.pdfW / cosAtLight)*GTerm;
  v1->pdfRev = (evalData.pdfFwd == 0) ? -1.0f*GTerm : pdfRevWP*GTerm;
  
  if(a_camDepth > 1)
    v2->pdfFwd = (pdfFwdAt1W == 0.0f) ? -1.0f*a_cv->lastGTerm : (pdfFwdAt1W / cosThetaPrev)*a_cv->lastGTerm;
  
  float envMisMult = 1.0f;
  if(lightType(a_pLight) == PLAIN_LIGHT_TYPE_SKY_DOME) // special case for env light, we will ignore MMLT MIS weight later, in connect kernel
  {
    float lgtPdf = a_explicitSam.pdf*a_lightPickProb;
    envMisMult   = misWeightHeuristic(lgtPdf, evalData.pdfFwd); // (lgtPdf*lgtPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);
  }

  const float explicitPdfW = fmax(a_explicitSam.pdf, DEPSILON2);
  return bsdfClamping((1.0f/a_lightPickProb)*(a_explicitSam.color*envMisMult)*brdfVal / explicitPdfW);
}

/**
\brief  Connect end points in SBDPT (Stochastic connection BDPT). Don't trace ray and don't compute shadow. You must compute shadow outside of this procedure.
\param  a_lv          - light  vertex we want to connect
\param  a_cv          - camera vertex we want to connect
\param  a_depth       - total trace depth equal to s+t;

\param  a_globals     - engine globals
\param  a_mltStorage  - materials storage 
\param  a_texStorage1 - general texture storage
\param  a_texStorage2 - aux texture storage
\param  a_ptList      - proc tex list data

\param  vSplitBefore  - out pdfArea array[split-1] 
\param  vSplit        - out pdfArea array[split+0] 
\param  vSplitAfter   - out pdfArea array[split+1] 
\return connection throughput color without shadow

*/

static float3 ConnectEndPointsP(__private const PathVertex* a_lv, __private const PathVertex* a_cv, const int a_depth,
                                __global const EngineGlobals* a_globals, __global const float4* a_mltStorage, texture2d_t a_texStorage1, texture2d_t a_texStorage2, __private const ProcTextureList* a_ptList,
                                __private PdfVertex* vSplitBefore, __private PdfVertex* vSplit, __private PdfVertex* vSplitAfter)
{
  if (!a_lv->valid || !a_cv->valid)
    return make_float3(0, 0, 0);

  const float3 diff = a_cv->hit.pos - a_lv->hit.pos;
  const float dist2 = fmax(dot(diff, diff), DEPSILON2);
  const float  dist = sqrt(dist2);
  const float3 lToC = diff / dist; // normalize(a_cv.hit.pos - a_lv.hit.pos)

  float3 lightBRDF = make_float3(0,0,0);
  float  lightVPdfFwdW = 0.0f;
  float  lightVPdfRevW = 0.0f;
  float  signOfNormalL = 1.0f;
  {
    ShadeContext sc;
    sc.wp = a_lv->hit.pos;
    sc.l  = lToC;                  // try to swap them ?
    sc.v  = (-1.0f)*a_lv->ray_dir; // try to swap them ?
    sc.n  = a_lv->hit.normal;
    sc.fn = a_lv->hit.flatNormal;
    sc.tg = a_lv->hit.tangent;
    sc.bn = a_lv->hit.biTangent;
    sc.tc = a_lv->hit.texCoord;

    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mltStorage, a_lv->hit.matId);
    if(pHitMaterial != 0)
    {
      BxDFResult evalData = materialEval(pHitMaterial, &sc, (EVAL_FLAG_FWD_DIR), /* global data --> */ a_globals, a_texStorage1, a_texStorage2, a_ptList);
      lightBRDF     = evalData.brdf + evalData.btdf;
      lightVPdfFwdW = evalData.pdfFwd;
      lightVPdfRevW = evalData.pdfRev;

      const bool underSurfaceL = (dot(lToC, a_lv->hit.normal) < -0.01f);
      if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && underSurfaceL)
        signOfNormalL = -1.0f;
    }
  }

  float3 camBRDF = make_float3(0, 0, 0);
  float  camVPdfRevW = 0.0f;
  float  camVPdfFwdW = 0.0f;
  float  signOfNormalC = 1.0f;
  {
    ShadeContext sc;
    sc.wp = a_cv->hit.pos;
    sc.l  = (-1.0f)*lToC;           // try to swap them ?
    sc.v  = (-1.0f)*a_cv->ray_dir;  // try to swap them ?
    sc.n  = a_cv->hit.normal;
    sc.fn = a_cv->hit.flatNormal;
    sc.tg = a_cv->hit.tangent;
    sc.bn = a_cv->hit.biTangent;
    sc.tc = a_cv->hit.texCoord;

    __global const PlainMaterial* pHitMaterial = materialAt(a_globals, a_mltStorage, a_cv->hit.matId);
    if(pHitMaterial != 0)
    {                                                                                                                                             
      BxDFResult evalData = materialEval(pHitMaterial, &sc, (EVAL_FLAG_DEFAULT), /* global data --> */ a_globals, a_texStorage1, a_texStorage2, a_ptList);  
      camBRDF       = evalData.brdf + evalData.btdf;                                                                                               
      camVPdfRevW   = evalData.pdfFwd;                                                                                                            
      camVPdfFwdW   = evalData.pdfRev;

      const bool underSurfaceC = (dot((-1.0f)*lToC, a_cv->hit.normal) < -0.01f);
      if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && underSurfaceC)
        signOfNormalC = -1.0f;
    }
  }

  const float cosAtLightVertex      = +signOfNormalL*dot(a_lv->hit.normal, lToC); // 
  const float cosAtCameraVertex     = -signOfNormalC*dot(a_cv->hit.normal, lToC); // 

  const float cosAtLightVertexPrev  = -dot(a_lv->hit.normal, a_lv->ray_dir);
  const float cosAtCameraVertexPrev = -dot(a_cv->hit.normal, a_cv->ray_dir);

  const float GTerm = cosAtLightVertex*cosAtCameraVertex / dist2;

  if (GTerm < 0.0f) // underSurfaceL || underSurfaceC
    return make_float3(0, 0, 0);

  // calc remaining PDFs
  //
  const float lightPdfFwdWP  = lightVPdfFwdW / fmax(cosAtLightVertex,  DEPSILON2);
  const float cameraPdfRevWP = camVPdfRevW   / fmax(cosAtCameraVertex, DEPSILON2);

  vSplit->pdfFwd = (lightPdfFwdWP  == 0.0f) ? -1.0f*GTerm : lightPdfFwdWP*GTerm;   // let s=2,t=1 => (a_spit == s == 2)
  vSplit->pdfRev = (cameraPdfRevWP == 0.0f) ? -1.0f*GTerm : cameraPdfRevWP*GTerm;  // let s=2,t=1 => (a_spit == s == 2)

  vSplitBefore->pdfRev = (lightVPdfRevW == 0.0f) ? -1.0f*a_lv->lastGTerm : a_lv->lastGTerm*(lightVPdfRevW / fmax(cosAtLightVertexPrev, DEPSILON));

  if (a_depth > 3)
    vSplitAfter->pdfFwd = (camVPdfFwdW == 0.0f) ? -1.0f*a_cv->lastGTerm : a_cv->lastGTerm*(camVPdfFwdW / fmax(cosAtCameraVertexPrev, DEPSILON));

  const bool fwdCanNotBeEvaluated = (lightPdfFwdWP < DEPSILON2)  || (a_depth > 3 && camVPdfFwdW < DEPSILON2);
  const bool revCanNotBeEvaluated = (cameraPdfRevWP < DEPSILON2) || (lightVPdfRevW < DEPSILON2);

  if (fwdCanNotBeEvaluated && revCanNotBeEvaluated)
    return make_float3(0, 0, 0);

  //bool lessOrNan1 = (!isfinite(lightBRDF.x) || (lightBRDF.x < 0)) || (!isfinite(lightBRDF.y) || (lightBRDF.y < 0)) || (!isfinite(lightBRDF.z) || (lightBRDF.z < 0));
  //bool lessOrNan2 = (!isfinite(camBRDF.x)   || (camBRDF.x < 0)) || (!isfinite(camBRDF.y) || (camBRDF.y < 0)) || (!isfinite(camBRDF.z) || (camBRDF.z < 0));
  //
  //if (maxcomp(lightBRDF) >= 0.35f || maxcomp(camBRDF) >= 0.35f || lessOrNan1 || lessOrNan2)
  //{
  //  std::cout << lightBRDF.x << lightBRDF.y << lightBRDF.z << std::endl;
  //  std::cout << camBRDF.x   << camBRDF.y << camBRDF.z << std::endl;
  //  std::cout << "GTerm = "  << GTerm << std::endl;
  //}

  return bsdfClamping(lightBRDF*camBRDF*GTerm); // fmin(GTerm,1000.0f);
}

/**
\brief  Get environment color considering PT mis (i.e. 2 strategies)
\param  rayDir           - input direction of ray that is going to hit environment (i.e. it miss all surfaces and got outside the scene)
\param  misPrev          - previous bounce MIS info
\param  flags            - ray flags
\param  a_globals        - engine globals
\param  a_mltStorage     - material storage (needed to get previous bounce material info)
\param  a_pdfStorage     - pdf storage
\param  a_shadingTexture - standart texture storage

\return environment color considering PT mis

 Although this function was designed for PT only, it can be used for 3Way and MMLT due to env lights don't have forward sampler in our renderer

*/

static inline float3 environmentColor(float3 rayDir, MisData misPrev, unsigned int flags, 
                                      __global const EngineGlobals* a_globals, 
                                      __global const float4*        a_mltStorage, 
                                      __global const float4*        a_pdfStorage, 
                                      texture2d_t                   a_shadingTexture)
{
  if (a_globals->skyLightId == -1)
    return make_float3(0, 0, 0);

  unsigned int rayBounceNum  = unpackBounceNum(flags);
  unsigned int diffBounceNum = unpackBounceNumDiff(flags);

  __global const PlainLight* pEnvLight = lightAt(a_globals, a_globals->skyLightId); // in_lights + a_globals->skyLightId;

  float3 envColor = skyLightGetIntensityTexturedENV(pEnvLight, rayDir, a_globals, a_pdfStorage, a_shadingTexture);

  // //////////////////////////////////////////////////////////////////////////////////////////////

  if (rayBounceNum > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0))
  {
    float lgtPdf    = lightPdfSelectRev(pEnvLight)*skyLightEvalPDF(pEnvLight, rayDir, a_globals, a_pdfStorage);
    float bsdfPdf   = misPrev.matSamplePdf;
    float misWeight = misWeightHeuristic(bsdfPdf, lgtPdf); // (bsdfPdf*bsdfPdf) / (lgtPdf*lgtPdf + bsdfPdf*bsdfPdf);

    envColor *= misWeight;
  }
  
  if (misPrev.prevMaterialOffset >= 0)
  {
    __global const PlainMaterial* pPrevMaterial = materialAtOffset(a_mltStorage, misPrev.prevMaterialOffset);                            // in_plainData + misPrev.prevMaterialOffset;
    if(pPrevMaterial != 0)
    {
      bool disableCaustics = (diffBounceNum > 0) && !(a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS) && materialCastCaustics(pPrevMaterial); // and prev material cast caustics
      if (disableCaustics)
        envColor = make_float3(0, 0, 0);
    }
  }

  // \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

  return envColor;
}

/**
\brief  Depending on the current shadow matte mode stored in a_globals->varsI[HRT_SHADOW_MATTE_BACK_MODE] 
        (MODE_SPHERICAL or MODE_CAM_PROJECTED) get back(screen) or backEnv(ray_dir)

\return back enviromnent color

*/

static inline float3 backColorOfSecondEnv(float3 ray_dir, float2 screen, __global const EngineGlobals* a_globals, texture2d_t in_texStorage1)
{
  const float texCoordX = screen.x / a_globals->varsF[HRT_WIDTH_F];
  const float texCoordY = screen.y / a_globals->varsF[HRT_HEIGHT_F];
  const int offset      = textureHeaderOffset(a_globals, a_globals->varsI[HRT_SHADOW_MATTE_BACK]);
 
  const float3 backColorMult = make_float3(a_globals->varsF[HRT_SHADOW_MATTE_BACK_COLOR_X], 
                                           a_globals->varsF[HRT_SHADOW_MATTE_BACK_COLOR_Y],
                                           a_globals->varsF[HRT_SHADOW_MATTE_BACK_COLOR_Z]);

  float3 envColor;       
  
  if(a_globals->varsI[HRT_SHADOW_MATTE_BACK_MODE] == MODE_SPHERICAL)
  {
    float sintheta = 0.0f;
    const float2 texCoord = sphereMapTo2DTexCoord(ray_dir, &sintheta);
    envColor = backColorMult*to_float3(read_imagef_sw4(in_texStorage1 + offset, texCoord, TEX_CLAMP_U | TEX_CLAMP_V, true));
  }
  else
    envColor = backColorMult*to_float3(read_imagef_sw4(in_texStorage1 + offset, make_float2(texCoordX, texCoordY), TEX_CLAMP_U | TEX_CLAMP_V, true));
  
  if(a_globals->varsF[HRT_BACK_TEXINPUT_GAMMA] != 1.0f)
  {
    envColor.x = sRGBToLinear(envColor.x);
    envColor.y = sRGBToLinear(envColor.y);
    envColor.z = sRGBToLinear(envColor.z);
  }
  
  return envColor;
}

/**
\brief  Add Perez Sun and Camera mapped texture to function "environmentColor".
\param  ray_pos          - input origin of ray (needed for Direct Light cylinder-attenuation if enabled).
\param  ray_dir          - input direction of ray that is going to hit environment (i.e. it miss all surfaces and got outside the scene)
\param  misPrev          - previous bounce MIS info
\param  flags            - ray flags
\param  screenX          - input screen ray X for caera mapped textures
\param  screenY          - input screen ray Y for caera mapped textures
\param  a_globals        - engine globals
\param  in_mtlStorage    - material storage (needed to get previous bounce material info)
\param  in_pdfStorage    - pdf storage
\param  in_texStorage1   - standart texture storage

\return environment color considering PT mis

 Although this function was designed for PT only, it can be used for 3Way and MMLT due to env lights don't have forward sampler in our renderer

*/

static inline float3 environmentColorExtended(float3 ray_pos, float3 ray_dir, MisData misPrev, unsigned int flags, int screenX, int screenY,
                                              __global const EngineGlobals* a_globals, 
                                              __global const float4*        in_mtlStorage, 
                                              __global const float4*        in_pdfStorage, 
                                              texture2d_t                   in_texStorage1)
{
  const int hitId = hitDirectLight(ray_dir, a_globals); 
  
  float3 envColor = make_float3(0, 0, 0);
  if (hitId >= 0) // hit any sun light
  {
    __global const PlainLight* pLight = a_globals->suns + hitId;
    envColor = lightBaseColor(pLight)*directLightAttenuation(pLight, ray_pos);
    const float pdfW = directLightEvalPDF(pLight, ray_dir);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ((unpackBounceNum(flags) > 0 && !(a_globals->g_flags & HRT_STUPID_PT_MODE) && (misPrev.isSpecular == 0))) //#TODO: check this for bug with 2 hdr env light (test 335)
      envColor = make_float3(0, 0, 0);
    else if (((misPrev.isSpecular == 1) && (a_globals->g_flags & HRT_ENABLE_PT_CAUSTICS)) || (a_globals->g_flags & HRT_STUPID_PT_MODE))
      envColor *= (1.0f / pdfW);
    if ((a_globals->g_flags & HRT_3WAY_MIS_WEIGHTS) != 0) //#TODO: fix IBPT (???)
      envColor = make_float3(0, 0, 0); 
  }
  else
  {
    envColor = environmentColor(ray_dir, misPrev, flags, a_globals, in_mtlStorage, in_pdfStorage, in_texStorage1);
    
    const uint rayBounce     = unpackBounceNum(flags);
    unsigned int otherFlags  = unpackRayFlags(flags);
    const int backTextureId  = a_globals->varsI[HRT_SHADOW_MATTE_BACK];
    const bool transparent   = ((otherFlags & RAY_EVENT_T) != 0) && ((otherFlags & RAY_EVENT_D) == 0) && ((otherFlags & RAY_EVENT_G) == 0); 
    
    if (backTextureId != INVALID_TEXTURE && (rayBounce == 0 || transparent))
      envColor = backColorOfSecondEnv(ray_dir, make_float2((float)screenX + 0.5f, (float)screenY + 0.5f), a_globals, in_texStorage1);
  }

  return envColor;
}

/**
\brief  Get surface emission. Considering both material and light to get correct value. No MIS applied inside.
\param  ray_pos          - ray origin (needed for 'directLightAttenuation' and e.t.c. deep inside)
\param  ray_dir          - ray direction
\param  pSurfElem        - surface element 
\param  flags            - ray flags
\param  a_wasSpecular    - set this true if previous bounce was specular

\param  pLight           - pointer to light if surface is a light-surface, nullptr otherwise
\param  a_pHitMaterial   - surface material

\param  a_texStorage     - standart testure storage
\param  a_pdfStorage     - pdf storage
\param  a_globals        - engine globals
\param  a_ptl            - proc. textures data list.

\return surface emission

 Get surface emission. Considering both material and light to get correct value. No MIS applied inside.

*/

static inline float3 emissionEval(const float3 ray_pos, const float3 ray_dir,  __private const SurfaceHit* pSurfElem, const uint flags, const bool a_wasSpecular, 
                                  __global const PlainLight* pLight, __global const PlainMaterial* a_pHitMaterial, __global const int4* a_texStorage, 
                                  __global const float4* a_pdfStorage, __global const EngineGlobals* a_globals, __private ProcTextureList* a_ptl)
{
  const float3 normal = pSurfElem->hfi ? (-1.0f)*pSurfElem->normal : pSurfElem->normal;

  bool hasIES = false;
  if (a_globals->lightsNum > 0 && pLight != 0)
    hasIES = (lightFlags(pLight) & LIGHT_HAS_IES) != 0;

  if (dot(ray_dir, normal) >= 0.0f && !hasIES)
    return make_float3(0, 0, 0);
  
  float3 outPathColor = materialEvalEmission(a_pHitMaterial, ray_dir, normal, pSurfElem->texCoord, a_globals, a_texStorage, a_ptl); 
  
	if ((materialGetFlags(a_pHitMaterial) & PLAIN_MATERIAL_FORBID_EMISSIVE_GI) && unpackBounceNumDiff(flags) > 0)
		outPathColor = make_float3(0, 0, 0);

  if (a_globals->lightsNum > 0 && pLight != 0)
  {
    outPathColor = lightGetIntensity(pLight, ray_pos, ray_dir, normal, pSurfElem->texCoord, flags, a_wasSpecular, 
                                     a_globals, a_texStorage, a_pdfStorage); 
  }

  return outPathColor;
}

#endif

/**
 \file
 \brief Metropolis Light Transport kernels.

 */

#include "cglobals.h"
#include "crandom.h"
#include "cfetch.h"

#include "clight.h"
#include "cmaterial.h"
#include "cbidir.h"

#define SPLIT_DL_BY_GRAMMAR false

/**
\brief Evaluate contib function. Not used.
\param in_color   - input color
\param out_colors - output contrib value

*/
__kernel void MLTEvalContribFunc(__global const float4* in_color, __global float* out_colors, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_colors[tid] = contribFunc(to_float3(in_color[tid]));
}


inline int TabIndex(const int vertId, const int tid, const int iNumElements)
{
  return tid + vertId*iNumElements;
}

__kernel void MMLTInitCameraPath(__global   uint* restrict a_flags,
                                 __global float4* restrict a_color,
                                 __global int2*   restrict a_split,
                                 __global float4* restrict a_vertexSup,

                                 //__global RandomGen* restrict a_gens,
                                 //__global float*     restrict a_mmltrands,

                                 const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int d = MMLT_GPU_TEST_DEPTH;
  const int s = 2; 

  a_flags[tid] = packBounceNum(0, 1);
  a_color[tid] = make_float4(1,1,1,1);
  a_split[tid] = make_int2(d,s);
  
  PathVertex resVertex;
  resVertex.ray_dir      = make_float3(0,0,0);
  resVertex.accColor     = make_float3(0,0,0);   
  resVertex.valid        = false; //(a_currDepth == a_targetDepth);     // #TODO: dunno if this is correct ... 
  resVertex.hitLight     = false;
  resVertex.wasSpecOnly  = false;
  WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                            a_vertexSup);

}

__kernel void CopyAccColorTo(__global const float4* restrict in_vertexSup, 
                             __global       float4* restrict out_color,
                             const int   iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  PathVertex resVertex;
  ReadPathVertexSupplement(in_vertexSup, tid, iNumElements,
                           &resVertex);

  if(resVertex.valid && resVertex.hitLight)
    out_color[tid] = to_float4(resVertex.accColor, 0.0f);
  else
    out_color[tid] = make_float4(0,0,0,0);
}


__kernel void MMLTCameraPathBounce(__global   float4*        restrict a_rpos,
                                   __global   float4*        restrict a_rdir,
                                   __global   uint*          restrict a_flags,
                                   __global RandomGen*       restrict out_gens,
                                  
                                   __global const int2*      restrict in_splitInfo,
                                   __global const Lite_Hit*  restrict in_hits,
                                   __global const int*       restrict in_instLightInstId,
                                   __global const float4*    restrict in_surfaceHit,
                                   __global const float4*    restrict in_procTexData,

                                   __global float4*          restrict a_color,
                                   __global float4*          restrict a_normalPrev,    // (!) stote prev normal here, instead of 'a_thoroughput'
                                   __global MisData*         restrict a_misDataPrev,
                                   __global float4*          restrict a_fog,
                                   __global PdfVertex*       restrict a_pdfVert,       // (!) MMLT pdfArray 
                                   __global float4*          restrict a_vertexSup,     // (!) MMLT out Path Vertex supplemental to surfaceHit data

                                   __global const float4*    restrict in_texStorage1,    
                                   __global const float4*    restrict in_texStorage2,
                                   __global const float4*    restrict in_mtlStorage,
                                   __global const float4*    restrict in_pdfStorage,   //

                                   __global const EngineGlobals*  restrict a_globals,
                                   const int   iNumElements, 
                                   const float mLightSubPathCount)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid]; // #NOTE: what if ray miss object just recently .. don't we need to do soem thing in MMLT? See original code.

  // (0) Ray is outside of scene, hit environment
  //
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE) // #TODO: read environment! 
  {
    float3 envColor = make_float3(0,0,0);
    
    PathVertex resVertex;
    resVertex.ray_dir     = to_float3(a_rdir[tid]);
    resVertex.accColor    = envColor*to_float3(a_color[tid]);   
    resVertex.valid       = false; //(a_currDepth == a_targetDepth);     // #TODO: dunno if this is correct ... 
    resVertex.hitLight    = true;
    resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);
    
    flags        = packRayFlags(flags, RAY_IS_DEAD);
    a_flags[tid] = flags;
  } 

  if (!rayIsActiveU(flags)) 
    return;

  const __global float* a_rptr = 0;   /////////////////////////////////////////////////// #TODO: INIT THIS POINTER WITH MMLT RANDS !!!

  // (0) Read "IntegratorMMLT::CameraPath" arguments and calc ray hit
  //
  const int2 splitData = in_splitInfo[tid];
  const int  d = splitData.x;
  const int  s = splitData.y;
  const int  t = d - s;               // note that s=2 means 1 light bounce and one connection!!!

  const bool a_haveToHitLightSource = (s == 0); // when s == 0, use only camera strategy, so have to hit light at some depth level   
  const int  a_fullPathDepth        = d;       
  const int  a_targetDepth          = t;
  const int  a_currDepth            = unpackBounceNum(flags); // #NOTE: first bounce must be equal to 1                           
  const int  prevVertexId           = a_fullPathDepth - a_currDepth + 1; 

  //__global const PlainLight* pLight = lightAt(a_globals, lightOffset);
  SurfaceHit surfElem;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &surfElem);

 
  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfElem.matId);

  const float3 ray_pos      = to_float3(a_rpos[tid]);
  const float3 ray_dir      = to_float3(a_rdir[tid]);
  const float3 a_prevNormal = to_float3(a_normalPrev[tid]);
  
  // (1)
  //
  const float cosHere = fabs(dot(ray_dir, surfElem.normal));
  const float cosPrev = fabs(dot(ray_dir, a_prevNormal));
 
  float GTerm = 1.0f;
  if (a_currDepth == 1)
  {
    float3 camDirDummy; float zDepthDummy;
    const float imageToSurfaceFactor = CameraImageToSurfaceFactor(surfElem.pos, surfElem.normal, a_globals,
                                                                  &camDirDummy, &zDepthDummy);
    const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
   
    PdfVertex vertLast;
    vertLast.pdfRev = cameraPdfA;
    vertLast.pdfFwd = 1.0f;
    a_pdfVert[TabIndex(a_fullPathDepth, tid, iNumElements)] = vertLast; //  a_perThread->pdfArray[a_fullPathDepth]
  }
  else
  {
    const float dist     = length(ray_pos - surfElem.pos); 
    GTerm = cosHere*cosPrev / fmax(dist*dist, DEPSILON2);
  }

  // (2)
  //
  const Lite_Hit liteHit  = in_hits[tid];
  const MisData a_misPrev = a_misDataPrev[tid];

  ProcTextureList ptl;        
  InitProcTextureList(&ptl);  
  ReadProcTextureList(in_procTexData, tid, iNumElements, 
                      &ptl);
  
  const int lightOffset   = (a_globals->lightsNum == 0 || liteHit.instId < 0) ? -1 : in_instLightInstId[liteHit.instId]; // #TODO: refactor this into function!
  __global const PlainLight* pLight = lightAt(a_globals, lightOffset);

  const float3 emission = emissionEval(ray_pos, ray_dir, &surfElem, flags, (a_misPrev.isSpecular == 1), pLight,
                                       pHitMaterial, in_texStorage1, in_pdfStorage, a_globals, &ptl);
  
  
  if (dot(emission, emission) > 1e-3f)
  {    
    if (a_currDepth == a_targetDepth && a_haveToHitLightSource)
    {
      const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, a_globals, in_texStorage1, in_pdfStorage);
      const float pdfLightWP    = lPdfFwd.pdfW           / fmax(cosHere, DEPSILON);
      const float pdfMatRevWP   = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);
      
      {
        PdfVertex v0,v1;

        v0.pdfFwd = lPdfFwd.pdfA / ((float)a_globals->lightsNum);
        v0.pdfRev = 1.0f;

        v1.pdfFwd = pdfLightWP*GTerm;
        v1.pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfMatRevWP*GTerm;

        a_pdfVert[TabIndex(0, tid, iNumElements)] = v0;
        a_pdfVert[TabIndex(1, tid, iNumElements)] = v1;
      } 
      
      PathVertex resVertex;
      resVertex.ray_dir     = ray_dir;
      resVertex.accColor    = emission*to_float3(a_color[tid]);   
      resVertex.lastGTerm   = 1.0f; ///////////////////////////////// ?????????????????????????????????????????????????
      resVertex.valid       = true;
      resVertex.hitLight    = true; 
      resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;
      WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                                a_vertexSup);

      a_flags[tid] = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
      return;
    }
    else // this branch could probably change in future, for simple emissive materials
    {
      PathVertex resVertex;
      resVertex.ray_dir     = ray_dir;
      resVertex.accColor    = make_float3(0,0,0);
      resVertex.valid       = false;
      resVertex.hitLight    = true; 
      resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;
      WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                                a_vertexSup);

      a_flags[tid] = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
      return;
    } 
  }
  else if (a_currDepth == a_targetDepth && !a_haveToHitLightSource) // #NOTE: what if a_targetDepth == 1 ?
  {
    PathVertex resVertex;
    resVertex.ray_dir     = ray_dir;
    resVertex.accColor    = make_float3(1, 1, 1)*to_float3(a_color[tid]);
    resVertex.valid       = true;
    resVertex.hitLight    = false;
    resVertex.wasSpecOnly = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false; 
 
    if (a_targetDepth != 1)
    {
      const float lastPdfWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON); // we store them to calculate fwd and rev pdf later when we connect end points
      resVertex.lastGTerm   = GTerm;                                            // because right now we can not do this until we don't know the light vertex
     
      PdfVertex vcurr;
      vcurr.pdfFwd = 1.0f; // write it later, inside ConnectShadow or ConnectEndPoints
      vcurr.pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : GTerm*lastPdfWP;
      a_pdfVert[TabIndex(prevVertexId, tid, iNumElements)] = vcurr;
    }
    else
      resVertex.lastGTerm = 1.0f;
    
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);

    a_flags[tid] = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
    return;
  }
  

  // (3) sample material, eval reverse and forward pdfs
  //  
  float allRands[MMLT_FLOATS_PER_BOUNCE];
  {
    RandomGen gen  = out_gens[tid];
    gen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

    RndMatAll(&gen, a_rptr, a_currDepth-1, 0, 0, 0, ////////////////////////////// #TODO: fix a_currDepth-1 !!!!!!!!!!!!!
              allRands);

    out_gens[tid] = gen;
  }         
  
  int matOffset = materialOffset(a_globals, surfElem.matId);

  MatSample matSam; int localOffset = 0; 
  MaterialSampleAndEvalBxDF(pHitMaterial, allRands, &surfElem, ray_dir, make_float3(1,1,1), flags,
                            a_globals, in_texStorage1, in_texStorage2, &ptl, 
                            &matSam, &localOffset);
  
  matOffset    = matOffset    + localOffset*(sizeof(PlainMaterial)/sizeof(float4));
  pHitMaterial = pHitMaterial + localOffset;

  const float3 bxdfVal = matSam.color; // *(1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosNext  = fabs(dot(matSam.direction, surfElem.normal));

  if (a_currDepth == 1)
  {
    if (isPureSpecular(matSam))  //  ow ... but if we met specular reflection when tracing from camera, we must put 0 because this path cannot be sample by light strategy at all.
    {                            //  a_perThread->pdfArray[a_fullPathDepth].pdfFwd = 0.0f;
      PdfVertex vertLast = a_pdfVert[TabIndex(a_fullPathDepth, tid, iNumElements)];
      vertLast.pdfFwd    = 0.0f;
      a_pdfVert[TabIndex(a_fullPathDepth, tid, iNumElements)] = vertLast;
    }
  }
  else
  {
    PdfVertex prevVert;

    if (!isPureSpecular(matSam))
    {
      ShadeContext sc;
      sc.wp = surfElem.pos;
      sc.l  = (-1.0f)*ray_dir;  // fliped; if compare to normal PT
      sc.v  = matSam.direction; // fliped; if compare to normal PT
      sc.n  = surfElem.normal;
      sc.fn = surfElem.flatNormal;
      sc.tg = surfElem.tangent;
      sc.bn = surfElem.biTangent;
      sc.tc = surfElem.texCoord;

      const float pdfFwdW  = materialEval(pHitMaterial, &sc, false, false, 
                                          a_globals, in_texStorage1, in_texStorage2, &ptl).pdfFwd;
      const float pdfFwdWP = pdfFwdW / fmax(cosHere, DEPSILON);

      prevVert.pdfFwd = pdfFwdWP*GTerm;
    }
    else
      prevVert.pdfFwd = -1.0f*GTerm;

    const float pdfCamPrevWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);
    prevVert.pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfCamPrevWP*GTerm;
    
    a_pdfVert[TabIndex(prevVertexId, tid, iNumElements)] = prevVert;
  }

  // (4) proceed to next bounce
  //  
  float3 accColor   = to_float3(a_color[tid]);
  const bool stopDL = SPLIT_DL_BY_GRAMMAR ? flagsHaveOnlySpecular(flags) : false;

  accColor *= (bxdfVal*cosNext / fmax(matSam.pdf, DEPSILON2));
  if (stopDL && a_haveToHitLightSource && a_currDepth + 1 == a_targetDepth) // exclude direct light
    accColor = make_float3(0, 0, 0);

  flags = flagsNextBounce(flags, matSam, a_globals);
  if (maxcomp(accColor) < 0.00001f)
  {
    PathVertex resVertex;
    resVertex.ray_dir     = make_float3(0, 0, 0);
    resVertex.accColor    = make_float3(0, 0, 0);
    resVertex.valid       = false;
    resVertex.hitLight    = false;
    resVertex.wasSpecOnly = false; 
    resVertex.lastGTerm   = 1.0f;                                            
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);
    flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
  }

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  a_flags[tid] = flags;
  a_color[tid] = to_float4(accColor,    0.0f);
  a_rpos [tid] = to_float4(nextRay_pos, 0.0f);
  a_rdir [tid] = to_float4(nextRay_dir, 0.0f);

  MisData misNext            = makeInitialMisData(); 
  misNext.matSamplePdf       = matSam.pdf;
  misNext.isSpecular         = (int)isPureSpecular(matSam);
  misNext.prevMaterialOffset = matOffset;
  misNext.cosThetaPrev       = fabs(+dot(nextRay_dir, surfElem.normal)); // update it withCosNextActually ...
  a_misDataPrev[tid]         = misNext;
}



__kernel void MMLTLightSampleForward(__global   float4*        restrict a_rpos,
                                     __global   float4*        restrict a_rdir,
                                     __global   uint*          restrict a_flags,
                                     __global RandomGen*       restrict out_gens,
                                    
                                     __global float4*          restrict a_color,
                                     __global PdfVertex*       restrict a_pdfVert,       // (!) MMLT pdfArray 
                                     __global float4*          restrict a_vertexSup,     // (!) MMLT out Path Vertex supplemental to surfaceHit data
                                     __global int*             restrict a_spec,          // (!) MMLTLightPathBounce only !!! prev bounce is specular.
  
                                     __global const float4*        restrict in_texStorage1,    
                                     __global const float4*        restrict in_pdfStorage,   //
                                     __global const EngineGlobals* restrict a_globals,
                                     const int   iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const __global float* a_rptr = 0;   /////////////////////////////////////////////////// #TODO: INIT THIS POINTER WITH MMLT RANDS !!!

  LightGroup2 lightRands;
  {
    RandomGen gen  = out_gens[tid];
    gen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

    RndLightMMLT(&gen, a_rptr,
                 &lightRands);

    out_gens[tid] = gen;
  }         

  float lightPickProb = 1.0f;
  const int lightId = SelectRandomLightFwd(lightRands.group2.z, a_globals,
                                           &lightPickProb);
  
  __global const PlainLight* pLight = lightAt(a_globals, lightId);
  
  LightSampleFwd sample;
  LightSampleForward(pLight, lightRands.group1, make_float2(lightRands.group2.x, lightRands.group2.y), 
                     a_globals, in_texStorage1, in_pdfStorage,
                     &sample);
  
  {
    PdfVertex v0;
    v0.pdfFwd = sample.pdfA*lightPickProb;
    v0.pdfRev = 1.0f;
    a_pdfVert[TabIndex(0, tid, iNumElements)] = v0;
  }

  float3 color = (1.0f/lightPickProb)*sample.color/(sample.pdfA*sample.pdfW);

  {  
    PathVertex lv;
    InitPathVertex(&lv);
    WritePathVertexSupplement(&lv, tid, iNumElements, 
                              a_vertexSup);
  }

  a_flags[tid] = packBounceNum(0, 1);
  a_color[tid] = to_float4(color,      0.0f);
  a_rpos [tid] = to_float4(sample.pos, sample.cosTheta);
  a_rdir [tid] = to_float4(sample.dir, sample.pdfW);
  a_spec [tid] = 0;
}

__kernel void MMLTLightPathBounce (__global   float4*        restrict a_rpos,
                                   __global   float4*        restrict a_rdir,
                                   __global   uint*          restrict a_flags,
                                   __global RandomGen*       restrict out_gens,
                                  
                                   __global const int2*      restrict in_splitInfo,
                                   __global const float4*    restrict in_surfaceHit,
                                   __global const float4*    restrict in_procTexData,

                                   __global float4*          restrict a_color,
                                   __global int*             restrict a_prevSpec,
                                   __global float4*          restrict a_fog,
                                   __global PdfVertex*       restrict a_pdfVert,       // (!) MMLT pdfArray 
                                   __global float4*          restrict a_vertexSup,     // (!) MMLT out Path Vertex supplemental to surfaceHit data

                                   __global const float4*    restrict in_texStorage1,    
                                   __global const float4*    restrict in_texStorage2,
                                   __global const float4*    restrict in_mtlStorage,
                                   __global const float4*    restrict in_pdfStorage,   //

                                   __global const EngineGlobals*  restrict a_globals,
                                   const int   iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  uint flags = a_flags[tid];
 
  // (0) Ray is outside of scene, hit environment
  //
  if (unpackRayFlags(flags) & RAY_GRAMMAR_OUT_OF_SCENE)  // #TODO: read environment! 
  {    
    PathVertex resVertex;
    resVertex.ray_dir     = make_float3(0,0,0);
    resVertex.accColor    = make_float3(0,0,0);   
    resVertex.valid       = false; //(a_currDepth == a_targetDepth);     // #TODO: dunno if this is correct ... 
    resVertex.hitLight    = true;
    resVertex.wasSpecOnly = false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);
    
    flags        = packRayFlags(flags, RAY_IS_DEAD);
    a_flags[tid] = flags;
  } 

  const int a_currDepth = unpackBounceNum(flags);
  const int2 splitData  = in_splitInfo[tid];
  const int  d = splitData.x;
  const int  s = splitData.y;
  const int  a_lightTraceDepth = s - 1;

  if (!rayIsActiveU(flags) || a_currDepth > a_lightTraceDepth) 
    return;

  const __global float* a_rptr = 0;   /////////////////////////////////////////////////// #TODO: INIT THIS POINTER WITH MMLT RANDS !!!

  SurfaceHit surfElem;
  ReadSurfaceHit(in_surfaceHit, tid, iNumElements, 
                 &surfElem);

  const float4 rpos_data = a_rpos[tid];
  const float4 rdir_data = a_rdir[tid];

  const float3 ray_pos = to_float3(rpos_data); const float a_prevLightCos = rpos_data.w;
  const float3 ray_dir = to_float3(rdir_data); const float a_prevPdf      = rdir_data.w;

  const float cosPrev = fabs(a_prevLightCos);
  const float cosCurr = fabs(-dot(ray_dir, surfElem.normal));
  const float dist    = length(surfElem.pos - ray_pos);

  // eval forward pdf
  //
  const float GTermPrev = (a_prevLightCos*cosCurr / fmax(dist*dist, DEPSILON2));
  const float prevPdfWP = a_prevPdf / fmax(a_prevLightCos, DEPSILON);
  
  const bool a_wasSpecular = (a_prevSpec[tid] == 1); 
  
  {
    PdfVertex vCurr;
    if (!a_wasSpecular)
      vCurr.pdfFwd = prevPdfWP*GTermPrev;
    else
      vCurr.pdfFwd = -1.0f*GTermPrev;
    vCurr.pdfRev = 1.0f;                                          //#NOTE: override it later!
    a_pdfVert[TabIndex(a_currDepth, tid, iNumElements)] = vCurr;
  }

  // are we done with LT pass ?
  //
  if (a_currDepth == a_lightTraceDepth)
  {
    PathVertex resVertex;
    resVertex.ray_dir     = ray_dir;
    resVertex.accColor    = to_float3(a_color[tid]);
    resVertex.lastGTerm   = GTermPrev;
    resVertex.valid       = true;
    resVertex.hitLight    = false; 
    resVertex.wasSpecOnly = false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);

    a_flags[tid] = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
    return;
  }

  // not done, next bounce
  // 
  ProcTextureList ptl;        
  InitProcTextureList(&ptl);  
  ReadProcTextureList(in_procTexData, tid, iNumElements, 
                      &ptl);

  __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, surfElem.matId);
  
  float allRands[MMLT_FLOATS_PER_BOUNCE];
  {
    RandomGen gen  = out_gens[tid];
    gen.maxNumbers = a_globals->varsI[HRT_MLT_MAX_NUMBERS];

    RndMatAll(&gen, a_rptr, a_currDepth-1, 0, 0, 0,     
              allRands);

    out_gens[tid] = gen;
  }         
  
  int matOffset = materialOffset(a_globals, surfElem.matId);
  
  MatSample matSam; int localOffset = 0; 
  MaterialSampleAndEvalBxDF(pHitMaterial, allRands, &surfElem, ray_dir, make_float3(1,1,1), flags,
                            a_globals, in_texStorage1, in_texStorage2, &ptl, 
                            &matSam, &localOffset);

  matOffset    = matOffset    + localOffset*(sizeof(PlainMaterial)/sizeof(float4));
  pHitMaterial = pHitMaterial + localOffset;
  
  const float cosNext  = fabs(+dot(matSam.direction, surfElem.normal));

  // calc new ray
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);
  
  // If we sampled specular event, then the reverse probability
  // cannot be evaluated, but we know it is exactly the same as
  // forward probability, so just set it. If non-specular event happened,
  // we evaluate the pdf
  //
  PdfVertex vCurr = a_pdfVert[TabIndex(a_currDepth, tid, iNumElements)];

  if (!isPureSpecular(matSam))
  {
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = (-1.0f)*ray_dir;
    sc.v  = (-1.0f)*nextRay_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const float pdfW         = materialEval(pHitMaterial, &sc, false, false, a_globals, in_texStorage1, in_texStorage2, &ptl).pdfFwd;
    const float prevPdfRevWP = pdfW / fmax(cosCurr, DEPSILON);
    vCurr.pdfRev = prevPdfRevWP*GTermPrev;
  }
  else
  {
    vCurr.pdfRev = -1.0f*GTermPrev;
  }
  
  a_pdfVert[TabIndex(a_currDepth, tid, iNumElements)] = vCurr;
  
  const float3 accColor = to_float3(a_color[tid])*matSam.color*cosNext*(1.0f / fmax(matSam.pdf, DEPSILON2));
  
  a_color[tid] = to_float4(accColor, 0.0f);
  a_rpos [tid] = to_float4(nextRay_pos, cosNext);
  a_rdir [tid] = to_float4(nextRay_dir, matSam.pdf);

  flags = flagsNextBounce(flags, matSam, a_globals);
  if (maxcomp(accColor) < 0.00001f)
  {
    PathVertex resVertex;
    resVertex.ray_dir     = ray_dir;
    resVertex.accColor    = to_float3(a_color[tid]);
    resVertex.lastGTerm   = GTermPrev;
    resVertex.valid       = false;
    resVertex.hitLight    = false; 
    resVertex.wasSpecOnly = false;
    WritePathVertexSupplement(&resVertex, tid, iNumElements, 
                              a_vertexSup);

    flags = packRayFlags(flags, unpackRayFlags(flags) | RAY_IS_DEAD);
  }

  a_flags   [tid] = flags;
  a_prevSpec[tid] = isPureSpecular(matSam) ? 1 : 0;
  
  
}


__kernel void MMLTMakeShadowRay(__global const int2  *  restrict in_splitInfo,
                                __global const float4*  restrict in_lv_hit,
                                __global const float4*  restrict in_lv_sup,
                                __global const float4*  restrict in_cv_hit,
                                __global const float4*  restrict in_cv_sup,
                                __global       float4*  restrict out_ray_pos,
                                __global       float4*  restrict out_ray_dir,
                                __global       int   *  restrict out_rflags,

                                __global const float4*         restrict in_mtlStorage,
                                __global const EngineGlobals*  restrict a_globals,
                                const int iNumElements)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;
  
  const int2 splitData      = in_splitInfo[tid];
  const int d               = splitData.x;
  const int s               = splitData.y;
  const int t               = d - s;  // note that s=2 means 1 light bounce and one connection!!!
  const int lightTraceDepth = s - 1;  // because the last light path is a connection anyway - to camera or to camera path
  const int camTraceDepth   = t;      //

  const float topUp = 1e10f;

  out_ray_pos[tid] = make_float4(0,topUp, 0, 0);
  out_ray_dir[tid] = make_float4(0,1,0,0);
  out_rflags [tid] = 0;
  
  PathVertex lv;
  ReadSurfaceHit(in_lv_hit, tid, iNumElements, 
                 &lv.hit);
  ReadPathVertexSupplement(in_lv_sup, tid, iNumElements, 
                           &lv);
  
  PathVertex cv;
  ReadSurfaceHit(in_cv_hit, tid, iNumElements, 
                 &cv.hit);
  ReadPathVertexSupplement(in_cv_sup, tid, iNumElements, 
                           &cv);

  /*
  if (lv.hit.matId >= 0)
  {
    float3 camDir; float zDepth;
    const float imageToSurfaceFactor = CameraImageToSurfaceFactor(lv.hit.pos, lv.hit.normal, a_globals,
                                                                  &camDir, &zDepth);
  
    float signOfNormal = 1.0f;
    if (lv.hit.matId >= 0)
    {
      __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, lv.hit.matId);
      if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, lv.hit.normal) < -0.01f)
        signOfNormal *= -1.0f;
    }
  
    out_ray_pos[tid] = to_float4(lv.hit.pos + epsilonOfPos(lv.hit.pos)*signOfNormal*lv.hit.normal, zDepth); // OffsRayPos(lv.hit.pos, lv.hit.normal, camDir);
    out_ray_dir[tid] = to_float4(camDir, as_float(-1));
  }
  */

  if (lightTraceDepth == -1)        // (3.1) -1 means we have full camera path, no conection is needed
  {

  }
  else
  {
    if (camTraceDepth == 0)         // (3.2) connect light vertex to camera (light tracing connection)
    {
      if (lv.valid)
      {
         float3 camDir; float zDepth;
         const float imageToSurfaceFactor = CameraImageToSurfaceFactor(lv.hit.pos, lv.hit.normal, a_globals,
                                                                       &camDir, &zDepth);
         float signOfNormal = 1.0f;
         if (lv.hit.matId >= 0)
         {
           __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, lv.hit.matId);
           if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, lv.hit.normal) < -0.01f)
             signOfNormal *= -1.0f;
         }

         out_ray_pos[tid] = to_float4(lv.hit.pos + epsilonOfPos(lv.hit.pos)*signOfNormal*lv.hit.normal, zDepth); // OffsRayPos(lv.hit.pos, lv.hit.normal, camDir);
         out_ray_dir[tid] = to_float4(camDir, as_float(-1));
      }
    }
    else if (lightTraceDepth == 0)  // (3.3) connect camera vertex to light (shadow ray)
    {
      if (cv.valid && !cv.wasSpecOnly) // cv.wasSpecOnly exclude direct light actually
      {
        
      }
    }
    else                            // (3.4) connect light and camera vertices (bidir connection)
    {
      if (cv.valid)
      {
        
      }
    }
  }
  

}

__kernel void MMLTConnect(__global const int2  *  restrict in_splitInfo,
                          __global const float4*  restrict in_lv_hit,
                          __global const float4*  restrict in_lv_sup,
                          __global const float4*  restrict in_cv_hit,
                          __global const float4*  restrict in_cv_sup,
                          __global const float4*  restrict in_procTexData,
                          __global const ushort4* restrict in_shadow,

                          __global PdfVertex*     restrict a_pdfVert,
                          __global       float4*  restrict out_color,
                          __global int2*          restrict out_zind,

                          __global const float4*         restrict in_texStorage1,    
                          __global const float4*         restrict in_texStorage2,
                          __global const float4*         restrict in_mtlStorage,
                          __global const float4*         restrict in_pdfStorage,  
                          __global const EngineGlobals*  restrict a_globals,
                          __constant ushort*             restrict a_mortonTable256,
                          const int iNumElements, const float mLightSubPathCount)
{
  const int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  const int2 splitData = in_splitInfo[tid];
  const int d = splitData.x;
  const int s = splitData.y;
  const int t = d - s;                // note that s=2 means 1 light bounce and one connection!!!
  const int lightTraceDepth = s - 1;  // because the last light path is a connection anyway - to camera or to camera path
  const int camTraceDepth   = t;      //
  
  PathVertex lv;
  ReadSurfaceHit(in_lv_hit, tid, iNumElements, 
                 &lv.hit);
  ReadPathVertexSupplement(in_lv_sup, tid, iNumElements, 
                           &lv);

  PathVertex cv;
  ReadSurfaceHit(in_cv_hit, tid, iNumElements, 
                 &cv.hit);
  ReadPathVertexSupplement(in_cv_sup, tid, iNumElements, 
                           &cv);

  ProcTextureList ptl;        
  InitProcTextureList(&ptl);  
  ReadProcTextureList(in_procTexData, tid, iNumElements, 
                      &ptl);


  float3 sampleColor  = make_float3(0,0,0);
  int x = 65535, y = 65535;
  
  if (lightTraceDepth == -1)        // (3.1) -1 means we have full camera path, no conection is needed
  {
    sampleColor = cv.accColor;
  }
  else
  {
    if (camTraceDepth == 0)         // (3.2) connect light vertex to camera (light tracing connection)
    {
      if (lv.valid)
      {
        float3 camDir; float zDepth;
        const float imageToSurfaceFactor = CameraImageToSurfaceFactor(lv.hit.pos, lv.hit.normal, a_globals,
                                                                      &camDir, &zDepth);
      
        __global const PlainMaterial* pHitMaterial = materialAt(a_globals, in_mtlStorage, lv.hit.matId);
        float signOfNormal = 1.0f;
        if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, lv.hit.normal) < -0.01f)
          signOfNormal = -1.0f;
    
        PdfVertex v0, v1;
        v0 = a_pdfVert[TabIndex(lightTraceDepth + 0, tid, iNumElements)];

        sampleColor = ConnectEyeP(&lv, mLightSubPathCount, camDir, imageToSurfaceFactor,
                                  a_globals, in_mtlStorage, in_texStorage1, in_texStorage2, &ptl,
                                  &v0, &v1, &x, &y);

        if (!isfinite(sampleColor.x) || !isfinite(sampleColor.y) || !isfinite(sampleColor.z) || imageToSurfaceFactor <= 0.0f)
          sampleColor = make_float3(0, 0, 0);

        a_pdfVert[TabIndex(lightTraceDepth + 0, tid, iNumElements)] = v0;
        a_pdfVert[TabIndex(lightTraceDepth + 1, tid, iNumElements)] = v1;
      }
    }
    else if (lightTraceDepth == 0)  // (3.3) connect camera vertex to light (shadow ray)
    {
      //if (cv.valid && !cv.wasSpecOnly) // cv.wasSpecOnly exclude direct light actually
      //{
      //  //float3 explicitColor = ConnectShadow(cv, &PerThread(), t);
      //  //sampleColor = cv.accColor*explicitColor;
      //}
    }
    else                            // (3.4) connect light and camera vertices (bidir connection)
    {
      //if (cv.valid)
      //{
      //  //float3 explicitColor = ConnectEndPoints(lv, cv, s, d, &PerThread());
      //  //sampleColor = cv.accColor*explicitColor*lv.accColor;
      //}
    }
  }

  sampleColor *= decompressShadow(in_shadow[tid]);

  if (!isfinite(sampleColor.x) || !isfinite(sampleColor.y) || !isfinite(sampleColor.z))
    sampleColor = make_float3(0, 0, 0);

  // #TODO: implement MIS weights here ... 
  //

  const int zid = (int)ZIndex(x, y, a_mortonTable256);
  if(out_zind != 0)
    out_zind[tid] = make_int2(zid, tid);  
  out_color [tid] = to_float4(sampleColor, as_float(packXY1616(x,y)));
}


#include <omp.h>
#include "CPUExp_Integrators.h"

void IntegratorThreeWay::SetMaxDepth(int a_depth)
{
  m_maxDepth = a_depth;
}

void IntegratorThreeWay::GetImageHDR(float4* a_imageHDR, int w, int h) const
{
  if (w != m_width || h != m_height)
  {
    std::cout << "IntegratorThreeWay::GetImageHDR, bad resolution" << std::endl;
    return;
  }

  const float scaleInv = 1.0f / float(m_spp + 1);

  #pragma omp parallel for
  for (int i = 0; i < m_width*m_height; i++)
  {
    float4 color = m_hdrData[i];
    a_imageHDR[i] = scaleInv*color;
  }
}

void IntegratorThreeWay::DoPass(std::vector<uint>& a_imageLDR)
{
  const int samplesPerPass = m_width*m_height;
  mLightSubPathCount = float(samplesPerPass);

  #pragma omp parallel for
  for (int i = 0; i < samplesPerPass; i++)
    DoLightPath();

  #pragma omp parallel for
  for (int y = 0; y < m_height; y++)
  {
    for (int x = 0; x < m_width; x++)
    {
      float3 ray_pos, ray_dir;
      std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);
  
      const float3 color = PathTrace(ray_pos, ray_dir);
      
      //const float maxCol = maxcomp(color);
      //if (maxCol > 10.0f && m_debugFirstBounceDiffuse && ThreadId() == 0)
      //  std::cout << color.x << " " << color.y << " " << color.z << std::endl;

      m_hdrData[y*m_width + x] += to_float4(color, 0.0f);
    }
  }

  constexpr float gammaPow = 1.0f / 2.2f;
  const float scaleInv = 1.0f / float(m_spp + 1);

  #pragma omp parallel for
  for (int i = 0; i < int(a_imageLDR.size()); i++)
  {
    float4 color = m_hdrData[i];

    color.x = powf(clamp(color.x*scaleInv, 0.0f, 1.0f), gammaPow);
    color.y = powf(clamp(color.y*scaleInv, 0.0f, 1.0f), gammaPow);
    color.z = powf(clamp(color.z*scaleInv, 0.0f, 1.0f), gammaPow);
    color.w = 1.0f;

    a_imageLDR[i] = RealColorToUint32(color);
  }

  //if (m_spp % 10 == 0)
  //  DebugSaveNoiseImage();

  RandomizeAllGenerators();

  std::cout << "IntegratorThreeWay: spp  = " << m_spp << std::endl;
  m_spp++;
}

void SaveLightSample(LightSampleFwd a_sam)
{
  static std::ofstream g_sout("z_lingtSample.txt");
  g_sout << a_sam.color.x  << " " << a_sam.color.y << " " << a_sam.color.z << " ";
  g_sout << a_sam.cosTheta << " ";
  g_sout << a_sam.dir.x    << " " << a_sam.dir.y << " " << a_sam.dir.z << " ";
  g_sout << a_sam.isPoint  << " ";
  g_sout << a_sam.pdfA     << " ";
  g_sout << a_sam.pdfW     << " ";
  g_sout << a_sam.pos.x    << " " << a_sam.pos.y << " " << a_sam.pos.z << std::endl;
}


void IntegratorThreeWay::DoLightPath()
{
  auto& rgen = randomGen();

  float lightPickProb = 1.0f;
  const int lightId = SelectRandomLightFwd(rndFloat2(&rgen), m_pGlobals,
                                           &lightPickProb);

  const PlainLight* pLight = lightAt(m_pGlobals, lightId);

  const float4 rands1 = rndFloat4(&rgen);
  const float2 rands2 = rndFloat2(&rgen);

  LightSampleFwd sample;
  LightSampleForward(pLight, rands1, rands2, m_pGlobals, m_texStorage, m_pdfStorage,
                     &sample);

  PerThread().pdfLightA0         = sample.pdfA;  // #TODO: move this to some other structure
  PerThread().selectedLightIdFwd = lightId;

  PerRayAcc acc0;
  float3 color     = sample.color/fmax(lightPickProb*sample.pdfA*sample.pdfW, DEPSILON2);
  acc0.pdfLightWP  = sample.pdfW/ fmax(sample.cosTheta, DEPSILON);
  acc0.pdfCameraWP = 1.0f;
  acc0.pdfGTerm    = 1.0f;
  acc0.pdfCamA0    = 1.0f;

  //SaveLightSample(sample);

  TraceLightPath(sample.pos, sample.dir, 1, sample.cosTheta, sample.pdfW, 
                 &acc0, color);
}

void IntegratorThreeWay::TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float a_prevLightCos, float a_prevPdf, 
                                        PerRayAcc* a_pAccData, float3 a_color)
{
  if (a_currDepth >= m_maxDepth)
    return;

  auto hit = rayTrace(ray_pos, ray_dir);
  if (!HitSome(hit))
    return;

  const SurfaceHit     surfElem     = surfaceEval(ray_pos, ray_dir, hit);
  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample      matSam       = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));

  // calc new ray
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  const float cosPrev = fabs(a_prevLightCos);
  const float cosCurr = fabs(-dot(ray_dir, surfElem.normal));
  const float cosNext = fabs(+dot(nextRay_dir, surfElem.normal));
  const float dist    = length(surfElem.pos - ray_pos);

  const float GTerm   = (a_prevLightCos*cosCurr / fmax(dist*dist, DEPSILON2));

  a_pAccData->pdfGTerm *= GTerm;
  if (a_currDepth == 1)
  {
    a_pAccData->pdfCamA0     = GTerm; // spetial case, multyply it by pdf later ... 
    const PlainLight* pLight = lightAt(m_pGlobals, PerThread().selectedLightIdFwd);
    a_pAccData->pdfSelectRev = lightPdfSelectRev(pLight);
  }

  ConnectEye(surfElem, ray_pos, ray_dir, a_currDepth, 
             a_pAccData, a_color);

  // eval reverse pdf
  //
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

    const float pdfW = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage).pdfFwd;

    a_pAccData->pdfCameraWP *= (pdfW/fmax(cosCurr, DEPSILON));
    a_pAccData->pdfLightWP  *= (matSam.pdf / fmax(cosNext, DEPSILON));

    if (a_currDepth == 1)
      a_pAccData->pdfCamA0  *= (pdfW/fmax(cosCurr, DEPSILON)); // now pdfRevA0 will do store correct product pdfWP[0]*G[0] (if [0] means light)
  }
  else
  {
    a_pAccData->pdfCameraWP *= 1.0f;
    a_pAccData->pdfLightWP  *= 1.0f;

    //if (a_currDepth == 1)
      //a_pAccData->pdfCameraWP = 0.0f;
  }

  a_color *= matSam.color*cosNext* (1.0f / fmax(matSam.pdf, DEPSILON));

  TraceLightPath(nextRay_pos, nextRay_dir, a_currDepth + 1, cosNext, matSam.pdf, 
                 a_pAccData, a_color);
}

void IntegratorThreeWay::ConnectEye(SurfaceHit a_hit, float3 ray_pos, float3 ray_dir, int a_currBounce, 
                                    PerRayAcc* a_pAccData, float3 a_color)
{
  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(a_hit.pos, a_hit.normal, m_pGlobals,
                                                                &camDir, &zDepth);
  if (imageToSurfaceFactor <= 0.0f)
    return;

  float  signOfNormal = 1.0f;
  float  pdfRevW      = 1.0f;
  float3 colorConnect = make_float3(1, 1, 1);
  {
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, a_hit.matId);
    if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, a_hit.normal) < -0.01f)
      signOfNormal = -1.0f;

    ShadeContext sc;
    sc.wp = a_hit.pos;
    sc.l  = camDir;          // seems like that sc.l = camDir, see smallVCM
    sc.v  = (-1.0f)*ray_dir; // seems like that sc.v = (-1.0f)*ray_dir, see smallVCM
    sc.n  = a_hit.normal;
    sc.fn = a_hit.flatNormal;
    sc.tg = a_hit.tangent;
    sc.bn = a_hit.biTangent;
    sc.tc = a_hit.texCoord;

    auto colorAndPdf = materialEval(pHitMaterial, &sc, false, true, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage);
    colorConnect     = colorAndPdf.brdf + colorAndPdf.btdf;
    pdfRevW          = colorAndPdf.pdfRev;
  }

  const float cosCurr  = fabs(dot(ray_dir, a_hit.normal));
  const float pdfRevWP = pdfRevW / fmax(cosCurr,DEPSILON); // pdfW po pdfWP

  float pdfCamA0 = a_pAccData->pdfCamA0;
  if (a_currBounce == 1)
    pdfCamA0 *= pdfRevWP; // see pdfRevWP? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.

  const float cancelImplicitLightHitPdf = (1.0f / fmax(pdfCamA0, DEPSILON2));

  const PlainLight* pLight     = lightAt(m_pGlobals, PerThread().selectedLightIdFwd);
  const float lightPickProbFwd = lightPdfSelectFwd(pLight, m_pGlobals);
  const float lightPickProbRev = a_pAccData->pdfSelectRev;

  // We put the virtual image plane at such a distance from the camera origin
  // that the pixel area is one and thus the image plane sampling pdf is 1.
  // The area pdf of a_hit.pos as sampled from the camera is then equal to
  // the conversion factor from image plane area density to surface area density
  //
  const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
  const float lightPdfA  = PerThread().pdfLightA0;

  const float pdfAccFwdA = 1.0f*a_pAccData->pdfLightWP*a_pAccData->pdfGTerm*(lightPdfA*lightPickProbFwd);
  const float pdfAccRevA = cameraPdfA*(pdfRevWP*a_pAccData->pdfCameraWP)*a_pAccData->pdfGTerm; // see pdfRevWP? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.
                                                                                               // we didn't eval reverse pdf yet. Imagine light ray hit surface and we immediately connect.
  const float pdfAccExpA = cameraPdfA*(pdfRevWP*a_pAccData->pdfCameraWP)*a_pAccData->pdfGTerm*(cancelImplicitLightHitPdf*(lightPdfA*lightPickProbRev));

  const float misWeight  = misWeightHeuristic3(pdfAccFwdA, pdfAccRevA, pdfAccExpA);

  ///////////////////////////////////////////////////////////////////////////////

  // We divide the contribution by surfaceToImageFactor to convert the(already
  // divided) pdf from surface area to image plane area, w.r.t. which the
  // pixel integral is actually defined. We also divide by the number of samples
  // this technique makes, which is equal to the number of light sub-paths
  //
  const float3 sampleColor = misWeight*a_color*(colorConnect*imageToSurfaceFactor/mLightSubPathCount);

  if (dot(sampleColor, sampleColor) > 1e-20f) // add final result to image
  {
    auto hit = rayTrace(a_hit.pos + epsilonOfPos(a_hit.pos)*signOfNormal*a_hit.normal, camDir);

    if (!HitSome(hit) || hit.t > zDepth)
    {
      const float2 posScreenSpace = worldPosToScreenSpace(a_hit.pos, m_pGlobals);

      int x = int(posScreenSpace.x + 0.5f);
      int y = int(posScreenSpace.y + 0.5f);

      if(x >= 0 && x < m_width && y >=0 && y < m_height)
      { 
        const int offset = y*m_width + x;
        
        #pragma omp atomic
        m_hdrData[offset].x += sampleColor.x;
        #pragma omp atomic
        m_hdrData[offset].y += sampleColor.y;
        #pragma omp atomic
        m_hdrData[offset].z += sampleColor.z;
      }
    }
  }
}


float3  IntegratorThreeWay::PathTraceAcc(float3 ray_pos, float3 ray_dir, const float a_cosPrev, MisData misPrev, int a_currDepth, uint flags,
                                         SurfaceHit* pFirstHit, PerRayAcc* a_accData)
{
  if (a_currDepth >= m_maxDepth)
    return float3(0,0,0);

  const Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
  {
    //a_accData->pdfLightW *= ... ; // #TODO: add pdf for environmentlight here and apply MIS
    return EnviromnentColor(ray_dir, misPrev, flags);
  }

  const SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  if (pFirstHit != nullptr && a_currDepth == 0)
    (*pFirstHit) = surfElem;

  PerRayAcc prevData = (*a_accData); // for 3 bounce we need to store (p0*G0)*(p1*G1) and do not include (p2*G2) to we could replace it with explicit strategy pdf

  float3 camDirDummy; float zDepthDummy;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(pFirstHit->pos, pFirstHit->normal, m_pGlobals,
                                                                &camDirDummy, &zDepthDummy);

  const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;

  const float cosHere = fabs(dot(ray_dir, surfElem.normal));
  const float cosPrev = a_cosPrev; // fabs(dot(a_prevHit.normal, ray_dir));
  const float dist    = length(ray_pos - surfElem.pos);
  const float GTerm   = cosHere*cosPrev / fmax(dist*dist, DEPSILON2);

  if (a_currDepth > 0)
    a_accData->pdfGTerm *= GTerm;

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-3f)
  {
    const int instId          = fetchInstId(hit);
    const int lightOffset     = m_geom.instLightInstId[instId];
    const PlainLight* pLight  = lightAt(m_pGlobals, lightOffset);

    const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, m_pGlobals, m_texStorage, m_pdfStorage);
    PerThread().mBounceDone   = a_currDepth;

    a_accData->pdfLightWP *= (lPdfFwd.pdfW/ fmax(cosHere, DEPSILON));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
    const float lightPdfA  = lPdfFwd.pdfA;
    const float cancelPrev = (misPrev.matSamplePdf / fmax(cosPrev, DEPSILON))*GTerm; // calcel previous pdfA 

    float pdfAccFwdA = 1.0f       * (a_accData->pdfLightWP*a_accData->pdfGTerm) * lightPdfA*lPdfFwd.pickProb;
    float pdfAccRevA = cameraPdfA * (a_accData->pdfCameraWP*a_accData->pdfGTerm);
    float pdfAccExpA = cameraPdfA * (a_accData->pdfCameraWP*a_accData->pdfGTerm)*(lightPdfA*lightPdfSelectRev(pLight) / fmax(cancelPrev, DEPSILON));

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

    const float misWeight = misWeightHeuristic3(pdfAccRevA, pdfAccFwdA, pdfAccExpA);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    return emission*misWeight;
  }

  // explicit sampling
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  MisData thisBounce; 
  float3 explicitColor(0, 0, 0);
  
  auto& gen = randomGen();
  float lightPickProb = 1.0f;
  int lightOffset = SelectRandomLightRev(rndFloat2_Pseudo(&gen), surfElem.pos, m_pGlobals,
                                         &lightPickProb);
  
  thisBounce.lightPickProb = lightPickProb;

  if ((!m_computeIndirectMLT || a_currDepth > 0) && lightOffset >= 0) // if need to sample direct light ?
  {
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
    
    ShadowSample explicitSam;
    LightSampleRev(pLight, rndFloat3(&gen), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   &explicitSam);
    
    const float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos); // explicitSam.direction;
    const float3 shadowRayPos = OffsRayPos(surfElem.pos, surfElem.normal, shadowRayDir); 
    const float3 shadow       = shadowTrace(shadowRayPos, shadowRayDir, explicitSam.maxDist*0.9995f);
    
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
    
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l  = shadowRayDir;
    sc.v  = (-1.0f)*ray_dir;
    sc.n  = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;
    
    const auto evalData = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage);
    
    const float  cosThetaOut1 = fmax(+dot(shadowRayDir, surfElem.normal), 0.0f);
    const float  cosThetaOut2 = fmax(-dot(shadowRayDir, surfElem.normal), 0.0f);
    const bool   underSurface = (dot(evalData.btdf, evalData.btdf)*cosThetaOut2 > 0.0f && dot(evalData.brdf, evalData.brdf)*cosThetaOut1 <= 0.0f);
    const float  cosThetaOut  = underSurface ? cosThetaOut2 : cosThetaOut1;  
    const float  cosAtLight   = explicitSam.cosAtLight;
                              
    const float3 brdfVal      = evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2;
    const float  bsdfRevWP    = (evalData.pdfFwd == 0.0f) ? 1.0f : evalData.pdfFwd / fmax(cosThetaOut, DEPSILON);
    const float  bsdfFwdWP    = (evalData.pdfRev == 0.0f) ? 1.0f : evalData.pdfRev / fmax(cosHere, DEPSILON);
    const float shadowDist    = length(surfElem.pos - explicitSam.pos);
    const float GTermShadow   = cosThetaOut*cosAtLight / fmax(shadowDist*shadowDist, DEPSILON); 
    
    const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, shadowRayDir, cosAtLight, m_pGlobals, m_texStorage, m_pdfStorage);
    
    float pdfFwdA1 = 1.0f;  // madness of IBPT. 
    if (a_currDepth > 0)    // Imagine ray that hit light source after (second???) bounce (or first ???). pdfAccFwdA = pdfLightA*PdfLightW*GTermShadow.
      pdfFwdA1 = bsdfFwdWP; // 
    
    float pdfAccFwdA = 1.0f*pdfFwdA1 * (prevData.pdfLightWP *prevData.pdfGTerm)*((lPdfFwd.pdfW / fmax(cosAtLight, DEPSILON))*GTermShadow)*(lPdfFwd.pdfA*lPdfFwd.pickProb);
    float pdfAccRevA = cameraPdfA    * (prevData.pdfCameraWP*prevData.pdfGTerm)*bsdfRevWP*GTermShadow;
    float pdfAccExpA = cameraPdfA    * (prevData.pdfCameraWP*prevData.pdfGTerm)*(lPdfFwd.pdfA*lightPickProb);
    if (explicitSam.isPoint)
      pdfAccRevA = 0.0f;
    
    const float misWeight  = misWeightHeuristic3(pdfAccExpA, pdfAccRevA, pdfAccFwdA);
    
    const float explicitPdfW = fmax(explicitSam.pdf, DEPSILON);
    
    explicitColor = ((1.0f/lightPickProb)*explicitSam.color*brdfVal/explicitPdfW)*misWeight*shadow;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));
  const float3 bxdfVal   = matSam.color; // *(1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosNext    = fabs(dot(matSam.direction, surfElem.normal));

  //if (matSam.flags & RAY_EVENT_D && a_currDepth == 0)
  //  m_debugFirstBounceDiffuse = true;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (!isPureSpecular(matSam))
  {
    a_accData->pdfCameraWP *= (matSam.pdf / fmax(cosNext, DEPSILON));

    if (a_currDepth > 0)
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

      const float pdfFwdW = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage).pdfFwd;

      a_accData->pdfLightWP *= (pdfFwdW / fmax(cosHere, DEPSILON));
    }
  }
  else
  {
    a_accData->pdfCameraWP *= 1.0f; // in the case of specular bounce pdfFwd = pdfRev = 1.0f;
    a_accData->pdfLightWP  *= 1.0f; //

    //  ow ... but if we met specular reflection when tracing from camera, we must put 0 because this path cannot be sample by light strategy at all.
    //
    if (a_currDepth == 0)
      a_accData->pdfLightWP = 0.0f;
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  thisBounce.isSpecular         = isPureSpecular(matSam);
  thisBounce.matSamplePdf       = matSam.pdf;
  thisBounce.prevMaterialOffset = -1;

  const float3 thoroughput = bxdfVal*cosNext / fmax(matSam.pdf, DEPSILON);

  const float cosPrevForNextBounce = fabs(dot(surfElem.normal, nextRay_dir));

  return explicitColor + thoroughput*PathTraceAcc(nextRay_pos, nextRay_dir, cosPrevForNextBounce, thisBounce, a_currDepth + 1, flags,
                                                  pFirstHit, a_accData);
}

float3 IntegratorThreeWay::PathTrace(float3 ray_pos, float3 ray_dir)
{
  SurfaceHit firstHit;
  PerRayAcc  acc;
  acc.pdfCameraWP = 1.0f;
  acc.pdfLightWP  = 1.0f;
  acc.pdfGTerm    = 1.0f;

  //m_debugFirstBounceDiffuse = false;

  return PathTraceAcc(ray_pos, ray_dir, 1.0f, makeInitialMisData(), 0, 0,
                      &firstHit, &acc);
}


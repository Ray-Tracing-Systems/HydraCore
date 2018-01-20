#include <omp.h>
#include "CPUExp_Integrators.h"

void IntegratorTwoWay::SetMaxDepth(int a_depth)
{
  m_maxDepth = a_depth;

}

void IntegratorTwoWay::DoPass(std::vector<uint>& a_imageLDR)
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

  RandomizeAllGenerators();

  std::cout << "IntegratorTwoWay: spp  = " << m_spp << std::endl;
  m_spp++;
}


void IntegratorTwoWay::DoLightPath()
{
  const int totalLights = m_pGlobals->lightsNum;
  const int lightId     = rndInt(&PerThread().gen, 0, totalLights);

  const float pdfLightSelectInv = (float)totalLights;

  const PlainLight* pLight = lightAt(m_pGlobals, lightId);

  auto& rgen = randomGen();
  const float4 rands1 = rndFloat4(&rgen);
  const float2 rands2 = rndFloat2(&rgen);

  LightSampleFwd sample;
  LightSampleForward(pLight, rands1, rands2, m_pGlobals, m_texStorage, m_pdfStorage,
                     &sample);

  PerThread().pdfLightA0 = sample.pdfA/pdfLightSelectInv;  // #TODO: move this to some other structure

  PerRayAcc acc0;
  float3 color     = pdfLightSelectInv*sample.color/(sample.pdfA*sample.pdfW);
  acc0.pdfLightWP  = sample.pdfW/sample.cosTheta;
  acc0.pdfCameraWP = 1.0f;
  acc0.pdfGTerm    = 1.0f;

  TraceLightPath(sample.pos, sample.dir, 1, sample.cosTheta, sample.pdfW, 
                 &acc0, color);
}

void IntegratorTwoWay::TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float a_prevLightCos, float a_prevPdf, 
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

  a_pAccData->pdfGTerm *= (a_prevLightCos*cosCurr / (dist*dist));

  ConnectEye(surfElem, ray_pos, ray_dir, a_currDepth, a_pAccData, a_color);

  // eval forward and reverse pdf
  //
  if(!isPureSpecular(matSam))
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

    a_pAccData->pdfLightWP  *= (matSam.pdf/fmax(cosNext, DEPSILON2));
    a_pAccData->pdfCameraWP *= (pdfW/fmax(cosCurr, DEPSILON2));
  }
  else
  {
    a_pAccData->pdfLightWP  *= 1.0f;
    a_pAccData->pdfCameraWP *= 1.0f; // in the case of specular bounce pdfFwd = pdfRev = 1.0f;
  }

  a_color *= matSam.color*cosNext* (1.0f / fmax(matSam.pdf, DEPSILON));

  TraceLightPath(nextRay_pos, nextRay_dir, a_currDepth + 1, cosNext, matSam.pdf, 
                 a_pAccData, a_color);
}

void IntegratorTwoWay::ConnectEye(SurfaceHit a_hit, float3 ray_pos, float3 ray_dir, int a_currBounce, PerRayAcc* a_pAccData, float3 a_accColor)
{
  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(a_hit.pos, a_hit.normal, m_pGlobals,
                                                                &camDir, &zDepth);
  if (imageToSurfaceFactor <= 0.0f)
    return;

  const float surfaceToImageFactor  = 1.f / imageToSurfaceFactor;

  float  pdfRevW = 1.0f;
  float3 colorConnect = make_float3(1, 1, 1);
  {
    const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, a_hit.matId);

    ShadeContext sc;
    sc.wp = a_hit.pos;
    sc.l  = camDir;          // seems like that sc.l = camDir, see smallVCM
    sc.v  = (-1.0f)*ray_dir; // seems like that sc.v = (-1.0f)*ray_dir, see smallVCM
    sc.n  = a_hit.normal;
    sc.fn = a_hit.flatNormal;
    sc.tg = a_hit.tangent;
    sc.bn = a_hit.biTangent;
    sc.tc = a_hit.texCoord;

    auto colorAndPdf = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage);
    colorConnect     = colorAndPdf.brdf;
    pdfRevW          = colorAndPdf.pdfRev;
  }

  const float cosCurr  = fabs(dot(ray_dir, a_hit.normal));
  const float pdfRevWP = pdfRevW / fmax(cosCurr,DEPSILON2); // pdfW po pdfWP

  // We put the virtual image plane at such a distance from the camera origin
  // that the pixel area is one and thus the image plane sampling pdf is 1.
  // The area pdf of a_hit.pos as sampled from the camera is then equal to
  // the conversion factor from image plane area density to surface area density
  //
  const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
  const float lightPdfA  = PerThread().pdfLightA0;

  const float pdfAccFwdA = lightPdfA*a_pAccData->pdfLightWP*a_pAccData->pdfGTerm;
  const float pdfAccRevA = cameraPdfA*(pdfRevWP*a_pAccData->pdfCameraWP)*a_pAccData->pdfGTerm; // see pdfRevW? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.
                                                                                               // we didn't eval reverse pdf yet. Imagine light ray hit surface and we immediately connect.

  const float misWeight  = misWeightHeuristic(pdfAccFwdA, pdfAccRevA);

  ///////////////////////////////////////////////////////////////////////////////

  // We divide the contribution by surfaceToImageFactor to convert the(already
  // divided) pdf from surface area to image plane area, w.r.t. which the
  // pixel integral is actually defined. We also divide by the number of samples
  // this technique makes, which is equal to the number of light sub-paths
  //
  const float3 sampleColor = misWeight*a_accColor*(colorConnect / (mLightSubPathCount*surfaceToImageFactor));

  if (dot(sampleColor, sampleColor) > 1e-6f) // add final result to image
  {
    auto hit = rayTrace(a_hit.pos + epsilonOfPos(a_hit.pos)*a_hit.normal, camDir);

    if (!HitSome(hit) || hit.t > zDepth)
    {
      const float2 posScreenSpace = worldPosToScreenSpace(a_hit.pos, m_pGlobals);

      int x = int(posScreenSpace.x + 0.5f);
      int y = int(posScreenSpace.y + 0.5f);

      if (x >= 0 && x <= m_width - 1 && y >= 0 && y <= m_height - 1)
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


float3 IntegratorTwoWay::PathTraceAcc(float3 ray_pos, float3 ray_dir, const SurfaceHit& a_prevHit, MisData misPrev, int a_currDepth, uint flags,
                                      SurfaceHit* pFirstHit, PerRayAcc* a_accData)
{
  if (a_currDepth >= m_maxDepth)
    return float3(0,0,0);

  const Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
  {
    //a_accData->pdfLightW *= ... ; // #TODO: add pdf for environmentlight here ?
    return EnviromnentColor(ray_dir, misPrev, flags);
  }

  const SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  if (pFirstHit != nullptr)
    (*pFirstHit) = surfElem;

  const float cosHere = fabs(dot(ray_dir, surfElem.normal));
  const float cosPrev = fabs(dot(a_prevHit.normal, ray_dir));

  if (a_currDepth > 0)
  {
    const float dist    = length(ray_pos - surfElem.pos);
    const float GTerm   = cosHere*cosPrev / fmax(dist*dist, DEPSILON2);
    a_accData->pdfGTerm *= GTerm;
  }

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-3f)
  {
    const int instId = fetchInstId(hit);
    const int lightOffset = m_geom.instLightInstId[instId];
    const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, m_pGlobals, m_texStorage, m_pdfStorage);

    const float pdfLightSelectInv = (float)m_pGlobals->lightsNum;

    PerThread().pdfLightA0   = lPdfFwd.pdfA/pdfLightSelectInv;
    PerThread().mBounceDone  = a_currDepth;

    a_accData->pdfLightWP *= (lPdfFwd.pdfW/fmax(cosHere, DEPSILON));

    return emission;
  }

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));
  const float3 bxdfVal   = matSam.color; // *(1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosNext    = fabs(dot(matSam.direction, surfElem.normal));

  // eval reverse and forward pdfs
  //
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
    a_accData->pdfCameraWP *= 1.0f;  // in the case of specular bounce pdfFwd = pdfRev = 1.0f;
    a_accData->pdfLightWP  *= 1.0f;  //

    //  ow ... but if we met specular reflection when tracing from camera, we must put 0 because this path cannot be sample by light strategy at all.
    //
    if (a_currDepth == 0)
      a_accData->pdfLightWP = 0.0f;
  }
 
  // proceed to next bounce
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  MisData thisBounce;
  thisBounce.isSpecular         = isPureSpecular(matSam);
  thisBounce.matSamplePdf       = matSam.pdf;
  thisBounce.prevMaterialOffset = -1;

  return (bxdfVal*cosNext / fmax(matSam.pdf, DEPSILON))*PathTraceAcc(nextRay_pos, nextRay_dir, surfElem, thisBounce, a_currDepth + 1, flags,
                                                                     nullptr, a_accData);
}

float3 IntegratorTwoWay::PathTrace(float3 ray_pos, float3 ray_dir)
{
  PerRayAcc acc;
  acc.pdfCameraWP = 1.0f;
  acc.pdfLightWP  = 1.0f;
  acc.pdfGTerm    = 1.0f;

  SurfaceHit firstHit;
  float3 color = PathTraceAcc(ray_pos, ray_dir, firstHit, makeInitialMisData(), 0, 0, 
                              &firstHit, &acc);

  if (length(color) < 1e-10f)
    return float3(0, 0, 0);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Compute pdf conversion factor from image plane area to surface area
  //
  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(firstHit.pos, firstHit.normal, m_pGlobals,
                                                                &camDir, &zDepth);

  const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
  const float lightPdfA  = PerThread().pdfLightA0;

  float pdfAccRevA = cameraPdfA * (acc.pdfCameraWP*acc.pdfGTerm);
  float pdfAccFwdA = lightPdfA  * (acc.pdfLightWP*acc.pdfGTerm);   // try to remove lightPdfA?, it should be inside acc.pdfLightW already!

  const int maxBounce = PerThread().mBounceDone;

  if (maxBounce == 0)
  {
    pdfAccFwdA = 1.0f;
    pdfAccRevA = 1.0f;
  }

  const float misWeight = misWeightHeuristic(pdfAccRevA, pdfAccFwdA);

  return misWeight*color;
}


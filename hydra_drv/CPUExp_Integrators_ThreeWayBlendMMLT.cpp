#include <omp.h>
#include "CPUExp_Integrators.h"

#include <array>
#include <algorithm>

void IntegratorThreeWayBlendMMLT::SetMaxDepth(int a_depth)
{
  m_maxDepth = a_depth;
  m_pIntegratorMMLT->SetMaxDepth(a_depth);
}

bool HR_SaveHDRImageToFileLDR(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scaleInv, const float a_gamma = 2.2f);
bool HR_SaveHDRImageToFileHDR(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scale = 1.0f);

void IntegratorThreeWayBlendMMLT::DoPass(std::vector<uint>& a_imageLDR)
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
  
      SurfaceHit firstHit;
      PerRayAcc  acc = InitialPerParAcc();
  
      float3 direct(0, 0, 0), indirect(0, 0, 0);
      PathTraceAcc(ray_pos, ray_dir, 1.0f, makeInitialMisData(), 0, 0,
                   &firstHit, &acc, &direct, &indirect);
  
      m_hdrDL[y*m_width + x] += to_float4(direct, 0.0f);
      m_hdrGI[y*m_width + x] += to_float4(indirect, 0.0f);
    }
  }

  float kScaleMLT = 1.0f;
  const int NSamplesBefore = 1;

  if (m_spp == NSamplesBefore)
  {
    const float scaleInv = 1.0f / float(m_spp + 1);
    //HR_SaveHDRImageToFileLDR(L"C:\\[Hydra]\\rendered_images\\02_dl.png", m_width, m_height, (const float*)m_hdrDL,   scaleInv, 2.2f);
    HR_SaveHDRImageToFileLDR(L"C:\\[Hydra]\\rendered_images\\03_gi.png", m_width, m_height, (const float*)m_hdrGI,   scaleInv, 2.2f);
    //HR_SaveHDRImageToFileLDR(L"C:\\[Hydra]\\rendered_images\\04_al.png", m_width, m_height, (const float*)m_hdrData, scaleInv, 2.2f);

    //HR_SaveHDRImageToFileHDR(L"C:\\[Hydra]\\rendered_images\\05_dl.hdr", m_width, m_height, (const float*)m_hdrDL, scaleInv);
    //HR_SaveHDRImageToFileHDR(L"C:\\[Hydra]\\rendered_images\\06_gi.hdr", m_width, m_height, (const float*)m_hdrGI, scaleInv);
    //HR_SaveHDRImageToFileHDR(L"C:\\[Hydra]\\rendered_images\\07_al.hdr", m_width, m_height, (const float*)m_hdrData, scaleInv);

    // (1) extract noise
    //
    //float normConst = 1.0f;
    std::vector<float> noise(m_width*m_height);
    //ExtractNoise(m_hdrGI, 1.0f, 
    //             noise, normConst);
    //
    //std::cout << "normConst = " << normConst << std::endl;
    //for (size_t i = 0; i < noise.size(); i++)
    //  noise[i] *= normConst;
    //{
    //  std::vector<float4> errImage(m_width*m_height);
    //  for (size_t i = 0; i < errImage.size(); i++)
    //  {
    //    const float val = noise[i];
    //    errImage[i] = float4(val, val, val, 1.0f);
    //  }
    //
    //  HR_SaveHDRImageToFileHDR(L"C:\\[Hydra]\\rendered_images\\07_noise.hdr", m_width, m_height, (const float*)&errImage[0]);
    //}

    // (2) compute gbuffer
    //
    std::cout << "begin gbuffer ... " << std::endl;
    std::vector<GBufferAll>  gbuff(m_width*m_height);
    CalcGBufferUncompressed(gbuff);
    //DebugSaveGbufferImage(L"C:/[Hydra]/rendered_images/torus_gbuff");
    std::cout << "end   gbuffer ... " << std::endl;

    std::array<int, 17> ids = {6,10,14,15,27,29,35,36,40,41,46,48,51,56,57,63,64};
    for (size_t i = 0; i < gbuff.size(); i++)
    {
      if (std::find(ids.begin(), ids.end(), gbuff[i].data1.matId) != ids.end())
        noise[i] = 1.0f;
      else
        noise[i] = 0.1f;
    }
    
    // (3) separable spread/blur
    //
    //SpreadNoise2(gbuff, noise); 
    m_noiseGI = noise;
    std::cout << "end   spread  ... " << std::endl;

    {
      std::vector<float4> errImage(m_width*m_height);
      for (size_t i = 0; i < errImage.size(); i++)
      {
        const float val = m_noiseGI[i];
        errImage[i] = float4(val, val, val, 1.0f);
      }

      HR_SaveHDRImageToFileHDR(L"C:\\[Hydra]\\rendered_images\\08_noise.hdr", m_width, m_height, (const float*)&errImage[0]);
    }

    m_pIntegratorMMLT->SetMaskPtr(&m_noiseGI[0]);
    m_pIntegratorMMLT->DoPassEstimateAvgBrightness();

  }
  else if (m_spp >= NSamplesBefore && (m_spp%2 == 0))
  {
    m_pIntegratorMMLT->DoPassIndirectMLT(m_hdrGI2);
    kScaleMLT = m_pIntegratorMMLT->EstimateScaleCoeff();
    //std::cout << "kScaleMLT = " << kScaleMLT << std::endl;
    //std::wstringstream strOut;
    //strOut << L"C:\\[Hydra]\\rendered_images\\" << m_spp << L"_gi.png";
    //std::wstring outPath = strOut.str();
    //HR_SaveHDRImageToFileLDR(outPath.c_str(), m_width, m_height, (const float*)m_hdrGI2, kScaleMLT, 2.2f);
  }

  // form final image
  //
  constexpr float gammaPow = 1.0f / 2.2f;
  const float scaleInv = 1.0f / float(m_spp + 1);

  //if (m_spp == 64)
  //{
  //  HR_SaveHDRImageToFileLDR(L"C:\\[Hydra]\\rendered_images\\02_dl.png", m_width, m_height, (const float*)m_hdrDL, scaleInv, 2.2f);
  //  HR_SaveHDRImageToFileLDR(L"C:\\[Hydra]\\rendered_images\\03_gi.png", m_width, m_height, (const float*)m_hdrGI, scaleInv, 2.2f);
  //  HR_SaveHDRImageToFileLDR(L"C:\\[Hydra]\\rendered_images\\04_gi.png", m_width, m_height, (const float*)m_hdrGI2, kScaleMLT, 2.2f);
  //  HR_SaveHDRImageToFileLDR(L"C:\\[Hydra]\\rendered_images\\05_al.png", m_width, m_height, (const float*)m_hdrData, 1.0f, 2.2f);
  //}

  if (m_spp <= NSamplesBefore)
  {
    for (int i = 0; i < int(a_imageLDR.size()); i++)
    {
      float4 color = m_hdrDL[i] + m_hdrGI[i];
      m_hdrData[i] = color*scaleInv;

      color.x = powf(clamp(color.x*scaleInv, 0.0f, 1.0f), gammaPow);
      color.y = powf(clamp(color.y*scaleInv, 0.0f, 1.0f), gammaPow);
      color.z = powf(clamp(color.z*scaleInv, 0.0f, 1.0f), gammaPow);
      color.w = 1.0f;

      a_imageLDR[i] = RealColorToUint32(color);
    }
  }
  else
  {
    for (int i = 0; i < int(a_imageLDR.size()); i++)
    {
      float alpha = m_noiseGI[i];
      float4 color;
      if (alpha > 0.1f)
      {
        color = kScaleMLT*m_hdrGI2[i]; // *alpha //#NOTE that you should not multiply alpha here because MLT already multiplied it !!!
        color += scaleInv*m_hdrDL[i];
        color += scaleInv*m_hdrGI[i] * (1.0f - alpha);
      }
      else
      {
        color  = scaleInv*m_hdrDL[i];
        color += scaleInv*m_hdrGI[i];
      }
      m_hdrData[i] = color;

      color.x = powf(clamp(color.x, 0.0f, 1.0f), gammaPow);
      color.y = powf(clamp(color.y, 0.0f, 1.0f), gammaPow);
      color.z = powf(clamp(color.z, 0.0f, 1.0f), gammaPow);
      color.w = 1.0f;

      a_imageLDR[i] = RealColorToUint32(color);
    }
  }

  RandomizeAllGenerators();
  m_pIntegratorMMLT->RandomizeAllGenerators();

  std::cout << "IntegratorThreeWayBlendMMLT: spp  = " << m_spp << std::endl;
  m_spp++;
}

void IntegratorThreeWayBlendMMLT::DoLightPath()
{
  const float totalLights  = float(m_pGlobals->lightsNum);
  const int lightId        = rndInt(&PerThread().gen, 0, m_pGlobals->lightsNum);
  const PlainLight* pLight = lightAt(m_pGlobals, lightId);

  auto& rgen = randomGen();
  const float4 rands1 = rndFloat4(&rgen);
  const float2 rands2 = rndFloat2(&rgen);

  LightSampleFwd sample;
  LightSampleForward(pLight, rands1, rands2, m_pGlobals, m_texStorage, m_pdfStorage,
                     &sample);

  PerThread().pdfLightA0 = sample.pdfA;              // #TODO: move this to some other structure

  PerRayAcc acc0;
  float3 color     = totalLights*sample.color/(sample.pdfA*sample.pdfW);
  acc0.pdfLightWP  = sample.pdfW/sample.cosTheta;
  acc0.pdfCameraWP = 1.0f;
  acc0.pdfGTerm    = 1.0f;
  acc0.pdfCamA0    = 1.0f;

  //SaveLightSample(sample);

  TraceLightPath(sample.pos, sample.dir, 1, sample.cosTheta, sample.pdfW, 
                 &acc0, color);
}

void IntegratorThreeWayBlendMMLT::TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float a_prevLightCos, float a_prevPdf, 
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

void IntegratorThreeWayBlendMMLT::ConnectEye(SurfaceHit a_hit, float3 ray_pos, float3 ray_dir, int a_currBounce, 
                                             PerRayAcc* a_pAccData, float3 a_color)
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
  const float pdfRevWP = pdfRevW / fmax(cosCurr,DEPSILON); // pdfW po pdfWP

  float pdfCamA0 = a_pAccData->pdfCamA0;
  if (a_currBounce == 1)
    pdfCamA0 *= pdfRevWP; // see pdfRevWP? this is just because on the first bounce a_pAccData->pdfCameraWP == 1.

  const float cancelImplicitLightHitPdf = (1.0f / fmax(pdfCamA0, DEPSILON2));

  const PlainLight* pLight = lightAt(m_pGlobals, PerThread().selectedLightIdFwd);
  const float lightPickProbFwd = lightPdfSelectFwd(pLight, m_pGlobals);
  const float lightPiclProbRev = a_pAccData->pdfSelectRev;

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
  const float pdfAccExpA = cameraPdfA*(pdfRevWP*a_pAccData->pdfCameraWP)*a_pAccData->pdfGTerm*(cancelImplicitLightHitPdf*(lightPdfA*lightPiclProbRev));

  const float misWeight  = misWeightHeuristic3(pdfAccFwdA, pdfAccRevA, pdfAccExpA);

  ///////////////////////////////////////////////////////////////////////////////

  // We divide the contribution by surfaceToImageFactor to convert the(already
  // divided) pdf from surface area to image plane area, w.r.t. which the
  // pixel integral is actually defined. We also divide by the number of samples
  // this technique makes, which is equal to the number of light sub-paths
  //
  const float3 sampleColor = misWeight*a_color*(colorConnect / (mLightSubPathCount*surfaceToImageFactor));

  if (dot(sampleColor, sampleColor) > 1e-6f) // add final result to image
  {
    auto hit = rayTrace(a_hit.pos + epsilonOfPos(a_hit.pos)*a_hit.normal, camDir);

    if (!HitSome(hit) || hit.t > zDepth)
    {
      const float2 posScreenSpace = worldPosToScreenSpace(a_hit.pos, m_pGlobals);

      int x = int(posScreenSpace.x + 0.5f);
      int y = int(posScreenSpace.y + 0.5f);

      if (x >= 0 && x < m_width && y >= 0 && y < m_height)
      { 
        const int offset = y*m_width + x;
        
        if (a_currBounce == 1)
        {
          #pragma omp atomic
          m_hdrDL[offset].x += sampleColor.x;
          #pragma omp atomic
          m_hdrDL[offset].y += sampleColor.y;
          #pragma omp atomic
          m_hdrDL[offset].z += sampleColor.z;
        }
        else
        { 
          #pragma omp atomic
          m_hdrGI[offset].x += sampleColor.x;
          #pragma omp atomic
          m_hdrGI[offset].y += sampleColor.y;
          #pragma omp atomic
          m_hdrGI[offset].z += sampleColor.z;
        }
      }

    }
  }
}


void  IntegratorThreeWayBlendMMLT::PathTraceAcc(float3 ray_pos, float3 ray_dir, const float a_cosPrev, MisData misPrev, int a_currDepth, uint flags,
                                                SurfaceHit* pFirstHit, PerRayAcc* a_accData, float3* a_pDirectLight, float3* a_pIndirectLight)
{
  if (a_currDepth >= m_maxDepth)
  {
    (*a_pDirectLight)   = float3(0, 0, 0);
    (*a_pIndirectLight) = float3(0, 0, 0);
    return;
  }

  const Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
  {
    //a_accData->pdfLightW *= ... ; // #TODO: add pdf for environmentlight here and apply MIS
    (*a_pDirectLight)   = EnviromnentColor(ray_dir, misPrev, flags);
    (*a_pIndirectLight) = float3(0, 0, 0);
    return;
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
    const int instId = fetchInstId(hit);
    const int lightOffset = m_geom.instLightInstId[instId];
    const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);

    const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, m_pGlobals, m_texStorage, m_pdfStorage);
    PerThread().mBounceDone  = a_currDepth;

    a_accData->pdfLightWP *= (lPdfFwd.pdfW/ fmax(cosHere, DEPSILON));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
    const float lightPdfA  = lPdfFwd.pdfA;
    const float cancelPrev = (misPrev.matSamplePdf / fmax(cosPrev, DEPSILON))*GTerm; // calcel previous pdfA 

    float pdfAccFwdA = 1.0f       * (a_accData->pdfLightWP*a_accData->pdfGTerm) * lightPdfA/float(m_pGlobals->lightsNum);
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

    (*a_pDirectLight  ) = emission*misWeight;
    (*a_pIndirectLight) = float3(0, 0, 0);
  }
  else if (!m_splitDLByGrammar && a_currDepth >= 1)
  {
    (*a_pDirectLight)   = float3(0, 0, 0);
    (*a_pIndirectLight) = float3(0, 0, 0);
  }

  // explicit sampling
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  MisData thisBounce; 
  float3 explicitColor(0, 0, 0);
  
  auto& gen = randomGen();
  float lightPickProb = 1.0f;
  int lightOffset     = SelectRandomLightRev(rndFloat2_Pseudo(&gen), surfElem.pos, m_pGlobals,
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
    
    const auto   evalData     = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage);
    
    const float  cosThetaOut1 = fmax(+dot(shadowRayDir, surfElem.normal), 0.0f);
    const float  cosThetaOut2 = fmax(-dot(shadowRayDir, surfElem.normal), 0.0f);
    const bool   underSurface = (dot(evalData.btdf, evalData.btdf)*cosThetaOut2 > 0.0f && dot(evalData.brdf, evalData.brdf)*cosThetaOut1 <= 0.0f);
    const float  cosThetaOut  = underSurface ? cosThetaOut2 : cosThetaOut1;  
    const float  cosAtLight   = explicitSam.cosAtLight;
                              
    const float3 brdfVal      = evalData.brdf*cosThetaOut1 + evalData.btdf*cosThetaOut2;
    const float  bsdfRevWP = (evalData.pdfFwd == 0.0f) ? 1.0f : evalData.pdfFwd / fmax(cosThetaOut, DEPSILON);
    const float  bsdfFwdWP = (evalData.pdfRev == 0.0f) ? 1.0f : evalData.pdfRev / fmax(cosHere, DEPSILON);
    const float shadowDist    = length(surfElem.pos - explicitSam.pos);
    const float GTermShadow   = cosThetaOut*cosAtLight / fmax(shadowDist*shadowDist, DEPSILON); 
    
    const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, shadowRayDir, cosAtLight, m_pGlobals, m_texStorage, m_pdfStorage);
    
    float pdfFwdA1 = 1.0f;        // madness of IBPT. 
    if (a_currDepth > 1)          // Imagine ray that hit light source after first bounce. pdfAccFwdA = pdfLightA*PdfLightW*GTermShadow.
      pdfFwdA1 = bsdfFwdWP*GTerm; // 
    
    const float pdfAccFwdA = 1.0f*pdfFwdA1 * (prevData.pdfLightWP *prevData.pdfGTerm)*((lPdfFwd.pdfW / fmax(cosAtLight, DEPSILON))*GTermShadow)*(lPdfFwd.pdfA/float(m_pGlobals->lightsNum));
    const float pdfAccRevA = cameraPdfA    * (prevData.pdfCameraWP*prevData.pdfGTerm)*bsdfRevWP*GTermShadow;
    const float pdfAccExpA = cameraPdfA    * (prevData.pdfCameraWP*prevData.pdfGTerm)*(lPdfFwd.pdfA*lightPickProb);
    
    const float misWeight  = misWeightHeuristic3(pdfAccExpA, pdfAccRevA, pdfAccFwdA);
    
    const float explicitPdfW = fmax(explicitSam.pdf, DEPSILON);
    
    explicitColor = ((1.0f/lightPickProb)*explicitSam.color*brdfVal /explicitPdfW)*misWeight*shadow;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));
  const float3 bxdfVal   = matSam.color; // *(1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosNext    = fabs(dot(matSam.direction, surfElem.normal));

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

  //return explicitColor + thoroughput*PathTraceAcc(nextRay_pos, nextRay_dir, cosPrevForNextBounce, thisBounce, a_currDepth + 1, flags,
  //                                                pFirstHit, a_accData);

  const bool wasSpecOnly = m_splitDLByGrammar ? flagsHaveOnlySpecular(flags) : (a_currDepth == 0);

  float3 direct, indirect;
  PathTraceAcc(nextRay_pos, nextRay_dir, cosPrevForNextBounce, thisBounce, a_currDepth + 1, flagsNextBounceLite(flags, matSam, m_pGlobals),
               pFirstHit, a_accData, &direct, &indirect);

  if (a_currDepth == 0 || wasSpecOnly)
  {
    (*a_pDirectLight)   += (explicitColor + thoroughput*direct);
    (*a_pIndirectLight) += (thoroughput*indirect);
  }
  else
  {
    (*a_pIndirectLight) += (explicitColor + thoroughput*(direct + indirect));
  }
  

}


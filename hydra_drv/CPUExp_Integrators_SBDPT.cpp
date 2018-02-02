#include <omp.h>
#include "CPUExp_Integrators.h"

void IntegratorSBDPT::SetMaxDepth(int a_depth)
{
  m_maxDepth = a_depth;
  for (size_t i = 0; i < m_perThread.size(); i++)
    m_perThread[i].pdfArray.resize(m_maxDepth + 1);
}

void IntegratorSBDPT::DoPass(std::vector<uint>& a_imageLDR)
{
  const int samplesPerPass = m_width*m_height;
  mLightSubPathCount = float(samplesPerPass);

  #pragma omp parallel for
  for (int i = 0; i < samplesPerPass; i++)
  {
    // select path depth and pair of (s,t) where 's' is a light source and 't' is the camera 
    //
    const int d = rndInt(&PerThread().gen, 2, m_maxDepth+1);       // #TODO: change rndInt_Pseudo for spetial bounce selector random.      
    const int s = rndInt(&PerThread().gen, 0, d+1);                // for example 'd' == 2; so 's' should be (0,1,2) for (s=2,t=0) and (s=1,t=1) and (s=0,t=2)
    const int t = d - s;                                           // note that s=2 means 1 light bounce and one connection!!!
    const float selectorInvPdf = float((d + 1)*(m_maxDepth - 1));  // 

    //const int s = 3;
    //const int t = 1;
    //const int d = s + t;
    //const float selectorInvPdf = 1.0f;

    for (int i = 0; i <= d; i++) // let say d == 3, so [0] is vertex in light and [3] is vertex on camera;
    {
      PerThread().pdfArray[i].pdfFwd = 1.0f;
      PerThread().pdfArray[i].pdfRev = 1.0f;
    }

    const int lightTraceDepth = s - 1;  // because the last light path is a connection anyway - to camera or to camera path
    const int camTraceDepth   = t;      //  
  
    int x = rndInt(&PerThread().gen, 0, m_width);            // light tracing can overwtite this variables
    int y = rndInt(&PerThread().gen, 0, m_height);           // light tracing can overwtite this variables

    PerThread().clearPathGrammar(d+1);

    // (1) trace path from camera with depth == camTraceDepth and (x,y)
    //
    PathVertex cv;
    InitPathVertex(&cv);

    if (camTraceDepth > 0)
    {
      float3 ray_pos, ray_dir;
      std::tie(ray_pos, ray_dir) = makeEyeRay(x, y);
    
      const bool haveToHitLight = (lightTraceDepth == -1);  // when lightTraceDepth == -1, use only camera strategy, so have to hit light at some depth level   

      cv = CameraPath(ray_pos, ray_dir, float3(0,0,1), makeInitialMisData(), 1, 0,
                      &PerThread(), camTraceDepth, haveToHitLight, d);
    }

    // (2) trace path from light with depth = lightTraceDepth;
    //
    PathVertex lv;
    InitPathVertex(&lv);

    if (lightTraceDepth > 0) 
      lv = LightPath(&PerThread(), lightTraceDepth);

    // (3) connect; this operation should also compute missing pdfA for camera and light 
    //
    float3 sampleColor(0, 0, 0);
    bool wasConnectEndPoints = false;

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
          sampleColor = ConnectEye(lv, lightTraceDepth,
                                   &PerThread(), &x, &y);
        }
      }
      else if (lightTraceDepth == 0)  // (3.3) connect camera vertex to light (shadow ray)
      {
        if (cv.valid)
        {
          float3 explicitColor = ConnectShadow(cv, &PerThread(), t);
          sampleColor = cv.accColor*explicitColor;
        }
      }
      else                            // (3.4) connect light and camera vertices (bidir connection)
      {
        if (cv.valid)
        {
          float3 explicitColor = ConnectEndPoints(lv, cv, s, d, &PerThread());
          sampleColor = cv.accColor*explicitColor*lv.accColor;
          wasConnectEndPoints = true;
        }
      }
    }

    // (4) calc MIS weights
    //
    float misWeight = 1.0f;
    if(dot(sampleColor, sampleColor) > 1e-12f)
    //if(false)
    {
      // lat say s=3,t=0, so pdfThisWay = pLightA*(pLightWP*G[0])*(pFwd[1]*G[1])*1 /// pLightA ==  pdfFwd[0]
      // let say s=2,t=1, so pdfThisWay = pLightA*(pLightWP*G[0])*1*pCamA          /// pCamA   ==  pdfRev[3]
      // let say s=1,t=2, so pdfThisWay = pLightA*1*(pRev[2]*G[1])*pCamA           /// (pRev[2]*G[1]) == pdfRev[2]
      // let say s=0,t=3, so pdfThisWay = 1*(pRev[1]*G[0])*(pRev[2]*G[1])*pCamA
      //
      float pdfThisWay = 1.0f;
      float pdfSumm    = 0.0f;
      int   validStrat = 0;

      for (int split = 0; split <= d; split++)
      {
        const int s1 = split;
        const int t1 = d - split;

        const bool specularMet = (split > 0) && (split < d) && (PerThread().pdfArray[split].pdfRev < 0.0f ||
                                                                PerThread().pdfArray[split].pdfFwd < 0.0f);
        float pdfOtherWay = specularMet ? 0.0f : 1.0f;
        if (split == d)
          pdfOtherWay = misHeuristicPower1(PerThread().pdfArray[d].pdfFwd);
      
        for (int i = 0; i < s1; i++)
          pdfOtherWay *= misHeuristicPower1(PerThread().pdfArray[i].pdfFwd);
        for (int i = s1 + 1; i <= d; i++)
          pdfOtherWay *= misHeuristicPower1(PerThread().pdfArray[i].pdfRev);

        if (pdfOtherWay != 0.0f)
          validStrat++;

        if (s1 == s && t1 == t)
          pdfThisWay = pdfOtherWay;

        pdfSumm += pdfOtherWay;
      }

      misWeight = pdfThisWay / fmax(pdfSumm, DEPSILON2);
     
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //#pragma omp critical
      //if (misWeight > 1e6f && wasConnectEndPoints)
      //{
      //  std::cout << std::endl;
      //  std::cout << "Path Grammar = " << PerThread().grammarCam.c_str() << std::endl;
      //  std::cout << "Valid strat  = " << validStrat << std::endl;
      //  std::cout << "sampleColor  = (" << sampleColor.x << ", " << sampleColor.y << ", " << sampleColor.z <<")" << std::endl;
      //  std::cout << "misWeight    = " << misWeight << std::endl;
      //  std::cout << "pdfThisWay   = " << pdfThisWay << std::endl;
      //  std::cout << "pdfSumm      = " << pdfSumm << std::endl;
      //  std::cout << "i            = " << i << std::endl;
      //  std::cout << std::endl;
      //
      //  for (int split = 0; split <= d; split++)
      //    std::cout << PerThread().pdfArray[split].pdfFwd << "\t";
      //  std::cout << std::endl;
      //  std::cout << std::endl;
      //  for (int split = 0; split <= d; split++)
      //    std::cout << PerThread().pdfArray[split].pdfRev << "\t";
      //  std::cout << std::endl;
      //  std::cout << std::endl;
      //  std::cout.flush();
      //  //DebugOutCurrPath(d);
      //  exit(0);
      //}
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    }

    sampleColor *= misWeight;
    sampleColor *= selectorInvPdf;

    // (5) contrib to image
    //
    if (dot(sampleColor, sampleColor) > 1e-20f && (x >= 0 && x < m_width && y >= 0 && y < m_height))
    { 
      const int offset = y*m_width + x;
      #pragma omp atomic
      m_hdrData[offset].x += sampleColor.x;
      #pragma omp atomic
      m_hdrData[offset].y += sampleColor.y;
      #pragma omp atomic
      m_hdrData[offset].z += sampleColor.z;  
    }
    //}
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

  std::cout << "IntegratorSBDPT: spp  = " << m_spp << std::endl;
  m_spp++;
}


PathVertex IntegratorSBDPT::LightPath(PerThreadData* a_perThread, int a_lightTraceDepth)
{
  const float totalLights = float(m_pGlobals->lightsNum);
  const int lightId       = rndInt(&PerThread().gen, 0, m_pGlobals->lightsNum);

  const PlainLight* pLight = lightAt(m_pGlobals, lightId);

  auto& rgen = randomGen();
  const float4 rands1 = rndFloat4(&rgen);
  const float2 rands2 = rndFloat2(&rgen);

  LightSampleFwd sample;
  LightSampleForward(pLight, rands1, rands2, m_pGlobals, m_texStorage, m_pdfStorage,
                     &sample);

  a_perThread->pdfArray[0].pdfFwd = sample.pdfA/totalLights;
  a_perThread->pdfArray[0].pdfRev = 1.0f;

  float3 color = totalLights*sample.color/(sample.pdfA*sample.pdfW);

  PathVertex lv;
  InitPathVertex(&lv);

  TraceLightPath(sample.pos, sample.dir, 1, sample.cosTheta, sample.pdfW, 
                 color, a_perThread, a_lightTraceDepth, false, &lv);
  return lv;
}

void IntegratorSBDPT::TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float a_prevLightCos, float a_prevPdf, 
                                     float3 a_color, PerThreadData* a_perThread, int a_lightTraceDepth, bool a_wasSpecular,
                                     PathVertex* a_pOutLightVertex)
{
  if (a_currDepth > a_lightTraceDepth)
    return;

  auto hit = rayTrace(ray_pos, ray_dir);
  if (!HitSome(hit))
    return;

  const SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  const float cosPrev = fabs(a_prevLightCos);
  const float cosCurr = fabs(-dot(ray_dir, surfElem.normal));
  const float dist    = length(surfElem.pos - ray_pos);

  // eval forward pdf
  //
  const float GTermPrev = (a_prevLightCos*cosCurr / fmax(dist*dist, DEPSILON2));
  const float prevPdfWP = a_prevPdf / fmax(a_prevLightCos, DEPSILON);
  
  if (!a_wasSpecular)
    a_perThread->pdfArray[a_currDepth].pdfFwd = prevPdfWP*GTermPrev;
  else
    a_perThread->pdfArray[a_currDepth].pdfFwd = -1.0f*GTermPrev;

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample      matSam       = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));

  // calc new ray
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  const float cosNext   = fabs(+dot(nextRay_dir, surfElem.normal));

  if (a_currDepth == a_lightTraceDepth)
  {
    a_pOutLightVertex->hit       = surfElem;
    a_pOutLightVertex->ray_dir   = ray_dir;
    a_pOutLightVertex->accColor  = a_color;
    a_pOutLightVertex->valid     = true;
    a_pOutLightVertex->lastGTerm = GTermPrev;
    return;
  }

  // If we sampled specular event, then the reverse probability
  // cannot be evaluated, but we know it is exactly the same as
  // forward probability, so just set it. If non-specular event happened,
  // we evaluate the pdf
  //
  if (!isPureSpecular(matSam))
  {
    ShadeContext sc;
    sc.wp = surfElem.pos;
    sc.l = (-1.0f)*ray_dir;
    sc.v = (-1.0f)*nextRay_dir;
    sc.n = surfElem.normal;
    sc.fn = surfElem.flatNormal;
    sc.tg = surfElem.tangent;
    sc.bn = surfElem.biTangent;
    sc.tc = surfElem.texCoord;

    const float pdfW         = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage).pdfFwd;
    const float prevPdfRevWP = pdfW / fmax(cosCurr, DEPSILON);
    a_perThread->pdfArray[a_currDepth].pdfRev = prevPdfRevWP*GTermPrev;
  }
  else
  {
    a_perThread->pdfArray[a_currDepth].pdfRev = -1.0f*GTermPrev;
  }

  a_color *= matSam.color*cosNext* (1.0f / fmax(matSam.pdf, DEPSILON2));

  TraceLightPath(nextRay_pos, nextRay_dir, a_currDepth + 1, cosNext, matSam.pdf, 
                 a_color, a_perThread, a_lightTraceDepth, isPureSpecular(matSam),
                 a_pOutLightVertex);
}

PathVertex IntegratorSBDPT::CameraPath(float3 ray_pos, float3 ray_dir, float3 a_prevNormal, MisData a_misPrev, int a_currDepth, uint flags,
                                       PerThreadData* a_perThread, int a_targetDepth, bool a_haveToHitLightSource, int a_fullPathDepth)

{
  const int prevVertexId = a_fullPathDepth - a_currDepth + 1; //#CHECK_THIS

  if (a_currDepth > a_targetDepth)
  {
    PathVertex resVertex;
    resVertex.valid    = false;
    resVertex.accColor = float3(0, 0, 0);
    return resVertex;
  }

  const Lite_Hit hit = rayTrace(ray_pos, ray_dir);

  if (HitNone(hit))
  {
    PathVertex resVertex;
    resVertex.valid    = false;
    resVertex.accColor = float3(0, 0, 0);
    return resVertex;
  }

  const SurfaceHit surfElem = surfaceEval(ray_pos, ray_dir, hit);

  PerThread().vert[a_currDepth] = surfElem.pos;

  const float cosHere = fabs(dot(ray_dir, surfElem.normal));
  const float cosPrev = fabs(dot(ray_dir, a_prevNormal));
  float GTerm = 1.0f;
  
  if (a_currDepth == 1)
  {
    float3 camDirDummy; float zDepthDummy;
    const float imageToSurfaceFactor = CameraImageToSurfaceFactor(surfElem.pos, surfElem.normal, m_pGlobals,
                                                                  &camDirDummy, &zDepthDummy);
    const float cameraPdfA = imageToSurfaceFactor / mLightSubPathCount;
    a_perThread->pdfArray[a_fullPathDepth].pdfRev = cameraPdfA;
    a_perThread->pdfArray[a_fullPathDepth].pdfFwd = 1.0f;

    PerThread().grammarCam.push_back('E');
  }
  else
  {
    const float dist = length(ray_pos - surfElem.pos);
    
    GTerm = cosHere*cosPrev / fmax(dist*dist, DEPSILON2);
  }

  float3 emission = emissionEval(ray_pos, ray_dir, surfElem, flags, a_misPrev, fetchInstId(hit));
  if (dot(emission, emission) > 1e-3f)
  {
    PathVertex resVertex;
    if (a_currDepth == a_targetDepth && a_haveToHitLightSource)
    {
      const int instId         = fetchInstId(hit);
      const int lightOffset    = m_geom.instLightInstId[instId];
      const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
      
      const LightPdfFwd lPdfFwd = lightPdfFwd(pLight, ray_dir, cosHere, m_pGlobals, m_texStorage, m_pdfStorage);
      const float pdfLightWP    = lPdfFwd.pdfW / fmax(cosHere, DEPSILON);
      const float pdfMatRevWP   = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);

      a_perThread->pdfArray[0].pdfFwd = lPdfFwd.pdfA/float(m_pGlobals->lightsNum);
      a_perThread->pdfArray[0].pdfRev = 1.0f;

      a_perThread->pdfArray[1].pdfFwd = pdfLightWP*GTerm;
      a_perThread->pdfArray[1].pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfMatRevWP*GTerm;

      PerThread().grammarCam.push_back('L');

      resVertex.accColor = emission;
      resVertex.valid    = true;
      return resVertex;
    }
    else // this branch could brobably change in future, for simple emissive materials
    {
      resVertex.accColor = float3(0, 0, 0);
      resVertex.valid    = false;
      return resVertex;
    }
   
  }
  else if (a_currDepth == a_targetDepth && !a_haveToHitLightSource) // #NOTE: what if a_targetDepth == 1 ?
  {
    PathVertex resVertex;
    resVertex.hit          = surfElem;
    resVertex.ray_dir      = ray_dir;
    resVertex.valid        = true;
    resVertex.accColor     = float3(1, 1, 1);

    if (a_targetDepth != 1)
    {
      const float lastPdfWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON); // we store them to calculate fwd and rev pdf later when we connect end points
      resVertex.lastGTerm   = GTerm;                                            // because right now we can not do this until we don't know the light vertex

      //a_perThread->pdfArray[prevVertexId].pdfFwd = ... // do this later, inside ConnectShadow or ConnectEndPoints
      a_perThread->pdfArray[prevVertexId].pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : GTerm*lastPdfWP;
    }
    else
      resVertex.lastGTerm = 1.0f;
    
    return resVertex;
  }

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, surfElem.matId);
  const MatSample matSam = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));
  const float3 bxdfVal   = matSam.color; // *(1.0f / fmaxf(matSam.pdf, 1e-20f));
  const float cosNext    = fabs(dot(matSam.direction, surfElem.normal));

  ///////////////////////////////////////////////////////////////////////////////////////////////////////// DEBUG
  if (isPureSpecular(matSam))
    PerThread().grammarCam.push_back('S');
  else if (isGlossy(matSam))
    PerThread().grammarCam.push_back('G');
  else
    PerThread().grammarCam.push_back('D');
  ///////////////////////////////////////////////////////////////////////////////////////////////////////// DEBUG

  // eval reverse and forward pdfs
  //
  if (a_currDepth == 1)
  {
    if (isPureSpecular(matSam))  //  ow ... but if we met specular reflection when tracing from camera, we must put 0 because this path cannot be sample by light strategy at all.
      a_perThread->pdfArray[a_fullPathDepth].pdfFwd = 0.0f;
  }
  else
  {
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

      const float pdfFwdW  = materialEval(pHitMaterial, &sc, false, false, /* global data --> */ m_pGlobals, m_texStorage, m_texStorage).pdfFwd;
      const float pdfFwdWP = pdfFwdW / fmax(cosHere, DEPSILON);

      a_perThread->pdfArray[prevVertexId].pdfFwd = pdfFwdWP*GTerm;
    }
    else
      a_perThread->pdfArray[prevVertexId].pdfFwd = -1.0f*GTerm;

    const float pdfCamPrevWP = a_misPrev.matSamplePdf / fmax(cosPrev, DEPSILON);
    a_perThread->pdfArray[prevVertexId].pdfRev = a_misPrev.isSpecular ? -1.0f*GTerm : pdfCamPrevWP*GTerm;
  }
  
 
  // proceed to next bounce
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  MisData thisBounce       = makeInitialMisData();
  thisBounce.isSpecular    = isPureSpecular(matSam);
  thisBounce.matSamplePdf  = matSam.pdf;

  PathVertex nextVertex = CameraPath(nextRay_pos, nextRay_dir, surfElem.normal, thisBounce, a_currDepth + 1, flags,
                                     a_perThread, a_targetDepth, a_haveToHitLightSource, a_fullPathDepth);

  nextVertex.accColor *= (bxdfVal*cosNext / fmax(matSam.pdf, DEPSILON2));

  return nextVertex;
}



float3 IntegratorSBDPT::ConnectEye(const PathVertex& a_lv, int a_ltDepth,
                                   PerThreadData* a_perThread, int* pX, int* pY)
{

  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(a_lv.hit.pos, a_lv.hit.normal, m_pGlobals,
                                                                &camDir, &zDepth);

  const PlainMaterial* pHitMaterial = materialAt(m_pGlobals, m_matStorage, a_lv.hit.matId);
  float signOfNormal = 1.0f;
  if ((materialGetFlags(pHitMaterial) & PLAIN_MATERIAL_HAVE_BTDF) != 0 && dot(camDir, a_lv.hit.normal) < -0.01f)
    signOfNormal = -1.0f;

  auto hit = rayTrace(a_lv.hit.pos + epsilonOfPos(a_lv.hit.pos)*signOfNormal*a_lv.hit.normal, camDir);
  
  float3 result(0, 0, 0);

  ::ConnectEyeP(a_lv, a_ltDepth, mLightSubPathCount, hit, 
                m_pGlobals, m_matStorage, m_texStorage, m_texStorageAux, 
                &a_perThread->pdfArray[0], pX, pY, &result);

  return result;
}


float3 IntegratorSBDPT::ConnectShadow(const PathVertex& a_cv, PerThreadData* a_perThread, const int a_camDepth)
{
  float3 explicitColor(0, 0, 0);

  const SurfaceHit& surfElem = a_cv.hit;

  auto& gen = randomGen();
  LightGroup2 lightSelector;
  RndLightMMLT(&gen, gen.rptr,
               &lightSelector);

  float lightPickProb = 1.0f;
  int lightOffset = SelectRandomLightRev(make_float2(lightSelector.group2.z, lightSelector.group2.w), surfElem.pos, m_pGlobals,
                                         &lightPickProb);

  if (!m_computeIndirectMLT && lightOffset >= 0) // if need to sample direct light ?
  {
   
    __global const PlainLight* pLight = lightAt(m_pGlobals, lightOffset);
    
    ShadowSample explicitSam;
    LightSampleRev(pLight, rndFloat3(&gen), surfElem.pos, m_pGlobals, m_pdfStorage, m_texStorage,
                   &explicitSam);
    
    const float3 shadowRayDir = normalize(explicitSam.pos - surfElem.pos); // explicitSam.direction;
    const float3 shadowRayPos = OffsRayPos(surfElem.pos, surfElem.normal, shadowRayDir);
    const float3 shadow       = shadowTrace(shadowRayPos, shadowRayDir, explicitSam.maxDist*0.9995f);
    
    if (dot(shadow, shadow) > 1e-12f)
    {
      explicitColor = shadow*ConnectShadowP(a_cv, a_camDepth, pLight, explicitSam, lightPickProb,
                                            m_pGlobals, m_matStorage, m_texStorage, m_texStorageAux, m_pdfStorage,
                                            &a_perThread->pdfArray[0]);
    }
  }

  return explicitColor;
}

float3 IntegratorSBDPT::ConnectEndPoints(const PathVertex& a_lv, const PathVertex& a_cv, const int a_spit, const int a_depth,
                                         PerThreadData* a_perThread)
{
  if (!a_lv.valid || !a_cv.valid)
    return float3(0, 0, 0);

  const float3 diff = a_cv.hit.pos - a_lv.hit.pos;
  const float dist2 = fmax(dot(diff, diff), DEPSILON2);
  const float  dist = sqrtf(dist2);
  const float3 lToC = diff / dist; 

  const float3 shadowRayDir = lToC;
  const float3 shadowRayPos = OffsRayPos(a_lv.hit.pos, a_lv.hit.normal, shadowRayDir); 
  const float3 shadow       = shadowTrace(shadowRayPos, shadowRayDir, dist*0.9995f);

  if (dot(shadow, shadow) < 1e-12f)
    return float3(0, 0, 0);
  else
    return shadow*ConnectEndPointsP(a_lv, a_cv, a_spit, a_depth,
                                    m_pGlobals, m_matStorage, m_texStorage, m_texStorageAux,
                                    &a_perThread->pdfArray[0]);
}




void IntegratorSBDPT::DebugOutCurrPath(int d)
{
  static std::ofstream fout("zpath.txt");

  if (!fout.is_open())
    return;

  #pragma omp critical
  {
    fout << d << "\t" << PerThread().grammarCam.c_str() << "\t";
    for (int i = 1; i <= d; i++)
      fout << PerThread().vert[i].x << " " << PerThread().vert[i].y << " " << PerThread().vert[i].z << "\t";
    fout << std::endl;
  }

}

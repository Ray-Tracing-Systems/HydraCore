#include <omp.h>
#include "CPUExp_Integrators.h"

void IntegratorLT::DoPass(std::vector<uint>& a_imageLDR)
{
  const int samplesPerPass = m_width*m_height;
  mLightSubPathCount = float(samplesPerPass);

  #ifdef NDEBUG
  #pragma omp parallel for
  #endif
  for (int i = 0; i < samplesPerPass; i++)
    DoLightPath(i);

  constexpr float gammaPow = 1.0f/2.2f;
  const float scaleInv     = 1.0f / float(m_spp + 1);

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

  m_spp++;
  std::cout << "IntegratorLT: spp  = " << m_spp << std::endl;

}


void IntegratorLT::DoLightPath(int iterId)
{
  auto& rgen = randomGen();

  float lightPickProb = 1.0f;
  const int lightId   = SelectRandomLightFwd(rndFloat1(&rgen), m_pGlobals,
                                             &lightPickProb);

  const PlainLight* pLight = lightAt(m_pGlobals, lightId);

  const float4 rands1 = rndFloat4(&rgen);
  const float2 rands2 = rndFloat2(&rgen);

  LightSampleFwd sample;
  LightSampleForward(pLight, rands1, rands2, m_pGlobals, m_texStorage, m_pdfStorage,
                     &sample);
  
  const float invPdf = 1.0f / (lightPickProb*fmax(sample.pdfA*sample.pdfW, DEPSILON2));

  PerThread().selectedLightIdFwd = lightId;

  // draw sample on light surface
  //
  if(lightType(pLight) != PLAIN_LIGHT_TYPE_DIRECT && lightType(pLight) != PLAIN_LIGHT_TYPE_SKY_DOME)
  {
    SurfaceHit dummyHit;
    dummyHit.pos    = sample.pos;
    dummyHit.normal = sample.norm;
    ConnectEye(dummyHit, sample.dir, sample.color*invPdf, 0);
  }

  TraceLightPath(sample.pos, sample.dir, 1, sample.color*invPdf);
}

void IntegratorLT::TraceLightPath(float3 ray_pos, float3 ray_dir, int a_currDepth, float3 a_accColor)
{
 if (a_currDepth >= m_maxDepth)
    return;

  auto hit = rayTrace(ray_pos, ray_dir);
  if (!HitSome(hit))
    return;

  auto surfElem = surfaceEval(ray_pos, ray_dir, hit);

  ConnectEye(surfElem, ray_dir, a_accColor, a_currDepth);

  // next bounce
  //
  const MatSample matSam   = std::get<0>(sampleAndEvalBxDF(ray_dir, surfElem));

  // calc new ray
  //
  const float3 nextRay_dir = matSam.direction;
  const float3 nextRay_pos = OffsRayPos(surfElem.pos, surfElem.normal, matSam.direction);

  const float cosLightNext = fabs(dot(nextRay_dir, surfElem.normal));

  TraceLightPath(nextRay_pos, nextRay_dir, a_currDepth + 1, cosLightNext*matSam.color*a_accColor*(1.0f / fmax(matSam.pdf, DEPSILON)));
}



void IntegratorLT::ConnectEye(SurfaceHit a_hit, float3 ray_dir, float3 a_accColor, int a_currBounce)
{
  // We put the virtual image plane at such a distance from the camera origin
  // that the pixel area is one and thus the image plane sampling pdf is 1.
  // The area pdf of aHitpoint as sampled from the camera is then equal to
  // the conversion factor from image plane area density to surface area density
  //
  float3 camDir; float zDepth;
  const float imageToSurfaceFactor = CameraImageToSurfaceFactor(a_hit.pos, a_hit.normal, m_pGlobals, make_float2(0,0),
                                                                &camDir, &zDepth);

  if (imageToSurfaceFactor <= 0.0f)
    return;

  float  signOfNormal  = 1.0f; 
  float3 colorConnect  = make_float3(1,1,1);
  if(a_currBounce > 0) // if 0, this is light surface
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
  
    BxDFResult matRes = materialEval(pHitMaterial, &sc, (EVAL_FLAG_FWD_DIR), /* global data --> */ m_pGlobals, m_texStorage, m_texStorageAux, &m_ptlDummy);
    colorConnect = matRes.brdf + matRes.btdf; 
  }

  // We divide the contribution by surfaceToImageFactor to convert the (already
  // divided) pdf from surface area to image plane area, w.r.t. which the
  // pixel integral is actually defined. We also divide by the number of samples
  // this technique makes, which is equal to the number of light sub-paths
  //
  float3 sampleColor = 1.0f*(a_accColor*colorConnect) * (imageToSurfaceFactor / mLightSubPathCount);

  // if(length(sampleColor) > 1000.0f)
  //   std::cout << "sampleColor = (" << sampleColor.x << ", " << sampleColor.y << ", " << sampleColor.z << std::endl;

  if (dot(sampleColor, sampleColor) > 1e-20f) // add final result to image
  {
    const float3 shadowRayPos = a_hit.pos + epsilonOfPos(a_hit.pos)*signOfNormal*a_hit.normal;

    auto hit = rayTrace(shadowRayPos, camDir);

    if (!HitSome(hit) || hit.t > zDepth)
    {
      const float2 posScreenSpace  = worldPosToScreenSpace(a_hit.pos, m_pGlobals);
      //const float2 posScreenSpace2 = worldPosToScreenSpaceWithDOF(a_hit.pos, m_pGlobals, float2(0,0));

      const int x = int(posScreenSpace.x + 0.5f);
      const int y = int(posScreenSpace.y + 0.5f);

      if(x >=0 && x <= m_width-1 && y >=0 && y <= m_height-1)
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


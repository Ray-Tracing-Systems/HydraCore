
float3 prtex1_mix(float3 x, float3 y, float a)
{
  return x*(1.0f - a) + y*a;
}

float4 prtex1_main(const SurfaceInfo* sHit, float3 color1, float3 color2, _PROCTEXTAILTAG_)
{
  const float3 pos  = readAttr_LocalPos(sHit);
  const float3 norm = readAttr_ShadeNorm(sHit);
  
  const float3 rayDir = hr_viewVectorHack;
  float cosAlpha      = fabs(dot(norm,rayDir));
  
  return prtex1_mix(color1, color2, cosAlpha);
}



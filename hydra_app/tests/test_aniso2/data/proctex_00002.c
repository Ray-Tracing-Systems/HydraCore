float3 prtex2_abs3(float3 a)
{
  return make_float3(fabs(a.x), fabs(a.y), fabs(a.z));
}


float4 prtex2_main(const SurfaceInfo* sHit, sampler2D texX1, sampler2D texY1, sampler2D texZ1, sampler2D texX2, sampler2D texY2, sampler2D texZ2, float blendSize, float mapScale, _PROCTEXTAILTAG_)
{
  const float3 norm = readAttr_ShadeNorm(sHit);
  const float3 pos = readAttr_WorldPos(sHit);

  sampler2D texX = norm.x < 0 ? texX1 : texX2;
  sampler2D texY = norm.y < 0 ? texY1 : texY2;
  sampler2D texZ = norm.z < 0 ? texZ1 : texZ2;

  float3 w = prtex2_abs3(norm);
  w.x = pow(w.x, blendSize);
  w.y = pow(w.y, blendSize);
  w.z = pow(w.z, blendSize);
  w = max(w, 0.00001) / dot(w, w); 

  float b = (w.x + w.y + w.z);
  w.x = w.x / b;
  w.y = w.y / b;
  w.z = w.z / b;

  float2 y_uv = make_float2(pos.x / mapScale, pos.z / mapScale);
  float2 x_uv = make_float2(pos.z / mapScale, pos.y / mapScale);
  float2 z_uv = make_float2(pos.x / mapScale, pos.y / mapScale);

  float4 texColX = texture2D(texX, x_uv, 0);
  float4 texColY = texture2D(texY, y_uv, 0);
  float4 texColZ = texture2D(texZ, z_uv, 0);

  float4 res = texColX * w.x + texColY * w.y + texColZ * w.z;

  return res;
}




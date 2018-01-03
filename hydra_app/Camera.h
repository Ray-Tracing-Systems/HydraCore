// © Copyright 2017 Vladimir Frolov, Ray Tracing Systems
//

#pragma once
#include "../../HydraAPI/hydra_api/LiteMath.h"
using namespace HydraLiteMath;

struct Camera
{
  Camera() : pos(0.0f, 1.0f, -10.0f), lookAt(0, 0, 0), up(0, 1, 0), fov(45.0f), tdist(100.0f) {}

  float3 pos;
  float3 lookAt;
  float  fov;
  float3 up;
  float  tdist;

  float3 forward() const { return normalize(lookAt - pos); }
  float3 right()   const { return cross(forward(), up); }

  void offsetOrientation(float a_upAngle, float rightAngle)
  {
    if (a_upAngle != 0.0f)  // rotate vertical
    {
      float3 direction = normalize(forward() * cosf(-TORADIANS*a_upAngle) + up * sinf(-TORADIANS*a_upAngle));

      up     = normalize(cross(right(), direction));
      lookAt = pos + tdist*direction;
    }

    if (rightAngle != 0.0f)  // rotate horizontal
    {
      float4x4 rot;
      rot.identity();
      rot.M(0, 0) = rot.M(2, 2) = cosf(-TORADIANS*rightAngle);
      rot.M(0, 2) = -sinf(-TORADIANS*rightAngle);
      rot.M(2, 0) = sinf(-TORADIANS*rightAngle);

      float3 direction2 = normalize(mul(rot, forward()));
      up     = normalize(mul(rot, up));
      lookAt = pos + tdist*direction2;
    }
  }

  void offsetPosition(float3 a_offset)
  {
    pos    += a_offset;
    lookAt += a_offset;
  }

protected:
  const float TORADIANS = 0.01745329251994329576923690768489f;
};

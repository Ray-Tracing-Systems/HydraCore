#pragma once

#include "../../../HydraCore/hydra_drv/IBVHBuilderAPI.h"
#include "../../../HydraAPI/hydra_api/aligned_alloc.h"

#include "../common/tutorial/tutorial_device.h"
using namespace embree;

namespace EarlySplit
{
  struct Box3f
  {
    Box3f() : vmin(1e38f, 1e38f, 1e38f), vmax(-1e38f, -1e38f, -1e38f) {}
    float3 vmin;
    float3 vmax;

    inline float3 center() const { return 0.5f*(vmin + vmax); }

    inline bool axisAligned(int axis, float split) const
    {
      float amin[3] = { vmin.x, vmin.y, vmin.z };
      float amax[3] = { vmax.x, vmax.y, vmax.z };

      return (amin[axis] == amax[axis]) && (amin[axis] == split);
    }

    inline void include(const float3 in_v)
    {
      vmin.x = fminf(vmin.x, in_v.x);
      vmin.y = fminf(vmin.y, in_v.y);
      vmin.z = fminf(vmin.z, in_v.z);

      vmax.x = fmaxf(vmax.x, in_v.x);
      vmax.y = fmaxf(vmax.y, in_v.y);
      vmax.z = fmaxf(vmax.z, in_v.z);
    }

    inline void intersect(const Box3f& in_box)
    {
      vmin.x = fmaxf(vmin.x, in_box.vmin.x);
      vmin.y = fmaxf(vmin.y, in_box.vmin.y);
      vmin.z = fmaxf(vmin.z, in_box.vmin.z);

      vmax.x = fminf(vmax.x, in_box.vmax.x);
      vmax.y = fminf(vmax.y, in_box.vmax.y);
      vmax.z = fminf(vmax.z, in_box.vmax.z);
    }
  };

  struct Triangle
  {
    Triangle() {}

    float4 A;
    float4 B;
    float4 C;
  };

  struct TriRef
  {
    ALIGNED_STRUCT
    Box3f box;
    int   triId;
    float metric;

    unsigned int dummy1;
    unsigned int dummy2;
    unsigned int dummy3;
    unsigned int geomID;
  };

  static void TriRefBoundsFunc(const TriRef* spheres, size_t item, RTCBounds* bounds_o)
  {
    const TriRef& tr = spheres[item];
    
    bounds_o->lower_x = tr.box.vmin.x;
    bounds_o->lower_y = tr.box.vmin.y;
    bounds_o->lower_z = tr.box.vmin.z;

    bounds_o->upper_x = tr.box.vmax.x;
    bounds_o->upper_y = tr.box.vmax.y;
    bounds_o->upper_z = tr.box.vmax.z;
  }

  static inline Box3f TriBounds(const Triangle& a_tri)
  {
    Box3f res;

    res.vmin.x = fminf(a_tri.A.x, fminf(a_tri.B.x, a_tri.C.x));
    res.vmin.y = fminf(a_tri.A.y, fminf(a_tri.B.y, a_tri.C.y));
    res.vmin.z = fminf(a_tri.A.z, fminf(a_tri.B.z, a_tri.C.z));

    res.vmax.x = fmaxf(a_tri.A.x, fmaxf(a_tri.B.x, a_tri.C.x));
    res.vmax.y = fmaxf(a_tri.A.y, fmaxf(a_tri.B.y, a_tri.C.y));
    res.vmax.z = fmaxf(a_tri.A.z, fmaxf(a_tri.B.z, a_tri.C.z));

    return res;
  }

  static inline float SurfaceArea(const Box3f& a_box)
  {
    float a = a_box.vmax.x - a_box.vmin.x;
    float b = a_box.vmax.y - a_box.vmin.y;
    float c = a_box.vmax.z - a_box.vmin.z;
    return 2.0f * (a*b + a*c + b*c);
  }

  static inline float SurfaceAreaOfTriangle(const Triangle& tri) { return length(cross(to_float3(tri.C) - to_float3(tri.A), to_float3(tri.B) - to_float3(tri.A))); }
  static inline float SurfaceAreaOfTriangle(const float3 v[3]) { return length(cross(v[2] - v[0], v[1] - v[0])); }

  static const float SubdivMetric(const Triangle& a_tri, const Box3f& a_box)
  {
    float triSA = SurfaceAreaOfTriangle(a_tri);
    float boxSA = SurfaceArea(a_box);
    return (boxSA*boxSA) / fmaxf(triSA, 1e-6f);
  }

  static inline float& f3_at(float3& a_f, const int a_index)
  {
    float* pArr = &a_f.x;
    return pArr[a_index];
  }


};

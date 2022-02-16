#include "CamHostPluginAPI.h"
#include <iostream>
#include <thread> // just for test big delay
#include <chrono> // std::chrono::seconds

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#include "../hydra_drv/cglobals.h"
#include "../../HydraAPI/hydra_api/HydraAPI.h"

class SimpleDOF : public IHostRaysAPI
{
public:
  SimpleDOF() { hr_qmc::init(table); m_globalCounter = 0; }
  
  void SetParameters(int a_width, int a_height, const float a_projInvMatrix[16], pugi::xml_node a_camNode) override
  {
    m_fwidth  = float(a_width);
    m_fheight = float(a_height);
    memcpy(&m_projInv, a_projInvMatrix, sizeof(float4x4));
    ReadParamsFromNode(a_camNode);
  }

  void ReadParamsFromNode(pugi::xml_node a_camNode);

  void MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize) override;
  void AddSamplesContribution(float* out_color4f, const float* colors4f, size_t in_blockSize, uint32_t a_width, uint32_t a_height) override;

  unsigned int table[hr_qmc::QRNG_DIMENSIONS][hr_qmc::QRNG_RESOLUTION];
  unsigned int m_globalCounter = 0;

  float m_fwidth  = 1024.0f;
  float m_fheight = 1024.0f;
  float4x4 m_projInv;

  float FOCAL_PLANE_DIST = 10.0f;
  float DOF_LENS_RADIUS  = 0.0f;
  bool  DOF_IS_ENABLED = false;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleDOF::ReadParamsFromNode(pugi::xml_node a_camNode)
{
  if (a_camNode.child(L"enable_dof").text().empty())
    return;

  int hasDof = a_camNode.child(L"enable_dof").text().as_int();
  if (hasDof > 0)
  {
    DOF_IS_ENABLED  = true;
    DOF_LENS_RADIUS = a_camNode.child(L"dof_lens_radius").text().as_float();
    
    // compute 'FOCAL_PLANE_DIST' from camPos and lookAt parameters 
    //
    float3 camPos, camLookAt;
    {
      const wchar_t* camPosStr = a_camNode.child(L"position").text().as_string();
      const wchar_t* camLAtStr = a_camNode.child(L"look_at").text().as_string();
      if (std::wstring(camPosStr) != L"")
      {
        std::wstringstream input(camPosStr);
        input >> camPos.x >> camPos.y >> camPos.z;
      }
      if (std::wstring(camLAtStr) != L"")
      {
        std::wstringstream input(camLAtStr);
        input >> camLookAt.x >> camLookAt.y >> camLookAt.z;
      }
    }
    FOCAL_PLANE_DIST = length(camPos - camLookAt);
  }
  else
    DOF_IS_ENABLED = false;
}

static inline int myPackXY1616(int x, int y) { return (y << 16) | (x & 0x0000FFFF); }

void SimpleDOF::MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize)
{
  #pragma omp parallel for
  for(int i=0;i<in_blockSize;i++)
  {
    const float rndX = hr_qmc::rndFloat(m_globalCounter+i, 0, table[0]);
    const float rndY = hr_qmc::rndFloat(m_globalCounter+i, 1, table[0]);
    
    const float x    = m_fwidth*rndX; 
    const float y    = m_fheight*rndY;

    float3 ray_pos = float3(0,0,0);
    float3 ray_dir = EyeRayDir(x, y, m_fwidth, m_fheight, m_projInv);

    if (DOF_IS_ENABLED) // dof is enabled
    {
      const float lenzX = hr_qmc::rndFloat(m_globalCounter+i, 2, table[0]);
      const float lenzY = hr_qmc::rndFloat(m_globalCounter+i, 3, table[0]);

      const float tFocus         = FOCAL_PLANE_DIST / (-ray_dir.z);
      const float3 focusPosition = ray_pos + ray_dir*tFocus;
      const float2 xy            = DOF_LENS_RADIUS*2.0f*MapSamplesToDisc(float2(lenzX - 0.5f, lenzY - 0.5f));
      ray_pos.x += xy.x;
      ray_pos.y += xy.y;
  
      ray_dir = normalize(focusPosition - ray_pos);
    }


    RayPart1 p1;
    p1.origin[0]   = ray_pos.x;
    p1.origin[1]   = ray_pos.y;
    p1.origin[2]   = ray_pos.z;
    p1.xyPosPacked = myPackXY1616(int(x), int(y));
   
    RayPart2 p2;
    p2.direction[0] = ray_dir.x;
    p2.direction[1] = ray_dir.y;
    p2.direction[2] = ray_dir.z;
    p2.dummy        = 0.0f;
    
    out_rayPosAndNear[i] = p1;
    out_rayDirAndFar [i] = p2;
  }

  //std::this_thread::sleep_for(std::chrono::milliseconds(50)); // test big delay

  m_globalCounter += unsigned(in_blockSize);
} 

void SimpleDOF::AddSamplesContribution(float* out_color4f, const float* colors4f, size_t in_blockSize, uint32_t a_width, uint32_t a_height)
{
  float4*       out_color = (float4*)out_color4f;
  const float4* colors    = (const float4*)colors4f;
  
  for (int i = 0; i < in_blockSize; i++)
  {
    const auto color = colors[i];
    const uint32_t packedIndex = as_int(color.w);
    const int x      = (packedIndex & 0x0000FFFF);         ///<! extract x position from color.w
    const int y      = (packedIndex & 0xFFFF0000) >> 16;   ///<! extract y position from color.w
    const int offset = y*a_width + x;

    if (x >= 0 && y >= 0 && x < a_width && y < a_height)
    {
      out_color[offset].x += color.x;
      out_color[offset].y += color.y;
      out_color[offset].z += color.z;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TableLens : public IHostRaysAPI
{
public:
  TableLens() { hr_qmc::init(table); m_globalCounter = 0; }
  
  void SetParameters(int a_width, int a_height, const float a_projInvMatrix[16], pugi::xml_node a_camNode) override
  {
    m_fwidth  = float(a_width);
    m_fheight = float(a_height);
    memcpy(&m_projInv, a_projInvMatrix, sizeof(float4x4));
    ReadParamsFromNode(a_camNode);
    RunTestRays();
  }

  void ReadParamsFromNode(pugi::xml_node a_camNode);
  void RunTestRays();

  void MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize) override;
  void AddSamplesContribution(float* out_color4f, const float* colors4f, size_t in_blockSize, uint32_t a_width, uint32_t a_height) override;

  unsigned int table[hr_qmc::QRNG_DIMENSIONS][hr_qmc::QRNG_RESOLUTION];
  unsigned int m_globalCounter = 0;

  float m_fwidth  = 1024.0f;
  float m_fheight = 1024.0f;
  float4x4 m_projInv;

  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  

  bool TraceLensesFromFilm(const float3 inRayPos, const float3 inRayDir, 
                           float3* outRayPos, float3* outRayDir) const;

  bool  IntersectSphericalElement(float radius, float zCenter, const float3 rayPos, const float3 rayDir, 
                                  float *t, float3 *n) const;

  struct LensElementInterface {
    float curvatureRadius;
    float thickness;
    float eta;
    float apertureRadius;
  };

  struct LensElementInterfaceWithId {
    LensElementInterface lensElement;
    int id;
  };
  
  
  std::vector<LensElementInterface> lines;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string  ws2s(const std::wstring& s);

void TableLens::ReadParamsFromNode(pugi::xml_node a_camNode)
{
  if (a_camNode.child(L"enable_dof").text().empty())
    return;
  
  auto opticalSys = a_camNode.child(L"optical_system");
  if(opticalSys == nullptr)
  {
    std::string camName = ws2s(a_camNode.attribute(L"name").as_string());
    std::cout << "[TableLens::ReadParamsFromNode]: node 'optical_system' is not found for camera " << camName.c_str() << std::endl;
    return;
  }

  std::vector<LensElementInterfaceWithId> ids;
  int currId = 0;
  for(auto line : opticalSys.children(L"line"))
  {
    LensElementInterface layer;
    int id = currId;
    if(line.attribute(L"id") != nullptr)
      id = line.attribute(L"id").as_int();
    layer.curvatureRadius = line.attribute(L"curvature_radius").as_float();
    layer.thickness       = line.attribute(L"thickness").as_float();
    layer.eta             = line.attribute(L"ior").as_float();
    if(line.attribute(L"semi_diameter") != nullptr)
      layer.apertureRadius  = 2.0f*line.attribute(L"semi_diameter").as_float();
    else if(line.attribute(L"aperture_radius") != nullptr)
      layer.apertureRadius  = 1.0f*line.attribute(L"aperture_radius").as_float();
    
    LensElementInterfaceWithId layer2;
    layer2.lensElement = layer;
    layer2.id          = id;
    ids.push_back(layer2);
    currId++;
  }
  
  // you may sort 'lines' by 'ids' if you want 
  //
  std::wstring order = opticalSys.attribute(L"order").as_string();
  if(order == L"scene_to_sensor")
    std::sort(ids.begin(), ids.end(), [](const auto& a, const auto& b) { return a.id > b.id; });
  else
    std::sort(ids.begin(), ids.end(), [](const auto& a, const auto& b) { return a.id < b.id; });

  lines.resize(ids.size());
  for(size_t i=0;i<ids.size(); i++)
    lines[i] = ids[i].lensElement;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon();

static inline bool Quadratic(float A, float B, float C, float *t0, float *t1) {
  // Find quadratic discriminant
  double discrim = (double)B * (double)B - 4. * (double)A * (double)C;
  if (discrim < 0.) 
    return false;
  double rootDiscrim = std::sqrt(discrim);
  float floatRootDiscrim   = rootDiscrim;
  //float floatRootDiscrimErr = MachineEpsilon * rootDiscrim;
  // Compute quadratic _t_ values
  float q;
  if ((float)B < 0)
      q = -.5 * (B - floatRootDiscrim);
  else
      q = -.5 * (B + floatRootDiscrim);
  *t0 = q / A;
  *t1 = C / q;
  if ((float)*t0 > (float)*t1) 
    std::swap(*t0, *t1);
  return true;
}

static inline bool Refract(const float3 wi, const float3 n, float eta, float3 *wt) {
  // Compute $\cos \theta_\roman{t}$ using Snell's law
  float cosThetaI  = dot(n, wi);
  float sin2ThetaI = std::max(float(0), float(1.0f - cosThetaI * cosThetaI));
  float sin2ThetaT = eta * eta * sin2ThetaI;
  // Handle total internal reflection for transmission
  if (sin2ThetaT >= 1) return false;
  float cosThetaT = std::sqrt(1 - sin2ThetaT);
  *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
  return true;
}


static inline float3 faceforward(const float3 n, const float3 v) { return (dot(n, v) < 0.f) ? -n : n; }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool TableLens::IntersectSphericalElement(float radius, float zCenter, const float3 rayPos, const float3 rayDir, 
                                          float *t, float3 *n) const
{
  // Compute _t0_ and _t1_ for ray--element intersection
  const float3 o = rayPos - float3(0, 0, zCenter);
  const float  A = rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z;
  const float  B = 2 * (rayDir.x * o.x + rayDir.y * o.y + rayDir.z * o.z);
  const float  C = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
  float  t0, t1;
  if (!Quadratic(A, B, C, &t0, &t1)) 
    return false;
  
  // Select intersection $t$ based on ray direction and element curvature
  bool useCloserT = (rayDir.z > 0.0f) ^ (radius < 0.0);
  *t = useCloserT ? std::min(t0, t1) : std::max(t0, t1);
  if (*t < 0.0f) 
    return false;
  
  // Compute surface normal of element at ray intersection point
  *n = normalize(o + (*t)*rayDir);
  *n = faceforward(*n, -1.0f*rayDir);
  return true;
}

bool TableLens::TraceLensesFromFilm(const float3 inRayPos, const float3 inRayDir, 
                                    float3* outRayPos, float3* outRayDir) const
{
  float elementZ = 0;
  // Transform _rCamera_ from camera to lens system space
  // 
  float3 rayPosLens = float3(inRayPos.x, inRayPos.y, +inRayPos.z);
  float3 rayDirLens = float3(inRayDir.x, inRayDir.y, +inRayDir.z);

  for(int i=0; i<lines.size(); i++)
  {
    const LensElementInterface& element = lines[i];                                  
    // Update ray from film accounting for interaction with _element_
    elementZ -= element.thickness;
    
    // Compute intersection of ray with lens element
    float t;
    float3 n;
    bool isStop = (element.curvatureRadius == 0.0f);
    if (isStop) 
    {
      // The refracted ray computed in the previous lens element
      // interface may be pointed towards film plane(+z) in some
      // extreme situations; in such cases, 't' becomes negative.
      if (rayDirLens.z >= 0.0f) 
        return false;
      t = (elementZ - rayPosLens.z) / rayDirLens.z;
    } 
    else 
    {
      const float radius  = element.curvatureRadius;
      const float zCenter = elementZ + element.curvatureRadius;
      if (!IntersectSphericalElement(radius, zCenter, rayPosLens, rayDirLens, &t, &n))
        return false;
    }

    // Test intersection point against element aperture
    const float3 pHit = rayPosLens + t*rayDirLens;
    const float r2    = pHit.x * pHit.x + pHit.y * pHit.y;
    if (r2 > element.apertureRadius * element.apertureRadius) 
      return false;
    
    rayPosLens = pHit;
    // Update ray path for from-scene element interface interaction
    if (!isStop) 
    {
      float3 wt;
      float etaI = lines[i+0].eta;                                                           // is this correct?
      float etaT = lines[i+1].eta;                                                           // is this correct? 
      if (!Refract(normalize((-1.0f)*rayDirLens), n, etaI / etaT, &wt))
        return false;
      rayDirLens = wt;
    }

  }

  // Transform _rLens_ from lens system space back to camera space
  //
  (*outRayPos) = float3(rayPosLens.x, rayPosLens.y, +rayPosLens.z);
  (*outRayDir) = float3(rayDirLens.x, rayDirLens.y, +rayDirLens.z);
  return false;  
}

void TableLens::RunTestRays()
{
  //float3 rayPos(0,0,1);
  //float3 rayDir(0,0,-1);
  float3 rayPos(3.5e-5f, 0, 0.999999821f);
  float3 rayDir(0,0,-1);
  bool res = TraceLensesFromFilm(rayPos, rayDir, &rayPos, &rayDir);
  int a = 2;
}


void TableLens::MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize)
{
  //#pragma omp parallel for
  for(int i=0;i<in_blockSize;i++)
  {
    const float rndX = hr_qmc::rndFloat(m_globalCounter+i, 0, table[0]);
    const float rndY = hr_qmc::rndFloat(m_globalCounter+i, 1, table[0]);
    
    const float x    = m_fwidth*rndX; 
    const float y    = m_fheight*rndY;

    float3 ray_pos = float3(0,0,0);
    float3 ray_dir = EyeRayDir(x, y, m_fwidth, m_fheight, m_projInv);

    if (!TraceLensesFromFilm(ray_pos, ray_dir, &ray_pos, &ray_dir)) 
    {
      ray_pos = float3(0,10000000.0,0.0); // shoot ray to the sky
      ray_dir = float3(0,1,0);
    }
    else
    {
      int a = 2;
    }

    RayPart1 p1;
    p1.origin[0]   = ray_pos.x;
    p1.origin[1]   = ray_pos.y;
    p1.origin[2]   = ray_pos.z;
    p1.xyPosPacked = myPackXY1616(int(x), int(y));
   
    RayPart2 p2;
    p2.direction[0] = ray_dir.x;
    p2.direction[1] = ray_dir.y;
    p2.direction[2] = ray_dir.z;
    p2.dummy        = 0.0f;
    
    out_rayPosAndNear[i] = p1;
    out_rayDirAndFar [i] = p2;
  }

  //std::this_thread::sleep_for(std::chrono::milliseconds(50)); // test big delay

  m_globalCounter += unsigned(in_blockSize);
} 

void TableLens::AddSamplesContribution(float* out_color4f, const float* colors4f, size_t in_blockSize, uint32_t a_width, uint32_t a_height)
{
  float4*       out_color = (float4*)out_color4f;
  const float4* colors    = (const float4*)colors4f;
  
  for (int i = 0; i < in_blockSize; i++)
  {
    const auto color = colors[i];
    const uint32_t packedIndex = as_int(color.w);
    const int x      = (packedIndex & 0x0000FFFF);         ///<! extract x position from color.w
    const int y      = (packedIndex & 0xFFFF0000) >> 16;   ///<! extract y position from color.w
    const int offset = y*a_width + x;

    if (x >= 0 && y >= 0 && x < a_width && y < a_height)
    {
      out_color[offset].x += color.x;
      out_color[offset].y += color.y;
      out_color[offset].z += color.z;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IHostRaysAPI* MakeHostRaysEmitter(int a_pluginId) ///<! you replace this function or make your own ... the example will be provided
{
  if(a_pluginId == 0)
    return nullptr;

  std::cout << "[MakeHostRaysEmitter]: create plugin #" << a_pluginId << std::endl;
  if(a_pluginId == 2)
    return new TableLens();
  else
    return new SimpleDOF();
}

void DeleteRaysEmitter(IHostRaysAPI* pObject) { delete pObject; }
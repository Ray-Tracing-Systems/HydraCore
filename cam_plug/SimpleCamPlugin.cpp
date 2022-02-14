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
  }

  void ReadParamsFromNode(pugi::xml_node a_camNode);

  void MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize) override;
  void AddSamplesContribution(float* out_color4f, const float* colors4f, size_t in_blockSize, uint32_t a_width, uint32_t a_height) override;

  unsigned int table[hr_qmc::QRNG_DIMENSIONS][hr_qmc::QRNG_RESOLUTION];
  unsigned int m_globalCounter = 0;

  float m_fwidth  = 1024.0f;
  float m_fheight = 1024.0f;
  float4x4 m_projInv;

  struct LensElementInterface {
    float curvatureRadius;
    float thickness;
    float eta;
    float apertureRadius;
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

  lines.resize(0);
  std::vector<int32_t> ids;
  for(auto line : opticalSys.children(L"line"))
  {
    LensElementInterface layer;
    int   id = line.attribute(L"id").as_int();
    layer.curvatureRadius = line.attribute(L"curvature_radius").as_float();
    layer.thickness       = line.attribute(L"thickness").as_float();
    layer.eta             = line.attribute(L"ior").as_float();
    layer.apertureRadius  = 2.0f*line.attribute(L"semi_diameter").as_float();
    lines.push_back(layer);
    ids.push_back(id);
  }

  // you may sort 'lines' by 'ids' if you want 
}

void TableLens::MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize)
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
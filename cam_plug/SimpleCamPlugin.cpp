#include "CamHostPluginAPI.h"
#include <iostream>

#include "../hydra_drv/cglobals.h"

#include "../../HydraAPI/hydra_api/HydraAPI.h"

class SimpleDOF : public IHostRaysAPI
{
public:
  SimpleDOF() { hr_qmc::init(table); m_globalCounter = 0; }
  
  void SetParameters(int a_width, int a_height, const float a_projInvMatrix[16])
  {
    m_fwidth  = float(a_width);
    m_fheight = float(a_height);
    memcpy(&m_projInv, a_projInvMatrix, sizeof(float4x4));
  }

  void MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize) override;
  void AddSamplesContribution(Color4f* out_color, const Color4f* colors, size_t in_blockSize, uint32_t a_width, uint32_t a_height) override;

  unsigned int table[hr_qmc::QRNG_DIMENSIONS][hr_qmc::QRNG_RESOLUTION];
  unsigned int m_globalCounter = 0;

  float m_fwidth  = 1024.0f;
  float m_fheight = 1024.0f;
  float4x4 m_projInv;
};


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

    float3 ray_dir = EyeRayDir(x, y, m_fwidth, m_fheight, m_projInv);

    RayPart1 p1;
    p1.origin[0] = 0.0f;
    p1.origin[1] = 0.0f;
    p1.origin[2] = 0.0f;
    p1.xyPosPacked = myPackXY1616(int(x), int(y));
   
    RayPart2 p2;
    p2.direction[0] = ray_dir.x;
    p2.direction[1] = ray_dir.y;
    p2.direction[2] = ray_dir.z;
    
    out_rayPosAndNear[i] = p1;
    out_rayDirAndFar [i] = p2;
  }

  m_globalCounter += unsigned(in_blockSize);
} 


static inline int my_float_as_int(float x) 
{
  int res; 
  memcpy(&res, &x, sizeof(float)); // modern C++ allow only this way, speed ik ok, check assembly with godbolt
  return res; 
}

void SimpleDOF::AddSamplesContribution(Color4f* out_color, const Color4f* colors, size_t in_blockSize, uint32_t a_width, uint32_t a_height)
{
  for (int i = 0; i < in_blockSize; i++)
  {
    const auto color = colors[i];
    const uint32_t packedIndex = my_float_as_int(color.w);
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


std::shared_ptr<IHostRaysAPI> MakeHostRaysEmitter(int a_pluginId) ///<! you replace this function or make your own ... the example will be provided
{
  if(a_pluginId == 0)
    return nullptr;

  std::cout << "[MakeHostRaysEmitter]: create plugin #" << a_pluginId << std::endl;
  return std::make_shared<SimpleDOF>();
}
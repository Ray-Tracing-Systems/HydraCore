#include "CamHostPluginAPI.h"
#include <iostream>

class SimpleDOF : public IHostRaysAPI
{
public:

  void MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize) override;
  void AddSamplesContribution(Color4f* out_color, const Color4f* colors, size_t in_blockSize, uint32_t a_width, uint32_t a_height) override;
};


void SimpleDOF::MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize)
{

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
  std::cout << "[MakeHostRaysEmitter]: create plugin #" << a_pluginId << std::endl;
  return std::make_shared<SimpleDOF>();
}
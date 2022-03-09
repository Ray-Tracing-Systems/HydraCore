#pragma once 

#include <cstdint>      // uint32_t
#include <cstddef>      // for size_t

struct RayPart1 
{
  float    origin[3];   ///<! ray origin, x,y,z
  uint32_t xyPosPacked; ///<! Packed x and y, on film plane, fixed point; 0x0000 => 0.0; 0xFFFF => 1.0; x stored in less significant bits.
};                      ///<! You may use packXY1616 function from cglobals.h to set this value

struct RayPart2 
{
  float direction[3];
  float dummy;
};

struct IHostRaysAPI
{ 
  virtual ~IHostRaysAPI() {}
  
  /** 
   \brief Set camera parameters
   \param a_width         - image width
   \param a_height        - image height
   \param a_projInvMatrix - inverse projection matrix; you may apply it to the ray to get correct perspective
   \param a_camNodeText   - all other camera parameters which you can directly by reading this node
  */
  virtual void SetParameters(int a_width, int a_height, const float a_projInvMatrix[16], const wchar_t* a_camNodeText) {}

  /** 
   \brief Put portion of rays in execution queue
   \param out_rayPosAndNear - packed ray origin    (x,y,z) and tNear (w)
   \param out_rayDirAndFar  - packed ray direction (x,y,z) and tFar  (w)
   \param in_blockSize      - ray portion size     (may depend on GPU/device, usually == 1024*512)
  
    Please note that it is assumed that rays are uniformly distributed over image plane (and all other integrated dimentions like position on lens) 
    for the whole period of time (all passes), the example will be provided.
  */
  virtual void MakeRaysBlock(RayPart1* out_rayPosAndNear, RayPart2* out_rayDirAndFar, size_t in_blockSize, int passId) = 0;
  
  /**
  \brief Add contribution
  \param out_color  - out float4 image of size a_width*a_height
  \param colors     - in float4 array of size a_size
  \param a_size     - array size
  \param a_width    - image width
  \param a_height   - image height
  */
  virtual void AddSamplesContribution(float* out_color4f, const float* colors4f, size_t in_blockSize, uint32_t a_width, uint32_t a_height, int passId) = 0;
  
  /**
  \brief Signal that rendering is finished and you can save image  
  */
  virtual void FinishRendering() = 0;
};

/** 
  \brief Create camera plugin implementation
  \param a_pluginId - plugin identifier 'cpu_plugin':  <camera id="0" ... cpu_plugin="1">
 */
IHostRaysAPI* MakeHostRaysEmitter(int a_pluginId);       ///<! you replace this function or make your own ... the example will be provided
void          DeleteRaysEmitter(IHostRaysAPI* pObject);  ///<! don't forget to provide delete fuction because your DLL has different heap

static constexpr int HOST_RAYS_PIPELINE_LENGTH = 3;
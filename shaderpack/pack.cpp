#include "../vsgl3/clHelper.h"

#include <vector>
#include <memory.h>
#include <unordered_map>

struct PlatformDevPair
{
  PlatformDevPair(cl_device_id a_dev, cl_platform_id a_platform) : dev(a_dev), platform(a_platform) {}
  
  cl_device_id   dev;
  cl_platform_id platform;
};

std::string cutSpaces(const std::string& a_rhs);
std::vector<PlatformDevPair> listAllOpenCLDevices(const bool a_silendMode = false);

struct CL_GLOBALS
{
  CL_GLOBALS() : ctx(0), cmdQueue(0), platform(0), device(0), m_maxWorkGroupSize(0), oclVer(100), use1DTex(false), liteCore(false),
                 cMortonTable(0), qmcTable(0), devIsCPU(false), cpuTrace(false) {}
  
  cl_context       ctx;                // OpenCL context
  cl_command_queue cmdQueue;           // OpenCL command que
  cl_platform_id   platform;           // OpenCL platform
  cl_device_id     device;             // OpenCL device
  
  cl_mem cMortonTable;
  cl_mem qmcTable;              // this is unrelated to previous. Table for Sobo/Niederreiter quasi random sequence.
  
  size_t m_maxWorkGroupSize;
  
  int  oclVer;
  bool use1DTex;
  bool liteCore;
  
  bool devIsCPU;
  bool cpuTrace;
  
} m_globals;


int main(int argc, const char** argv)
{
  int selectedDeviceId = 0;
  bool SAVE_BUILD_LOG = false;

#ifdef WIN32
  std::string inputfolder = "../../hydra_drv";
#else
  std::string inputfolder = HYDRA_DRV_PATH;
#endif
  
  std::cout << "inputfolder = " << HYDRA_DRV_PATH << std::endl;
  
  #ifdef WIN32
  int initRes = clewInit(L"opencl.dll");
  if (initRes == -1)
  {
    std::cerr << "[cl_core]: failed to load opencl.dll " << std::endl;
    return 0;
  }
  #else
  int initRes = 0;
  #endif
  
  std::vector<PlatformDevPair> devList = listAllOpenCLDevices();
  
  char deviceName[1024];
  
  for (size_t i = 0; i < devList.size(); i++)
  {
    memset(deviceName, 0, 1024);
    CHECK_CL(clGetDeviceInfo(devList[i].dev, CL_DEVICE_NAME, 1024, deviceName, NULL));
    std::string devName2 = cutSpaces(deviceName);
    std::cout << "[cl_core]: device name = " << devName2.c_str() << std::endl;
  }
  
  m_globals.liteCore   = false;
  m_globals.device     = devList[selectedDeviceId].dev;
  m_globals.platform   = devList[selectedDeviceId].platform;
  
  memset(deviceName, 0, 1024);
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_NAME, 1024, deviceName, NULL));
  std::cout << std::endl;
  
  std::string devName2 = cutSpaces(deviceName);
  std::cout << "[cl_core]: using device  : " << devName2.c_str() << std::endl;
  
  // get OpenCL version
  //
  memset(deviceName, 0, 1024);
  clGetPlatformInfo(m_globals.platform, CL_PLATFORM_VERSION, 1024, deviceName, NULL);
  
  std::string versionStr = std::string(deviceName);
  std::cout << "[cl_core]: using cl_ver  : " << m_globals.oclVer << std::endl;
  std::cout << std::endl;
  
  
  std::string specDefines = "";
  
  if (m_globals.liteCore)
    specDefines = " -D BUGGY_INTEL "; // -D ENABLE_OPACITY_TEX  -D SHADOW_TRACE_COLORED_SHADOWS
  else
    specDefines = " -D SHADOW_TRACE_COLORED_SHADOWS -D ENABLE_OPACITY_TEX -D ENABLE_BLINN "; // -D NEXT_BOUNCE_RR
  
  if (!m_globals.devIsCPU && !m_globals.liteCore)
    specDefines += " -D RAYTR_THREAD_COMPACTION ";
  
  
  //
  //
  int clglIsSupported = clglSharingIsSupported(m_globals.device);
  
  cl_int ciErr1;
  
  
  if (ciErr1 != CL_SUCCESS || m_globals.ctx == 0)
  {
    m_globals.ctx = clCreateContext(0, 1, &m_globals.device, NULL, NULL, &ciErr1);
  }
  
  if (ciErr1 != CL_SUCCESS)
    std::cerr <<  "Error in clCreateContext" << std::endl;
  
  m_globals.cmdQueue = clCreateCommandQueue(m_globals.ctx, m_globals.device, 0, &ciErr1);
  
  if (ciErr1 != CL_SUCCESS)
  {
    std::cerr << "[cl_core]: clCreateCommandQueue(1) status = " << ciErr1 << std::endl;
    std::cerr << "Error in clCreateCommandQueue(1)" << std::endl;
  }
  
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &m_globals.m_maxWorkGroupSize, NULL));
  
  std::string sshaderpath  = inputfolder + "/shaders/screen.cl";   // !!!! the hole in security !!!
  std::string tshaderpath  = inputfolder + "/shaders/trace.cl";    // !!!! the hole in security !!!
  std::string soshaderpath = inputfolder + "/shaders/sort.cl";     // !!!! the hole in security !!!
  std::string ishaderpath  = inputfolder + "/shaders/image.cl";    // !!!! the hole in security !!!
  std::string mshaderpath  = inputfolder + "/shaders/mlt.cl";      // !!!! the hole in security !!!
  std::string lshaderpath  = inputfolder + "/shaders/light.cl";    // !!!! the hole in security !!!
  std::string yshaderpath  = inputfolder + "/shaders/material.cl"; // !!!! the hole in security !!!
  
  bool inDevelopment        = false;
  std::string loadEncrypted = "crypt"; // ("crypt", "load", "")
  
  //
  //
  std::string optionsGeneral = "-cl-mad-enable -cl-no-signed-zeros -cl-single-precision-constant -cl-denorms-are-zero "; // -cl-uniform-work-group-size
  std::string optionsInclude = "-I " + inputfolder + " -D OCL_COMPILER ";  // put function that will find shader include folder
  
  if (SAVE_BUILD_LOG)
    optionsGeneral += "-cl-nv-verbose ";
  
  std::string options = optionsGeneral + optionsInclude + specDefines; // + " -cl-nv-maxrregcount=32 ";
  std::cout << "[cl_core]: packing cl programs ..." << std::endl;
  
  //m_progressBar("Compiling shaders", 0.1f);
  std::cout << "[cl_core]: packing " << ishaderpath.c_str() << "    ..." << std::endl;
  CLProgram imagep = CLProgram(m_globals.device, m_globals.ctx, ishaderpath.c_str(), options.c_str(), "", loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << soshaderpath.c_str() << "     ..." << std::endl;
  CLProgram sort   = CLProgram(m_globals.device, m_globals.ctx, soshaderpath.c_str(), options.c_str(), "", loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << mshaderpath.c_str() << "      ... " << std::endl;
  CLProgram mlt    = CLProgram(m_globals.device, m_globals.ctx, mshaderpath.c_str(), options.c_str(), "", loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << sshaderpath.c_str() <<  "   ... " << std::endl;
  CLProgram screen = CLProgram(m_globals.device, m_globals.ctx, sshaderpath.c_str(), options.c_str(), "", loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << tshaderpath.c_str() << "    ..." << std::endl;
  CLProgram trace  = CLProgram(m_globals.device, m_globals.ctx, tshaderpath.c_str(), options.c_str(), "", loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << lshaderpath.c_str() << "    ..." << std::endl;
  CLProgram lightp = CLProgram(m_globals.device, m_globals.ctx, lshaderpath.c_str(), options.c_str(), "", loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << yshaderpath.c_str() << " ..." << std::endl;
  CLProgram material = CLProgram(m_globals.device, m_globals.ctx, yshaderpath.c_str(), options.c_str(), "", loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing cl programs complete" << std::endl << std::endl;
  
  return 0;
}

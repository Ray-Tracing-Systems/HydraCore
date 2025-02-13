#include "GPUOCLLayer.h"
#include "crandom.h"

#include "hydra_api/xxhash.h"
#include "hydra_api/ssemath.h"
#include "hydra_api/HydraInternal.h" // for mkdir

#include "cl_scan_gpu.h"
#include "../cam_plug/CamHostPluginAPI.h"
#include "hydra_api/HydraDLib.h"

const ushort* getGgxTable();
const ushort* getTranspTable();

extern "C" void initQuasirandomGenerator(unsigned int table[QRNG_DIMENSIONS_K][QRNG_RESOLUTION_K]);

#include <algorithm>
#undef min
#undef max

#include <future>
#include <chrono>
#include <iomanip>

constexpr bool SAVE_BUILD_LOG = false;

std::wstring g_internalLibPath = L"";
std::wstring g_internalSateFile = L"";

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUOCLLayer::CL_SCREEN_BUFFERS::free()
{
  if (color0) clReleaseMemObject(color0);
  if (pbo)    clReleaseMemObject(pbo);

  color0 = 0;
  pbo    = 0;
}

void GPUOCLLayer::CL_BUFFERS_RAYS::free()
{
  if(rayPos)   { clReleaseMemObject(rayPos);   rayPos   = nullptr; }
  if(rayDir)   { clReleaseMemObject(rayDir);   rayDir   = nullptr; }
  if(hits)     { clReleaseMemObject(hits);     hits     = nullptr; }
  if(rayFlags) { clReleaseMemObject(rayFlags); rayFlags = nullptr; }
  if(surfId)   { clReleaseMemObject(surfId);    surfId  = nullptr; }
                                                                                                      
  if (hitSurfaceAll)   { clReleaseMemObject(hitSurfaceAll);  hitSurfaceAll    = nullptr; }
  if (hitProcTexData)  { clReleaseMemObject(hitProcTexData); hitProcTexData   = nullptr;}

  if (pathThoroughput) { clReleaseMemObject(pathThoroughput); pathThoroughput = nullptr; }
  if (pathMisDataPrev) { clReleaseMemObject(pathMisDataPrev); pathMisDataPrev = nullptr; }
  if (pathShadeColor)  { clReleaseMemObject(pathShadeColor);  pathShadeColor  = nullptr; }
                      
  if (pathAccColor)    { clReleaseMemObject(pathAccColor);    pathAccColor = nullptr;    }
  if (pathAuxColor)    { clReleaseMemObject(pathAuxColor);    pathAuxColor = nullptr;    }
  if (randGenState)    { clReleaseMemObject(randGenState);    randGenState = nullptr;    }

  if (pathShadow8B)       { clReleaseMemObject(pathShadow8B);       pathShadow8B       = nullptr; }
  if (pathShadow8BAux)    { clReleaseMemObject(pathShadow8BAux);    pathShadow8BAux    = nullptr; }
  if (pathShadow8BAuxCPU) { clReleaseMemObject(pathShadow8BAuxCPU); pathShadow8BAuxCPU = nullptr; }

  if (pathAuxColorCPU) { clReleaseMemObject(pathAuxColorCPU); pathAuxColorCPU = nullptr; }

  if (lsamRev)         { clReleaseMemObject(lsamRev);  lsamRev  = nullptr; }
  if (lshadow)         { clReleaseMemObject(lshadow);  lshadow  = nullptr; }
  if (shadowTemp1i)    { clReleaseMemObject(shadowTemp1i);  shadowTemp1i = nullptr; }

  if (shadowRayPos)    { clReleaseMemObject(shadowRayPos); shadowRayPos = nullptr; }
  if (shadowRayDir)    { clReleaseMemObject(shadowRayDir); shadowRayDir = nullptr; }
  if (accPdf)          { clReleaseMemObject(accPdf);       accPdf       = nullptr; }

  if(oldFlags)         { clReleaseMemObject(oldFlags);  oldFlags  = nullptr; }
  if(oldRayDir)        { clReleaseMemObject(oldRayDir); oldRayDir = nullptr; }
  if(oldColor)         { clReleaseMemObject(oldColor);  oldColor  = nullptr; }
                       
  if (fogAtten)        { clReleaseMemObject(fogAtten);   fogAtten   = nullptr; }
  if (samZindex)       { clReleaseMemObject(samZindex);  samZindex  = nullptr; }
  if (aoCompressed)    { clReleaseMemObject(aoCompressed);   aoCompressed  = nullptr; }
  if (aoCompressed2)   { clReleaseMemObject(aoCompressed2);  aoCompressed2 = nullptr; }
  if (lightOffsetBuff) { clReleaseMemObject(lightOffsetBuff);  lightOffsetBuff = nullptr; }
  if (packedXY)        { clReleaseMemObject(packedXY);   packedXY   = nullptr; }
  if (debugf4)         { clReleaseMemObject(debugf4);    debugf4    = nullptr; }

  if(atomicCounterMem) { clReleaseMemObject(atomicCounterMem); atomicCounterMem = nullptr;}
}

size_t GPUOCLLayer::CL_BUFFERS_RAYS::resize(cl_context ctx, cl_command_queue cmdQueue, size_t a_size, bool a_cpuShare, bool a_cpuFB, bool a_evalSurfaId)
{
  free();

  cl_mem_flags shareFlags = a_cpuShare ? CL_MEM_ALLOC_HOST_PTR : 0;

  size_t buff1Size = sizeof(cl_float)*a_size;
  size_t currSize  = 0;

  cl_int ciErr1;

  MEGABLOCKSIZE = a_size; // /2
  rayPos   = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 4;
  rayDir   = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4*sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  hits     = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, sizeof(Lite_Hit)*MEGABLOCKSIZE,   NULL, &ciErr1);   currSize += buff1Size * 1;
  rayFlags = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, sizeof(uint)*MEGABLOCKSIZE, NULL, &ciErr1);         currSize += buff1Size * 1;
  if(a_evalSurfaId) {
    surfId = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint)*MEGABLOCKSIZE, NULL, &ciErr1);         
    currSize += buff1Size * 1;
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  const size_t sizeOfHit = SURFACE_HIT_SIZE_IN_F4*sizeof(float4)*MEGABLOCKSIZE;
  hitSurfaceAll          = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeOfHit, NULL, &ciErr1);                            currSize += sizeOfHit;
  hitProcTexData         = nullptr;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  pathThoroughput = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);          currSize += buff1Size * 4;
  pathMisDataPrev = clCreateBuffer(ctx, CL_MEM_READ_WRITE,     sizeof(MisData) *MEGABLOCKSIZE, NULL, &ciErr1);          currSize += buff1Size * 1;
  pathShadeColor  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);          currSize += buff1Size * 4;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  pathAccColor = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);  currSize += buff1Size * 4;
  randGenState = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += (sizeof(RandomGen)*MEGABLOCKSIZE);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  lsamRev      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 12 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);  currSize += buff1Size * 12;

  shadowRayPos = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  shadowRayDir = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  accPdf       = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 1 * sizeof(PerRayAcc)*MEGABLOCKSIZE, NULL, &ciErr1);  currSize += buff1Size * 1;
  shadowTemp1i = clCreateBuffer(ctx, CL_MEM_READ_WRITE,              1 * sizeof(PerRayAcc)*MEGABLOCKSIZE, NULL, &ciErr1);  currSize += buff1Size * 1;
                                                                                                                           
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  oldFlags      = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 1 * sizeof(uint)*MEGABLOCKSIZE,     NULL, &ciErr1);   currSize += buff1Size * 1;
  oldRayDir     = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  oldColor      = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  lshadow  = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_ushort)*MEGABLOCKSIZE, NULL, &ciErr1);       currSize += buff1Size * 4;
  fogAtten = clCreateBuffer(ctx, CL_MEM_READ_WRITE,              4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);        currSize += buff1Size * 4;


  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  if (a_cpuFB || FORCE_DRAW_SHADOW)
  {
    pathAuxColor       = clCreateBuffer(ctx, CL_MEM_READ_WRITE,                         4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 4;
    pathAuxColorCPU    = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);  

    pathShadow8B       = clCreateBuffer(ctx, CL_MEM_READ_WRITE,                         1 * sizeof(cl_uint8)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += (a_size * sizeof(cl_uint8));
    pathShadow8BAux    = clCreateBuffer(ctx, CL_MEM_READ_WRITE,                         1 * sizeof(cl_uint8)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += (a_size * sizeof(cl_uint8));
    pathShadow8BAuxCPU = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 1 * sizeof(cl_uint8)*MEGABLOCKSIZE, NULL, &ciErr1); 
  }
  else
  {
    pathAuxColor       = nullptr;
    pathAuxColorCPU    = nullptr;
    pathShadow8B       = nullptr;
    pathShadow8BAux    = nullptr;
    pathShadow8BAuxCPU = nullptr;
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  samZindex       = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*sizeof(int)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 2;
  packedXY        = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1*sizeof(int)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 1;
  lightOffsetBuff = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1*sizeof(int)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 1;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");
 
  std::cout << "[cl_core]: MEGABLOCK SIZE = " << MEGABLOCKSIZE << std::endl;

  atomicCounterMem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1*sizeof(int), NULL, &ciErr1);
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("can't alloc atomic counter memory");
 

  return currSize;
}

void GPUOCLLayer::CL_SCENE_DATA::free()
{
  for (int i = 0; i < bvhNumber; i++)
  {
    if (bvhBuff[i])        clReleaseMemObject(bvhBuff[i]);
    if (objListBuff[i])    clReleaseMemObject(objListBuff[i]);
    if (alphTstBuff[i])    clReleaseMemObject(alphTstBuff[i]);

    bvhBuff    [i] = nullptr;
    objListBuff[i] = nullptr;
    alphTstBuff[i] = nullptr;
  }
  bvhNumber = 0;

  if(matrices)       clReleaseMemObject(matrices);
  if(instLightInst)  clReleaseMemObject(instLightInst);

  matrices          = nullptr;
  instLightInst     = nullptr;

  matricesSize      = 0;
  instLightInstSize = 0;

  if (remapLists != nullptr) { clReleaseMemObject(remapLists); remapLists = nullptr; }
  if (remapTable != nullptr) { clReleaseMemObject(remapTable); remapTable = nullptr; }
  if (remapInst  != nullptr) { clReleaseMemObject(remapInst);  remapInst  = nullptr;  }
  remapListsSize = 0;
  remapTableSize = 0;
  remapInstSize  = 0;

  for (auto p : namedBuffers)
  {
    if (p.second != nullptr)
      clReleaseMemObject(p.second);
    p.second = nullptr;
  }
  namedBuffers.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct PlatformDevPair
{
  PlatformDevPair(cl_device_id a_dev, cl_platform_id a_platform) : dev(a_dev), platform(a_platform) {}

  cl_device_id   dev;
  cl_platform_id platform;
};

std::string cutSpaces(const std::string& a_rhs)
{
  int pos = 0;
  for (int i = 0; i < a_rhs.size(); i++)
  {
    if (a_rhs[i] != ' ')
      break;
    pos++;
  }

  return a_rhs.substr(pos, a_rhs.size() - pos);
}

std::vector<PlatformDevPair> listAllOpenCLDevices(const bool a_silendMode = false)
{
  const int MAXPLATFORMS            = 4;
  const int MAXDEVICES_PER_PLATFORM = 8;

  cl_platform_id platforms[MAXPLATFORMS];
  cl_device_id   devices[MAXDEVICES_PER_PLATFORM];

  cl_uint factPlatroms = 0;
  cl_uint factDevs = 0;

  std::vector<PlatformDevPair> result;

  CHECK_CL(clGetPlatformIDs(MAXPLATFORMS, platforms, &factPlatroms));

  if (!a_silendMode)
  {
    char temp[1024];
    std::cout << std::endl;
    for (size_t i = 0; i < factPlatroms; i++)
    {
      std::cout << "platform " << i << " : " << std::endl;

      memset(temp, 0, 1024);
      clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 1024, temp, NULL);
      std::cout << "CL_PLATFORM_PROFILE = " << temp << std::endl;

      memset(temp, 0, 1024);
      clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 1024, temp, NULL);
      std::cout << "CL_PLATFORM_VERSION = " << temp << std::endl;

      memset(temp, 0, 1024);
      clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, temp, NULL);
      std::cout << "CL_PLATFORM_NAME    = " << temp << std::endl;

      memset(temp, 0, 1024);
      clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 1024, temp, NULL);
      std::cout << "CL_PLATFORM_VENDOR  = " << temp << std::endl;

      std::cout << std::endl;
    }
  }

  for (size_t i = 0; i < factPlatroms; i++)
  {
    CHECK_CL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAXDEVICES_PER_PLATFORM, devices, &factDevs));
    for (cl_uint j = 0; j < factDevs; j++)
      result.push_back(PlatformDevPair(devices[j], platforms[i]));
  }

  return result;

}

std::string deviceHash(cl_device_id a_devId, cl_platform_id a_platform)
{
  char deviceName[1024];
  memset(deviceName, 0, 1024);

  CHECK_CL(clGetDeviceInfo(a_devId, CL_DEVICE_NAME, 1024, deviceName, NULL));
  size_t len = strnlen(deviceName, 1024);

  uint64_t hashVal = XXH64(deviceName, len, uint64_t(8589934592) + uint64_t(2737126414));

  memset(deviceName, 0, 1024);
  CHECK_CL(clGetPlatformInfo(a_platform, CL_PLATFORM_VERSION, 1024, deviceName, NULL));
  len = strnlen(deviceName, 1024);

  hashVal = XXH64(deviceName, len, hashVal);
  
  memset(deviceName, 0, 1024);
  snprintf(deviceName, 1024, "%llu", (long long unsigned int)hashVal);
  return std::string(deviceName);
}

bool deviceIsCPU(cl_device_id a_devId)
{
  cl_device_type devType = 0;
  CHECK_CL(clGetDeviceInfo(a_devId, CL_DEVICE_TYPE, sizeof(cl_device_type), &devType, NULL));
  return (devType == CL_DEVICE_TYPE_CPU);
}

//bool deviceSupportMemSharing(cl_device_id a_devId)
//{
  //cl_device_svm_capabilities caps;
  //CHECK_CL(clGetDeviceInfo(a_devId, CL_DEVICE_SVM_CAPABILITIES,sizeof(cl_device_svm_capabilities),&caps,0));
  //return true;
//}

void PrintCLDevicesListToFile(const char* a_fileName, const std::vector<PlatformDevPair>& devList)
{
  std::ofstream deviceListFile(HydraInstallPath() + "logs\\devlist.txt");

  char deviceName[1024];

  for (size_t i = 0; i < devList.size(); i++)
  {
    memset(deviceName, 0, 1024);
    CHECK_CL(clGetDeviceInfo(devList[i].dev, CL_DEVICE_NAME, 1024, deviceName, NULL));

    cl_device_type devType = CL_DEVICE_TYPE_GPU;
    CHECK_CL(clGetDeviceInfo(devList[i].dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &devType, NULL));

    std::string devName2 = cutSpaces(deviceName);

    deviceListFile << i << "; " << devName2.c_str() << "; ";

    memset(deviceName, 0, 1024);
    clGetPlatformInfo(devList[i].platform, CL_PLATFORM_VERSION, 1024, deviceName, NULL);
    deviceListFile << deviceName << "; ";
    
    if (devType == CL_DEVICE_TYPE_CPU)
      deviceListFile << "CPU;";
    else
      deviceListFile << "GPU;";

    deviceListFile << std::endl;
  }

  deviceListFile << devList.size() << "; Hydra CPU; Pure C/C++; CPU;" << std::endl;
  deviceListFile.flush();
  deviceListFile.close();
  return;
}

std::vector<HRRenderDeviceInfoListElem> g_deviceList;
std::wstring s2ws(const std::string& s);

const HRRenderDeviceInfoListElem* GPUOCLLayer::ListDevices() const
{
  auto devList = listAllOpenCLDevices(true);
  g_deviceList.resize(devList.size()+1);

  char deviceName[1024];

  for (size_t i = 0; i < devList.size(); i++)
  {
    g_deviceList[i].id = int32_t(i);
    memset(deviceName, 0, 1024);
    CHECK_CL(clGetDeviceInfo(devList[i].dev, CL_DEVICE_NAME, 1024, deviceName, NULL));
    std::wstring devNameW = s2ws(deviceName);
    wcsncpy(g_deviceList[i].name, devNameW.c_str(), 256);
    
    memset(deviceName, 0, 1024);
    CHECK_CL(clGetPlatformInfo(devList[i].platform, CL_PLATFORM_VERSION, 1024, deviceName, NULL));
    std::wstring driverName = s2ws(deviceName);
    wcsncpy(g_deviceList[i].driver, driverName.c_str(), 256);

    cl_device_type devType = CL_DEVICE_TYPE_GPU;
    CHECK_CL(clGetDeviceInfo(devList[i].dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &devType, NULL));
    g_deviceList[i].isCPU = (devType == CL_DEVICE_TYPE_CPU);

    g_deviceList[i].isEnabled = false;
    g_deviceList[i].next      = &g_deviceList[i + 1];
  }

  size_t last = g_deviceList.size() - 1;
  g_deviceList[last].id = -1;
  wcsncpy(g_deviceList[last].name,   L"Hydra CPU", 256);
  wcsncpy(g_deviceList[last].driver, L"Pure C/C++", 256);
  g_deviceList[last].isCPU     = true;
  g_deviceList[last].isEnabled = false;
  g_deviceList[last].next      = nullptr;

  if (g_deviceList.size() == 0)
    return nullptr;
  else
    return &g_deviceList[0];
}

//void TestPathVertexReadWrite();
const ushort* getGgxTable();
const ushort* getTranspTable();

GPUOCLLayer::GPUOCLLayer(int w, int h, int a_flags, int a_deviceId) : Base(w, h, a_flags)
{ 
  //TestPathVertexReadWrite();

  m_initFlags = a_flags;
  for (int i = 0; i < MEM_TAKEN_OBJECTS_NUM; i++)
    m_memoryTaken[i] = 0;
  
  InitEngineGlobals(&m_globsBuffHeader, getGgxTable(), getTranspTable());
  
  #ifdef WIN32

  // enforce UNICODE define (for some reason cmake compile definitions don't work)
  // but we use wchar_t* name L"opencl.dll" anyway -> need unicode library loader
#ifndef UNICODE
#define UNICODE
#endif

  int initRes = clewInit(L"opencl.dll");
  #else
  int initRes = 0;
  #endif
  
  if (initRes == -1)
  {
    std::cerr << "[cl_core]: failed to load opencl.dll " << std::endl;

    if (a_flags & GPU_RT_HW_LIST_OCL_DEVICES)
      PrintCLDevicesListToFile((HydraInstallPath() + "logs\\devlist.txt").c_str(), std::vector<PlatformDevPair>());

    exit(0);
  }

  //CHECK_CL(clGetPlatformIDs(1, &m_globals.platform, NULL));
  //CHECK_CL(clGetDeviceIDs(m_globals.platform, CL_DEVICE_TYPE_GPU, 1, &m_globals.device, NULL));

  std::vector<PlatformDevPair> devList = listAllOpenCLDevices();

  char deviceName[1024];

  for (size_t i = 0; i < devList.size(); i++)
  {
    memset(deviceName, 0, 1024);
    CHECK_CL(clGetDeviceInfo(devList[i].dev, CL_DEVICE_NAME, 1024, deviceName, NULL));
    std::string devName2 = cutSpaces(deviceName);
    std::cout << "[cl_core]: device name = " << devName2.c_str() << std::endl;
  }


  if (a_flags & GPU_RT_HW_LIST_OCL_DEVICES)
  {
    PrintCLDevicesListToFile((HydraInstallPath() + "logs\\devlist.txt").c_str(), devList);
    return;
  }
  
  m_globals.liteCore = ((a_flags & GPU_RT_LITE_CORE) != 0);

  //m_globals.liteCore   = true;
  bool forceNoGLSharing = true;

  if (m_globals.liteCore)
    std::cout << "[cl_core]: using lite core "<< std::endl;

  int selectedDeviceId = a_deviceId;

  if (selectedDeviceId >= devList.size())
  {
    std::cerr << "[cl_core]: CRITICAL ERROR! No device with id = " << selectedDeviceId << " have found! " << std::endl;
    std::cerr.flush();
    exit(0);
  }

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

  if (versionStr.find("OpenCL 2.0") != std::string::npos)
    m_globals.oclVer = 200;
  else if (versionStr.find("OpenCL 1.2") != std::string::npos)
    m_globals.oclVer = 120;
  else if (versionStr.find("OpenCL 1.1") != std::string::npos)
    m_globals.oclVer = 110;
  else
    m_globals.oclVer = 100;

  std::cout << "[cl_core]: using cl_ver  : " << m_globals.oclVer << std::endl;
  std::cout << std::endl;

  size_t   paramValueSize = 0;
  cl_ulong maxBufferSize  = 0;
  cl_ulong memTotal       = 0;
  cl_ulong maxShmemSize   = 0;
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxBufferSize, NULL));
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memTotal, &paramValueSize));
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &maxShmemSize, NULL));
  
  
  std::cout << "[cl_core]: totalMemSize  = " << memTotal      / cl_ulong(1024*1024) << "\tMB" << std::endl;
  std::cout << "[cl_core]: maxBufferSize = " << maxBufferSize / cl_ulong(1024*1024) << "\tMB" << std::endl;
  std::cout << "[cl_core]: maxSharedSize = " << maxShmemSize  / cl_ulong(1024)      << "\tKB" << std::endl;

  m_globals.devIsCPU = deviceIsCPU(m_globals.device);
  m_globals.cpuTrace = false; // m_globals.devIsCPU && !(a_flags & CPU_RT_PURE_CL);

  if (m_globals.devIsCPU)
    std::cout << "[cl_core]: device is a CPU " << std::endl;
  else
    std::cout << "[cl_core]: device is a GPU " << std::endl;

  if (m_globals.cpuTrace)
    std::cout << "[cl_core]: use CPU trace " << std::endl;
  else
    std::cout << "[cl_core]: use OCL trace " << std::endl;

  //
  //
  m_clglSharing = 0;
  int clglIsSupported = clglSharingIsSupported(m_globals.device);

  cl_int ciErr1 = CL_SUCCESS;
 
  if ((a_flags & GPU_RT_NOWINDOW) || (clglIsSupported == 0))
  {
    m_globals.ctx = clCreateContext(0, 1, &m_globals.device, NULL, NULL, &ciErr1);
    m_clglSharing = false;
  }
  else if(!forceNoGLSharing)
  {
    #ifdef WIN32
    cl_context_properties properties[] = {
      CL_GL_CONTEXT_KHR,   (cl_context_properties)wglGetCurrentContext(), // WGL Context
      CL_WGL_HDC_KHR,      (cl_context_properties)wglGetCurrentDC(),      // WGL HDC
      CL_CONTEXT_PLATFORM, (cl_context_properties)m_globals.platform,     // OpenCL platform
      0
    };

    m_globals.ctx = clCreateContext(properties, 1, &m_globals.device, NULL, NULL, &ciErr1);
    m_clglSharing = true;
    #else
    ciErr1        = -1;
    m_clglSharing = false;
    #endif
  }

  if (ciErr1 != CL_SUCCESS || m_globals.ctx == 0)
  {
    m_globals.ctx = clCreateContext(0, 1, &m_globals.device, NULL, NULL, &ciErr1);
    m_clglSharing = false;
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in clCreateContext");

  m_globals.cmdQueue = clCreateCommandQueue(m_globals.ctx, m_globals.device, 0, &ciErr1); 
  if (ciErr1 != CL_SUCCESS)
  {
    std::cerr << "[cl_core]: clCreateCommandQueue(1) status = " << ciErr1 << std::endl;
    RUN_TIME_ERROR("Error in clCreateCommandQueue(1)");
  }

  m_globals.cmdQueueDevToHost = clCreateCommandQueue(m_globals.ctx, m_globals.device, 0, &ciErr1);
  if (ciErr1 != CL_SUCCESS)
  {
    std::cerr << "[cl_core]: clCreateCommandQueue(2) status = " << ciErr1 << std::endl; 
    RUN_TIME_ERROR("Error in clCreateCommandQueue(2)");
  }

  m_globals.cmdQueueHostToDev = clCreateCommandQueue(m_globals.ctx, m_globals.device, 0, &ciErr1);
  if (ciErr1 != CL_SUCCESS)
  {
    std::cerr << "[cl_core]: clCreateCommandQueue(3) status = " << ciErr1 << std::endl; 
    RUN_TIME_ERROR("Error in clCreateCommandQueue(3)");
  }

  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &m_globals.m_maxWorkGroupSize, NULL));

  std::string sshaderpath  = "../hydra_drv/shaders/screen.cl";   // !!!! the hole in security !!!
  std::string tshaderpath  = "../hydra_drv/shaders/trace.cl";    // !!!! the hole in security !!!
  std::string soshaderpath = "../hydra_drv/shaders/sort.cl";     // !!!! the hole in security !!!
  std::string ishaderpath  = "../hydra_drv/shaders/image.cl";    // !!!! the hole in security !!!
  std::string mshaderpath  = "../hydra_drv/shaders/mlt.cl";      // !!!! the hole in security !!!
  std::string lshaderpath  = "../hydra_drv/shaders/light.cl";    // !!!! the hole in security !!!
  std::string yshaderpath  = "../hydra_drv/shaders/material.cl"; // !!!! the hole in security !!!

  const std::string installPath2 = HydraInstallPath();
  
  if (!isFileExists(sshaderpath))  sshaderpath  = installPath2 + "shaders/screen.cl";
  if (!isFileExists(tshaderpath))  tshaderpath  = installPath2 + "shaders/trace.cl";
  if (!isFileExists(soshaderpath)) soshaderpath = installPath2 + "shaders/sort.cl";
  if (!isFileExists(ishaderpath))  ishaderpath  = installPath2 + "shaders/image.cl";
  if (!isFileExists(mshaderpath))  mshaderpath  = installPath2 + "shaders/mlt.cl";
  if (!isFileExists(lshaderpath))  lshaderpath  = installPath2 + "shaders/light.cl";
  if (!isFileExists(yshaderpath))  yshaderpath  = installPath2 + "shaders/material.cl";

  std::string devHash = deviceHash(m_globals.device, m_globals.platform);

  if (m_globals.liteCore)
    devHash += "1";
  else
    devHash += "0";

  std::string shaderpath = installPath2 + "shadercache/";
  hr_mkdir(shaderpath.c_str());

  std::string sshaderpathBin  = shaderpath + "screen_" + devHash + ".bin";
  std::string tshaderpathBin  = shaderpath + "tracex_" + devHash + ".bin";
  std::string tshaderpathBin2 = shaderpath + "trace2_" + devHash + ".bin";
  std::string soshaderpathBin = shaderpath + "sortxx_" + devHash + ".bin";
  std::string ioshaderpathBin = shaderpath + "imagex_" + devHash + ".bin";
  std::string moshaderpathBin = shaderpath + "mltxxx_" + devHash + ".bin";
  std::string loshaderpathBin = shaderpath + "lightx_" + devHash + ".bin";
  std::string yoshaderpathBin = shaderpath + "matsxx_" + devHash + ".bin";

  bool inDevelopment = (a_flags & GPU_RT_IN_DEVELOPMENT);
  std::string loadEncrypted = "load"; // ("crypt", "load", "")
  if (inDevelopment)
    loadEncrypted = "";

  if ((a_flags & GPU_RT_CLEAR_SHADER_CACHE) || inDevelopment)
  {
    std::remove(sshaderpathBin.c_str());
    std::remove(tshaderpathBin.c_str());
    std::remove(tshaderpathBin2.c_str());
    std::remove(soshaderpathBin.c_str());
    std::remove(ioshaderpathBin.c_str());
    std::remove(moshaderpathBin.c_str());
    std::remove(loshaderpathBin.c_str());
    std::remove(yoshaderpathBin.c_str());
  }

  const bool doublesForTriIntersection = ((a_flags & GPU_RT_DOUBLE_FOR_TRIS) != 0);

  std::string options = GetOCLShaderCompilerOptions();
  std::cout << "[cl_core]: building cl programs ..." << std::endl;

  //m_progressBar("Compiling shaders", 0.1f);
  std::cout << "[cl_core]: building " << ishaderpath.c_str() << "    ..." << std::endl;
  m_progs.imagep = CLProgram(m_globals.device, m_globals.ctx, ishaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, ioshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << soshaderpath.c_str() << "     ..." << std::endl;
  m_progs.sort   = CLProgram(m_globals.device, m_globals.ctx, soshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, soshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << sshaderpath.c_str() <<  "   ... " << std::endl;
  m_progs.screen = CLProgram(m_globals.device, m_globals.ctx, sshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, sshaderpathBin, SAVE_BUILD_LOG);
  
  if(doublesForTriIntersection)
  {
    std::string options2 = options + " -D DOUBLE_RAY_TRIANGLE";
    std::cout << "[cl_core]: building " << tshaderpath.c_str()     << " (double) " << std::endl;
    std::cout << "[cl_core]: shdr_bin " << tshaderpathBin2.c_str() << " (double) " << std::endl;    
    m_progs.trace  = CLProgram(m_globals.device, m_globals.ctx, tshaderpath.c_str(), options2.c_str(), HydraInstallPath(), loadEncrypted, tshaderpathBin2, SAVE_BUILD_LOG);
  }
  else
  {
    std::cout << "[cl_core]: building " << tshaderpath.c_str()    << " (float)" << std::endl;
    std::cout << "[cl_core]: shdr_bin " << tshaderpathBin.c_str() << " (float) " << std::endl;    
    m_progs.trace  = CLProgram(m_globals.device, m_globals.ctx, tshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, tshaderpathBin, SAVE_BUILD_LOG);
  }

  std::cout << "[cl_core]: building " << lshaderpath.c_str() << "    ..." << std::endl;
  m_progs.lightp = CLProgram(m_globals.device, m_globals.ctx, lshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, loshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << yshaderpath.c_str() << " ..." << std::endl;
  m_progs.material = CLProgram(m_globals.device, m_globals.ctx, yshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, yoshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << mshaderpath.c_str() << "      ... " << std::endl;
  m_progs.mlt = CLProgram(m_globals.device, m_globals.ctx, mshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, moshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: build cl programs complete" << std::endl << std::endl;

  if (!inDevelopment)
  {
    if (!isFileExists(ioshaderpathBin))
      m_progs.imagep.saveBinary(ioshaderpathBin);

    if (!isFileExists(soshaderpathBin))
      m_progs.sort.saveBinary(soshaderpathBin);

    if (!isFileExists(moshaderpathBin))
      m_progs.mlt.saveBinary(moshaderpathBin);

    if (!isFileExists(sshaderpathBin))
      m_progs.screen.saveBinary(sshaderpathBin);
    
    if(doublesForTriIntersection) 
    {
      if(!isFileExists(tshaderpathBin2))
        m_progs.trace.saveBinary(tshaderpathBin2);
    }
    else
    {
      if(!isFileExists(tshaderpathBin))
        m_progs.trace.saveBinary(tshaderpathBin);
    }

    if (!isFileExists(loshaderpathBin))
      m_progs.lightp.saveBinary(loshaderpathBin);

    if (!isFileExists(yoshaderpathBin))
      m_progs.material.saveBinary(yoshaderpathBin);
  }

  // create morton table
  //
  m_globals.cMortonTable = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(MortonTable256Host), MortonTable256Host, &ciErr1);   if (ciErr1 != CL_SUCCESS) RUN_TIME_ERROR("Error in clCreateBuffer");

  // create qmc table for Niederreiter sequence
  //
  unsigned int tableCPU[QRNG_DIMENSIONS_K][QRNG_RESOLUTION_K];
  initQuasirandomGenerator(tableCPU);
  m_globals.qmcTable = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, QRNG_DIMENSIONS_K * QRNG_RESOLUTION_K * sizeof(unsigned int), &tableCPU, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error when create qmcTable");

  float2 qmc[GBUFFER_SAMPLES];
  float2 qmc2[PMPIX_SAMPLES];
  
  PlaneHammersley(&qmc[0].x, GBUFFER_SAMPLES);
  PlaneHammersley(&qmc2[0].x, PMPIX_SAMPLES);

  m_globals.hammersley2DGBuff = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qmc),  qmc,  &ciErr1);
  m_globals.hammersley2D256   = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qmc2), qmc2, &ciErr1);

  waitIfDebug(__FILE__, __LINE__);

  m_spp               = 0.0f;
  m_sppDL             = 0.0f;
  m_sppDone           = 0.0f;
  m_sppContrib        = 0.0f;
  m_avgBrightness     = 1.0f;
  m_tablesBeenUpdated = false;
  
  //if(!scan_alloc_internal(65536*16, m_globals.ctx))
  //  RUN_TIME_ERROR("Error in scan_alloc_internal");
  //testScanFloatsAnySize();
}

void GPUOCLLayer::RecompileProcTexShaders(const std::string& a_shaderPath)
{
  std::string options = GetOCLShaderCompilerOptions();
 
  #ifndef RECOMPILE_PROCTEX_FROM_STRING
  std::cout << "[cl_core]: recompile " << a_shaderPath.c_str() << " ..." << std::endl;
  m_progs.texproc = CLProgram(m_globals.device, m_globals.ctx, a_shaderPath.c_str(), options, HydraInstallPath(), "", "", SAVE_BUILD_LOG);
  #else 
  std::cout << "[cl_core]: recompile from string ..." << std::endl;
  m_progs.texproc = CLProgram(m_globals.device, m_globals.ctx, a_shaderPath, options, HydraInstallPath(), nullptr);
  #endif

  if(m_rays.hitProcTexData != nullptr)
    clReleaseMemObject(m_rays.hitProcTexData);

  cl_int ciErr1 = 0;
  m_rays.hitProcTexData = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, F4_PROCTEX_SIZE*sizeof(float4)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  //currSize += (F4_PROCTEX_SIZE*sizeof(float4)*m_rays.MEGABLOCKSIZE); //#TODO: ACCOUNT THIS MEM FOR MEM INFO

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in RecompileProcTexShaders -> clCreateBuffer for proc tex");
}

void GPUOCLLayer::FinishAll()
{
  if (m_globals.cmdQueue == 0)
    return;

  CHECK_CL(clFinish(m_globals.cmdQueue));
}

GPUOCLLayer::~GPUOCLLayer()
{
  FinishAll();
  
  m_camPlugin.free();
  MLT_Free();
  kmlt.free();
  m_rays.free();
  m_screen.free();
  m_scene.free();
  
  if (m_globals.cMortonTable)     { clReleaseMemObject(m_globals.cMortonTable);      m_globals.cMortonTable      = nullptr; }
  if (m_globals.qmcTable)         { clReleaseMemObject(m_globals.qmcTable);          m_globals.qmcTable          = nullptr; }
  if (m_globals.hammersley2DGBuff){ clReleaseMemObject(m_globals.hammersley2DGBuff); m_globals.hammersley2DGBuff = nullptr; }
  if (m_globals.hammersley2D256)  { clReleaseMemObject(m_globals.hammersley2D256);   m_globals.hammersley2D256   = nullptr; }

  if(m_globals.cmdQueue)          { clReleaseCommandQueue(m_globals.cmdQueue);          m_globals.cmdQueue          = nullptr; }
  if(m_globals.cmdQueueDevToHost) { clReleaseCommandQueue(m_globals.cmdQueueDevToHost); m_globals.cmdQueueDevToHost = nullptr; }
  if(m_globals.cmdQueueHostToDev) { clReleaseCommandQueue(m_globals.cmdQueueHostToDev); m_globals.cmdQueueHostToDev = nullptr; }
  if(m_globals.ctx)               { clReleaseContext     (m_globals.ctx);               m_globals.ctx               = nullptr; }
}

size_t GPUOCLLayer::CalcMegaBlockSize(int a_flags)
{
  const size_t memAmount = GetAvaliableMemoryAmount(true);
  const size_t MB = size_t(1024*1024);

  int MEGABLOCK_SIZE = 1024 * 512;

  if (m_globals.devIsCPU)
    MEGABLOCK_SIZE = 256 * 256;
  else if (memAmount <= size_t(256)*MB)
    MEGABLOCK_SIZE = 256 * 256;
  else if (memAmount <= size_t(1024)*MB)
    MEGABLOCK_SIZE = 512 * 512;
  else if (memAmount <= size_t(4*1024)*MB)
    MEGABLOCK_SIZE = 1024 * 512;
  else
    MEGABLOCK_SIZE = 1024 * 1024;

  if (a_flags & GPU_MLT_ENABLED_AT_START)
  {
    MEGABLOCK_SIZE = 524288;
    if (a_flags & GPU_MMLT_THREADS_262K)
      MEGABLOCK_SIZE = 262144;
    else if (a_flags & GPU_MMLT_THREADS_131K)
      MEGABLOCK_SIZE = 131072;
    else if (a_flags & GPU_MMLT_THREADS_65K)
      MEGABLOCK_SIZE = 65536;
    else if (a_flags & GPU_MMLT_THREADS_16K)
      MEGABLOCK_SIZE = 16384;

    if (m_globals.devIsCPU)
      MEGABLOCK_SIZE = 16384;
  }

  return MEGABLOCK_SIZE;
}

std::string GPUOCLLayer::GetOCLShaderCompilerOptions()
{
  std::string specDefines = "";

  if (m_globals.liteCore)
    specDefines = " -D BUGGY_INTEL "; // -D ENABLE_OPACITY_TEX  -D SHADOW_TRACE_COLORED_SHADOWS
  else
    specDefines = " -D SHADOW_TRACE_COLORED_SHADOWS -D ENABLE_OPACITY_TEX -D ENABLE_BLINN "; // -D NEXT_BOUNCE_RR

  if (!m_globals.devIsCPU && !m_globals.liteCore)
    specDefines += " -D RAYTR_THREAD_COMPACTION ";

  std::string optionsGeneral = "-cl-mad-enable -cl-no-signed-zeros -cl-single-precision-constant -cl-denorms-are-zero "; // -cl-uniform-work-group-size 
  std::string optionsInclude = "-I ../hydra_drv -I " + HydraInstallPath() + "/shaders -D OCL_COMPILER ";             // put function that will find shader include folder

  if (SAVE_BUILD_LOG)
    optionsGeneral += "-cl-nv-verbose ";

  return optionsGeneral + optionsInclude + specDefines; // + " -cl-nv-maxrregcount=32 ";
}


void GPUOCLLayer::ResizeScreen(int width, int height, int a_flags)
{
  if (m_width == width && m_height == height)
    return;

  Base::ResizeScreen(width, height, a_flags);

  m_screen.free();

  //
  //
  m_screen.m_cpuFrameBuffer = (a_flags & GPU_RT_CPU_FRAMEBUFFER); 

  cl_int ciErr1 = CL_SUCCESS;

  if (m_screen.m_cpuFrameBuffer)
  {
    m_screen.color0                 = nullptr;
    m_screen.targetFrameBuffPointer = nullptr;
    m_screen.pbo                    = nullptr;
    m_screen.m_cpuFbufChannels      = m_vars.m_varsI[HRT_FBUF_CHANNELS];

    if(m_pExternalImage == nullptr)
      m_screen.color0CPU.resize(width * height * m_screen.m_cpuFbufChannels);

    std::cout << "[cl_core]: use CPU framebuffer" << std::endl;
  }
  else
  {
    m_screen.color0 = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*width*height, NULL, &ciErr1);

    if (ciErr1 != CL_SUCCESS)
      RUN_TIME_ERROR("[cl_core]: Failed to create cl full screen color buffers ");

    m_screen.targetFrameBuffPointer = m_screen.color0;

    if (ciErr1 != CL_SUCCESS)
      RUN_TIME_ERROR("[cl_core]: Failed to create cl full screen zblocks buffer ");

    if (m_initFlags & GPU_RT_NOWINDOW)
      m_screen.pbo = nullptr;
    else
      m_screen.pbo = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(cl_uint)*width*height, NULL, &ciErr1);

    std::cout << "[cl_core]: use GPU framebuffer" << std::endl;
  }

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create cl full screen uint buffer (LDR image) ");

  m_width  = width;
  m_height = height;

  const size_t MEGABLOCK_SIZE = CalcMegaBlockSize(a_flags); 

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create cl half screen zblocks buffer ");

  if(m_screen.color0 != nullptr)
    memsetf4(m_screen.color0, float4(0, 0, 0, 0), m_width*m_height);

  if (m_screen.pbo != nullptr)
    memsetu32(m_screen.pbo, 0, m_width*m_height);

  m_memoryTaken[MEM_TAKEN_RAYS] = m_rays.resize(m_globals.ctx, m_globals.cmdQueue, MEGABLOCK_SIZE, m_globals.cpuTrace, m_screen.m_cpuFrameBuffer, true); // TODO: fix true, pass this flag elsewhere, to early for check (m_vars.m_varsI[HRT_ENABLE_SURFACE_PACK] == 1)

  MLT_Alloc_For_PT_QMC(1, kmlt.xVectorQMC); // Allocate memory for testing QMC/KMLT F(xVec,bounceNum); THIS IS IMPORTANT CALL! It sets internal KMLT variables

  memsetf4(m_rays.rayPos, float4(0, 0, 0, 0), m_rays.MEGABLOCKSIZE);
  memsetf4(m_rays.rayDir, float4(0, 0, 0, 0), m_rays.MEGABLOCKSIZE);
  CHECK_CL(clFinish(m_globals.cmdQueue)); 

  // estimate mem taken
  //
  m_memoryTaken[MEM_TAKEN_SCREEN] = 0;
  if(!m_screen.m_cpuFrameBuffer)
    m_memoryTaken[MEM_TAKEN_SCREEN] += (sizeof(cl_float) * 4)*width*height;                                                                             

  const float scaleX = sqrtf( float(width)  / 1024.0f );
  const float scaleY = sqrtf( float(height) / 1024.0f );

  m_vars.m_varsF[HRT_MLT_SCREEN_SCALE_X] = clamp(scaleX, 1.0f, 4.0f);
  m_vars.m_varsF[HRT_MLT_SCREEN_SCALE_Y] = clamp(scaleY, 1.0f, 4.0f);

}

size_t GPUOCLLayer::GetAvaliableMemoryAmount(bool allMem)
{
  cl_ulong memTotal = 0;
  size_t   paramValueSize = 0;
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memTotal, &paramValueSize));

  if (allMem)
    return size_t(memTotal);

  long long int memLeft = (long long int)(memTotal);

  for (int i = 0; i < MEM_TAKEN_OBJECTS_NUM; i++)
    memLeft -= m_memoryTaken[i];

  if (memLeft < 0)
    memLeft = 0;

  return memLeft;
}

size_t GPUOCLLayer::GetMaxBufferSizeInBytes()
{
  cl_ulong maxBufferSize = 0;
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxBufferSize, NULL));
  return size_t(maxBufferSize);
}

size_t GPUOCLLayer::GetMemoryTaken()
{
  size_t taken = 0;
  for (int i = 0; i < MEM_TAKEN_OBJECTS_NUM; i++)
    taken += m_memoryTaken[i];
  return taken;
}


const char* GPUOCLLayer::GetDeviceName(int* pOCLVer) const
{
  memset(m_deviceName, 0, 1024);
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_NAME, 1024, m_deviceName, NULL));

  std::string devName2 = cutSpaces(m_deviceName);
  strncpy(m_deviceName, devName2.c_str(), 1024);

  if (pOCLVer != nullptr)
    (*pOCLVer) = m_globals.oclVer;

  return m_deviceName;
}

extern int g_maxCPUThreads;

void GPUOCLLayer::GetLDRImage(uint* data, int width, int height) const
{
  cl_int ciErr1          = CL_SUCCESS;
  cl_mem tempLDRBuff     = m_screen.pbo;
  bool needToFreeLDRBuff = false;

  size_t size = m_width * m_height;
  
  if (m_screen.m_cpuFrameBuffer) 
  {
    if (m_passNumber - 1 <= 0) // remember about pipelined copy!!
      return;

    int width2, height2, channels2;
    const float* color0 = GetCPUScreenBuffer(0, width2, height2, channels2);
    const float* color1 = GetCPUScreenBuffer(1, width2, height2, channels2);

    float normConst   = 1.0f / m_spp; // 1.0f / float(m_passNumber - 1); // remember about pipelined copy!!
    float normConstDL = 1.0f / m_sppDL; 

    if (m_vars.m_flags & HRT_ENABLE_MMLT && (m_vars.m_flags & HRT_ENABLE_SBPT) == 0)  
      normConst = EstimateMLTNormConst((float4*)color0, width, height);

    //if (!HydraSSE::g_useSSE || channels2 == 1) // this is old-style gamma-correction
    if(true)
    {
      #pragma omp parallel for num_threads(g_maxCPUThreads)
      for (int i = 0; i < size; i++)  // #TODO: use sse and fast pow
      {
        float4 color {};
        if(channels2 == 1)
          color = float4{color0[i], color0[i], color0[i], color0[i]};
        else if(channels2 >= 4)
          color = float4{color0[i * 4 + 0], color0[i * 4 + 1], color0[i * 4 + 2], color0[i * 4 + 3]};

        color.x = linearToSRGB(color.x*normConst);
        color.y = linearToSRGB(color.y*normConst);
        color.z = linearToSRGB(color.z*normConst);
        color.w = linearToSRGB(color.w*normConst);
        data[i] = RealColorToUint32(ToneMapping4(color));
      }
    }
    else // todo: remove all this shit!
    {
      const float gammaInv   = 1.0f / m_vars.m_varsF[HRT_IMAGE_GAMMA];
      const __m128 powerf4   = _mm_set_ps(gammaInv, gammaInv, gammaInv, gammaInv);
      const __m128 normc     = _mm_set_ps(normConst, normConst, normConst, normConst);
      const __m128 normc2    = _mm_set_ps(normConstDL, normConstDL, normConstDL, normConstDL);
      const __m128 const_255 = _mm_set_ps1(255.0f);

      const float* dataHDR  = (const float*)color0;
      const float* dataHDR1 = (const float*)color1;

      if (m_vars.m_flags & HRT_ENABLE_MMLT && (m_vars.m_flags & HRT_ENABLE_SBPT) == 0)
      {
        assert(channels2 == 4);
        if(channels2 != 4)
        {
          std::cerr << "GPUOCLLayer::GetLDRImage(HRT_ENABLE_MMLT): internal CPU image channels != 4" << std::endl;
          std::cerr.flush();
        }
        else
        {
          if (color1 != nullptr && color0 != nullptr)
          {
            #pragma omp parallel for num_threads(g_maxCPUThreads)
            for (int i = 0; i < size; i++)
            {
              const __m128 colorDL = _mm_mul_ps(normc2, _mm_load_ps(dataHDR1 + i * 4));
              const __m128 colorIL = _mm_mul_ps(normc, _mm_load_ps(dataHDR + i * 4));
              const __m128 color2 = HydraSSE::powf4(_mm_add_ps(colorDL, colorIL), powerf4);
              const __m128i rgba = _mm_cvtps_epi32(_mm_min_ps(_mm_mul_ps(color2, const_255), const_255));
              const __m128i out = _mm_packus_epi32(rgba, _mm_setzero_si128());
              const __m128i out2 = _mm_packus_epi16(out, _mm_setzero_si128());
              data[i] = _mm_cvtsi128_si32(out2);
            }

          }
          else if (color0 != nullptr)
          {
            #pragma omp parallel for  num_threads(g_maxCPUThreads)
            for (int i = 0; i < size; i++)
            {
              data[i] = HydraSSE::gammaCorr(dataHDR + i * 4, normc, powerf4);
            }
          }
          else
          {
            std::cerr << "GPUOCLLayer::GetLDRImage(HRT_ENABLE_MMLT): both internal CPU images == nullptr!!!" << std::endl;
            std::cerr.flush();
          }
        }
      }
      else
      {
        if(channels2 == 4)
        {
          #pragma omp parallel for num_threads(g_maxCPUThreads)
          for (int i = 0; i < size; i++)
          {
            data[i] = HydraSSE::gammaCorr(dataHDR + i * 4, normc, powerf4);
          }
        }
      }
    }
  }
  else
  {
    if (tempLDRBuff == 0)
    {
      std::cerr << "[cl_core]: null m_screen.pbo, alloc temp buffer in host memory " << std::endl;
      cvex::vector<float4> hdrData(width*height);
      GetHDRImage(&hdrData[0], width, height);

      #pragma omp parallel for num_threads(g_maxCPUThreads)
      for (int i = 0; i < size; i++)  // #TODO: use sse and fast pow
      {
        float4 color = hdrData[i];
        color.x = linearToSRGB(color.x);
        color.y = linearToSRGB(color.y);
        color.z = linearToSRGB(color.z);
        color.w = linearToSRGB(color.w);
        data[i] = RealColorToUint32(ToneMapping4(color));
      }
      
    }
    else
    {
      CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, tempLDRBuff, CL_TRUE, 0, m_width*m_height * sizeof(uint), data, 0, NULL, NULL));
    }
    
  }

}

std::tuple<double, double, double> HydraRender::ColorSummImage4f(const float* a_image4f, int a_width, int a_height);

float GPUOCLLayer::EstimateMLTNormConst(const float4* data, int width, int height) const // #TODO: opt this with simdpp library ...   
{
  double summ[3];
  std::tie(summ[0], summ[1], summ[2]) = HydraRender::ColorSummImage4f((const float*)data, width, height);
  double avgDiv       = 1.0/double(width*height);
  float avgBrightness = contribFunc(float3(avgDiv*summ[0], avgDiv*summ[1], avgDiv*summ[2]));
  return m_avgBrightness / fmax(avgBrightness, DEPSILON2);
}

void GPUOCLLayer::GetHDRImage(float4* data, int width, int height) const
{
  if (m_passNumber - 1 <= 0 || m_spp <= 1e-5f) // remember about pipelined copy!!
    return;

  float normConst = 1.0f;
  if(m_sppDone > 0)
    normConst = (1.0f / m_sppDone); // valid only for CPU FB, remember about pipelined copy. m_spp and m_sppDone are different!!! 
  else
    normConst = (1.0f / m_spp);     // may be valid if FB is on GPU

  if (m_screen.m_cpuFrameBuffer)
  {
    if(m_screen.m_cpuFbufChannels != 4)
      return;

    if (m_vars.m_flags & HRT_ENABLE_MMLT)  
      normConst = EstimateMLTNormConst((float4*)m_screen.color0CPU.data(), width, height);

    for (size_t i = 0; i < (width*height); i++)
    {
      for(int j = 0; j < m_screen.m_cpuFbufChannels; ++j)
        data[i][j] = m_screen.color0CPU[i * m_screen.m_cpuFbufChannels + j] * normConst;
    }
  }
  else if(m_screen.color0 != nullptr)
  {
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_screen.color0, CL_TRUE, 0, m_width*m_height * sizeof(cl_float4), data, 0, NULL, NULL));
    for (size_t i = 0; i < (width*height); i++)
      data[i] = data[i] * normConst;
  }
}

typedef IHostRaysAPI* (*MakeEmitterFT)(int);
typedef void (*DeleteEmitterFT)(IHostRaysAPI*);

//IHostRaysAPI* MakeHostRaysEmitter(int a_pluginId);       ///<! you replace this function or make your own ... the example will be provided
//void          DeleteRaysEmitter(IHostRaysAPI* pObject);

void GPUOCLLayer::InitPathTracing(int seed, std::vector<int32_t>* pInstRemapTable)
{
  std::cout << "[cl_core]: InitRandomGen seed = " << seed << std::endl;
  
  runKernel_InitRandomGen(m_rays.randGenState, m_rays.MEGABLOCKSIZE, seed);
  m_passNumber = 0;
  m_spp        = 0.0f;
  m_sppDone    = 0.0f;
  m_sppContrib = 0.0f;
  m_passNumberForQMC = 0;

  ClearAccumulatedColor();
  
  // reset camera plugin if we have it ...
  //
  const int cpuPluginId = m_camNode.attribute(L"cpu_plugin").as_int();
  if(m_camPlugin.pCamPlugin == nullptr && cpuPluginId != 0) // actual plugin init happened first time
  {
    const wchar_t* dllPath = m_camNode.attribute(L"cpu_plugin_dll").as_string();
    if(std::wstring(dllPath) != L"")
    {
      const std::string dllPath2 = ws2s(std::wstring(dllPath));
      std::cout << "[INFO]: load 'IHostRaysAPI' plugin from '" << dllPath2.c_str() << "'" << std::endl; 
      
      HydraDllHandle* pHandle = HydraLoadLibrary(dllPath);
      if(pHandle == nullptr)
        std::cout << "[ERROR]: can't load 'IHostRaysAPI' plugin from '" << dllPath2.c_str() << "'" << std::endl; 
      
      MakeEmitterFT   MakeFunc = (MakeEmitterFT)HydraGetProcAddress(pHandle, "MakeHostRaysEmitter");
      DeleteEmitterFT DelFunc  = (DeleteEmitterFT)HydraGetProcAddress(pHandle, "DeleteRaysEmitter");

      m_camPlugin.pCamPlugin = std::shared_ptr<IHostRaysAPI>(MakeFunc(m_vars.m_varsI[HRT_USE_CPU_PLUGIN]), DelFunc);
    }
    else
      m_camPlugin.pCamPlugin = std::shared_ptr<IHostRaysAPI>(MakeHostRaysEmitter(m_vars.m_varsI[HRT_USE_CPU_PLUGIN]), DeleteRaysEmitter);
    
    std::wstringstream strout, strout2;
    
    uint64_t address = reinterpret_cast<uint64_t>((void*)pInstRemapTable->data());
    strout2 << std::noshowbase << std::setw(16) << std::setfill(L"0"[0]) << address;
    auto str2 = strout2.str();
    m_settingsNode.force_child(L"remapInstAddress").text() = str2.c_str();
    m_settingsNode.force_child(L"remapInstSize").text()    = int(pInstRemapTable->size());
    m_settingsNode.force_child(L"xmlfilepath").text()      = g_internalLibPath.c_str();

    strout << L"<?xml version=\"1.0\"?>" << std::endl;
    m_camNode.print(strout);
    m_settingsNode.print(strout);
    const std::wstring nodeData = strout.str(); 
    m_camPlugin.pCamPlugin->SetParameters(m_width, m_height, m_globsBuffHeader.mProjInverse, nodeData.c_str());

    m_camPlugin.free();
    cl_int ciErr1 = CL_SUCCESS;
    size_t totalSize = sizeof(RayPart1)*m_rays.MEGABLOCKSIZE + sizeof(RayPart2)*m_rays.MEGABLOCKSIZE;
    m_camPlugin.camRayGPU[0] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, totalSize, NULL, &ciErr1);
    m_camPlugin.camRayGPU[1] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, totalSize, NULL, &ciErr1);
    m_camPlugin.camRayCPU[0] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, totalSize, NULL, &ciErr1);
    m_camPlugin.camRayCPU[1] = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, totalSize, NULL, &ciErr1);
    m_camPlugin.accumBuff    = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, m_rays.MEGABLOCKSIZE*sizeof(float4), NULL, &ciErr1);
  
    if (ciErr1 != CL_SUCCESS)
      RUN_TIME_ERROR("[HostRaysAPI]: Error in create rays buffers for Host Camera Plugin");
  }
}

void GPUOCLLayer::ClearAccumulatedColor()
{
  if (m_screen.m_cpuFrameBuffer && !m_screen.color0CPU.empty())
    memset(m_screen.color0CPU.data(), 0, m_width * m_height * m_screen.m_cpuFbufChannels * sizeof(float));
  else if(m_screen.color0 != nullptr)
    memsetf4(m_screen.color0, make_float4(0, 0, 0, 0.0f), m_width*m_height); // #TODO: change this for 2D memset to support large resolutions!!!!

  m_mlt.mppDone = 0.0;
  m_spp         = 0.0f;
}

MRaysStat GPUOCLLayer::GetRaysStat()
{
  return m_stat;
}

void GPUOCLLayer::ResetPerfCounters()
{
  memset(&m_stat, 0, sizeof(MRaysStat));
}


const float* GPUOCLLayer::GetCPUScreenBuffer(int a_layerId, int& width, int& height, int& channels) const
{
  if(!m_screen.m_cpuFrameBuffer)
    return nullptr;
     
  const float* resultPtr = nullptr;
  width  = m_width;
  height = m_height;
  
  if (m_pExternalImage != nullptr)
  {
    resultPtr = m_pExternalImage->ImageData(a_layerId);
    width     = m_pExternalImage->Header()->width;
    height    = m_pExternalImage->Header()->height;
    channels  = m_pExternalImage->Header()->channels;
  }
  else
  {
    channels = m_screen.m_cpuFbufChannels;
    if(a_layerId == 0)
      resultPtr = reinterpret_cast<const float *>(m_screen.color0CPU.data());
    else if(a_layerId == 1)
      resultPtr = reinterpret_cast<const float *>(m_mlt.colorDLCPU.data());
  }

  //std::cout << "GPUOCLLayer::GetCPUScreenBuffer channels = " << channels;

  return resultPtr;     
}



void GPUOCLLayer::BeginTracingPass()
{
  static int firstCall = 0;
  firstCall++;

  m_timer.start();

  const int minBounce  = m_vars.m_varsI[HRT_MMLT_FIRST_BOUNCE];
  const int maxBounce  = m_vars.m_varsI[HRT_TRACE_DEPTH];
  const int BURN_ITERS = m_vars.m_varsI[HRT_MMLT_BURN_ITERS];

  //m_vars.m_flags |= HRT_ENABLE_PT_CAUSTICS;
  //UpdateVarsOnGPU(m_vars);

  bool asyncPluginMode = true; // enable parallel evaluation of ray block in CPU plugin

  if((m_vars.m_flags & HRT_ENABLE_SBPT) != 0)
  {
    #ifdef SBDPT_INDIRECT_ONLY
    const int sbptBounceBeg = 3;
    #else
    const int sbptBounceBeg = 1;
    #endif
    SBDPT_Pass(sbptBounceBeg, maxBounce, NUM_MMLT_PASS);
    //SBDPT_Pass(1, 2, NUM_MMLT_PASS);
  }
  else if (m_vars.m_flags & HRT_ENABLE_MMLT)                 // SBDPT or MMLT pass
  { 
    #ifndef SBDPT_CHECK_BOUNCE
    #ifndef SBDPT_INDIRECT_ONLY
    DL_Pass(maxBounce, NUM_MMLT_PASS/2);  //#NOTE: strange bug, DL contribute to IL if reverse order
    #endif 
    #endif
    MMLT_Pass(NUM_MMLT_PASS, minBounce, maxBounce, BURN_ITERS);   
  }
  else if((m_vars.m_flags & HRT_PRODUCTION_IMAGE_SAMPLING) != 0 && (m_vars.m_flags & HRT_UNIFIED_IMAGE_SAMPLING) != 0)
  {
    if(firstCall > 1) // stupid hack to calc gbuffer instead of render when first hrCommit is called.
    RunProductionSamplingMode();
  }
  else if (m_vars.m_flags & HRT_UNIFIED_IMAGE_SAMPLING) // PT or LT pass
  { 
    const int minBounce = 1;
    const int maxBounce = m_vars.m_varsI[HRT_TRACE_DEPTH];
    std::future<int> pluginExecution;
    
    if ((m_vars.m_flags & HRT_FORWARD_TRACING) != 0)    // LT 
    {
      m_vars.m_varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] = 0;  // explicit disable reading random numbers from buffer
      m_vars.m_varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES] = 0;
      UpdateVarsOnGPU(m_vars);

      EvalLT(nullptr, minBounce, maxBounce, m_rays.MEGABLOCKSIZE, 
             m_rays.pathAccColor);
    }
    //else if( ... ) // Kelemen MLT is enabled
    //{ 
    //  KMLT_Pass(NUM_MMLT_PASS, minBounce, maxBounce, 128); // BURN_ITERS
    //}
    else if(m_vars.m_varsI[HRT_USE_CPU_PLUGIN] >= 1)
    {
      m_vars.m_varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] = 0;
      m_vars.m_varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES] = 0;
      UpdateVarsOnGPU(m_vars);
      
    
      int buffId = m_passNumber % 2;
      if(asyncPluginMode)
        pluginExecution = std::async(std::launch::async, &GPUOCLLayer::DoCamPluginRays, this, buffId, m_passNumber);
      else
        DoCamPluginRays(buffId, m_passNumber);
  
      if(m_passNumber >= 1)
      {
        const int samplesPerPass = m_vars.m_varsI[HRT_SAMPLES_PER_PASS];
        memsetf4(m_camPlugin.accumBuff, float4(0,0,0,0), m_rays.MEGABLOCKSIZE);
        for(int i=0;i<samplesPerPass;i++)
        {
          runKernel_TakeHostRays(m_camPlugin.camRayGPU[1-buffId], m_rays.rayPos, m_rays.rayDir, m_rays.pathAccColor, m_rays.MEGABLOCKSIZE);
          trace1D_Rev(minBounce, maxBounce, m_rays.rayPos, m_rays.rayDir, m_rays.MEGABLOCKSIZE, m_rays.pathAccColor);
          runKernel_AccumColor(m_rays.pathAccColor, m_rays.surfId, m_camPlugin.accumBuff, m_rays.MEGABLOCKSIZE, 1.0f/float(samplesPerPass));
        }
      }

      AddContributionToScreen(m_camPlugin.accumBuff, m_rays.samZindex);
      
      if(asyncPluginMode)
        pluginExecution.get();
    }
    else                                                // PT 
    { 
      //m_vars.m_flags |= HRT_INDIRECT_LIGHT_MODE; // for test
      //m_vars.m_flags |= HRT_DIRECT_LIGHT_MODE;   // for test
     
      m_vars.m_varsI[HRT_KMLT_OR_QMC_LGT_BOUNCES] = kmlt.maxBounceQMC;
      m_vars.m_varsI[HRT_KMLT_OR_QMC_MAT_BOUNCES] = kmlt.maxBounceQMC;
      UpdateVarsOnGPU(m_vars);

      if(kmlt.maxBounceQMC != 0) 
      {
        runKernel_MakeEyeRaysQMC(m_rays.MEGABLOCKSIZE, m_passNumberForQMC,
                                 m_rays.samZindex, kmlt.xVectorQMC);
      }
      else
      {
        runKernel_MakeEyeSamplesOnly(m_rays.MEGABLOCKSIZE, m_passNumberForQMC,
                                     m_rays.samZindex, kmlt.xVectorQMC);
      }
      
      EvalPT(kmlt.xVectorQMC, m_rays.samZindex, minBounce, maxBounce, m_rays.MEGABLOCKSIZE,
             m_rays.pathAccColor);

      AddContributionToScreen(m_rays.pathAccColor, m_rays.samZindex);
    }
    
  }
  else if(!m_screen.m_cpuFrameBuffer)
  { 
    DrawNormals();
  }
}

int GPUOCLLayer::DoCamPluginRays(int buffId, int a_passId)
{
  auto start = std::chrono::high_resolution_clock::now();

  cl_int ciErr1 = CL_SUCCESS;
  const size_t fullSize = m_rays.MEGABLOCKSIZE*sizeof(RayPart1) + m_rays.MEGABLOCKSIZE*sizeof(RayPart2);
  
  RayPart1* rays1 = (RayPart1*)clEnqueueMapBuffer(m_globals.cmdQueueHostToDev, m_camPlugin.camRayCPU[buffId], CL_TRUE, CL_MAP_WRITE, 0, fullSize, 0, 0, 0, &ciErr1);
  RayPart2* rays2 = (RayPart2*)(rays1 + m_rays.MEGABLOCKSIZE);
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[HostRaysAPI]: Error in 'clEnqueueMapBuffer' for Host Camera Plugin");
  
  m_camPlugin.pCamPlugin->MakeRaysBlock(rays1, rays2, m_rays.MEGABLOCKSIZE, a_passId);
  
  clEnqueueUnmapMemObject(m_globals.cmdQueueHostToDev, m_camPlugin.camRayCPU[buffId], rays1, 0, 0, 0);
  clEnqueueCopyBuffer    (m_globals.cmdQueueHostToDev, m_camPlugin.camRayCPU[buffId], m_camPlugin.camRayGPU[buffId], 0, 0, fullSize, 0, nullptr, nullptr);
  
  m_camPlugin.pipeTime[0] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
  return 0;
}


void GPUOCLLayer::EndTracingPass()
{
  if (!m_screen.m_cpuFrameBuffer)
  {
    CHECK_CL(clFinish(m_globals.cmdQueue));
  }

  m_passNumberForQMC++;
  if (m_vars.m_flags & HRT_UNIFIED_IMAGE_SAMPLING)
  {
    const float passScale = (m_vars.m_flags & HRT_ENABLE_MMLT) ? float(NUM_MMLT_PASS) : 1.0f;
    m_spp += passScale*float(double(m_rays.MEGABLOCKSIZE) / double(m_width*m_height));

    const float time = m_timer.getElapsed();
    if (m_passNumberForQMC % 4 == 0 && m_passNumberForQMC > 0)
    {
      const float halfIfIBPT = (m_vars.m_flags & HRT_3WAY_MIS_WEIGHTS) ? 0.5f : 1.0f;
      auto precOld = std::cout.precision(2);
      std::cout << "spp =\t" << int(m_spp) << "\tspeed = " << passScale*halfIfIBPT * float(m_rays.MEGABLOCKSIZE) / (1e6f*time) << " M(samples)/s         \r";
      std::cout.precision(precOld);
      std::cout.flush();
    }
  }
  else
  {
    m_spp              = 0;
    m_passNumberForQMC = 0;
    m_sppDL            = 0;
  }
}


IHWLayer* CreateOclImpl(int w, int h, int a_flags, int a_deviceId) 
{
  return new GPUOCLLayer(w, h, a_flags, a_deviceId);
}


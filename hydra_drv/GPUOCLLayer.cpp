#include "GPUOCLLayer.h"
#include "crandom.h"

#include "../../HydraAPI/hydra_api/xxhash.h"
#include "../../HydraAPI/hydra_api/ssemath.h"

#include "cl_scan_gpu.h"

extern "C" void initQuasirandomGenerator(unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]);

#include <algorithm>
#undef min
#undef max

constexpr bool SAVE_BUILD_LOG    = false;
constexpr bool FORCE_DRAW_SHADOW = false;

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
  //scan_free_internal();
}

void GPUOCLLayer::CL_BUFFERS_RAYS::free()
{
  if(rayPos)   { clReleaseMemObject(rayPos);   rayPos   = nullptr; }
  if(rayDir)   { clReleaseMemObject(rayDir);   rayDir   = nullptr; }
  if(hits)     { clReleaseMemObject(hits);     hits     = nullptr; }
  if(rayFlags) { clReleaseMemObject(rayFlags); rayFlags = nullptr; }
                                                                                                     
  if (hitPosNorm)          { clReleaseMemObject(hitPosNorm);          hitPosNorm          = nullptr; }
  if (hitTexCoord)         { clReleaseMemObject(hitTexCoord);         hitTexCoord         = nullptr; }
  if (hitMatId)            { clReleaseMemObject(hitMatId);            hitMatId            = nullptr; }
  if (hitTangent)          { clReleaseMemObject(hitTangent);          hitTangent          = nullptr; }
  if (hitFlatNorm)         { clReleaseMemObject(hitFlatNorm);         hitFlatNorm         = nullptr; }
  if (hitPrimSize)         { clReleaseMemObject(hitPrimSize);         hitPrimSize         = nullptr; }
  if (hitNormUncompressed) { clReleaseMemObject(hitNormUncompressed); hitNormUncompressed = nullptr; }
  if (hitProcTexData)      { clReleaseMemObject(hitProcTexData);      hitProcTexData      = nullptr;}

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

  if (lsam1)           { clReleaseMemObject(lsam1); lsam1 = nullptr; }
  if (lsam2)           { clReleaseMemObject(lsam2); lsam2 = nullptr; }
  if (lsamCos)         { clReleaseMemObject(lsamCos); lsamCos = nullptr; }

  if (shadowRayPos)    { clReleaseMemObject(shadowRayPos); shadowRayPos = nullptr; }
  if (shadowRayDir)    { clReleaseMemObject(shadowRayDir); shadowRayDir = nullptr; }
  if (accPdf)          { clReleaseMemObject(accPdf);       accPdf       = nullptr; }

  if(oldFlags)         { clReleaseMemObject(oldFlags);  oldFlags  = nullptr; }
  if(oldRayDir)        { clReleaseMemObject(oldRayDir); oldRayDir = nullptr; }
  if(oldColor)         { clReleaseMemObject(oldColor);  oldColor  = nullptr; }
  if(lightNumberLT)    { clReleaseMemObject(lightNumberLT); lightNumberLT = nullptr; }

  if (lsamProb)        { clReleaseMemObject(lsamProb); lsamProb = nullptr; }
  if (lshadow)         { clReleaseMemObject(lshadow);  lshadow  = nullptr; }
                       
  if (fogAtten)        { clReleaseMemObject(fogAtten);   fogAtten   = nullptr; }
  if (samZindex)       { clReleaseMemObject(samZindex);  samZindex  = nullptr; }
  if (aoCompressed)    { clReleaseMemObject(aoCompressed);   aoCompressed  = nullptr; }
  if (aoCompressed2)   { clReleaseMemObject(aoCompressed2);  aoCompressed2 = nullptr; }
  if (lightOffsetBuff) { clReleaseMemObject(lightOffsetBuff);  lightOffsetBuff = nullptr; }
  if (packedXY)        { clReleaseMemObject(packedXY);   packedXY   = nullptr; }
  if (debugf4)         { clReleaseMemObject(debugf4);    debugf4    = nullptr; }

  if(atomicCounterMem) { clReleaseMemObject(atomicCounterMem); atomicCounterMem = nullptr;}
}

size_t GPUOCLLayer::CL_BUFFERS_RAYS::resize(cl_context ctx, cl_command_queue cmdQueue, size_t a_size, bool a_cpuShare, bool a_cpuFB)
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

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  hitPosNorm  = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 4;
  hitTexCoord = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);              currSize += buff1Size * 2;
  hitMatId    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(HitMatRef)*MEGABLOCKSIZE, NULL, &ciErr1);                 currSize += buff1Size * 2;
  hitTangent  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(Hit_Part4)*MEGABLOCKSIZE, NULL, &ciErr1);                 currSize += buff1Size * 2;
  hitFlatNorm = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint)*MEGABLOCKSIZE, NULL, &ciErr1);                      currSize += buff1Size * 1;
  hitPrimSize = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float)*MEGABLOCKSIZE, NULL, &ciErr1);                     currSize += buff1Size * 1;
  hitNormUncompressed = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float4)*MEGABLOCKSIZE, NULL, &ciErr1);            currSize += buff1Size * 4;
  hitProcTexData      = nullptr;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  pathThoroughput = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);          currSize += buff1Size * 4;
  pathMisDataPrev = clCreateBuffer(ctx, CL_MEM_READ_WRITE,     sizeof(MisData) *MEGABLOCKSIZE, NULL, &ciErr1);          currSize += buff1Size * 1;
  pathShadeColor  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);          currSize += buff1Size * 4;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  pathAccColor = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);  currSize += buff1Size * 4;
  randGenState = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1 * sizeof(RandomGen)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 1;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  lsam1        = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);     currSize += buff1Size * 4;
  lsam2        = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);     currSize += buff1Size * 4;
  lsamCos      = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);     currSize += buff1Size * 4;
                                                                                                                          
  shadowRayPos = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  shadowRayDir = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  accPdf       = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 1 * sizeof(PerRayAcc)*MEGABLOCKSIZE, NULL, &ciErr1);  currSize += buff1Size * 1;
                                                                                                                           
  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  oldFlags      = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 1 * sizeof(uint)*MEGABLOCKSIZE,     NULL, &ciErr1);   currSize += buff1Size * 1;
  oldRayDir     = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  oldColor      = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 4 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);   currSize += buff1Size * 4;
  lightNumberLT = clCreateBuffer(ctx, CL_MEM_READ_WRITE | shareFlags, 1 * sizeof(int)*MEGABLOCKSIZE,      NULL, &ciErr1);   currSize += buff1Size * 1;

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in resize rays buffers");

  lsamProb = clCreateBuffer(ctx, CL_MEM_READ_WRITE,              1 * sizeof(cl_float)*MEGABLOCKSIZE, NULL, &ciErr1);        currSize += buff1Size * 1;
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

  samZindex       = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1*2*sizeof(int)*MEGABLOCKSIZE, NULL, &ciErr1); currSize += buff1Size * 2;
  packedXY        = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int)*MEGABLOCKSIZE, NULL, &ciErr1);     currSize += buff1Size*1;
  lightOffsetBuff = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int)*MEGABLOCKSIZE, NULL, &ciErr1);     currSize += buff1Size * 1;

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


GPUOCLLayer::GPUOCLLayer(int w, int h, int a_flags, int a_deviceId) : Base(w, h, a_flags)
{ 
  m_initFlags = a_flags;
  for (int i = 0; i < MEM_TAKEN_OBJECTS_NUM; i++)
    m_memoryTaken[i] = 0;
  
  InitEngineGlobals(&m_globsBuffHeader);
  
  #ifdef WIN32
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

  cl_int ciErr1;
 
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

  std::string sshaderpathBin  = installPath2 + "shadercache/" + "screen_" + devHash + ".bin";
  std::string tshaderpathBin  = installPath2 + "shadercache/" + "tracex_" + devHash + ".bin";
  std::string soshaderpathBin = installPath2 + "shadercache/" + "sortxx_" + devHash + ".bin";
  std::string ioshaderpathBin = installPath2 + "shadercache/" + "imagex_" + devHash + ".bin";
  std::string moshaderpathBin = installPath2 + "shadercache/" + "mltxxx_" + devHash + ".bin";
  std::string loshaderpathBin = installPath2 + "shadercache/" + "lightx_" + devHash + ".bin";
  std::string yoshaderpathBin = installPath2 + "shadercache/" + "matsxx_" + devHash + ".bin";

  bool inDevelopment = (a_flags & GPU_RT_IN_DEVELOPMENT);
  std::string loadEncrypted = "load"; // ("crypt", "load", "")
  if (inDevelopment)
    loadEncrypted = "";

  if ((a_flags & GPU_RT_CLEAR_SHADER_CACHE) || inDevelopment)
  {
    std::remove(sshaderpathBin.c_str());
    std::remove(tshaderpathBin.c_str());
    std::remove(soshaderpathBin.c_str());
    std::remove(ioshaderpathBin.c_str());
    std::remove(moshaderpathBin.c_str());
    std::remove(loshaderpathBin.c_str());
    std::remove(yoshaderpathBin.c_str());
  }

  std::string options = GetOCLShaderCompilerOptions();
  std::cout << "[cl_core]: building cl programs ..." << std::endl;

  //m_progressBar("Compiling shaders", 0.1f);
  std::cout << "[cl_core]: building " << ishaderpath.c_str() << "    ..." << std::endl;
  m_progs.imagep = CLProgram(m_globals.device, m_globals.ctx, ishaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, ioshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << soshaderpath.c_str() << "     ..." << std::endl;
  m_progs.sort   = CLProgram(m_globals.device, m_globals.ctx, soshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, soshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << mshaderpath.c_str() << "      ... " << std::endl;
  m_progs.mlt    = CLProgram(m_globals.device, m_globals.ctx, mshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, moshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << sshaderpath.c_str() <<  "   ... " << std::endl;
  m_progs.screen = CLProgram(m_globals.device, m_globals.ctx, sshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, sshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << tshaderpath.c_str() << "    ..." << std::endl;
  m_progs.trace  = CLProgram(m_globals.device, m_globals.ctx, tshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, tshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << lshaderpath.c_str() << "    ..." << std::endl;
  m_progs.lightp = CLProgram(m_globals.device, m_globals.ctx, lshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, loshaderpathBin, SAVE_BUILD_LOG);

  std::cout << "[cl_core]: building " << yshaderpath.c_str() << " ..." << std::endl;
  m_progs.material = CLProgram(m_globals.device, m_globals.ctx, yshaderpath.c_str(), options.c_str(), HydraInstallPath(), loadEncrypted, yoshaderpathBin, SAVE_BUILD_LOG);

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

    if (!isFileExists(tshaderpathBin))
      m_progs.trace.saveBinary(tshaderpathBin);

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
  unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION];
  initQuasirandomGenerator(tableCPU);
  m_globals.qmcTable = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int), &tableCPU, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error when create qmcTable");

  float2 qmc[GBUFFER_SAMPLES];
  float2 qmc2[PMPIX_SAMPLES];
  
  PlaneHammersley(&qmc[0].x, GBUFFER_SAMPLES);
  PlaneHammersley(&qmc2[0].x, PMPIX_SAMPLES);

  m_globals.hammersley2DGBuff = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qmc),  qmc,  &ciErr1);
  m_globals.hammersley2D256   = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(qmc2), qmc2, &ciErr1);

  waitIfDebug(__FILE__, __LINE__);

  m_raysWasSorted = false;
  m_spp           = 0.0f;
  m_sppDone       = 0.0f;
  m_sppContrib    = 0.0f;
}

void GPUOCLLayer::RecompileProcTexShaders(const char* a_shaderPath)
{
  std::string options = GetOCLShaderCompilerOptions();

  std::cout << "[cl_core]: recompile " << a_shaderPath << " ..." << std::endl;
  m_progs.texproc = CLProgram(m_globals.device, m_globals.ctx, a_shaderPath, options, HydraInstallPath(), "", "", SAVE_BUILD_LOG);

  cl_int ciErr1 = 0;
  m_rays.hitProcTexData = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE, sizeof(ProcTextureList)*m_rays.MEGABLOCKSIZE, NULL, &ciErr1);
  //currSize += sizeof(ProcTextureList)*MEGABLOCKSIZE; //#TODO: ACCOUNT THIS MEM FOR MEM INFO

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

  m_mlt.free();
  m_rays.free();
  m_screen.free();
  m_scene.free();

  if (m_globals.cMortonTable)     { clReleaseMemObject(m_globals.cMortonTable);      m_globals.cMortonTable      = nullptr; }
  if (m_globals.qmcTable)         { clReleaseMemObject(m_globals.qmcTable);          m_globals.qmcTable          = nullptr; }
  if (m_globals.hammersley2DGBuff){ clReleaseMemObject(m_globals.hammersley2DGBuff); m_globals.hammersley2DGBuff = nullptr; }
  if (m_globals.hammersley2D256)  { clReleaseMemObject(m_globals.hammersley2D256);   m_globals.hammersley2D256   = nullptr; }

  if(m_globals.cmdQueue)          { clReleaseCommandQueue(m_globals.cmdQueue);          m_globals.cmdQueue          = nullptr; }
  if(m_globals.cmdQueueDevToHost) { clReleaseCommandQueue(m_globals.cmdQueueDevToHost); m_globals.cmdQueueDevToHost = nullptr; }
  if(m_globals.ctx)               { clReleaseContext     (m_globals.ctx);               m_globals.ctx               = nullptr; }
}

size_t GPUOCLLayer::CalcMegaBlockSize()
{
  const size_t memAmount = GetAvaliableMemoryAmount(true);
  const size_t MB = size_t(1024*1024);

  if (m_globals.devIsCPU)
  {
    return 256 * 256;
  }
  else if (memAmount <= size_t(256)*MB)
  {
    return 256 * 256;
  }
  else if (memAmount <= size_t(1024)*MB)
  {
    return 512 * 512;
  }
  else if (memAmount <= size_t(4*1024)*MB)
  {
    return 1024 * 512;
  }
  else
    return 1024 * 1024;

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

    if(m_pExternalImage == nullptr)
      m_screen.color0CPU.resize(width*height);

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

  const size_t MEGABLOCK_SIZE = CalcMegaBlockSize(); 

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("[cl_core]: Failed to create cl half screen zblocks buffer ");

  //scan_alloc_internal(m_width*m_height, m_globals.ctx);

  if(m_screen.color0 != nullptr)
    memsetf4(m_screen.color0, float4(0, 0, 0, 0), m_width*m_height);

  if (m_screen.pbo != nullptr)
    memsetu32(m_screen.pbo, 0, m_width*m_height);

  m_memoryTaken[MEM_TAKEN_RAYS] = m_rays.resize(m_globals.ctx, m_globals.cmdQueue, MEGABLOCK_SIZE, m_globals.cpuTrace, m_screen.m_cpuFrameBuffer);

  memsetf4(m_rays.rayPos, float4(0, 0, 0, 0), m_rays.MEGABLOCKSIZE);
  memsetf4(m_rays.rayDir, float4(0, 0, 0, 0), m_rays.MEGABLOCKSIZE);
  CHECK_CL(clFinish(m_globals.cmdQueue)); 

  // estimate mem taken
  //
  m_memoryTaken[MEM_TAKEN_SCREEN] = 0;
  if(!m_screen.m_cpuFrameBuffer)
    m_memoryTaken[MEM_TAKEN_SCREEN] += (sizeof(cl_float) * 4)*width*height;                                                                             

  const float scaleX = float(width) / 1024.0f;
  const float scaleY = float(height) / 1024.0f;

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


void GPUOCLLayer::GetLDRImage(uint* data, int width, int height) const
{
  cl_int ciErr1          = CL_SUCCESS;
  cl_mem tempLDRBuff     = m_screen.pbo;
  bool needToFreeLDRBuff = false;

  const float gammaInv   = 1.0f / m_vars.m_varsF[HRT_IMAGE_GAMMA];
  const int size         = m_width*m_height;

  if (m_screen.m_cpuFrameBuffer) 
  {
    if (m_passNumber - 1 <= 0) // remember about pipelined copy!!
      return;

    const float normConst = 1.0f / m_spp; // 1.0f / float(m_passNumber - 1); // remember about pipelined copy!!

    if (!HydraSSE::g_useSSE)
    {
      #pragma omp parallel for
      for (int i = 0; i < size; i++)  // #TODO: use sse and fast pow
      {
        float4 color = m_screen.color0CPU[i];
        color.x = powf(color.x*normConst, gammaInv);
        color.y = powf(color.y*normConst, gammaInv);
        color.z = powf(color.z*normConst, gammaInv);
        color.w = powf(color.w*normConst, gammaInv);
        data[i] = RealColorToUint32(ToneMapping4(color));
      }
    }
    else
    {
      const __m128 powerf4 = _mm_set_ps(gammaInv, gammaInv, gammaInv, gammaInv);
      const __m128 normc   = _mm_set_ps(normConst, normConst, normConst, normConst);

      const float* dataHDR = (const float*)&m_screen.color0CPU[0];

      #pragma omp parallel for
      for (int i = 0; i < size; i++)
      {
        data[i] = HydraSSE::gammaCorr(dataHDR + i*4, normc, powerf4);
      }
    }
  }
  else
  {
    if (tempLDRBuff == 0)
    {
      std::cerr << "[cl_core]: null m_screen.pbo, alloc temp buffer in host memory " << std::endl;
      std::vector<float4> hdrData(width*height);
      GetHDRImage(&hdrData[0], width, height);

      #pragma omp parallel for
      for (int i = 0; i < size; i++)  // #TODO: use sse and fast pow
      {
        float4 color = hdrData[i];
        color.x = powf(color.x, gammaInv);
        color.y = powf(color.y, gammaInv);
        color.z = powf(color.z, gammaInv);
        color.w = powf(color.w, gammaInv);
        data[i] = RealColorToUint32(ToneMapping4(color));
      }
      
    }
    else
    {
      CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, tempLDRBuff, CL_TRUE, 0, m_width*m_height * sizeof(uint), data, 0, NULL, NULL));
    }
    
  }

}

void GPUOCLLayer::GetHDRImage(float4* data, int width, int height) const
{
  if (m_passNumber - 1 <= 0 || m_spp <= 1e-5f) // remember about pipelined copy!!
    return;

  const float normConst = (1.0f / m_spp);

  if (m_screen.m_cpuFrameBuffer)
  {
    for (size_t i = 0; i < (width*height); i++)
      data[i] = m_screen.color0CPU[i] * normConst;
  }
  else if(m_screen.color0 != nullptr)
  {
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_screen.color0, CL_TRUE, 0, m_width*m_height * sizeof(cl_float4), data, 0, NULL, NULL));
    for (size_t i = 0; i < (width*height); i++)
      data[i] = data[i] * normConst;
  }
}

void GPUOCLLayer::ContribToExternalImageAccumulator(IHRSharedAccumImage* a_pImage)
{
  if (!m_screen.m_cpuFrameBuffer)
  {
    if(m_screen.color0CPU.size() != m_width * m_height)
      m_screen.color0CPU.resize(m_width*m_height);
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_screen.color0, CL_TRUE, 0, m_width*m_height * sizeof(cl_float4), m_screen.color0CPU.data(), 0, NULL, NULL));
  }

  float* input = (float*)m_screen.color0CPU.data();
  if (input == nullptr)
  {
    std::cerr << "GPUOCLLayer::ContribToExternalImageAccumulator: nullptr internal image" << std::endl;
    return;
  }

  if (a_pImage == nullptr)
  {
    std::cerr << "GPUOCLLayer::ContribToExternalImageAccumulator: nullptr external image" << std::endl;
    return;
  }

  const bool lockSuccess = a_pImage->Lock(100); // can wait 100 ms for success lock

  if (lockSuccess)
  {
    float* output  = a_pImage->ImageData(0);
    const int size = m_width*m_height;

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
      const __m128 color1 = _mm_load_ps(input  + i * 4);
      const __m128 color2 = _mm_load_ps(output + i * 4);
      _mm_store_ps(output + i * 4, _mm_add_ps(color1, color2));
    }

    a_pImage->Header()->counterRcv++;
    a_pImage->Header()->spp += m_spp;
    a_pImage->Unlock();

    ClearAccumulatedColor();
  
    m_sppContrib += m_spp;
    m_sppDone    += m_spp;
  }
  
}


void GPUOCLLayer::InitPathTracing(int seed)
{
  std::cout << "[cl_core]: InitRandomGen seed = " << seed << std::endl;
  
  runKernel_InitRandomGen(m_rays.randGenState, m_rays.MEGABLOCKSIZE, seed);
  m_passNumber = 0;
  m_spp        = 0.0f;
  m_sppDone    = 0.0f;
  m_sppContrib = 0.0f;
  m_passNumberForQMC = 0;

  ClearAccumulatedColor();
}

void GPUOCLLayer::ClearAccumulatedColor()
{
  if (m_screen.m_cpuFrameBuffer && m_screen.color0CPU.size() != 0)
    memset(m_screen.color0CPU.data(), 0, m_width*m_height*sizeof(float4));
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


void GPUOCLLayer::CopyShadowTo(cl_mem a_color, size_t a_size)
{
  size_t localWorkSize   = CMP_RESULTS_BLOCK_SIZE;
  int iSize              = int(a_size);
  a_size                 = roundBlocks(a_size, int(localWorkSize));

  cl_kernel kern         = m_progs.screen.kernel("CopyShadowTo");

  CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&m_rays.pathShadow8B));
  CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&a_color)); 
  CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&iSize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

void GPUOCLLayer::DrawNormals()
{
  cl_kernel makeRaysKern = m_progs.screen.kernel("MakeEyeRays");

  int iter = 0;
  for (size_t offset = 0; offset < m_width*m_height; offset += m_rays.MEGABLOCKSIZE)
  {
    size_t localWorkSize = 256;
    size_t globalWorkSize = m_rays.MEGABLOCKSIZE;
    cl_int iOffset = cl_int(offset);

    if (offset + globalWorkSize > (m_width*m_height))
      globalWorkSize = (m_width*m_height) - offset;

    CHECK_CL(clSetKernelArg(makeRaysKern, 0, sizeof(cl_int), (void*)&iOffset));
    CHECK_CL(clSetKernelArg(makeRaysKern, 1, sizeof(cl_mem), (void*)&m_rays.rayPos));
    CHECK_CL(clSetKernelArg(makeRaysKern, 2, sizeof(cl_mem), (void*)&m_rays.rayDir));
    CHECK_CL(clSetKernelArg(makeRaysKern, 3, sizeof(cl_int), (void*)&m_width));
    CHECK_CL(clSetKernelArg(makeRaysKern, 4, sizeof(cl_int), (void*)&m_height));
    CHECK_CL(clSetKernelArg(makeRaysKern, 5, sizeof(cl_mem), (void*)&m_scene.allGlobsData));

    if (globalWorkSize % localWorkSize != 0)
      globalWorkSize = (globalWorkSize / localWorkSize)*localWorkSize;

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, makeRaysKern, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
    
    trace1DPrimaryOnly(m_rays.rayPos, m_rays.rayDir, m_screen.color0, globalWorkSize, offset); //  m_screen.colorSubBuffers[iter]
    iter++;
  }

  cl_mem tempLDRBuff = m_screen.pbo;

  //if (!(m_initFlags & GPU_RT_NOWINDOW))
  //  CHECK_CL(clEnqueueAcquireGLObjects(m_globals.cmdQueue, 1, &tempLDRBuff, 0, 0, 0));

  cl_kernel colorKern = m_progs.screen.kernel("RealColorToRGB256");
  
  size_t global_item_size[2] = { size_t(m_width), size_t(m_height) };
  size_t local_item_size[2]  = { 16, 16 };

  RoundBlocks2D(global_item_size, local_item_size);
  
  CHECK_CL(clSetKernelArg(colorKern, 0, sizeof(cl_mem), (void*)&m_screen.color0));
  CHECK_CL(clSetKernelArg(colorKern, 1, sizeof(cl_mem), (void*)&tempLDRBuff));
  CHECK_CL(clSetKernelArg(colorKern, 2, sizeof(cl_int), (void*)&m_width));
  CHECK_CL(clSetKernelArg(colorKern, 3, sizeof(cl_int), (void*)&m_height));
  CHECK_CL(clSetKernelArg(colorKern, 4, sizeof(cl_mem), (void*)&m_globals.cMortonTable));
  CHECK_CL(clSetKernelArg(colorKern, 5, sizeof(cl_float), (void*)&m_globsBuffHeader.varsF[HRT_IMAGE_GAMMA]));
  
  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, colorKern, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
  
  //if (!(m_initFlags & GPU_RT_NOWINDOW))
  //  CHECK_CL(clEnqueueReleaseGLObjects(m_globals.cmdQueue, 1, &tempLDRBuff, 0, 0, 0));
}


void GPUOCLLayer::ConnectEyePass(cl_mem in_rayFlags, cl_mem in_hitPos, cl_mem in_hitNorm, cl_mem in_rayDirOld, cl_mem in_color, int a_bounce, size_t a_size)
{
  runKernel_EyeShadowRays(in_rayFlags, in_hitPos, m_rays.hitFlatNorm, in_rayDirOld,
                          m_rays.shadowRayPos, m_rays.shadowRayDir, a_size);

  runKernel_ShadowTrace(in_rayFlags, m_rays.shadowRayPos, m_rays.shadowRayDir,
                        m_rays.lshadow, a_size);

  runKernel_ProjectSamplesToScreen(in_rayFlags, in_hitPos, in_hitNorm, m_rays.shadowRayDir, in_rayDirOld, in_color,
                                   m_rays.pathShadeColor, m_rays.samZindex, a_size, a_bounce);

  AddContributionToScreen(m_rays.pathShadeColor); // because GPU contributio for LT could be very expensieve (imagine point light)
}

void DebugSaveFuckingGBufferAsManyImages(int a_width, int a_height, const std::vector<GBufferAll>& gbuffer, const wchar_t* a_path);

void GPUOCLLayer::EvalGBuffer(IHRSharedAccumImage* a_pAccumImage, const std::vector<int32_t>& a_instIdByInstId)
{
  // std::vector<float4> data1(m_width*m_height);
  // std::vector<float4> data2(m_width*m_height);

  if (a_pAccumImage == nullptr)
    return;

  if (a_pAccumImage->Header()->gbufferIsEmpty != 1)
    return;
  
  bool locked = false;
  for (int i = 0; i < 20; i++)
  {
    locked = a_pAccumImage->Lock(100);
    if (locked)
      break;
    else
      std::cout << "GPUOCLLayer::EvalGBuffer: trying to lock shared image " << std::endl;
  }

  if (!locked)
    return;

  if (a_pAccumImage->Header()->gbufferIsEmpty != 1) // some other process already have computed gbuffer
  {
    a_pAccumImage->Unlock();
    return;
  }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #TODO: refactor this
  float4* data1 = nullptr;
  float4* data2 = nullptr;
  if (a_pAccumImage->Header()->depth == 4)     // 
  {
    data1 = (float4*)a_pAccumImage->ImageData(2);
    data2 = (float4*)a_pAccumImage->ImageData(3);
  }
  else if (a_pAccumImage->Header()->depth == 3) // 
  {
    data1 = (float4*)a_pAccumImage->ImageData(1);
    data2 = (float4*)a_pAccumImage->ImageData(2);
  }
  else
  {
    std::cerr << "GPUOCLLayer::EvalGBuffer: wrong shared image layers num; num = " << a_pAccumImage->Header()->depth << std::endl;
    a_pAccumImage->Unlock();
    return;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #TODO: refactor this
 

  size_t  bufferSize = m_rays.MEGABLOCKSIZE;
  int32_t lineSize   = m_width * GBUFFER_SAMPLES;

  if (bufferSize % lineSize != 0)
    bufferSize -= (bufferSize % lineSize);

  assert(bufferSize % lineSize == 0);

  int32_t linesPerBlock = int32_t(bufferSize / lineSize);

  for (int32_t line = 0; line < m_height; line += linesPerBlock)
  {
    int32_t yBegin = line;
    int32_t yEnd   = line + linesPerBlock;
    if (yEnd > m_height)
      yEnd = m_height;

    int32_t finalSize = (yEnd - yBegin)*lineSize;

    // (1) generate eye rays
    //
    runKernel_MakeEyeRaysSpp(GBUFFER_SAMPLES, yBegin, finalSize, nullptr,
                             m_rays.rayPos, m_rays.rayDir);
    
    // (2) trace1D with single bounce
    //
    memsetu32(m_rays.rayFlags, 0, finalSize);                                                              // fill flags with zero
    memsetf4 (m_rays.hitMatId, make_float4(0, 0, 0, 0), (finalSize * sizeof(HitMatRef)) / sizeof(float4)); // fill accumulated rays dist with zero
     
    runKernel_Trace     (m_rays.rayPos, m_rays.rayDir, m_rays.hits, finalSize);
    runKernel_ComputeHit(m_rays.rayPos, m_rays.rayDir, finalSize);

    // (3) get compressed samples
    //
    runKernel_GetGBufferSamples(m_rays.rayDir, m_rays.pathAccColor, m_rays.pathShadeColor, GBUFFER_SAMPLES, finalSize);

    // (4) trace some more bounces to get alpha.
    //
    memsetf4(m_rays.pathThoroughput, make_float4(1, 1, 1, 1), finalSize);

    int maxBounce = m_vars.m_varsI[HRT_TRACE_DEPTH];
    if (maxBounce < 2)
      maxBounce = 2;
    for (int bounce = 1; bounce < maxBounce; bounce++)
    {
      runKernel_NextTransparentBounce(m_rays.rayPos, m_rays.rayDir, m_rays.pathThoroughput, finalSize);
      if (bounce == maxBounce - 1)
        break;
      runKernel_Trace                (m_rays.rayPos, m_rays.rayDir, m_rays.hits,            finalSize);
      runKernel_ComputeHit           (m_rays.rayPos, m_rays.rayDir,                         finalSize);
    }

    runKernel_PutAlphaToGBuffer(m_rays.pathThoroughput, m_rays.pathAccColor, finalSize);

    // (5) pass them to the host mem
    //
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.pathAccColor,    CL_FALSE, 0, (finalSize/GBUFFER_SAMPLES)*sizeof(float4), &data1[line*m_width], 0, NULL, NULL));
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.pathShadeColor,  CL_FALSE, 0, (finalSize/GBUFFER_SAMPLES)*sizeof(float4), &data2[line*m_width], 0, NULL, NULL));
  
    for (int x = 0; x < m_width; x++)
    {
      int* pInstId  = (int*)(&data2[line*m_width + x].w);
      int oldInstId = (*pInstId);
      if (oldInstId >= 0 && oldInstId <= a_instIdByInstId.size())
        (*pInstId) = a_instIdByInstId[oldInstId];
    }
  }

  clFinish(m_globals.cmdQueue);

  #pragma omp parallel for
  for (int32_t line = 0; line < m_height; line++)
  {
    for (int x = 0; x < m_width; x++)
    {
      int oldInstId = as_int(data2[line*m_width + x].w);
      if (oldInstId >= 0 && oldInstId < a_instIdByInstId.size())
        data2[line*m_width + x].w = as_float(a_instIdByInstId[oldInstId]);
    }
  }
  
  // //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // std::vector<GBufferAll> gbuffer(m_width*m_height);
  // #pragma omp parallel for
  // for (int i = 0; i < int(gbuffer.size()); i++)
  // {
  //   GBufferAll all;
  //   all.data1 = unpackGBuffer1(data1[i]);
  //   all.data2 = unpackGBuffer2(data2[i]);
  //   gbuffer[i] = all;
  // }
  // DebugSaveFuckingGBufferAsManyImages(m_width, m_height, gbuffer, L"gbufferout");
  // //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (a_pAccumImage != nullptr)
  {
    a_pAccumImage->Header()->gbufferIsEmpty = 0;
    a_pAccumImage->Unlock();
  }
}

std::vector<int> GPUOCLLayer::MakeAllPixelsList()
{
  std::vector<int> allPixels(m_width*m_height);

  //#pragma omp parallel for
  //for(int y=0;y<m_height;y++)
  //{
  //  for(int x=0;x<m_width;x++)
  //    allPixels[y*m_width+x] = packXY1616(x,y);
  //}
  //return allPixels;
  
  const int TILE_SIZE = 32;

  const int maxXRounded = (int(m_width) / int(TILE_SIZE)) * TILE_SIZE;
  const int maxYRounded = (int(m_height) / int(TILE_SIZE)) * TILE_SIZE;

  int top = 0;

  for(int ty=0; ty<maxYRounded; ty+=TILE_SIZE)
  {
    for(int tx=0; tx<maxXRounded; tx+=TILE_SIZE)
    { 
      for(int y1=0;y1<TILE_SIZE;y1++)
      {
        const int y = ty + y1;
        for(int x1=0;x1<TILE_SIZE;x1++)
        {
          const int x = tx + x1;
          allPixels[top + y1*TILE_SIZE + x1] = packXY1616(x,y);  // ZIndexHost(x1,y1)
        }
      }

      top += (TILE_SIZE*TILE_SIZE);
    }
  }

  // push borders
  //
  const int remX = m_width  - maxXRounded;
  const int remY = m_height - maxYRounded;

  for(int y = 0; y < m_height; y++)
  {
    for(int x = maxXRounded; x < m_width; x++)
    {
      allPixels[top] = packXY1616(x,y);
      top++;
    }
  }

  for(int x=0;x<maxXRounded;x++)
  {
    for(int y = maxYRounded; y < m_height; y++)
    {
      allPixels[top] = packXY1616(x,y);
      top++;
    }
  }

  assert(top == m_width*m_height);

  return allPixels;
}

void GPUOCLLayer::RunProductionSamplingMode()
{
  std::cout << "ProductionSamplingMode begin" << std::endl; std::cout.flush();
  
  Timer timer(true); 

  if(m_screen.color0CPU.size() != m_width*m_height)
  {
    m_screen.color0CPU.resize(m_width*m_height);
    memset(m_screen.color0CPU.data(), 0, m_screen.color0CPU.size()*sizeof(float4));
  }

  // (1) create pixels list
  //
  std::vector<int> allPixels = MakeAllPixelsList();

  const int numPasses     = int( int64_t(m_width*m_height)*int64_t(PMPIX_SAMPLES) / int64_t(GetRayBuffSize()) );
  const int pixelsPerPass = GetRayBuffSize() / PMPIX_SAMPLES;

  cl_int ciErr1 = CL_SUCCESS;

  cl_mem pixCoordGPU = clCreateBuffer(m_globals.ctx, CL_MEM_READ_ONLY,  pixelsPerPass*sizeof(int),    nullptr, &ciErr1);
  cl_mem pixColorGPU = clCreateBuffer(m_globals.ctx, CL_MEM_WRITE_ONLY, pixelsPerPass*sizeof(float4), nullptr, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in clCreateBuffer, RunProductionSamplingMode");

  std::vector<float4> pixColors(pixelsPerPass);

  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, pixCoordGPU, CL_TRUE, 0, // CL_FALSECL_TRUE 
                                pixelsPerPass*sizeof(int), (void*)(allPixels.data() + 0), 0, NULL, NULL));

  int currPos = 0;
  bool earlyExit = false;
  for(int pass = 0; pass < numPasses; pass++)
  { 
    if(m_pExternalImage != nullptr)
    {
      bool q1 = false, q2 = false;
      int maxSamplesPerPixel = 0;

      if(m_pExternalImage != nullptr)
      {
        maxSamplesPerPixel = m_vars.m_varsI[HRT_MAX_SAMPLES_PER_PIXEL];
        auto pHeader = m_pExternalImage->Header();
        std::string msg(m_pExternalImage->MessageSendData());
        q1 = (pHeader->spp >= maxSamplesPerPixel);
        q2 = (msg.find("exitnow") != std::string::npos);
      }

      if(q1 || q2) // to quit immediately
      {
        m_sppDone    = maxSamplesPerPixel;
        m_sppContrib = maxSamplesPerPixel;
        earlyExit    = true;
        break;
      }
    }

    //std::cerr << "g_immediateExit = " << g_immediateExit << std::endl; 

    // (2) take a part of list and put it to the GPU 
    //
    //CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, pixCoordGPU, CL_TRUE, 0, 
      //                            pixelsPerPass*sizeof(int), (void*)(allPixels.data() + currPos), 0, NULL, NULL));

    // (3) generate PMPIX_SAMPLES rays per each pixel 
    //

	const int pixelsDone       = pass * pixelsPerPass;
	const int pixelsInThisPass = (pixelsDone + pixelsPerPass <= allPixels.size()) ? pixelsPerPass : int(allPixels.size() - pixelsDone);
    const int finalSize        = PMPIX_SAMPLES*pixelsInThisPass;

    runKernel_MakeEyeRaysSpp(PMPIX_SAMPLES, 0, finalSize, pixCoordGPU,
                             m_rays.rayPos, m_rays.rayDir);

    // (4) trace rays/paths
    //
    runKernel_ClearAllInternalTempBuffers(finalSize);
    trace1D(m_rays.rayPos, m_rays.rayDir, m_rays.pathAccColor, finalSize);
    runKernel_GetShadowToAlpha(m_rays.pathAccColor, m_rays.pathShadow8B, finalSize);

    // (5) average colors
    //
    runKernel_ReductionFloat4Average(m_rays.pathAccColor, pixColorGPU, finalSize, PMPIX_SAMPLES);

    // (6) copy resulting colors to the CPU and add them to the image
    //
    CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, pixColorGPU, CL_TRUE, 0, 
                                 pixelsInThisPass*sizeof(float4), pixColors.data(), 0, NULL, NULL));
    
    if(pass < numPasses-1) // copy next pixels portion asynchronious
    {
      CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, pixCoordGPU, CL_FALSE, 0, 
                                    pixelsInThisPass*sizeof(int), (void*)(allPixels.data() + currPos + pixelsPerPass), 0, NULL, NULL));
      clFlush(m_globals.cmdQueue);
    }

    const float multf = float(PMPIX_SAMPLES);
    for(int pixId = 0; pixId < pixelsInThisPass; pixId++) // contribute to image here
    {
      const int pixelPacked = allPixels[currPos + pixId];
      const int x           = (pixelPacked & 0x0000FFFF);
      const int y           = (pixelPacked & 0xFFFF0000) >> 16;
      m_screen.color0CPU[y*m_width + x] += (pixColors[pixId]*multf); 
    }

    currPos += pixelsPerPass;
    if(pass % 16 == 0)
    {
      std::cout << "production rendering: " << 100.0f*float(pass)/float(numPasses) << "% \r";
      std::cout.flush();
    }
  } // for

  m_globals.m_passNumberQMC += PMPIX_SAMPLES;

  std::cout << std::endl;

  clReleaseMemObject(pixCoordGPU); pixCoordGPU = nullptr;
  clReleaseMemObject(pixColorGPU); pixColorGPU = nullptr;

  m_spp        += PMPIX_SAMPLES;
  m_passNumber += 2; // just for GetLDRImage works correctly it have to be not 0, see pipelined copy for common pt ... ;
  
  const float renderingTime = timer.getElapsed();
  const int maxSamplesPerPixel = m_vars.m_varsI[HRT_MAX_SAMPLES_PER_PIXEL];

  std::cout << "ProductionSamplingMode end, time = " << renderingTime << "s" << std::endl; std::cout.flush();

  if(m_pExternalImage != nullptr && !earlyExit)
  {
    float4* resultPtr = (float4*)m_pExternalImage->ImageData(0);
    const int width   = m_pExternalImage->Header()->width;
    const int height  = m_pExternalImage->Header()->height;

    const bool lockSuccess = m_pExternalImage->Lock(1000); // can wait 1s for success lock
    
    if (lockSuccess)
    {
      const int size = m_width*m_height;

      #pragma omp parallel for
      for(int i=0;i<size;i++)
      {
        const float4 color = m_screen.color0CPU[i];
        resultPtr[i] += color;
        m_screen.color0CPU[i] = float4(0,0,0,0);
      }

      m_pExternalImage->Header()->counterRcv++;
      m_pExternalImage->Header()->spp += PMPIX_SAMPLES;
      m_sppContrib                    += PMPIX_SAMPLES;
        
      m_pExternalImage->Unlock();
      
      //std::cerr << "m_sppContrib        = " << m_sppContrib << std::endl;
      //std::cerr << "HRT_CONTRIB_SAMPLES = " << m_vars.m_varsI[HRT_CONTRIB_SAMPLES] << std::endl;
      //std::cerr << "flags.prod.mode     = " << (m_vars.m_flags & HRT_PRODUCTION_IMAGE_SAMPLING) << std::endl;

      if(m_vars.m_varsI[HRT_BOX_MODE_ON] == 1 && m_sppContrib >= m_vars.m_varsI[HRT_CONTRIB_SAMPLES])  // to quit immediately
        exit(0);
    }
  
    m_sppDone += PMPIX_SAMPLES;
    
    if(m_pExternalImage->Header()->spp >= maxSamplesPerPixel) // to quit immediately
    {
      m_sppDone    = maxSamplesPerPixel;
      m_sppContrib = maxSamplesPerPixel;
    }
  }

}

void GPUOCLLayer::TraceSBDPTPass(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size)
{
  int maxBounce   = 3;

  for (int bounce = 0; bounce < maxBounce; bounce++)
  {
    runKernel_Trace(a_rpos, a_rdir, m_rays.hits, a_size);
    runKernel_ComputeHit(a_rpos, a_rdir, a_size);

    runKernel_HitEnvOrLight(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, bounce, a_size);
    
    runKernel_NextBounce(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, a_size);
  }

}

void GPUOCLLayer::BeginTracingPass()
{
  m_timer.start();

  if (m_vars.m_flags & HRT_ENABLE_MMLT)                 // SBDPT or MMLT pass
  {
    // (1) EyeSample
    //
    m_raysWasSorted = false;
    runKernel_MakeEyeRays(m_rays.rayPos, m_rays.rayDir, m_rays.samZindex, m_rays.MEGABLOCKSIZE, m_passNumberForQMC);

    // (2) trace; 
    //
    TraceSBDPTPass(m_rays.rayPos, m_rays.rayDir, m_rays.pathAccColor, m_rays.MEGABLOCKSIZE);

    // (3) Connect
    //

    // (4) Contrib to screen
    //
    AddContributionToScreen(m_rays.pathAccColor);

  }
  else if((m_vars.m_flags & HRT_PRODUCTION_IMAGE_SAMPLING) != 0 && (m_vars.m_flags & HRT_UNIFIED_IMAGE_SAMPLING) != 0)
  {
    RunProductionSamplingMode();
  }
  else if (m_vars.m_flags & HRT_UNIFIED_IMAGE_SAMPLING) // PT or LT pass
  {
    // (1) Generate random rays and generate multiple references via Z-index
    //
    if (m_vars.m_flags & HRT_FORWARD_TRACING)
    {
      runKernel_MakeLightRays(m_rays.rayPos, m_rays.rayDir, m_rays.pathAccColor, m_rays.MEGABLOCKSIZE);

      if ((m_vars.m_flags & HRT_DRAW_LIGHT_LT) ) 
        ConnectEyePass(m_rays.rayFlags, m_rays.lsam1, m_rays.hitNormUncompressed, nullptr, m_rays.pathAccColor, -1, m_rays.MEGABLOCKSIZE);
    }
    else
    {
      m_raysWasSorted = false;
      runKernel_MakeEyeRays(m_rays.rayPos, m_rays.rayDir, m_rays.samZindex, m_rays.MEGABLOCKSIZE, m_passNumberForQMC);
    }

    // (2) Compute sample colors
    //
    trace1D(m_rays.rayPos, m_rays.rayDir, m_rays.pathAccColor, m_rays.MEGABLOCKSIZE);

    // (3) accumulate colors
    //
    if ((m_vars.m_flags & HRT_FORWARD_TRACING) == 0)
      AddContributionToScreen(m_rays.pathAccColor);
    
  }
  else if(!m_screen.m_cpuFrameBuffer)
  { 
    DrawNormals();
  }
}


void GPUOCLLayer::AddContributionToScreen(cl_mem& in_color)
{
  if (m_screen.m_cpuFrameBuffer)
  {
    float4* resultPtr = nullptr;
    int width         = m_width;
    int height        = m_height;

    if (m_pExternalImage != nullptr)
    {
      resultPtr = (float4*)m_pExternalImage->ImageData(0);
      width     = m_pExternalImage->Header()->width;
      height    = m_pExternalImage->Header()->height;
    }
    else
      resultPtr = &m_screen.color0CPU[0];

    AddContributionToScreenCPU(in_color, m_rays.samZindex, int(m_rays.MEGABLOCKSIZE), width, height,
                               resultPtr);
  }
  else
    AddContributionToScreenGPU(in_color, m_rays.samZindex, int(m_rays.MEGABLOCKSIZE), m_width, m_height, m_passNumber,
                               m_screen.color0, m_screen.pbo);

  m_passNumber++;
}

/**
\brief Add contribution 
\param out_color  - out float4 image of size a_width*a_height 
\param colors     - in float4 array of size a_size
\param a_size     - array size
\param a_width    - image width
\param a_height   - image height

*/
void AddSamplesContribution(float4* out_color, const float4* colors, int a_size, int a_width, int a_height)
{
  for (int i = 0; i < a_size; i++)
  {
    const float4 color    = colors[i];
    const int packedIndex = as_int(color.w);
    const int x           = (packedIndex & 0x0000FFFF);
    const int y           = (packedIndex & 0xFFFF0000) >> 16;
    const int offset      = y*a_width + x;

    if (x >= 0 && y >= 0 && x < a_width && y < a_height)
    {
      out_color[offset].x += color.x; 
      out_color[offset].y += color.y; 
      out_color[offset].z += color.z; 
    }
  }
}

/**
\brief Add contribution with storing shadows in the fourth channel
\param out_color  - out float4 image of size a_width*a_height
\param colors     - in float4 array of size a_size
\param shadows    - in cl_uint8 array of compressed shadow value
\param a_size     - array size
\param a_width    - image width
\param a_height   - image height

*/
void AddSamplesContributionS(float4* out_color, const float4* colors, const unsigned char* shadows, int a_size, int a_width, int a_height)
{
  const float multInv = 1.0f / 255.0f;

  for (int i = 0; i < a_size; i++)
  {
    const float4 color    = colors[i];
    const auto   shad     = shadows[i];

    const int packedIndex = as_int(color.w);
    const int x           = (packedIndex & 0x0000FFFF);
    const int y           = (packedIndex & 0xFFFF0000) >> 16;
    const int offset      = y * a_width + x;

    if (x >= 0 && y >= 0 && x < a_width && y < a_height)
    {
      out_color[offset].x += color.x;
      out_color[offset].y += color.y;
      out_color[offset].z += color.z;
      out_color[offset].w += multInv * float(shad);
    }
  }
}

void GPUOCLLayer::AddContributionToScreenCPU(cl_mem& in_color, cl_mem in_indices, int a_size, int a_width, int a_height, float4* out_color)
{
  // (1) compute compressed index in color.w; use runKernel_MakeEyeRaysAndClearUnified for that task if CPU FB is enabled!!!
  //
  size_t szLocalWorkSize = 256;
  cl_int iNumElements    = cl_int(a_size);
  size_t size            = roundBlocks(size_t(a_size), int(szLocalWorkSize));

  if ((m_vars.m_flags & HRT_FORWARD_TRACING) == 0) // lt already pack index to color, so, don't do that again!!!
  {
    cl_kernel kern = m_progs.screen.kernel("PackIndexToColorW");

    CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&m_rays.packedXY));
    CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&in_color));
    CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&iNumElements));
    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &size, &szLocalWorkSize, 0, NULL, NULL));
  }

  clFlush(m_globals.cmdQueue);

  const bool ltPassOfIBPT = (m_vars.m_flags & HRT_3WAY_MIS_WEIGHTS) && (m_vars.m_flags & HRT_FORWARD_TRACING); 

  Timer copyTimer(true);
  const bool measureTime = false;

  float timeCopy    = 0.0f;
  float timeContrib = 0.0f;

  // (2) sync copy of data (sync asyncronious call in future, pin pong) and eval contribution 
  //
  if (m_passNumber != 0)
  {
    clEnqueueCopyBuffer(m_globals.cmdQueueDevToHost, m_rays.pathAuxColor, m_rays.pathAuxColorCPU, 0, 0, a_size * sizeof(float4), 0, nullptr, nullptr);
    if (m_storeShadowInAlphaChannel)
      clEnqueueCopyBuffer(m_globals.cmdQueueDevToHost, m_rays.pathShadow8BAux, m_rays.pathShadow8BAuxCPU, 0, 0, a_size * sizeof(float4), 0, nullptr, nullptr);

    cl_int ciErr1  = 0;
    float4* colors = (float4*)clEnqueueMapBuffer(m_globals.cmdQueueDevToHost, m_rays.pathAuxColorCPU, CL_TRUE, CL_MAP_READ, 0, a_size * sizeof(float4), 0, 0, 0, &ciErr1);

    cl_uint8* shadows = nullptr;
    if (m_storeShadowInAlphaChannel)
      shadows = (cl_uint8*)( clEnqueueMapBuffer(m_globals.cmdQueueDevToHost, m_rays.pathShadow8BAuxCPU, CL_TRUE, CL_MAP_READ, 0, a_size * sizeof(cl_uint8), 0, 0, 0, &ciErr1) );


    if (measureTime)
    {
      //clFinish(m_globals.cmdQueueDevToHost);
      timeCopy = copyTimer.getElapsed();
    }
  
    const float contribSPP = float(double(a_size) / double(a_width*a_height));
    
    bool lockSuccess = (m_pExternalImage == nullptr);
    if (m_pExternalImage != nullptr)
      lockSuccess = m_pExternalImage->Lock(250); // can wait 250 ms for success lock
    
    if (lockSuccess)
    {
      if (m_storeShadowInAlphaChannel)
        AddSamplesContributionS(out_color, colors, (const unsigned char*)shadows, int(size), a_width, a_height);
      else
        AddSamplesContribution(out_color, colors, int(size), a_width, a_height);

      if (m_pExternalImage != nullptr) //#TODO: if ((m_vars.m_flags & HRT_FORWARD_TRACING) == 0) IT IS DIFFERENT FOR LT !!!!!!!!!!
      {
        if (!ltPassOfIBPT) // don't update counters if this is only first pass of two-pass IBPT
        {
          m_pExternalImage->Header()->counterRcv++;
          m_pExternalImage->Header()->spp += contribSPP;
          m_sppContrib                    += contribSPP;
        }
        m_pExternalImage->Unlock();
      }
    }
  
    m_sppDone += contribSPP;

    if (measureTime && lockSuccess)
    {
      timeContrib = copyTimer.getElapsed();
      std::cout << "time copy    = " << timeCopy*1000.0f << std::endl;
      std::cout << "time contrib = " << (timeContrib - timeCopy)*100.0f << std::endl;
    }

    clEnqueueUnmapMemObject(m_globals.cmdQueueDevToHost, m_rays.pathAuxColorCPU, colors, 0, 0, 0);
    if (m_storeShadowInAlphaChannel)
      clEnqueueUnmapMemObject(m_globals.cmdQueueDevToHost, m_rays.pathShadow8BAuxCPU, shadows, 0, 0, 0);
  }

  clFinish(m_globals.cmdQueueDevToHost);
  clFinish(m_globals.cmdQueue);


  if (measureTime)
  {
    std::cout << "time total   = " << copyTimer.getElapsed()*1000.0f << std::endl;
    std::cout << std::endl;
  }

  // (3) swap color and shadow8 buffers
  //
  {
    cl_mem temp         = m_rays.pathAuxColor;
    m_rays.pathAuxColor = in_color;
    in_color            = temp;

    temp                   = m_rays.pathShadow8BAux;
    m_rays.pathShadow8BAux = m_rays.pathShadow8B;
    m_rays.pathShadow8B    = temp;
  }
}

void GPUOCLLayer::EndTracingPass()
{
  if (!m_screen.m_cpuFrameBuffer)
  {
    CHECK_CL(clFinish(m_globals.cmdQueue));
  }

  if (m_vars.m_flags & HRT_UNIFIED_IMAGE_SAMPLING)
  {
    m_spp += float(double(m_rays.MEGABLOCKSIZE) / double(m_width*m_height));
    m_passNumberForQMC++;

    const float time = m_timer.getElapsed();
    if (m_passNumberForQMC % 4 == 0 && m_passNumberForQMC > 0)
    {
      const float halfIfIBPT = (m_vars.m_flags & HRT_3WAY_MIS_WEIGHTS) ? 0.5f : 1.0f;

      auto precOld = std::cout.precision(2);
      std::cout << "spp =\t" << int(m_spp) << "\tspeed = " << halfIfIBPT * float(m_rays.MEGABLOCKSIZE) / (1e6f*time) << " M(samples)/s         \r";
      std::cout.precision(precOld);
      std::cout.flush();
    }
  }
  else
  {
    m_spp              = 0;
    m_passNumberForQMC = 0;
  }

}

void GPUOCLLayer::CopyForConnectEye(cl_mem in_flags,  cl_mem in_raydir,  cl_mem in_color,
                                           cl_mem out_flags, cl_mem out_raydir, cl_mem out_color, size_t a_size)
{
  cl_kernel kernX      = m_progs.lightp.kernel("CopyAndPackForConnectEye");

  size_t localWorkSize = 256;
  int            isize = int(a_size);
  a_size               = roundBlocks(a_size, int(localWorkSize));
 
  CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&in_flags));
  CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&in_raydir));
  CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_mem), (void*)&in_color));

  CHECK_CL(clSetKernelArg(kernX, 3, sizeof(cl_mem), (void*)&out_flags));
  CHECK_CL(clSetKernelArg(kernX, 4, sizeof(cl_mem), (void*)&out_raydir));
  CHECK_CL(clSetKernelArg(kernX, 5, sizeof(cl_mem), (void*)&out_color));
  CHECK_CL(clSetKernelArg(kernX, 6, sizeof(cl_int), (void*)&isize));

  CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  waitIfDebug(__FILE__, __LINE__);
}

int GPUOCLLayer::CountNumActiveThreads(cl_mem a_rayFlags, size_t a_size)
{
  int zero = 0;
  CHECK_CL(clEnqueueWriteBuffer(m_globals.cmdQueue, m_rays.atomicCounterMem, CL_TRUE, 0, 
                                sizeof(int), &zero, 0, NULL, NULL));

  {
    cl_kernel kern = m_progs.screen.kernel("CountNumLiveThreads");
  
    size_t szLocalWorkSize = 256;
    cl_int iNumElements    = cl_int(a_size);
    a_size                 = roundBlocks(a_size, int(szLocalWorkSize));
  
    CHECK_CL(clSetKernelArg(kern, 0, sizeof(cl_mem),  (void*)&m_rays.atomicCounterMem));
    CHECK_CL(clSetKernelArg(kern, 1, sizeof(cl_mem),  (void*)&a_rayFlags));
    CHECK_CL(clSetKernelArg(kern, 2, sizeof(cl_int),  (void*)&iNumElements));
  
    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kern, 1, NULL, &a_size, &szLocalWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }

  int counter = 0;
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.atomicCounterMem, CL_TRUE, 0, 
                              sizeof(int), &counter, 0, NULL, NULL));

  return counter;
}

void GPUOCLLayer::trace1D(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size)
{
  // trace rays
  //
  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    m_timer.start();
  }

  float timeForSample    = 0.0f;
  float timeForBounce    = 0.0f;
  float timeForTrace     = 0.0f;
  float timeBeforeShadow = 0.0f;
  float timeForShadow    = 0.0f;
  float timeStart        = 0.0f;

  float timeNextBounceStart = 0.0f;
  float timeForNextBounce   = 0.0f;

  float timeForHitStart = 0.0f;
  float timeForHit      = 0.0f;

  int measureBounce = m_vars.m_varsI[HRT_MEASURE_RAYS_TYPE];
  int maxBounce     = m_vars.m_varsI[HRT_TRACE_DEPTH];

  for (int bounce = 0; bounce < maxBounce; bounce++)
  {
    const bool measureThisBounce = (bounce == measureBounce);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeStart = m_timer.getElapsed();
    }

    runKernel_Trace(a_rpos, a_rdir, m_rays.hits, a_size);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeForHitStart = m_timer.getElapsed();
      timeForTrace    = timeForHitStart - timeStart;
    }

    runKernel_ComputeHit(a_rpos, a_rdir, a_size);

    if ((m_vars.m_flags & HRT_FORWARD_TRACING) == 0)
      runKernel_HitEnvOrLight(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, bounce, a_size);

    if (FORCE_DRAW_SHADOW && bounce == 1)
    {
      CopyShadowTo(a_outColor, m_rays.MEGABLOCKSIZE);
      break;
    }

    if((m_vars.m_flags & HRT_PRODUCTION_IMAGE_SAMPLING) != 0 && (bounce%2 == 0)) // opt for empty environment rendering.
    {
      if(CountNumActiveThreads(m_rays.rayFlags, a_size) == 0)
        break;
    }

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeBeforeShadow = m_timer.getElapsed();
      timeForHit       = timeBeforeShadow - timeForHitStart;
    }

    if (m_vars.m_flags & HRT_FORWARD_TRACING)
    {
      // postpone 'ConnectEyePass' call to the end of bounce; 
      // ConnectEyePass(m_rays.rayFlags, m_rays.hitPosNorm, m_rays.hitNormUncompressed, a_rdir, a_outColor, bounce, a_size);
      CopyForConnectEye(m_rays.rayFlags, a_rdir,             a_outColor, 
                        m_rays.oldFlags, m_rays.oldRayDir,   m_rays.oldColor, a_size);
    }
    else if (m_vars.shadePassEnable(bounce))
    {
      ShadePass(a_rpos, a_rdir, m_rays.pathShadeColor, a_size, measureThisBounce);
    }

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeNextBounceStart = m_timer.getElapsed();
      timeForShadow       = timeNextBounceStart - timeBeforeShadow;
    }

    runKernel_NextBounce(m_rays.rayFlags, a_rpos, a_rdir, a_outColor, a_size);

    if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS] && measureThisBounce)
    {
      clFinish(m_globals.cmdQueue);
      timeForBounce     = (m_timer.getElapsed() - timeStart);
      timeForNextBounce = (m_timer.getElapsed() - timeNextBounceStart);
    }

    if (m_vars.m_flags & HRT_FORWARD_TRACING)
    {
      ConnectEyePass(m_rays.oldFlags, m_rays.hitPosNorm, m_rays.hitNormUncompressed, m_rays.oldRayDir, m_rays.oldColor, bounce, a_size);
      if (m_vars.m_flags & HRT_3WAY_MIS_WEIGHTS)
        runKernel_UpdateForwardPdfFor3Way(m_rays.oldFlags, m_rays.oldRayDir, m_rays.rayDir, m_rays.accPdf, a_size);
    }

  }


  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    timeForSample = m_timer.getElapsed();
  }

  m_stat.raysPerSec      = float(a_size) / timeForTrace;
  m_stat.traversalTimeMs = timeForTrace*1000.0f;
  m_stat.sampleTimeMS    = timeForSample*1000.0f;
  m_stat.bounceTimeMS    = timeForBounce*1000.0f;
  //m_stat.shadowTimeMs    = timeForShadow*1000.0f;
  m_stat.evalHitMs       = timeForHit*1000.0f;
  m_stat.nextBounceMs    = timeForNextBounce*1000.0f;

  m_stat.samplesPerSec    = float(a_size) / timeForSample;
  m_stat.traceTimePerCent = int( ((timeForTrace + timeForShadow) / timeForBounce)*100.0f );
  //std::cout << "measureBounce = " << measureBounce << std::endl;
}


void GPUOCLLayer::trace1DPrimaryOnly(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size, size_t a_offset)
{
  cl_kernel kernShowN = m_progs.trace.kernel("ShowNormals");
  cl_kernel kernShowT = m_progs.trace.kernel("ShowTexCoord");
  cl_kernel kernFill  = m_progs.trace.kernel("ColorIndexTriangles");

  //cl_kernel kern = m_progs.screen.kernel("FillColorTest");

  size_t localWorkSize = 256;
  int    isize   = int(a_size);
  int    ioffset = int(a_offset);

  // trace rays
  //
  memsetu32(m_rays.rayFlags, 0, a_size);                                         // fill flags with zero data
  memsetf4(a_outColor, make_float4(0, 0, 0, 0), a_size, a_offset);               // fill initial out color with black

  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    m_timer.start();
  }
  
  runKernel_Trace(a_rpos, a_rdir, m_rays.hits, a_size);
  
  if (m_vars.m_varsI[HRT_ENABLE_MRAYS_COUNTERS])
  {
    clFinish(m_globals.cmdQueue);
    m_stat.raysPerSec = float(a_size) / m_timer.getElapsed();
  }
  
  runKernel_ComputeHit(a_rpos, a_rdir, a_size, true);

  //
  //
  if (true)
  {
    CHECK_CL(clSetKernelArg(kernShowN, 0, sizeof(cl_mem), (void*)&m_rays.hitPosNorm));
    CHECK_CL(clSetKernelArg(kernShowN, 1, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernShowN, 2, sizeof(cl_int), (void*)&isize));  
    CHECK_CL(clSetKernelArg(kernShowN, 3, sizeof(cl_int), (void*)&ioffset));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernShowN, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
  else if (false)
  {
    CHECK_CL(clSetKernelArg(kernShowT, 0, sizeof(cl_mem), (void*)&m_rays.hitTexCoord));
    CHECK_CL(clSetKernelArg(kernShowT, 1, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernShowT, 2, sizeof(cl_int), (void*)&isize));
    CHECK_CL(clSetKernelArg(kernShowT, 3, sizeof(cl_int), (void*)&ioffset));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernShowT, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }
  else
  {
    CHECK_CL(clSetKernelArg(kernFill, 0, sizeof(cl_mem), (void*)&m_rays.hits));
    CHECK_CL(clSetKernelArg(kernFill, 1, sizeof(cl_mem), (void*)&a_outColor));
    CHECK_CL(clSetKernelArg(kernFill, 2, sizeof(cl_int), (void*)&isize));
    CHECK_CL(clSetKernelArg(kernFill, 3, sizeof(cl_int), (void*)&ioffset));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernFill, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
    waitIfDebug(__FILE__, __LINE__);
  }

}


IHWLayer* CreateOclImpl(int w, int h, int a_flags, int a_deviceId) 
{
  return new GPUOCLLayer(w, h, a_flags, a_deviceId);
}








#ifdef WIN32
#include "../../HydraAPI/clew/clew.h"
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>

#include <string.h>

#include "clHelper.h"

static std::string ToString(int i)
{
  std::stringstream out;
  out << i;
  return out.str();
}

static void RunTimeError(const char* file, int line, std::string msg)
{
  throw std::runtime_error(std::string("Run time error at ") + file + std::string(", line ") + ToString(line) + ": " + msg);
}


void checkCLFun(cl_int cErr, char* file, int line)
{
  if (cErr != CL_SUCCESS)
  {
    const char* err = getOpenCLErrorString(cErr);
    RunTimeError(file, line, err);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string CutPathToFolder(const std::string& a_pathToFile)
{
  size_t posOfLastSlash = a_pathToFile.find_last_of('\\');

  if (posOfLastSlash == std::string::npos)
    posOfLastSlash = a_pathToFile.find_last_of('/');

  if (posOfLastSlash == std::string::npos)
    return "";

  return a_pathToFile.substr(0, posOfLastSlash);
}

std::string GetFileNameFromIncludeExpr(const std::string& a_str, size_t a_pos)
{
  size_t first = a_str.find_first_of('"', a_pos);
  size_t second = a_str.find_first_of('"', first + 1);

  if (first == std::string::npos || second == std::string::npos)
  {
    first = a_str.find_first_of('<', a_pos);
    second = a_str.find_first_of('>', first + 1);
  }

  if (first == std::string::npos || second == std::string::npos)
    return "";

  return a_str.substr(first + 1, second - first - 1);
}


std::string HydraInstallPath();

size_t ReplaceIncludeWithFile(std::string& a_str, size_t a_pos, const std::string& a_fileName)
{
  std::ifstream sourceFile(a_fileName.c_str());

  if (!sourceFile.is_open())
    sourceFile.open(HydraInstallPath() + a_fileName);

  if (!sourceFile.is_open())
  {
    std::cerr << "[helperProg]: " << "can't load shader include " << a_fileName.c_str() << std::endl;
    return a_pos + 1; // proceed next include
  }

  std::string source;

  sourceFile.seekg(0, std::ios::end);
  source.reserve(sourceFile.tellg());
  sourceFile.seekg(0, std::ios::beg);
  source.assign((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());

  size_t endOfLine = a_str.find_first_of('\n', a_pos);

  a_str = a_str.substr(0, a_pos) + source + a_str.substr(endOfLine, a_str.size());
  return 0; // start search include files with the beggining
}

void IncludeFiles(std::string& a_shader, const std::string& a_pathToFile)
{
  size_t found = 0;

  while (true)
  {
    found = a_shader.find("#include", found);
    bool commented = false;

    // look back and see whether it was commented or nor
    //
    if (found != std::string::npos)
    {
      for (int i = 1; i<10; i++)
      {
        if (int(found) - i < 0)
          break;

        if (a_shader[found - i] == '\n')
          break;

        if (a_shader[found - i] == '/' && a_shader[found - i - 1] == '/')
        {
          commented = true;
          break;
        }
      }
    }

    if (found != std::string::npos && !commented)
    {
      std::string pathToFolder = CutPathToFolder(a_pathToFile);
      std::string fileName     = GetFileNameFromIncludeExpr(a_shader, found);

      found = ReplaceIncludeWithFile(a_shader, found, pathToFolder + "/" + fileName);
    }
    else if (commented)
    {
      found++;
    }
    else
      break;
  };
}

void LoadTextFromFileSimple(const std::string& a_fileName, std::string& a_shaderSource)
{
  std::ifstream vertSourceFile(a_fileName.c_str());

  if (!vertSourceFile.is_open())
    throw std::runtime_error(std::string("cant open file: ") + a_fileName);

  vertSourceFile.seekg(0, std::ios::end);
  a_shaderSource.reserve(vertSourceFile.tellg());

  vertSourceFile.seekg(0, std::ios::beg);
  a_shaderSource.assign((std::istreambuf_iterator<char>(vertSourceFile)), std::istreambuf_iterator<char>());
}

void LoadTextFromFile(const std::string& a_fileName, std::string& a_shaderSource)
{
  LoadTextFromFileSimple(a_fileName, a_shaderSource);
  IncludeFiles(a_shaderSource, a_fileName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int clglSharingIsSupported(cl_device_id device_id)
{
  size_t ext_size = 4096;
  char* ext_string = (char*)malloc(ext_size);
  cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, ext_size, ext_string, &ext_size);

  // Search for GL support in extension string (space delimited)
  int glSupported = IsExtensionSupported(CL_GL_SHARING_EXT, ext_string, ext_size);
  //std::cout << "[cl_core]: GL sharing support " << glSupported << std::endl;
  free(ext_string);

  return glSupported;

}

char *strnstr(const char *haystack, const char *needle, size_t len)
{
  int i;
  size_t needle_len;

  /* segfault here if needle is not NULL terminated */
  if (0 == (needle_len = strlen(needle)))
    return (char *)haystack;

  /* Limit the search if haystack is shorter than 'len' */
  len = strnlen(haystack, len);

  for (i = 0; i<(int)(len - needle_len); i++)
  {
    if ((haystack[0] == needle[0]) &&
      (0 == strncmp(haystack, needle, needle_len)))
      return (char *)haystack;

    haystack++;
  }
  return NULL;
}


int IsExtensionSupported(const char* support_str, const char* ext_string, size_t ext_buffer_size)
{
  size_t offset = 0;
  const char* space_substr = strnstr(ext_string + offset, " ", ext_buffer_size - offset);
  size_t space_pos = space_substr ? space_substr - ext_string : 0;
  while (space_pos < ext_buffer_size)
  {
    if (strncmp(support_str, ext_string + offset, space_pos) == 0)
    {
      // Device supports requested extension!
      //printf("[cl_core]: Found extension support �%s�!\n", support_str);
      return 1;
    }
    // Keep searching -- skip to next token string
    offset = space_pos + 1;
    space_substr = strnstr(ext_string + offset, " ", ext_buffer_size - offset);
    space_pos = space_substr ? space_substr - ext_string : 0;
  }
  printf("Warning: Extension not supported �%s�!\n", support_str);
  return 0;
}


const char * getOpenCLErrorString(cl_int err)
{
  switch (err)
  {
  case CL_SUCCESS: return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE: return "CL_MAP_FAILURE";

  case CL_INVALID_VALUE: return "CL_INVALID_VALUE ";
  case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE ";
  case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM ";
  case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE ";
  case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT ";
  case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES ";
  case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE ";
  case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR ";
  case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT ";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ";
  case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE ";
  case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER ";
  case CL_INVALID_BINARY: return "CL_INVALID_BINARY ";
  case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS ";
  case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM ";
  case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE ";
  case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME ";
  case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION ";
  case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL ";
  case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX ";
  case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE ";
  case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE ";
  case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS ";
  case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION ";
  case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE ";
  case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE ";
  case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET ";
  case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST ";
  case CL_INVALID_EVENT: return "CL_INVALID_EVENT ";
  case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION ";
  case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT ";
  case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE ";
  case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL ";
  case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE ";

  default: return "Unknown OpenCL error";
  }
}

// begin XTEA: http://en.wikipedia.org/wiki/XTEA

typedef unsigned int uint32_t;

void encipher(unsigned int num_rounds, uint32_t v[2], uint32_t const key[4])  // take 64 bits of data in v[0] and v[1] and 128 bits of key[0] - key[3] 
{
  unsigned int i;
  uint32_t v0 = v[0], v1 = v[1], sum = 0, delta = 0x9E3779B9;
  for (i = 0; i < num_rounds; i++) 
  {
    v0 += (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + key[sum & 3]);
    sum += delta;
    v1 += (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + key[(sum >> 11) & 3]);
  }

  v[0] = v0; v[1] = v1;
}

// \\ end XTEA

void decipher(unsigned int num_rounds, uint32_t v[2], uint32_t const key[4])  // take 64 bits of data in v[0] and v[1] and 128 bits of key[0] - key[3] 
{
  unsigned int i;
  uint32_t v0 = v[0], v1 = v[1], delta = 0x9E3779B9, sum = delta*num_rounds;
  for (i = 0; i < num_rounds; i++) 
  {
    v1 -= (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + key[(sum >> 11) & 3]);
    sum -= delta;
    v0 -= (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + key[sum & 3]);
  }

  v[0] = v0; v[1] = v1;
}

void SaveAndCrypt(const std::string& computeSource, const std::string& ecrypted)
{
  uint32_t key[4] = { 0x4AF0AB01, 0x12F06B71, 0x254A600, 0x55535690 };

  int buffSize = int(computeSource.size() / 4);

  uint32_t* buffer = new uint32_t[buffSize];
  memcpy(buffer, computeSource.c_str(), computeSource.size());

  for (int offs = 0; offs < buffSize; offs += 2)
    encipher(4, buffer + offs, key);

  std::ofstream fout(ecrypted, std::ios::binary);

  fout.write((const char*)&buffSize, sizeof(int));
  fout.write((const char*)buffer, buffSize*sizeof(int));
  fout.close();

  delete[] buffer;
}

std::string LoadAndDecrypt(const std::string& ecrypted)
{
  uint32_t key[4] = { 0x4AF0AB01, 0x12F06B71, 0x254A600, 0x55535690 };

  std::ifstream fin(ecrypted, std::ios::binary);

  if (!fin.is_open())
    return "";

  int buffSize = 0;
  fin.read((char*)&buffSize, sizeof(int));

  uint32_t* buffer = new uint32_t[buffSize];
  fin.read((char*)buffer, buffSize*sizeof(int));

  for (int offs = 0; offs < buffSize; offs += 2)
    decipher(4, buffer + offs, key);

  std::string resStr = std::string((char*)buffer);
  delete [] buffer;

  for (size_t i = resStr.size() - 32; i < resStr.size(); i++)
    resStr[i] = '\0';

  return resStr;
}


std::vector<unsigned char> LoadShaderFromBinary(const std::string& binPath)
{
  std::vector<unsigned char> data(0);
  std::ifstream fin(binPath.c_str(), std::ios::binary);

  if (!fin.is_open())
    return data;

  int header[4] = {0,0,0,0};

  fin.read((char*)header, sizeof(int) * 4);
  int size = header[0];

  if (size > 0)
  {
    data.resize(size);
    fin.read((char*)&data[0], size);
  }

  return data;
}


CLProgram::CLProgram() : m_refCounter(0)
{
  program   = NULL;
  m_ctx     = 0;
  m_dev     = 0;
  m_lastErr = CL_SUCCESS;
  m_programLength = 0;
}

CLProgram::CLProgram(cl_device_id a_devId, cl_context a_ctx, 
                     const std::string& cs_path, const std::string& options, const std::string& includeFolderPath, 
                     const std::string& encrypted, const std::string& binPath, bool a_saveLog) : m_refCounter(0)
{
  m_ctx = a_ctx;
  m_dev = a_devId;

  std::vector<unsigned char> binShaderCode = LoadShaderFromBinary(binPath);
  bool loadBinarySuccess = false;

  // load from binary
  //
  if (binShaderCode.size() != 0)
  {
    cl_int errNum = 0;
    cl_int binaryStatus;
    size_t binarySize = binShaderCode.size();
    const unsigned char *programBinary = &binShaderCode[0];

    program = clCreateProgramWithBinary(a_ctx, 1, &a_devId, &binarySize, &programBinary, &binaryStatus, &errNum);

    loadBinarySuccess = true;
    if (errNum != CL_SUCCESS)
    {
      std::cerr << "[cl_core]: failed to load program binary" << std::endl;
      loadBinarySuccess = false;
    }

    if (binaryStatus != CL_SUCCESS)
    {
      std::cerr << "[cl_core]: invalid binary for a device" << std::endl;
      loadBinarySuccess = false;
    }

    if (loadBinarySuccess)
      std::cout << "[cl_core]: loaded " + cs_path + " from binary" << std::endl;
  }
  
  // load from text source
  //
  if (!loadBinarySuccess)
  {
    std::string computeSource;
    if (encrypted != "")
    {
      std::string ecryptedPath = cs_path.substr(0, cs_path.size() - 2) + "xx";

      if (encrypted == "crypt")
      {
        LoadTextFromFile(cs_path, computeSource);

        if (computeSource.size() % 8 != 0)
        {
          size_t n = (computeSource.size() / 8) * 8 + 8 - computeSource.size();
          computeSource = computeSource + std::string(n, '\n');
        }

        SaveAndCrypt(computeSource, ecryptedPath);
      }
      else
      {
        computeSource = LoadAndDecrypt(ecryptedPath);  // try to load encrypted shaders

        if (computeSource == "")
          LoadTextFromFileSimple(cs_path, computeSource); // if falied load from source
      }
    }
    else
      LoadTextFromFileSimple(cs_path, computeSource);

    const char* tmpComputeSource = computeSource.c_str();
    m_programLength = computeSource.size();

    program = clCreateProgramWithSource(a_ctx, 1, (const char **)&tmpComputeSource, &m_programLength, &m_lastErr);

    if (m_lastErr != CL_SUCCESS)
      std::cerr << "clCreateProgramWithSource error = " << getOpenCLErrorString(m_lastErr) << std::endl;
  }

  m_lastErr = clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);

  if (m_lastErr != CL_SUCCESS)
  {
    char* buffer = (char*)malloc(204800);
    memset(buffer, 0, 204800);
    size_t len = 204800;
    cl_int err = clGetProgramBuildInfo(program, a_devId, CL_PROGRAM_BUILD_LOG, 204800, buffer, &len);
    if (err != CL_SUCCESS)
      std::cerr << "clGetProgramBuildInfo error = " << getOpenCLErrorString(err) << std::endl;

    std::cerr << "cl program compilation failed!\tfile: " << cs_path.c_str() << std::endl;
    std::cerr << buffer << std::endl;

    free(buffer);
    throw std::runtime_error(std::string("cl program compilation failed!\tfile : ") + cs_path + std::string(", cl error type = ") + getOpenCLErrorString(m_lastErr));
  }
  else if (a_saveLog)
  {
    char* buffer = (char*)malloc(204800);
    memset(buffer, 0, 204800);
    size_t len = 204800;
    cl_int err = clGetProgramBuildInfo(program, a_devId, CL_PROGRAM_BUILD_LOG, 204800, buffer, &len);
    if (err != CL_SUCCESS)
      std::cerr << "clGetProgramBuildInfo error = " << getOpenCLErrorString(err) << std::endl;

    auto nameBegin = cs_path.find_last_of('/') + 1;

    const std::string name = cs_path.substr(nameBegin, cs_path.size() - nameBegin - 3);

    if (name.size() > 0)
    {
      const std::string logName = "z_" + name + "_log.txt";
      std::ofstream fout(logName.c_str());
      fout << buffer << std::endl;
      fout.close();
    }

    free(buffer);
  }

  if (!Link())
    throw std::runtime_error(std::string("cl program build failed!\tfile : ") + cs_path + std::string(", cl error type =  ") + getOpenCLErrorString(m_lastErr));

  m_refCounter++;
}


void CLProgram::saveBinary(const std::string& a_fileName)
{
  cl_uint numDevices = 0;
  cl_int errNum = 0;

  errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "[cl_core]: CLProgram::saveBinary, error querying for number of devices." << std::endl;
    return;
  }

  if (numDevices != 1)
    std::cerr << "[cl_core]: CLProgram::saveBinary, numDevices = " << numDevices << std::endl;

  size_t programBinarySize;

  errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices, &programBinarySize, NULL);
  if (errNum != CL_SUCCESS || programBinarySize == 0)
  {
    std::cerr << "[cl_core]: CLProgram::saveBinary, error querying for program binary sizes." << std::endl;
    return;
  }

  std::vector<unsigned char> programBinData(programBinarySize);
  
  unsigned char* pBegin = &programBinData[0];
  unsigned char** programBinaries = &pBegin;

  errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices, programBinaries, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "[cl_core]: CLProgram::saveBinary, Error querying for program binaries" << std::endl;
    return;
  }

  int iSizeBin = int(programBinData.size());
  int header[4] = { iSizeBin, 0, 0, 0 };

  std::ofstream fout(a_fileName.c_str(), std::ios::binary | std::ios::trunc);
  fout.write((const char*)header, sizeof(int) * 4);
  fout.write((const char*)&programBinData[0], programBinData.size());
  fout.close();

}

cl_kernel CLProgram::kernel(const std::string& name) const
{
  auto p = kernels.find(name);

  if (p == kernels.end())
  {
    cl_kernel ckKernel = clCreateKernel(program, name.c_str(), const_cast<cl_int*>(&m_lastErr));
    
    if (m_lastErr != CL_SUCCESS)
      throw std::runtime_error(std::string("clCreateKernel : ") + getOpenCLErrorString(m_lastErr));
    
    kernels[name] = ckKernel;
    return ckKernel;
  }
  else
    return p->second;
}


CLProgram::~CLProgram()
{
  m_refCounter--;

  if (m_refCounter <= 0) // delete resources
  {
    for (auto p = kernels.begin(); p != kernels.end(); ++p)
    {
      if (p->second)
        clReleaseKernel(p->second);
    }

    clReleaseProgram(program);
  }

}


CLProgram& CLProgram::operator=(const CLProgram& a_prog)
{
  program   = a_prog.program;
  m_ctx     = a_prog.m_ctx;
  m_dev     = a_prog.m_dev;
  m_lastErr = a_prog.m_lastErr;
  kernels   = a_prog.kernels;
  m_programLength = a_prog.m_programLength;

  a_prog.m_refCounter++;
  m_refCounter = 1;

  return *this;
}


//#ifndef CL_PROGRAM_NUM_KERNELS
//#define CL_PROGRAM_NUM_KERNELS 0x1167
//#endif

//#ifndef CL_PROGRAM_KERNEL_NAMES
//#define CL_PROGRAM_KERNEL_NAMES 0x1168
//#endif

bool CLProgram::Link()
{
  return (m_lastErr == CL_SUCCESS);
}


size_t roundWorkGroupSize(size_t a_size, size_t a_blockSize)
{
  if((a_size % a_blockSize) == 0)
    return a_size;

  return a_size + (a_blockSize - (a_size % a_blockSize));
}


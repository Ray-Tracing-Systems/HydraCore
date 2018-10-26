//////////////////////////////////////////////////////////////////
// clHelper.h Author: Vladimir Frolov, 2014, Graphics & Media Lab.
//////////////////////////////////////////////////////////////////
#pragma once
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <string>

#ifdef WIN32
#include "../../HydraAPI/clew/clew.h"
#else
#include <CL/cl.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

int IsExtensionSupported(const char* support_str, const char* ext_string, size_t ext_buffer_size);
const char * getOpenCLErrorString(cl_int err);

void checkCLFun(cl_int cErr, const char* const file, int line);
int clglSharingIsSupported(cl_device_id device_id);

#define CHECK_CL(call) checkCLFun((call), __FILE__, __LINE__);

size_t roundWorkGroupSize(size_t a_size, size_t a_blockSize);


struct CLProgram
{

  CLProgram();
  CLProgram(cl_device_id a_devId, cl_context a_ctx, const std::string cs_path, const std::string options, 
            const std::string includeFolderPath = "", const std::string encryptedBufferPath = "", const std::string binPath = "", bool a_saveLog = false);

  CLProgram(cl_device_id a_devId, cl_context a_ctx, 
            const std::string& computeSource, const std::string options, const std::string includeFolderPath, void* unusedPtr); // load from source

  CLProgram(const CLProgram& a_prog);

  virtual ~CLProgram();
  CLProgram& operator=(const CLProgram& a_prog);

  cl_kernel kernel(const std::string& name) const;

  cl_program program;

  void saveBinary(const std::string& a_fileName);

protected:

  bool Link();

  cl_context   m_ctx;
  cl_device_id m_dev;
  cl_int       m_lastErr;

  mutable size_t m_programLength;
  mutable int    m_refCounter;
  mutable std::map<std::string, cl_kernel> kernels;

};


#include "input.h"
#include <time.h>

Input::Input()
{
  enableOpenGL1 = false;
  noWindow      = false;
  exitStatus    = false;

  runTests      = false;
  inTestsFolder = L"../../HydraAPI/main/";

  //inLibraryPath = "temp/";
  //inLibraryPath = "tests/test_35";
  //inLibraryPath = "tests/test_42";      ///< cornell box with teapot
  
  inLibraryPath = "D:/PROG/HydraAPI/main/tests/test_76";
  //inLibraryPath = "D:/PROG/HydraAPI/main/tests_f/test_004";
  //inLibraryPath = "D:/PROG/HydraAPI/main/tests_f/test_205";
  //inLibraryPath = "C:/[Hydra]/pluginFiles/scenelib";
 
  inDeviceId = 1;
  enableMLT  = false;

  winWidth   = 1024;
  winHeight  = 1024;

  camMoveSpeed     = 2.5f;
  mouseSensitivity = 0.1f;
  saveInterval     = 0.0f;

  // dynamic data
  //
  pathTracingEnabled  = false;
  lightTracingEnabled = false;
  cameraFreeze        = false;
  inSeed              = clock();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
\brief  Read bool from hash map. if no value was found (by key), don't overwrite 'pParam'
\param  a_params - hash map to read from
\param  a_name   - key
\param  pParam   - output parameter. if no value was found (by key), it does not touched.
\return found value or false if nothing was found

*/
static bool ReadBoolCmd(const std::unordered_map<std::string, std::string>& a_params, const std::string& a_name, bool* pParam = nullptr)
{
  const auto p = a_params.find(a_name);
  if (p != a_params.end())
  {
    const int val = atoi(p->second.c_str());
    const bool result = (val > 0);

    if (pParam != nullptr)
      (*pParam) = result;

    return result;
  }
  else
    return false;
}


/**
\brief  Read int from hash map. if no value was found (by key), don't overwrite 'pParam'
\param  a_params - hash map to read from
\param  a_name   - key
\param  pParam   - output parameter. if no value was found (by key), it does not touched.
\return found value or false if nothing was found

*/
static int ReadIntCmd(const std::unordered_map<std::string, std::string>& a_params, const std::string& a_name, int* pParam = nullptr)
{
  const auto p = a_params.find(a_name);
  if (p != a_params.end())
  {
    const int val = atoi(p->second.c_str());
    if (pParam != nullptr)
      (*pParam) = val;
    return val;
  }
  else
    return 0;
}

/**
\brief  Read float from hash map. if no value was found (by key), don't overwrite 'pParam'
\param  a_params - hash map to read from
\param  a_name   - key
\param  pParam   - output parameter. if no value was found (by key), it does not touched.
\return found value or false if nothing was found

*/
static float ReadFloatCmd(const std::unordered_map<std::string, std::string>& a_params, const std::string& a_name, float* pParam = nullptr)
{
  const auto p = a_params.find(a_name);
  if (p != a_params.end())
  {
    const float val = float(atof(p->second.c_str()));
    if (pParam != nullptr)
      (*pParam) = val;
    return val;
  }
  else
    return 0;
}

/**
\brief  Read string from hash map. if no value was found (by key), don't overwrite 'pParam'
\param  a_params - hash map to read from
\param  a_name   - key
\param  pParam   - output parameter. if no value was found (by key), it does not touched.
\return found value or "" if nothing was found

*/
static std::string ReadStringCmd(const std::unordered_map<std::string, std::string>& a_params, const std::string& a_name, std::string* pParam = nullptr)
{
  const auto p = a_params.find(a_name);
  if (p != a_params.end())
  {
    if (pParam != nullptr)
      (*pParam) = p->second;

    return p->second;
  }
  else
    return std::string("");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Input::ParseCommandLineParams(const std::unordered_map<std::string, std::string>& a_params)
{
  ReadBoolCmd(a_params,   "-nowindow",        &noWindow);
  ReadBoolCmd(a_params,   "-enable_mlt",      &enableMLT);
  ReadBoolCmd(a_params,   "-cl_list_devices", &listDevicesAndExit);
  ReadBoolCmd(a_params,   "-alloc_image_b",   &allocInternalImageB);
 
  if (listDevicesAndExit)
    noWindow = true;

  ReadIntCmd (a_params,   "-seed",         &inSeed);
  ReadIntCmd (a_params,   "-cl_device_id", &inDeviceId);
  ReadFloatCmd(a_params,  "-saveinterval", &saveInterval);

  ReadIntCmd(a_params,    "-width",        &winWidth);
  ReadIntCmd(a_params,    "-height",       &winHeight);


  ReadStringCmd(a_params, "-inputlib", &inLibraryPath);
  // ReadStringCmd(a_params, "-imageB",   &inImageBName);
  // ReadStringCmd(a_params, "-mutexB",   &inMutexBName);
  // ReadStringCmd(a_params, "-imageA",   &inImageAName);
  // ReadStringCmd(a_params, "-mutexA",   &inMutexAName);
  
  ReadStringCmd(a_params, "-out",      &outLDRImage); 
  ReadStringCmd(a_params, "-logdir",   &inLogDirCust);


}


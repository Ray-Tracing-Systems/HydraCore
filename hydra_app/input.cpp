#include "input.h"
#include <time.h>
#include <iostream>

extern bool g_hydraApiDisableSceneLoadInfo;

Input::Input()
{
  g_hydraApiDisableSceneLoadInfo = true;
  
  //noWindow      = false;             ///< run 'console_main', else run 'window_main'
  //inLibraryPath = "tests/test_42_beckmann"; ///< cornell box with teapot
  //inLibraryPath = "tests/test_aniso";
  //inLibraryPath = "tests/test_42_with_mirror";  
  //inLibraryPath = "tests/test_42_ggx";
  //inLibraryPath = "tests/test_42";
  //inLibraryPath = "../../Diser/DiffuseReference/01_CornellBoxEmpty_direct/scenelib";
  //inLibraryPath = "tests/test_223_small"; ///< cornell box with sphere
  //inLibraryPath = "tests/test_224";
  //inLibraryPath = "tests/test_224_sphere";
  //inLibraryPath = "tests/test_224_sphere_microfacet";
  //inLibraryPath = "tests/test_pool";
  //inLibraryPath = "C:/[Hydra]/pluginFiles/scenelib/";

  //inLibraryPath = "/home/frol/PROG/HydraRepos/HydraAPI-tests/tests_f/test_119";
  //inLibraryPath = "/home/frol/PROG/HydraRepos/HydraAPI/main/demos/demo_02";
  //inLibraryPath = "tests/demo_05";
  inLibraryPath = "tests/demo_06";
  //inLibraryPath = "tests/test_42";

  inDevelopment = false; ///< recompile shaders each time; note that nvidia have their own shader cache!
  inDeviceId    = 0;     ///< opencl device id
  cpuFB         = true;  ///< store frame buffer on CPU. Automaticly enabled if
  enableMLT     = false; ///< if use MMLT, you MUST enable it early, when render process just started (here or via command line).
  boxMode       = false; ///< special 'in the box' mode when render don't react to any commands

  winWidth      = 1024; //1368;
  winHeight     = 1024; //912;

  enableOpenGL1 = false; ///< if you want to draw scene for some debug needs with OpenGL1.
  exitStatus    = false;
  runTests      = false;

  camMoveSpeed     = 2.5f;
  mouseSensitivity = 0.1f;
  saveInterval     = 0.0f;

  // dynamic data
  //
  pathTracingEnabled  = false;
  lightTracingEnabled = false;
  ibptEnabled         = false;
  sbptEnabled         = false;
  mmltEnabled         = false;
  
  cameraFreeze        = false;
  inSeed              = clock();

  getGBufferBeforeRender = false; ///< if external application that ise HydraAPI ask to calc gbuffer;
  productionPTMode       = false;

  maxSamplesContrib      = 1000000;
  mmltThreads            = (1024 * 1024) / 2;

  outDir   = ""; 
  inMethod = "";
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
void hr_setrenderpath(const std::string& a_rhs);

void Input::ParseCommandLineParams(const std::unordered_map<std::string, std::string>& a_params)
{
  ReadBoolCmd(a_params,   "-nowindow",        &noWindow);
  ReadBoolCmd(a_params,   "-cpu_fb",          &cpuFB);
  ReadBoolCmd(a_params,   "-enable_mlt",      &enableMLT);

  ReadBoolCmd(a_params,   "-cl_list_devices", &listDevicesAndExit);
  ReadBoolCmd(a_params,   "-listdevices",     &listDevicesAndExit);
  ReadBoolCmd(a_params,   "-list_devices",    &listDevicesAndExit);
  ReadBoolCmd(a_params,   "-listdev",         &listDevicesAndExit);
  
  ReadBoolCmd(a_params,   "-alloc_image_b",   &allocInternalImageB);
  ReadBoolCmd(a_params,   "-evalgbuffer",     &getGBufferBeforeRender);
  ReadBoolCmd(a_params,   "-boxmode",         &boxMode);
 
  if (listDevicesAndExit)
    noWindow = true;

  ReadIntCmd (a_params,   "-seed",         &inSeed);
  ReadIntCmd (a_params,   "-cl_device_id", &inDeviceId);
  ReadFloatCmd(a_params,  "-saveinterval", &saveInterval);

  ReadIntCmd(a_params,    "-width",        &winWidth);
  ReadIntCmd(a_params,    "-height",       &winHeight);
  
  ReadIntCmd(a_params,    "-maxsamples",       &maxSamples);
  ReadIntCmd(a_params,    "-contribsamples",   &maxSamplesContrib);
  ReadIntCmd(a_params,    "-mmltthreads",      &mmltThreads);
  
  ReadStringCmd(a_params, "-inputlib",    &inLibraryPath);
  ReadStringCmd(a_params, "-statefile",   &inTargetState);
  ReadStringCmd(a_params, "-method",      &inMethod);
  
  ReadStringCmd(a_params, "-out",         &outLDRImage); 
  ReadStringCmd(a_params, "-outdir",      &outDir);
  ReadStringCmd(a_params, "-outall",      &outAllDir);
  ReadStringCmd(a_params, "-logdir",      &inLogDirCust);
  ReadStringCmd(a_params, "-sharedimage", &inSharedImageName);
  
  if(inTargetState != "")
    inLibraryPath = inLibraryPath + "/" + inTargetState;
 
  const auto p = a_params.find("-hydradir");
  if (p != a_params.end())
    hr_setrenderpath(p->second + "/");

  //std::cout << "Input::ParseCommandLineParams, inLibraryPath = " << inLibraryPath.c_str() << std::endl;
  //std::cout << "Input::ParseCommandLineParams, boxMode       = " << boxMode               << std::endl;
}


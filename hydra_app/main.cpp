#include "main.h"

#include "hydra_api/HydraRenderDriverAPI.h"
#include "../hydra_drv/RenderDriverRTE.h"


#ifdef WIN32
  #include <windows.h> 
#else
  #include <unistd.h>
  #include <csignal>
#endif

//#include <chrono>
//#include <thread>

using pugi::xml_node;
using pugi::xml_attribute;

Input g_input;

bool g_normalExit = false;
IHRSharedAccumImage* g_pExternalImage = nullptr;
//extern bool g_hydraapipostprocessloaddll;
//extern bool g_hydraApiDisableSceneLoadInfo;

void InfoCallBack(const wchar_t* message, const wchar_t* callerPlace, HR_SEVERITY_LEVEL a_level)
{
  if (a_level >= HR_SEVERITY_WARNING)
    std::wcerr << callerPlace << L": " << message << std::endl;
  else if(a_level == HR_SEVERITY_INFO)
    std::wcout << callerPlace << L": " << message;
  else
    std::wcout << callerPlace << L": " << message << std::endl;
}

void destroy()
{
  if (g_normalExit)
    return;

  delete g_pExternalImage;
  g_pExternalImage = nullptr;

  std::wcout << L"[main]: destroy()" << std::endl;
  hrSceneLibraryClose();
}

// init and call one of main loops depen on mode
//
std::shared_ptr<IHRRenderDriver> g_pDriver = nullptr;

#ifdef WIN32
BOOL WINAPI HandlerExit(_In_ DWORD fdwControl)
{
  if (g_normalExit)
    return TRUE;

  //std::wcout << L"[main]: HandlerExit()" << std::endl;
  hrSceneLibraryClose();
  g_pDriver->~IHRRenderDriver();
  g_pDriver = nullptr;
  
  delete g_pExternalImage;
  g_pExternalImage = nullptr;

  return TRUE;
}
#else
bool destroyedBySig  = false;
void sig_handler(int signo)
{
  if(destroyedBySig)
    return;

  switch(signo)
  {
    case SIGINT : std::cerr << "\n[hydra], SIGINT";      break;
    case SIGABRT: std::cerr << "\n[hydra], SIGABRT";     break;
    case SIGILL : std::cerr << "\n[hydra], SIGINT";      break;
    case SIGTERM: std::cerr << "\n[hydra], SIGILL";      break;
    case SIGSEGV: std::cerr << "\n[hydra], SIGSEGV";     break;
    case SIGFPE : std::cerr << "\n[hydra], SIGFPE";      break;
    case SIGTSTP : std::cerr << "\n[hydra], SIGTSTP";      break;

    default     : std::cerr << "\n[hydra], SIG_UNKNOWN"; break;
      break;
  }

  delete g_pExternalImage;
  g_pExternalImage = nullptr;

  std::cerr << "[hydra]: hrSceneLibraryClose()" << std::endl;
  hrSceneLibraryClose();
  
  if(g_pDriver != nullptr)
  {
    g_pDriver->~IHRRenderDriver();
    g_pDriver = nullptr;
  }
  destroyedBySig = true;

}
#endif


void window_main (std::shared_ptr<IHRRenderDriver> a_pDriverPointer);
void console_main(std::shared_ptr<IHRRenderDriver> a_pDriverPointer, IHRSharedAccumImage* a_pSharedImage);
void tests_main  (std::shared_ptr<IHRRenderDriver> a_pDriverPointer);

extern int g_width;
extern int g_height;

int main(int argc, const char** argv)
{
//  g_hydraapipostprocessloaddll = false; // don't load post process dll's by HydraAPI

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #ifdef WIN32
  wchar_t NPath[512];
  GetCurrentDirectoryW(512, NPath);
  std::wcout << L"[main]: curr_dir = " << NPath << std::endl;
  #else
  //std::string workingDir = "../../hydra_app";
  //chdir(workingDir.c_str());
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr)
    std::cout << "[main]: curr_dir = " << cwd << std::endl;
  #endif
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  hrInfoCallback(&InfoCallBack);
  hrErrorCallerPlace(L"main");  // for debug needs only
  
  // if application will terminated in an unusual way, you have to call hrDestroy to safely free all resourced
  //
  atexit(&destroy);
#ifdef WIN32
  SetConsoleCtrlHandler(&HandlerExit, TRUE);  // if some one kill console :)
#else
  {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = sig_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = SA_RESETHAND;
    sigaction(SIGINT,  &sigIntHandler, NULL);
    sigaction(SIGABRT, &sigIntHandler, NULL);
    sigaction(SIGILL,  &sigIntHandler, NULL);
    sigaction(SIGTERM, &sigIntHandler, NULL);
    sigaction(SIGSEGV, &sigIntHandler, NULL);
    sigaction(SIGFPE,  &sigIntHandler, NULL);
    sigaction(SIGTSTP,  &sigIntHandler, NULL);
  }
#endif

  // read input parameters
  //
  std::unordered_map<std::string, std::string> cmdParams;

  for (int i = 0; i < argc; )
  {
    std::cout << "argv[" << i << "] =\t" << argv[i] << std::endl;
    
    if (argv[i][0] == '-' && i + 1 < argc)
    {
      std::cout << "argv[" << i+1 << "] =\t" << argv[i+1] << std::endl;
      const char* paramName = argv[i];
      const char* paramVal  = argv[i + 1];
      cmdParams[paramName]  = paramVal;
      i += 2;
    }
    else
    {
      const char* paramName = argv[i];
      cmdParams[paramName]  = "";
      i += 1;
    }
  }

  g_input.ParseCommandLineParams(cmdParams);

  if (g_input.inLogDirCust != "")
  {
    std::stringstream devStr;
    devStr << g_input.inDeviceId;
    
    const std::string stdpath = g_input.inLogDirCust + "stdout" + devStr.str() + ".txt";
    const std::string errpath = g_input.inLogDirCust + "stderr" + devStr.str() + ".txt";
    
    std::cout << "[main]: redirect stdout to " << stdpath.c_str() << std::endl;
    std::cout << "[main]: redirect stderr to " << errpath.c_str() << std::endl;

    auto r1 = freopen(stdpath.c_str(), "wt", stdout);
    auto r2 = freopen(errpath.c_str(), "wt", stderr);

    if (r1 == nullptr || r2 == nullptr)
      std::cerr << "[main]: freopen failed!" << std::endl;
  }
  

  try
  {
    if (g_input.runTests)
    {
      g_pDriver = std::shared_ptr<IHRRenderDriver>(CreateDriverRTE(L"", g_input.winWidth, g_input.winHeight, g_input.inDeviceId, GPU_RT_NOWINDOW | GPU_RT_DO_NOT_PRINT_PASS_NUMBER, nullptr));
      
      std::cout << "[main]: detached render driver was created for [tests_main]" << std::endl;

      tests_main(g_pDriver);
    }
    else if (g_input.noWindow)
    {
      int flags = GPU_RT_NOWINDOW;
      if (g_input.listDevicesAndExit)
      {
        flags |= GPU_RT_HW_LIST_OCL_DEVICES;
        g_input.inDeviceId = 0;
        g_pExternalImage   = nullptr;
      }
      else // connect to external framebuffer if have such
      {
        char errMsg[256];
        auto externalImageName = g_input.inSharedImageName.c_str();
        g_pExternalImage       = CreateImageAccum();
        bool externalImageIsOk = g_pExternalImage->Attach(externalImageName, errMsg);

        if (!externalImageIsOk)
        {
          std::cerr << "failed to attach to external image, enter to boxmode" << std::endl;
          delete g_pExternalImage;
          g_pExternalImage = nullptr;
          g_input.boxMode  = true;
        }
      }

      if (g_pExternalImage != nullptr && g_input.allocInternalImageB)
        flags |= GPU_RT_ALLOC_INTERNAL_IMAGEB;

      if (g_input.cpuFB)
        flags |= GPU_RT_CPU_FRAMEBUFFER;

      if(g_input.inDevelopment)
        flags |= GPU_RT_IN_DEVELOPMENT;

      if (g_input.enableMLT)
      {
        flags |= GPU_MLT_ENABLED_AT_START;
      
        if (g_input.mmltThreads == 262144)
          flags |= GPU_MMLT_THREADS_262K;
        else if (g_input.mmltThreads == 131072)
          flags |= GPU_MMLT_THREADS_131K;
        else if (g_input.mmltThreads == 65536)
          flags |= GPU_MMLT_THREADS_65K;
        else if (g_input.mmltThreads == 16384)
          flags |= GPU_MMLT_THREADS_16K;
      }
      g_pDriver = std::shared_ptr<IHRRenderDriver>(CreateDriverRTE(L"", g_input.winWidth, g_input.winHeight, g_input.inDeviceId, flags, g_pExternalImage));

      std::cout << "[main]: detached render driver was created for [console_main]" << std::endl;

      if(!g_input.listDevicesAndExit)
        console_main(g_pDriver, g_pExternalImage);
      else
      {
        std::string path = "C:\\[Hydra]\\logs\\devlist.txt";
        std::wofstream fout(path);

        HRRenderRef renderRef = hrRenderCreateFromExistingDriver(L"HydraInternalRTE", g_pDriver);
        auto pList = hrRenderGetDeviceList(renderRef);

        while (pList != nullptr)
        {
          fout << pList->id << L"; " << pList->name << L"; " << pList->driver << L"; ";
          if (pList->isCPU)
            fout << L"CPU; " << std::endl;
          else
            fout << L"GPU; " << std::endl;
          pList = pList->next;
        }

        fout.close();
      }
    }
    else
    {
      int flags = 0;

      if (g_input.cpuFB)
        flags |= GPU_RT_CPU_FRAMEBUFFER;
      
      if(g_input.inDevelopment)
        flags |= GPU_RT_IN_DEVELOPMENT;

      if (g_input.enableMLT)
        flags |= GPU_MLT_ENABLED_AT_START;
      
      if (g_input.enableMLT)
      {
        flags |= GPU_MLT_ENABLED_AT_START;
      
        if (g_input.mmltThreads == 262144)
          flags |= GPU_MMLT_THREADS_262K;
        else if (g_input.mmltThreads == 131072)
          flags |= GPU_MMLT_THREADS_131K;
        else if (g_input.mmltThreads == 65536)
          flags |= GPU_MMLT_THREADS_65K;
        else if (g_input.mmltThreads == 16384)
          flags |= GPU_MMLT_THREADS_16K;
      }
      g_pDriver = std::shared_ptr<IHRRenderDriver>(CreateDriverRTE(L"", g_width, g_height, g_input.inDeviceId, flags, nullptr));

      std::cout << "[main]: detached render driver was created for [window_main]" << std::endl;
      //g_hydraApiDisableSceneLoadInfo = true;
      window_main(g_pDriver);
    }
  }
  catch (std::runtime_error& e)
  {
    std::cout << "std::runtime_error: " << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "unknown exception" << std::endl;
  }

  delete g_pExternalImage;
  g_pExternalImage = nullptr;

  hrErrorCallerPlace(L"main"); // for debug needs only
  hrSceneLibraryClose();
  
  g_pDriver = nullptr;
  
#ifdef WIN32
  SetConsoleCtrlHandler(NULL, FALSE);
#endif

  std::cout << std::endl;
  std::cout << "normal exit" << std::endl;
  g_normalExit = true;

  return 0;
}


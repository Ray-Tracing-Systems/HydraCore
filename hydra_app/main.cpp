#include "main.h"

#include "../../HydraAPI/hydra_api/RenderDriverHydraLegacyStuff.h"
#include "../../HydraAPI/hydra_api/HydraRenderDriverAPI.h"
#include "../hydra_drv/RenderDriverRTE.h"

#include <limits>

#ifdef WIN32
  #include <windows.h> 
#else
  #include <unistd.h>
#endif

using pugi::xml_node;
using pugi::xml_attribute;

Input g_input;

void InfoCallBack(const wchar_t* message, const wchar_t* callerPlace, HR_SEVERITY_LEVEL a_level)
{
  if (a_level >= HR_SEVERITY_WARNING)
    std::wcerr << callerPlace << L": " << message << std::endl;
  else
    std::wcout << callerPlace << L": " << message << std::endl;
}

bool g_normalExit = false;
IHRSharedAccumImage* g_pExternalImage = nullptr;

void destroy()
{
  if (g_normalExit)
    return;

  std::wcout << L"[main]: destroy()" << std::endl;
  hrDestroy();

  delete g_pExternalImage; 
  g_pExternalImage = nullptr;
}

#ifdef WIN32
BOOL WINAPI HandlerExit(_In_ DWORD fdwControl)
{
  if (g_normalExit)
    return TRUE;

  //std::wcout << L"[main]: HandlerExit()" << std::endl;
  hrDestroy();
  
  delete g_pExternalImage;
  g_pExternalImage = nullptr;

  return TRUE;
}
#endif


void window_main (std::shared_ptr<IHRRenderDriver> a_pDriverPointer);
void console_main(std::shared_ptr<IHRRenderDriver> a_pDriverPointer, IHRSharedAccumImage* a_pSharedImage);
void tests_main  (std::shared_ptr<IHRRenderDriver> a_pDriverPointer);

extern int g_width;
extern int g_height;

int main(int argc, const char** argv)
{
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #ifdef WIN32
  wchar_t NPath[512];
  GetCurrentDirectoryW(512, NPath);
  std::wcout << L"[main]: curr_dir = " << NPath << std::endl;
  #else
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr)
    std::cout << "[main]: curr_dir = " << cwd << std::endl;
  #endif
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  hrInit(L"HydraApplication");

  hrInfoCallback(&InfoCallBack);
  hrErrorCallerPlace(L"main");  // for debug needs only

  atexit(&destroy);                           // if application will terminated in an unusual way, you have to call hrDestroy to safely free all resourced
#ifdef WIN32
  SetConsoleCtrlHandler(&HandlerExit, TRUE);  // if some one kill console :)
#endif

  // read input parameters
  //
  std::unordered_map<std::string, std::string> cmdParams;

  for (int i = 0; i < argc; )
  {
    if (argv[i][0] == '-' && i + 1 < argc)
    {
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
    const std::string stdpath = g_input.inLogDirCust + "stdout.txt";
    const std::string errpath = g_input.inLogDirCust + "stderr.txt";
    
    std::cout << "[main]: redirect stdout to " << stdpath.c_str() << std::endl;
    std::cout << "[main]: redirect stderr to " << errpath.c_str() << std::endl;

    auto r1 = freopen(stdpath.c_str(), "wt", stdout);
    auto r2 = freopen(errpath.c_str(), "wt", stderr);

    if (r1 == nullptr || r2 == nullptr)
      std::cerr << "[main]: freopen failed!" << std::endl;
  }


  // init and call one of main loops depen on mode
  //
  std::shared_ptr<IHRRenderDriver> pDriver = nullptr;

  try
  {
    if (g_input.runTests)
    {
      pDriver = std::shared_ptr<IHRRenderDriver>(CreateDriverRTE(L"", g_input.winWidth, g_input.winHeight, g_input.inDeviceId, GPU_RT_NOWINDOW | GPU_RT_DO_NOT_PRINT_PASS_NUMBER, nullptr));
      
      std::cout << "[main]: detached render driver was created for [tests_main]" << std::endl;

      tests_main(pDriver);
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
        const char* externalImageName = "hydraimage";
        g_pExternalImage       = CreateImageAccum();
        bool externalImageIsOk = g_pExternalImage->Attach(externalImageName, errMsg);

        if (!externalImageIsOk)
        {
          delete g_pExternalImage;
          g_pExternalImage = nullptr;
        }
      }

      if (g_pExternalImage != nullptr && g_input.allocInternalImageB)
        flags |= GPU_RT_ALLOC_INTERNAL_IMAGEB;

      if (g_input.cpuFB)
        flags |= GPU_RT_CPU_FRAMEBUFFER;

      pDriver = std::shared_ptr<IHRRenderDriver>(CreateDriverRTE(L"", g_input.winWidth, g_input.winHeight, g_input.inDeviceId, flags, g_pExternalImage));

      std::cout << "[main]: detached render driver was created for [console_main]" << std::endl;

      if(!g_input.listDevicesAndExit)
        console_main(pDriver, g_pExternalImage);
      else
      {
        std::string path = "C:\\[Hydra]\\logs\\devlist.txt";
        std::wofstream fout(path);

        HRRenderRef renderRef = hrRenderCreateFromExistingDriver(L"HydraInternalRTE", pDriver);
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
      pDriver = std::shared_ptr<IHRRenderDriver>(CreateDriverRTE(L"", g_width, g_height, g_input.inDeviceId, 0, nullptr));

      std::cout << "[main]: detached render driver was created for [window_main]" << std::endl;

      window_main(pDriver);
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

  hrErrorCallerPlace(L"main"); // for debug needs only
  hrDestroy();
  
  pDriver = nullptr; // destroy render driber explicitly
  
  delete g_pExternalImage;
  g_pExternalImage = nullptr;

  atexit(nullptr); // don't call hrDestroy twice
#ifdef WIN32
  SetConsoleCtrlHandler(NULL, FALSE);
#endif

  std::cout << std::endl;
  std::cout << "normal exit" << std::endl;
  g_normalExit = true;

  return 0;
}


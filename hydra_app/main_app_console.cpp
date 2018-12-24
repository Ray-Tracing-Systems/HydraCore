#include "../hydra_drv/RenderDriverRTE.h"
#include "../../HydraAPI/hydra_api/HydraLegacyUtils.h"

#include "main.h"

#include <chrono>
#include <thread>

using pugi::xml_node;
using pugi::xml_attribute;
using namespace HydraXMLHelpers;

bool g_exitDueToSamplesLimit = false;

extern Input g_input;
extern Camera g_cam;

static HRCameraRef    camRef;
static HRSceneInstRef scnRef;
static HRRenderRef    renderRef;

enum RENDERSTATE {STATE_WAIT = 0, STATE_RENDER = 1};

static RENDERSTATE g_state = STATE_WAIT;

static int g_sessionId = 0;
static int g_commandId = 0;


HAPI void hrDrawPassOnly(HRSceneInstRef a_pScn, HRRenderRef a_pRender, HRCameraRef a_pCam);
HAPI void hrRenderEvalGbuffer(HRRenderRef a_pRender);


static void DispatchCommand(const char* message)
{
  if (message != nullptr)
  {
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "[msg_rcv]: " << message << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;

    std::string action  = "";
    std::string layer   = "color";
    std::string sessId  = "0";
    std::string messId  = "0";
    std::string pass_id = "0";
    std::string stateFile = "";

    // read values
    {
      std::stringstream inCmd(message);

      char name[256];       //#TODO: this is unsafe; make it safe;
      char value[256];      //#TODO: this is unsafe; make it safe;

      while (inCmd.good())
      {
        inCmd >> name >> value;
        if (std::string(name) == "-action")
          action = value;

        if (std::string(name) == "-layer")
          layer = value;

        if (std::string(name) == "-sid")
          sessId = value;

        if (std::string(name) == "-mid")
          messId = value;

        if (std::string(name) == "-pass_id")
          pass_id = value;
  
        if (std::string(name) == "-statefile")
          g_input.inStateFile = value;
      }

      if (action == "start")
      {
        g_state     = STATE_RENDER;
        g_sessionId = atoi(sessId.c_str());
      }
      else if (action == "exitnow")
        g_input.exitStatus = true;
    }
    
    g_commandId = atoi(messId.c_str());
  }

}

bool InitSceneLibAndRTE(HRCameraRef& a_camRef, HRSceneInstRef& a_scnRef, HRRenderRef&  a_renderRef, std::shared_ptr<IHRRenderDriver> a_pDriver)
{
  hrErrorCallerPlace(L"InitSceneLibAndRTE");

  const std::wstring libraryPath = (g_input.inStateFile == "") ? s2ws(g_input.inLibraryPath) : s2ws(g_input.inLibraryPath + "/" + g_input.inStateFile);
  //const std::string libPathFinal = ws2s(libraryPath);
  //std::cout << "InitSceneLibAndRTE: " << libPathFinal.c_str() << std::endl;
  
  int32_t stateId = hrSceneLibraryOpen(libraryPath.c_str(), HR_OPEN_EXISTING);
  if (stateId < 0)
   return false;
  
  HRSceneLibraryInfo scnInfo = hrSceneLibraryInfo();

  if (scnInfo.camerasNum == 0) // create some default camera
    a_camRef = hrCameraCreate(L"defaultCam");

  a_renderRef = hrFindRenderByTypeName(L"HydraModern");
  a_camRef.id = 0;
  a_scnRef.id = 0;

  // copy render settings for newly created render of type L"HydraInternalRTE"
  //
  HRRenderRef renderRef2 = hrRenderCreateFromExistingDriver(L"HydraInternalRTE", a_pDriver);

  hrRenderOpen(renderRef2, HR_WRITE_DISCARD);
  {
    auto node2 = hrRenderParamNode(renderRef2);

    hrRenderOpen(a_renderRef, HR_OPEN_READ_ONLY);
    {
      auto node = hrRenderParamNode(a_renderRef);
      for (auto child = node.first_child(); child != nullptr; child = child.next_sibling())
        node2.append_copy(child);

      if (g_input.enableMLT)
        node2.child(L"method_secondary").text() = L"mmlt";
      
      if (g_input.runTests)
        node2.force_child(L"seed").text() = 777;
      else
      {
        auto currtime = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(currtime).time_since_epoch().count();
        node2.force_child(L"seed").text() = int(now_ms & 0xEFFFFFFF);
      }
      
      // override rendering method if it as set via command line
      //
      if(g_input.inMethod != "")
      {
        if(g_input.inMethod == "pt" || g_input.inMethod == "PT" || g_input.inMethod == "pathtracing")
        {
           node2.child(L"method_primary").text()   = L"pathtracing";
           node2.child(L"method_secondary").text() = L"pathtracing";
           node2.child(L"method_tertiary").text()  = L"pathtracing";
           node2.child(L"method_caustic").text()   = L"pathtracing";
        }
        else if (g_input.inMethod == "lt" || g_input.inMethod == "LT" || g_input.inMethod == "lighttracing")
        {
           node2.child(L"method_primary").text()   = L"lighttracing";
           node2.child(L"method_secondary").text() = L"lighttracing";
           node2.child(L"method_tertiary").text()  = L"lighttracing";
           node2.child(L"method_caustic").text()   = L"lighttracing"; 
        }
        else if (g_input.inMethod == "ibpt" || g_input.inMethod == "IBPT")
        {
           node2.child(L"method_primary").text()   = L"IBPT";
           node2.child(L"method_secondary").text() = L"IBPT";
           node2.child(L"method_tertiary").text()  = L"IBPT";
           node2.child(L"method_caustic").text()   = L"IBPT"; 
        }
        else if (g_input.inMethod == "mmlt" || g_input.inMethod == "MMLT")
        {
           node2.child(L"method_primary").text()   = L"pathtracing";
           node2.child(L"method_secondary").text() = L"MMLT";
           node2.child(L"method_tertiary").text()  = L"MMLT";
           node2.child(L"method_caustic").text()   = L"MMLT"; 
        }
        else
        {
          std::cerr << "unknown rendering method(command line '-method'): " << g_input.inMethod;
          exit(0);
        }
      }
      
    }
    hrRenderClose(a_renderRef);
  }
  hrRenderClose(renderRef2);

  a_renderRef = renderRef2; // well, yep! :)

  // auto pList = hrRenderGetDeviceList(a_renderRef);
  // while (pList != nullptr)
  // {
  //   std::wcout << L"device id = " << pList->id << L", name = " << pList->name << L", driver = " << pList->driver << std::endl;
  //   pList = pList->next;
  // }

  hrRenderEnableDevice(a_renderRef, g_input.inDeviceId, true);

  return true;
}

static bool g_firstCall = true;

//
static void Draw(std::shared_ptr<IHRRenderDriver> a_pDetachedRenderDriverPointer)
{
  static Timer timer;
  static int saveImageLast = 0; ///< used for saving intermediate image if this is enabled

  if (g_firstCall)
  {
    std::cout << "[main]: loading scene ... " << std::endl;
    if (!InitSceneLibAndRTE(camRef, scnRef, renderRef, a_pDetachedRenderDriverPointer))
    {
      std::cerr << "can not load scene library at " << g_input.inLibraryPath << std::endl;
      exit(0);
    }
  
    hrRenderOpen(renderRef, HR_OPEN_EXISTING);
    {
      auto paramNode = hrRenderParamNode(renderRef);
      if(g_input.outLDRImage != "")
      {
        paramNode.force_child(L"boxmode").text()        = 1;
        paramNode.force_child(L"contribsamples").text() = g_input.maxSamplesContrib;
      }
      else
        paramNode.force_child(L"boxmode").text() = g_input.boxMode ? 1 : 0;
    }
    hrRenderClose(renderRef);
    std::cout << "[main]: commit scene ... " << std::endl;

    hrCommit(scnRef, renderRef, camRef);
    timer.start();
    g_firstCall = false;
    return;
  }
  
  hrErrorCallerPlace(L"Draw");

  //hrCommit(scnRef, renderRef, camRef);
  hrDrawPassOnly(scnRef, renderRef, camRef);
  
  HRRenderUpdateInfo info = hrRenderHaveUpdate(renderRef);
  
  if (info.finalUpdate)  // save final image
  {
    if (g_input.outLDRImage != "")
    {
      const std::wstring outStr = s2ws(g_input.outLDRImage);
      hrRenderSaveFrameBufferLDR(renderRef, outStr.c_str());
    }
    g_input.exitStatus = true;
  }

  if (g_input.saveInterval > 1.0f)  // save intermediate image if this is enabled
  {
    const float time     = timer.getElapsed();
    const int saveImages = int(time / g_input.saveInterval);

    // std::cout << "saveImages    = " << saveImages << std::endl;
    // std::cout << "saveImageLast = " << saveImageLast << std::endl;
    // std::cout << "time          = " << time << std::endl;
    // std::cout << "saveInterval  = " << g_input.saveInterval << std::endl;

    if (saveImages > saveImageLast)
    {
      std::wstringstream fname1, fname2;
      if(g_input.outDir == "")
      {
        #ifdef WIN32
        fname1 << L"C:/[Hydra]/rendered_images/a_" << int(time) << L".png";
        fname2 << L"C:/[Hydra]/rendered_images/b_" << int(time) << L".hdr";
        #else
        fname1 << L"/home/frol/hydra/rendered_images/a_" << int(time) << L".png";
        fname2 << L"/home/frol/hydra/rendered_images/b_" << int(time) << L".hdr";
        #endif
      }
      else
      {
        std::wstring dir = s2ws(g_input.outDir);
        fname1 << dir.c_str() << L"/LDR_" << int(int(time)/60) << L"min.png";
        fname2 << dir.c_str() << L"/HDR_" << int(int(time)/60) << L"min.hdr";
      }
    
      const std::wstring outStr1 = fname1.str();
      const std::wstring outStr2 = fname2.str();
      hrRenderSaveFrameBufferLDR(renderRef, outStr1.c_str());
      hrRenderSaveFrameBufferHDR(renderRef, outStr2.c_str());
      saveImageLast = saveImages;
      std::wcout << L"image " << outStr1.c_str() << L" saved " << std::endl;
    }
  }

}

//
static void GetGBuffer(std::shared_ptr<IHRRenderDriver> a_pDetachedRenderDriverPointer)
{
  hrErrorCallerPlace(L"GetGBuffer");

  InitSceneLibAndRTE(camRef, scnRef, renderRef, a_pDetachedRenderDriverPointer);
  
  hrRenderOpen(renderRef, HR_OPEN_EXISTING); // #TODO: refector; this is needed here due to we update settings only once if g_firstCall == true
  {
    auto paramNode = hrRenderParamNode(renderRef);
    paramNode.force_child(L"boxmode").text()        = g_input.boxMode ? 1 : 0;
    paramNode.force_child(L"maxsamples").text()     = g_input.maxSamples;
    paramNode.force_child(L"contribsamples").text() = g_input.maxSamplesContrib;
  }
  hrRenderClose(renderRef);
  
  hrCommit(scnRef, renderRef, camRef);
  g_firstCall = false;

  hrRenderEvalGbuffer(renderRef);  
}


void console_main(std::shared_ptr<IHRRenderDriver> a_pDetachedRenderDriverPointer, IHRSharedAccumImage* a_pSharedImage)
{
  static int g_prevMessageId = 0;
  
  if (a_pSharedImage == nullptr) // selfEmployed, don't wait commands from main process
    g_state = STATE_RENDER;
  
  // GetGBuffer(a_pDetachedRenderDriverPointer);
  // g_input.exitStatus = true;
    
  if(g_input.boxMode && g_state == STATE_WAIT) // don't wait for commands in 'box mode'
    g_state = STATE_RENDER;
    
  while (!g_input.exitStatus)
  {
    
    if (a_pSharedImage != nullptr && !g_input.boxMode)
    {
      auto pHeader = a_pSharedImage->Header();
      if (pHeader->counterSnd > g_prevMessageId)
      {
        DispatchCommand(a_pSharedImage->MessageSendData());
        g_prevMessageId = pHeader->counterSnd;
      }
    }

    if (g_state == STATE_RENDER)
    {
      if (g_input.getGBufferBeforeRender)
      {
        std::cout << "[main]: begin gbuffer" << std::endl;
        GetGBuffer(a_pDetachedRenderDriverPointer);
        std::cout << "[main]: end gbuffer" << std::endl;
        g_input.getGBufferBeforeRender = false;
      }

      Draw(a_pDetachedRenderDriverPointer);
    }
    else
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
  
    if(g_exitDueToSamplesLimit)
      g_input.exitStatus = true;
  }

  
}

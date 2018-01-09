#include "../hydra_drv/RenderDriverRTE.h"
#include "../../HydraAPI/hydra_api/RenderDriverHydraLegacyStuff.h"

#include "main.h"

#include <chrono>
#include <thread>

using pugi::xml_node;
using pugi::xml_attribute;
using namespace HydraXMLHelpers;

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
HAPI IHRRenderDriver* hrRenderDriverPointer(HRRenderRef a_pRender);

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

  const std::wstring libraryPath = s2ws(g_input.inLibraryPath);

  int32_t stateId = hrSceneLibraryOpen(libraryPath.c_str(), HR_OPEN_EXISTING);
  if (stateId < 0)
   return false;

  HRSceneLibraryInfo scnInfo = hrSceneLibraryInfo();

  if (scnInfo.camerasNum == 0) // create some default camera
    a_camRef = hrCameraCreate(L"defaultCam");

  a_renderRef.id = 0;
  a_camRef.id    = 0;
  a_scnRef.id    = 0;

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
      {
        node2.child(L"method_secondary").text() = L"mlt";
        node2.child(L"method_tertiary").text()  = L"mlt";
        if(std::wstring(node2.child(L"method_caustic").text().as_string()) != L"none")
          node2.child(L"method_caustic").text()   = L"mlt";
      }

      if (g_input.runTests)
        node2.force_child(L"seed").text() = 777;
      else
        node2.force_child(L"seed").text() = GetTickCount();

    }
    hrRenderClose(a_renderRef);
  }
  hrRenderClose(renderRef2);

  a_renderRef = renderRef2; // well, yep! :)

  auto pList = hrRenderGetDeviceList(a_renderRef);

  while (pList != nullptr)
  {
    std::wcout << L"device id = " << pList->id << L", name = " << pList->name << L", driver = " << pList->driver << std::endl;
    pList = pList->next;
  }

  hrRenderEnableDevice(a_renderRef, g_input.inDeviceId, true);

  return true;
}


//
static void Draw(std::shared_ptr<IHRRenderDriver> a_pDetachedRenderDriverPointer)
{
  static Timer timer;
  static bool firstCall    = true;  
  static int saveImageLast = 0; ///< used for saving intermediate image if this is enabled

  if (firstCall)
  {
    InitSceneLibAndRTE(camRef, scnRef, renderRef, a_pDetachedRenderDriverPointer);
    std::cout << "[main]: Init() done" << std::endl;
    hrCommit(scnRef, renderRef, camRef);
    timer.start();
    firstCall = false;
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
      fname1 << L"C:/[Hydra]/rendered_images/a_" << time << L".png";
      fname2 << L"C:/[Hydra]/rendered_images/b_" << time << L".hdr";
      const std::wstring outStr1 = fname1.str();
      const std::wstring outStr2 = fname2.str();
      hrRenderSaveFrameBufferLDR(renderRef, outStr1.c_str());
      hrRenderSaveFrameBufferHDR(renderRef, outStr2.c_str());
      saveImageLast = saveImages;
      std::wcout << L"image " << outStr1.c_str() << L" saved " << std::endl;
    }
  }

}

void console_main(std::shared_ptr<IHRRenderDriver> a_pDetachedRenderDriverPointer, IHRSharedAccumImage* a_pSharedImage)
{
  if (a_pSharedImage == nullptr) // selfEmployed, don't wait commands from main process
    g_state = STATE_RENDER;

  static int prevMessageId = 0;

  while (!g_input.exitStatus)
  {    
    if (a_pSharedImage != nullptr)
    {
      auto pHeader = a_pSharedImage->Header();

      if (pHeader->counterSnd > prevMessageId)
      {
        DispatchCommand(a_pSharedImage->MessageSendData());
        prevMessageId = pHeader->counterSnd;
      }
    }

    if (g_state == STATE_RENDER)
      Draw(a_pDetachedRenderDriverPointer);
    else
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  
}

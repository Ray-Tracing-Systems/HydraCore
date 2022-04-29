#include "../hydra_drv/RenderDriverRTE.h"
#include "hydra_api/HydraLegacyUtils.h"

#include "main.h"

#include <chrono>
#include <thread>

using pugi::xml_node;
using pugi::xml_attribute;
using namespace HydraXMLHelpers;

bool g_exitDueToSamplesLimit = false;
int  g_maxCPUThreads = 4;

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

  HRInitInfo initInfo;
  initInfo.vbSize                    = 1024; // do not allocate vb.
  initInfo.computeMeshBBoxes         = false;
  initInfo.sortMaterialIndices       = false;
  initInfo.copyTexturesToLocalFolder = false;
  initInfo.localDataPath             = true;

  int32_t stateId = hrSceneLibraryOpen(libraryPath.c_str(), HR_OPEN_EXISTING, initInfo);
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

static bool g_firstCall     = true;
static bool g_startTimerNow = false;

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

      if(g_input.overrideMaxSamplesInCMD) 
      {
        paramNode.force_child(L"minRaysPerPixel").text() = g_input.maxSamples;
        paramNode.force_child(L"maxRaysPerPixel").text() = g_input.maxSamples;
      }

      if (g_input.overrideMaxCPUThreads) {
        g_maxCPUThreads = g_input.maxCPUThreads;
        paramNode.force_child(L"max_cpu_threads").text() = g_input.maxCPUThreads;
      }

      for(auto param : g_input.m_allParams) 
      {  
        bool isKey = param.first.size() > 0 && param.first[0] == '-'; 
        std::wstring paramName  = s2ws(param.first.substr(1));
        std::wstring paramValue = s2ws(param.second);
        if(paramName.size() > 0 && isKey && paramName.find(L"\\") == std::wstring::npos && paramName.find(L"/") == std::wstring::npos)
          paramNode.force_child(paramName.c_str()).text() = paramValue.c_str();
      }
    }
    hrRenderClose(renderRef);
    std::cout << "[main]: commit scene ... " << std::endl;

    hrCommit(scnRef, renderRef, camRef);
    timer.start();
    g_firstCall   = false;
    saveImageLast = 0;
    return;
  }
  else if(g_startTimerNow)
  {
    timer.start();
    saveImageLast = 0;
    g_startTimerNow = false;
  }
  
  hrErrorCallerPlace(L"Draw");

  //hrCommit(scnRef, renderRef, camRef);
  hrDrawPassOnly(scnRef, renderRef, camRef);
  
  HRRenderUpdateInfo info = hrRenderHaveUpdate(renderRef);
  
  if (info.finalUpdate && g_input.boxMode)  // save final image
  {
    if (g_input.outLDRImage != "")
    {
      std::cout << "save final image to " << g_input.outLDRImage.c_str() << std::endl; 
      const std::wstring outStr = s2ws(g_input.outLDRImage);      

      if(outStr.find(L".hdr") != std::wstring::npos || outStr.find(L".exr") != std::wstring::npos)
        hrRenderSaveFrameBufferHDR(renderRef, outStr.c_str());
      else
        hrRenderSaveFrameBufferLDR(renderRef, outStr.c_str());

      if(g_input.outAllDir != "")
      {
        std::cout << "save gbuffer     to " << g_input.outAllDir.c_str() << std::endl; 
        
        const std::wstring hdrImgName = s2ws(g_input.outAllDir + "/00_color.exr");
        const std::wstring depthName  = s2ws(g_input.outAllDir + "/01_depth.png");
        const std::wstring normsName  = s2ws(g_input.outAllDir + "/02_norms.png");
        const std::wstring diffcName  = s2ws(g_input.outAllDir + "/03_diffc.png");
        const std::wstring coverName  = s2ws(g_input.outAllDir + "/04_bords.png");
        const std::wstring shadowName = s2ws(g_input.outAllDir + "/05_shadw.png");
        const std::wstring alphaName  = s2ws(g_input.outAllDir + "/06_alpha.png");

        hrRenderSaveFrameBufferHDR (renderRef, hdrImgName.c_str());
        hrRenderSaveGBufferLayerLDR(renderRef, depthName.c_str(), L"depth");
        hrRenderSaveGBufferLayerLDR(renderRef, normsName.c_str(), L"normals");
        hrRenderSaveGBufferLayerLDR(renderRef, diffcName.c_str(), L"diffcolor");
        hrRenderSaveGBufferLayerLDR(renderRef, coverName.c_str(), L"coverage");

        hrRenderSaveGBufferLayerLDR(renderRef, alphaName.c_str(), L"alpha");
        hrRenderSaveGBufferLayerLDR(renderRef, shadowName.c_str(),L"shadow");

        /*
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out2.png", L"depth");
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out3.png", L"normals");
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out4.png", L"texcoord");
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out5.png", L"diffcolor");
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out6.png", L"alpha");
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out7.png", L"shadow");

        const unsigned int palette[20] = { 0xffff0000, 0xff00ff00, 0xff0000ff, 0xff0082c8,
                                           0xfff58231, 0xff911eb4, 0xff46f0f0, 0xfff032e6,
                                           0xffd2f53c, 0xfffabebe, 0xff008080, 0xffe6beff,
                                           0xffaa6e28, 0xfffffac8, 0xff800000, 0xffaaffc3,
                                           0xff808000, 0xffffd8b1, 0xff000080, 0xff808080 };
      
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out8.png", L"matid",  (const int32_t*)palette, 20);
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out9.png", L"objid",  (const int32_t*)palette, 20);
        hrRenderSaveGBufferLayerLDR(renderRef, L"tests_images/test_77/z_out10.png", L"instid", (const int32_t*)palette, 20);
      
        */
      }
    }
    g_input.exitStatus = true;
  }

  if (g_input.saveInterval > 1.0f)  // save intermediate image if this is enabled
  {
    const float time     = timer.getElapsed();
    const int saveImages = int(time / g_input.saveInterval);

    //std::cout << "saveImages    = " << saveImages << std::endl;
    //std::cout << "saveImageLast = " << saveImageLast << std::endl;
    //std::cout << "time          = " << time << std::endl;
    //std::cout << "saveInterval  = " << g_input.saveInterval << std::endl;

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
        fname2 << L"/home/frol/hydra/rendered_images/b_" << int(time) << L".exr";
        #endif
      }
      else
      {
        std::wstring dir = s2ws(g_input.outDir);
        fname1 << dir.c_str() << std::fixed << L"/LDR_" << std::setfill(L"0"[0]) << std::setw(3) << int(int(time)/60) << L"min.png"; 
        fname2 << dir.c_str() << std::fixed << L"/HDR_" << std::setfill(L"0"[0]) << std::setw(3) << int(int(time)/60) << L"min.exr";
      }
    
      const std::wstring outStr1 = fname1.str();
      const std::wstring outStr2 = fname2.str();
      hrRenderSaveFrameBufferLDR(renderRef, outStr1.c_str());
      hrRenderSaveFrameBufferHDR(renderRef, outStr2.c_str());
      saveImageLast = saveImages*2-1; // to save 1, 2, 4, 8, ... ; and saveImages to save 1, 2, 3, 4, ... 
      std::wcout << L"image " << outStr1.c_str() << L" saved " << std::endl;
    }
  }

}

//
static void GetGBuffer(std::shared_ptr<IHRRenderDriver> a_pDetachedRenderDriverPointer)
{
  hrErrorCallerPlace(L"GetGBuffer");

  InitSceneLibAndRTE(camRef, scnRef, renderRef, a_pDetachedRenderDriverPointer);
  
  hrRenderOpen(renderRef, HR_OPEN_EXISTING); // #TODO: refactor; this is needed here due to we update settings only once if g_firstCall == true
  {
    auto paramNode = hrRenderParamNode(renderRef);
    paramNode.force_child(L"boxmode").text()        = g_input.boxMode ? 1 : 0;
    paramNode.force_child(L"maxsamples").text()     = g_input.maxSamples;
    paramNode.force_child(L"contribsamples").text() = g_input.maxSamplesContrib;
  }
  hrRenderClose(renderRef);
  
  hrCommit(scnRef, renderRef, camRef);
  g_firstCall     = false;
  g_startTimerNow = true;

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

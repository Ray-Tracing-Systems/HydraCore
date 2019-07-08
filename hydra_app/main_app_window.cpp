#include "main.h"
#include "Camera.h"

#include "../../HydraAPI/hydra_api/HydraRenderDriverAPI.h"
IHRRenderDriver* CreateDriverRTE(const wchar_t* a_cfg, int w, int h, int a_devId, int a_flags);

#ifndef WIN32
unsigned int GetTickCount();
#endif

#include <wchar.h>

using pugi::xml_node;
using pugi::xml_attribute;

using namespace HydraXMLHelpers;

GLFWwindow* g_window = nullptr;
int         g_width  = 1024;
int         g_height = 1024;

extern Input g_input;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static bool   g_captureMouse         = false;
static bool   g_capturedMouseJustNow = false;
static double g_scrollY              = 0.0f;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Camera g_cam;

static HRCameraRef    camRef;
static HRSceneInstRef scnRef;
static HRRenderRef    renderRef;


static void Init(std::shared_ptr<IHRRenderDriver> pDriverImpl)
{
  hrErrorCallerPlace(L"Init");

  std::wstring libraryPath = s2ws(g_input.inLibraryPath);

  HRInitInfo initInfo;
  initInfo.vbSize                    = 1024; // do not allocate vb.
  initInfo.computeMeshBBoxes         = false;
  initInfo.sortMaterialIndices       = false;
  initInfo.copyTexturesToLocalFolder = false;
  initInfo.localDataPath             = true;

  int32_t stateId = hrSceneLibraryOpen(libraryPath.c_str(), HR_OPEN_EXISTING, initInfo);
  if (stateId < 0)
    exit(0);

  HRSceneLibraryInfo scnInfo = hrSceneLibraryInfo();

  if (scnInfo.camerasNum == 0) // create some default camera
    camRef = hrCameraCreate(L"defaultCam");

  renderRef.id = 0;
  camRef.id    = 0;
  scnRef.id    = 0;

  // TODO: set current camera parameters here
  //
  hrCameraOpen(camRef, HR_OPEN_READ_ONLY);
  {
    xml_node camNode = hrCameraParamNode(camRef);

    ReadFloat3(camNode.child(L"position"), &g_cam.pos.x);
    ReadFloat3(camNode.child(L"look_at"),  &g_cam.lookAt.x);
    ReadFloat3(camNode.child(L"up"),       &g_cam.up.x);
    g_cam.fov = ReadFloat(camNode.child(L"fov"));
  }
  hrCameraClose(camRef);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  HRRenderRef renderRef2;

  if (g_input.enableOpenGL1)
    renderRef2 = hrRenderCreate(L"opengl1DrawBvh");
  else
    renderRef2 = hrRenderCreateFromExistingDriver(L"HydraInternalRTE", pDriverImpl);

  hrRenderOpen(renderRef2, HR_WRITE_DISCARD);
  {
    auto node2 = hrRenderParamNode(renderRef2);

    hrRenderOpen(renderRef, HR_OPEN_READ_ONLY);
    {
      auto node = hrRenderParamNode(renderRef);
      for (auto child = node.first_child(); child != nullptr; child = child.next_sibling())
        node2.append_copy(child);

      if (g_input.enableMLT)
      {
        node2.child(L"method_secondary").text() = L"mlt";
        node2.child(L"method_tertiary").text()  = L"mlt";
        if(std::wstring(node2.child(L"method_caustic").text().as_string()) != L"none")
          node2.child(L"method_caustic").text() = L"mlt";
      }

      if (g_input.runTests)
        node2.force_child(L"seed").text() = 777;
      else
        node2.force_child(L"seed").text() = GetTickCount();

    }
    hrRenderClose(renderRef);

    node2.force_child(L"width").text()  = g_width;
    node2.force_child(L"height").text() = g_height;
    node2.force_child(L"method_primary").text() = L"raytracing";
  }
  hrRenderClose(renderRef2);
  renderRef = renderRef2; // well, yep! :)

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // auto pList = hrRenderGetDeviceList(renderRef);
  // while (pList != nullptr)
  // {
  //   std::wcout << L"device id = " << pList->id << L", name = " << pList->name << L", driver = " << pList->driver << std::endl;
  //   pList = pList->next;
  // }

  hrRenderEnableDevice(renderRef, g_input.inDeviceId, true);

  hrCommit(scnRef, renderRef, camRef);

  //exit(0);
}

static void Update(float secondsElapsed)
{
  //move position of camera based on WASD keys, and XZ keys for up and down
  if (glfwGetKey(g_window, 'S'))
    g_cam.offsetPosition(secondsElapsed * g_input.camMoveSpeed * -g_cam.forward());
  else if (glfwGetKey(g_window, 'W'))
    g_cam.offsetPosition(secondsElapsed * g_input.camMoveSpeed * g_cam.forward());
  
  if (glfwGetKey(g_window, 'A'))
    g_cam.offsetPosition(secondsElapsed * g_input.camMoveSpeed * -g_cam.right());
  else if (glfwGetKey(g_window, 'D'))
    g_cam.offsetPosition(secondsElapsed * g_input.camMoveSpeed * g_cam.right());
  
  if (glfwGetKey(g_window, 'F'))
    g_cam.offsetPosition(secondsElapsed * g_input.camMoveSpeed * -g_cam.up);
  else if (glfwGetKey(g_window, 'R'))
    g_cam.offsetPosition(secondsElapsed * g_input.camMoveSpeed * g_cam.up);
  
  //rotate camera based on mouse movement
  //
  if (g_captureMouse)
  {
    if(g_capturedMouseJustNow)
      glfwSetCursorPos(g_window, 0, 0);
  
    double mouseX, mouseY;
    glfwGetCursorPos(g_window, &mouseX, &mouseY);
    g_cam.offsetOrientation(g_input.mouseSensitivity * float(mouseY), g_input.mouseSensitivity * float(mouseX));
    glfwSetCursorPos(g_window, 0, 0); //reset the mouse, so it doesn't go out of the window
    g_capturedMouseJustNow = false;
  }
  
  //increase or decrease field of view based on mouse wheel
  //
  const float zoomSensitivity = -0.2f;
  float fieldOfView = g_cam.fov + zoomSensitivity * (float)g_scrollY;
  if(fieldOfView < 1.0f) fieldOfView   = 1.0f;
  if(fieldOfView > 180.0f) fieldOfView = 180.0f;
  g_cam.fov = fieldOfView;
  g_scrollY = 0;
 
}

//
static void Draw(void)
{
  hrErrorCallerPlace(L"Draw");

  static GLfloat	rtri  = 25.0f; // Angle For The Triangle ( NEW )
  static GLfloat	rquad = 40.0f;
  static float    g_FPS = 60.0f;
  static int      frameCounter = 0;
  static Timer    timer(true);
  static Timer    timerTotal(true);
  static int      saveImageLast = 0; ///< used for saving intermediate image if this is enabled

  const float DEG_TO_RAD = float(M_PI) / 180.0f;

  hrCameraOpen(camRef, HR_OPEN_EXISTING);
  {
    xml_node camNode = hrCameraParamNode(camRef);

    WriteFloat3(camNode.child(L"position"), &g_cam.pos.x);
    WriteFloat3(camNode.child(L"look_at"),  &g_cam.lookAt.x);
    WriteFloat3(camNode.child(L"up"),       &g_cam.up.x);
    WriteFloat( camNode.child(L"fov"),      g_cam.fov); 
  }
  hrCameraClose(camRef);

  hrRenderOpen(renderRef, HR_OPEN_EXISTING);
  {
    xml_node settingsNode = hrRenderParamNode(renderRef);

    if(g_input.mmltEnabled)
    {
      settingsNode.child(L"method_primary").text() = L"pathtracing";
      settingsNode.child(L"method_secondary").text() = L"MMLT";
    }
    else if(g_input.ibptEnabled)
      settingsNode.child(L"method_primary").text() = L"IBPT";
    else if (g_input.sbptEnabled)
       settingsNode.child(L"method_primary").text() = L"SBPT";
    else if(g_input.pathTracingEnabled)
    {
      settingsNode.child(L"method_primary").text() = L"pathtracing";
      if(g_input.productionPTMode)
        settingsNode.force_child(L"offline_pt").text() = 1;
      else
        settingsNode.force_child(L"offline_pt").text() = 0;
    }
    else if(g_input.lightTracingEnabled)
      settingsNode.child(L"method_primary").text() = L"lighttracing";
    else
      settingsNode.child(L"method_primary").text() = L"raytracing";
  }
  hrRenderClose(renderRef);


  hrCommit(scnRef, renderRef, camRef);

  if (!g_input.enableOpenGL1)
  {
    glViewport(0, 0, g_width, g_height);
    std::vector<int32_t> image(g_width * g_height);
  
    hrRenderGetFrameBufferLDR1i(renderRef, g_width, g_height, &image[0]);
  
    //for (int i = 0; i < g_width * g_height; i++) // draw shadow as red color
    //{
    //  int32_t pixelOld = image[i];
    //  int32_t alpha    = (pixelOld & 0xFF000000) >> 24;
    //  image[i]         = alpha;
    //}

    glFlush();
    glDisable(GL_TEXTURE_2D);
    glDrawPixels(g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);
  }

  // count fps 
  //
  const float coeff = 100.0f / fmax(g_FPS, 1.0f);
  rtri += coeff*0.2f;
  rquad -= coeff*0.15f;

  if (frameCounter % 10 == 0)
  {
    std::stringstream out; out.precision(4);
    g_FPS = (10.0f / timer.getElapsed());
    out << "FPS = " << g_FPS;
    glfwSetWindowTitle(g_window, out.str().c_str());
    timer.start();
  }

  frameCounter++;

  if (g_input.saveInterval > 1.0f)  // save intermediate image if this is enabled
  {
    const float time     = timerTotal.getElapsed();
    const int saveImages = int(time / g_input.saveInterval);

    //std::cout << "saveImages    = " << saveImages << std::endl;
    //std::cout << "saveImageLast = " << saveImageLast << std::endl;
    //std::cout << "time          = " << time << std::endl;
    //std::cout << "saveInterval  = " << g_input.saveInterval << std::endl;

    if (saveImages > saveImageLast)
    {
      std::wstringstream fname1, fname2;
      #ifdef WIN32
      fname1 << L"C:/[Hydra]/rendered_images/a_" << int(time) << L".png";
      fname2 << L"C:/[Hydra]/rendered_images/b_" << int(time) << L".hdr";
      #else
      fname1 << L"/home/frol/hydra/rendered_images/a_" << int(time) << L".png";
      fname2 << L"/home/frol/hydra/rendered_images/b_" << int(time) << L".hdr";
      #endif
      const std::wstring outStr1 = fname1.str();
      const std::wstring outStr2 = fname2.str();
      hrRenderSaveFrameBufferLDR(renderRef, outStr1.c_str());
      hrRenderSaveFrameBufferHDR(renderRef, outStr2.c_str());
      saveImageLast = saveImages;
      std::wcout << L"image " << outStr1.c_str() << L" saved " << std::endl;
      std::wcout.flush();
    }
  }
}


// 
static void key(GLFWwindow* window, int k, int s, int action, int mods)
{
  if (action != GLFW_PRESS) 
    return;

  g_input.camMoveSpeed = 2.5f;

  switch (k) {
  case GLFW_KEY_Z:
    break;
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    break;
  case GLFW_KEY_UP:
    //view_rotx += 5.0;
    break;
  case GLFW_KEY_DOWN:
    //view_rotx -= 5.0;
    break;
  case GLFW_KEY_LEFT:
    //view_roty += 5.0;
    break;
  case GLFW_KEY_RIGHT:
    //view_roty -= 5.0;
    break;

  case GLFW_KEY_LEFT_SHIFT:
    g_input.camMoveSpeed = 10.0f;
    break;

  case GLFW_KEY_P:
    g_input.pathTracingEnabled  = !g_input.pathTracingEnabled;
    g_input.cameraFreeze        = !g_input.cameraFreeze;
    g_input.productionPTMode    = false;

    g_input.lightTracingEnabled = false;
    g_input.ibptEnabled         = false;
    g_input.sbptEnabled         = false;
    g_input.mmltEnabled         = false;
    break;

  case GLFW_KEY_O:
    g_input.pathTracingEnabled  = !g_input.pathTracingEnabled;
    g_input.cameraFreeze        = !g_input.cameraFreeze;
    g_input.productionPTMode    = true;

    g_input.lightTracingEnabled = false;
    g_input.ibptEnabled         = false;
    g_input.sbptEnabled         = false;
    g_input.mmltEnabled         = false;
    break;

  case GLFW_KEY_L:
    g_input.lightTracingEnabled = !g_input.lightTracingEnabled;
    g_input.cameraFreeze        = !g_input.cameraFreeze;
    g_input.productionPTMode    = false;

    g_input.pathTracingEnabled  = false;
    g_input.ibptEnabled         = false;
    g_input.sbptEnabled         = false;
    g_input.mmltEnabled         = false;
    break;

  case GLFW_KEY_B:
    g_input.ibptEnabled        = !g_input.ibptEnabled;
    g_input.cameraFreeze       = !g_input.cameraFreeze;
    g_input.productionPTMode   = false;

    g_input.lightTracingEnabled = false;
    g_input.pathTracingEnabled  = false;
    g_input.sbptEnabled         = false;
    g_input.mmltEnabled         = false;
    break;

  case GLFW_KEY_V:
    g_input.sbptEnabled         = !g_input.sbptEnabled;
    g_input.cameraFreeze        = !g_input.cameraFreeze;
    g_input.productionPTMode    = false;

    g_input.lightTracingEnabled = false;
    g_input.pathTracingEnabled  = false;
    g_input.ibptEnabled         = false;
    g_input.mmltEnabled         = false;
    break;

  case GLFW_KEY_M:
    g_input.mmltEnabled         = !g_input.mmltEnabled;
    g_input.cameraFreeze        = !g_input.cameraFreeze;
    g_input.pathTracingEnabled  = false;
    g_input.lightTracingEnabled = false;
    g_input.productionPTMode    = false;
    g_input.ibptEnabled         = false;
    g_input.sbptEnabled         = false;
    break;


  default:
    return;
  }


}

// new window size 
static void reshape(GLFWwindow* window, int width, int height)
{
  hrErrorCallerPlace(L"reshape");

  g_width  = width;
  g_height = height;
  
  hrRenderOpen(renderRef, HR_OPEN_EXISTING);
  {
    pugi::xml_node node = hrRenderParamNode(renderRef);

    const wchar_t* drvName = node.attribute(L"name").as_string();
    
    wchar_t temp[256];
    swprintf(temp, 256, L"%d", g_width);
    node.child(L"width").text().set(temp);
    swprintf(temp, 256, L"%d", g_height);
    node.child(L"height").text().set(temp);
  }
  hrRenderClose(renderRef);
  
  hrCommit(scnRef, renderRef);
}


// records how far the y axis has been scrolled
void OnScroll(GLFWwindow* window, double deltaX, double deltaY) 
{
  g_scrollY += deltaY;
}

void OnMouseButtonClicked(GLFWwindow* window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    g_captureMouse = !g_captureMouse;


  if (g_captureMouse)
  {
    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    g_capturedMouseJustNow = true;
  }
  else
    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

}

void OnError(int errorCode, const char* msg) 
{
  throw std::runtime_error(msg);
}


void window_main(std::shared_ptr<IHRRenderDriver> a_pDriverImpl)
{
  g_width  = g_input.winWidth;
  g_height = g_input.winHeight;

  if (!glfwInit())
  {
    fprintf(stderr, "Failed to initialize GLFW\n");
    exit(EXIT_FAILURE);
  }

  glfwSetErrorCallback(OnError);

  glfwWindowHint(GLFW_DEPTH_BITS, 24);

  g_window = glfwCreateWindow(g_width, g_height, "Hydra GLFW Window", NULL, NULL);
  if (!g_window)
  {
    fprintf(stderr, "Failed to open GLFW window\n");
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  // Set callback functions
  glfwSetFramebufferSizeCallback(g_window, reshape);
  glfwSetKeyCallback(g_window, key);
  glfwSetScrollCallback(g_window, OnScroll);
  glfwSetMouseButtonCallback(g_window, OnMouseButtonClicked);

  glfwMakeContextCurrent(g_window);
  //gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
  glfwSwapInterval(0);
  
  glfwGetFramebufferSize(g_window, &g_width, &g_height);
  glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  reshape(g_window, g_width, g_height);

  // Parse command-line options
  Init(a_pDriverImpl);

  // Main loop
  //
  double lastTime = glfwGetTime();
  while (!glfwWindowShouldClose(g_window))
  {
    glfwPollEvents();

    double thisTime = glfwGetTime();
    const float diffTime = float(thisTime - lastTime);
    lastTime = thisTime;

    Update(diffTime);
    Draw();

    // Swap buffers
    glfwSwapBuffers(g_window);

    //exit program if escape key is pressed
    if (glfwGetKey(g_window, GLFW_KEY_ESCAPE))
      glfwSetWindowShouldClose(g_window, GL_TRUE);
  }

  // Terminate GLFW
  glfwTerminate();

}

void extSwapBuffers() { glfwSwapBuffers(g_window); } 

#include "../hydra_drv/RenderDriverRTE.h"
#include "hydra_api/HydraLegacyUtils.h"

#include "main.h"
#include "hydra_api/HR_HDRImageTool.h"

using pugi::xml_node;
using pugi::xml_attribute;
using namespace HydraXMLHelpers;

extern Input g_input;
extern Camera g_cam;

static HRCameraRef    camRef;
static HRSceneInstRef scnRef;
static HRRenderRef    renderRef;


static std::wstring tail(std::wstring const& source, size_t const length) 
{
  if (length >= source.size())
    return source;
  return source.substr(source.size() - length);
}

bool InitSceneLibAndRTE(HRCameraRef& a_camRef, HRSceneInstRef& a_scnRef, HRRenderRef&  a_renderRef, std::shared_ptr<IHRRenderDriver> a_pDriver);
HAPI void hrDrawPassOnly(HRSceneInstRef a_pScn, HRRenderRef a_pRender, HRCameraRef a_pCam);

float ImagesMSE(const std::wstring& a_path1, const std::wstring& a_path2)
{
  std::string path1 = ws2s(a_path1);
  std::string path2 = ws2s(a_path2);
  
  int w1, h1, w2, h2, chan1, chan2;
  std::vector<float> data1, data2;
  
  HydraRender::LoadImageFromFile(path1, data1, w1, h1, chan1);
  HydraRender::LoadImageFromFile(path2, data2, w2, h2, chan2);
  
  if (w1 != w2 || h1 != h2)
    return 10000.0f;
 
  return float(w1*h1)*HydraRender::MSE(data1, data2);
}

void tests_main(std::shared_ptr<IHRRenderDriver> a_pDetachedRenderDriverPointer)
{
  //g_pDetachedRenderDriverPointer = std::shared_ptr<IHRRenderDriver>(CreateDriverRTE(L"", g_input.winWidth, g_input.winHeight, g_input.inDeviceId, GPU_RT_NOWINDOW | GPU_RT_DO_NOT_PRINT_PASS_NUMBER));
  //std::cout << "[main]: detached render driver was created " << std::endl;

#ifdef WIN32
  const std::wstring testFolder       = g_input.inTestsFolder + L"tests_f";
  const std::wstring testFolderImages = g_input.inTestsFolder + L"tests_images";
 
  std::vector<std::wstring> directories = hr_listfiles(testFolder.c_str());
 
  std::cout << "begin tests" << std::endl;

  std::wofstream testOut("z_tests.txt");
  testOut.precision(2);

  const int begin = 0;
  const int end   = 100000;
  int curr = 0;

  const std::wstring testWeWantToRun = L"test_210"; // L"test_105";

  for (auto dir : directories)
  {
    auto tailOfName = tail(dir, 2);
    if (tailOfName.find_first_of(L".") != std::wstring::npos || tailOfName.find_first_of(L"..") != std::wstring::npos)
      continue;

    if (curr < begin)
    {
      testOut << curr << L":\ttest\t" << dir << "\t SKIPED!" << std::endl;
      curr++;
      continue;
    }

    if (testWeWantToRun != L"")
    {
      if (dir.find(testWeWantToRun) == std::wstring::npos)
      {
        curr++;
        continue;
      }
    }

    std::wcout << std::endl;
    std::wcout << L"========================================================" << std::endl << L"(" << curr << L"): ";
    std::wcout << L"test " << dir.c_str() << std::endl;
    std::wcout << L"========================================================" << std::endl;

    g_input.inLibraryPath = ws2s(dir);
    if (!InitSceneLibAndRTE(camRef, scnRef, renderRef, a_pDetachedRenderDriverPointer))
    {
      testOut << curr << L":\ttest\t" << dir << "\t FAILED!" << " -- can't load scene library" << std::endl;
      continue;
    }
    hrCommit(scnRef, renderRef, camRef);

    // get correct file names for saving and comparing images
    //
    const auto posOfEnd                 = dir.find_first_of(L"tests_f") + 7;
    const std::wstring imageFolderName  = dir.substr(posOfEnd, dir.size());
    const std::wstring imageFolderName2 = testFolderImages + imageFolderName;
    const std::wstring outName          = imageFolderName2 + L"/w_out.png";
    const std::wstring refName1         = imageFolderName2 + L"/w_ref.png";
    const std::wstring refName2         = imageFolderName2 + L"/z_ref.png";

    #ifdef WIN32
    std::ifstream fin(refName1);
    #else
    std::string temp(refName1.begin(), refName1.end());
    std::ifstream fin(temp);
    #endif
    const std::wstring refName = fin.good() ? refName1 : refName2;
    fin.close();

    // begin rendering
    //
    std::cout.precision(2);
    bool finished = false;
    do
    {
      hrDrawPassOnly(scnRef, renderRef, camRef);
      auto info = hrRenderHaveUpdate(renderRef);

      if (info.finalUpdate)
      {
        std::cout << "progress = " << std::fixed << 100.0f*info.progress << std::endl;
        std::cout << std::endl << "saving image ... " << std::endl;
        hrRenderSaveFrameBufferLDR(renderRef, outName.c_str());
        finished = true;
      }
      else
      {
        std::cout << "progress = " << std::fixed << 100.0f*info.progress << "%                        \r";
        std::cout.flush();
      }

    } while (!finished);

    // end rendering, check image
    //
    const float mse = ImagesMSE(outName, refName);

    if (mse < 50.0f)
      testOut << curr << L":\ttest\t" << dir << "\t PASSED!" << std::endl;
    else
      testOut << curr << L":\ttest\t" << dir << "\t FAILED!\tMSE = " << std::fixed << mse << std::endl;

    std::cout << "MSE = " << mse << std::endl;

    curr++;
  }

  #endif

  a_pDetachedRenderDriverPointer = nullptr;

  std::cout << "end tests" << std::endl;
}

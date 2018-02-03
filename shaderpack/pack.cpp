#include "../vsgl3/clHelper.h"

#include <vector>
#include <memory.h>
#include <unordered_map>

int main(int argc, const char** argv)
{
  int selectedDeviceId = 0;
  bool SAVE_BUILD_LOG = false;

#ifdef WIN32
  std::string inputfolder = "../hydra_drv";
#else
  std::string inputfolder = HYDRA_DRV_PATH;
#endif
  
  std::cout << "inputfolder = " << inputfolder.c_str() << std::endl;

  
  std::string specDefines = " -D SHADOW_TRACE_COLORED_SHADOWS -D ENABLE_OPACITY_TEX -D ENABLE_BLINN "; // -D NEXT_BOUNCE_RR
  
  std::string sshaderpath  = inputfolder + "/shaders/screen.cl";   // !!!! the hole in security !!!
  std::string tshaderpath  = inputfolder + "/shaders/trace.cl";    // !!!! the hole in security !!!
  std::string soshaderpath = inputfolder + "/shaders/sort.cl";     // !!!! the hole in security !!!
  std::string ishaderpath  = inputfolder + "/shaders/image.cl";    // !!!! the hole in security !!!
  std::string mshaderpath  = inputfolder + "/shaders/mlt.cl";      // !!!! the hole in security !!!
  std::string lshaderpath  = inputfolder + "/shaders/light.cl";    // !!!! the hole in security !!!
  std::string yshaderpath  = inputfolder + "/shaders/material.cl"; // !!!! the hole in security !!!
  
  bool inDevelopment        = false;
  std::string loadEncrypted = "crypt"; // ("crypt", "load", "")
  
  //
  //
  std::string optionsGeneral = "-cl-mad-enable -cl-no-signed-zeros -cl-single-precision-constant -cl-denorms-are-zero "; // -cl-uniform-work-group-size
  std::string optionsInclude = "-I " + inputfolder + " -D OCL_COMPILER ";  // put function that will find shader include folder
  
  if (SAVE_BUILD_LOG)
    optionsGeneral += "-cl-nv-verbose ";
  
  std::string options = optionsGeneral + optionsInclude + specDefines; // + " -cl-nv-maxrregcount=32 ";
  std::cout << "[cl_core]: packing cl programs ..." << std::endl;
  
  //m_progressBar("Compiling shaders", 0.1f);
  std::cout << "[cl_core]: packing " << ishaderpath.c_str() << "    ..." << std::endl;
  CLProgram imagep = CLProgram(nullptr, nullptr, ishaderpath.c_str(), options.c_str(), inputfolder.c_str(),
                               loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << soshaderpath.c_str() << "     ..." << std::endl;
  CLProgram sort   = CLProgram(nullptr, nullptr, soshaderpath.c_str(), options.c_str(), inputfolder.c_str(),
                               loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << mshaderpath.c_str() << "      ... " << std::endl;
  CLProgram mlt    = CLProgram(nullptr, nullptr, mshaderpath.c_str(), options.c_str(), inputfolder.c_str(),
                               loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << sshaderpath.c_str() <<  "   ... " << std::endl;
  CLProgram screen = CLProgram(nullptr, nullptr, sshaderpath.c_str(), options.c_str(), inputfolder.c_str(),
                               loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << tshaderpath.c_str() << "    ..." << std::endl;
  CLProgram trace  = CLProgram(nullptr, nullptr, tshaderpath.c_str(), options.c_str(), inputfolder.c_str(),
                               loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << lshaderpath.c_str() << "    ..." << std::endl;
  CLProgram lightp = CLProgram(nullptr, nullptr, lshaderpath.c_str(), options.c_str(), inputfolder.c_str(),
                               loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing " << yshaderpath.c_str() << " ..." << std::endl;
  CLProgram material = CLProgram(nullptr, nullptr, yshaderpath.c_str(), options.c_str(), inputfolder.c_str(),
                                 loadEncrypted, "", SAVE_BUILD_LOG);
  
  std::cout << "[cl_core]: packing cl programs complete" << std::endl << std::endl;
  
  return 0;
}


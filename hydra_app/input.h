#pragma once

#include <string>
#include <unordered_map>

struct Input
{
  Input();

  // fixed data
  //
  bool noWindow;
  bool exitStatus;
  bool enableOpenGL1;
  bool enableMLT;
  bool allocInternalImageB;
  bool runTests;     ///< run all functional tests from HydraAPI folder 
  bool listDevicesAndExit;
  bool cpuFB;
  bool inDevelopment;
  bool getGBufferBeforeRender;
  bool boxMode;

  std::string   inLibraryPath;
  std::string   inTargetState;
  std::string   inStateFile;
  
  std::string   inLogDirCust;
  std::string   inSharedImageName;

  std::wstring  inTestsFolder;
  std::string   inMethod;     // override for rendering method

  std::string   outLDRImage;
  std::string   outDir;       // override for output directory

  int32_t     inSeed;
  int32_t     inDeviceId;
  int32_t     winWidth;
  int32_t     winHeight;
  
  int32_t     maxSamples;
  int32_t     maxSamplesContrib;
  int32_t     mmltThreads;

  // mouse and keyboad/oher gui input
  //
  float camMoveSpeed;
  float mouseSensitivity;
  float saveInterval;

  // dynamic data
  //

  bool pathTracingEnabled;
  bool lightTracingEnabled;
  bool ibptEnabled;
  bool sbptEnabled;
  bool mmltEnabled;
  bool cameraFreeze;
  bool productionPTMode;

  void ParseCommandLineParams(const std::unordered_map<std::string, std::string>& a_params);
};

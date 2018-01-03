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
  bool runTests;     ///< run all functional tests from HydraAPI folder 
  bool listDevicesAndExit;

  std::string   inLibraryPath;
  std::string   inImageBName;
  std::string   inMutexBName;
  std::string   inImageAName;
  std::string   inMutexAName;
  std::string   outLDRImage;
  std::string   inLogDirCust;

  std::wstring  inTestsFolder;

  int32_t     inSeed;
  int32_t     inDeviceId;
  int32_t     winWidth;
  int32_t     winHeight;

  // mouse and keyboad/oher gui input
  //
  float camMoveSpeed;
  float mouseSensitivity;
  float saveInterval;

  // dynamic data
  //

  bool pathTracingEnabled;
  bool cameraFreeze;
  bool enableABConn;

  void ParseCommandLineParams(const std::unordered_map<std::string, std::string>& a_params);
};

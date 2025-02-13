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
  bool getGBufferBeforeRender = false;
  bool saveGBufferAfterRender = false;
  bool boxMode;
  bool doubleRT;

  std::string   inLibraryPath;
  std::string   inTargetState;
  std::string   inStateFile;
  
  std::string   inLogDirCust;
  std::string   inSharedImageName;

  std::wstring  inTestsFolder;
  std::string   inMethod;     // override for rendering method

  std::string   outLDRImage;
  std::string   outAllDir;
  std::string   outDir;       // override for output directory

  std::string   outAlbedo;
  std::string   outNormal;
  std::string   outDepth;
  std::string   outAlpha;
  std::string   outShadow;
  std::string   outCoverage;
  std::string   outMatId;
  std::string   outObjId;
  std::string   outInstId;

  int32_t     inDeviceId;
  int32_t     winWidth;
  int32_t     winHeight;
  
  int32_t     maxSamples = 0;
  int32_t     maxSamplesContrib;
  int32_t     mmltThreads;
  int32_t     maxCPUThreads = 4;
  bool        overrideMaxSamplesInCMD = false;
  bool        overrideMaxCPUThreads   = false;

  // mouse and keyboad/oher gui input
  //
  float camMoveSpeed;
  float mouseSensitivity;
  float saveInterval = 0.0f;

  // dynamic data
  //

  bool pathTracingEnabled;
  bool lightTracingEnabled;
  bool ibptEnabled;
  bool sbptEnabled;
  bool mmltEnabled;
  bool cameraFreeze;
  bool productionPTMode;

  std::unordered_map<std::string, std::string> m_allParams;

  void ParseCommandLineParams(const std::unordered_map<std::string, std::string>& a_params);
};

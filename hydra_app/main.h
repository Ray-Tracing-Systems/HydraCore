#pragma once

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES // Make MS math.h define M_PI
#endif

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <iostream>
#include <sstream>

#include "utils/Timer.h"

#if defined(USE_GL)
#if defined(_MSC_VER)
  #include <GLFW/glfw3.h>
  #pragma comment(lib, "glfw3dll.lib")
#else
  #include <GLFW/glfw3.h>
#endif
#endif

#include "input.h"

#include "hydra_api/HydraAPI.h"
#include "hydra_api/HydraXMLHelpers.h"

#include "Camera.h"

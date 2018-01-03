#pragma once

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES // Make MS math.h define M_PI
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <sstream>

#include "Timer.h"

#if defined(_MSC_VER)
  #include <GLFW/glfw3.h>
  #pragma comment(lib, "glfw3dll.lib")
#else
  #include <GLFW/glfw3.h>
#endif

#include "input.h"
#include "mesh_utils.h"

#include "../../HydraAPI/hydra_api/HydraAPI.h"
#include "../../HydraAPI/hydra_api/HydraXMLHelpers.h"

#include "Camera.h"



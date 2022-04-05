set(CMAKE_CXX_STANDARD 14)
project(vsgl3 C CXX)

set(SOURCE_FILES
    clHelper.cpp
    clHelper.h)

include_directories(${HYDRA_API_UTILS_SRC_DIR})

ADD_LIBRARY(vsgl3 STATIC ${SOURCE_FILES})



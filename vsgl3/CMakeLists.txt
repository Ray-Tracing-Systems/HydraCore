set(CMAKE_CXX_STANDARD 14)
project(vsgl3 C CXX)

set(SOURCE_FILES
    clHelper.cpp
    clHelper.h
    )


ADD_LIBRARY(vsgl3 STATIC ${SOURCE_FILES})


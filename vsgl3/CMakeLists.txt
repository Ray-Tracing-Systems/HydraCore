set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
    clHelper.cpp
    clHelper.h
    Timer.cpp
    Timer.h
    )

ADD_LIBRARY(vsgl3 STATIC ${SOURCE_FILES})


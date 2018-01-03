set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
    clew.h
    clew.c
    )

ADD_LIBRARY(clew STATIC ${SOURCE_FILES})
#ADD_EXECUTABLE(bvh_access bvh_access.cpp)


set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
        input.cpp
        main_app_console.cpp
        main_app_tests.cpp
        main.cpp)

if(USE_GL)
    set(SOURCE_FILES ${SOURCE_FILES}
            main_app_window.cpp)
endif()

set(EXTERNAL_SRC
        ${HYDRA_API_UTILS_SRC_DIR}/Timer.h
        ${HYDRA_API_UTILS_SRC_DIR}/Timer.cpp)

set(ADDITIONAL_INCLUDE_DIRS ${ADDITIONAL_INCLUDE_DIRS}
        ../LIBRARY/include)

include_directories(${ADDITIONAL_INCLUDE_DIRS})
include_directories(${HYDRA_API_ROOT_DIR})
include_directories(${HYDRA_API_UTILS_SRC_DIR})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE(WARNING "DEBUG IS ON")
    add_definitions(-DDEBUG -D_DEBUG)
endif()

link_directories(${CMAKE_SOURCE_DIR}/LIBRARY/lib_x64_linux/)
link_directories(${ADDITIONAL_LIBRARY_DIRS})

find_package(Threads REQUIRED)
find_package(OpenMP)
if(USE_GL OR WIN32)
    find_package(OpenGL REQUIRED)
endif()

add_executable(hydra ${SOURCE_FILES} ${EXTERNAL_SRC})

if(OpenMP_CXX_FOUND)
    target_link_libraries(hydra PUBLIC OpenMP::OpenMP_CXX)
endif()


find_library(HYDRA_API_LIB_DEBUG            hydra_api  HINTS ${HYDRA_API_LIB_DIR}/Debug NO_DEFAULT_PATH)
find_library(HYDRA_API_IES_PARSER_LIB_DEBUG ies_parser HINTS ${HYDRA_API_LIB_DIR}/Debug NO_DEFAULT_PATH)
find_library(HYDRA_API_MIKKTSPACE_DEBUG     mikktspace HINTS ${HYDRA_API_LIB_DIR}/Debug NO_DEFAULT_PATH)
find_library(HYDRA_API_CORTO_DEBUG          corto      HINTS ${HYDRA_API_LIB_DIR}/Debug NO_DEFAULT_PATH)


find_library(HYDRA_API_LIB_RELEASE            hydra_api  HINTS ${HYDRA_API_LIB_DIR}/Release NO_DEFAULT_PATH)
find_library(HYDRA_API_IES_PARSER_LIB_RELEASE ies_parser HINTS ${HYDRA_API_LIB_DIR}/Release NO_DEFAULT_PATH)
find_library(HYDRA_API_MIKKTSPACE_RELEASE     mikktspace HINTS ${HYDRA_API_LIB_DIR}/Release NO_DEFAULT_PATH)
find_library(HYDRA_API_CORTO_RELEASE          corto      HINTS ${HYDRA_API_LIB_DIR}/Release NO_DEFAULT_PATH)


#MESSAGE(WARNING "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR} )
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

target_link_libraries(hydra LINK_PUBLIC hydra_drv)
target_link_libraries(hydra LINK_PUBLIC debug ${HYDRA_API_LIB_DEBUG} ${HYDRA_API_IES_PARSER_LIB_DEBUG} ${HYDRA_API_MIKKTSPACE_DEBUG} ${HYDRA_API_CORTO_DEBUG})
target_link_libraries(hydra LINK_PUBLIC optimized ${HYDRA_API_LIB_RELEASE} ${HYDRA_API_IES_PARSER_LIB_RELEASE} ${HYDRA_API_MIKKTSPACE_RELEASE} ${HYDRA_API_CORTO_RELEASE})
target_link_libraries(hydra LINK_PUBLIC freeimage vsgl3)

if(WIN32)
	target_link_libraries(hydra LINK_PUBLIC debug ${HYDRA_API_CLEW_DEBUG})
	target_link_libraries(hydra LINK_PUBLIC optimized ${HYDRA_API_CLEW_RELEASE})
    if(USE_GL)
        target_link_libraries(hydra LINK_PUBLIC ${OPENGL_gl_LIBRARY} glfw3dll)
    endif()
else()
    find_package(OpenCL REQUIRED)
    target_compile_options(hydra PRIVATE -fpermissive -Wnarrowing)
    target_link_libraries(hydra LINK_PUBLIC
            hydrabvhbuilder embree embree_sse42 sys tasking simd lexers
            Threads::Threads rt dl OpenCL::OpenCL)
    if(USE_GL)
        target_link_libraries(hydra LINK_PUBLIC ${OPENGL_gl_LIBRARY} glfw)
    endif()

    install(TARGETS hydra DESTINATION hydra)

    install(FILES ${CMAKE_SOURCE_DIR}/hydra_drv/cfetch.h
            ${CMAKE_SOURCE_DIR}/hydra_drv/cglobals.h
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/texproc.cl
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/image.xx
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/light.xx
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/material.xx
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/mlt.xx
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/screen.xx
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/sort.xx
            ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/trace.xx
            DESTINATION hydra/shaders)

    install(CODE "if(EXISTS \${ENV}\${CMAKE_INSTALL_PREFIX}\${HYDRA_DIR_NAME}/shadercache)
                FILE(REMOVE_RECURSE \${ENV}\${CMAKE_INSTALL_PREFIX}\${HYDRA_DIR_NAME}/shadercache)
              ENDIF()")
    install(CODE "FILE(MAKE_DIRECTORY \${ENV}\${CMAKE_INSTALL_PREFIX}\${HYDRA_DIR_NAME}/shadercache)")
endif()
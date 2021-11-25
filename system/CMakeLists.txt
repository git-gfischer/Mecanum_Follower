CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.1)
project(yoloDepth)

SET(MY_DIR ${libfreenect2_examples_SOURCE_DIR})
SET(DEPENDS_DIR "${MY_DIR}/../depends" CACHE STRING "Dependency directory")

OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)

# The example build system is standalone and will work out-of-tree with these files copied
SET(freenect2_ROOT_DIR ${MY_DIR}/..)
SET(flextGL_SOURCES ${freenect2_ROOT_DIR}/src/flextGL.cpp)
SET(flextGL_INCLUDE_DIRS ${freenect2_ROOT_DIR}/src) # for flextGL.h

FIND_PACKAGE(PkgConfig)    # try find PKGConfig as it will be used if found
LIST(APPEND CMAKE_MODULE_PATH ${freenect2_ROOT_DIR}/cmake_modules) # FindGLFW3.cmake

IF(TARGET freenect2)
  MESSAGE(STATUS "Using in-tree freenect2 target")
  SET(freenect2_LIBRARIES freenect2)
  SET(freenect2_DLLS ${LIBFREENECT2_DLLS})
ELSE()
  FIND_PACKAGE(freenect2 REQUIRED)
  # Out-of-tree build will have to have DLLs manually copied.
ENDIF()

INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIR})

IF(ENABLE_OPENGL)
  #FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
  IF(GLFW3_FOUND AND OPENGL_FOUND)
    INCLUDE_DIRECTORIES(
      ${GLFW3_INCLUDE_DIRS}
      ${flextGL_INCLUDE_DIRS}
    )

    LIST(APPEND ${freenect2_DLLS} ${GLFW3_DLL})
    LIST(APPEND yoloDepth ${flextGL_SOURCES})

    LIST(APPEND freenect2_LIBRARIES
      ${GLFW3_LIBRARIES}
      ${OPENGL_gl_LIBRARY}
    )
    ADD_DEFINITIONS(-DEXAMPLES_WITH_OPENGL_SUPPORT=1)
  ENDIF()
ENDIF(ENABLE_OPENGL)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(hungarian 
            Tracking/Hungarian.cpp 
            Tracking/Hungarian.h)

add_library(kalman 
            Tracking/KalmanTracker.cpp 
            Tracking/KalmanTracker.h)

add_library(control
            Mecanum_Controller/Mecanum_controller.cpp
            Mecanum_Controller/Mecanum_controller.h)


#add executable
add_executable(yoloDepth main.cpp)

TARGET_LINK_LIBRARIES(yoloDepth ${freenect2_LIBRARIES})
TARGET_LINK_LIBRARIES(yoloDepth ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(yoloDepth hungarian)
TARGET_LINK_LIBRARIES(yoloDepth kalman)
TARGET_LINK_LIBRARIES(yoloDepth control)
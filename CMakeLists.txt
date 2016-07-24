cmake_minimum_required(VERSION 2.6)

project(siamese_network_ex)

message(STATUS ${DLIB_CMAKE_FILE})
if (NOT (DEFINED DLIB_CMAKE_FILE))
  message(FATAL_ERROR "DLIB_CMAKE_FILE must be defined as the path to dlib/cmake.")
else()
  include(${DLIB_CMAKE_FILE})
endif()

add_executable(${PROJECT_NAME} siamese_network_ex.cpp)
target_link_libraries(${PROJECT_NAME} dlib)


cmake_minimum_required(VERSION 2.6)

project(siamese_network_ex)

# Find dlib
if (NOT (DEFINED DLIB_DIR))
  message(FATAL_ERROR "DLIB_DIR must be defined as the directory containing dlib/cmake.")
else()
  include(${DLIB_DIR}/dlib/cmake)
endif()

if (${DLIB_VERSION} VERSION_GREATER 19.0.0)
  if (${DLIB_VERSION} VERSION_GREATER 19.2.0)
    add_definitions(-DNEW_DLIB_LOSS)
  endif()
else()
  message(FATAL_ERROR "dlib must be at least version 19.0.0. ${DLIB_VERSION} found.")
endif()

add_executable(${PROJECT_NAME} siamese_network_ex.cpp)
target_link_libraries(${PROJECT_NAME} dlib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)

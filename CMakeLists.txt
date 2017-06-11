cmake_minimum_required(VERSION 2.6)

project(siamese_network_ex)

# Find and build the dlib directory
if (DEFINED DLIB_DIR)
  find_path(DLIB_DIR
    dlib/cmake
    PATHS ${DLIB_DIR} $ENV{DLIB_DIR})

  if (EXISTS ${DLIB_DIR})
    include(${DLIB_DIR}/dlib/cmake)
  else()
    message(FATAL_ERROR "DLIB_DIR must contain 'dlib/cmake'.")
  endif()
else()
  message(FATAL_ERROR "DLIB_DIR must be defined as the path to dlib's root directory.")
endif()

# Check the version of dlib
if (${DLIB_VERSION} VERSION_GREATER 19.0.0)
  # If the version is greater than 19.2.0, define a flag that the code will use
  # to implement the new loss layer interface.
  if (${DLIB_VERSION} VERSION_GREATER 19.2.0)
    add_definitions(-DNEW_DLIB_LOSS)
  endif()

  # If the version is less than 19.4.0, make an ALIAS target dlib::dlib with the
  # target produced by dlib's in-project build.
  if (${DLIB_VERSION} VERSION_LESS 19.4.0)
    add_library(dlib::dlib ALIAS dlib)
  endif()
else()
  message(FATAL_ERROR "dlib must be at least version 19.0.0. ${DLIB_VERSION} found.")
endif()

add_executable(${PROJECT_NAME} siamese_network_ex.cpp)
target_link_libraries(${PROJECT_NAME} dlib::dlib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)

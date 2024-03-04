## template for a CMake C++ Library
cmake_minimum_required(VERSION 3.14)
project(libpressio_nvcomp VERSION "0.0.6" LANGUAGES CXX CUDA)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

include(CTest)
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

#compiler flags and standard conformance checks
check_cxx_compiler_flag("-fno-omit-frame-pointer" HAVE_NO_OMIT_FRAME_POINTER)
set(NO_OMIT_FRAME_POINTER_FLAG "")
if(HAVE_NO_OMIT_FRAME_POINTER)
  set(NO_OMIT_FRAME_POINTER_FLAG "-fno-omit-frame-pointer")
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "build libpressio_nvcomp as a shared library" ON)

find_package(LibPressio REQUIRED)
find_package(std_compat REQUIRED)
find_package(nvcomp REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(libpressio_nvcomp
  ./src/pressio-nvcomp.cc
  )
target_link_libraries(
  libpressio_nvcomp
  PRIVATE
  LibPressio::libpressio
  CUDA::cudart
  nvcomp::nvcomp
  )
target_include_directories(libpressio_nvcomp
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pressio-nvcomp>
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libpressio-nvcomp.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libpressio-nvcomp.pc
  @ONLY
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libpressio-nvcomp_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/libpressio_nvcomp_version.h
  )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/libpressio_nvcomp_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pressio-nvcomp)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio-nvcomp.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)
install(TARGETS libpressio_nvcomp 
  EXPORT LibPressioNVCompTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT LibPressioNVCompTargets NAMESPACE LibPressioNVComp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioNVComp/)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/LibPressioNVCompConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioNVCompConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioNVComp
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioNVCompConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LibPressioNVCompConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LibPressioNVCompConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioNVComp
)


option(USE_CLANG_TIDY "include clang-tidy warnings in the build log" OFF)
if(USE_CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy)
  set_target_properties(pressio-nvcomp PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY}")
  set_target_properties(pressio-nvcomp PROPERTIES C_CLANG_TIDY "${CLANG_TIDY}")
endif()

option(USE_INCLUDE_WHAT_YOU_USE "include include-what-you-use warnings in the build log" OFF)
if(USE_INCLUDE_WHAT_YOU_USE)
  find_program(INCLUDE_WHAT_YOU_USE include-what-you-use)
  set_target_properties(pressio-nvcomp PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE}")
  set_target_properties(pressio-nvcomp PROPERTIES C_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE}")
endif()


option(BUILD_DOCS "build the documentation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED dot)
  set(DOXYGEN_MAN_LINKS YES)
  set(DOXYGEN_GENERATE_MAN YES)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_EXTRACT_LOCAL_METHODS YES)
  set(DOXYGEN_EXTRACT_STATIC YES)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
  doxygen_add_docs(
    docs
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/include
    COMMENT "Generate Documenation"
  )
endif()
# vim: ft=cmake :

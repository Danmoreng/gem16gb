cmake_minimum_required(VERSION 3.28)

project(g4 VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(G4_ENABLE_CUDA "Build the CUDA capability path" OFF)
option(G4_BUILD_TESTS "Build host tests" ON)
option(G4_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(G4_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)

include(CompilerWarnings)
include(Sanitizers)
include(CudaArchitectures)

add_library(g4_core STATIC
  src/model/checkpoint_loader.cpp
  src/model/config.cpp
  src/model/manifest.cpp
  src/model/safetensors.cpp
  src/runtime/capabilities.cpp
  src/util/json.cpp
)
# GCC 16 emits a false positive from std::variant's move constructor at -O3.
# Keep the suppression local to the JSON translation unit.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_property(SOURCE src/util/json.cpp APPEND PROPERTY COMPILE_OPTIONS -Wno-maybe-uninitialized)
endif()
target_include_directories(g4_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_features(g4_core PUBLIC cxx_std_20)
g4_set_project_warnings(g4_core)
g4_enable_sanitizers(g4_core)

if(G4_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "G4_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  g4_configure_cuda_architectures()
  target_sources(g4_core PRIVATE src/cuda/dispatch.cu)
  target_link_libraries(g4_core PRIVATE CUDA::cudart)
  target_compile_definitions(g4_core
    PUBLIC G4_HAS_CUDA=1
    PRIVATE G4_COMPILED_CUDA_ARCH="${G4_CUDA_ARCHITECTURE}"
  )
  set_target_properties(g4_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  message(STATUS "g4 CUDA capability scaffold: enabled for ${CMAKE_CUDA_ARCHITECTURES}")
else()
  target_compile_definitions(g4_core PUBLIC G4_HAS_CUDA=0)
  message(STATUS "g4 CUDA capability scaffold: disabled")
endif()

foreach(tool IN ITEMS inspect run bench)
  add_executable(g4-${tool} src/cli/${tool}_main.cpp)
  target_link_libraries(g4-${tool} PRIVATE g4_core)
  g4_set_project_warnings(g4-${tool})
  g4_enable_sanitizers(g4-${tool})
endforeach()

include(CTest)
if(BUILD_TESTING AND G4_BUILD_TESTS)
  add_executable(g4-unit-tests
    tests/unit/json_test.cpp
    tests/unit/safetensors_test.cpp
    tests/unit/test_main.cpp
  )
  target_include_directories(g4-unit-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit")
  target_link_libraries(g4-unit-tests PRIVATE g4_core)
  g4_set_project_warnings(g4-unit-tests)
  g4_enable_sanitizers(g4-unit-tests)
  add_test(NAME g4-unit COMMAND g4-unit-tests)
endif()

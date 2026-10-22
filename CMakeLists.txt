cmake_minimum_required(VERSION 3.16)
project(HpcAssignment8 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SIMDSEARCH_NATIVE "Compile with -march=native (non-portable binaries)" OFF)
option(BUILD_TESTING "Build the automated tests" ON)
set(ENABLE_MPI "AUTO" CACHE STRING "Part 2 MPI support: AUTO (use if found), ON (required), OFF")
set_property(CACHE ENABLE_MPI PROPERTY STRINGS AUTO ON OFF)
set(MPIEXEC_EXTRA_FLAGS "" CACHE STRING "Extra flags for mpiexec in ctest (e.g. --oversubscribe for Open MPI)")

# --- OpenMP (required by both parts) ---------------------------------------
find_package(OpenMP REQUIRED COMPONENTS CXX)

# --- Build info recorded into benchmark metadata ---------------------------
string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
set(HPC_BUILD_TYPE_STR "${CMAKE_BUILD_TYPE}")
if(NOT HPC_BUILD_TYPE_STR)
  set(HPC_BUILD_TYPE_STR "multi-config")
endif()
set(HPC_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}} ${OpenMP_CXX_FLAGS}")
if(SIMDSEARCH_NATIVE)
  string(APPEND HPC_CXX_FLAGS_STR " -march=native")
endif()
string(APPEND HPC_CXX_FLAGS_STR " -ffp-contract=off")
string(STRIP "${HPC_CXX_FLAGS_STR}" HPC_CXX_FLAGS_STR)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/common/hpc_build_info.h.in
               ${CMAKE_BINARY_DIR}/generated/hpc_build_info.h @ONLY)

# --- Header-only shared utilities ------------------------------------------
add_library(hpc_common INTERFACE)
target_include_directories(hpc_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/common/include
  ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(hpc_common INTERFACE OpenMP::OpenMP_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Identical floating-point results for identical source expressions in every
  # translation unit (no compiler-chosen FMA contraction differences).
  target_compile_options(hpc_common INTERFACE -ffp-contract=off -Wall -Wextra)
  if(SIMDSEARCH_NATIVE)
    target_compile_options(hpc_common INTERFACE -march=native)
  endif()
endif()

if(BUILD_TESTING)
  enable_testing()
endif()

add_subdirectory(part1_simdsearch)
add_subdirectory(part2_matvec)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP     : ${OpenMP_CXX_VERSION}")

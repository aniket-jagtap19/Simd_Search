# ---- serial + OpenMP (always built) ----------------------------------------
add_library(matvec_core STATIC src/matvec.cpp)
target_include_directories(matvec_core PUBLIC include)
target_link_libraries(matvec_core PUBLIC hpc_common)

add_executable(bench_matvec_omp src/bench_matvec_omp.cpp)
target_link_libraries(bench_matvec_omp PRIVATE matvec_core)

if(BUILD_TESTING)
  add_executable(test_matvec tests/test_matvec.cpp)
  target_link_libraries(test_matvec PRIVATE matvec_core)
  add_test(NAME matvec_serial_openmp COMMAND test_matvec)
endif()

# ---- MPI (independently switchable: -DENABLE_MPI=AUTO|ON|OFF) --------------
set(_want_mpi TRUE)
if(ENABLE_MPI STREQUAL "OFF")
  set(_want_mpi FALSE)
endif()

if(_want_mpi)
  if(ENABLE_MPI STREQUAL "ON")
    find_package(MPI REQUIRED COMPONENTS CXX)
  else()
    find_package(MPI QUIET COMPONENTS CXX)
  endif()
endif()

if(_want_mpi AND MPI_CXX_FOUND)
  message(STATUS "MPI        : enabled (${MPI_CXX_COMPILER})")
  add_library(matvec_mpi_core STATIC src/matvec_mpi.cpp)
  target_link_libraries(matvec_mpi_core PUBLIC matvec_core MPI::MPI_CXX)

  add_executable(matvec_mpi src/matvec_mpi_main.cpp)
  target_link_libraries(matvec_mpi PRIVATE matvec_mpi_core)

  if(BUILD_TESTING)
    add_executable(test_matvec_mpi tests/test_matvec_mpi.cpp)
    target_link_libraries(test_matvec_mpi PRIVATE matvec_mpi_core)
    separate_arguments(_extra UNIX_COMMAND "${MPIEXEC_EXTRA_FLAGS}")
    foreach(np 1 2 3 4 8)
      add_test(NAME matvec_mpi_np${np}
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${_extra}
                       $<TARGET_FILE:test_matvec_mpi> --expect-np ${np} ${MPIEXEC_POSTFLAGS})
    endforeach()
  endif()
else()
  message(STATUS "MPI        : NOT built (ENABLE_MPI=${ENABLE_MPI}; MPI not found or disabled). "
                 "Part 1 and the serial/OpenMP matvec are unaffected.")
endif()

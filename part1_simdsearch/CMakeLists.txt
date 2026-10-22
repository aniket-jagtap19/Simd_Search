add_library(simdsearch_core STATIC src/dataset.cpp src/io.cpp src/search.cpp)
target_include_directories(simdsearch_core PUBLIC include)
target_link_libraries(simdsearch_core PUBLIC hpc_common)

add_executable(simdsearch src/main_cli.cpp)
target_link_libraries(simdsearch PRIVATE simdsearch_core)

add_executable(bench_search src/bench_search.cpp)
target_link_libraries(bench_search PRIVATE simdsearch_core)

if(BUILD_TESTING)
  add_executable(test_simdsearch tests/test_simdsearch.cpp)
  target_link_libraries(test_simdsearch PRIVATE simdsearch_core)
  add_test(NAME simdsearch_unit COMMAND test_simdsearch WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

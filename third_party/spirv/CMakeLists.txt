include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonSPIRV ${CMAKE_CURRENT_SOURCE_DIR}/triton_spirv.cc)
  target_link_libraries(TritonSPIRV PUBLIC PRIVATE Python3::Module pybind11::headers)
endif()

option(BUILD_SPIRV_OPT "build spirv-opt to debug" OFF)
add_subdirectory(tool)

option(TEST_SPIRV_CC "build triton_spirv.cc to test" OFF)
if(TEST_SPIRV_CC)
  execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REPLACE "-I" "" PYBIND11_CLEANED "${PYBIND11_RAW}")
  separate_arguments(PYBIND11_INCLUDES UNIX_COMMAND "${PYBIND11_CLEANED}")
  add_triton_plugin(TritonSPIRV ${CMAKE_CURRENT_SOURCE_DIR}/triton_spirv.cc)
  target_include_directories(TritonSPIRV PUBLIC PRIVATE ${PYBIND11_INCLUDES})
endif()

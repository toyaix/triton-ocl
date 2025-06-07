get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

if(BUILD_SPIRV_OPT)
  add_subdirectory(spirv-opt)
endif()

add_subdirectory(triton-spirv-translate)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(triton-spirv-translate EmitOpenCL.cpp)
llvm_update_compile_flags(triton-spirv-translate)
target_link_libraries(triton-spirv-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
)
mlir_check_link_libraries(triton-spirv-translate)

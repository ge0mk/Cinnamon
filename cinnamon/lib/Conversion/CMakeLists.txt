add_mlir_library(CinmCommonPatterns
    CommonPatterns.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
)

if (TORCH_MLIR_DIR)
    add_subdirectory(TorchToCinm)
endif()
if (CINM_BUILD_GPU_SUPPORT)
    add_subdirectory(CnmToGPU)
endif()
add_subdirectory(CnmToUPMEM)
add_subdirectory(UPMEMToLLVM)
add_subdirectory(CinmToCnm)
add_subdirectory(CinmToCim)
add_subdirectory(CimToMemristor)
add_subdirectory(MemristorToFunc)

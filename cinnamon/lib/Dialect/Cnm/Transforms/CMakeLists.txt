add_mlir_dialect_library(CnmTransforms
    Bufferize.cpp
    CnmOpFusion.cpp
    HoistWorkgroups.cpp
    SPIRVAttachAttributes.cpp

DEPENDS
    CnmIncGen
    CnmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRGPUDialect
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
)

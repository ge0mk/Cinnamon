add_mlir_dialect_library(CinmTransforms
    TilingPass.cpp
    LoopOptimizationPasses.cpp

DEPENDS
    CinmIncGen
    CinmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRTransformUtils
)

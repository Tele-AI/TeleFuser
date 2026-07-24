cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(tf-kernel LANGUAGES CXX CUDA)

# utils
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/detect_gpu.cmake)

# CMake
cmake_policy(SET CMP0169 OLD)
cmake_policy(SET CMP0177 NEW)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
# Disable stripping during installation
set(CMAKE_INSTALL_DO_STRIP OFF)

# Python
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
    message("CUDA_VERSION ${CUDA_VERSION} >= 13.0")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.8")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.4")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.1")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.1")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "11.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 11.8")
endif()

# Torch
find_package(Torch REQUIRED)
clear_cuda_arches(CMAKE_FLAG)

# Third Party repos
# cutlass v4.x (main cutlass for most operations)
FetchContent_Declare(
    repo-cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        57e3cfb47a2d9e0d46eb6335c3dc411498efa198
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass)

# cutlass v3.3 (for block sparse attention, incompatible with v4.x)
FetchContent_Declare(
    repo-cutlass-v3.3
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        v3.3.0
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass-v3.3)

# flashinfer - disable submodule initialization, use tf-kernel's cutlass and spdlog
FetchContent_Declare(
    repo-flashinfer
    GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer.git
    GIT_TAG        bc29697ba20b7e6bdb728ded98f04788e16ee021
    GIT_SHALLOW    OFF
    GIT_SUBMODULES ""  # Disable submodule initialization (cutlass, spdlog)
)
FetchContent_Populate(repo-flashinfer)

# spdlog (for flashinfer logging)
FetchContent_Declare(
    repo-spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG        v1.15.0
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-spdlog)


# Target SM architecture selection
# Options: AUTO (detect local GPU), ALL (all supported architectures), SM80, SM90, SM100
set(TF_KERNEL_TARGET_SM "ALL" CACHE STRING "Target SM architecture: AUTO, ALL, SM80, SM90, SM100")
set_property(CACHE TF_KERNEL_TARGET_SM PROPERTY STRINGS AUTO ALL SM80 SM90 SM100)
message(STATUS "TF_KERNEL_TARGET_SM=${TF_KERNEL_TARGET_SM}")

# Detect GPU architecture if AUTO mode
if(TF_KERNEL_TARGET_SM STREQUAL "AUTO")
    detect_gpu_arch(DETECTED_SM)
    if(DETECTED_SM)
        message(STATUS "Auto-detected GPU SM architecture: ${DETECTED_SM}")
        set(TF_KERNEL_TARGET_SM "${DETECTED_SM}")
    else()
        message(WARNING "Could not auto-detect GPU architecture, falling back to ALL_SM")
        set(TF_KERNEL_TARGET_SM "ALL_SM")
    endif()
endif()

# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

# Configure gencode below SM90
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(DEFAULT_ENABLE_BELOW_SM90 OFF)
    message(STATUS "For aarch64, disable gencode below SM90 by default")
else()
    set(DEFAULT_ENABLE_BELOW_SM90 ON)
endif()

option(ENABLE_BELOW_SM90 "Enable gencode below SM90" ${DEFAULT_ENABLE_BELOW_SM90})

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
)

# Base CUDA flags (without gencode arch flags)
set(TF_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=tf-kernel"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    "-std=c++17"
    "-DFLASHINFER_ENABLE_F16"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"

    # Supress warnings
    "-Xcompiler=-Wno-clang-format-violations"
    "-Xcompiler=-Wno-conversion"
    "-Xcompiler=-Wno-deprecated-declarations"
    "-Xcompiler=-Wno-terminate"
    "-Xcompiler=-Wfatal-errors"
    "-Xcompiler=-ftemplate-backtrace-limit=1"
    "-Xcudafe=--diag_suppress=177"   # variable was declared but never referenced
    "-Xcudafe=--diag_suppress=2361"  # invalid narrowing conversion from "char" to "signed char"

    # uncomment to debug
    # "--ptxas-options=-v"
    # "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
)

# SM90-specific gencode flags (only sm_90 and sm_90a)
set(TF_KERNEL_SM90_CUDA_FLAGS
    "-gencode=arch=compute_90a,code=sm_90a"
)

# SM100+ specific gencode flags (sm_100a and above)
set(TF_KERNEL_SM100_CUDA_FLAGS
    "-gencode=arch=compute_100a,code=sm_100a"
    "-gencode=arch=compute_120a,code=sm_120a"
)

# CMake cannot represent architecture-specific targets such as sm_90a through
# CUDA_ARCHITECTURES. Disable its default architecture (often sm_52) and pass
# the same explicit -gencode flags to the separable device-link step.
function(tf_kernel_configure_device_link target cuda_flags_var)
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES OFF)
    foreach(cuda_flag IN LISTS ${cuda_flags_var})
        if(cuda_flag MATCHES "^-gencode=")
            string(REPLACE "," "$<COMMA>" device_link_flag "${cuda_flag}")
            target_link_options(${target} PRIVATE "$<DEVICE_LINK:${device_link_flag}>")
        endif()
    endforeach()
endfunction()

set(TF_KERNEL_COMPILE_THREADS 32 CACHE STRING "Set compilation threads, default 32")

# When TF_KERNEL_COMPILE_THREADS value is less than 1, set it to 1
if (NOT TF_KERNEL_COMPILE_THREADS MATCHES "^[0-9]+$")
    message(FATAL_ERROR "TF_KERNEL_COMPILE_THREADS must be an integer, but was set to '${TF_KERNEL_COMPILE_THREADS}'.")
elseif (TF_KERNEL_COMPILE_THREADS LESS 1)
    message(STATUS "TF_KERNEL_COMPILE_THREADS was set to a value less than 1. Using 1 instead.")
    set(TF_KERNEL_COMPILE_THREADS 1)
endif()

list(APPEND TF_KERNEL_CUDA_FLAGS
    "--threads=${TF_KERNEL_COMPILE_THREADS}"
)

option(TF_KERNEL_ENABLE_BF16             "Enable BF16"             ON)
option(TF_KERNEL_ENABLE_FP8              "Enable FP8"              ON)
option(TF_KERNEL_ENABLE_FP4              "Enable FP4"              ON)
option(TF_KERNEL_ENABLE_SM90A            "Enable SM90A"            OFF)
option(TF_KERNEL_ENABLE_SM100A           "Enable SM100A"           OFF)

if (TF_KERNEL_ENABLE_BF16)
    list(APPEND TF_KERNEL_CUDA_FLAGS
        "-DFLASHINFER_ENABLE_BF16"
    )
endif()

if (TF_KERNEL_ENABLE_FP8)
    list(APPEND TF_KERNEL_CUDA_FLAGS
        "-DFLASHINFER_ENABLE_FP8"
        "-DFLASHINFER_ENABLE_FP8_E4M3"
        "-DFLASHINFER_ENABLE_FP8_E5M2"
    )
endif()

# Below SM90 gencode flags (shared by both targets)
set(TF_KERNEL_BELOW_SM90_FLAGS)
if (ENABLE_BELOW_SM90)
    list(APPEND TF_KERNEL_BELOW_SM90_FLAGS
        "-gencode=arch=compute_80,code=sm_80"
        "-gencode=arch=compute_89,code=sm_89"
    )
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        list(APPEND TF_KERNEL_BELOW_SM90_FLAGS
            "-gencode=arch=compute_87,code=sm_87"
        )
    endif()
endif()

# SM90a feature flag (CUDA 12.4+)
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    set(TF_KERNEL_ENABLE_FA3 ON)
endif()

# SM100+ additional gencode flags
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR TF_KERNEL_ENABLE_SM100A)
    # refer sm_121, sm_110 and sm_101 description  https://github.com/pytorch/pytorch/pull/156176
    if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
        list(APPEND TF_KERNEL_SM100_CUDA_FLAGS
            "-gencode=arch=compute_103a,code=sm_103a"
            "--compress-mode=size"
        )
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            list(APPEND TF_KERNEL_SM100_CUDA_FLAGS
                "-gencode=arch=compute_110a,code=sm_110a"
                "-gencode=arch=compute_121a,code=sm_121a"
            )
        endif()
    else()
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            list(APPEND TF_KERNEL_SM100_CUDA_FLAGS
                "-gencode=arch=compute_101a,code=sm_101a"
            )
        endif()
    endif()
endif()

if (TF_KERNEL_ENABLE_FP4 AND "${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8")
    set(ENABLE_NVFP4 ON)
    list(APPEND TF_KERNEL_CUDA_FLAGS
        "-DENABLE_NVFP4=1"
    )
    # Note: ENABLE_NVFP4 is only defined for SM100 target via target_compile_definitions
    # This ensures FP4 symbols are only exported in SM100 builds where the implementation is linked
    message("ENABLE FP4 OPERATORS")
else()
    set(ENABLE_NVFP4 OFF)
endif()

# All source files for DiT transformer diffusion model
# NOTE: Only include essential operators for DiT model
set(SOURCES
    "csrc/common_extension.cc"

    # Elementwise operations
    "csrc/elementwise/activation.cu"
    "csrc/elementwise/cast.cu"
    "csrc/elementwise/copy.cu"
    "csrc/elementwise/fused_add_rms_norm_kernel.cu"
    "csrc/elementwise/pos_enc.cu"
    "csrc/elementwise/rope.cu"
    "csrc/elementwise/topk.cu"

    # GEMM operations (quantization support)
    "csrc/gemm/fp8_gemm_kernel.cu"
    "csrc/gemm/int8_gemm_kernel.cu"
    "csrc/gemm/fp8_blockwise_gemm_kernel.cu"
    "csrc/gemm/bmm_fp8.cu"

    # Quantization operations
    "csrc/gemm/per_tensor_quant_fp8.cu"
    "csrc/gemm/per_token_group_quant_8bit.cu"
    "csrc/gemm/per_token_group_quant_8bit_v2.cu"
    "csrc/gemm/per_token_quant_fp8.cu"

    # Memory operations
    "csrc/memory/store.cu"
    "csrc/memory/weak_ref_tensor.cpp"

    # FlashInfer operations
    "${repo-flashinfer_SOURCE_DIR}/csrc/norm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/renorm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/sampling.cu"

    # SageAttention fused operations
    "csrc/sageattn2/fused/fused.cu"

    # SageAttention qattn operations (architecture specific)
    "csrc/sageattn2/qattn/qk_int_sv_f16_cuda_sm80.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu"
)
set(SM_90_SOURCES
    "csrc/sageattn2/qattn/qk_int_sv_f8_cuda_sm90.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu"
)
set(SM_80_SOURCES

    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu"
)
set(SM_100_SOURCES
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu"
    "csrc/sageattn2/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu"
)
if (ENABLE_NVFP4)
    list(APPEND SM_100_SOURCES
        "csrc/gemm/nvfp4_expert_quant.cu"
        "csrc/gemm/nvfp4_quant_entry.cu"
        "csrc/gemm/nvfp4_quant_kernels.cu"
        "csrc/gemm/nvfp4_scaled_mm_entry.cu"
        "csrc/gemm/nvfp4_scaled_mm_kernels.cu"
        "csrc/sageattn3/blackwell/api.cu"
        "csrc/sageattn3/quantization/fp4_quantization_4d.cu"
    )
endif()

set(INCLUDES
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${repo-flashinfer_SOURCE_DIR}/include
    ${repo-flashinfer_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/examples/common
    ${repo-spdlog_SOURCE_DIR}/include
)

# Determine which SM targets to build based on TF_KERNEL_TARGET_SM
set(BUILD_SM80 OFF)
set(BUILD_SM90 OFF)
set(BUILD_SM100 OFF)

if(TF_KERNEL_TARGET_SM STREQUAL "ALL")
    set(BUILD_SM80 ON)
    set(BUILD_SM90 ON)
    set(BUILD_SM100 ON)
elseif(TF_KERNEL_TARGET_SM STREQUAL "SM80")
    set(BUILD_SM80 ON)
elseif(TF_KERNEL_TARGET_SM STREQUAL "SM90")
    set(BUILD_SM90 ON)
elseif(TF_KERNEL_TARGET_SM STREQUAL "SM100")
    set(BUILD_SM100 ON)
else()
    message(FATAL_ERROR "Invalid TF_KERNEL_TARGET_SM value: ${TF_KERNEL_TARGET_SM}. Must be one of: AUTO, ALL, SM80, SM90, SM100")
endif()

message(STATUS "Build targets: SM80=${BUILD_SM80}, SM90=${BUILD_SM90}, SM100=${BUILD_SM100}")

# =========================== Block Sparse Attention (CUTLASS v3.3) =============================
# Block Sparse Attention requires CUTLASS v3.3, which is incompatible with v4.x
# We create a separate static library with its own CUTLASS version
set(BLOCK_SPARSE_ATTN_CPP_SOURCES
    "csrc/block_sparse_attn/flash_api.cpp"
)
set(BLOCK_SPARSE_ATTN_CUDA_SOURCES
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim32_fp16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim32_fp16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim32_bf16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim32_bf16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim64_fp16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim64_fp16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim64_bf16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim64_bf16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim128_fp16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim128_fp16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim128_bf16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_fwd_block_hdim128_bf16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim32_fp16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim32_fp16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim32_bf16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim32_bf16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim64_fp16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim64_fp16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim64_bf16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim64_bf16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim128_fp16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim128_fp16_causal_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim128_bf16_sm80.cu"
    "csrc/block_sparse_attn/src/flash_bwd_block_hdim128_bf16_causal_sm80.cu"
)
set(BLOCK_SPARSE_ATTN_SOURCES ${BLOCK_SPARSE_ATTN_CPP_SOURCES} ${BLOCK_SPARSE_ATTN_CUDA_SOURCES})

set(BLOCK_SPARSE_ATTN_CUTLASS_INCLUDES
    ${repo-cutlass-v3.3_SOURCE_DIR}/include
    ${repo-cutlass-v3.3_SOURCE_DIR}/tools/util/include
    ${PROJECT_SOURCE_DIR}/csrc/block_sparse_attn
    ${PROJECT_SOURCE_DIR}/csrc/block_sparse_attn/src
)

# Set specific include directories for block_sparse_attn source files
# These need CUTLASS v3.3 which conflicts with v4.x used by other sources
set_source_files_properties(${BLOCK_SPARSE_ATTN_SOURCES}
    PROPERTIES
    INCLUDE_DIRECTORIES "${BLOCK_SPARSE_ATTN_CUTLASS_INCLUDES}"
)

# =========================== Common SM90 Build ============================= #
if(BUILD_SM90)
    message(STATUS "Building SM90 target")
    Python_add_library(common_ops_sm90_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES} ${SM_90_SOURCES} ${BLOCK_SPARSE_ATTN_SOURCES})

    # SM90 target: only use sm_90/sm_90a + below_sm90 flags, NOT sm_100+
    target_compile_options(common_ops_sm90_build PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${TF_KERNEL_CUDA_FLAGS}  ${TF_KERNEL_SM90_CUDA_FLAGS} -use_fast_math>
    )
    tf_kernel_configure_device_link(common_ops_sm90_build TF_KERNEL_SM90_CUDA_FLAGS)
    target_include_directories(common_ops_sm90_build PRIVATE ${INCLUDES})
    # Set output name and separate build directory to avoid conflicts
    set_target_properties(common_ops_sm90_build PROPERTIES
        OUTPUT_NAME "common_ops"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm90"
    )
endif()

# =========================== Common SM80 Build ============================= #
if(BUILD_SM80)
    message(STATUS "Building SM80 target")
    Python_add_library(common_ops_sm80_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES} ${SM_80_SOURCES} ${BLOCK_SPARSE_ATTN_SOURCES})

    # SM80 target: use below_sm90 flags only (for sm80-89)
    target_compile_options(common_ops_sm80_build PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${TF_KERNEL_CUDA_FLAGS} ${TF_KERNEL_BELOW_SM90_FLAGS}>
    )
    tf_kernel_configure_device_link(common_ops_sm80_build TF_KERNEL_BELOW_SM90_FLAGS)
    target_include_directories(common_ops_sm80_build PRIVATE ${INCLUDES})
    # Set output name and separate build directory to avoid conflicts
    set_target_properties(common_ops_sm80_build PROPERTIES
        OUTPUT_NAME "common_ops"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm80"
    )
endif()

# =========================== Common SM100+ Build ============================= #
if(BUILD_SM100)
    message(STATUS "Building SM100 target")
    Python_add_library(common_ops_sm100_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES} ${SM_100_SOURCES} ${BLOCK_SPARSE_ATTN_SOURCES})

    # SM100+ target: only use sm100+ gencode flags (sm_100a, sm_120a, etc.)
    target_compile_options(common_ops_sm100_build PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${TF_KERNEL_CUDA_FLAGS} ${TF_KERNEL_SM100_CUDA_FLAGS}>
    )
    tf_kernel_configure_device_link(common_ops_sm100_build TF_KERNEL_SM100_CUDA_FLAGS)
    target_include_directories(common_ops_sm100_build PRIVATE ${INCLUDES})
    # Set output name and separate build directory to avoid conflicts
    set_target_properties(common_ops_sm100_build PROPERTIES
        OUTPUT_NAME "common_ops"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm100"
    )
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
    message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

# Link libraries to main targets and install
if(BUILD_SM90)
    target_link_libraries(common_ops_sm90_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)
    install(TARGETS common_ops_sm90_build LIBRARY DESTINATION tf_kernel/sm90)
endif()

if(BUILD_SM80)
    target_link_libraries(common_ops_sm80_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)
    install(TARGETS common_ops_sm80_build LIBRARY DESTINATION tf_kernel/sm80)
endif()

if(BUILD_SM100)
    target_link_libraries(common_ops_sm100_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)
    # Only define ENABLE_NVFP4 for SM100 target where FP4 implementation is linked
    if(ENABLE_NVFP4)
        target_compile_definitions(common_ops_sm100_build PRIVATE ENABLE_NVFP4=1)
    endif()
    install(TARGETS common_ops_sm100_build LIBRARY DESTINATION tf_kernel/sm100)
endif()

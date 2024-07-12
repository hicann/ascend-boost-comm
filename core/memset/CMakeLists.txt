include(${PROJECT_SOURCE_DIR}/cmake/kernel_config.cmake)

set(OPS_PROJECT_ROOT_DIR "$ENV{CODE_ROOT}")
set(MKI_SCRIPT_DIR "$ENV{SCRIPT_DIR}")
set(OPS_THIRD_PARTY_DIR "$ENV{THIRD_PARTY_DIR}")

include_directories(
    ${PROJECT_SOURCE_DIR}/3rdparty/securec/include
)

link_directories(
    $ENV{CODE_ROOT}/lib
    ${ASCEND_DRIVER_DIR}/lib64/common
    $ENV{ASCEND_HOME_PATH}/lib64
)

set(BUILD_MemsetOperation_MemsetKernel_ascend310p ON)
set(BUILD_MemsetOperation_MemsetKernel_ascend910b ON)

set(op_name MemsetOperation)

add_kernel(memset ascend310p vector
    kernel/memset.cpp MemsetKernel)

add_kernel(memset ascend910b vector
    kernel/memset.cpp MemsetKernel)

# set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
add_custom_command(
    OUTPUT ${BINARY_SRC_LIST}
    DEPENDS ${BINARY_TARGET_LIST}
    COMMAND python3 $ENV{CODE_ROOT}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type ascendc
)
add_custom_target(srcs ALL
    DEPENDS ${BINARY_SRC_LIST}
)
# add_library(mki_memset SHARED ${BINARY_SRC_LIST})
# add_dependencies(mki_memset srcs)

# set(BINARY_SRC_LIST "" CACHE INTERNAL "src path")
# set(BINARY_TARGET_LIST "" CACHE INTERNAL "target list")

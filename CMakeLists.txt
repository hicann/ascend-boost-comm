cmake_minimum_required(VERSION 3.8)
project("mki")
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TEST_FRAMEWORK "BUILD_TEST_FRAMEWORK" OFF)
option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)
option(USE_PYTHON_TEST "USE_PYTHON_TEST" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
message(STATUS "BUILD_TEST_FRAMEWORK:${BUILD_TEST_FRAMEWORK}")
message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")
message(STATUS "USE_PYTHON_TEST:${USE_PYTHON_TEST}")
message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")

include(${PROJECT_SOURCE_DIR}/cmake/host_config.cmake)
set(ASCEND_DRIVER_DIR /usr/local/Ascend/driver)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/include
    ${PROJECT_SOURCE_DIR}/3rdparty/metadef/inc/external
    ${PROJECT_SOURCE_DIR}/3rdparty/nlohmannJson/include
    ${ASCEND_DRIVER_DIR}/kernel/libc_sec/include
)

link_directories(
    ${ASCEND_DRIVER_DIR}/lib64/common
    $ENV{ASCEND_HOME_PATH}/lib64
)

file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/core/mki/*.cpp")

# Add memset kernel to libmki (Not available)
add_subdirectory(${PROJECT_SOURCE_DIR}/core/memset)

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
add_library(mki_static STATIC ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki_static srcs)
target_link_libraries(mki_static PUBLIC dl pthread c_sec mmpa)
target_compile_definitions(mki_static PRIVATE fe=${NAMESPACE}Fe)

add_library(mki SHARED ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki srcs)
target_link_libraries(mki PUBLIC dl pthread c_sec mmpa)
target_compile_definitions(mki PRIVATE fe=${NAMESPACE}Fe)

add_executable(sym_check ${PROJECT_SOURCE_DIR}/core/sym_check.cpp)
target_link_libraries(sym_check PRIVATE mki)

if(BUILD_TEST_FRAMEWORK)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests/framework)
endif()

# cmake install stage
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/output/mki" CACHE PATH "..." FORCE)
endif()

# cp files
install(FILES
    ${PROJECT_SOURCE_DIR}/scripts/set_env.sh DESTINATION ./)

file(GLOB_RECURSE SCRIPTS_FILES "${PROJECT_SOURCE_DIR}/scripts/*.py")
list(APPEND SCRIPTS_FILES 
    ${PROJECT_SOURCE_DIR}/scripts/build_env_setup.sh
    ${PROJECT_SOURCE_DIR}/scripts/build_ops.sh
)
install(FILES ${SCRIPTS_FILES} DESTINATION ./scripts)

install(DIRECTORY
    ${PROJECT_SOURCE_DIR}/core/include/mki DESTINATION include)

# install libs
install(TARGETS mki_static DESTINATION lib)
install(TARGETS mki DESTINATION lib)

set(BINARY_SRC_LIST "" CACHE INTERNAL "src path")
set(BINARY_TARGET_LIST "" CACHE INTERNAL "target list")

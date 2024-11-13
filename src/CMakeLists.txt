file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*/*.cpp")
list(FILTER SOURCE_FILES EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/mki_loader/.*")

file(GLOB_RECURSE LOADER_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/mki_loader/*.cpp")
# Add memset kernel to libmki
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/utils/memset/kernel)

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
add_library(mki_static STATIC ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki_static ${BINARY_TARGET_LIST})
target_link_libraries(mki_static PUBLIC dl pthread c_sec mmpa runtime)
target_compile_definitions(mki_static PRIVATE ${NAMESPACE}=Mki)

add_library(mki SHARED ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki ${BINARY_TARGET_LIST})
target_link_libraries(mki PUBLIC dl pthread c_sec mmpa profapi runtime)
target_compile_definitions(mki PRIVATE ${NAMESPACE}=Mki)

add_library(mki_loader STATIC ${LOADER_SOURCE_FILES})
target_link_libraries(mki_loader PUBLIC mki dl pthread c_sec mmpa)

add_executable(sym_check ${CMAKE_CURRENT_LIST_DIR}/sym_check.cpp)
target_link_libraries(sym_check PRIVATE mki_loader mki)

string(TOUPPER ${NAMESPACE} NAMESPACE_UPPER)
add_custom_command(
    OUTPUT ops.h
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/include/mki_loader/ops.h
    COMMAND sh -c "sed -e 's/OpSpace/${NAMESPACE}/g' -e 's/MKI_LOADER_OPS_H/${NAMESPACE_UPPER}_LOADER_OPS_H/' ${CMAKE_CURRENT_LIST_DIR}/include/mki_loader/ops.h > ops.h"
    VERBATIM)

add_custom_target(${NAMESPACE}_ops_h ALL DEPENDS ops.h)

# install libs and includes
install(TARGETS mki_static DESTINATION lib)
install(TARGETS mki DESTINATION lib)
install(TARGETS mki_loader DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/include DESTINATION . PATTERN "mki_loader/ops.h" EXCLUDE)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ops.h DESTINATION ./include/mki_loader)

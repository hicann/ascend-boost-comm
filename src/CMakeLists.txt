file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*/*.cpp")
list(FILTER SOURCE_FILES EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/mki_loader/.*")

file(GLOB_RECURSE LOADER_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/mki_loader/*.cpp")
# Add memset kernel to libmki
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/utils/memset/kernel)

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
add_library(mki_static STATIC ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki_static binary_encode_srcs)
target_link_libraries(mki_static PUBLIC dl pthread c_sec mmpa runtime)

add_library(mki SHARED ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki binary_encode_srcs)
target_link_libraries(mki PUBLIC dl pthread c_sec mmpa profapi runtime)

add_library(mki_loader STATIC ${LOADER_SOURCE_FILES})
target_link_libraries(mki_loader PUBLIC mki dl pthread c_sec mmpa)

add_executable(sym_check ${CMAKE_CURRENT_LIST_DIR}/sym_check.cpp)
target_link_libraries(sym_check PRIVATE mki_loader mki)

# install libs
install(TARGETS mki_static DESTINATION lib)
install(TARGETS mki DESTINATION lib)
install(TARGETS mki_loader DESTINATION lib)

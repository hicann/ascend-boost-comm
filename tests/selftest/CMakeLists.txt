aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/api TEST_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/operationir TEST_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/utils TEST_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/common TEST_SRC)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/googletest googletest EXCLUDE_FROM_ALL)
add_executable(mki_selftest ${TEST_SRC})
target_include_directories(mki_selftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/metadef/inc)
target_include_directories(mki_selftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_include_directories(mki_selftest PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(mki_selftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib)
target_link_libraries(mki_selftest PRIVATE mki gtest gtest_main)
target_compile_options(mki_selftest PRIVATE
    -Wno-sign-compare
    -Wno-narrowing
)
target_compile_definitions(mki_selftest PRIVATE fe=MkiFe)

install(TARGETS mki_selftest DESTINATION bin)

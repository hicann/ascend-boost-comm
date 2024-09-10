
execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_auto.py
                                --namespace ${NAMESPACE}
                                --params_path ${PROJECT_SOURCE_DIR}/include/atbops)
add_library(mki_test_autogen SHARED ${CMAKE_CURRENT_LIST_DIR}/op_desc_json.cpp)
target_link_libraries(mki_test_autogen PRIVATE atbops)
target_compile_definitions(mki_test_autogen PRIVATE OpSpace=${NAMESPACE})

install(TARGETS mki_test_autogen DESTINATION tests)

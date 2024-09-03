macro(add_operation op srcs)
    if (BUILD_${op})
        add_compile_definitions(OperationPlaceHolder="${op}")
        add_library(${op} OBJECT ${srcs})
        set(op_name ${op})
        set(ops_objects ${ops_objects} ${op} PARENT_SCOPE)
    endif()
endmacro()

macro(add_kernel kernel soc channel srcs tac)
    if (BUILD_${op_name}_${tac}_${soc})
        set(${kernel}_${soc}_output
            ${CMAKE_BINARY_DIR}/op_kernels/${soc}/${op_name}/${tac}/${kernel}.o)
        set(PYTHON_ARGS 
            "--soc" "${soc}"
            "--channel" "${channel}"
            "--srcs" "${CMAKE_CURRENT_LIST_DIR}/${srcs}"
            "--dst" "${${kernel}_${soc}_output}"
            "--code_root" "${OPS_THIRD_PARTY_DIR}/.."
            "--kernel" "${kernel}"
        )
        string(TOLOWER ${soc} soc_lower)
        string(LENGTH ${soc} soc_length)
        if(${soc_lower} STREQUAL "ascend910b")
            list(APPEND PYTHON_ARGS "--build_type" "msDebug")
        endif()
        add_custom_command(
            OUTPUT ${${kernel}_${soc}_output}
            DEPENDS ${srcs}
            WORKING_DIRECTORY ${OPS_PROJECT_ROOT_DIR}
            COMMAND python3 ${MKI_SCRIPT_DIR}/compile_ascendc.py ${PYTHON_ARGS}
        )
        add_custom_target(ascendc_${kernel}_${soc} ALL
            DEPENDS ${${kernel}_${soc}_output}
        )
        set(LOCAL_BINARY_SRC_LIST ${LOCAL_BINARY_SRC_LIST} $ENV{CACHE_DIR}/obj/${soc}/${op_name}/${kernel}.cpp)
        set(BINARY_SRC_LIST ${BINARY_SRC_LIST} ${LOCAL_BINARY_SRC_LIST} PARENT_SCOPE)
        set(LOCAL_BINARY_TARGET_LIST ${LOCAL_BINARY_TARGET_LIST} ascendc_${kernel}_${soc})
        set(BINARY_TARGET_LIST ${BINARY_TARGET_LIST} ${LOCAL_BINARY_TARGET_LIST} PARENT_SCOPE)
    endif()
endmacro()

macro(add_kernel_bin tac soc)
    set(${tac}_${soc}_dir
        ${CMAKE_BINARY_DIR}/op_kernels/${soc}/${op_name}/${tac})
    file(MAKE_DIRECTORY ${${tac}_${soc}_dir})
    set(${tac}_${soc}_output ${${tac}_${soc}_dir}/${tac}.o)
    add_custom_command(
        OUTPUT ${${tac}_${soc}_output}
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/kernel/${tac}_${soc}.txt
        WORKING_DIRECTORY ${OPS_PROJECT_ROOT_DIR}
        COMMAND xxd -r -ps
            ${CMAKE_CURRENT_LIST_DIR}/kernel/${tac}_${soc}.txt
            ${${tac}_${soc}_output}
        COMMAND cp
            ${CMAKE_CURRENT_LIST_DIR}/kernel/${tac}_${soc}.json
            ${${tac}_${soc}_dir}/${tac}.json
    )
    add_custom_target(binary_${tac}_${soc} ALL
        DEPENDS ${${tac}_${soc}_output}
    )
endmacro()

macro(return_value_check return_value msg)
    if(NOT return_value EQUAL 0)
        message(FATAL_ERROR ${msg})
    endif()
endmacro()

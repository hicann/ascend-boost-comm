#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -e
BUILD_DIR=$(dirname $(readlink -f $0))
CANN_DIR=$(readlink -f $BUILD_DIR/../CANN)
export SCRIPT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=$(pwd)
cd $SCRIPT_DIR
cd ..
export CODE_ROOT=$(pwd)
export CACHE_DIR=$CODE_ROOT/build
export OUTPUT_DIR=$CODE_ROOT/output
export BUILD_CONFIG_DIR=$BUILD_DIR/../configs

export THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
BUILD_TEST_FRAMEWORK=OFF
COMPILE_OPTIONS=""
DEVICE_CODE_PACK_SWITCH=ON
USE_VERBOSE=OFF
ENABLE_COVERAGE=OFF
USE_CXX11_ABI=""
IS_RELEASE=False
BUILD_OPTION_LIST="testframework release example dev debug unittest clean help"
BUILD_CONFIGURE_LIST=("--output=.*" "--use_cxx11_abi=0" "--use_cxx11_abi=1"
                      "--verbose" "--no_werror" "--coverage" "--namespace=.*" "--msdebug")

# install cann
function fn_install_cann_and_kernel()
{
    cd $CANN_DIR
    chmod +x *.run
    cann_install_path="/home/slave1/Ascend/ascend-toolkit"
    if [ ! -d "$cann_install_path" ];then
        ./CANN-runtime-*.run --full --quiet --nox11 --install-path=${cann_install_path}/runtime
        ./CANN-compiler-*.run --full --pylocal --quiet --nox11 --install-path=${cann_install_path}/compiler & \
        ./CANN-opp-*.run --full --quiet --nox11 --install-path=${cann_install_path}/opp & \
        ./CANN-toolkit-*.run --full --pylocal --quiet --nox11 --install-path=${cann_install_path}/toolkit & \
        ./CANN-aoe-*.run --full --quiet --nox11 --install-path=${cann_install_path}/aoe & \
        ./Ascend910B-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}/910b_opp_kernel & \
        ./Ascend310P-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}/310p_opp_kernel & \
        ./Ascend910-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}/910_opp_kernel & \
        ./Ascend310B-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}/310b_opp_kernel & \
        ./CANN-hccl-*.run --full --quiet --nox11 --install-path=${cann_install_path}/hccl
        wait
        rsync -avh --size-only ${cann_install_path}/compiler/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/opp/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/toolkit/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/aoe/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/910b_opp_kernel/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/310p_opp_kernel/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/910_opp_kernel/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/310b_opp_kernel/ ${cann_install_path}/runtime/ & \
        rsync -avh --size-only ${cann_install_path}/hccl/ ${cann_install_path}/runtime/
        wait
        mv ${cann_install_path}/runtime ${cann_install_path}
    fi
    set +e
    source /home/slave1/Ascend/ascend-toolkit/latest/bin/setenv.bash
    export ASCEND_HOME_PATH=/home/slave1/Ascend/ascend-toolkit/latest
    set -e
    cd -
}

function fn_build_googletest()
{
    GTEST_DIR=$THIRD_PARTY_DIR/googletest
    if [ -d "$GTEST_DIR" ]; then
        return $?
    fi
    [[ ! -d "$THIRD_PARTY_DIR" ]] && mkdir -p $THIRD_PARTY_DIR
    cd $THIRD_PARTY_DIR
    wget --no-check-certificate https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
    tar -xf v1.13.0.tar.gz
    mv googletest-1.13.0 googletest
    rm v1.13.0.tar.gz
}

function fn_build_nlohmann_json()
{
    if [ -d "$THIRD_PARTY_DIR/nlohmannJson" ]; then
        return $?
    fi
    cd $CACHE_DIR
    rm -rf nlohmann
    mkdir nlohmann
    cd nlohmann
    git clone -b v3.11.3 --depth 1 https://github.com/nlohmann/json.git
    mkdir -p $THIRD_PARTY_DIR/nlohmannJson
    mv json/include $THIRD_PARTY_DIR/nlohmannJson
    cd $CACHE_DIR
    rm -rf nlohmann
}

function fn_build_dependency()
{
    ARCH=$(uname -m)
    CCEC_COMPILER_DIR=$THIRD_PARTY_DIR/compiler/ccec_compiler
    TIKCPP_DIR=$THIRD_PARTY_DIR/compiler/tikcpp

    [[ -d "$THIRD_PARTY_DIR/compiler" ]] && rm -rf $THIRD_PARTY_DIR/compiler
    mkdir -p $THIRD_PARTY_DIR/compiler
    ln -s $ASCEND_HOME_PATH/compiler/ccec_compiler $CCEC_COMPILER_DIR
    ln -s $ASCEND_HOME_PATH/compiler/tikcpp $TIKCPP_DIR
}

function fn_build_release_3rdparty()
{
    fn_build_dependency
    fn_build_nlohmann_json
}

function fn_init_pytorch_env()
{
    export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import importlib.util; spec=importlib.util.find_spec("torch_npu"); \
                                                   print(spec.submodule_search_locations[0])')"
    echo "PYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH"
    echo "PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    echo "PYTORCH_NPU_INSTALL_PATH=$PYTORCH_NPU_INSTALL_PATH"
}

function fn_init_use_cxx11_abi()
{
    res=$(python3 -c "import torch" &> /dev/null || echo "torch_not_exist")
    if [ "$res" == "torch_not_exist" ]; then
        echo "Warning: Torch is not installed!"
        [[ "$USE_CXX11_ABI" == "" ]] && USE_CXX11_ABI=ON
        echo "USE_CXX11_ABI=$USE_CXX11_ABI"
        return 0
    fi

    if [ "$USE_CXX11_ABI" == "" ]; then
        if [ "$(python3 -c 'import torch; print(torch.compiled_with_cxx11_abi())')" == "True" ]; then
            USE_CXX11_ABI=ON
        else
            USE_CXX11_ABI=OFF
        fi
    fi
    echo "USE_CXX11_ABI=$USE_CXX11_ABI"
}

function fn_compile_and_install()
{
    cmake $1 $2
    if [ "$USE_VERBOSE" == "ON" ];then
        VERBOSE=1 make -j
    else
        make -j
    fi
    make install
}

function fn_config_json_copy()
{
    CONFIG_JSON_DIR=$CODE_ROOT/configs
    CONFIG_JSON_DEST=$OUTPUT_DIR/mki/configs
    if [ -d "$CONFIG_JSON_DEST" ];then
        rm -rf $CONFIG_JSON_DEST
    fi
    mkdir -p $CONFIG_JSON_DEST
    cp $CONFIG_JSON_DIR/build_config.json $CONFIG_JSON_DEST/build_config.json
}

function fn_cmake_configs_copy()
{
    CMAKE_CONFIGS_DIR=$CODE_ROOT/cmake
    CMAKE_CONFIGS_OUT_DIR=$OUTPUT_DIR/mki/cmake
    if [ -d "$CMAKE_CONFIGS_OUT_DIR" ];then
        rm -rf $CMAKE_CONFIGS_OUT_DIR
    fi
    mkdir -p $CMAKE_CONFIGS_OUT_DIR
    for cfg_name in $(ls $CMAKE_CONFIGS_DIR);do
        cp $CMAKE_CONFIGS_DIR/$cfg_name $CMAKE_CONFIGS_OUT_DIR/$cfg_name
    done
}

function fn_build()
{
    local_ascend_path="/usr/local/Ascend/ascend-toolkit"
    if [ "$IS_RELEASE" == "True" ];then
        if [ ! -d "$local_ascend_path" ];then
            fn_install_cann_and_kernel
        else
            . /usr/local/Ascend/ascend-toolkit/set_env.sh
        fi
    fi
    if [ -z $ASCEND_HOME_PATH ];then
        echo "env ASCEND_HOME_PATH not exists, build fail"
        exit 1
    fi

    [ -n "$CACHE_DIR" ] && rm -rf $CACHE_DIR
    [[ ! -d "$CACHE_DIR" ]] && mkdir $CACHE_DIR
    [[ ! -d "$OUTPUT_DIR" ]] && mkdir -p $OUTPUT_DIR
    [[ ! -d $THIRD_PARTY_DIR ]] && mkdir -p $THIRD_PARTY_DIR

    fn_build_release_3rdparty

    cd $CACHE_DIR
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_INSTALL_PREFIX=$OUTPUT_DIR/mki"
    if command -v ccache &> /dev/null; then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    fi
    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"

    echo "** Mki framework build and install!"
    fn_compile_and_install "$CODE_ROOT" "$COMPILE_OPTIONS"

    fn_config_json_copy
    fn_cmake_configs_copy
}

function fn_make_tar_package()
{
    cd $OUTPUT_DIR
    tar -czf mki.tar.gz mki
    rm -rf mki
}

function fn_main()
{
    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="dev"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]};do
            if [[ $1 =~ $item ]];then
                cfg_flag=1
                break 1
            fi
        done
        if [[ $cfg_flag == 1 ]];then
            arg1="dev"
        else
            echo "argument $1 is unknown, please type 'build.sh help' for more imformation"
            exit 1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        --output=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the output directory is not set. This should be set like --output=<outputDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export OUTPUT_DIR=$(cd $arg2; pwd)
            fi
            ;;
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            ;;
        "--verbose")
            USE_VERBOSE=ON
            ;;
        "--coverage")
            ENABLE_COVERAGE=ON
            ;;
        --namespace=*)
            ;;
        "--no_werror")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DNO_WERROR=ON"
            ;;
        "--msdebug")
            CHIP_TYPE=$(npu-smi info -m | grep -oE 'Ascend\s*\S+' | head -n 1 | tr -d ' ' | tr '[:upper:]' '[:lower:]')
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_MSDEBUG=ON -DCHIP_TYPE=${CHIP_TYPE}"
            ;;
        esac
        shift
    }
    done

    fn_init_use_cxx11_abi
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=$USE_CXX11_ABI -DENABLE_COVERAGE=$ENABLE_COVERAGE"
    case "${arg1}" in
        "testframework")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST_FRAMEWORK=ON"
            fn_init_pytorch_env
            fn_build
            ;;
        "release")
            IS_RELEASE=True
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            fn_build
            fn_make_tar_package
            ;;
        "example")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_EXAMPLE=ON"
            fn_init_pytorch_env
            fn_build
            ;;
        "dev")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            fn_build
            ;;
        "debug")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug"
            fn_build
            ;;
        "unittest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug -DBUILD_UNIT_TEST=ON"
            fn_build_googletest
            fn_build
            ;;
        "clean")
            [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR
            [[ -d "$OUTPUT_DIR" ]] && rm -rf $OUTPUT_DIR
            [[ -d $THIRD_PARTY_DIR ]] && rm -rf $THIRD_PARTY_DIR
            echo "clear all build history."
            ;;
        *)
            echo "build.sh testframework|release|example|dev|debug|unittest|clean"\
            "--output=<dir>|--force_clean|--use_cxx11_abi=0|--use_cxx11_abi=1"\
            "|--no_werror|--verbose|--coverage|--namespace=<namespace>|--msdebug"
            ;;
    esac
}

fn_main "$@"

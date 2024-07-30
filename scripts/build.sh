#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

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
DEPENDENCY_DIR=2024-04-13

MKI_SOURCE_DIR=/tmp/asdops_dependency/$DEPENDENCY_DIR

export THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
BUILD_TEST_FRAMEWORK=OFF
FORCE_CLEAN=OFF
COMPILE_OPTIONS=""
INCREMENTAL_SWITCH=OFF
DEVICE_CODE_PACK_SWITCH=ON
USE_VERBOSE=OFF
USE_CXX11_ABI=""
IS_RELEASE=False
SKIP_BUILD=OFF
CMC_URL_COMMON=https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/Baize%20C/AscendTransformerBoost/1.0.0/asdops_dependency/common
CMC_URL=https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/Baize%20C/AscendTransformerBoost/1.0.0/asdops_dependency/$DEPENDENCY_DIR
BUILD_OPTION_LIST="testframework example unittest pythontest alltest all_test_run debug release help dev clean"
BUILD_CONFIGURE_LIST=("--output=.*" "--cache=.*" "--incremental" "--gcov" "--force_clean" "--use_cxx11_abi=0"
                      "--use_cxx11_abi=1" "--build_config=.*" "--skip_build" "--no_werror" "--namespace=.*")

# install cann
function fn_install_cann_and_kernel()
{
    cd $CANN_DIR
    chmod +x *.run
    cann_install_path="/home/slave1/Ascend/ascend-toolkit"
    if [ ! -d cann_install_path ];then
        ./CANN-runtime-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-compiler-*.run --full --pylocal --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-opp-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-toolkit-*.run --full --pylocal --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-aoe-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./Ascend910B-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./Ascend310P-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-hccl-*.run --full --quiet --nox11 --install-path=${cann_install_path}
    fi
    set +e
    source /home/slave1/Ascend/ascend-toolkit/latest/bin/setenv.bash
    export ASCEND_HOME_PATH=/home/slave1/Ascend/ascend-toolkit/latest
    set -e
    cd -
}

function fn_build_nlohmann_json()
{
    if [ -d "$THIRD_PARTY_DIR/nlohmannJson" ]; then
        return $?
    fi
    cd $CACHE_DIR
    wget --no-check-certificate $CMC_URL_COMMON/nlohmannjson-v3.11.2.tar.gz
    tar -xf nlohmannjson-v3.11.2.tar.gz
    rm nlohmannjson-v3.11.2.tar.gz
    rm -rf ./nlohmannJson/.git
    mv ./nlohmannJson $THIRD_PARTY_DIR
}

function fn_build_huawei_secure_c()
{
    if [ -d "$THIRD_PARTY_DIR/securec" ]; then
        return $?
    fi
    cd $CACHE_DIR
    wget --no-check-certificate $CMC_URL_COMMON/huawei_secure_c-tag_Huawei_Secure_C_V100R001C01SPC012B002_00001.tar.gz
    tar -xf huawei_secure_c-tag_Huawei_Secure_C_V100R001C01SPC012B002_00001.tar.gz
    rm huawei_secure_c-tag_Huawei_Secure_C_V100R001C01SPC012B002_00001.tar.gz
    mv huawei_secure_c* securec
    mv securec $THIRD_PARTY_DIR
}

function fn_build_dependency()
{
    ARCH=$(uname -m)
    METADEF_DIR=$THIRD_PARTY_DIR/metadef
    CCEC_COMPILER_DIR=$THIRD_PARTY_DIR/compiler/ccec_compiler
    TIKCPP_DIR=$THIRD_PARTY_DIR/compiler/tikcpp

    if [ "$IS_RELEASE" == "True" ];then
        mkdir -p $THIRD_PARTY_DIR/compiler
        ln -s $ASCEND_HOME_PATH/compiler/ccec_compiler $CCEC_COMPILER_DIR
        ln -s $ASCEND_HOME_PATH/compiler/tikcpp $TIKCPP_DIR
    fi

    check_dependency_cache
    if [[ ! -d "$METADEF_DIR" ]]; then
        cp -r $MKI_SOURCE_DIR/metadef $METADEF_DIR
    fi
    mkdir -p $THIRD_PARTY_DIR/compiler
    if [ ! -d $CCEC_COMPILER_DIR ];then
        if [ $ARCH = "aarch64" ];then
            ln -s $MKI_SOURCE_DIR/ccec_compiler/aarch64-linux $CCEC_COMPILER_DIR
        else
            ln -s $MKI_SOURCE_DIR/ccec_compiler/x86_64-linux $CCEC_COMPILER_DIR
        fi
    fi

    if [ ! -d $TIKCPP_DIR ];then
        ln -s $MKI_SOURCE_DIR/tikcpp $TIKCPP_DIR
    fi
}

function fn_build_release_3rdparty()
{
    fn_build_dependency
    fn_build_huawei_secure_c
    fn_build_nlohmann_json
}

function fn_init_pytorch_env()
{
    export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
    echo "PYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH"
    echo "PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    if [ -f $PYTORCH_NPU_INSTALL_PATH/include/torch_npu/csrc/core/npu/NPUFormat.h ]; then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DNPU_FORMAT_H_VALID=ON"
    fi
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

function fn_platform_configs_copy()
{
    INI_DIR=$CODE_ROOT/configs/platform_configs
    PLAT_DIR=$OUTPUT_DIR/mki/configs/platform_configs
    if [ -d "$PLAT_DIR" ];then
        rm -rf $PLAT_DIR
    fi
    mkdir -p $PLAT_DIR
    for ini_name in $(ls $INI_DIR);do
        cp $INI_DIR/$ini_name $PLAT_DIR/$ini_name
    done
}

function fn_tbe_info_copy()
{
    INFO_DIR=$CODE_ROOT/configs
    INFO_DEST_DIR=$OUTPUT_DIR/mki/configs
    if [ -d "$INFO_DEST_DIR" ];then
        rm -rf $INFO_DEST_DIR
    fi
    mkdir -p $INFO_DEST_DIR
    for info_name in $(ls $INFO_DIR);do
        cp -r $INFO_DIR/$info_name $INFO_DEST_DIR/$info_name
    done
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

    if [ "$SKIP_BUILD" == "ON" ]; then
        echo "info: skip mki build because SKIP_BUILD is on."
        return 0
    fi

    if [ "$INCREMENTAL_SWITCH" == "OFF" ];then
        [ -n "$CACHE_DIR" ] && rm -rf $CACHE_DIR
    fi
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

    fn_platform_configs_copy
    fn_cmake_configs_copy
    fn_tbe_info_copy
}

function fn_build_example()
{
    cd $CODE_ROOT/example
    bash $SCRIPT_DIR/build_ops.sh --mkidir=$OUTPUT_DIR/mki --build_config=$CODE_ROOT/configs
}

function fn_make_tar_package()
{
    cd $OUTPUT_DIR
    tar -czf mki.tar.gz mki
    rm -rf mki
}

function check_dependency_cache()
{
    if [ "$IS_RELEASE" == "True" ];then
        return
    fi
    if [ "$FORCE_CLEAN" == "ON" ];then
        echo "CLEAN ALL DEPENDENCY CACHE !!!!!"
        rm $MKI_SOURCE_DIR -rf
        rm $OUTPUT_DIR -rf
        rm $CACHE_DIR -rf
    fi
    [[ ! -d "$MKI_SOURCE_DIR" ]] && mkdir -p $MKI_SOURCE_DIR
    cd $MKI_SOURCE_DIR
    if [ ! -d "$MKI_SOURCE_DIR/ccec_compiler" ]; then
        [[ ! -f $MKI_SOURCE_DIR/ccec_compiler.tar.gz ]] && wget --no-check-certificate $CMC_URL/ccec_compiler.tar.gz
        tar xf ccec_compiler.tar.gz
        rm ccec_compiler.tar.gz
    fi
    if [ ! -d "$MKI_SOURCE_DIR/metadef" ]; then
        [[ ! -f $MKI_SOURCE_DIR/metadef.tar.gz ]] && wget --no-check-certificate $CMC_URL/metadef.tar.gz
        tar xf metadef.tar.gz
    fi
    if [ ! -d "$MKI_SOURCE_DIR/tikcpp" ]; then
        [[ ! -f $MKI_SOURCE_DIR/tikcpp.tar.gz ]] && wget --no-check-certificate $CMC_URL/tikcpp.tar.gz
        tar xf tikcpp.tar.gz
        rm tikcpp.tar.gz
    fi
    echo "dependency_cache is ready"
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
        --cache=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the cache directory is not set. This should be set like --cache=<cacheDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export CACHE_DIR=$(cd $arg2; pwd)
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
        "--incremental")
            INCREMENTAL_SWITCH=ON
            ;;
        "--force_clean")
            FORCE_CLEAN=ON
            ;;
        --build_config=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the config directory is not set. This should be set like --build_config=<configFileDir>"
            else
                first_char=${arg2: 0: 1}
                if [[ "$first_char" == "/" ]];then
                    export BUILD_CONFIG_DIR=$arg2
                else
                    export BUILD_CONFIG_DIR=$CURRENT_DIR"/"$arg2
                fi
            fi
            ;;
        --namespace=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the namespace is not set. This should be set like --namespace=Mki"
            else
                COMPILE_OPTIONS="${COMPILE_OPTIONS} -DNAMESPACE:STRING=$arg2"
            fi
            ;;
        "--skip_build")
            SKIP_BUILD=ON
            ;;
        "--no_werror")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DNO_WERROR=ON"
            ;;
        esac
        shift
    }
    done

    fn_init_use_cxx11_abi
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=$USE_CXX11_ABI"
    case "${arg1}" in
        "testframework")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_TEST_FRAMEWORK=ON"
            BUILD_TEST_FRAMEWORK=ON
            fn_init_pytorch_env
            fn_build
            ;;
        "debug")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug"
            fn_build
            ;;
        "release")
            IS_RELEASE=True
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            fn_build
            fn_make_tar_package
            ;;
        "dev")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            fn_build
            ;;
        "example")
            fn_build_example
            ;;
        "clean")
            [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR
            [[ -d "$OUTPUT_DIR" ]] && rm -rf $OUTPUT_DIR
            [[ -d $THIRD_PARTY_DIR ]] && rm -rf $THIRD_PARTY_DIR
            echo "clear all build history."
            ;;
        *)
            echo "build.sh testframework|example|dev|debug|release|clean"\
            "--incremental|--force_clean|--output=<dir>|--cache=<dir>|--use_cxx11_abi=0"\
            "|--use_cxx11_abi=1|--build_config=<dir>|--skip_build|--no_werror|--namespace=<namespace>"
            ;;
    esac
}

fn_main "$@"

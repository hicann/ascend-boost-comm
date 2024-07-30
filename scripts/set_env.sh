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
path="${BASH_SOURCE[0]}"
if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    mki_path=$(cd $(dirname $path); pwd )
    export MKI_HOME_PATH="${mki_path}"
    export LD_LIBRARY_PATH=$MKI_HOME_PATH/lib:$MKI_HOME_PATH/ops_lib:$MKI_HOME_PATH/tests:$LD_LIBRARY_PATH
    export PATH=$MKI_HOME_PATH/bin:$PATH

    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    export MKI_LOG_TO_STDOUT=0
    export MKI_LOG_LEVEL=INFO
    export MKI_LOG_TO_FILE=0
    export MKI_LOG_TO_FILE_FLUSH=0
    export MKI_LOG_TO_BOOST_TYPE=atb #算子库对应加速库日志类型，默认transformer
    export MKI_LOG_PATH=~
else
    echo "There is no 'set_env.sh' to import"
fi

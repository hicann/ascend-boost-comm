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
if [[ -f "$path" ]] && [[ "$path" =~ 'set_ops_env.sh' ]];then
    mki_ops_path=$(cd $(dirname $path); pwd )
    export MKI_OPS_HOME_PATH="${mki_ops_path}"
    export LD_LIBRARY_PATH=$MKI_OPS_HOME_PATH/ops_lib:$MKI_OPS_HOME_PATH/tests:$LD_LIBRARY_PATH
else
    echo "There is no 'set_ops_env.sh' to import"
fi

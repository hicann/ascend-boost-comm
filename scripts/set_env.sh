#!/bin/bash
# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

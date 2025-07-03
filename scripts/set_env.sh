#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
path="${BASH_SOURCE[0]}"
if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    mki_path=$(cd $(dirname $path); pwd )
    export MKI_HOME_PATH="${mki_path}"
    export ATB_HOME_PATH="${mki_path}"
    export LD_LIBRARY_PATH=$MKI_HOME_PATH/lib:$MKI_HOME_PATH/tests:$LD_LIBRARY_PATH
    export PATH=$MKI_HOME_PATH/bin:$PATH

    export PYTORCH_INSTALL_PATH="$(python3 -c 'import importlib.util; SPEC=importlib.util.find_spec("torch"); \
                                               print(SPEC.submodule_search_locations[0])')"
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    export ASDOPS_LOG_TO_STDOUT=0
    export ASDOPS_LOG_LEVEL=INFO
    export ASDOPS_LOG_TO_FILE=0
    export ASDOPS_LOG_TO_FILE_FLUSH=0
    export ASDOPS_LOG_TO_BOOST_TYPE=atb #算子库对应加速库日志类型，默认transformer
    export ASDOPS_LOG_PATH=~
else
    echo "There is no 'set_env.sh' to import"
fi

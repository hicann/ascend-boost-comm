# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
from op_test import OpTest
import pandas as pd
import logging
from pathlib import Path
from typing import Union
from case_runner import __test_runner


def CaseInject(cls: Union[None, OpTest] = None, csv_path: str = ".") -> Union[OpTest, callable]:
    logging.info("loading csv testcase...")

    def __csv_filter(file_name):
        return os.path.splitext(file_name)[1].lower() == '.csv'



    def decorator(__cls: OpTest) -> OpTest:
        setattr(__cls, '__test_runner', __test_runner)
        csv_files_path = []
        if Path(csv_path).is_file():
            csv_files_path = [csv_path]
        else:

            csv_files_path = filter(__csv_filter, os.listdir(csv_path))
        setattr(__cls, 'test_cases', {})
        for file in csv_files_path:
            csv_case = pd.read_csv(file, sep='|')
            for case in csv_case.iterrows():
                case_data = case[1]
                case_name = case_data['CaseName']
                __cls.test_cases[case_name] = {
                    "case": case_data, "file_path": file}
                if getattr(__cls, f'test_{case_name}', None) is None:
                    def __test_func(self:OpTest): 
                        return self.__test_runner(case_name)
                    setattr(__cls,
                            f'test_{case_name}', __test_func)
                    logging.info(f"loading case \"{case_name}\" OK!")
                else:
                    logging.info("found duplicate case name. ignored.")
            logging.info(f"load csv {file} OK")
        return __cls
    if cls is not None:
        return decorator(cls)
    else:
        return decorator

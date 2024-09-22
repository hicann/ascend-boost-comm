# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import inspect
import logging
from pathlib import Path
from typing import Type, Union

from mkipythontest.case import Case, GeneralizedCase
from mkipythontest.optest import OpTest
from mkipythontest.utils.soc import on_soc

from .parser import BaseParser, DefaultCsvParser


def run_case(self: OpTest):
    """运行函数

    :param self: 测试类
    """
    self.run_case(self._testMethodName.replace("test_", "", 1))


def add_case(test_class: Type[OpTest], case: Union[Case, list[Case]]) -> None:
    """
    向测试类增加用例

    :param test_class: 测试类
    :param case: 用例对象或列表
    :return:
    """
    case_list = case
    if isinstance(case, Case):
        case_list = [case]
    for case in case_list:
        if case.case_name in test_class.test_cases:
            logging.info("found duplicate case name. ignored.")
        test_class.test_cases[case.case_name] = case

        __test_func = run_case
        if case.soc_version:
            __test_func = on_soc(soc_name=case.soc_version)(__test_func)
        if case.expected_error:
            pass
        setattr(test_class,
                f'test_{case.case_name}', __test_func)
        logging.info(f"loading case \"{case.case_name}\" OK!")


def case_inject(cls: Union[None, Type[OpTest]] = None,
                csv_path: str = ".",
                parser: Type[BaseParser] = DefaultCsvParser) -> Union[OpTest, callable]:
    """
    用例注入装饰器

    :param cls: 测试类
    :param csv_path: csv加载路径
    :param parser: csv解析器类
    :return: 注入用例后测试类
    """
    csv_parser: BaseParser = parser()

    def empty_decorator(test_class: Type[OpTest]) -> Type[OpTest]:
        return test_class

    def decorator(test_class: Type[OpTest]) -> Type[OpTest]:

        if not Path(csv_path).is_file():
            logging.error("not csv file")

        cases = csv_parser.parse(csv_path)
        for case in cases:
            if isinstance(case, GeneralizedCase):
                for real_case in case.to_case_list():
                    add_case(test_class, real_case)
            else:
                add_case(test_class, case)
        return test_class

    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    caller_file = caller_frame.f_globals['__file__']
    defined_file = inspect.getfile(cls)

    if caller_file != defined_file:
        logging.info("not directly run test. passed.")
        decorator = empty_decorator
    else:
        logging.info("loading csv testcase...")

    if cls is not None:
        return decorator(cls)
    else:
        return decorator

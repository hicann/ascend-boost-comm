
from functools import lru_cache
from inspect import getmembers, isfunction
from typing import Iterable, Optional, Type, Union
from unittest import TestCase, TestSuite

import pandas
from mkipythontest import OpTest
from mkipythontest.case import Case
from mkipythontest.case.injector import add_case
from mkipythontest.case.parser import BaseParser


@lru_cache
def dict_in(source: dict, target: dict) -> bool:
    """检查source是否为target的子集

    :param source: 源dict
    :param target: 目标dict
    :return: 是否为子集
    """
    for key, value in source.items():
        if key not in target:
            return False
        if target[key] != value:
            return False
    return True


@lru_cache
def get_value_partial_match(key: dict, target: dict[dict, any]) -> Optional[any]:
    """返回目标dict中子集匹配的value

    :param key: dict键
    :param target: 目标dic't
    :return: 命中返回value，未命中返回None
    """
    for key_target, value in target.items():
        if dict_in(key_target, key):
            return value
    return None


def get_test_name_list(test_class: Type[TestCase]) -> list[str]:
    """返回一个测试类中所有case的名称列表

    :param test_class: 测试类
    :return: 名称列表
    """
    test_name_list = []
    functions = getmembers(test_class, isfunction)
    for name, _ in functions:
        if name.startswith("test"):
            test_name_list.append(name)
    return test_name_list


def remove_test(test_class: Type[TestCase]) -> Type[TestCase]:
    """移除一个测试类中所有case

    :param test_class: 测试类
    :return: 清空case的测试类
    """
    test_name_list = get_test_name_list(test_class)
    for test_name in test_name_list:
        delattr(test_class, test_name)
    return test_class


class BatchTestSuite:
    def __init__(self):
        self._test_classes: dict[str, dict[dict[str, any], Type[OpTest]]] = {}

    def has_registered(self, op_name: str, restricted_param: dict) -> bool:
        """检查对应参数是否注册

        :param op_name: 算子名称
        :param restricted_param: 算子其他选择参数
        :return: 是否注册
        """
        if op_name not in self._test_classes:
            return False
        for key_param, _ in self._test_classes.items():
            if dict_in(key_param, restricted_param):
                return True
        return False

    def register_test_class(self, test_class: Type[OpTest], op_name: str, restricted_param: dict = {}):
        """测试类注册

        :param test_class: 测试类
        :param op_name: 算子名称
        :param restricted_param: 算子其他参数, defaults to {}
        """
        if self.has_registered(op_name, restricted_param):
            return
        test_class = remove_test(test_class)
        if op_name not in self._test_classes:
            self._test_classes[op_name] = {}
        self._test_classes[op_name][restricted_param] = test_class

    def get_test_classes(self, has_case: bool = False) -> list[OpTest]:
        """获取当前注册所有测试类的list

        :param has_case: 只取加载到case的类, defaults to False
        :return: 测试类列表
        """
        test_classes_list = []
        for _, value_class_param in self._test_classes.items():
            for _, value_class in value_class_param.items():
                # (~a)|(a&b)=(~a)|b
                if not has_case or get_test_name_list(value_class):
                    test_classes_list.append(value_class)
        return test_classes_list

    @lru_cache
    def get_test_class(self, op_name: str, op_param: dict) -> Optional[Type[OpTest]]:
        """
        获取测试类

        :param op_name:
        :param op_param:
        :return:
        """

        if op_name not in self._test_classes:
            return None
        return get_value_partial_match(op_param, self._test_classes[op_name])

    def add_test_csv(self, csv_path: Union[str, Iterable[str]], parser: BaseParser) -> None:
        if isinstance(csv_path, str):
            csv_path = [csv_path]
        case_list: list[Case] = []
        for csv_path_i in csv_path:
            case_list.extend(parser.parse(csv_path_i))
        for case in case_list:
            op_name = case.op_name
            op_param = case.op_param
            test_class = self.get_test_class(op_name, op_param)
            add_case(test_class, case)

    def all_test_run(self):
        batch_test_suite = TestSuite()
        for _, test_class in self._test_classes.items():
            batch_test_suite.addTest(test_class())
        batch_test_suite.run()
        
    def summarize_result(self)->pandas.DataFrame:
        pass
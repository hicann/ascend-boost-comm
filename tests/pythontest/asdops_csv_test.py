import os
from functools import lru_cache
from importlib import import_module
from typing import Type
from unittest import TestResult, TestSuite

import torch
from case import Case
from case.injector import add_case, case_inject
from case.parser import DefaultCsvParser
from op_test import OpTest

from tests.pythontest.example import Example


def args():
    pass


# 如何应对不同kernel对应不同optype的情形？（如何解决ATB石山设计）
# 解决方案：打不过就加入，使用atb石山设计，但改进一下

TEST_CLASSES = {
    "ElewiseOperation":
        {
            {"elewiseType", 1}: Example
        }
}


class BatchTestManager:
    def __init__(self):
        self.test_classes = {}

    def __dict_key_match(self, dict_param: dict, test_dict_param: dict) -> bool:
        """
        参数匹配

        :param dict_param:
        :param test_dict_param:
        :return:
        """
        for key_name, key_value in dict_param.items():
            if key_name not in test_dict_param:
                return False
            if key_value != test_dict_param[key_name]:
                return False
        return True

    def __remove_test_function(self, test_class: Type[OpTest]) -> Type[OpTest]:

        """
        删除测试函数
        """

        # 获取类的所有属性
        attrs = vars(test_class)

        # 遍历类的属性，查找以 'test' 开头的方法
        for attr_name, attr_value in list(attrs.items()):
            if callable(attr_value) and attr_name.startswith('test_'):
                delattr(test_class, attr_name)
        return test_class

    @lru_cache
    def get_test_class(self, op_name: str, op_param: dict) -> Type[OpTest]:
        """
        获取测试类

        :param op_name:
        :param op_param:
        :return:
        """
        if (op_name, op_param) in self.test_classes:
            test_class = self.test_classes[op_name, op_param]
            return self.__remove_test_function(test_class)
        for key_param, value_class in self.test_classes[op_name].items():
            if self.__dict_key_match(op_param, key_param):
                self.test_classes[op_name, op_param] = value_class
                return value_class
        return OpTest

    def add_test_case(self, case: Case) -> None:
        op_name = case.op_name
        op_param = case.op_param
        test_class = self.get_test_class(op_name, op_param)
        add_case(test_class, case)

    def get_test_suite(self) -> TestSuite:
        batch_test_suite = TestSuite()
        for _, test_class in self.test_classes.items():
            batch_test_suite.addTest(test_class())
        return batch_test_suite


if __name__ == "__main__":
    parser = DefaultCsvParser()
    batch_test_cases = parser.parse("./test_cases/batch_test_cases.csv")
    test_classes = []
    batch_test_manager = BatchTestManager()
    for case in batch_test_cases:
        batch_test_manager.add_test_case(case)
    batch_test_suite = batch_test_manager.get_test_suite()
    batch_test_suite.run()

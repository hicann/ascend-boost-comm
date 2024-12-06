# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
from time import time
from typing import Any, Callable, Optional, Type, Union
from unittest import TestSuite, TextTestRunner, defaultTestLoader

from mkipythontest import OpTest
from mkipythontest.case import Case
from mkipythontest.case.load import add_case
from mkipythontest.utils.misc import get_test_name_list
from mkipythontest.utils.singleton import Singleton


@Singleton
class BatchTestManager:
    def __init__(self):
        self._test_classes: dict[str,
                                 dict[frozenset[tuple[str, Any]], Type[OpTest]]] = {}

    def __param_match(self, 
                      op_param: dict[str, Any], 
                      registered_class_param: frozenset[tuple[str, Any]]) -> bool:
        for k, v in registered_class_param:
            # 不存在某键时生效
            if v is None and k in op_param:
                return False
            if not op_param.get(k, None) == v:
                return False
        return True

    def has_registered(self, op_name: str, op_param=None) -> bool:
        """Check if there is registered test class.

        :param op_name: operation name
        :param op_param: other specific param
        :return: if there is registered test class
        """
        if op_param is None:
            op_param = {}
        if op_name not in self._test_classes:
            return False
        for registered_class_param, _ in self._test_classes[op_name].items():
            if self.__param_match(op_param, registered_class_param):
                return True
        return False

    def register_test_class(self,
                            test_class: Type[OpTest],
                            op_name: str,
                            restricted_param=None):
        """Register a test class into the manager.

        :param test_class: test class
        :param op_name: operation name
        :param restricted_param: other specific param, defaults to {}
        """
        if restricted_param is None:
            restricted_param = {}
        if self.has_registered(op_name, restricted_param):
            return
        if op_name not in self._test_classes:
            self._test_classes[op_name] = {}
        self._test_classes[op_name][frozenset(
            (restricted_param.items()))] = test_class

    def get_test_classes(self, has_case: bool = False) -> list[Type[OpTest]]:
        """Get a list of all registered test class.

        :param has_case: return test classes with case, defaults to False
        :return: 测试类列表
        """
        test_classes_list = []
        for _, value_class_param in self._test_classes.items():
            for _, value_class in value_class_param.items():
                # (~a)|(a&b)=(~a)|b
                if not has_case or get_test_name_list(value_class):
                    test_classes_list.append(value_class)
        return list(set(test_classes_list))

    def get_test_class(self, op_name: str, op_param: dict) -> Optional[Type[OpTest]]:
        """Get test class by operation name and operation param.

        :param op_name: operation name
        :param op_param: operation param
        :return: if there is corresponding test class, return it
        """

        if op_name not in self._test_classes:
            return None
        for key_params, test_class in self._test_classes[op_name].items():
            if self.__param_match(op_param, key_params):
                return test_class
        return None

    def add_case(self, case: Case):
        """Add case to the manager

        :param case: case
        """
        op_name = case.op_name
        op_param = case.op_param
        test_class = self.get_test_class(op_name, op_param)
        if test_class is not None:
            add_case(test_class, case)

    def all_test_run(self):
        """Run all tests.
        """
        suite = TestSuite()
        for test_class in self.get_test_classes(has_case=True):
            suite.addTests(defaultTestLoader.loadTestsFromTestCase(test_class))
        TextTestRunner().run(suite)


def RegisterTestClass(op_name: str, op_param: Union[dict, list[dict]] = {}) -> Callable:
    if isinstance(op_param, dict):
        op_param = [op_param]

    def decorator(cls: Type[OpTest]) -> Type[OpTest]:
        manager = BatchTestManager()

        for param in op_param:
            manager.register_test_class(cls, op_name, param)
            logging.info(
                "Registered class %s for operation %s, param is %s", cls, op_name, param)
        return cls

    return decorator

@Singleton
class AnyValueClass:
    def __init__(self):
        self.__hash: int = int(time())

    def __eq__(self, other):
        return True

    def __hash__(self):
        return self.__hash


AnyValue = AnyValueClass()

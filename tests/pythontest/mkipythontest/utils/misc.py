# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from functools import lru_cache
from inspect import getmembers, isfunction
from typing import Callable, Iterable, Optional, Type, TypeVar
from unittest import TestCase

T = TypeVar('T')


def update_copy(src_dict: dict, update_dict: dict) -> dict:
    """Update source dict with another dict and return the updated dict

    :param src_dict: source dict
    :param update_dict: the dict contains update contents
    :return: updated dict
    """
    src_dict_copy = src_dict.copy()
    src_dict_copy.update(update_dict)
    return src_dict_copy


@lru_cache
def is_sub_dict(src_dict: dict, target_dict: dict) -> bool:
    """Check if a dict is the sub dict of another dict

    :param src_dict: source dict
    :param target_dict: the dict contains update contents
    :return: if source dict is the sub dict of target dict
    """
    return all(key in target_dict and target_dict[key] == value for key, value in src_dict.items())


@lru_cache
def get_value_partial_match(key: dict, target: dict[dict, T]) -> Optional[T]:
    """Key is a dict and target is a dict whose key is a dict.
    If a key(dict) can match the given key, return the corresponding value.

    :param key: given key(dict)
    :param target: where to find the key
    :return: return the corresponding value if there is matched key(dict)
    """
    for key_target, value in target.items():
        if is_sub_dict(key_target, key):
            return value
    return None


def split_and_map(
        input_string: str,
        mapper: Callable[[str], T],
        delimiter: str = ';',
) -> list[T]:
    """Split a string and make mapping the result list 

    :param input_string: input string
    :param mapper: mapping function
    :param delimiter: the delimiter fot spliting, defaults to ';'
    :return: output list
    """
    return list(map(mapper, input_string.split(delimiter)))


def split_and_map_dict(input_string: str,
                       mapper: dict,
                       delimiter: str = ';',
                       skip_nonexistent_key: bool = False,
                       nonexistent_key_default_value: Optional[T] = None) -> list[T]:
    result = split_and_map(input_string, lambda k: mapper.get(
        k, nonexistent_key_default_value), delimiter)
    if skip_nonexistent_key:
        try:
            result.remove(None)
        except ValueError as _:
            pass
    return result


def iterable_to_dict(iterable: Iterable[T],
                     key: Callable[[int, T], str] = lambda i, v: str(i),
                     value: Callable[[int, T], T] = lambda i, v: v):
    """Convert an iterable to a dict.

    :param iterable: iterable
    :param key: key, defaults to lambda: get the index
    :param value: value, defaults to lambda: get the value
    :return: converted dict
    """
    return {key(i, v): value(i, v) for i, v in enumerate(iterable)}


def get_test_name_list(test_class: Type[TestCase]) -> list[str]:
    """Get a list of test name in a test class

    :param test_class: test class
    :return: a list of test name
    """
    test_name_list = []
    functions = getmembers(test_class, isfunction)
    for name, _ in functions:
        if name.startswith("test"):
            test_name_list.append(name)
    return test_name_list


def remove_test(test_class: Type[TestCase]) -> Type[TestCase]:
    """Remove all test in a test class

    :param test_class: test class
    :return: test class without any test
    """
    test_name_list = get_test_name_list(test_class)
    for test_name in test_name_list:
        delattr(test_class, test_name)
    return test_class

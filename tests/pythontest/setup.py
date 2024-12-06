# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from setuptools import setup, find_packages

setup(
    name='mkipythontest',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch","numpy","pandas"
    ],
    author='yuantao',
    author_email='taoyuan15@h-partners.com',
    description='Mind-KernelInfra python test framework',
    license='Mulan PSL',
    keywords='mki python test',
    url='https://gitee.com/Ascend/Mind-KernelInfra'
)

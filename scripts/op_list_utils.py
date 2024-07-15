# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendOpCommonLib is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import yaml
import os
import re
import logging
import argparse

pattern_operation = r"add_operation\(.+?\)"
pattern_kernel = r"add_kernel\(.+?\)"
SOC_LIST = ['ascend310b', 'ascend310p', 'ascend910b']


def parse_operation(add_operation_text):
    param_list = add_operation_text.split('(')[1].strip(')').split(' ')
    return param_list[0]


def parse_kernel(add_kernel_text):
    param_list = add_kernel_text.split('(')[1].strip(')').split(' ')
    param_list = [x for x in param_list if x]
    return param_list[4], param_list[1]


def build_op_list(ops_src_root_dir, dst_yaml_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s')
    if os.path.exists(dst_yaml_path):
        os.remove(dst_yaml_path)
    op_kernel_list = {}
    for path, _, file_list in os.walk(ops_src_root_dir):
        for file in file_list:
            if file == 'CMakeLists.txt':
                file_path = os.path.join(path, file)
                with open (file_path) as fd:
                    context = fd.read().replace('\n', '')
                    match_operation = re.search(pattern_operation, context)
                    if not match_operation:
                        break
                    add_operation_text = match_operation.group()
                    match_kernel_text_list = re.findall(pattern_kernel, context)
                    if len(match_kernel_text_list) == 0:
                        break
                    op_name = parse_operation(add_operation_text)
                    op_kernel_list[op_name] = {}
                    for add_kernel_text in match_kernel_text_list:
                        kernel_name, soc = parse_kernel(add_kernel_text)
                        if kernel_name in op_kernel_list[op_name]:
                            op_kernel_list[op_name][kernel_name][soc] = True
                        else:
                            op_kernel_list[op_name][kernel_name] = {soc: True}

    with open(dst_yaml_path, 'w') as yf:
        logging.debug(yaml.dump(op_kernel_list))
        yf.write(yaml.dump(op_kernel_list, indent=4))
        logging.info(f'op list is build at {dst_yaml_path}')


def input_args_check(src_root_dir, dst_yaml_dir):
    if not os.path.exists(os.path.join(src_root_dir, 'CMakeLists.txt')):
        logging.error(f'input dir {src_root_dir} is not root directory')
        exit(1)
    if not os.path.exists(dst_yaml_dir):
        logging.error(f'dest dir {dst_yaml_dir} do not exist')
        exit(1)


def build_cmake_options(yaml_file_path, cmake_option_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s')
    if os.path.basename(yaml_file_path) != 'op_list.yaml':
        logging.error(f'{yaml_file_path}: file name error')
        exit(1)
    if not os.path.exists(os.path.dirname(cmake_option_path)):
        logging.error(f'{os.path.dirname(cmake_option_path)} do not exist')
        exit(1)
    option_list = []
    try:
        with open(yaml_file_path) as f:
            op_kernel_list = yaml.load(f, Loader=yaml.Loader)
            for op_name in op_kernel_list:
                assert op_name.endswith('Operation'), f'{op_name} is an invalid operation name'
                kernel_list = op_kernel_list[op_name]
                kernel_built_count = 0
                for kernel_name in kernel_list:
                    assert kernel_name.endswith('Kernel'), f'{kernel_name} is an invalid kernel name'
                    soc_list = kernel_list[kernel_name]
                    for soc in soc_list:
                        assert soc in SOC_LIST, f'{soc} is an invalid soc type'
                        assert isinstance(soc_list[soc], bool), f'{soc_list[soc]} is not bool type'
                        op_switch = 'ON' if soc_list[soc] else 'OFF'
                        option_name = f'BUILD_{op_name}_{kernel_name}_{soc}'
                        option_list.append(f'set({option_name} {op_switch})')
                        if soc_list[soc]:
                            kernel_built_count += 1
                operation_switch = 'ON' if kernel_built_count > 0 else 'OFF'
                option_list.append(f'set(BUILD_{op_name} {operation_switch})')

    except FileExistsError:
        logging.error(f'can not open yaml file, {yaml_file_path} do not exist')
        exit(1)
    except yaml.YAMLError as e:
        if isinstance(e, yaml.scanner.ScannerError) and hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            logging.error(f"{yaml_file_path} parsed failed at line {mark.line}, col {mark.column + 1}")
        else:
            logging.error(f'an error occurred when parse {yaml_file_path}')
        exit(1)
    except AssertionError as e:
        logging.error(e.args[0])
        exit(1)

    with open(cmake_option_path, 'w') as cmake_file:
        cmake_file.write('\n'.join(option_list))
        logging.info(f'cmake file is build at {cmake_option_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s')
    scripts_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_root_dir', type=str, required=False,
                        default=f'{scripts_dir}/../../..',
                        help='root directory that contains a CMakeLists.txt file')
    parser.add_argument('-d', '--dst_yaml_dir', type=str, required=False,
                        default=f'{scripts_dir}/../../../configs',
                        help='destination path of op list yaml file')
    input_args = parser.parse_args()
    src_root_dir = os.path.abspath(input_args.src_root_dir)
    dst_yaml_dir = os.path.abspath(input_args.dst_yaml_dir)
    input_args_check(src_root_dir, dst_yaml_dir)
    dst_yaml_path = os.path.join(dst_yaml_dir, 'op_list.yaml')
    build_op_list(src_root_dir, dst_yaml_path)

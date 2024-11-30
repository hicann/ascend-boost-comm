import json
import os
from typing import Type
import torch
from abc import ABCMeta
from mkipythontest.constant import ERROR_INFO, ErrorType


class Runner(metaclass=ABCMeta):
    def __init__(self, op_desc_json_str: str):
        pass

    def execute(self, at_in_tensors: list[torch.Tensor], at_out_tensors: list[torch.Tensor]):
        pass


TORCH_HOME_ENV = "MKI_HOME_PATH"
TORCH_SO_NAME = "tests/libmki_torch.so"
TORCH_CLASS_NAME = "MkiTorch"


class MkiTorchRunner(Runner):
    __has_loaded = False

    @classmethod
    def load_mki_torch(cls):
        MKI_HOME_PATH = os.environ.get(TORCH_HOME_ENV)
        if MKI_HOME_PATH is None:
            raise RuntimeError(
                f"env {TORCH_HOME_ENV} not exist, source set_env.sh")
        LIB_PATH = os.path.join(MKI_HOME_PATH, TORCH_SO_NAME)
        torch.classes.load_library(LIB_PATH)
        cls.__has_loaded = True

    @classmethod
    def has_loaded(cls):
        return cls.__has_loaded

    def __init__(self, op_desc_json_str: str):
        if not MkiTorchRunner.has_loaded():
            MkiTorchRunner.load_mki_torch()
        mki_class_wrapper = getattr(torch.classes, TORCH_CLASS_NAME)
        mki_class = getattr(mki_class_wrapper, TORCH_CLASS_NAME)
        self.mki = mki_class(json.dumps(op_desc_json_str))

    def execute(self, at_in_tensors: list[torch.Tensor], at_out_tensors: list[torch.Tensor]) -> ErrorType:
        run_result_value = self.mki.execute(at_in_tensors, at_out_tensors)
        # Update when MKI update
        if run_result_value == "ok":
            run_result_value = ErrorType.NO_ERROR.value
        else:
            run_result_value = ErrorType.UNDEFINED.value
        run_result = ERROR_INFO.get(run_result_value, ErrorType.UNDEFINED)
        return run_result

def get_runner()->Type[Runner]:
    runner_path = os.environ.get("MKI_RUNNER_PATH")
    if not runner_path:
        return MkiTorchRunner
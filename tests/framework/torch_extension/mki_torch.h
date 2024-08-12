/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * MindKernelInfra is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef MKI_TORCH_H
#define MKI_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <nlohmann/json.hpp>
#include "mki/kernel.h"
#include "mki/utils/SVector/SVector.h"
#include "mki_loader/ops.h"

namespace MkiTest {
class MkiTorch : public torch::CustomClassHolder {
public:
    explicit MkiTorch(std::string opDescJsonStr) : opDescJsonStr_(opDescJsonStr) {}
    ~MkiTorch() {}
    std::string Execute(std::vector<at::Tensor> atInTensors, std::vector<at::Tensor> atOutTensors);
    c10::intrusive_ptr<MkiTorch> clone() const { return c10::make_intrusive<MkiTorch>(opDescJsonStr_); }

private:
    std::string RunOp(Mki::LaunchParam &launchParam);
    std::string RunOpPerf(Mki::LaunchParam &launchParam, int runTimes);
    std::string ExecuteImpl(std::vector<at::Tensor> &atInTensors, std::vector<at::Tensor> &atOutTensors);
    void *GetCurrentStream() const;

    Mki::Tensor AtTensor2MkiTensor(const at::Tensor &atTensor);
    void ContiguousAtTensor(std::vector<at::Tensor> &atTensors);

    void SetUpTensors(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam,
                      std::vector<at::Tensor> &atInTensors, std::vector<at::Tensor> &atOutTensors);

    std::string SaveTensorsToBuf(Mki::SVector<Mki::Tensor> &mkiTensors);
    std::string GetTensorsFromBuf(Mki::SVector<Mki::Tensor> &mkiTensors);

    Mki::Kernel *GetKernelInstance(Mki::LaunchParam &launchParam);

    std::string AddWorkspace(const Mki::KernelInfo &kernelInfo, Mki::RunInfo &runInfo);
    std::string FreeWorkspace(const Mki::KernelInfo &kernelInfo, Mki::RunInfo &runInfo);

    std::string InitTilingAtMkiTorch(std::shared_ptr<Mki::Kernel> kernel, const Mki::LaunchParam &launchParam,
                                     Mki::RunInfo &runInfo);

private:
    bool perfFlag_{false};
    bool launchWithTiling_{true};
    uint8_t *deviceLaunchBuffer_{nullptr};
    uint8_t *tensorTempBufList_[20];
    std::string opDescJsonStr_;
    std::string opName_;
};
}

#endif

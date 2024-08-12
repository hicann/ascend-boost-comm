/*
 * copyright (c) 2024 Huawei Technologies Co., Ltd.
 * MindKernelInfra is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *             http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include "mki/base/operation_base.h"
#include "mki/base/kernel_base.h"
#include "mki/kernel.h"
#include "mki/kernel_info.h"

namespace {
	using namespace Mki;

	class KernelNew : public Kernel{
		public:
			KernelNew(const std::string& kernelName) :  Kernel(), kernelName_(kernelName) {}
			~KernelNew() override{};
			Kernel *Clone() const override { 
				Mki::Kernel *kernel = new KernelNew("Test");
				return kernel; 
			};
			void Reset() override{};
			bool CanSupport(const LaunchParam &launchParam) const override { return true; };
			uint64_t GetTilingSize(const LaunchParam &launchParam) const override { return 1; };
			Status Init(const LaunchParam &launchParam) override { return Status::OkStatus(); };
			Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override { return Status::OkStatus(); };
			void SetLaunchWithTiling(bool flag) override{};
			void SetTilingHostAddr(uint8_t *addr, uint64_t len) override{};
			std::string GetName() const override { return kernelName_; };
			const KernelInfo &GetKernelInfo() const override {return kernelInfo_;};
			KernelType GetType() const override { return Mki::KERNEL_TYPE_AI_CORE; };

		private:
			Mki::KernelInfo kernelInfo_;
			std::string kernelName_;
	};

	TEST(KernelBaseTest,all){
		LaunchParam launchParam;
		Tensor inTensor;
		Tensor outTensor=inTensor;
		launchParam.AddInTensor(inTensor);
		launchParam.AddOutTensor(outTensor);
		Mki::KernelInfo kernelInfo;

		Mki::Kernel *kernel = new KernelNew("Test");
		EXPECT_EQ(kernel->CanSupport(launchParam),true);
		EXPECT_EQ(kernel->GetTilingSize(launchParam),1);
		EXPECT_EQ(kernel->GetName(), "Test");
		EXPECT_EQ(kernel->GetType(), Mki::KERNEL_TYPE_AI_CORE);
	}
}// namespace

/*copyright (c) 2024 Huawei Technologies Co., Ltd.
 * MindKernelInfra is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *              http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include "mki/bin_handle.h"

namespace{
	using namespace Mki;

	class BinHandleTest : public ::testing::Test {
		protected:
			void SetUp() override {
				BinaryBasicInfo binInfo;
				binInfo.binaryBuf = reinterpret_cast<const uint8_t*>("binary data");
				binInfo.binaryLen = 10;
				binInfo.targetSoc = "target_soc";

				binHandle = new BinHandle(&binInfo);
			}

			void TearDown() override {
				delete binHandle;
			}

			BinHandle* binHandle;
	};

	TEST_F(BinHandleTest, ConstructorAndDestructor) {
		EXPECT_NE(binHandle->GetHandle(), nullptr);
	}

	TEST_F(BinHandleTest, Init) {
		std::string kernelName = "test_binhandle";
		EXPECT_EQ(binHandle->Init(kernelName),false);
	}

	TEST_F(BinHandleTest, GetKernelTilingSize) {
		std::string kernelName = "test_binhandle";
		EXPECT_EQ(binHandle->Init(kernelName),false);
		EXPECT_EQ(binHandle->GetKernelTilingSize(), 0);
	}

	TEST_F(BinHandleTest, GetKernelCoreType) {
		std::string kernelName = "test_binhandle";
		EXPECT_EQ(binHandle->Init(kernelName),false);
		EXPECT_EQ(binHandle->GetKernelCoreType(), -1);
	}

	TEST_F(BinHandleTest, GetKernelCompileInfo) {
		std::string kernelName = "test_binhandle";
		EXPECT_EQ(binHandle->Init(kernelName),false);
		EXPECT_NE(binHandle->GetKernelCompileInfo(), "");
	}
} // namespace

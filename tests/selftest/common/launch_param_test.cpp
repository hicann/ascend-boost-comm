#include <gtest/gtest.h>
#include "mki/launch_param.h"

namespace{
class LaunchParamTest : public ::testing::Test {
	protected:
		Mki::LaunchParam launchParam;
		Mki::Tensor tensor1, tensor2;

		void SetUp() override {
			tensor1.desc = Mki::TensorDesc{};
			tensor1.deviceData = malloc(128);  // 示例数据
			tensor1.dataSize = 128;
			tensor1.hostData = malloc(128);

			tensor2.desc = Mki::TensorDesc{};
			tensor2.deviceData = malloc(256);  // 示例数据
			tensor2.dataSize = 256;
			tensor2.hostData = malloc(256);
		}

		void TearDown() override {
			free(tensor1.deviceData);
			free(tensor1.hostData);
			free(tensor2.deviceData);
			free(tensor2.hostData);
		}
};

TEST_F(LaunchParamTest, AddInputTensor) {
	EXPECT_EQ(launchParam.GetInTensorCount(), 0);
	launchParam.AddInTensor(tensor1);
	EXPECT_EQ(launchParam.GetInTensorCount(), 1);
	launchParam.AddInTensor(tensor2);
	EXPECT_EQ(launchParam.GetInTensorCount(), 2);
}

TEST_F(LaunchParamTest, GetInputTensor) {
	launchParam.AddInTensor(tensor1);
	launchParam.AddInTensor(tensor2);
	EXPECT_EQ(launchParam.GetInTensor(0).dataSize, tensor1.dataSize);
	EXPECT_EQ(launchParam.GetInTensor(1).dataSize, tensor2.dataSize);
}

TEST_F(LaunchParamTest, AddOutputTensor) {
	EXPECT_EQ(launchParam.GetOutTensorCount(), 0);
	launchParam.AddOutTensor(tensor1);
	EXPECT_EQ(launchParam.GetOutTensorCount(), 1);
	launchParam.AddOutTensor(tensor2);
	EXPECT_EQ(launchParam.GetOutTensorCount(), 2);
}

TEST_F(LaunchParamTest, GetOutputTensor) {
	launchParam.AddOutTensor(tensor1);
	launchParam.AddOutTensor(tensor2);
	EXPECT_EQ(launchParam.GetOutTensor(0).dataSize, tensor1.dataSize);
	EXPECT_EQ(launchParam.GetOutTensor(1).dataSize, tensor2.dataSize);
}

TEST_F(LaunchParamTest, Reset) {
	launchParam.AddInTensor(tensor1);
	launchParam.AddOutTensor(tensor2);
	EXPECT_EQ(launchParam.GetInTensorCount(), 1);
	EXPECT_EQ(launchParam.GetOutTensorCount(), 1);

	launchParam.Reset();
	EXPECT_EQ(launchParam.GetInTensorCount(), 0);
	EXPECT_EQ(launchParam.GetOutTensorCount(), 0);
}

TEST_F(LaunchParamTest, AssignmentOperator) {
	Mki::LaunchParam launchParam2;
	launchParam.AddInTensor(tensor1);
	launchParam.AddOutTensor(tensor2);

	launchParam2 = launchParam;

	EXPECT_EQ(launchParam2.GetInTensorCount(), 1);
	EXPECT_EQ(launchParam2.GetOutTensorCount(), 1);
	EXPECT_EQ(launchParam2.GetInTensor(0).dataSize, tensor1.dataSize);
	EXPECT_EQ(launchParam2.GetOutTensor(0).dataSize, tensor2.dataSize);
}

TEST_F(LaunchParamTest, ToString) {
	launchParam.AddInTensor(tensor1);
	launchParam.AddOutTensor(tensor2);

	std::string result = launchParam.ToString();
	EXPECT_NE(result.find("intensors[0]"), std::string::npos);
	EXPECT_NE(result.find("outtensors[0]"), std::string::npos);
}
} // namespace

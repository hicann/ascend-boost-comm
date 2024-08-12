#include <gtest/gtest.h>
#include "mki/types.h"
namespace{
using namespace Mki;
TEST(GetTensorElementSizeTest, HandleValidInput)
{
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_UNDEFINED), 0);
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_FLOAT), sizeof(float));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_FLOAT16), 2);
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_INT8), sizeof(int8_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_INT32), sizeof(int32_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_UINT8), sizeof(uint8_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_INT16), sizeof(int16_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_UINT16), sizeof(uint16_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_UINT32), sizeof(uint32_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_INT64), sizeof(int64_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_UINT64), sizeof(uint64_t));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_DOUBLE), sizeof(double));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_BOOL), sizeof(bool));
	EXPECT_EQ(GetTensorElementSize(TensorDType::TENSOR_DTYPE_BF16), 2);
}

TEST(GetTensorElementSizeTest, HandleInvalidInput){
	TensorDType invalid_value = static_cast<TensorDType>(999);
	EXPECT_EQ(GetTensorElementSize(invalid_value), 0);
}

TEST(GetDTypeWithStrTest, HandleValidInput){
	EXPECT_EQ(GetDTypeWithStr("float"), TensorDType::TENSOR_DTYPE_FLOAT);
	EXPECT_EQ(GetDTypeWithStr("float16"), TensorDType::TENSOR_DTYPE_FLOAT16);
	EXPECT_EQ(GetDTypeWithStr("int8"), TensorDType::TENSOR_DTYPE_INT8);
	EXPECT_EQ(GetDTypeWithStr("int32"), TensorDType::TENSOR_DTYPE_INT32);
	EXPECT_EQ(GetDTypeWithStr("uint8"), TensorDType::TENSOR_DTYPE_UINT8);
	EXPECT_EQ(GetDTypeWithStr("int16"), TensorDType::TENSOR_DTYPE_INT16);
	EXPECT_EQ(GetDTypeWithStr("uint16"), TensorDType::TENSOR_DTYPE_UINT16);
	EXPECT_EQ(GetDTypeWithStr("uint32"), TensorDType::TENSOR_DTYPE_UINT32);
	EXPECT_EQ(GetDTypeWithStr("int64"), TensorDType::TENSOR_DTYPE_INT64);
	EXPECT_EQ(GetDTypeWithStr("uint64"), TensorDType::TENSOR_DTYPE_UINT64);
	EXPECT_EQ(GetDTypeWithStr("double"), TensorDType::TENSOR_DTYPE_DOUBLE);
	EXPECT_EQ(GetDTypeWithStr("bool"), TensorDType::TENSOR_DTYPE_BOOL);
	EXPECT_EQ(GetDTypeWithStr("string"), TensorDType::TENSOR_DTYPE_STRING);
	EXPECT_EQ(GetDTypeWithStr("complex64"), TensorDType::TENSOR_DTYPE_COMPLEX64);
	EXPECT_EQ(GetDTypeWithStr("complex128"), TensorDType::TENSOR_DTYPE_COMPLEX128);
	EXPECT_EQ(GetDTypeWithStr("bf16"), TensorDType::TENSOR_DTYPE_BF16);
}

TEST(GetDTypeWithStrTest, HandleInvalidInput){
	std::string invalid_value = "999";
	EXPECT_EQ(GetDTypeWithStr(invalid_value), TensorDType::TENSOR_DTYPE_UNDEFINED);
}

TEST(GetStrWithDTypeTest, HandleValidInput){
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_FLOAT), "float");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_FLOAT16), "float16");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_INT8), "int8");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_INT32), "int32");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_UINT8), "uint8");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_INT16), "int16");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_UINT16), "uint16");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_UINT32), "uint32");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_INT64), "int64");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_UINT64), "uint64");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_DOUBLE), "double");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_BOOL), "bool");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_STRING), "string");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_COMPLEX64), "complex64");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_COMPLEX128), "complex128");
	EXPECT_EQ(GetStrWithDType(TensorDType::TENSOR_DTYPE_BF16), "bf16");
}

TEST(GetStrWithDTypeTest, HandleInvalidInput){
	int invalid_value = 999;
	EXPECT_EQ(GetStrWithDType(invalid_value), "undefined");
}

TEST(GetFormatWithStrTest, HandleValidInput){
	EXPECT_EQ(GetFormatWithStr("nchw"), TensorFormat::TENSOR_FORMAT_NCHW);
	EXPECT_EQ(GetFormatWithStr("nchc"), TensorFormat::TENSOR_FORMAT_NHWC);
	EXPECT_EQ(GetFormatWithStr("nd"), TensorFormat::TENSOR_FORMAT_ND);
	EXPECT_EQ(GetFormatWithStr("nc1hc0"), TensorFormat::TENSOR_FORMAT_NC1HWC0);
	EXPECT_EQ(GetFormatWithStr("fractal_z"), TensorFormat::TENSOR_FORMAT_FRACTAL_Z);
	EXPECT_EQ(GetFormatWithStr("nc1hwc0_c04"), TensorFormat::TENSOR_FORMAT_NC1HWC0_C04);
	EXPECT_EQ(GetFormatWithStr("hwcn"), TensorFormat::TENSOR_FORMAT_HWCN);
	EXPECT_EQ(GetFormatWithStr("ndhwc"), TensorFormat::TENSOR_FORMAT_NDHWC);
	EXPECT_EQ(GetFormatWithStr("fractal_nz"), TensorFormat::TENSOR_FORMAT_FRACTAL_NZ);
	EXPECT_EQ(GetFormatWithStr("ncdhw"), TensorFormat::TENSOR_FORMAT_NCDHW);
	EXPECT_EQ(GetFormatWithStr("ndc1hwc0"), TensorFormat::TENSOR_FORMAT_NDC1HWC0);
	EXPECT_EQ(GetFormatWithStr("fractal_z_3d"), TensorFormat::TENSOR_FORMAT_FRACTAL_Z_3D);
}

TEST(GetFormatWithStrTest, HandleInvalidInput){
	std::string invalid_value = "999";
	EXPECT_EQ(GetFormatWithStr(invalid_value), TensorFormat::TENSOR_FORMAT_UNDEFINED);
}

TEST(GetStrWithFormatTest, HandleValidInput){
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_NCHW), "nchw");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_NHWC), "nchc");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_ND), "nd");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_NC1HWC0), "nc1hc0");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_FRACTAL_Z), "fractal_z");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_NC1HWC0_C04), "nc1hwc0_c04");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_HWCN), "hwcn");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_NDHWC), "ndhwc");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_FRACTAL_NZ), "fractal_nz");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_NCDHW), "ncdhw");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_NDC1HWC0), "ndc1hwc0");
	EXPECT_EQ(GetStrWithFormat(TensorFormat::TENSOR_FORMAT_FRACTAL_Z_3D), "fractal_z_3d");
}

TEST(GetStrWithFormatTest, HandleInvalidInput){
	int invalid_value = 999;
	EXPECT_EQ(GetStrWithFormat(invalid_value), "undefined");
}
} // namespace

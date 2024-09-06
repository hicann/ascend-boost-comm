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
#include <gtest/gtest.h>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include "mki/utils/operationir/operation_ir.h"
#include "mki/utils/operationir/operation_ir_cfg.h"

namespace Mki {

TEST(TestOperationIr, ParseIniSuccess)
{
    char buf[256];
    auto result = getcwd(buf, sizeof(buf));
    ASSERT_NE(result, nullptr);
    std::string current_path(buf);
    std::string fileName = current_path + "/test2.ini";
    std::ofstream file(fileName);
    file << "[TestOperation1] \n input0.name=x \n input0.dtype=float,bf16 \n input0.format=nd,nd \n "
        "input0.optional=true\n input1.name=y \n input1.dtype=float,float16 \n input1.format=nd,fractal_nz \n "
        "output0.name=out \n output0.dtype=float,bf16 \n output0.format=nd,nd\n output0.optional=true\n "
        "[TestOperation2] \n input0.name=in \n input0.dtype=float,bf16 \n input0.format=nd\n input0.optional=true\n "
        "output0.name=out \n output0.dtype=float,bf16 \n output0.format=nd,nd\n output0.optional=true";
    file.close();

    OperationIrCfg opIrCfg;
    Status st = opIrCfg.Load(fileName);
    ASSERT_EQ(st.Ok(), true);
    OperationIr *opIr = nullptr;
    opIr = opIrCfg.GetOperationIr("TestOperation1");
    ASSERT_NE(opIr, nullptr);
    ASSERT_EQ(opIr->IsValid(), true);
    ASSERT_EQ(opIr->GetSupportSize(), 2);
    ASSERT_EQ(opIr->GetInTensorInfoIrs().size(), 2);
    ASSERT_EQ(opIr->GetInTensorInfoIrs()[0].isOptional, true);
    EXPECT_EQ(opIr->GetInTensorInfoIrs()[0].supportedDtypes,
              (std::vector<TensorDType>{TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_BF16}));
    EXPECT_EQ(opIr->GetInTensorInfoIrs()[0].supportedFormats,
              (std::vector<TensorFormat>{TENSOR_FORMAT_ND, TENSOR_FORMAT_ND}));
    ASSERT_EQ(opIr->GetInTensorInfoIrs()[1].isOptional, false);
    EXPECT_EQ(opIr->GetInTensorInfoIrs()[1].supportedDtypes,
              (std::vector<TensorDType>{TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_FLOAT16}));
    EXPECT_EQ(opIr->GetInTensorInfoIrs()[1].supportedFormats,
              (std::vector<TensorFormat>{TENSOR_FORMAT_ND, TENSOR_FORMAT_FRACTAL_NZ}));
    ASSERT_EQ(opIr->GetOutTensorInfoIrs().size(), 1);
    ASSERT_EQ(opIr->GetOutTensorInfoIrs()[0].isOptional, true);
    EXPECT_EQ(opIr->GetOutTensorInfoIrs()[0].supportedDtypes,
              (std::vector<TensorDType>{TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_BF16}));
    EXPECT_EQ(opIr->GetOutTensorInfoIrs()[0].supportedFormats,
              (std::vector<TensorFormat>{TENSOR_FORMAT_ND, TENSOR_FORMAT_ND}));

    opIr = opIrCfg.GetOperationIr("TestOperation2");
    ASSERT_NE(opIr, nullptr);
    ASSERT_EQ(opIr->IsValid(), false);
    std::remove(fileName.c_str());
}

TEST(TestOperationIr, ParseIniInvalid)
{
    char buf[256];
    auto result = getcwd(buf, sizeof(buf));
    ASSERT_NE(result, nullptr);
    std::string current_path(buf);
    std::string fileName = current_path + "/test2.ini";
    std::ofstream file(fileName);
    // invalid index
    file << "[TestOperationInvalid1] \n input0.name=x \n input99.dtype=float,bf16 \n input0.format=nd,nd";
    file.close();
    OperationIrCfg opIrCfg;
    Status st = opIrCfg.Load(fileName);
    ASSERT_EQ(st.Ok(), false);

    // invalid file
    st = opIrCfg.Load("invalid.ini");
    ASSERT_EQ(st.Ok(), false);

    file.open(fileName, std::ios::trunc);
    // invalid name
    file << "[] \n input0.dtype=float,bf16 \n";
    file.close();
    st = opIrCfg.Load(fileName);
    ASSERT_EQ(st.Ok(), false);

    file.open(fileName, std::ios::trunc);
    // invalid name
    file << "[?????] \n input0.dtype=float,bf16 \n";
    file.close();
    st = opIrCfg.Load(fileName);
    ASSERT_EQ(st.Ok(), false);

    file.open(fileName, std::ios::trunc);
    // invalid dtype
    file << "[TestOperationInvalid] \n input0.dtype=float16,int8,int32,bool,string,complex64,float88,bf16 \n";
    file.close();
    st = opIrCfg.Load(fileName);
    ASSERT_EQ(st.Ok(), false);

    file.open(fileName, std::ios::trunc);
    // invalid format
    file << "[TestOperationInvalid] \n "
            "input0.dtype=float,bf16,uint8,int16,uint16,uint32,int64,uint64,double,complex128 \n "
            "input0.format=nchw,nhwc,nc1hwc0,fractal_z,nc1hwc0_c04,hwcn,ndhwc,fractal_nz,ncdhw,"
            "ndc1hwc0,fractal_z_3d,aaa,nd ";
    file.close();
    st = opIrCfg.Load(fileName);
    ASSERT_EQ(st.Ok(), false);

    std::remove(fileName.c_str());
}
} // namespace Mki

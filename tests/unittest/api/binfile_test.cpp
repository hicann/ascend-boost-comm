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
#include "mki/utils/bin_file/bin_file.h"
#include "mki/utils/file_system/file_system.h"

namespace Mki {
TEST(BinFileTest, all)
{
    {
        BinFile binFile;
        binFile.AddAttr("attr1", "3");
        char objBuffer[10];
        binFile.AddObject("obj1", objBuffer, sizeof(objBuffer));
        binFile.Write("1.bin");
    }
    {
        BinFile binFile;
        binFile.Read("1.bin");
        std::vector<std::pair<std::string, std::string>> attrMap;
        binFile.GetAllAttrs(attrMap);
        EXPECT_EQ(attrMap.size(), 1);
        EXPECT_EQ(attrMap[0].second, "3");

        std::vector<std::pair<std::string, std::pair<char *, uint64_t>>> objMap;
        binFile.GetAllObjects(objMap);
        EXPECT_EQ(objMap.size(), 1);
        EXPECT_EQ(objMap[0].second.second, 10);
        FileSystem::DeleteFile("1.bin");
    }
}

TEST(BinFileTest, all0)
{
    {
        BinFile binFile;
        binFile.AddAttr("attr1", "3");
        char objBuffer[10];
        binFile.AddObject("obj1", objBuffer, sizeof(objBuffer));
        binFile.Write("2.bin");
    }
    {
        BinFile binFile;
        binFile.Read("2.bin");
        std::vector<std::pair<std::string, std::string>> attrMap;
        binFile.GetAllAttrs(attrMap);
        EXPECT_EQ(attrMap.size(), 1);
        EXPECT_EQ(attrMap[0].second, "3");

        std::vector<std::pair<std::string, std::pair<char *, uint64_t>>> objMap;
        binFile.GetAllObjects(objMap);
        EXPECT_EQ(objMap.size(), 1);
        EXPECT_EQ(objMap[0].second.second, 10);
        FileSystem::DeleteFile("2.bin");
    }
}

TEST(BinFileTest, all1)
{
    {
        BinFile binFile;
        binFile.AddAttr("attr1", "3");
        char objBuffer[10];
        binFile.AddObject("obj1", objBuffer, sizeof(objBuffer));
        binFile.Write("3.bin");
    }
    {
        BinFile binFile;
        binFile.Read("3.bin");
        std::vector<std::pair<std::string, std::string>> attrMap;
        binFile.GetAllAttrs(attrMap);
        EXPECT_EQ(attrMap.size(), 1);
        EXPECT_EQ(attrMap[0].second, "3");

        std::vector<std::pair<std::string, std::pair<char *, uint64_t>>> objMap;
        binFile.GetAllObjects(objMap);
        EXPECT_EQ(objMap.size(), 1);
        EXPECT_EQ(objMap[0].second.second, 10);
        FileSystem::DeleteFile("3.bin");
    }
}

TEST(CheckNameTest, EmptyName) {
    BinFile file;
    std::string name = "";
    const std::string value = "1";
    Status status = file.AddAttr(name, value);
    EXPECT_FALSE(status.Ok());
    EXPECT_EQ(file.AddAttr(name, value).Ok(), false);
}

TEST(CheckNameTest, NameStartsWithDollar) {
    BinFile file;
    std::string name = "$John";
    const std::string value = "1";
    Status status = file.AddAttr(name, value);
    EXPECT_FALSE(status.Ok());
    EXPECT_EQ(file.AddAttr(name, value).Ok(), false);
}

TEST(CheckNameTest, ValidName) {
    BinFile file;
    std::string name = "John";
    const std::string value = "1";
    Status result = file.AddAttr(name, value);
    EXPECT_TRUE(result.Ok());
}

TEST(CheckNameTest, ValidName0)
{
    BinFile file;
    std::string name = "test_object";
    uint64_t objLen = 0;
    std::vector<char> objBuffer(1, 'a');
    Status status = file.AddObject(name, objBuffer.data(), objLen);
    EXPECT_EQ(status.Ok(), true);
}

TEST(CheckNameTest, ValidName2)
{
    BinFile file;
    std::string name = "John";
    const std::string value = "1";
    Status result = file.AddAttr(name, value);
    EXPECT_TRUE(result.Ok());
    result = file.AddAttr(name, value);
    EXPECT_EQ(result.Message(), "attr:John already exists");
}

TEST(CheckNameTest, Read)
{
    BinFile file;
    std::string name = "test_object";
    Status result = file.Read(name);
    EXPECT_EQ(result.Ok(), false);
}

TEST(FileSystemTest, Write)
{
    BinFile file;
    std::string dirPath = "";
    Status result = file.Write(dirPath);
    EXPECT_EQ(result.Message(), "open file fail");
}

TEST(BinFileTest, invalidChar)
{
    {
        BinFile binFile;
        binFile.AddAttr("$asd", "3");
        char objBuffer[10];
        binFile.AddObject("obj1", objBuffer, sizeof(objBuffer));
        binFile.Write("1.bin");
    }
    {
        BinFile binFile;
        binFile.Read("1.bin");
        std::vector<std::pair<std::string, std::string>> attrMap;
        binFile.GetAllAttrs(attrMap);
        EXPECT_EQ(attrMap.size(), 0);
    }
}

TEST(BinFileTest, emptyChar)
{
    {
        BinFile binFile;
        binFile.AddAttr("", "3");
        char objBuffer[10];
        binFile.AddObject("obj1", objBuffer, sizeof(objBuffer));
        binFile.Write("1.bin");
    }
    {
        BinFile binFile;
        binFile.Read("1.bin");
        std::vector<std::pair<std::string, std::string>> attrMap;
        binFile.GetAllAttrs(attrMap);
        EXPECT_EQ(attrMap.size(), 0);
    }
}

TEST(BinFileTest, exceedMaxLen)
{
    {
        BinFile binFile;
        std::string name(300, ' ');
        binFile.AddAttr(name, "s");
        char objBuffer[10];
        binFile.AddObject("obj1", objBuffer, sizeof(objBuffer));
        binFile.Write("1.bin");
    }
    {
        BinFile binFile;
        binFile.Read("1.bin");
        std::vector<std::pair<std::string, std::string>> attrMap;
        binFile.GetAllAttrs(attrMap);
        EXPECT_EQ(attrMap.size(), 0);
    }
}
} // namespace Mki

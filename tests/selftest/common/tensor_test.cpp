#include <gtest/gtest.h>
#include "mki/tensor.h"
#include "../schedule/common/tensor.cpp"
namespace{
TEST(CalNumelTest, HandleValidInput){
	Mki::SVector<int64_t> v = {2, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 24);

	v = {1, 2, 3};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 6);

	v = {1000, 1000, 1000};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 1000000000);
}

TEST(CalNumelTest, HandleInvalidInput){
	Mki::SVector<int64_t> v = {-1, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 0);

	v = {2, 0, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 0);

	v = {2, std::numeric_limits<int64_t>::max(), 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 0);
}

TEST(CalNumelTest, HandleSubrangeInput){
	Mki::SVector<int64_t> v = {2, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 2), 6);

	v = {2, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 1, 3), 12);
}

TEST(ViewTest, HandleValidInput) {
	Mki::TensorDesc td;
	td.dims = {3, 2, 4};
	Mki::SVector<int64_t> ms = {2, 3, 4};
	td.View(ms);
	EXPECT_EQ(td.dims, ms);

	Mki::SVector<int64_t> ms2 = {1, 2, 3};
	td.View(ms2);
	EXPECT_NE(td.dims, ms2);
}

TEST(ViewTest, HandleInvalidInput) {
	Mki::TensorDesc td;
	Mki::SVector<int64_t> ms = {2, 3, 4};
	td.View(ms);
	EXPECT_TRUE(td.dims.empty());
}

TEST(ToStringTest, HandleEmptyDims){
	Mki::TensorDesc td;
	td.dims = {};
	EXPECT_EQ(td.ToString(), "dtype:undefined, format:undefined, dims:[]");
}

TEST(ToStringTest, HandleSingleElementDims){
	Mki::TensorDesc td;
	td.dims = {10};
	EXPECT_EQ(td.ToString(), "dtype:undefined, format:undefined, dims:[10]");
}

TEST(ToStringTest, HandleMultipleElementsDims){
	Mki::TensorDesc td;
	td.dims = {1, 2, 3};
	EXPECT_EQ(td.ToString(), "dtype:undefined, format:undefined, dims:[1, 2, 3]");
}
} //namespace

//
// Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MKI_UTILS_FP16_T_H
#define MKI_UTILS_FP16_T_H

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <ostream>

namespace Mki {
/**
 * @ingroup Fp16T
 * @brief   Half precision float
 *          bit15:       1 bit SIGN      +---+-----+------------+
 *          bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
 *          bit0-9:      10bit MAN       +---+-----+------------+
 */
struct Fp16T {
    uint16_t val = 0;

public:
    /**
     * @ingroup Fp16T constructor
     * @brief   Constructor without any param(default constructor)
     */
    Fp16T(void) { val = 0x0u; }

    /**
     * @ingroup all type constructor
     * @brief   Constructor with all type
     */
    template <typename T>
    explicit Fp16T(const T& value) { *this = value; }

    /**
     * @ingroup Fp16T constructor
     * @brief   Constructor with an uint16_t value
     */
    explicit Fp16T(const uint16_t& uiVal) : val(uiVal) {}

    /**
     * @ingroup Fp16T constructor
     * @brief   Constructor with a Fp16T object(copy constructor)
     */
    Fp16T(const Fp16T& fp) : val(fp.val) {}

    /**
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be added
     * @brief   Override addition operator to performing Fp16T addition
     * @return  Return Fp16T result of adding this and fp
     */
    Fp16T operator+(const Fp16T fp) const;

    /**
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be subtracted
     * @brief   Override addition operator to performing Fp16T subtraction
     * @return  Return Fp16T result of subtraction fp from this
     */
    Fp16T operator-(const Fp16T fp) const;

    /**
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be multiplied
     * @brief   Override multiplication operator to performing Fp16T multiplication
     * @return  Return Fp16T result of multiplying this and fp
     */
    Fp16T operator*(const Fp16T fp) const;

    /**
     * @ingroup Fp16T math operator divided
     * @param [in] fp Fp16T object to be divided
     * @brief   Override division operator to performing Fp16T division
     * @return  Return Fp16T result of division this by fp
     */
    Fp16T operator/(const Fp16T fp) const;

    /**
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be added
     * @brief   Override addition operator to performing Fp16T addition
     * @return  Return Fp16T result of adding this and fp
     */
    Fp16T operator+=(const Fp16T fp);

    /**
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be subtracted
     * @brief   Override addition operator to performing Fp16T subtraction
     * @return  Return Fp16T result of subtraction fp from this
     */
    Fp16T operator-=(const Fp16T fp);

    /**
     * @ingroup Fp16T math operator
     * @param [in] fp Fp16T object to be multiplied
     * @brief   Override multiplication operator to performing Fp16T multiplication
     * @return  Return Fp16T result of multiplying this and fp
     */
    Fp16T operator*=(const Fp16T fp);

    /**
     * @ingroup Fp16T math operator divided
     * @param [in] fp Fp16T object to be divided
     * @brief   Override division operator to performing Fp16T division
     * @return  Return Fp16T result of division this by fp
     */
    Fp16T operator/=(const Fp16T fp);

    /**
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T if-equal comparison
     * @return  Return boolean result of if-equal comparison of this and fp.
     */
    bool operator==(const Fp16T& fp) const;

    /**
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T not-equal comparison
     * @return  Return boolean result of not-equal comparison of this and fp.
     */
    bool operator!=(const Fp16T& fp) const;

    /**
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T greater-than comparison
     * @return  Return boolean result of greater-than comparison of this and fp.
     */
    bool operator>(const Fp16T& fp) const;

    /**
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T greater-equal comparison
     * @return  Return boolean result of greater-equal comparison of this and fp.
     */
    bool operator>=(const Fp16T& fp) const;

    /**
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T less-than comparison
     * @return  Return boolean result of less-than comparison of this and fp.
     */
    bool operator<(const Fp16T& fp) const;

    /**
     * @ingroup Fp16T math compare operator
     * @param [in] fp Fp16T object to be compared
     * @brief   Override basic comparison operator to performing Fp16T less-equal comparison
     * @return  Return boolean result of less-equal comparison of this and fp.
     */
    bool operator<=(const Fp16T& fp) const;

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] fp Fp16T object to be copy to Fp16T
     * @brief   Override basic evaluation operator to copy Fp16T to a new Fp16T
     * @return  Return Fp16T result from fp
     */
    Fp16T& operator=(const Fp16T& fp);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] fVal float object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert float to Fp16T
     * @return  Return Fp16T result from fVal
     */
    Fp16T& operator=(const float& fVal);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] dVal double object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert double to Fp16T
     * @return  Return Fp16T result from dVal
     */
    Fp16T& operator=(const double& dVal);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] iVal float object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert float to Fp16T
     * @return  Return Fp16T result from iVal
     */
    Fp16T& operator=(const int8_t& iVal);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] uiVal uint8_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert uint8_t to Fp16T
     * @return  Return Fp16T result from uiVal
     */
    Fp16T& operator=(const uint8_t& uiVal);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] iVal int16_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert int16_t to Fp16T
     * @return  Return Fp16T result from iVal
     */
    Fp16T& operator=(const int16_t& iVal);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] uiVal uint16_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert uint16_t to Fp16T
     * @return  Return Fp16T result from uiVal
     */
    Fp16T& operator=(const uint16_t& uiVal);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] iVal int32_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert int32_t to Fp16T
     * @return  Return Fp16T result from iVal
     */
    Fp16T& operator=(const int32_t& iVal);

    /**
     * @ingroup Fp16T math evaluation operator
     * @param [in] uiVal uint32_t object to be converted to Fp16T
     * @brief   Override basic evaluation operator to convert uint32_t to Fp16T
     * @return  Return Fp16T result from uiVal
     */
    Fp16T& operator=(const uint32_t& uiVal);

    /**
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to float/fp32
     * @return  Return float/fp32 value of Fp16T
     */
    operator float() const;

    /**
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to double/fp64
     * @return  Return double/fp64 value of Fp16T
     */
    operator double() const;

    /**
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to int8_t
     * @return  Return int8_t value of Fp16T
     */
    operator int8_t() const;

    /**
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to uint8_t
     * @return  Return uint8_t value of Fp16T
     */
    operator uint8_t() const;

    /**
     * @ingroup Fp16T conversion
     * @brief   Override convert operator to convert Fp16T to int16_t
     * @return  Return int16_t value of Fp16T
     */
    operator int16_t() const;

    /**
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to uint16_t
     * @return  Return uint16_t value of Fp16T
     */
    operator uint16_t() const;

    /**
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to int32_t
     * @return  Return int32_t value of Fp16T
     */
    operator int32_t() const;

    /**
     * @ingroup Fp16T math conversion
     * @brief   Override convert operator to convert Fp16T to int64_t
     * @return  Return int64_t value of Fp16T
     */
    operator uint32_t() const;

    /**
     * @ingroup Fp16T judgment method
     * @param [in] fp Fp16T object to be judgement
     * @brief   whether a Fp16T is inifinite
     * @return  Returns 1:+INF -1:-INF 0:not INF
     */
    int IsInf() const;

    /**
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to float/fp32
     * @return  Return float/fp32 value of Fp16T
     */
    float ToFloat();

    /**
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to double/fp64
     * @return  Return double/fp64 value of Fp16T
     */
    double ToDouble();

    /**
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to int8_t
     * @return  Return int8_t value of Fp16T
     */
    int8_t ToInt8();

    /**
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to uint8_t
     * @return  Return uint8_t value of Fp16T
     */
    uint8_t ToUInt8();

    /**
     * @ingroup Fp16T conversion
     * @brief   Convert Fp16T to int16_t
     * @return  Return int16_t value of Fp16T
     */
    int16_t ToInt16();

    /**
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to uint16_t
     * @return  Return uint16_t value of Fp16T
     */
    uint16_t ToUInt16();

    /**
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to int32_t
     * @return  Return int32_t value of Fp16T
     */
    int32_t ToInt32();

    /**
     * @ingroup Fp16T math conversion
     * @brief   Convert Fp16T to int64_t
     * @return  Return int64_t value of Fp16T
     */
    uint32_t ToUInt32();
};

using fp16_t = Fp16T;

std::ostream &operator<<(std::ostream &os, const Fp16T &fp);
}; // namespace Mki

#endif // MKI_UTILS_FP16_T_H

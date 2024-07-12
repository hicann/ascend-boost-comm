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

#include "mki/utils/fp16/fp16_t.h"

#define FP16_EXTRAC_SIGN(x) (((x) >> 15) & 1)
#define FP16_EXTRAC_EXP(x) (((x) >> 10) & FP16_MAX_EXP)
#define FP16_EXTRAC_MAN(x) ((((x) >> 0) & 0x3FF) | (((((x) >> 10) & 0x1F) > 0 ? 1 : 0) * 0x400))
#define FP16_CONSTRUCTOR(s, e, m) \
    (static_cast<uint16_t>(((s) << FP16_SIGN_INDEX) | ((e) << FP16_MAN_LEN) | ((m) & FP16_MAX_MAN)))
#define FP16_IS_ZERO(x) (((x) & FP16_ABS_MAX) == 0)
#define FP16_IS_DENORM(x) ((((x) & FP16_EXP_MASK) == 0))
#define FP16_IS_INVALID(x) (((x) & FP16_EXP_MASK) == FP16_EXP_MASK)
#define FP32_CONSTRUCTOR(s, e, m) (((s) << FP32_SIGN_INDEX) | ((e) << FP32_MAN_LEN) | ((m) & FP32_MAX_MAN))

namespace {
const uint16_t CONST_1 = 1;
const uint16_t CONST_2 = 2;
const uint16_t CONST_3 = 3;
const uint16_t CONST_7 = 7;
const uint16_t CONST_10 = 10;
const uint16_t CONST_11 = 11;
const uint16_t CONST_15 = 15;
const uint16_t CONST_31 = 31;
const uint16_t CONST_32 = 32;

const uint16_t FP16_EXP_BIAS = 15;
const uint16_t FP16_EXP_MASK = 0x7C00;
const uint16_t FP16_MAN_LEN = 10;
const uint16_t FP16_MAN_MASK = 0x03FF;
const uint16_t FP16_MAN_HIDE_BIT = 0x0400;
const uint16_t FP16_SIGN_INDEX = 15;
const uint16_t FP16_SIGN_MASK = 0x8000;
const uint16_t FP16_MAX = 0x7BFF;
const uint16_t FP16_MIN = 0xFBFF;
const uint16_t FP16_ABS_MAX = 0x7FFF;
const uint16_t FP16_MAX_EXP = 0x001F;
const uint16_t FP16_MAX_MAN = 0x03FF;
const uint16_t FP32_EXP_BIAS = 127;
const uint16_t FP32_MAN_LEN = 23;
const uint16_t FP32_SIGN_INDEX = 31;
const uint32_t FP32_SIGN_MASK = 0x80000000u;
const uint32_t FP32_EXP_MASK = 0x7F800000u;
const uint32_t FP32_MAN_MASK = 0x007FFFFFu;
const uint32_t FP32_MAN_HIDE_BIT = 0x00800000u;
const uint32_t FP32_ABS_MAX = 0x7FFFFFFFu;
const uint32_t FP32_MAX_MAN = 0x7FFFFF;
const uint16_t FP64_EXP_BIAS = 1023;
const uint16_t FP64_MAN_LEN = 52;
const uint16_t FP64_SIGN_INDEX = 63;
const uint64_t FP64_SIGN_MASK = 0x8000000000000000LLu;
const uint64_t FP64_EXP_MASK = 0x7FF0000000000000LLu;
const uint64_t FP64_MAN_MASK = 0x000FFFFFFFFFFFFFLLu;
const uint64_t FP64_MAN_HIDE_BIT = 0x0010000000000000LLu;
const uint8_t INT8_T_MAX = 0x7F;
const uint8_t BIT_LEN8_MAX = 0xFF;
const uint16_t INT16_T_MAX = 0x7FFF;
const uint16_t BIT_LEN16_MAX = 0xFFFF;
const uint32_t INT32_T_MAX = 0x7FFFFFFFu;
const uint32_t BIT_LEN32_MAX = 0xFFFFFFFFu;

enum class Fp16RoundMode {
    ROUND_TO_NEAREST = 0, /**< round to nearest even */
    ROUND_BY_TRUNCATED,   /**< round by truncated    */
    ROUND_MODE_RESERVED,
};
const Fp16RoundMode FP16_ROUND_MODE = Fp16RoundMode::ROUND_TO_NEAREST;
}

namespace Mki {
void ExtractFP16(const uint16_t& val, uint16_t* s, int16_t* e, uint16_t* m)
{
    // 1.Extract
    *s = FP16_EXTRAC_SIGN(val);
    *e = FP16_EXTRAC_EXP(val);
    *m = FP16_EXTRAC_MAN(val);

    // Denormal
    if ((*e) == 0) {
        *e = 1;
    }
}

template <typename T>
void ReverseMan(bool negative, T* man)
{
    if (negative) {
        *man = (~(*man)) + 1;
    }
}

template <typename T>
T RightShift(T man, int16_t shift)
{
    int bits = sizeof(T) * 8;
    T mask = ((static_cast<T>(1u)) << (static_cast<unsigned int>(bits - 1)));
    for (int i = 0; i < shift; i++) {
        man = ((man & mask) | (man >> 1));
    }
    return man;
}

template <typename T>
T GetManSum(int16_t ea, const T &ma, int16_t eb, const T &mb)
{
    T sum = 0;
    if (ea != eb) {
        T mTmp = 0;
        int16_t eTmp = std::abs(ea - eb);
        if (ea > eb) {
            mTmp = mb;
            mTmp = RightShift(mTmp, eTmp);
            sum = ma + mTmp;
        } else {
            mTmp = ma;
            mTmp = RightShift(mTmp, eTmp);
            sum = mTmp + mb;
        }
    } else {
        sum = ma + mb;
    }
    return sum;
}

template <typename T>
T ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, T man, uint16_t shift = 0)
{
    man = (man >> shift) + ((bit1 && (bitLeft || bit0)) ? 1 : 0);
    return man;
}

template <typename T>
int16_t GetManBitLength(T man)
{
    int16_t len = 0;
    while (man) {
        man >>= 1;
        len++;
    }
    return len;
}

static bool IsRoundOne(uint64_t man, uint16_t truncLen)
{
    uint16_t shiftOut = truncLen - CONST_2;
    uint64_t mask0 = 0x4;
    mask0 = mask0 << shiftOut;
    uint64_t mask1 = 0x2;
    mask1 = mask1 << shiftOut;
    uint64_t mask2 = mask1 - 1;

    bool lastBit = ((man & mask0) > 0);
    bool truncHigh = false;
    bool truncLeft = false;
    if (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) {
        truncHigh = ((man & mask1) > 0);
        truncLeft = ((man & mask2) > 0);
    }
    return (truncHigh && (truncLeft || lastBit));
}

/**
 * @ingroup Fp16T public method
 * @param [in] exp       exponent of Fp16T value
 * @param [in] man       exponent of Fp16T value
 * @brief   normalize Fp16T value
 * @return
 */
static void Fp16Normalize(int16_t& exp, uint16_t& man)
{
    if (exp >= FP16_MAX_EXP) {
        exp = FP16_MAX_EXP - 1;
        man = FP16_MAX_MAN;
    } else if (exp == 0 && man == FP16_MAN_HIDE_BIT) {
        exp++;
        man = 0;
    }
}

/**
 * @ingroup Fp16T math conversion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to float/fp32
 * @return  Return float/fp32 value of fpVal which is the value of Fp16T object
 */
static float Fp16ToFloat(const uint16_t& fpVal)
{
    float ret;

    uint16_t hfSign = 0;
    uint16_t hfMan = 0;
    int16_t hfExp = 0;
    ExtractFP16(fpVal, &hfSign, &hfExp, &hfMan);
    while ((hfMan != 0) && ((hfMan & FP16_MAN_HIDE_BIT) == 0)) {
        hfMan <<= 1;
        hfExp--;
    }

    uint32_t sRet = 0;
    uint32_t eRet = 0;
    uint32_t mRet = 0;
    uint32_t fVal = 0;
    sRet = hfSign;
    if (hfMan == 0) {
        eRet = 0;
        mRet = 0;
    } else {
        eRet = static_cast<uint32_t>(hfExp - FP16_EXP_BIAS + FP32_EXP_BIAS);
        mRet = hfMan & FP16_MAN_MASK;
        mRet = mRet << (FP32_MAN_LEN - FP16_MAN_LEN);
    }
    fVal = FP32_CONSTRUCTOR(sRet, eRet, mRet);
    uint32_t* ptrFVal = &fVal;
    float* ptrRet = reinterpret_cast<float *>(ptrFVal);
    ret = *ptrRet;

    return ret;
}

/**
 * @ingroup Fp16T math conversion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to double/fp64
 * @return  Return double/fp64 value of fpVal which is the value of Fp16T object
 */
static double Fp16ToDouble(const uint16_t& fpVal)
{
    double ret;

    uint16_t hfSign = 0;
    uint16_t hfMan = 0;
    int16_t hfExp;
    ExtractFP16(fpVal, &hfSign, &hfExp, &hfMan);

    while ((hfMan != 0) && ((hfMan & FP16_MAN_HIDE_BIT) == 0)) {
        hfMan <<= 1;
        hfExp--;
    }

    uint64_t sRet = 0;
    uint64_t eRet = 0;
    uint64_t mRet = 0;
    uint64_t fVal = 0;
    sRet = hfSign;
    if (hfMan == 0) {
        eRet = 0;
        mRet = 0;
    } else {
        eRet = static_cast<uint64_t>(hfExp - FP16_EXP_BIAS + FP64_EXP_BIAS);
        mRet = hfMan & FP16_MAN_MASK;
        mRet = mRet << (FP64_MAN_LEN - FP16_MAN_LEN);
    }
    fVal = (sRet << FP64_SIGN_INDEX) | (eRet << FP64_MAN_LEN) | (mRet);
    uint64_t* ptrFVal = &fVal;
    double* ptrRet = reinterpret_cast<double *>(ptrFVal);
    ret = *ptrRet;

    return ret;
}

/**
 * @ingroup Fp16T math conversion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to int8_t
 * @return  Return int8_t value of fpVal which is the value of Fp16T object
 */
static int8_t Fp16ToInt8(const uint16_t& fpVal)
{
    int8_t ret = 0;
    uint8_t retV = 0;
    uint8_t sRet = 0;
    uint8_t mRet = 0;
    uint16_t hfE = 0;
    uint16_t hfM = 0;

     // 1.get sRet and shift it to bit0.
    sRet = FP16_EXTRAC_SIGN(fpVal);

    // 2.get hfE and hfM
    hfE = FP16_EXTRAC_EXP(fpVal);
    hfM = FP16_EXTRAC_MAN(fpVal);

    if (FP16_IS_DENORM(fpVal)) { // Denormalized number
        retV = 0;
        ret = *(reinterpret_cast<uint8_t *>(&retV));
        return ret;
    }

    uint16_t shiftOut = 0;
    uint8_t overflowFlag = 0;
    uint64_t longIntM = hfM;

    if (FP16_IS_INVALID(fpVal)) { // Inf or NaN
        overflowFlag = 1;
    } else {
        while (hfE != FP16_EXP_BIAS) {
            if (hfE > FP16_EXP_BIAS) {
                hfE--;
                longIntM = longIntM << 1;
                if (sRet == 1 && longIntM >= 0x20000u) { // sign=1,negative number(<0)
                    longIntM = 0x20000u; // 10 0000 0000 0000 0000  10(Fp16T-man)+7(int8)=17bit
                    overflowFlag = 1;
                    break;
                } else if (sRet != 1 && longIntM >= 0x1FFFFu) { // sign=0,positive number(>0)
                    longIntM = 0x1FFFFu; // 01 1111 1111 1111 1111  10(Fp16T-man)+7(int8)
                    overflowFlag = 1;
                    break;
                }
            } else {
                hfE++;
                shiftOut++;
            }
        }
    }
    if (overflowFlag != 0) {
        retV = INT8_T_MAX + sRet;
    } else {
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint8_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN8_MAX);
        needRound = needRound && ((sRet == 0 && mRet < INT8_T_MAX) || (sRet == 1 && mRet <= INT8_T_MAX));
        if (needRound) {
            mRet++;
        }
        if (sRet != 0) {
            mRet = (~mRet) + CONST_1;
        }
        if (mRet == 0) {
            sRet = 0;
        }
    // Generate final result
        retV = (sRet << CONST_7) | (mRet);
    }

    ret = *(reinterpret_cast<int8_t *>(&retV));
    return ret;
}

/**
 * @ingroup Fp16T math conversion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to uint8_t
 * @return  Return uint8_t value of fpVal which is the value of Fp16T object
 */
static uint8_t Fp16ToUInt8(const uint16_t& fpVal)
{
    uint8_t ret = 0;
    uint8_t sRet = 0;
    uint8_t mRet = 0;
    uint16_t hfE = 0;
    uint16_t hfM = 0;

     // 1.get sRet and shift it to bit0.
    sRet = FP16_EXTRAC_SIGN(fpVal);

    // 2.get hfE and hfM
    hfE = FP16_EXTRAC_EXP(fpVal);
    hfM = FP16_EXTRAC_MAN(fpVal);

    if (FP16_IS_DENORM(fpVal)) { // Denormalized number
        return 0;
    }

    if (FP16_IS_INVALID(fpVal)) { // Inf or NaN
        mRet = ~0;
    } else {
        uint16_t shiftOut = 0;
        uint64_t longIntM = hfM;
        uint8_t overflowFlag = 0;

        while (hfE != FP16_EXP_BIAS) {
            if (hfE > FP16_EXP_BIAS) {
                hfE--;
                longIntM = longIntM << 1;
                if (longIntM >= 0x40000Lu) { // overflow 0100 0000 0000 0000 0000
                    longIntM = 0x3FFFFLu; // 11 1111 1111 1111 1111   10(Fp16T-man)+8(uint8)=18bit
                    overflowFlag = 1;
                    mRet = ~0;
                    break;
                }
            } else {
                hfE++;
                shiftOut++;
            }
        }
        if (overflowFlag == 0) {
            bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
            mRet = static_cast<uint8_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN8_MAX);
            if (needRound && mRet != BIT_LEN8_MAX) {
                mRet++;
            }
        }
    }

    if (sRet == 1) { // Negative number
        mRet = 0;
    }
    // Generate final result
    ret = mRet;

    return ret;
}

/**
 * @ingroup Fp16T math conversion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to int16_t
 * @return  Return int16_t value of fpVal which is the value of Fp16T object
 */
static int16_t Fp16ToInt16(const uint16_t& fpVal)
{
    // 1.get sRet and shift it to bit0.
    uint16_t sRet = FP16_EXTRAC_SIGN(fpVal);

    // 2.get hfE and hfM
    uint16_t hfE = FP16_EXTRAC_EXP(fpVal);
    uint16_t hfM = FP16_EXTRAC_MAN(fpVal);

    int16_t ret = 0;
    uint16_t retV = 0;
    if (FP16_IS_DENORM(fpVal)) { // Denormalized number
        retV = 0;
        ret = *(reinterpret_cast<uint8_t *>(&retV));
        return ret;
    }

    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;
    uint8_t overflowFlag = 0;

    if (FP16_IS_INVALID(fpVal)) { // Inf or NaN
        overflowFlag = 1;
    } else {
        while (hfE != FP16_EXP_BIAS) {
            if (hfE > FP16_EXP_BIAS) {
                hfE--;
                longIntM = longIntM << 1;
                if (sRet == 1 && longIntM > 0x2000000Lu) { // sign=1,negative number(<0)
                    overflowFlag = 1;
                    longIntM = 0x2000000Lu; // 10(Fp16T-man)+15(int16)=25bit
                    break;
                } else if (sRet != 1 && longIntM >= 0x1FFFFFFLu) { // sign=0,positive number(>0) Overflow
                    overflowFlag = 1;
                    longIntM = 0x1FFFFFFLu; // 10(Fp16T-man)+15(int16)=25bit
                    break;
                }
            } else {
                shiftOut++;
                hfE++;
            }
        }
    }
    uint16_t mRet = 0;
    if (overflowFlag != 0) {
        retV = INT16_T_MAX + sRet;
    } else {
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint16_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
        if (needRound && mRet < INT16_T_MAX) {
            mRet++;
        }
        if (sRet != 0) {
            mRet = (~mRet) + CONST_1;
        }
        if (mRet == 0) {
            sRet = 0;
        }
        // Generate final result
        retV = (sRet << CONST_15) | (mRet);
    }

    ret = *(reinterpret_cast<int16_t *>(&retV));
    return ret;
}

/**
 * @ingroup Fp16T math conversion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to uint16_t
 * @return  Return uint16_t value of fpVal which is the value of Fp16T object
 */
static uint16_t Fp16ToUInt16(const uint16_t& fpVal)
{
    uint16_t ret = 0;
    uint16_t sRet = 0;
    uint16_t mRet = 0;
    uint16_t hfM = 0;
    uint16_t hfE = 0;

    // 1.get sRet and shift it to bit0.
    sRet = FP16_EXTRAC_SIGN(fpVal);

    // 2.get hfE and hfM
    hfM = FP16_EXTRAC_MAN(fpVal);
    hfE = FP16_EXTRAC_EXP(fpVal);

    if (FP16_IS_DENORM(fpVal)) { // Denormalized number
        return 0;
    }

    if (FP16_IS_INVALID(fpVal)) { // Inf or NaN
        mRet = ~0;
    } else {
        uint64_t longIntM = hfM;
        uint16_t shiftOut = 0;

        while (hfE != FP16_EXP_BIAS) {
            if (hfE > FP16_EXP_BIAS) {
                hfE--;
                longIntM = longIntM << 1;
            } else {
                hfE++;
                shiftOut++;
            }
        }
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint16_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
        if (needRound && mRet != BIT_LEN16_MAX) {
            mRet++;
        }
    }

    if (sRet == 1) { // Negative number
        mRet = 0;
    }
    // Generate final result
    ret = mRet;

    return ret;
}

/**
 * @ingroup Fp16T math convertion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to int32_t
 * @return  Return int32_t value of fpVal which is the value of Fp16T object
 */
static int32_t Fp16ToInt32(const uint16_t& fpVal)
{
    int32_t ret = 0;
    uint32_t retV = 0;
    uint32_t sRet = 0;
    uint32_t mRet = 0;
    uint16_t hfE = 0;
    uint16_t hfM = 0;

    // 1.get sRet and shift it to bit0.
    sRet = FP16_EXTRAC_SIGN(fpVal);

    // 2.get hfE and hfM
    hfE = FP16_EXTRAC_EXP(fpVal);
    hfM = FP16_EXTRAC_MAN(fpVal);

    if (FP16_IS_INVALID(fpVal)) { // Inf or NaN
        retV = INT32_T_MAX + sRet;
    } else {
        uint16_t shiftOut = 0;
        uint64_t longIntM = hfM;
        while (hfE != FP16_EXP_BIAS) {
            if (hfE > FP16_EXP_BIAS) {
                longIntM = longIntM << 1;
                hfE--;
            } else {
                shiftOut++;
                hfE++;
            }
        }
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint32_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN32_MAX);
        if (needRound && mRet < INT32_T_MAX) {
            mRet++;
        }

        if (sRet == 1) {
            mRet = (~mRet) + 1;
        }
        if (mRet == 0) {
            sRet = 0;
        }
    // Generate final result
        retV = (sRet << CONST_31) | (mRet);
    }

    ret = *(reinterpret_cast<int32_t *>(&retV));
    return ret;
}

/**
 * @ingroup Fp16T math conversion static method
 * @param [in] fpVal uint16_t value of Fp16T object
 * @brief   Convert Fp16T to uint32_t
 * @return  Return uint32_t value of fpVal which is the value of Fp16T object
 */
static uint32_t Fp16ToUInt32(const uint16_t& fpVal)
{
    uint32_t ret = 0;
    uint32_t sRet = 0;
    uint32_t mRet = 0;
    uint16_t hfE = 0;
    uint16_t hfM = 0;

    // 1.get sRet and shift it to bit0.
    sRet = FP16_EXTRAC_SIGN(fpVal);

    // 2.get hfE and hfM
    hfE = FP16_EXTRAC_EXP(fpVal);
    hfM = FP16_EXTRAC_MAN(fpVal);

    if (FP16_IS_DENORM(fpVal)) { // Denormalized number
        return 0u;
    }

    if (FP16_IS_INVALID(fpVal)) { // Inf or NaN
        mRet = ~0u;
    } else {
        uint64_t longIntM = hfM;
        uint16_t shiftOut = 0;
        while (hfE != FP16_EXP_BIAS) {
            if (hfE > FP16_EXP_BIAS) {
                longIntM = longIntM << 1;
                hfE--;
            } else {
                shiftOut++;
                hfE++;
            }
        }
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint32_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN32_MAX);
        if (needRound && mRet != BIT_LEN32_MAX) {
            mRet++;
        }
    }

    if (sRet == 1) { // Negative number
        mRet = 0;
    }
    // Generate final result
    ret = mRet;

    return ret;
}

/**
 * @ingroup Fp16T math operator
 * @param [in] v1 left operator value of Fp16T object
 * @param [in] v2 right operator value of Fp16T object
 * @brief   Performing Fp16T addition
 * @return  Return Fp16T result of adding this and fp
 */
static uint16_t Fp16Add(uint16_t v1, uint16_t v2)
{
    uint16_t ret = 0;
    uint16_t sA = 0;
    uint16_t sB = 0;
    int16_t eA = 0;
    int16_t eB = 0;
    uint32_t mA = 0;
    uint32_t mB = 0;
    uint16_t sRet = 0;
    uint16_t mRet = 0;
    int16_t eRet = 0;
    uint16_t maTmp = 0;
    uint16_t mbTmp = 0;

    uint16_t shiftOut = 0;
    // 1.Extract
    ExtractFP16(v1, &sA, &eA, &maTmp);
    ExtractFP16(v2, &sB, &eB, &mbTmp);
    mA = maTmp;
    mB = mbTmp;

    uint16_t sum;
    if (sA != sB) {
        ReverseMan(sA > 0, &mA);
        ReverseMan(sB > 0, &mB);
        sum = static_cast<uint16_t>(GetManSum(eA, mA, eB, mB));
        sRet = (sum & FP16_SIGN_MASK) >> FP16_SIGN_INDEX;
        ReverseMan(sRet > 0, &mA);
        ReverseMan(sRet > 0, &mB);
    } else {
        sum = static_cast<uint16_t>(GetManSum(eA, mA, eB, mB));
        sRet = sA;
    }

    if (sum == 0) {
        shiftOut = CONST_3;
        mA = mA << shiftOut;
        mB = mB << shiftOut;
    }

    int16_t eTmp = 0;
    uint32_t mTrunc = 0;

    eRet = std::max(eA, eB);
    eTmp = std::abs(eA - eB);
    if (eA > eB) {
        mTrunc = (mB << (CONST_32 - static_cast<uint16_t>(eTmp)));
        mB = RightShift(mB, eTmp);
    } else if (eA < eB) {
        mTrunc = (mA << (CONST_32 - static_cast<uint16_t>(eTmp)));
        mA = RightShift(mA, eTmp);
    }
    // calculate mantissa
    mRet = static_cast<uint16_t>(mA + mB);

    uint16_t mMin = static_cast<uint16_t>(FP16_MAN_HIDE_BIT << shiftOut);
    uint16_t mMax = mMin << 1;
    // Denormal
    while (mRet < mMin && eRet > 0) { // the value of mRet should not be smaller than 2^23
        mRet = mRet << 1;
        mRet += (FP32_SIGN_MASK & mTrunc) >> FP32_SIGN_INDEX;
        mTrunc = mTrunc << 1;
        eRet = eRet - 1;
    }
    while (mRet >= mMax) { // the value of mRet should be smaller than 2^24
        mTrunc = mTrunc >> 1;
        mTrunc = mTrunc | (FP32_SIGN_MASK * (mRet & 1));
        mRet = mRet >> 1;
        eRet = eRet + 1;
    }

    bool isLastBit = ((mRet & 1) > 0);
    bool isTruncHigh = 0;
    bool isTruncLeft = 0;
    isTruncHigh = (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) && ((mTrunc & FP32_SIGN_MASK) > 0);
    isTruncLeft = (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) && ((mTrunc & FP32_ABS_MAX) > 0);
    mRet = ManRoundToNearest(isLastBit, isTruncHigh, isTruncLeft, mRet, shiftOut);
    while (mRet >= mMax) {
        mRet = mRet >> 1;
        eRet = eRet + 1;
    }

    if (eRet == 0 && mRet <= mMax) {
        mRet = mRet >> 1;
    }
    Fp16Normalize(eRet, mRet);

    ret = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
    return ret;
}

/**
 * @ingroup Fp16T math operator
 * @param [in] v1 left operator value of Fp16T object
 * @param [in] v2 right operator value of Fp16T object
 * @brief   Performing Fp16T subtraction
 * @return  Return Fp16T result of subtraction fp from this
 */
static uint16_t Fp16Sub(uint16_t v1, uint16_t v2)
{
    // Reverse
    uint16_t tmp = ((~(v2)) & FP16_SIGN_MASK) | (v2 & FP16_ABS_MAX);
    uint16_t ret = Fp16Add(v1, tmp);
    return ret;
}

/**
 * @ingroup Fp16T math operator
 * @param [in] v1 left operator value of Fp16T object
 * @param [in] v2 right operator value of Fp16T object
 * @brief   Performing Fp16T multiplication
 * @return  Return Fp16T result of multiplying this and fp
 */
static uint16_t Fp16Mul(uint16_t v1, uint16_t v2)
{
    uint16_t sA = 0;
    uint16_t sB = 0;
    int16_t eA = 0;
    int16_t eB = 0;
    uint32_t mA = 0;
    uint32_t mB = 0;
    uint16_t sRet = 0;
    uint16_t mRet = 0;
    int16_t eRet = 0;
    uint32_t mulM = 0;
    uint16_t maTmp = 0;
    uint16_t mbTmp = 0;

    // 1.Extract
    ExtractFP16(v1, &sA, &eA, &maTmp);
    ExtractFP16(v2, &sB, &eB, &mbTmp);
    mA = maTmp;
    mB = mbTmp;

    eRet = eA + eB - FP16_EXP_BIAS - CONST_10;
    mulM = mA * mB;
    sRet = sA ^ sB;

    uint32_t mMin = FP16_MAN_HIDE_BIT;
    uint32_t mMax = mMin << 1;
    uint32_t mTrunc = 0;
    // the value of mRet should not be smaller than 2^23
    while (mulM < mMin && eRet > 1) {
        mulM = mulM << 1;
        eRet = eRet - 1;
    }
    while (mulM >= mMax || eRet < 1) {
        mTrunc = mTrunc >> 1;
        mTrunc = mTrunc | (FP32_SIGN_MASK * (mulM & 1));
        mulM = mulM >> 1;
        eRet = eRet + 1;
    }
    bool bLastBit = ((mulM & 1) > 0);
    bool bTruncHigh = (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) && ((mTrunc & FP32_SIGN_MASK) > 0);
    bool bTruncLeft = (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) && ((mTrunc & FP32_ABS_MAX) > 0);
    mulM = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mulM);

    while (mulM >= mMax || eRet < 0) {
        mulM = mulM >> 1;
        eRet = eRet + 1;
    }

    if (eRet == 1 && mulM < FP16_MAN_HIDE_BIT) {
        eRet = 0;
    }
    mRet = static_cast<uint16_t>(mulM);

    Fp16Normalize(eRet, mRet);

    uint16_t ret = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
    return ret;
}

/**
 * @ingroup Fp16T math operator divided
 * @param [in] v1 left operator value of Fp16T object
 * @param [in] v2 right operator value of Fp16T object
 * @brief   Performing Fp16T division
 * @return  Return Fp16T result of division this by fp
 */
static uint16_t Fp16Div(uint16_t v1, uint16_t v2)
{
    uint16_t ret;
    if (FP16_IS_ZERO(v2)) { // result is inf
    // throw "Fp16T division by zero.";
        uint16_t sA = FP16_EXTRAC_SIGN(v1);
        uint16_t sB = FP16_EXTRAC_SIGN(v2);
        uint16_t sRet = sA ^ sB;
        ret = FP16_CONSTRUCTOR(sRet, FP16_MAX_EXP - 1, FP16_MAX_MAN);
    } else if (FP16_IS_ZERO(v1)) {
        ret = 0u;
    } else {
        uint16_t sA = 0;
        uint16_t sB = 0;
        int16_t eA = 0;
        int16_t eB = 0;
        uint64_t mA = 0;
        uint64_t mB = 0;
        float mDiv = 0;
        uint16_t maTmp = 0;
        uint16_t mbTmp = 0;
        // 1.Extract
        ExtractFP16(v1, &sA, &eA, &maTmp);
        ExtractFP16(v2, &sB, &eB, &mbTmp);
        mA = maTmp;
        mB = mbTmp;

        uint64_t mTmp = 0;
        if (eA > eB) {
            mTmp = mA;
            uint16_t tmp = static_cast<uint16_t>(eA - eB);
            for (int i = 0; i < tmp; i++) {
                mTmp = mTmp << 1;
            }
            mA = mTmp;
        } else if (eA < eB) {
            mTmp = mB;
            uint16_t tmp = static_cast<uint16_t>(eB - eA);
            for (int i = 0; i < tmp; i++) {
                mTmp = mTmp << 1;
            }
            mB = mTmp;
        }
        mDiv = static_cast<float>(mA * 1.0f / mB);
        Fp16T fpDiv = static_cast<Fp16T>(mDiv); /*lint !e524*/
        ret = fpDiv.val;
        if (sA != sB) {
            ret |= FP16_SIGN_MASK;
        }
    }
    return ret;
}

// operate
Fp16T Fp16T::operator+(const Fp16T fp) const
{
    uint16_t retVal = Fp16Add(val, fp.val);
    Fp16T ret(retVal);
    return ret;
}

Fp16T Fp16T::operator-(const Fp16T fp) const
{
    uint16_t retVal = Fp16Sub(val, fp.val);
    Fp16T ret(retVal);
    return ret;
}

Fp16T Fp16T::operator*(const Fp16T fp) const
{
    uint16_t retVal = Fp16Mul(val, fp.val);
    Fp16T ret(retVal);
    return ret;
}

Fp16T Fp16T::operator/(const Fp16T fp) const
{
    uint16_t retVal = Fp16Div(val, fp.val);
    Fp16T ret(retVal);
    return ret;
}

Fp16T Fp16T::operator+=(const Fp16T fp)
{
    val = Fp16Add(val, fp.val);
    return *this;
}

Fp16T Fp16T::operator-=(const Fp16T fp)
{
    val = Fp16Sub(val, fp.val);
    return *this;
}

Fp16T Fp16T::operator*=(const Fp16T fp)
{
    val = Fp16Mul(val, fp.val);
    return *this;
}

Fp16T Fp16T::operator/=(const Fp16T fp)
{
    val = Fp16Div(val, fp.val);
    return *this;
}

// compare
bool Fp16T::operator==(const Fp16T& fp) const
{
    if (FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val)) {
        return true;
    } else {
        return ((val & BIT_LEN16_MAX) == (fp.val & BIT_LEN16_MAX)); // bit compare
    }
}

bool Fp16T::operator!=(const Fp16T& fp) const
{
    if (FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val)) {
        return false;
    } else {
        return ((val & BIT_LEN16_MAX) != (fp.val & BIT_LEN16_MAX)); // bit compare
    }
}

bool Fp16T::operator>(const Fp16T& fp) const
{
    uint16_t sA = 0;
    uint16_t sB = 0;
    uint16_t eA = 0;
    uint16_t eB = 0;
    uint16_t mA = 0;
    uint16_t mB = 0;

    // 1.Extract
    sA = FP16_EXTRAC_SIGN(val);
    sB = FP16_EXTRAC_SIGN(fp.val);
    eA = FP16_EXTRAC_EXP(val);
    eB = FP16_EXTRAC_EXP(fp.val);
    mA = FP16_EXTRAC_MAN(val);
    mB = FP16_EXTRAC_MAN(fp.val);

    // Compare
    if ((sA == 0) && (sB > 0)) { // +  -
        // -0=0
        return !(FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val));
    } else if ((sA == 0) && (sB == 0)) { // + +
        if (eA > eB) { // eA - eB >= 1; Va always larger than Vb
            return true;
        } else if (eA == eB) {
            return mA > mB;
        } else {
            return false;
        }
    } else if ((sA > 0) && (sB > 0)) { // - -    opposite to  + +
        if (eA < eB) {
            return true;
        } else if (eA == eB) {
            return mA < mB;
        } else {
            return false;
        }
    } else { // -  +
        return false;
    }
}

bool Fp16T::operator>=(const Fp16T& fp) const
{
    if ((*this) > fp) {
        return true;
    } else if ((*this) == fp) {
        return true;
    } else {
        return false;
    }
}

bool Fp16T::operator<(const Fp16T& fp) const
{
    if ((*this) >= fp) {
        return false;
    } else {
        return true;
    }
}

bool Fp16T::operator<=(const Fp16T& fp) const
{
    if ((*this) > fp) {
        return false;
    } else {
        return true;
    }
}

// evaluation
Fp16T& Fp16T::operator=(const Fp16T& fp)
{
    if (this == &fp) {
        return *this;
    }
    val = fp.val;
    return *this;
}

Fp16T& Fp16T::operator=(const float& fVal)
{
    uint16_t sRet = 0;
    uint16_t mRet = 0;
    int16_t eRet = 0;
    uint32_t eF = 0;
    uint32_t mF = 0;
    uint32_t ui32V = *(reinterpret_cast<const uint32_t *>(&fVal)); // 1:8:23bit sign:exp:man
    uint32_t mLenDelta = 0;

    sRet = static_cast<uint16_t>((ui32V & FP32_SIGN_MASK) >> FP32_SIGN_INDEX); // 4Byte->2Byte
    eF = (ui32V & FP32_EXP_MASK) >> FP32_MAN_LEN; // 8 bit exponent
    mF = (ui32V & FP32_MAN_MASK); // 23 bit mantissa dont't need to care about denormal
    mLenDelta = FP32_MAN_LEN - FP16_MAN_LEN;

    bool needRound = false;
    // Exponent overflow/NaN converts to signed inf/NaN
    if (eF > 0x8Fu) { // 0x8Fu:142=127+15
        eRet = FP16_MAX_EXP - 1;
        mRet = FP16_MAX_MAN;
    } else if (eF <= 0x70u) { // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
        eRet = 0;
        if (eF >= 0x67) { // 0x67:103=127-24 Denormal
            mF = (mF | FP32_MAN_HIDE_BIT);
            uint16_t shiftOut = FP32_MAN_LEN;
            uint64_t mTmp = (static_cast<uint64_t>(mF)) << (eF - 0x67);

            needRound = IsRoundOne(mTmp, shiftOut);
            mRet = static_cast<uint16_t>(mTmp >> shiftOut);
            if (needRound) {
                mRet++;
            }
        } else if (eF == 0x66 && mF > 0) { // 0x66:102 Denormal 0<f_v<min(Denormal)
            mRet = 1;
        } else {
            mRet = 0;
        }
    } else { // Regular case with no overflow or underflow
        needRound = IsRoundOne(mF, mLenDelta);
        mRet = static_cast<uint16_t>(mF >> mLenDelta);
        if (needRound) {
            mRet++;
        }
        eRet = static_cast<int16_t>(eF - 0x70u);
        if ((mRet & FP16_MAN_HIDE_BIT) != 0) {
            eRet++;
        }
    }

    Fp16Normalize(eRet, mRet);
    val = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
    return *this;
}

Fp16T& Fp16T::operator=(const int8_t& iVal)
{
    uint16_t sRet = 0;
    uint16_t eRet = 0;
    uint16_t mRet = 0;

    sRet = static_cast<uint16_t>(((static_cast<uint8_t>(iVal)) & 0x80) >> CONST_7);
    mRet = static_cast<uint16_t>(((static_cast<uint8_t>(iVal)) & INT8_T_MAX));

    if (mRet == 0) {
        eRet = 0;
    } else {
        if (sRet != 0) { // negative number(<0)
            mRet = static_cast<uint16_t>(std::abs(iVal)); // complement
        }

        eRet = FP16_MAN_LEN;
        while ((mRet & FP16_MAN_HIDE_BIT) == 0) {
            eRet = eRet - CONST_1;
            mRet = mRet << CONST_1;
        }
        eRet = eRet + FP16_EXP_BIAS;
    }

    val = FP16_CONSTRUCTOR(sRet, eRet, mRet);
    return *this;
}

Fp16T& Fp16T::operator=(const uint8_t& uiVal)
{
    uint16_t sRet = 0;
    uint16_t eRet = 0;
    uint16_t mRet = uiVal;
    if (mRet != 0) {
        eRet = FP16_MAN_LEN;
        while ((mRet & FP16_MAN_HIDE_BIT) == 0) {
            mRet = mRet << CONST_1;
            eRet = eRet - CONST_1;
        }
        eRet = eRet + FP16_EXP_BIAS;
    }

    val = FP16_CONSTRUCTOR(sRet, eRet, mRet);
    return *this;
}

Fp16T& Fp16T::operator=(const int16_t& iVal)
{
    if (iVal == 0) {
        val = 0;
    } else {
        uint16_t sRet;
        uint16_t uiVal = *(reinterpret_cast<const uint16_t *>(&iVal));
        sRet = static_cast<uint16_t>(uiVal >> CONST_15);
        if (sRet != 0) {
            int16_t iValM = -iVal;
            uiVal = *(reinterpret_cast<uint16_t *>(&iValM));
        }
        uint32_t mTmp = (uiVal & FP32_ABS_MAX);

        uint16_t mMin = FP16_MAN_HIDE_BIT;
        uint16_t mMax = mMin << 1;
        uint16_t len = static_cast<uint16_t>(GetManBitLength(mTmp));
        if (mTmp != 0) {
            int16_t eRet;

            if (len > CONST_11) {
                eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
                uint16_t eTmp = len - CONST_11;
                uint32_t truncMask = 1;
                for (int i = 1; i < eTmp; i++) {
                    truncMask = (truncMask << 1) + 1;
                }
                uint32_t mTrunc = (mTmp & truncMask) << (CONST_32 - eTmp);
                for (int i = 0; i < eTmp; i++) {
                    mTmp = (mTmp >> 1);
                    eRet = eRet + 1;
                }
                bool bLastBit = ((mTmp & 1) > 0);
                bool bTruncHigh = false;
                bool bTruncLeft = false;
                if (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) { // trunc
                    bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
                    bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
                }
                mTmp = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mTmp);
                while (mTmp >= mMax || eRet < 0) {
                    mTmp = mTmp >> 1;
                    eRet = eRet + 1;
                }
            } else {
                eRet = FP16_EXP_BIAS;
                mTmp = mTmp << (CONST_11 - len);
                eRet = eRet + (len - 1);
            }
            uint16_t mRet = static_cast<uint16_t>(mTmp);
            val = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
        } else {
        }
    }
    return *this;
}

Fp16T& Fp16T::operator=(const uint16_t& uiVal)
{
    if (uiVal == 0) {
        val = 0;
    } else {
        int16_t eRet;
        uint16_t mRet = uiVal;

        uint16_t mMin = FP16_MAN_HIDE_BIT;
        uint16_t mMax = mMin << 1;
        uint16_t len = static_cast<uint16_t>(GetManBitLength(mRet));
        if (len > CONST_11) {
            eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
            uint32_t mTrunc;
            uint32_t truncMask = 1;
            uint16_t eTmp = len - CONST_11;
            for (int i = 1; i < eTmp; i++) {
                truncMask = (truncMask << 1) + 1;
            }
            mTrunc = (mRet & truncMask) << (CONST_32 - eTmp);
            for (int i = 0; i < eTmp; i++) {
                mRet = (mRet >> 1);
                eRet = eRet + 1;
            }
            bool bLastBit = ((mRet & 1) > 0);
            bool bTruncHigh = false;
            bool bTruncLeft = false;
            if (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) { // trunc
                bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
                bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
            }
            mRet = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mRet);
            while (mRet >= mMax || eRet < 0) {
                mRet = mRet >> 1;
                eRet = eRet + 1;
            }
            if (FP16_IS_INVALID(val)) {
                val = FP16_MAX;
            }
        } else {
            eRet = FP16_EXP_BIAS;
            mRet = mRet << (CONST_11 - len);
            eRet = eRet + (len - 1);
        }
        val = FP16_CONSTRUCTOR(0u, static_cast<uint16_t>(eRet), mRet);
    }
    return *this;
}

Fp16T& Fp16T::operator=(const int32_t& iVal)
{
    if (iVal == 0) {
        val = 0;
    } else {
        uint32_t uiVal = *(reinterpret_cast<const uint32_t *>(&iVal));
        uint16_t sRet = static_cast<uint16_t>(uiVal >> CONST_31);
        if (sRet != 0) {
            int32_t iValM = -iVal;
            uiVal = *(reinterpret_cast<uint32_t *>(&iValM));
        }
        int16_t eRet;
        uint32_t mTmp = (uiVal & FP32_ABS_MAX);

        uint32_t mMin = FP16_MAN_HIDE_BIT;
        uint32_t mMax = mMin << 1;
        uint16_t len = static_cast<uint16_t>(GetManBitLength(mTmp));
        if (len > CONST_11) {
            eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
            uint32_t truncMask = 1;
            uint16_t eTmp = len - CONST_11;
            for (int i = 1; i < eTmp; i++) {
                truncMask = (truncMask << 1) + 1;
            }
            uint32_t mTrunc = (mTmp & truncMask) << (CONST_32 - eTmp);
            for (int i = 0; i < eTmp; i++) {
                eRet = eRet + 1;
                mTmp = (mTmp >> 1);
            }
            bool bLastBit = ((mTmp & 1) > 0);
            bool bTruncHigh = false;
            bool bTruncLeft = false;
            if (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) { // trunc
                bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
                bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
            }
            mTmp = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mTmp);
            while (mTmp >= mMax || eRet < 0) {
                eRet = eRet + 1;
                mTmp = mTmp >> 1;
            }
            if (eRet >= FP16_MAX_EXP) {
                mTmp = FP16_MAX_MAN;
                eRet = FP16_MAX_EXP - 1;
            }
        } else {
            eRet = FP16_EXP_BIAS;
            mTmp = mTmp << (CONST_11 - len);
            eRet = eRet + (len - 1);
        }
        uint16_t mRet = static_cast<uint16_t>(mTmp);
        val = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
    }
    return *this;
}

Fp16T& Fp16T::operator=(const uint32_t& uiVal)
{
    if (uiVal == 0) {
        val = 0;
    } else {
        int16_t eRet;
        uint32_t mTmp = uiVal;

        uint32_t mMin = FP16_MAN_HIDE_BIT;
        uint32_t mMax = mMin << 1;
        uint16_t len = static_cast<uint16_t>(GetManBitLength(mTmp));
        if (len > CONST_11) {
            eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
            uint32_t mTrunc = 0;
            uint32_t truncMask = 1;
            uint16_t eTmp = len - CONST_11;
            for (int i = 1; i < eTmp; i++) {
                truncMask = (truncMask << 1) + 1;
            }
            mTrunc = (mTmp & truncMask) << (CONST_32 - eTmp);
            for (int i = 0; i < eTmp; i++) {
                mTmp = (mTmp >> 1);
                eRet = eRet + 1;
            }
            bool bLastBit = ((mTmp & 1) > 0);
            bool bTruncHigh = false;
            bool bTruncLeft = false;
            if (Fp16RoundMode::ROUND_TO_NEAREST == FP16_ROUND_MODE) { // trunc
                bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
                bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
            }
            mTmp = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mTmp);
            while (mTmp >= mMax || eRet < 0) {
                mTmp = mTmp >> 1;
                eRet = eRet + 1;
            }
            if (eRet >= FP16_MAX_EXP) {
                eRet = FP16_MAX_EXP - 1;
                mTmp = FP16_MAX_MAN;
            }
        } else {
            eRet = FP16_EXP_BIAS;
            mTmp = mTmp << (CONST_11 - len);
            eRet = eRet + (len - 1);
        }
        uint16_t mRet = static_cast<uint16_t>(mTmp);
        val = FP16_CONSTRUCTOR(0u, static_cast<uint16_t>(eRet), mRet);
    }
    return *this;
}

Fp16T& Fp16T::operator=(const double& dVal)
{
    uint16_t sRet = 0;
    uint16_t mRet = 0;
    int16_t eRet = 0;
    uint64_t eD = 0;
    uint64_t mD = 0;
    uint64_t ui64V = *(reinterpret_cast<const uint64_t *>(&dVal)); // 1:11:52bit sign:exp:man
    uint32_t mLenDelta = 0;

    sRet = static_cast<uint16_t>((ui64V & FP64_SIGN_MASK) >> FP64_SIGN_INDEX); // 4Byte
    eD = (ui64V & FP64_EXP_MASK) >> FP64_MAN_LEN; // 10 bit exponent
    mD = (ui64V & FP64_MAN_MASK); // 52 bit mantissa
    mLenDelta = FP64_MAN_LEN - FP16_MAN_LEN;

    bool needRound = false;
    // Exponent overflow/NaN converts to signed inf/NaN
    if (eD >= 0x410u) { // 0x410:1040=1023+16
        eRet = FP16_MAX_EXP - 1;
        mRet = FP16_MAX_MAN;
        val = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
    } else if (eD <= 0x3F0u) { // Exponent underflow converts to denormalized half or signed zero
        // 0x3F0:1008=1023-15
        /**
         * Signed zeros, denormalized floats, and floats with small
         * exponents all convert to signed zero half precision.
         */
        eRet = 0;
        if (eD >= 0x3E7u) { // 0x3E7u:999=1023-24 Denormal
            // Underflows to a denormalized value
            mD = (FP64_MAN_HIDE_BIT | mD);
            uint16_t shiftOut = FP64_MAN_LEN;
            uint64_t mTmp = (static_cast<uint64_t>(mD)) << (eD - 0x3E7u);

            needRound = IsRoundOne(mTmp, shiftOut);
            mRet = static_cast<uint16_t>(mTmp >> shiftOut);
            if (needRound) {
                mRet++;
            }
        } else if (eD == 0x3E6u && mD > 0) {
            mRet = 1;
        } else {
            mRet = 0;
        }
    } else { // Regular case with no overflow or underflow
        eRet = static_cast<int16_t>(eD - 0x3F0u);

        needRound = IsRoundOne(mD, mLenDelta);
        mRet = static_cast<uint16_t>(mD >> mLenDelta);
        if (needRound) {
            mRet++;
        }
        if ((mRet & FP16_MAN_HIDE_BIT) != 0) {
            eRet++;
        }
    }

    Fp16Normalize(eRet, mRet);
    val = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
    return *this;
}

// convert
Fp16T::operator float() const { return Fp16ToFloat(val); }

Fp16T::operator double() const { return Fp16ToDouble(val); }

Fp16T::operator int8_t() const { return Fp16ToInt8(val); }

Fp16T::operator uint8_t() const { return Fp16ToUInt8(val); }

Fp16T::operator int16_t() const { return Fp16ToInt16(val); }

Fp16T::operator uint16_t() const { return Fp16ToUInt16(val); }

Fp16T::operator int32_t() const { return Fp16ToInt32(val); }

Fp16T::operator uint32_t() const { return Fp16ToUInt32(val); }

int Fp16T::IsInf() const
{
    if ((val & FP16_ABS_MAX) == FP16_EXP_MASK) {
        if ((val & FP16_SIGN_MASK) != 0) {
            return -1;
        } else {
            return 1;
        }
    } else {
        return 0;
    }
}

float Fp16T::ToFloat() { return Fp16ToFloat(val); }

double Fp16T::ToDouble() { return Fp16ToDouble(val); }

int8_t Fp16T::ToInt8() { return Fp16ToInt8(val); }

uint8_t Fp16T::ToUInt8() { return Fp16ToUInt8(val); }

int16_t Fp16T::ToInt16() { return Fp16ToInt16(val); }

uint16_t Fp16T::ToUInt16() { return Fp16ToUInt16(val); }

int32_t Fp16T::ToInt32() { return Fp16ToInt32(val); }

uint32_t Fp16T::ToUInt32() { return Fp16ToUInt32(val); }

std::ostream &operator<<(std::ostream &os, const Fp16T &fp)
{
    os << static_cast<float>(fp);
    return os;
}
} // namespace fe

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
#ifndef MKI_UTILS_STATUS_STATUS_H
#define MKI_UTILS_STATUS_STATUS_H
#include <string>

namespace Mki {
class Status {
public:
    Status() noexcept;
    ~Status();
    Status(const Status &other);
    Status &operator=(const Status &other);
    Status(Status &&other) noexcept;
    Status &operator=(Status &&other);
    bool Ok() const;
    std::string ToString() const;
    int Code() const;
    std::string Message() const;

public:
    static Status OkStatus();
    static Status FailStatus(int code, const std::string &msg = "");

private:
    Status(int code, const std::string &msg);
    int GetLen() const;
    int GetCode() const;
    const char *GetMsg() const;

private:
    uint8_t *rep_{nullptr};
};

#define OP_TILING_CHECK_STATUS_RETURN(ret)                                                                         \
    do {                                                                                                           \
        if (!(ret).Ok()) {                                                                                         \
            MKI_LOG_ERROR << (ret).ToString();                                                                     \
            return ret;                                                                                            \
        }                                                                                                          \
    } while (0)

#define OP_TILING_CHECK_STATUS_RETURN_VOID(ret)                                                                    \
    do {                                                                                                           \
        if (!(ret).Ok()) {                                                                                         \
            MKI_LOG_ERROR << (ret).ToString();                                                                     \
            return;                                                                                                \
        }                                                                                                          \
    } while (0)
} // namespace Mki
#endif

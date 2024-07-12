/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    char *rep_{nullptr};
};

#define OP_TILING_CHECK_STATUS_RETURN(ret)                                                                             \
    do {                                                                                                               \
        if (!(ret).Ok()) {                                                                                             \
            MKI_LOG_ERROR << (ret).ToString();                                                                         \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

#define OP_TILING_CHECK_STATUS_RETURN_VOID(ret)                                                                        \
    do {                                                                                                               \
        if (!(ret).Ok()) {                                                                                             \
            MKI_LOG_ERROR << (ret).ToString();                                                                         \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)
} // namespace Mki
#endif

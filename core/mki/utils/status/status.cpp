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
#include <securec.h>
#include "mki/utils/log/log.h"
#include "mki/utils/status/status.h"

namespace Mki {
constexpr int MAX_LOG_STRING_SIZE = 1024;
Status::Status() noexcept : rep_(nullptr){};

Status::~Status()
{
    delete[] rep_;
    rep_ = nullptr;
};

Status::Status(const Status &other)
{
    if (other.rep_ == nullptr) {
        rep_ = nullptr;
        return;
    }

    int len = other.GetLen();
    if (len > MAX_LOG_STRING_SIZE) {
        return;
    }
    rep_ = new char[len];
    auto ret = memcpy_s(rep_, len, other.rep_, len);
    MKI_LOG_IF(ret != EOK, ERROR) << "memcpy failed";
}

Status &Status::operator=(const Status &other)
{
    if (this == &other) {
        return *this;
    }

    if (other.rep_ == nullptr) {
        delete[] rep_;
        rep_ = nullptr;
    } else {
        delete[] rep_;
        rep_ = nullptr;
        int len = other.GetLen();
        if (len > MAX_LOG_STRING_SIZE) {
            return *this;
        }
        rep_ = new char[len];
        auto ret = memcpy_s(rep_, len, other.rep_, len);
        MKI_LOG_IF(ret != EOK, ERROR) << "memcpy failed";
    }

    return *this;
}

Status::Status(Status &&other) noexcept
{
    rep_ = other.rep_;
    other.rep_ = nullptr;
}

Status &Status::operator=(Status &&other)
{
    std::swap(rep_, other.rep_);
    return *this;
}

bool Status::Ok() const { return rep_ == nullptr; }

std::string Status::ToString() const
{
    if (rep_ == nullptr) {
        return "ok";
    }

    return "code:" + std::to_string(GetCode()) + ", msg:" + GetMsg();
}

int Status::Code() const
{
    if (rep_ == nullptr) {
        return 0;
    }
    return GetCode();
}

std::string Status::Message() const
{
    if (rep_ == nullptr) {
        return "";
    }
    return GetMsg();
}

Status::Status(int code, const std::string &msg)
{
    if (msg.size() > MAX_LOG_STRING_SIZE) {
        return;
    }
    int len = static_cast<int>(sizeof(int) + sizeof(int) + msg.size() + 1);
    rep_ = new char[len];
    int *lenPtr = reinterpret_cast<int *>(rep_);
    *lenPtr = len;
    int *codePtr = reinterpret_cast<int *>(rep_ + sizeof(int));
    *codePtr = code;
    char *msgPtr = rep_ + sizeof(int) + sizeof(int);
    auto ret = memcpy_s(msgPtr, len - sizeof(int) - sizeof(int), msg.data(), msg.size());
    MKI_LOG_IF(ret != EOK, ERROR) << "memcpy failed";
    rep_[len - 1] = 0;
}

Status Status::OkStatus()
{
    Status st;
    return st;
}

Status Status::FailStatus(int code, const std::string &msg)
{
    Status st(code, msg);
    return st;
}

int Status::GetLen() const
{
    int *lenPtr = reinterpret_cast<int *>(rep_);
    return *lenPtr;
}

int Status::GetCode() const
{
    int *codePtr = reinterpret_cast<int *>(rep_ + sizeof(int));
    return *codePtr;
}

const char *Status::GetMsg() const
{
    char *msgPtr = rep_ + sizeof(int) + sizeof(int);
    return msgPtr;
}
} // namespace Mki

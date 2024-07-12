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
#ifndef MKI_UTILS_SVECTOR_SVECTOR_H
#define MKI_UTILS_SVECTOR_SVECTOR_H
#include <vector>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>
#include <initializer_list>
#include "mki/utils/compare/compare.h"

namespace Mki {
constexpr size_t MAX_SVECTOR_SIZE = 256;
constexpr size_t DEFAULT_SVECTOR_SIZE = 48;
constexpr bool CHECK_BOUND = true;

struct MaxSizeExceeded : public std::exception {};

template <class T, class... Args>
typename std::enable_if<std::is_trivially_destructible<T>::value>::type ConstructInPlace(
    T *ptr, Args &&...args) noexcept(noexcept(T(std::forward<Args>(args)...)))
{
    new (ptr) T(std::forward<Args>(args)...);
}

template <class T, class... Args>
typename std::enable_if<!std::is_trivially_destructible<T>::value>::type ConstructInPlace(
    T *ptr, Args &&...args) noexcept(std::is_nothrow_destructible<T>::value && noexcept(T(std::forward<Args>(args)...)))

{
    if (ptr != nullptr) {
        ptr->~T();
    }

    new (ptr) T(std::forward<Args>(args)...);
}

template <class T, std::size_t MAX_SIZE = DEFAULT_SVECTOR_SIZE> class SVector {
public:
    constexpr SVector() : size_(0)
    {
        static_assert(MAX_SIZE > 0 && MAX_SIZE <= MAX_SVECTOR_SIZE);
        for (std::size_t i = 0; i < MAX_SIZE; ++i) {
            storage_[i] = T{};
        }
    }

    SVector(std::initializer_list<T> list)
    {
        static_assert(MAX_SIZE > 0 && MAX_SIZE <= MAX_SVECTOR_SIZE);
        if (CHECK_BOUND && list.size() > MAX_SIZE) {
            throw MaxSizeExceeded();
        }
        size_ = list.size();
        size_t i = 0;
        for (auto it = list.begin(); it != list.end() && i < size_; ++it) {
            storage_[i++] = *it;
        }
    }

    explicit SVector(std::size_t size, const T &value = 0) : size_(0)
    {
        static_assert(MAX_SIZE > 0 && MAX_SIZE <= MAX_SVECTOR_SIZE);
        if (CHECK_BOUND && size > MAX_SIZE) {
            throw MaxSizeExceeded();
        }

        size_ = size;
        for (std::size_t i = 0; i < size_; ++i) {
            storage_[i] = value;
        }
    }

    void push_back(const T &val) noexcept((!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value)
    {
        if (CHECK_BOUND && size_ == MAX_SIZE) {
            throw MaxSizeExceeded();
        }
        storage_[size_++] = val;
    }

    template <class... Args>
    void emplace_back(Args &&...args) noexcept(
        (!CHECK_BOUND) && noexcept(ConstructInPlace(std::declval<T *>(), std::forward<Args>(args)...)))
    {
        if (CHECK_BOUND && size_ == MAX_SIZE) {
            throw MaxSizeExceeded();
        }
        ConstructInPlace(storage_ + (size_++), std::forward<Args>(args)...);
    }

    T *begin() noexcept { return &storage_[0]; }

    const T *begin() const noexcept { return &storage_[0]; }

    T *end() noexcept
    {
        return (&storage_[0]) + size_;
    }

    const T *end() const noexcept
    {
        return (&storage_[0]) + size_;
    }

    T *erase(const T *first, const T *last) noexcept(!CHECK_BOUND)
    {
        if (first < begin() || last > end() || first > last) {
            throw std::out_of_range("erase out of range");
        }
        T *dst = begin() + (first - begin());
        if (last == end()) {
            size_ = first - begin();
        } else {
            for (auto src = last; src != end(); ++dst, ++src) {
                *dst = std::move(*src);
            }
            size_ = dst - begin();
            dst -= 1;
        }
        // This might happen if there is undefined behavior
        if (CHECK_BOUND && (size_ > MAX_SIZE)) {
            throw MaxSizeExceeded();
        }

        return dst;
    }

    T *erase(T *at) noexcept(!CHECK_BOUND)
    {
        if (at < begin() || at >= end()) {
            throw std::out_of_range("erase out of range");
        }
        T *dst = begin() + (at - begin());
        for (auto src = at + 1; src != end(); ++dst, ++src) {
            *dst = std::move(*src);
        }
        size_ -= 1;

        if (CHECK_BOUND && (size_ > MAX_SIZE)) {
            throw MaxSizeExceeded();
        }

        return begin() + (at - begin());
    }

    T &operator[](std::size_t i)
    {
        if (size_ == 0 || i >= size_) {
            throw std::out_of_range("out of range");
        }
        return storage_[i];
    }

    const T &operator[](std::size_t i) const
    {
        if (size_ == 0 || i >= size_) {
            throw std::out_of_range("out of range");
        }
        return storage_[i];
    }

    T &at(std::size_t i)
    {
        if (size_ == 0 || i >= size_) {
            throw std::out_of_range("out of range");
        }
        return storage_[i];
    }

    const T &at(std::size_t i) const
    {
        if (size_ == 0 || i >= size_) {
            throw std::out_of_range("out of range");
        }
        return storage_[i];
    }

    std::size_t size() const noexcept { return size_; }

    T *insert(const T *pos, const T &value) noexcept(
        (!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value)
    {
        if (CHECK_BOUND && size_ == MAX_SIZE) {
            throw MaxSizeExceeded();
        }
        if (pos < begin() || pos > end()) {
            throw std::out_of_range("insert out of range");
        }

        auto it = end();
        while (it != pos) {
            *it = *(it - 1);
            it -= 1;
        }
        *it = value;
        size_ += 1;
        return it;
    }

    void insert(const std::size_t pos, const T &value) noexcept
        ((!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value)
    {
        if (CHECK_BOUND && size_ == MAX_SIZE) {
            throw MaxSizeExceeded();
        }

        if (pos > size_) {
            throw MaxSizeExceeded();
        }

        for (auto it = size_; it != pos; it--) {
            storage_[it] = storage_[it - 1];
        }
        storage_[pos] = value;
        size_ += 1;
    }

    bool empty() const noexcept { return size_ == 0; }

    void clear() noexcept { size_ = 0; }

    T *data() noexcept { return &storage_[0]; }

    const T *data() const noexcept { return &storage_[0]; }

    void resize(std::size_t size) noexcept(!CHECK_BOUND)
    {
        if (CHECK_BOUND && size > MAX_SIZE) {
            throw MaxSizeExceeded();
        }

        size_ = size;
    }

    bool operator==(const SVector<T> &other) const
    {
        if (size_ != other.size_) {
            return false;
        }
        for (size_t i = 0; i < size_; ++i) {
            if (!Utils::Compare<T>::IsEqual(storage_[i], other.storage_[i])) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const SVector<T> &other) const
    {
        if (size_ != other.size_) {
            return true;
        }
        for (size_t i = 0; i < size_; ++i) {
            if (!Utils::Compare<T>::IsEqual(storage_[i], other.storage_[i])) {
                return true;
            }
        }
        return false;
    }

    bool operator<(const SVector<T> &other) const
    {
        if (size_ != other.size_) {
            return size_ < other.size_;
        }
        for (size_t i = 0; i < size_; ++i) {
            if (storage_[i] != other.storage_[i]) {
                return storage_[i] < other.storage_[i];
            }
        }
        return false;
    }

private:
    T storage_[MAX_SIZE + 1];
    std::size_t size_{0};
};

template <class T, std::size_t MAX_SIZE>
std::ostream &operator<<(std::ostream &os, const SVector<T, MAX_SIZE> &svector)
{
    if (svector.size() == 0) {
        return os;
    }

    std::string str = ",";
    for (size_t i = 0; i < svector.size(); ++i) {
        os << svector.at(i);
        if (i != svector.size() - 1) {
            os << str;
        }
    }

    return os;
}
} // namespace Mki
#endif

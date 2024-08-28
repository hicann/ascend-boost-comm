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
#include "mki/utils/file_system/file_system.h"
#include <cstring>
#include <cstdlib>
#include <climits>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/strings/str_checker.h"

namespace Mki {
constexpr size_t MAX_PATH_LEN = 256;

void FileSystem::GetDirChildItems(const std::string &dirPath, std::vector<std::string> &itemPaths)
{
    GetDirChildItemsImpl(dirPath, true, true, itemPaths);
}

void FileSystem::GetDirChildFiles(const std::string &dirPath, std::vector<std::string> &filePaths)
{
    GetDirChildItemsImpl(dirPath, true, false, filePaths);
}

void FileSystem::GetDirChildDirs(const std::string &dirPath, std::vector<std::string> &dirPaths)
{
    GetDirChildItemsImpl(dirPath, false, true, dirPaths);
}

bool FileSystem::Exists(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }
    return true;
}

bool FileSystem::IsDir(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }

    return S_ISDIR(st.st_mode);
}

std::string FileSystem::Join(const std::vector<std::string> &paths)
{
    std::string retPath;
    for (const auto &path : paths) {
        if (retPath.empty()) {
            retPath.append(path);
        } else {
            retPath.append("/" + path);
        }
    }
    return retPath;
}

int64_t FileSystem::FileSize(const std::string &filePath)
{
    struct stat st;
    if (stat(filePath.c_str(), &st) < 0) {
        return -1;
    }
    return st.st_size;
}

std::string FileSystem::BaseName(const std::string &filePath)
{
    std::string fileName;
    std::string realPath = FileSystem::PathCheckAndRegular(filePath);
    MKI_CHECK(!realPath.empty(), "file path is invalid", return "");
    const char *str = strrchr(realPath.c_str(), '/');
    if (str) {
        fileName = str + 1;
    } else {
        fileName = realPath;
    }
    return fileName;
}

std::string FileSystem::DirName(const std::string &path)
{
    if (path.size() < 1) {return "";}
    int32_t idx = static_cast<int32_t>(path.size() - 1);
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    std::string sub = path.substr(0, static_cast<uint32_t>(idx));
    const char *str = strrchr(sub.c_str(), '/');
    if (str == nullptr) {
        return ".";
    }
    idx = str - sub.c_str() - 1;
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    if (idx < 0) {
        return "/";
    }
    return path.substr(0, static_cast<uint32_t>(idx) + 1);
}

bool FileSystem::ReadFile(const std::string &filePath, uint8_t *buffer, uint64_t bufferSize)
{
    std::string realPath = FileSystem::PathCheckAndRegular(filePath);
    MKI_CHECK(!realPath.empty(), "File path is invalid", return false);
    int64_t fileSize = FileSystem::FileSize(realPath);
    if (fileSize < 0 || fileSize > MAX_FILE_SIZE) {
        MKI_LOG(ERROR) << "File size is invalid";
        return false;
    }
    std::ifstream fd(realPath.c_str(), std::ios::binary);
    if (!fd.is_open() || buffer == nullptr) {
        MKI_LOG(ERROR) << "open file failed or buffer is NULL";
        return false;
    }

    if (bufferSize > static_cast<uint64_t>(fileSize)) {
        MKI_LOG(ERROR) << "read buffer size is out of range, "
                       << " bufferSize : , " << bufferSize << " fileSize : , " << fileSize;
        return false;
    }
    fd.read(reinterpret_cast<char *>(buffer), bufferSize);
    return true;
}

bool FileSystem::WriteFile(const char *codeBuf, uint64_t codeLen, const std::string &filePath, const mode_t mode)
{
    if (codeLen > MAX_FILE_SIZE || codeBuf == nullptr) {
        MKI_LOG(ERROR) << "codeLen or codeBuf is invalid, codeLen : " << codeLen;
        return false;
    }

    char resolvedDir[PATH_MAX] = {0};
    MKI_CHECK(realpath(DirName(filePath).c_str(), resolvedDir) != nullptr,
              filePath << " realpath resolved fail", return false);

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode);
    if (fd < 0) {
        MKI_LOG(ERROR) << "open file failed or codeBuf is NULL";
        return false;
    }

    auto writeSize = write(fd, codeBuf, codeLen);
    (void)close(fd);
    if (writeSize != static_cast<ssize_t>(codeLen)) {
        MKI_LOG(ERROR) << "write file failed, writeSize : " << writeSize << ", codeLen : " << codeLen;
        return false;
    }
    return true;
}

void FileSystem::DeleteFile(const std::string &filePath)
{
    (void)remove(filePath.c_str());
}

bool FileSystem::Rename(const std::string &filePath, const std::string &newFilePath)
{
    MKI_CHECK((CheckNameValid(filePath) && CheckNameValid(newFilePath)), "invalid rename path", return false);
    int ret = rename(filePath.c_str(), newFilePath.c_str());
    return ret == 0;
}

bool FileSystem::MakeDir(const std::string &dirPath, int mode)
{
    MKI_CHECK(CheckNameValid(dirPath), "invalid mkdir path", return false);
    int ret = mkdir(dirPath.c_str(), mode);
    return ret == 0;
}

bool FileSystem::Makedirs(const std::string &dirPath, const mode_t mode)
{
    uint32_t offset = 0;
    uint32_t pathLen = dirPath.size();
    do {
        const char *str = strchr(dirPath.c_str() + offset, '/');
        offset = (str == nullptr) ? pathLen : str - dirPath.c_str() + 1;
        std::string curPath = dirPath.substr(0, offset);
        if (!Exists(curPath)) {
            if (!MakeDir(curPath, mode)) {
                return false;
            }
        }
    } while (offset != pathLen);
    return true;
}

void FileSystem::GetDirChildItemsImpl(const std::string &dirPath, bool matchFile, bool matchDir,
                                      std::vector<std::string> &itemPaths)
{
    std::string realPath = FileSystem::PathCheckAndRegular(dirPath, false);
    MKI_CHECK(!realPath.empty(), "file path is invalid", return);
    struct stat st;
    if (stat(realPath.c_str(), &st) < 0 || !S_ISDIR(st.st_mode)) {
        return;
    }

    DIR *dirHandle = opendir(realPath.c_str());
    if (dirHandle == nullptr) {
        return;
    }

    struct dirent *dp = nullptr;
    while ((dp = readdir(dirHandle)) != nullptr) {
        const int parentDirNameLen = 2;
        if (strncmp(dp->d_name, ".", 1) == 0 || strncmp(dp->d_name, "..", parentDirNameLen) == 0) {
            continue;
        }
        std::string itemPath = FileSystem::PathCheckAndRegular(realPath + "/" + dp->d_name);
        if (itemPath.empty()) {
            MKI_LOG(ERROR) << "File path is invalid";
            closedir(dirHandle);
            return;
        }

        stat(itemPath.c_str(), &st);
        if (itemPaths.size() > MAX_FILE_NUM) {
            MKI_LOG(ERROR) << "Max file number exceeded ";
            break;
        }
        if ((matchDir && S_ISDIR(st.st_mode)) || (matchFile && S_ISREG(st.st_mode))) {
            itemPaths.push_back(itemPath.c_str());
        }
    }

    closedir(dirHandle);
}

bool FileSystem::IsSymLink(const std::string &filePath)
{
    struct stat buf {};
    if (lstat(filePath.c_str(), &buf) != 0) {
        return false;
    }
    return S_ISLNK(buf.st_mode);
}

std::string FileSystem::PathCheckAndRegular(const std::string &path, bool symlinkCheck, bool parentReferenceCheck)
{
    // 1. check if "path" is empty
    if (path.empty()) {
        MKI_LOG(ERROR) << "path string is NULL";
        return "";
    }

    // 2. check the length of "path"
    if (path.size() >= PATH_MAX) {
        MKI_LOG(ERROR) << "file path " << path.c_str() << " is too long!";
        return "";
    }

    // 3. check if "path" contains parent directory reference
    if (parentReferenceCheck && path.find("..") != std::string::npos) {
        MKI_LOG(ERROR) << "file path " << path.c_str() << " contains parent directory reference!";
        return "";
    }

    // 4. check if is symbolic link
    if (symlinkCheck) {
        std::string regularPath = RemoveTrailingSlash(path);
        if (IsSymLink(regularPath)) {
            MKI_LOG(ERROR) << "file path " << path.c_str() << " is symbolic link!";
            return "";
        }
    }

    // 5. get the real path
    char resolvedPath[PATH_MAX] = {0};
    std::string res = "";

    if (realpath(path.c_str(), resolvedPath) != nullptr) {
        res = resolvedPath;
    } else {
        MKI_LOG(ERROR) << "path " << path.c_str() << " is not exist";
    }
    return res;
}

std::string FileSystem::RemoveTrailingSlash(const std::string &path)
{
    size_t lastNonSlash = path.find_last_not_of("/");
    // If there is no non-/ character and the path is not enpty, / is returned.
    if (lastNonSlash == std::string::npos && path.size() > 1) {
        return "/";
    // When the last non-/ character is not the last character,
    // return the substring from the beginning of the path to the last non-/ character.
    } else if (lastNonSlash != std::string::npos && lastNonSlash != path.size() - 1) {
        return path.substr(0, lastNonSlash + 1);
    } else {
        return path;
    }
}
} // namespace Mki

/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
#ifndef ATHENA_LOG_H
#define ATHENA_LOG_H

#include <athena/core/AbstractLoger.h>
#include <athena/core/Logger.h>
#include <iostream>
#include <memory>

namespace athena {
class LogHolder {
    std::unique_ptr<core::AbstractLogger> mLog;
    std::unique_ptr<core::AbstractLogger> mErr;

    template <typename LoggerType, typename... Args>
    void setStream(std::unique_ptr<core::AbstractLogger>& stream, Args&&... args) {
        stream.reset(new LoggerType(std::forward<Args>(args)...));
    }
 public:
    LogHolder()
        : mLog(std::make_unique<core::Logger>(std::cout)),
          mErr(std::make_unique<core::Logger>(std::cerr)) {}
    ~LogHolder() = default;
    LogHolder(const LogHolder& rhs) = delete;
    LogHolder(LogHolder&& rhs) noexcept = delete;

    LogHolder& operator=(const LogHolder& rhs) = delete;
    LogHolder& operator=(LogHolder&& rhs) noexcept = delete;

    template <typename LoggerType, typename... Args>
    friend void setLogStream(Args&&... args);
    template <typename LoggerType, typename... Args>
    friend void setErrStream(Args&&... args);
    friend core::AbstractLogger &log();
    friend core::AbstractLogger &err();
};

extern const std::unique_ptr<LogHolder> logHolder;

template <typename LoggerType, typename... Args>
void setLogStream(Args&&... args) {
    logHolder->setStream<LoggerType>(logHolder->mLog, std::forward<Args>(args)...);
}
template <typename LoggerType, typename... Args>
void setErrStream(Args&&... args) {
    logHolder->setStream<LoggerType>(logHolder->mErr, std::forward<Args>(args)...);
}
core::AbstractLogger &log();
core::AbstractLogger &err();
}

#endif //ATHENA_LOG_H

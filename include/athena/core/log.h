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
namespace {
std::unique_ptr<core::AbstractLogger> mLog = std::make_unique<core::Logger>(std::cout);
std::unique_ptr<core::AbstractLogger> mErr = std::make_unique<core::Logger>(std::cerr);

template <typename LoggerType, typename... Args>
void setStream(std::unique_ptr<core::AbstractLogger>& stream, Args&&... args) {
    stream.reset(new LoggerType(std::forward<Args>(args)...));
}
}

core::AbstractLogger &log();
core::AbstractLogger &err();

template <typename LoggerType, typename... Args>
void setLogStream(Args&&... args) {
    setStream<LoggerType>(mLog, std::forward<Args>(args)...);
}

template <typename LoggerType, typename... Args>
void setErrStream(Args&&... args) {
    setStream<LoggerType>(mErr, std::forward<Args>(args)...);
}
}

#endif //ATHENA_LOG_H

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
#include <athena/core/log.h>
#include <athena/core/AbstractLoger.h>
#include <athena/core/Logger.h>
#include <iostream>
#include <memory>

namespace athena {
namespace {
std::unique_ptr<core::AbstractLogger> mLog = std::make_unique<core::Logger>(std::cout);
std::unique_ptr<core::AbstractLogger> mErr = std::make_unique<core::Logger>(std::cerr);
}

core::AbstractLogger &log() {
    return *mLog;
}

core::AbstractLogger &err() {
    return *mErr;
}
}
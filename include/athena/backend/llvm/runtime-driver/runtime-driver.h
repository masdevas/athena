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

#ifndef ATHENA_RUNTIME_DRIVER_H
#define ATHENA_RUNTIME_DRIVER_H

#include <athena/core/FatalError.h>

#include <dlfcn.h>
#include <string>
#include <string_view>

namespace athena::backend {
class RuntimeDriver {
    void* mLibraryHandle;
    void (*mFaddPointer)(void* a, size_t ca, void* b, size_t cb, void* c);

    void* getFunction(std::string_view nameFunction);

    public:
    RuntimeDriver();
    RuntimeDriver(std::string_view nameLibrary);
    RuntimeDriver(const RuntimeDriver& rhs) = delete;
    RuntimeDriver(RuntimeDriver&& rhs) noexcept = default;
    ~RuntimeDriver();

    RuntimeDriver& operator=(const RuntimeDriver& rhs) = delete;
    RuntimeDriver& operator=(RuntimeDriver&& rhs) noexcept;

    void load(std::string_view nameLibrary);
    void unload();
    void reload(std::string_view nameLibrary);
    bool isLoaded() const;

    void fadd(void* a, size_t ca, void* b, size_t cb, void* c) {
        mFaddPointer(a, ca, b, cb, c);
    }
};

extern RuntimeDriver kRuntimeDriver;

extern "C" {
void inline fadd(void* a, size_t ca, void* b, size_t cb, void* c) {
    kRuntimeDriver.fadd(a, ca, b, cb, c);
}
}
}  // namespace athena::backend

#endif  // ATHENA_RUNTIME_DRIVER_H

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

#include <athena/backend/llvm/runtime-driver/runtime-driver.h>

namespace athena::backend {

RuntimeDriver kRuntimeDriver;

RuntimeDriver::RuntimeDriver() : mLibraryHandle(nullptr) {}
RuntimeDriver::RuntimeDriver(std::string_view nameLibrary) {
    load(nameLibrary);
}
RuntimeDriver::~RuntimeDriver() { unload(); }
RuntimeDriver& RuntimeDriver::operator=(RuntimeDriver&& rhs) noexcept {
    unload();
    mLibraryHandle     = rhs.mLibraryHandle;
    mFaddPointer       = rhs.mFaddPointer;
    rhs.mLibraryHandle = nullptr;
    rhs.mFaddPointer   = nullptr;
    return *this;
}
void* RuntimeDriver::getFunction(std::string_view nameFunction) {
    if (void* function = dlsym(mLibraryHandle, nameFunction.data());
        !function) {
        ::athena::core::FatalError(1, "RuntimeDriver: " + std::string(dlerror()));
        return nullptr;
    } else {
        return function;
    }
}
void RuntimeDriver::load(std::string_view nameLibrary) {
    if (mLibraryHandle = dlopen(nameLibrary.data(), RTLD_LAZY);
        !mLibraryHandle) {
        ::athena::core::FatalError(1, "RuntimeDriver: " + std::string(dlerror()));
    }
    mFaddPointer =
        reinterpret_cast<void (*)(void*, size_t, void*, size_t, void*)>(
            getFunction("fadd"));
}
void RuntimeDriver::unload() {
    if (mLibraryHandle && dlclose(mLibraryHandle)) {
        ::athena::core::FatalError(1, "RuntimeDriver: " + std::string(dlerror()));
    }
    mLibraryHandle = nullptr;
}
void RuntimeDriver::reload(std::string_view nameLibrary) {
    unload();
    load(nameLibrary);
}
bool RuntimeDriver::isLoaded() const { return mLibraryHandle != nullptr; }
}  // namespace athena::backend

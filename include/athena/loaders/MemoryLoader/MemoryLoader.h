/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_MEMORYLOADER_H
#define ATHENA_MEMORYLOADER_H

#include <athena/core/AbstractLoader.h>
#include <athena/core/Allocator.h>

namespace athena::loaders {
/**
 * Load data from RAM to Tensor
 */
class MemoryLoader : public core::AbstractLoader {
    private:
    void *mData;
    size_t mSize;

    public:
    MemoryLoader(void *data, size_t size) : mData(data), mSize(size){};
    MemoryLoader(const MemoryLoader &) = default;
    MemoryLoader(MemoryLoader &&src) noexcept;
    ~MemoryLoader() = default;
    MemoryLoader &operator=(const MemoryLoader &) = default;
    MemoryLoader &operator=(MemoryLoader &&src) noexcept;

    virtual void load(core::Allocator *, core::inner::Tensor *) override;
    std::string getLoadCName() const override {
        static const std::string loadName = "MemoryLoaderLoad";
        return loadName;
    };
    std::string getCreateCName() const override {
        static const std::string createName = "CreateMemoryLoader";
        return createName;
    };

    virtual std::string getName() const override {
        return "MemoryLoader";
    }

    std::string serialize() const override;

    static AbstractLoader *deserialize(const std::string &data) {
        new core::FatalError(-1, "MemoryLoader is not serializable");
    }
};
}  // namespace athena::loaders

extern "C" {
void MemoryLoaderLoad(void *, void *, void *);
void *CreateMemoryLoader(void *, size_t);
}

#endif  // ATHENA_MEMORYLOADER_H

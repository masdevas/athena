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

#ifndef ATHENA_ABSTRACTLOADER_H
#define ATHENA_ABSTRACTLOADER_H

#include <athena/core/Allocator.h>

#include <string>
#include <string_view>

namespace athena::core {

/**
 * Loaders is a concept that helps Athena put user data into Graph
 */
class AbstractLoader {
    public:
    /**
     * Do actual data load
     */
    virtual void load(Allocator *, inner::Tensor *) = 0;
    /**
     * Get C-style function name that does actual load
     * For backend usage only
     * @return C-style function name string
     */
    virtual std::string getLoadCName() const = 0;
    /**
     * Get C-style function name that creates loader object
     * For backend usage only
     * @return C-style function name string
     */
    virtual std::string getCreateCName() const = 0;
};

/**
 * Dummy loader for testing purposes only
 */
class DummyLoader : public AbstractLoader {
    public:
    void load(Allocator *, inner::Tensor *tensor) override {}
    std::string getLoadCName() const override {
        static const std::string loadName = "DummyLoad";
        return loadName;
    }
    virtual std::string getCreateCName() const override {
        static const std::string createName = "DummyCreate";
        return createName;
    }
};
}  // namespace athena::core

#endif  // ATHENA_ABSTRACTLOADER_H

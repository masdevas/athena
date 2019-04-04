/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#ifndef ATHENA_TABLE_H
#define ATHENA_TABLE_H

#include <athena/core/inner/NullRecord.h>

#include <vector>

namespace athena::core::inner {
constexpr size_t kKUndefinedIndex = 0;

template <typename Content>
class Table {
 private:
    Content const mNullRecord;
    std::vector<Content> mRegisteredContents;

 public:
    Table();
    Table(const Table &) = delete;
    Table(Table &&rhs) = delete;
    ~Table() = default;

    Table &operator=(const Table &rhs) = delete;
    Table &operator=(Table &&rhs) = delete;
    Content& operator[](size_t index);

    template <typename ...Args>
    size_t registerRecord(Args&&... args);
    size_t size();
    void clear();
};

template <typename Content>
Table<Content>::Table() : mNullRecord(NullRecord<Content>::value) {
    mRegisteredContents.emplace_back(mNullRecord);
}

template <typename Content>
Content& Table<Content>::operator[](size_t index) {
    return mRegisteredContents[index];
}

template <typename Content>
template <typename ...Args>
size_t Table<Content>::registerRecord(Args&&... args) {
    mRegisteredContents.emplace_back(std::forward<Args>(args)...);
    return mRegisteredContents.size() - 1;
}

template <typename Content>
size_t Table<Content>::size() {
    return mRegisteredContents.size();
}

template <typename Content>
void Table<Content>::clear() {
    mRegisteredContents.clear();
    mRegisteredContents.emplace_back(mNullRecord);
}
}

#endif //ATHENA_TABLE_H

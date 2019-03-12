/*
 * Copyright (c) 2019 Alexander Batashev. All rights reserved.
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
#ifndef ATHENA_VMALLOCATIONTABLE_H
#define ATHENA_VMALLOCATIONTABLE_H

#include <vector>
#include <athena/core/Tensor.h>

namespace athena::backend::llvm {

struct AllocationTableRecord {
    size_t address;
    core::Tensor *tensor;
};

class VMAllocationTable {
 private:
    std::vector<AllocationTableRecord> mRegisteredTensors;
    size_t mNextCell;

 public:
    VMAllocationTable() : mNextCell(1) {};
    ~VMAllocationTable() = default;
    VMAllocationTable(const VMAllocationTable&) = delete;
    VMAllocationTable& operator=(const VMAllocationTable&) = delete;

    void registerTensor(core::Tensor *tensor);
    core::Tensor *lookup(size_t address);
};
}

#endif //ATHENA_VMALLOCATIONTABLE_H

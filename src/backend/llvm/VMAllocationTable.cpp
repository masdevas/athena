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
#include <athena/backend/llvm/VMAllocationTable.h>
using namespace athena::backend::llvm;
using namespace athena::core;

void VMAllocationTable::registerTensor(Tensor *tensor) {
    if (tensor != nullptr && tensor->getVirtualAddress() == 0) {
        tensor->setVirtualAddress(mNextCell);
        mRegisteredTensors.push_back(AllocationTableRecord{mNextCell, tensor});
        mNextCell += tensor->getShape().getTotalSize() + 1;
    }
}
Tensor *VMAllocationTable::lookup(size_t address) {
    if (address >= mNextCell) {
        return nullptr;
    }

    Tensor *result = nullptr;

    for (auto &record : mRegisteredTensors) {
        if (record.address <= address &&
            record.address + record.tensor->getShape().getTotalSize() >
                address) {
            result = record.tensor;  // todo extract subtensor
            break;
        }
    }

    return result;
}

/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "CpuDevice.h"
bool athena::backend::llvm::CPUDevice::isPartitionSupported(
    athena::backend::llvm::Device::PartitionDomain) {
    return false;
}
athena::backend::llvm::DeviceContainer
athena::backend::llvm::CPUDevice::partition(
    athena::backend::llvm::Device::PartitionDomain) {
    return DeviceContainer();
}

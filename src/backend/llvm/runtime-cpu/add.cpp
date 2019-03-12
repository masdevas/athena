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

#include <athena/backend/llvm/runtime/add.h>

#include <iostream>

void fadd(void *a, size_t ca, void *b, size_t cb, void *c) {
    auto af = reinterpret_cast<float *>(a);
    auto bf = reinterpret_cast<float *>(b);
    auto cf = reinterpret_cast<float *>(c);

    for (int i = 0; i < ca; i++) {
        cf[i] = af[i] + bf[i];
        std::cout << cf[i] << "\n";
    }
}

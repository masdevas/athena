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


#ifndef ATHENA_ALLOCATE_H
#define ATHENA_ALLOCATE_H

#if __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif

void allocate(void *allocator, void *tensor);
size_t get_fast_pointer(void *allocator, void *tensor);

#if __cplusplus
}
#endif

#endif //ATHENA_ALLOCATE_H

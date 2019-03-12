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
#ifndef ATHENA_ADD_H
#define ATHENA_ADD_H

#if __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif

void fadd(void *a, size_t ca, void *b, size_t cb, void *c);

#if __cplusplus
}
#endif

#endif  // ATHENA_ADD_H

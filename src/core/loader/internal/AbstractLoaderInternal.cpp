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

#include <athena/core/loader/internal/AbstractLoaderInternal.h>

namespace athena::core::internal {
AbstractLoaderInternal::AbstractLoaderInternal(utils::WeakPtr<ContextInternal> context,
                                               utils::Index publicIndex, utils::String name)
  : Entity(std::move(context), publicIndex, std::move(name)) {}
}

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
#include <athena/core/Clear.h>
#include <athena/core/inner/GlobalTables.h>

namespace athena::core {
void clearAll() {
    inner::getGraphTable().clear();
    inner::getNodeTable().clear();
    inner::getAllocationTable().clear();
}
}
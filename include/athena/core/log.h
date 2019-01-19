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
#ifndef ATHENA_LOG_H
#define ATHENA_LOG_H

#include "AbstractLoger.h"


namespace athena {
core::AbstractLogger &log();
core::AbstractLogger &err();
}

#endif //ATHENA_LOG_H

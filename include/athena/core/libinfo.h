/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <string>

#ifndef ATHENA_LIBINFO_H
#define ATHENA_LIBINFO_H

extern "C" {
const char* getMajorVersion();
const char* getMinorVersion();
const char* getPatchVersion();
const char* getBuildVersion();
}

#endif  // ATHENA_LIBINFO_H

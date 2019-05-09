# coding=utf-8
# Copyright (c) 2018 Athena. All rights reserved.
# https://athenaframework.ml
#
# Licensed under MIT license.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as et
import platform

main_env_var_name = "ATHENA_TEST_ENVIRONMENT"
ci_environment_code = "ci"
dev_environment_code = "dev"

commands_tag = "commands"
set_command_tag = "set"
value_tag = "value"

alias_attrib_name = "name"
alias_attrib_env = "env"
alias_attrib_os = "os"


def fatal_error(*args, error_code=1):
    print("(!) Integration test system error:", *args, file=sys.stderr)
    sys.exit(error_code)


def get_env_code() -> str:
    environment = os.environ.get(main_env_var_name)
    if environment is None or environment == "dev":
        return dev_environment_code
    elif environment == "ci":
        return ci_environment_code
    else:
        fatal_error("Undefined value of ATHENA_TEST_ENVIRONMENT:", environment)


def substitute_env_vars(value) -> str:
    found = re.findall(r'\$[a-zA-Z][\w]*', value)
    for found_item in found:
        env_var_name = found_item[1:]
        env_var = os.environ.get(env_var_name)
        if env_var is None:
            fatal_error("Environment variable", "\"" + env_var_name + "\"", "exists")
        value = value.replace(found_item, env_var)
    return value


def set_env_var_with_value(var_name, value):
    value = substitute_env_vars(value)
    os.environ[var_name] = value


def set_env_var(set_command, env_code):
    target_os = set_command.attrib.get(alias_attrib_os)
    if target_os is not None and target_os != platform.system():
        return

    var_name = set_command.attrib.get(alias_attrib_name)
    if len(set_command) == 0:
        if set_command.attrib.get(alias_attrib_env) is not None:
            fatal_error("Single-line set-command can't have an attribute \"env\"")
        set_env_var_with_value(var_name, set_command.text)
    else:
        for value in set_command:
            if value.tag == value_tag:
                if value.attrib.get(alias_attrib_env) == env_code:
                    set_env_var_with_value(var_name, value.text)
                    break
            else:
                fatal_error("Unknown tag", value.tag)
        else:
            fatal_error("Value for variable", "\"" + var_name + "\"",
                        "for env", "\"" + env_code + "\"", "is not defined")


def get_commands_tree_from_xml(file_name: str):
    if not os.path.isfile(file_name):
        return None
    return et.ElementTree(file=file_name)


def exec_commands(tree_commands, env_code: str):
    if tree_commands is None:
        return
    commands = tree_commands.getroot()
    if commands.tag != commands_tag:
        fatal_error("Configure file hasn't tag \"commands\"")
    for command in commands:
        if command.tag == set_command_tag:
            set_env_var(command, env_code)
        else:
            fatal_error("Unknown command:", command.tag)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        fatal_error("Wrong count arguments for script. Args: ", *sys.argv)
    target = sys.argv[1]
    name_config = target + ".xml"
    commands_tree = get_commands_tree_from_xml(name_config)
    req_env_code = get_env_code()
    exec_commands(commands_tree, req_env_code)
    print("-- Integration test output for target", "\"" + target + "\"", ":")
    try:
        subprocess.check_call(["./" + target], stdout=sys.stdout, shell=True)
    except subprocess.CalledProcessError as exc:
        fatal_error("Error code from subprocess", "\"" + target + "\"", ":", exc.returncode,
                    "\nOutput: ", exc.output, error_code=exc.returncode)

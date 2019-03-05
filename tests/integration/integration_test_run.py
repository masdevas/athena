
import sys
import os
import subprocess
import xml.etree.ElementTree as et

ci_environment_config = "TestSuite_ci.xml"
dev_environment_config = "TestSuite_dev.xml"
main_environment_name = "ATHENA_TEST_ENVIRONMENT"

def get_name_config():
    name_config = None
    environment = os.environ.get(main_environment_name)
    if environment is None or environment == "dev":
        name_config = dev_environment_config
    elif environment == "ci":
        name_config = ci_environment_config
    else:
        print("-- Error python script: Undefined value of \
ATHENA_TEST_ENVIRONMENT: ", environment, file=sys.stderr)
        sys.exit()
    return name_config

def set_env_variables(name_config : str):
    if not os.path.isfile(name_config):
        return
    tree = et.ElementTree(file=name_config)
    root = tree.getroot()
    for child in root:
        for name, value in zip(child.attrib.keys(), child.attrib.values()):
            os.environ[name] = value

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("-- Error python script: Wrong count arguments", file=sys.stderr)
        sys.exit()
    target = sys.argv[1]
    name_config = get_name_config()
    set_env_variables(name_config)
    print("-- Integration test output", target, ":")
    subprocess.run(["./" + target], stdout=sys.stdout)
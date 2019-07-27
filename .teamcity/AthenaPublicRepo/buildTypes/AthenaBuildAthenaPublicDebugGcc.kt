package AthenaPublicRepo.buildTypes

import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.exec
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.script

object AthenaBuildAthenaPublicDebugGcc : BuildType({
    name = "[Debug][GCC] Build"

    params {
        param("cc", "gcc-8")
        param("image_name", "gcc")
        param("cxx", "g++-8")
        param("repo", "athenaml/athena")
        param("env.SRC_DIR", "%system.teamcity.build.checkoutDir%")
        param("env.ATHENA_BINARY_DIR", "%teamcity.build.checkoutDir%/install_Debug_GCC")
        param("env.BUILD_PATH", "%teamcity.build.checkoutDir%/build_Debug_GCC")
    }

    vcs {
        root(AthenaPublicRepo.vcsRoots.AthenaPublic)
    }

    steps {
        exec {
            name = "Build"
            path = "scripts/build.py"
            arguments = "--ninja --build-type Debug --install-dir=%env.ATHENA_BINARY_DIR%  --build-type Debug %env.BUILD_PATH% %env.SRC_DIR%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
            dockerRunParameters = "-e CC=%cc% -e CXX=%cxx% -e ATHENA_BINARY_DIR=%env.ATHENA_BINARY_DIR% -e ATHENA_TEST_ENVIRONMENT=ci"
        }
        script {
            name = "Install"
            scriptContent = "cd %env.BUILD_PATH% && cmake --build . --target install;"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
        }
        exec {
            name = "Test"
            path = "scripts/test.sh"
            arguments = "%env.BUILD_PATH%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
        feature {
            type = "xml-report-plugin"
            param("xmlReportParsing.reportType", "ctest")
            param("xmlReportParsing.reportDirs", "+:build_Debug_GCC/Testing/**/*.xml")
        }
    }

    requirements {
        equals("docker.server.osType", "linux")
    }
})

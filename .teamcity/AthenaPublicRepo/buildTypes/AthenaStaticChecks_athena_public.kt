package AthenaPublicRepo.buildTypes

import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.ExecBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.exec
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.script
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.failOnText

object AthenaStaticChecks_athena_public : BuildType({
    name = "Static Checks"

    artifactRules = """
        +:format-fixes.diff
        +:tidy-fixes.yaml
        +:infer-out/** => infer.zip
    """.trimIndent()

    params {
        param("env.SRC_DIR", "%system.teamcity.build.checkoutDir%")
        param("target_branch", "")
        param("repo", "athenaml/athena")
    }

    vcs {
        root(AthenaPublicRepo.vcsRoots.AthenaPublic)
    }

    steps {
        exec {
            name = "Get compile commands"
            path = "scripts/build.py"
            arguments = "--no-build --ninja --disable-tests --compile-commands %teamcity.build.checkoutDir%/comp_comm %teamcity.build.checkoutDir%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/llvm8:latest"
            dockerRunParameters = "-e CC=clang-8 -e CXX=clang++-8"
        }
        script {
            name = "Clang Tidy"
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            scriptContent = "run-clang-tidy-8.py -export-fixes=tidy-fixes.yaml -header-filter='%teamcity.build.checkoutDir%/include' -p=%teamcity.build.checkoutDir%/comp_comm %teamcity.build.checkoutDir%/src"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/llvm8:latest"
            dockerPull = true
        }
        script {
            name = "Infer"
            enabled = false
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            scriptContent = "infer --compilation-database %teamcity.build.checkoutDir%/comp_comm/compilation_database.json --html"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/infer:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
        }
        exec {
            name = "Clang Format"
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            path = "scripts/clang-format.sh"
            arguments = "%target_branch%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/llvm8:latest"
            dockerImagePlatform = ExecBuildStep.ImagePlatform.Linux
            dockerPull = true
            dockerRunParameters = "-e CLANG_FORMAT_DIFF_EXEC=/usr/lib/llvm-8/share/clang/clang-format-diff.py"
        }
    }

    failureConditions {
        failOnText {
            enabled = false
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "warning:"
            failureMessage = "There were errors on static checks"
            reverse = false
        }
        failOnMetricChange {
            metric = BuildFailureOnMetric.MetricType.ARTIFACT_SIZE
            threshold = 40
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.MORE
            compareTo = value()
            param("anchorBuild", "lastSuccessful")
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
    }

    cleanup {
        artifacts(builds = 1, days = 7)
    }
})

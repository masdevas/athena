package AthenaPublicRepo.buildTypes

import AthenaAlexFork.vcsRoots.AthenaAlex
import AthenaPublicRepo.vcsRoots.AthenaPublic
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.sshAgent
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.script
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs

object UpdateDocs : BuildType({
    name = "Update Docs"

    enablePersonalBuilds = false
    artifactRules = """
        +:website/build => docs.zip
        +:website/*.log => logs.zip
    """.trimIndent()
    type = BuildTypeSettings.Type.DEPLOYMENT
    maxRunningBuilds = 1
    publishArtifacts = PublishMode.ALWAYS

    vcs {
        root(AthenaPublicRepo.vcsRoots.AthenaPublic)
        root(AthenaPublicRepo.vcsRoots.HttpsGithubComAthenamlWebsiteRefsHeadsMaster, "+:. => website")

        checkoutMode = CheckoutMode.ON_AGENT
    }

    steps {
        script {
            name = "Generate doxygen docs"
            scriptContent = """
                scripts/build.py --no-build --docs-only --build-type Debug %teamcity.build.checkoutDir%/doxygen %teamcity.build.checkoutDir%
                cmake --build %teamcity.build.checkoutDir%/doxygen --target docs
            """.trimIndent()
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/docs:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
        }
        script {
            name = "Build website"
            scriptContent = """
                cd %teamcity.build.checkoutDir%/website
                npm install
                npm run build
                mkdir -p %teamcity.build.checkoutDir%/website/build/athenaml.github.io/api/develop
                cp -R %teamcity.build.checkoutDir%/doxygen/docs/* %teamcity.build.checkoutDir%/website/build/athenaml.github.io/api/develop/
            """.trimIndent()
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/docs:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
        }
        script {
            name = "Deploy"
            scriptContent = """
                cd %teamcity.build.checkoutDir%/website
                git config --global user.email "bot@getathena.ml"
                git config --global user.name "Athena Build Bot"
                git clone git@github.com:athenaml/athenaml.github.io.git
                cp -R build/athenaml.github.io/* athenaml.github.io/
                cd athenaml.github.io
                git add .
                git commit -m "Update website"
                git push --all
                cd ..
                rm -r athenaml.github.io
            """.trimIndent()
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
        }
    }

    triggers {
        vcs {
            branchFilter = """
                +:refs/heads/master
                +:<default>
            """.trimIndent()
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
        sshAgent {
            teamcitySshKey = "teamcity_github"
        }
        pullRequests {
            vcsRootExtId = "${AthenaPublic.id}"
            provider = github {
                authType = vcsRoot()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
        pullRequests {
            vcsRootExtId = "${AthenaAlex.id}"
            provider = github {
                authType = vcsRoot()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
    }

    requirements {
        equals("system.agent.name", "Default Agent")
    }
})

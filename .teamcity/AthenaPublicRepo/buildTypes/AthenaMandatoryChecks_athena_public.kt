package AthenaPublicRepo.buildTypes

import AthenaPublicRepo.vcsRoots.AthenaPublic
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.VcsTrigger
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs

object AthenaMandatoryChecks_athena_public : BuildType({
    name = "Pull Request / Post commit"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(AthenaPublicRepo.vcsRoots.AthenaPublic)

        showDependenciesChanges = true
    }

    triggers {
        vcs {
            quietPeriodMode = VcsTrigger.QuietPeriodMode.USE_DEFAULT
            triggerRules = "+:root=${AthenaPublic.id}:**"

            branchFilter = """
                +:pull/*
                +:develop
                +:master
            """.trimIndent()
            watchChangesInDependencies = true
            perCheckinTriggering = true
            groupCheckinsByCommitter = true
            enableQueueOptimization = false
        }
    }

    features {
        pullRequests {
            vcsRootExtId = "${AthenaPublic.id}"
            provider = github {
                authType = vcsRoot()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
        commitStatusPublisher {
            vcsRootExtId = "${AthenaPublic.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "credentialsJSON:c3f84f17-e571-495d-a804-ea7a25b20d89"
                }
            }
        }
        commitStatusPublisher {
            vcsRootExtId = "${AthenaPublic.id}"
            publisher = upsource {
                serverUrl = "https://upsource.getathena.ml"
                projectId = "ATHENA"
                userName = "admin"
                password = "credentialsJSON:3185188d-44f7-4583-b5f5-3b43968674f3"
            }
        }
    }

    dependencies {
        snapshot(AthenaBuildAthenaPublicDebugClang) {
        }
        snapshot(AthenaBuildAthenaPublicDebugGcc) {
        }
        snapshot(AthenaBuildAthenaPublicReleaseClang) {
        }
        snapshot(AthenaBuildAthenaPublicReleaseGcc) {
        }
        snapshot(AthenaStaticChecks_athena_public) {
        }
    }
})

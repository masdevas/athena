package AthenaAndreyFork.buildTypes

import AthenaAndreyFork.vcsRoots.AthenaAndrey
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.VcsTrigger
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs

object AthenaMandatoryChecks_athena_andrey : BuildType({
    name = "Pull Request / Post commit"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(AthenaAndreyFork.vcsRoots.AthenaAndrey)

        showDependenciesChanges = true
    }

    triggers {
        vcs {
            quietPeriodMode = VcsTrigger.QuietPeriodMode.USE_DEFAULT
            triggerRules = "+:root=${AthenaAndrey.id}:**"

            branchFilter = """
                +:*
                -:develop
                -:master
                -:pull/*
                -:pull/*/merge
            """.trimIndent()
            watchChangesInDependencies = true
            perCheckinTriggering = true
            groupCheckinsByCommitter = true
            enableQueueOptimization = false
        }
    }

    dependencies {
        snapshot(AthenaBuildAthenaAndreyDebugClang) {
        }
        snapshot(AthenaBuildAthenaAndreyDebugGcc) {
        }
        snapshot(AthenaBuildAthenaAndreyReleaseClang) {
        }
        snapshot(AthenaBuildAthenaAndreyReleaseGcc) {
        }
        snapshot(AthenaStaticChecks_athena_andrey) {
        }
    }
})

package AthenaAlexFork.buildTypes

import AthenaAlexFork.vcsRoots.AthenaAlex
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.VcsTrigger
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs

object AthenaMandatoryChecks_athena_alex : BuildType({
    name = "Pull Request / Post commit"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(AthenaAlexFork.vcsRoots.AthenaAlex)

        showDependenciesChanges = true
    }

    triggers {
        vcs {
            quietPeriodMode = VcsTrigger.QuietPeriodMode.USE_DEFAULT
            triggerRules = "+:root=${AthenaAlex.id}:**"

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
        snapshot(AthenaBuildAthenaAlexDebugClang) {
        }
        snapshot(AthenaBuildAthenaAlexDebugGcc) {
        }
        snapshot(AthenaBuildAthenaAlexReleaseClang) {
        }
        snapshot(AthenaBuildAthenaAlexReleaseGcc) {
        }
        snapshot(AthenaStaticChecks_athena_alex) {
        }
    }
})

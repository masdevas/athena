package AthenaPublicRepo.buildTypes

import AthenaPublicRepo.vcsRoots.AthenaPublic
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.investigationsAutoAssigner
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.vcsLabeling
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.schedule

object Daily : BuildType({
    name = "Daily"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(AthenaPublicRepo.vcsRoots.AthenaPublic)

        checkoutMode = CheckoutMode.ON_SERVER
        showDependenciesChanges = true
    }

    triggers {
        schedule {
            schedulingPolicy = daily {
                timezone = "Europe/Moscow"
            }
            branchFilter = "+:<default>"
            triggerRules = "+:"
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }

    features {
        investigationsAutoAssigner {
            defaultAssignee = "abatashev"
        }
        vcsLabeling {
            vcsRootId = "${AthenaPublic.id}"
            labelingPattern = "nightly-27-07-2019"
            branchFilter = ""
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

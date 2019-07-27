package AthenaAndreyFork

import AthenaAndreyFork.buildTypes.*
import AthenaAndreyFork.vcsRoots.*
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.Project

object Project : Project({
    id("AthenaAndreyFork")
    name = "Andrey Fork"

    vcsRoot(AthenaAndrey)

    buildType(AthenaBuildAthenaAndreyDebugGcc)
    buildType(AthenaBuildAthenaAndreyDebugClang)
    buildType(AthenaBuildAthenaAndreyReleaseClang)
    buildType(AthenaBuildAthenaAndreyReleaseGcc)
    buildType(AthenaMandatoryChecks_athena_andrey)
    buildType(AthenaStaticChecks_athena_andrey)
})

package AthenaAlexFork

import AthenaAlexFork.buildTypes.*
import AthenaAlexFork.vcsRoots.*
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.Project

object Project : Project({
    id("AthenaAlexFork")
    name = "Alex Fork"

    vcsRoot(AthenaAlex)

    buildType(AthenaStaticChecks_athena_alex)
    buildType(AthenaBuildAthenaAlexDebugClang)
    buildType(AthenaBuildAthenaAlexDebugGcc)
    buildType(AthenaMandatoryChecks_athena_alex)
    buildType(AthenaBuildAthenaAlexReleaseClang)
    buildType(AthenaBuildAthenaAlexReleaseGcc)
})

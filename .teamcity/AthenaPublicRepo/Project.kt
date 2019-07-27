package AthenaPublicRepo

import AthenaPublicRepo.buildTypes.*
import AthenaPublicRepo.vcsRoots.*
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.Project

object Project : Project({
    id("AthenaPublicRepo")
    name = "Athena Public"

    vcsRoot(HttpsGithubComAthenamlAthenamlGithubIoRefsHeadsMaster)
    vcsRoot(HttpsGithubComAthenamlWebsiteRefsHeadsMaster)
    vcsRoot(AthenaPublic)

    buildType(AthenaBuildAthenaPublicDebugGcc)
    buildType(AthenaMandatoryChecks_athena_public)
    buildType(Daily)
    buildType(AthenaStaticChecks_athena_public)
    buildType(AthenaBuildAthenaPublicReleaseGcc)
    buildType(AthenaBuildAthenaPublicDebugClang)
    buildType(AthenaBuildAthenaPublicReleaseClang)
    buildType(UpdateDocs)
})

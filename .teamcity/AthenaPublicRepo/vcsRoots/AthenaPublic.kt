package AthenaPublicRepo.vcsRoots

import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

object AthenaPublic : GitVcsRoot({
    name = "athena_public"
    url = "https://github.com/athenaml/athena"
    branch = "refs/heads/develop"
    branchSpec = """
        +:refs/heads/master
        +:refs/pull/*
        +:refs/pull/*/merge
    """.trimIndent()
    authMethod = password {
        userName = "athenamlbot"
        password = "credentialsJSON:c3f84f17-e571-495d-a804-ea7a25b20d89"
    }
})

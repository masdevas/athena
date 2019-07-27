package athena.settings

import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

object AthenaAlex : GitVcsRoot({
    name = "athena_alex"
    url = "https://github.com/alexbatashev/athena"
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

object AthenaAndrey : GitVcsRoot({
    name = "athena_andrey"
    url = "https://github.com/masdevas/athena"
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

object HttpsGithubComAthenamlAthenamlGithubIoRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/athenaml.github.io#refs/heads/master"
    url = "https://github.com/athenaml/athenaml.github.io"
    authMethod = password {
        userName = "athenamlbot"
        password = "credentialsJSON:c3f84f17-e571-495d-a804-ea7a25b20d89"
    }
})

object HttpsGithubComAthenamlWebsiteRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/website#refs/heads/master"
    url = "https://github.com/athenaml/website"
    authMethod = password {
        userName = "athenamlbot"
        password = "credentialsJSON:c3f84f17-e571-495d-a804-ea7a25b20d89"
    }
})

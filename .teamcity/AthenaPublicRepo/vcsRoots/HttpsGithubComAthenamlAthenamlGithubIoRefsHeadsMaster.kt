package AthenaPublicRepo.vcsRoots

import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

object HttpsGithubComAthenamlAthenamlGithubIoRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/athenaml.github.io#refs/heads/master"
    url = "https://github.com/athenaml/athenaml.github.io"
    authMethod = password {
        userName = "athenamlbot"
        password = "credentialsJSON:c3f84f17-e571-495d-a804-ea7a25b20d89"
    }
})

package patches.vcsRoots

import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.ui.*
import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

/*
This patch script was generated by TeamCity on settings change in UI.
To apply the patch, change the vcsRoot with id = 'AthenaAndrey'
accordingly, and delete the patch script.
*/
changeVcsRoot(RelativeId("AthenaAndrey")) {
    val expected = GitVcsRoot({
        id("AthenaAndrey")
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
            password = "zxx24d3eb84f48caad36dc56a3252875b70b7fd09cd46cb8ab54153e54fc1042983f410212252a6a0cf775d03cbe80d301b"
        }
    })

    check(this == expected) {
        "Unexpected VCS root settings"
    }

    (this as GitVcsRoot).apply {
        authMethod = password {
            userName = "athenamlbot"
            password = "credentialsJSON:c3f84f17-e571-495d-a804-ea7a25b20d89"
        }
    }

}

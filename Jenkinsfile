pipeline {
    agent {
        docker {
          image 'tnitn_ci2:py312'
          args '-v /mnt/jenkins/jenkinsci:/home/jenkinsci -v $HOME/.cache:/root/.cache --shm-size=4g --entrypoint=""'
        }
  }
    stages {
        stage('Print hello') {
            steps {
                echo 'Hello world!'
            }
        }
    }
}
pipeline {
  agent {
        docker {
          image 'nvcr.io/nvidia/pytorch:22.12-py3'
          args '--user 0:128 -v /home/jenkinsci:/home/jenkinsci -v $HOME/.cache:/root/.cache --shm-size=3g --entrypoint=""'
        }
  }
  options {
    timeout(time: 2, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: true)
  }

  stages {

    stage('Add git safe directory'){
      steps{
//         sh 'git config --global user.name "jenkinsci"'
//         sh 'git config --global user.email "$(whoami)@$(hostname)"'
        sh 'git config --global --add safe.directory /var/lib/jenkins/workspace/NTP_$GIT_BRANCH'
        sh 'git config --global --add safe.directory /home/jenkinsci/workspace/NTP_$GIT_BRANCH'
      }
    }

    stage('PyTorch version') {
      steps {
        sh 'python -c "import torch; print(torch.__version__)"'
        sh 'python -c "import torchvision; print(torchvision.__version__)"'
      }
    }

    stage('Install test requirements') {
      steps {
        sh 'apt-get update && apt-get install -y bc'
      }
    }



    stage('NeMo Installation') {
      steps {
        sh './reinstall.sh release'
      }
    }



    stage('L0: TN/ITN Tests CPU') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('En TN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --text="1" --cache_dir /home/jenkinsci/nlp/text_norm/ci/grammars/01-30-23'
          }
        }
        stage('En ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --language en --text="twenty" --cache_dir /home/jenkinsci/nlp/text_norm/ci/grammars/01-30-23'
          }
        }
        stage('Test En non-deterministic TN & Run all En TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/en/ -m "not pleasefixme" --cpu --tn_cache_dir /home/jenkinsci/nlp/text_norm/ci/grammars/01-30-23'
          }
        }

      }
    }

    stage('L2: NeMo text processing') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Eng TN') {
          steps {
            sh 'cd tools/text_processing_deployment && python pynini_export.py --output=/home/jenkinsci/nlp/text_norm/output/ --grammars=tn_grammars --cache_dir /home/jenkinsci/nlp/text_norm/ci/grammars/01-30-23 --language=en && ls -R /home/jenkinsci/nlp/text_norm/output/ && echo ".far files created "|| exit 1'
            sh 'cd NTP_text_processing/text_normalization/ &&  python normalize.py --input_file=/home/jenkinsci/nlp/text_norm/ci/test.txt --input_case="lower_cased" --language=en --output_file=/home/jenkinsci/nlp/text_norm/output/test.pynini.txt --verbose'
            sh 'cat /home/jenkinsci/nlp/text_norm/output/test.pynini.txt'
            sh 'cmp --silent /home/jenkinsci/nlp/text_norm/output/test.pynini.txt /home/jenkinsci/nlp/text_norm/ci/test_goal_py_05-25.txt || exit 1'
            sh 'rm -rf /home/jenkinsci/nlp/text_norm/output/*'
          }
        }

        stage('L2: Eng ITN export') {
          steps {
            sh 'cd tools/text_processing_deployment && python pynini_export.py --output=/home/jenkinsci/nlp/text_denorm/output/ --grammars=itn_grammars --cache_dir /home/jenkinsci/nlp/text_norm/ci/grammars/01-30-23 --language=en && ls -R /home/jenkinsci/nlp/text_denorm/output/ && echo ".far files created "|| exit 1'
            sh 'cd NTP_text_processing/inverse_text_normalization/ &&  python inverse_normalize.py --input_file=/home/jenkinsci/nlp/text_denorm/ci/test.txt --language=en --output_file=/home/jenkinsci/nlp/text_denorm/output/test.pynini.txt --verbose'
            sh 'cmp --silent /home/jenkinsci/nlp/text_denorm/output/test.pynini.txt /home/jenkinsci/nlp/text_denorm/ci/test_goal_py.txt || exit 1'
            sh 'rm -rf /home/jenkinsci/nlp/text_denorm/output/*'
          }
        }

      }
    }
  }


  post {
    always {
      sh 'chmod -R 777 .'
      cleanWs()
    }
  }
}

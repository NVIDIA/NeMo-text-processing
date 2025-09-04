pipeline {
  agent {
        docker {
          image 'tnitn_ci:py312'
          args '--user 0:128 -v /home/jenkinsci:/home/jenkinsci -v $HOME/.cache:/root/.cache --shm-size=8g --entrypoint=""'
        }
  }
  options {
    timeout(time: 4, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: true)
  }
  environment {

    AR_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/04-24-24-0'
    DE_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/10-23-24-0'
    EN_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/09-04-24-0'
    ES_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/09-25-24-0'
    ES_EN_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/08-30-24-0'
    FR_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/04-07-25-0'
    HU_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/07-16-24-0'
    PT_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    RU_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    VI_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    SV_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    ZH_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/11-13-24-0'
    IT_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/08-22-24-0'
    HY_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/03-12-24-0'
    MR_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/03-12-24-1'
    JA_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/10-17-24-1'
    HI_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/04-22-25-0'
    DEFAULT_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
  }
  stages {

    stage('Add git safe directory'){
      steps{
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


    stage('L0: Create EN TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: Test utils') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/audio_based_utils/ --cpu'
          }
        }
        stage('L0: En TN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --text="1" --cache_dir ${EN_TN_CACHE}'
          }
        }
        stage('L0: En TN non-deterministic grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize_with_audio.py --text="1" --cache_dir ${EN_TN_CACHE}'
          }
        }
        stage('L0: En ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --language en --text="twenty" --cache_dir ${EN_TN_CACHE}'
          }
        }

      }
    }



// L1 Tests starts here

    stage('L1: TN/ITN Tests CPU') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      stages {
        stage('L1: Test EN non-deterministic TN & Run all En TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/en/ -m "not pleasefixme" --cpu --tn_cache_dir ${EN_TN_CACHE}'
          }
        }
      }
    }

    stage('L2: Sparrowhawk Tests') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: EN ITN Run Sparrowhawk test - Lower Cased Input') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES=""  cp -r /workspace/sparrowhawk/documentation/grammars /workspace/sparrowhawk/documentation/grammars_en_itn_grammars_lower_cased && cd tools/text_processing_deployment && bash sh_test.sh --MODE="test_itn_grammars" --OVERWRITE_CACHE=False --FAR_PATH=${EN_TN_CACHE}/SH_ITN --LANGUAGE="en"'
            sh 'CUDA_VISIBLE_DEVICES="" cd tests/nemo_text_processing/en && bash test_sparrowhawk_inverse_text_normalization.sh /workspace/sparrowhawk/documentation/grammars_en_itn_grammars_lower_cased `pwd`'

          }
        }
        stage('L2: EN ITN Run Sparrowhawk test - Cased Input') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES=""  cp -r /workspace/sparrowhawk/documentation/grammars /workspace/sparrowhawk/documentation/grammars_en_itn_grammars_cased && cd tools/text_processing_deployment && bash sh_test.sh --MODE="test_itn_grammars" --INPUT_CASE="cased" --OVERWRITE_CACHE=False --FAR_PATH=${EN_TN_CACHE}/SH_ITN_cased --LANGUAGE="en"'
            sh 'CUDA_VISIBLE_DEVICES="" cd tests/nemo_text_processing/en && bash test_sparrowhawk_inverse_text_normalization_cased.sh /workspace/sparrowhawk/documentation/grammars_en_itn_grammars_cased `pwd`'

          }
        }
        stage('L2: EN TN Run Sparrowhawk test') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES=""  cp -r /workspace/sparrowhawk/documentation/grammars /workspace/sparrowhawk/documentation/grammars_en_tn_grammars_cased && cd tools/text_processing_deployment && bash sh_test.sh --MODE="test_tn_grammars" --INPUT_CASE="cased" --OVERWRITE_CACHE=False --FAR_PATH=${EN_TN_CACHE}/SH_TN --GRAMMARS="tn_grammars" --LANGUAGE="en" '
            sh 'CUDA_VISIBLE_DEVICES="" cd tests/nemo_text_processing/en && bash test_sparrowhawk_normalization.sh /workspace/sparrowhawk/documentation/grammars_en_tn_grammars_cased `pwd`'
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
            sh 'TIME=`date +"%Y-%m-%d-%T"` && NORM_OUTPUT_DIR=/home/jenkinsci/TestData/text_norm/output_${TIME} && \
            cd tools/text_processing_deployment && python pynini_export.py --output=$NORM_OUTPUT_DIR --grammars=tn_grammars --cache_dir ${EN_TN_CACHE} --language=en && ls -R $NORM_OUTPUT_DIR && echo ".far files created "|| exit 1'
            sh 'TIME=`date +"%Y-%m-%d-%T"` && NORM_OUTPUT_DIR=/home/jenkinsci/TestData/text_norm/output_${TIME} && mkdir $NORM_OUTPUT_DIR && \
            cd nemo_text_processing/text_normalization/ &&  python normalize.py --input_file=/home/jenkinsci/TestData/text_norm/ci/test.txt --input_case="lower_cased" --language=en --output_file=$NORM_OUTPUT_DIR/test.pynini.txt --verbose && \
            cat $NORM_OUTPUT_DIR/test.pynini.txt && \
            cmp --silent $NORM_OUTPUT_DIR/test.pynini.txt /home/jenkinsci/TestData/text_norm/ci/test_goal_py.txt || exit 1 && \
            rm -rf $NORM_OUTPUT_DIR'
          }
        }

        stage('L2: Eng ITN export') {
          steps {
            sh 'TIME=`date +"%Y-%m-%d-%T"` && DENORM_OUTPUT_DIR=/home/jenkinsci/TestData/text_denorm/output_${TIME} && \
            cd tools/text_processing_deployment && python pynini_export.py --output=$DENORM_OUTPUT_DIR --grammars=itn_grammars --cache_dir ${EN_TN_CACHE} --language=en && ls -R $DENORM_OUTPUT_DIR && echo ".far files created "|| exit 1'
            sh 'TIME=`date +"%Y-%m-%d-%T"` && DENORM_OUTPUT_DIR=/home/jenkinsci/TestData/text_denorm/output_${TIME} && mkdir $DENORM_OUTPUT_DIR && \
            cd nemo_text_processing/inverse_text_normalization/ &&  python inverse_normalize.py --input_file=/home/jenkinsci/TestData/text_denorm/ci/test.txt --language=en --output_file=$DENORM_OUTPUT_DIR/test.pynini.txt --verbose && \
            cmp --silent $DENORM_OUTPUT_DIR/test.pynini.txt /home/jenkinsci/TestData/text_denorm/ci/test_goal_py.txt || exit 1 && \
            rm -rf $DENORM_OUTPUT_DIR'
          }
        }


        stage('L2: Eng alignment TN') {
          steps {
            sh 'TIME=`date +"%Y-%m-%d-%T"` && NORM_OUTPUT_DIR=/home/jenkinsci/TestData/text_norm/output_${TIME} && mkdir $NORM_OUTPUT_DIR && \
            cd nemo_text_processing/fst_alignment && python alignment.py --text="2615 Forest Av, 90501 CA, Santa Clara. 10kg, 12/16/2018" --grammar=tn --rule=tokenize_and_classify --fst=${EN_TN_CACHE}/en_tn_True_deterministic_cased__tokenize.far 2>&1 | tee $NORM_OUTPUT_DIR/pred.txt && \
            cmp --silent $NORM_OUTPUT_DIR/pred.txt /home/jenkinsci/TestData/text_norm/ci/alignment_gold.txt || exit 1 && \
            rm -rf $NORM_OUTPUT_DIR'
          }
        }

        stage('L2: Eng alignment ITN') {
          steps {
            sh 'TIME=`date +"%Y-%m-%d-%T"` && DENORM_OUTPUT_DIR=/home/jenkinsci/TestData/text_denorm/output_${TIME} && mkdir $DENORM_OUTPUT_DIR && \
            cd nemo_text_processing/fst_alignment && python alignment.py --text="one million twenty three thousand two hundred eleven ten kilograms one hundred twenty three dollars and twenty five cents" --grammar=itn --rule=tokenize_and_classify --fst=${EN_TN_CACHE}/en_itn_lower_cased.far 2>&1 | tee $DENORM_OUTPUT_DIR/pred.txt && \
            cmp --silent $DENORM_OUTPUT_DIR/pred.txt /home/jenkinsci/TestData/text_denorm/ci/alignment_gold.txt || exit 1 && \
            rm -rf $DENORM_OUTPUT_DIR'
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

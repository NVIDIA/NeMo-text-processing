pipeline {
  agent {
        docker {
          image 'tnitn_ci'
          args '--user 0:128 -v /home/jenkinsci:/home/jenkinsci -v $HOME/.cache:/root/.cache --shm-size=4g --entrypoint=""'
        }
  }
  options {
    timeout(time: 2, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: true)
  }
  environment {

    AR_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    DE_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    EN_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-14-23-0'
    ES_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/08-29-23-0'
    ES_EN_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-13-23-1'
    FR_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/08-16-23-1'
    HU_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    PT_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    RU_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    VI_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    SV_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/06-08-23-0'
    ZH_TN_CACHE='/home/jenkinsci/TestData/text_norm/ci/grammars/07-27-23-0'
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

    stage('L0: Create DE TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: DE TN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=de --text="1" --cache_dir ${DEFAULT_TN_CACHE}'
          }
        }
        stage('L0: DE ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=de --text="ein hundert " --cache_dir ${DEFAULT_TN_CACHE}'
          }
        }

      }
    }

    stage('L0: Create ES TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: ES TN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=es --text="1" --cache_dir ${ES_TN_CACHE}'
          }
        }
        stage('L0: ES ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=es --text="ciento uno " --cache_dir ${ES_TN_CACHE}'
          }
        }

      }
    }

    stage('L0: Create Codeswitched ES/EN TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {

        stage('L0: ES/EN ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=es_en --text="ciento uno " --cache_dir ${ES_EN_TN_CACHE}'
          }
        }

      }
    }

    stage('L0: Create AR TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: AR TN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=ar --text="2" --cache_dir ${AR_TN_CACHE}'
          }
        }
        stage('L0: AR ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=ar --text="اثنان " --cache_dir ${AR_TN_CACHE}'
          }
        }

      }
    }

    stage('L0: Create FR TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
       // stage('L0: FR TN grammars') {
       //  steps {
       //     sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=fr --text="2" --cache_dir ${FR_TN_CACHE}'
       //   }
       // }
        stage('L0: FR ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=fr --text="cent " --cache_dir ${FR_TN_CACHE}'
          }
        }

      }
    }
    stage('L0: Create HU TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: HU TN grammars') {
         steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=hu --text="100" --cache_dir ${HU_TN_CACHE}'
          }
        }
       // stage('L0: HU ITN grammars') {
       //   steps {
       //     sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=hu --text="száz " --cache_dir ${HU_TN_CACHE}'
       //   }
       // }
      }
    }
    stage('L0: Create VI TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
       // stage('L0: VI TN grammars') {
       //  steps {
       //     sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=vi --text="2" --cache_dir ${VI_TN_CACHE}'
       //   }
       // }
        stage('L0: VI ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=vi --text="một ngàn " --cache_dir ${VI_TN_CACHE}'
          }
        }

      }
    }

    stage('L0: Create PT TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
       // stage('L0: PT TN grammars') {
       //  steps {
       //     sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=pt --text="2" --cache_dir ${DEFAULT_TN_CACHE}'
       //   }
       // }
        stage('L0: PT ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=pt --text="dez " --cache_dir ${PT_TN_CACHE}'
          }
        }

      }
    }
    stage('L0: Create RU TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: RU TN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize_with_audio.py --lang=ru --text="03" --cache_dir ${RU_TN_CACHE}'
          }
        }
        stage('L0: RU ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=ru --text="три " --cache_dir ${RU_TN_CACHE}'
          }
        }
      }
    }
    stage('L0: Create SV TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: SV TN grammars') {
         steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=sv --text="100" --cache_dir ${SV_TN_CACHE}'
          }
        }
      //  stage('L0: SV ITN grammars') {
      //    steps {
      //      sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=sv --text="hundra " --cache_dir ${SV_TN_CACHE}'
      //    }
      //  }
      }
    }
    stage('L0: Create ZH TN/ITN Grammars') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L0: ZH TN grammars') {
         steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py --lang=zh --text="你" --cache_dir ${ZH_TN_CACHE}'
          }
        }
        stage('L0: ZH ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --lang=zh --text="二零零二年一月二十八日 " --cache_dir ${ZH_TN_CACHE}'
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
        stage('L1: Run all DE TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/de/ -m "not pleasefixme" --cpu --tn_cache_dir ${DE_TN_CACHE}'
          }
        }
        stage('L1: Run all ES TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/es/ -m "not pleasefixme" --cpu --tn_cache_dir ${ES_TN_CACHE}'
          }
        }
        stage('L1: Run all Codeswitched ES/EN TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/es_en/ -m "not pleasefixme" --cpu --tn_cache_dir ${ES_EN_TN_CACHE}'
          }
        }
        stage('L1: Run all AR TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/ar/ -m "not pleasefixme" --cpu --tn_cache_dir ${AR_TN_CACHE}'
          }
        }
        stage('L1: Run all FR TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/fr/ -m "not pleasefixme" --cpu --tn_cache_dir ${FR_TN_CACHE}'
          }
        }
        stage('L1: Run all PT TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/pt/ -m "not pleasefixme" --cpu --tn_cache_dir ${PT_TN_CACHE}'
          }
        }
        stage('L1: Run all VI TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/vi/ -m "not pleasefixme" --cpu --tn_cache_dir ${VI_TN_CACHE}'
          }
        }
        stage('L1: Run all RU TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/ru/ -m "not pleasefixme" --cpu --tn_cache_dir ${RU_TN_CACHE}'
          }
        }
        // stage('L1: Run all SV TN/ITN tests (restore grammars from cache)') {
        //   steps {
        //     sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/sv/ -m "not pleasefixme" --cpu --tn_cache_dir ${SV_TN_CACHE}'
        //   }
        // }
        stage('L1: Run all ZH TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/zh/ -m "not pleasefixme" --cpu --tn_cache_dir ${ZH_TN_CACHE}'
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

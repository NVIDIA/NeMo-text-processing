# What does this PR do ?

Add a one line overview of what this PR aims to accomplish.


# Before your PR is "Ready for review"
**Pre checks**:
- [ ] Have you signed your commits? Use ``git commit -s`` to sign.
- [ ] Do all unittests finish successfully before sending PR?
   1) ``pytest`` or (if your machine does not have GPU) ``pytest --cpu`` from the root folder (given you marked your test cases accordingly `@pytest.mark.run_only_on('CPU')`).
   2) Sparrowhawk tests ``bash tools/text_processing_deployment/export_grammars.sh --MODE=test ...``
- [ ] If you are adding a new feature: Have you added test cases for both `pytest` and Sparrowhawk [here](tests/nemo_text_processing).
- [ ] Have you added ``__init__.py`` for every folder and subfolder, including `data` folder which has .TSV files?
- [ ] Have you followed codeQL results and removed unused variables and imports (report is at the bottom of the PR in github review box) ?
- [ ] Have you added the correct license header `Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.` to all newly added Python files?
- [ ] If you copied [nemo_text_processing/text_normalization/en/graph_utils.py](nemo_text_processing/text_normalization/en/graph_utils.py) your header's second line should be `Copyright 2015 and onwards Google, Inc.`. See an example [here](https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/text_normalization/en/graph_utils.py#L2).
- [ ] Remove import guards (`try import: ... except: ...`) if not already done.
- [ ] If you added a new language or a new feature please update the [NeMo documentation](https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization/wfst/wfst_text_normalization.rst) (lives in different repo).
- [ ] Have you added your language support to [tools/text_processing_deployment/pynini_export.py](tools/text_processing_deployment/pynini_export.py).
  


**PR Type**:
- [ ] New Feature
- [ ] Bugfix
- [ ] Documentation
- [ ] Test

If you haven't finished some of the above items you can still open "Draft" PR.

# Contributions are welcome!

We do all of NeMo-text-processing's development in the open. Contributions from the open-source community are welcome.


# Pull Requests (PR) Guidelines

**Send your PRs to the `main` branch**

1) Make sure your PR does one thing. Have a clear answer to "What does this PR do?".
2) Make sure you sign your commits. E.g. use ``git commit -s`` when you commit.
3) Make sure to add test cases for both `pytest` and Sparrowhawk [here](tests/nemo_text_processing).
4) Make sure all unittests finish successfully before sending PR:
   1) ``pytest`` or (if your machine does not have GPU) ``pytest --cpu`` from the root folder (given you marked your test cases accordingly `@pytest.mark.run_only_on('CPU')`).
   2) Sparrowhawk tests ``bash tools/text_processing_deployment/export_grammars.sh --MODE=test ...``
5) If you are adding a **new** Python file with a license header, the first line needs to be `Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.` (change `2023` to the current year).
6) If your `text_normalization/LANG/graph_utils.py` is mainly copied from [nemo_text_processing/text_normalization/en/graph_utils.py](nemo_text_processing/text_normalization/en/graph_utils.py) your header's second line should be `Copyright 2015 and onwards Google, Inc.`. See an example [here](https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/text_normalization/en/graph_utils.py#L2).
7) Add ``__init__.py`` for every folder and subfolder.
8) Remove import guards (`try import: ... except: ...`) if not already done.
9) follow codeQL results and remove unused variables and imports (report is at the bottom of the PR in github review box)
10) Add your language support to [tools/text_processing_deployment/pynini_export.py](tools/text_processing_deployment/pynini_export.py).
11) Optional: if you added a new language or a new feature please update the [NeMo documentation](https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization/wfst/wfst_text_normalization.rst) (lives in different repo).
12) Send your PR and request a review



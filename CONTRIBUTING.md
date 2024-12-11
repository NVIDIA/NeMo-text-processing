# Contributions are welcome!

We do all of NeMo-Text-Processing's development in the open. Contributions from the open-source community are welcome.


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

# Notes for Language Contribution
1) `en/graph_utils.py` and `en/utils.py` are the de facto parents for all other `graph_util` and `utils` functions, respectively. Please refrain from duplicating code logics into new `graph_utils.py` and `utils.py` files and default to imports from `en/graph_utils.py` and `en/utils.py` instead. `LANG/graph_utils.py` and `LANG/utils.py` files should only contain new methods and variables. Not all new languages will require a submodule specific `graph_utils.py` or `utils.py` file.

2) NeMo-Text-Processing allows creation of FST graphs through two backends: the Python based library itself (via [Pynini](https://www.opengrm.org/twiki/bin/view/GRM/Pynini) backend) and C++ based [Sparrowhawk](https://github.com/google/sparrowhawk/tree/master) in an upstream repo. Due to the typical tradeoffs between these Python and C++ development [languages](https://www.youtube.com/watch?v=VioxsWYzoJk), the NeMo-Text-Processing library assumes development to be performed with the Python library for final deployment in Sparrowhawk/C++. This dual framework approach can lead to issues in development, notably in the case of tagging additional properties during tokenization.

When writing taggers for semiotic classes, you may need to tag additional token properties (e.g. grammatical gender, case) for accurate verbalization. For example, the Spanish ordinal `21.º` carries masculine gender and is verbalized with a specific spelling. As such, it would be desired for the TN tagger to tokenize the string with the gender property included. 

Naively, one may be tempted to simply include the property string `gender: "masc"` and check for this string during the verbalization phase. **This is not advised.** While the NeMo-Text-Processing library itself will permit any custom string in the tagger, Sparrowhawk limits permissible strings, and will fail with custom property strings. Given the performance loss in not providing Sparrowhawk support, we cannot integrate new graphs that cause Sparrowhawk failure. As such, tagged properties should be limited to Sparrowhawk supported strings. 

For all classes, Sparrowhawk supports the `morphosyntactic_features` property, and it is recommended to default to this property for tagging additional features. For example:

`21.º" -> ordinal { integer: "vigésimo primero" morphosyntactic_features: "masc" }`

For additional Sparrowhawk supported properties by class, see [here](https://github.com/yzhang123/sparrowhawk/blob/test/src/proto/semiotic_classes.proto)

N.B. The same limitation applies for novel semiotic classes as well. Only predefined classes are supported in Sparrowhawk.

3) Between the tagging and verbalizing stages, both the NeMo-Text-Processing and Sparrowhawk engines permute order of tagged properties. That is, assuming the tagger parsed `1ᵉʳ juillet` as:

`date { month: "juillet" day: "1" } }`

the verbalizer will receive as input both

`date { month: "juillet" day: "1" }`

and

`date { day: "1" month: "juillet" }`

While this eases construction of verbalization graphs, permutation can be computationally expensive. If you know that the tagger output will not require permutation of token properties, you can improve model performance by including the `preserve_order: "true"` property:

`date { day: "1" month: "juillet" preserve_order: "true" }`

import pynini
import pytest
from parameterized import parameterized

from nemo_text_processing.inverse_text_normalization.te.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.verbalize_final import VerbalizeFinalFst

from ..utils import parse_test_case_file


class TestCardinal:
    tagger = ClassifyFst(overwrite_cache=True).fst
    verbalizer = VerbalizeFinalFst().fst

    @parameterized.expand(parse_test_case_file('te/data_inverse_text_normalization/test_cases_cardinal.txt'))
    #@pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        tagged = pynini.shortestpath(test_input @ self.tagger).string()
        pred = pynini.shortestpath(tagged @ self.verbalizer).string()
        assert pred == expected
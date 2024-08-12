import pytest
from parameterized import parameterized

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

from ..utils import CACHE_DIR, parse_test_case_file


class TestFullSentences:
    inverse_normalizer_en = InverseNormalizer(lang='he', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('he/data_inverse_text_normalization/test_full_sentences.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer_en.inverse_normalize(test_input, verbose=False)
        assert pred == expected

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
from parameterized import parameterized

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

from ..utils import CACHE_DIR, parse_test_case_file


class TestElectronic:
    """
    ITN Electronic test suite — Hindi.

    Covers: email · https/http/www URL · plain domain · Windows path ·
            Unix path · IP address · alphanumeric codes ·
            chemical formulas (named + subscript) ·
            subdomain patterns (srv- db- lt- web- laptop- desktop- email-)

    Test cases: hi/data_inverse_text_normalization/test_cases_electronic.txt
    Format per line: spoken_hindi~expected_written_form
    """

    inverse_normalizer = InverseNormalizer(lang='hi', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(
        parse_test_case_file('hi/data_inverse_text_normalization/test_cases_electronic.txt')
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected, (
            f"\nInput:    {test_input}"
            f"\nExpected: {expected}"
            f"\nGot:      {pred}"
        )

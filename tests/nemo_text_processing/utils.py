# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import List, Tuple

CACHE_DIR = None
RUN_AUDIO_BASED_TESTS = False


def set_cache_dir(path: str = None):
    """
    Sets cache directory for TN/ITN unittests. Default is None, e.g. no cache during tests.
    """
    global CACHE_DIR
    CACHE_DIR = path


def set_audio_based_tests(run_audio_based: bool = False):
    """
    Sets audio-based test mode for TN/ITN unittests. Default is False, e.g. audio-based tests will be skipped.
    """
    global RUN_AUDIO_BASED_TESTS
    RUN_AUDIO_BASED_TESTS = run_audio_based


def parse_test_case_file(file_name: str):
    """
    Prepares tests pairs for ITN and TN tests
    """
    test_pairs = []
    with open(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + file_name, 'r') as f:
        for line in f:
            components = line.strip("\n").split("~")
            spoken = components[0]

            """
            Some transformations can have multiple correct forms. Instead of
            asserting against a single expected value, we assert that the
            output matches any of the correct forms.

                Example:    200 can be "doscientos" or "doscientas" in Spanish
                Test data:  200~doscientos~doscientas
                Evaluation: ASSERT "doscientos" in ["doscientos", "doscientas"]
            """
            written = components[1] if len(components) == 2 else components[1:]
            test_pairs.append((spoken, written))
    return test_pairs


def get_test_cases_multiple(file_name: str = 'data_text_normalization/en/test_cases_normalize_with_audio.txt'):
    """
    Prepares tests pairs for audio based TN tests
    """
    test_pairs = []
    with open(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + file_name, 'r') as f:
        written = None
        normalized_options = []
        for line in f:
            if line.startswith('~'):
                if written:
                    test_pairs.append((written, normalized_options))
                    normalized_options = []
                written = line.strip().replace('~', '')
            else:
                normalized_options.append(line.strip())
    test_pairs.append((written, normalized_options))
    return test_pairs


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for comparison."""
    return re.sub(r'\s+', ' ', text.strip())


def extract_projected_content(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract all projected content from a string with paired brackets format.
    
    Args:
        text: Text containing paired projections like "text [normalized][original] more text"
    
    Returns:
        List of (prefix_text, normalized, original) tuples where prefix_text is the text before this projection
    
    Examples:
        "i can use your [card ending in eight eight seven six][card ending in 8876]" 
            -> [("i can use your ", "card ending in eight eight seven six", "card ending in 8876")]
    """
    import re

    # Pattern to match paired brackets with NO space between them
    pattern = r'\[([^\]]+)\]\[([^\]]+)\]'

    result = []
    last_end = 0

    for match in re.finditer(pattern, text):
        # Get the text before this projection pair (don't strip - preserve spacing)
        prefix = text[last_end : match.start()]
        normalized = match.group(1)
        original = match.group(2)
        result.append((prefix, normalized, original))
        last_end = match.end()

    # If there's text after the last projection, we need to handle it
    remaining = text[last_end:]
    if remaining:
        result.append((remaining, "", ""))

    return result


def reconstruct_sentences(text: str) -> Tuple[str, str]:
    """
    Reconstruct the normalized output and original input from projected text.
    
    Args:
        text: Text with paired projections like "text [normalized][original] more text"
        
    Returns:
        Tuple of (normalized_sentence, original_sentence)
    """
    projections = extract_projected_content(text)

    if not projections:
        # No projections found, return the text as-is for both
        return text, text

    normalized_parts = []
    original_parts = []

    for prefix, normalized, original in projections:
        if normalized and original:
            # This prefix has a projection following it
            normalized_parts.append(prefix + normalized)
            original_parts.append(prefix + original)
        else:
            # This is trailing text with no projection
            normalized_parts.append(prefix)
            original_parts.append(prefix)

    normalized_sentence = "".join(normalized_parts)
    original_sentence = "".join(original_parts)

    return normalize_whitespace(normalized_sentence), normalize_whitespace(original_sentence)

def remove_spaces_around_punctuation(text):
    """Remove spaces that are adjacent to punctuation characters."""
    punctuation_chars = r'[,!?.;:(){}[\]<>/@#$%^&*+=|\\`~"\'-]'
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+(?=' + punctuation_chars + ')', '', text)
    
    # Remove spaces after punctuation
    text = re.sub(r'(?<=' + punctuation_chars + r')\s+', '', text)
    
    return text

def assert_projecting_output(predicted: str, expected_output: str, original_input: str) -> None:
    """
    Enhanced assertion for projecting tests with paired bracket format.
    
    Args:
        predicted: The actual output from the normalizer (with [normalized][original] pairs)
        expected_output: The expected normalized output
        original_input: The original input that should be reconstructed
        
    Raises:
        AssertionError: If the projecting output doesn't match expectations
    """
    import re
    
    predicted = normalize_whitespace(predicted)
    expected_output = normalize_whitespace(expected_output)
    original_input = normalize_whitespace(original_input)

    # Case 1: No normalization occurred, no projection needed
    if expected_output == original_input:
        assert predicted == expected_output, (
            f"No normalization expected but got different output.\n"
            f"Expected: '{expected_output}'\n"
            f"Got:      '{predicted}'"
        )
        return

    # Case 2: Extract and reconstruct both sentences
    reconstructed_normalized, reconstructed_original = reconstruct_sentences(predicted)

    # Verify the normalized version matches expected output
    assert reconstructed_normalized == expected_output, (
        f"Reconstructed normalized output doesn't match expected.\n"
        f"Expected:      '{expected_output}'\n"
        f"Reconstructed: '{reconstructed_normalized}'\n"
        f"Full output:   '{predicted}'"
    )

    # Check if original input contains punctuation
    # Define punctuation characters that trigger space-agnostic comparison
    punctuation_chars = r'[,!?.;:(){}[\]<>/@#$%^&*+=|\\`~"\'-]'
    
    if re.search(punctuation_chars, original_input):
        # Remove all spaces for comparison when punctuation is present
        original_input_no_spaces = re.sub(r'\s+', '', original_input)
        original_input_no_spaces = re.sub(r'``', '"', original_input_no_spaces)
        original_input_no_spaces = re.sub(r'-', 'to', original_input_no_spaces)
        reconstructed_original_no_spaces = re.sub(r'\s+', '', reconstructed_original)
        reconstructed_original_no_spaces = re.sub(r'``', '"', reconstructed_original_no_spaces)
        reconstructed_original_no_spaces = re.sub(r'-', 'to', reconstructed_original_no_spaces)

        assert reconstructed_original_no_spaces == original_input_no_spaces, (
            f"Reconstructed original input doesn't match actual input (ignoring spaces).\n"
            f"Expected:      '{original_input}' -> '{original_input_no_spaces}'\n"
            f"Reconstructed: '{reconstructed_original}' -> '{reconstructed_original_no_spaces}'\n"
            f"Full output:   '{predicted}'"
        )
    else:
        # No punctuation, use exact comparison
        assert reconstructed_original == original_input, (
            f"Reconstructed original input doesn't match actual input.\n"
            f"Expected:      '{original_input}'\n"
            f"Reconstructed: '{reconstructed_original}'\n"
            f"Full output:   '{predicted}'"
        )

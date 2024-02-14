# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Dict

from cdifflib import CSequenceMatcher

from nemo_text_processing.utils.logging import logger

MATCH = "match"
NONMATCH = "non-match"
SEMIOTIC_TAG = "[SEMIOTIC_SPAN]"


def _get_alignment(a: str, b: str) -> Dict:
    """
    Constructs alignment between a and b

    Returns:
        a dictionary, where keys are a's word index and values is a Tuple that contains span from b, and whether it
            matches a or not, e.g.:
                >>> a = "a b c"
                >>> b = "a b d f"
                >>> print(_get_alignment(a, b))
                {0: (0, 1, 'match'), 1: (1, 2, 'match'), 2: (2, 4, 'non-match')}
    """
    a = a.lower().split()
    b = b.lower().split()

    s = CSequenceMatcher(None, a, b, autojunk=False)
    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    s = s.get_matching_blocks()

    diffs = {}
    non_match_start_l = 0
    non_match_start_r = 0
    for match in s:
        l_start, r_start, length = match
        if non_match_start_l < l_start:
            while non_match_start_l < l_start:
                diffs[non_match_start_l] = (non_match_start_r, r_start, NONMATCH)
                non_match_start_l += 1

        for len_ in range(length):
            diffs[l_start + len_] = (r_start + len_, r_start + 1 + len_, MATCH)
        non_match_start_l = l_start + length
        non_match_start_r = r_start + length
    return diffs


def adjust_boundaries(norm_raw_diffs: Dict, norm_pred_diffs: Dict, raw: str, norm: str, pred_text: str, verbose=False):
    """
    Adjust alignment boundaries by taking norm--raw texts and norm--pred_text alignments, and creating raw-pred_text alignment
        alignment.

    norm_raw_diffs: output of _get_alignment(norm, raw)
    norm_pred_diffs: output of  _get_alignment(norm, pred_text)
    raw: input text
    norm: output of default normalization (deterministic)
    pred_text: ASR prediction
    verbose: set to True to output intermediate output of adjustments (for debugging)

    Return:
        semiotic_spans: List[str] - List of semiotic spans from raw text
        pred_texts: List[str] - List of pred_texts correponding to semiotic_spans
        norm_spans: List[str] - List of normalized texts correponding to semiotic_spans
        raw_text_masked_list: List[str] - List of words from raw text where every semiotic span is replaces with SEMIOTIC_TAG
        raw_text_mask_idx: List[int] - List of indexes of SEMIOTIC_TAG in raw_text_masked_list

        e.g.:
            >>> raw = 'This is #4 ranking on G.S.K.T.'
            >>> pred_text = 'this iss for ranking on g k p'
            >>> norm = 'This is nubmer four ranking on GSKT'

            output:
            semiotic_spans: ['is #4', 'G.S.K.T.']
            pred_texts: ['iss for', 'g k p']
            norm_spans: ['is nubmer four', 'GSKT']
            raw_text_masked_list: ['This', '[SEMIOTIC_SPAN]', 'ranking', 'on', '[SEMIOTIC_SPAN]']
            raw_text_mask_idx: [1, 4]
    """

    raw_pred_spans = []
    word_id = 0
    while word_id < len(norm.split()):
        norm_raw, norm_pred = norm_raw_diffs[word_id], norm_pred_diffs[word_id]
        # if there is a mismatch in norm_raw and norm_pred, expand the boundaries of the shortest mismatch to align with the longest one
        # e.g., norm_raw = (1, 2, 'match') norm_pred = (1, 5, 'non-match') => expand norm_raw until the next matching sequence or the end of string to align with norm_pred
        if (norm_raw[2] == MATCH and norm_pred[2] == NONMATCH) or (norm_raw[2] == NONMATCH and norm_pred[2] == MATCH):
            mismatched_id = word_id
            non_match_raw_start = norm_raw[0]
            non_match_pred_start = norm_pred[0]
            done = False
            word_id += 1
            while word_id < len(norm.split()) and not done:
                norm_raw, norm_pred = norm_raw_diffs[word_id], norm_pred_diffs[word_id]
                if norm_raw[2] == MATCH and norm_pred[2] == MATCH:
                    non_match_raw_end = norm_raw_diffs[word_id - 1][1]
                    non_match_pred_end = norm_pred_diffs[word_id - 1][1]
                    word_id -= 1
                    done = True
                else:
                    word_id += 1
            if not done:
                non_match_raw_end = len(raw.split())
                non_match_pred_end = len(pred_text.split())
            raw_pred_spans.append(
                (
                    mismatched_id,
                    (non_match_raw_start, non_match_raw_end, NONMATCH),
                    (non_match_pred_start, non_match_pred_end, NONMATCH),
                )
            )
        else:
            raw_pred_spans.append((word_id, norm_raw, norm_pred))
        word_id += 1

    # aggregate neighboring spans with the same status
    spans_merged_neighbors = []
    last_status = None
    for idx, item in enumerate(raw_pred_spans):
        if last_status is None:
            last_status = item[1][2]
            raw_start = item[1][0]
            pred_text_start = item[2][0]
            norm_span_start = item[0]
            raw_end = item[1][1]
            pred_text_end = item[2][1]
        elif last_status is not None and last_status == item[1][2]:
            raw_end = item[1][1]
            pred_text_end = item[2][1]
        else:
            spans_merged_neighbors.append(
                [[norm_span_start, item[0]], [raw_start, raw_end], [pred_text_start, pred_text_end], last_status]
            )
            last_status = item[1][2]
            raw_start = item[1][0]
            pred_text_start = item[2][0]
            norm_span_start = item[0]
            raw_end = item[1][1]
            pred_text_end = item[2][1]

    if last_status == item[1][2]:
        raw_end = item[1][1]
        pred_text_end = item[2][1]
        spans_merged_neighbors.append(
            [[norm_span_start, item[0]], [raw_start, raw_end], [pred_text_start, pred_text_end], last_status]
        )
    else:
        spans_merged_neighbors.append(
            [
                [raw_pred_spans[idx - 1][0], len(norm.split())],
                [item[1][0], len(raw.split())],
                [item[2][0], len(pred_text.split())],
                item[1][2],
            ]
        )

    raw_list = raw.split()
    pred_text_list = pred_text.split()
    norm_list = norm.split()

    # increase boundaries between raw and pred_text if some spans contain empty pred_text
    extended_spans = []
    raw_norm_spans_corrected_for_pred_text = []
    idx = 0
    while idx < len(spans_merged_neighbors):
        item = spans_merged_neighbors[idx]

        cur_semiotic = " ".join(raw_list[item[1][0] : item[1][1]])
        cur_pred_text = " ".join(pred_text_list[item[2][0] : item[2][1]])
        cur_norm_span = " ".join(norm_list[item[0][0] : item[0][1]])
        logger.debug(f"cur_semiotic: {cur_semiotic}")
        logger.debug(f"cur_pred_text: {cur_pred_text}")
        logger.debug(f"cur_norm_span: {cur_norm_span}")

        # if cur_pred_text is an empty string
        if item[2][0] == item[2][1]:
            # for the last item
            if idx == len(spans_merged_neighbors) - 1 and len(raw_norm_spans_corrected_for_pred_text) > 0:
                last_item = raw_norm_spans_corrected_for_pred_text[-1]
                last_item[0][1] = item[0][1]
                last_item[1][1] = item[1][1]
                last_item[2][1] = item[2][1]
                last_item[-1] = item[-1]
            else:
                raw_start, raw_end = item[0]
                norm_start, norm_end = item[1]
                pred_start, pred_end = item[2]
                while idx < len(spans_merged_neighbors) - 1 and not (
                    (pred_end - pred_start) > 2 and spans_merged_neighbors[idx][-1] == MATCH
                ):
                    idx += 1
                    raw_end = spans_merged_neighbors[idx][0][1]
                    norm_end = spans_merged_neighbors[idx][1][1]
                    pred_end = spans_merged_neighbors[idx][2][1]
                cur_item = [[raw_start, raw_end], [norm_start, norm_end], [pred_start, pred_end], NONMATCH]
                raw_norm_spans_corrected_for_pred_text.append(cur_item)
                extended_spans.append(len(raw_norm_spans_corrected_for_pred_text) - 1)
            idx += 1
        else:
            raw_norm_spans_corrected_for_pred_text.append(item)
            idx += 1

    semiotic_spans = []
    norm_spans = []
    pred_texts = []
    raw_text_masked = ""
    for idx, item in enumerate(raw_norm_spans_corrected_for_pred_text):
        cur_semiotic = " ".join(raw_list[item[1][0] : item[1][1]])
        cur_pred_text = " ".join(pred_text_list[item[2][0] : item[2][1]])
        cur_norm_span = " ".join(norm_list[item[0][0] : item[0][1]])

        if idx == len(raw_norm_spans_corrected_for_pred_text) - 1:
            cur_norm_span = " ".join(norm_list[item[0][0] : len(norm_list)])
        if (item[-1] == NONMATCH and cur_semiotic != cur_norm_span) or (idx in extended_spans):
            raw_text_masked += " " + SEMIOTIC_TAG
            semiotic_spans.append(cur_semiotic)
            pred_texts.append(cur_pred_text)
            norm_spans.append(cur_norm_span)
        else:
            raw_text_masked += " " + " ".join(raw_list[item[1][0] : item[1][1]])
    raw_text_masked_list = raw_text_masked.strip().split()

    raw_text_mask_idx = [idx for idx, x in enumerate(raw_text_masked_list) if x == SEMIOTIC_TAG]

    if verbose:
        print("+" * 50)
        print("raw_pred_spans:")
        for item in spans_merged_neighbors:
            print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")

        print("+" * 50)
        print("spans_merged_neighbors:")
        for item in spans_merged_neighbors:
            print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
        print("+" * 50)
        print("raw_norm_spans_corrected_for_pred_text:")
        for item in raw_norm_spans_corrected_for_pred_text:
            print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
        print("+" * 50)

    return semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx


def get_alignment(raw: str, norm: str, pred_text: str, verbose: bool = False):
    """
    Aligns raw text with deterministically normalized text and ASR output, finds semiotic spans
    """
    for value in [raw, norm, pred_text]:
        if value is None or value == "":
            return [], [], [], [], []

    norm_pred_diffs = _get_alignment(norm, pred_text)
    norm_raw_diffs = _get_alignment(norm, raw)

    semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx = adjust_boundaries(
        norm_raw_diffs, norm_pred_diffs, raw, norm, pred_text, verbose
    )

    if verbose:
        for i in range(len(semiotic_spans)):
            print("=" * 40)
            # print(i)
            print(f"semiotic : {semiotic_spans[i]}")
            print(f"pred text: {pred_texts[i]}")
            print(f"norm     : {norm_spans[i]}")
            print("=" * 40)

    return semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx


if __name__ == "__main__":
    raw = 'This is a #4 ranking on G.S.K.T.'
    pred_text = 'this iss p k for ranking on g k p'
    norm = 'This is nubmer four ranking on GSKT'

    output = get_alignment(raw, norm, pred_text, True)
    print(output)

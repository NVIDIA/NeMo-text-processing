from cdifflib import CSequenceMatcher

from nemo.utils import logging

logging.setLevel("INFO")

MATCH = "match"
NONMATCH = "non-match"
SEMIOTIC_TAG = "[SEMIOTIC_SPAN]"


def _get_alignment(a, b):
    """

	TODO: add

	Returns:
		list of Tuple(pred start and end, gt start and end) subsections
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


def get_semiotic_spans(a, b):
    """returns list of different substrings between and b

	Returns:
		list of Tuple(pred start and end, gt start and end) subsections
	"""

    b = b.lower().split()
    s = CSequenceMatcher(None, a.lower().split(), b, autojunk=False)

    # do it here to preserve casing
    a = a.split()

    result = []
    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    a_start_non_match, b_start_non_match = 0, 0
    # get not matching blocks
    for match in s.get_matching_blocks():
        a_start_match, b_start_match, match_len = match
        # we're widening the semiotic span to include 1 context word from each side, so not considering 1-word match here
        if match_len > 1:
            if a_start_non_match < a_start_match:
                result.append([[a_start_non_match, a_start_match], [b_start_non_match, b_start_match]])
            a_start_non_match = a_start_match + match_len
            b_start_non_match = b_start_match + match_len

    if a_start_non_match < len(a):
        result.append([[a_start_non_match, len(a)], [b_start_non_match, len(b)]])

    # add context (1 word from both sides)
    result_with_context = []
    for item in result:
        try:
            start_a, end_a = item[0]
            start_b, end_b = item[1]
        except:
            import pdb

            pdb.set_trace()
        if start_a - 1 >= 0 and start_b - 1 >= 0:
            start_a -= 1
            start_b -= 1
        if end_a + 1 <= len(a) and end_b + 1 <= len(b):
            end_a += 1
            end_b += 1
        result_with_context.append([[start_a, end_a], [start_b, end_b]])

    result = result_with_context
    semiotic_spans = []
    default_normalization = []
    a_masked = []
    start_idx = 0
    masked_idx = []
    for item in result:
        cur_start, cur_end = item[0]
        if start_idx < cur_start:
            a_masked.extend(a[start_idx:cur_start])
        start_idx = cur_end
        a_masked.append(SEMIOTIC_TAG)
        masked_idx.append(len(a_masked) - 1)

    if start_idx < len(a):
        a_masked.extend(a[start_idx:])

    for item in result:
        a_diff = ' '.join(a[item[0][0] : item[0][1]])
        b_diff = ' '.join(b[item[1][0] : item[1][1]])
        semiotic_spans.append(a_diff)
        default_normalization.append(b_diff)
        logging.debug(f"a: {a_diff}")
        logging.debug(f"b: {b_diff}")
        logging.debug("=" * 20)
    return result, semiotic_spans, a_masked, masked_idx, default_normalization


def _print_alignment(diffs, l, r):
    l = l.split()
    r = r.split()
    for l_idx, item in diffs.items():
        start, end, match_state = item
        print(f"{l[l_idx]} -- {r[start: end]} -- {match_state}")


def _adjust_span(semiotic_spans, norm_pred_diffs, pred_norm_diffs, norm_raw_diffs, raw: str, pred_text: str):
    """

	:param semiotic_spans:
	:param norm_pred_diffs:
	:param pred_norm_diffs:
	:param norm_raw_diffs:
	:param raw:
	:param pred_text:
	:return:
	"""
    standard_start = 0
    raw_text_list = raw.split()
    pred_text_list = pred_text.split()

    text_for_audio_based = {"semiotic": [], "standard": "", "pred_text": []}

    for idx, (raw_span, norm_span) in enumerate(semiotic_spans):
        raw_start, raw_end = raw_span
        raw_end -= 1

        # get the start of the span
        pred_text_start, _, status_norm_pred = norm_pred_diffs[norm_span[0]]
        new_norm_start = pred_norm_diffs[pred_text_start][0]
        raw_text_start = norm_raw_diffs[new_norm_start][0]

        # get the end of the span
        _, pred_text_end, status_norm_pred = norm_pred_diffs[norm_span[1] - 1]

        new_norm_end = pred_norm_diffs[pred_text_end - 1][1]
        raw_text_end = norm_raw_diffs[new_norm_end - 1][1]

        if standard_start < raw_text_start:
            text_for_audio_based["standard"] += " " + " ".join(raw_text_list[standard_start:raw_text_start])

        cur_semiotic_span = f"{' '.join(raw_text_list[raw_text_start:raw_text_end])}"
        cur_pred_text = f"{' '.join(pred_text_list[pred_text_start:pred_text_end])}"

        text_for_audio_based["semiotic"].append(cur_semiotic_span)
        text_for_audio_based["pred_text"].append(cur_pred_text)
        text_for_audio_based["standard"] += f" {SEMIOTIC_TAG} "

        standard_start = raw_text_end

    if standard_start < len(raw_text_list):
        text_for_audio_based["standard"] += ' '.join(raw_text_list[standard_start:])

    text_for_audio_based["standard"] = text_for_audio_based["standard"].replace("  ", " ").strip()
    return text_for_audio_based


def adjust_boundaries(norm_raw_diffs, norm_pred_diffs, raw, norm, pred_text):
    adjusted = []
    word_id = 0
    while word_id < len(norm.split()):
        norm_raw, norm_pred = norm_raw_diffs[word_id], norm_pred_diffs[word_id]
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
            adjusted.append(
                (
                    mismatched_id,
                    (non_match_raw_start, non_match_raw_end, NONMATCH),
                    (non_match_pred_start, non_match_pred_end, NONMATCH),
                )
            )
        else:
            adjusted.append((word_id, norm_raw, norm_pred))
        word_id += 1

    adjusted2 = []
    last_status = None
    for idx, item in enumerate(adjusted):
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
            adjusted2.append(
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
        adjusted2.append(
            [[norm_span_start, item[0]], [raw_start, raw_end], [pred_text_start, pred_text_end], last_status]
        )
    else:
        adjusted2.append(
            [
                [adjusted[idx - 1][0], len(norm.split())],
                [item[1][0], len(raw.split())],
                [item[2][0], len(pred_text.split())],
                item[1][2],
            ]
        )

    raw_list = raw.split()
    pred_text_list = pred_text.split()
    norm_list = norm.split()

    extended_spans = []
    adjusted3 = []
    idx = 0
    while idx < len(adjusted2):
        item = adjusted2[idx]
        # cur_semiotic = " ".join(raw_list[item[1][0] : item[1][1]])
        # cur_pred_text = " ".join(pred_text_list[item[2][0] : item[2][1]])
        # cur_norm_span = " ".join(norm_list[item[0][0] : item[0][1]])

        # if cur_pred_text is an empty string
        if item[2][0] == item[2][1]:
            # for the last item
            if idx == len(adjusted2) - 1 and len(adjusted3) > 0:
                last_item = adjusted3[-1]
                last_item[0][1] = item[0][1]
                last_item[1][1] = item[1][1]
                last_item[2][1] = item[2][1]
                last_item[-1] = item[-1]
            else:
                raw_start, raw_end = item[0]
                norm_start, norm_end = item[1]
                pred_start, pred_end = item[2]
                while idx < len(adjusted2) - 1 and not ((pred_end - pred_start) > 2 and adjusted2[idx][-1] == MATCH):
                    idx += 1
                    raw_end = adjusted2[idx][0][1]
                    norm_end = adjusted2[idx][1][1]
                    pred_end = adjusted2[idx][2][1]
                cur_item = [[raw_start, raw_end], [norm_start, norm_end], [pred_start, pred_end], NONMATCH]
                adjusted3.append(cur_item)
                extended_spans.append(len(adjusted3) - 1)
            idx += 1
        else:
            adjusted3.append(item)
            idx += 1

    semiotic_spans = []
    norm_spans = []
    pred_texts = []
    raw_text_masked = ""
    for idx, item in enumerate(adjusted3):
        cur_semiotic = " ".join(raw_list[item[1][0] : item[1][1]])
        cur_pred_text = " ".join(pred_text_list[item[2][0] : item[2][1]])
        cur_norm_span = " ".join(norm_list[item[0][0] : item[0][1]])

        if idx == len(adjusted3) - 1:
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

    # print("+" * 50)
    # print("adjusted:")
    # for item in adjusted2:
    #     print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
    #
    # print("+" * 50)
    # print("adjusted2:")
    # for item in adjusted2:
    #     print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
    # print("+" * 50)
    # print("adjusted3:")
    # for item in adjusted3:
    #     print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
    # print("+" * 50)
    return semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx


def get_alignment(raw, norm, pred_text, verbose: bool = False):
    norm_pred_diffs = _get_alignment(norm, pred_text)
    norm_raw_diffs = _get_alignment(norm, raw)

    semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx = adjust_boundaries(
        norm_raw_diffs, norm_pred_diffs, raw, norm, pred_text
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
    raw = "This, example: number 1,500 can be a very long one!, and can fail to produce valid normalization for such an easy number like 10,125 or dollar value $5349.01, and can fail to terminate, and can fail to terminate, and can fail to terminate, 452."
    norm = "This, example: number one thousand five hundred can be a very long one!, and can fail to produce valid normalization for such an easy number like ten thousand one hundred twenty five or dollar value five thousand three hundred and forty nine dollars and one cent, and can fail to terminate, and can fail to terminate, and can fail to terminate, four fifty two."
    pred_text = "this w example nuber viteen hundred can be a very h lowne one and can fail to produce a valid normalization for such an easy number like ten thousand one hundred twenty five or dollar value five thousand three hundred and fortyn nine dollars and one cent and can fail to terminate and can fail to terminate and can fail to terminate four fifty two"

    raw = 'It immediately took the #4 ranking on Apple and Google app stores prior to any featuring and'
    pred_text = 'it immediately took the number four ranking on apple and google apt stores prior to any featuring'
    norm = 'It immediately took the hash four ranking on Apple and Google app stores prior to any featuring and'

    get_alignment(raw, norm, pred_text, True)

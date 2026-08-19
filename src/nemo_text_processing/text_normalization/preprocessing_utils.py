from typing import List


def _split(sentences: List[str], delimiter: str, max_len: int, min_len: int):
    """
    Splits sentences based by the specified delimiter. Will attempt to split and combine sentences to get target
        min/max length.

    Args:
        sentences: Sentences to split into segments.
        delimiter: symbol to split by
        max_len: the maximum number of symbols in the output sentences (the result will be the closest len match)
        min_len: the minimum number of the output sentences (the result will be the closest len match)
    """
    result = []
    for sent in sentences:
        if len(sent) < max_len:
            result.append(sent)
            continue

        split_sent = sent.split(delimiter)
        # keep the delimiter
        split_sent = [(s + delimiter).strip() for s in split_sent[:-1]] + [split_sent[-1]]

        if "," in delimiter:
            # split based on comma usually results in too short utterance, combine sentences
            # that result in a single word split. It's usually not recommended to do that for other delimiters.
            comb = []
            for s in split_sent:
                # if the previous sentence is too short, combine it with the current sentence
                if len(comb) > 0 and (len(comb[-1].split()) <= min_len or len(s.split()) <= min_len):
                    comb[-1] = comb[-1] + " " + s
                else:
                    comb.append(s)
            result.extend(comb)
        else:
            result.extend(split_sent)
    return result


def additional_split(sentences: List[str], split_on_symbols: str, max_len: int = 1000, min_len: int = 2) -> List[str]:
    """
    Splits sentences by split_on_symbols.

    Args:
        sentences: Sentences to split into segments.
        split_on_symbols: Symbols to split sentences if eos sentence split resulted in a long sequence.
            Use '|' as a separator between symbols, for example: ';|:| ', will attempt to split each sentence
            by semi-colon ";", colon ":", and space " ".
        max_len: the maximum number of symbols in the output sentences (the result will be the closest len match)
        min_len: the minimum number of the output sentences (the result will be the closest len match)
    """
    if len(split_on_symbols) == 0:
        return sentences

    split_on_symbols = split_on_symbols.split("|")

    another_sent_split = []
    for sent in sentences:
        split_sent = [sent]
        for delimiter in split_on_symbols:
            if len(delimiter) == 0:
                continue
            split_sent = _split(
                split_sent, delimiter + " " if delimiter != " " else delimiter, max_len=max_len, min_len=min_len
            )
        another_sent_split.extend(split_sent)

    sentences = [s.strip() for s in another_sent_split if s.strip()]
    return sentences

import pynini
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    delete_zero_or_one_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time in Hebrew
        e.g. time { hours: "2" minutes: "55" suffix: "לילה" } -> 2:55
        e.g. time { hours: "2" minutes: "57" suffix: "בוקר" } -> 2:57
        e.g. time { prefix: "ב" hours: "6" minutes: "32" suffix: "ערב" } -> ב-18:32
        e.g. time { prefix: "בשעה" hours: "2" minutes: "10" suffix: "צהריים" } -> בשעה-14:10

    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour_to_noon = pynini.string_map([
            ("12", "12"),
            ("1", "13"),
            ("2", "14"),
            ("3", "15"),
            ("4", "16"),
            ("5", "17"),
            ("6", "18"),
        ])

        hour_to_evening = pynini.string_map([
            ("5", "17"),
            ("6", "18"),
            ("7", "19"),
            ("8", "20"),
            ("9", "21"),
            ("10", "22"),
            ("11", "23"),
        ])

        hour_to_night = pynini.string_map([
            ("8", "20"),
            ("9", "21"),
            ("10", "22"),
            ("11", "23"),
            ("12", "0"),
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
            ("4", "4"),
        ])

        day_suffixes = pynutil.delete("suffix: \"בוקר\"")
        noon_suffixes = pynutil.delete("suffix: \"צהריים\"")
        evening_suffixes = pynutil.delete("suffix: \"ערב\"")
        night_suffixes = pynutil.delete("suffix: \"לילה\"")

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )

        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )

        prefix = (
            pynutil.delete("prefix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.insert("-")
            + pynutil.delete("\"")
        )

        optional_prefix = pynini.closure(prefix + delete_zero_or_one_space, 0, 1)
        optional_suffix = pynini.closure(delete_space + day_suffixes, 0, 1)
        graph = hour + delete_space + pynutil.insert(":") + minute + optional_suffix

        for hour_to, suffix in zip([hour_to_noon, hour_to_evening, hour_to_night], [noon_suffixes, evening_suffixes, night_suffixes]):
            graph |= (
                hour @ hour_to
                + delete_space
                + pynutil.insert(":")
                + minute
                + delete_space
                + suffix
            )

        graph |= optional_prefix + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


if __name__ == "__main__":
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    time = TimeFst().fst
    # apply_fst('time { hours: "2" minutes: "10" suffix: "בוקר" }', time)
    # apply_fst('time { hours: "2" minutes: "01" suffix: "צהריים" }', time)
    # apply_fst('time { hours: "12" minutes: "03" suffix: "צהריים" }', time)
    # apply_fst('time { hours: "2" minutes: "20" suffix: "בוקר" }', time)
    # apply_fst('time { hours: "3" minutes: "00" suffix: "לילה" }', time)
    # apply_fst('time { hours: "2" minutes: "55" suffix: "לילה" }', time)
    # apply_fst('time { hours: "2" minutes: "57" suffix: "בוקר" }', time)
    # apply_fst('time { prefix: "ב" hours: "6" minutes: "32" suffix: "ערב" }', time)
    # apply_fst('time { prefix: "בשעה" hours: "2" minutes: "10" suffix: "צהריים" }', time)
import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.el.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)

GREEK_LOWER = pynini.union(
    "α",
    "β",
    "γ",
    "δ",
    "ε",
    "ζ",
    "η",
    "θ",
    "ι",
    "κ",
    "λ",
    "μ",
    "ν",
    "ξ",
    "ο",
    "π",
    "ρ",
    "σ",
    "ς",
    "τ",
    "υ",
    "φ",
    "χ",
    "ψ",
    "ω",
)
GREEK_UPPER = pynini.union(
    "Α",
    "Β",
    "Γ",
    "Δ",
    "Ε",
    "Ζ",
    "Η",
    "Θ",
    "Ι",
    "Κ",
    "Λ",
    "Μ",
    "Ν",
    "Ξ",
    "Ο",
    "Π",
    "Ρ",
    "Σ",
    "Τ",
    "Υ",
    "Φ",
    "Χ",
    "Ψ",
    "Ω",
)
GREEK_ALPHA = pynini.union(GREEK_LOWER, GREEK_UPPER)


class MeasureFst(GraphFst):
    def __init__(self, cardinal: GraphFst, decimal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_no_tokens
        graph_unit = pynini.string_file(get_abs_path("data/measure/units.tsv")).optimize()

        delete_opt_space = pynini.closure(pynutil.delete(" "), 0, 1)

        cardinal_space = pynutil.insert("integer: \"") + cardinal_graph + pynutil.insert("\"")

        default_units = pynutil.insert("units: \"") + graph_unit + pynutil.insert("\"")

        tagger_graph = cardinal_space + delete_opt_space + default_units

        integer = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        unit = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        verbalizer_graph = integer + delete_space + insert_space + unit

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()

        graph_standard = pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")

        math_operations = pynini.string_file(get_abs_path("data/measure/math_operation.tsv"))
        delimiter = pynini.accep(" ") | pynutil.insert(" ")

        math = (
            (cardinal_graph | NEMO_ALPHA)
            + delimiter
            + math_operations
            + (delimiter | NEMO_ALPHA)
            + cardinal_graph
            + delimiter
            + pynini.cross("=", "ίσον")
            + delimiter
            + (cardinal_graph | NEMO_ALPHA)
        )

        math |= (
            (cardinal_graph | NEMO_ALPHA)
            + delimiter
            + pynini.cross("=", "ίσον")
            + delimiter
            + (cardinal_graph | NEMO_ALPHA)
            + delimiter
            + math_operations
            + delimiter
            + cardinal_graph
        )

        math = (
            pynutil.insert('units: "math" cardinal { integer: "') + math + pynutil.insert('" } preserve_order: true')
        )

        address = self.get_address_graph(cardinal)
        address = pynutil.insert('units: "address" integer: "') + address + pynutil.insert('" preserve_order: true')

        graph = graph_standard | math | address
        self.fst = self.add_tokens(graph).optimize()

    def get_address_graph(self, cardinal):
        address_num = pynini.compose(NEMO_DIGIT ** (1, 2), cardinal.graph_no_tokens)
        address_num |= address_num + insert_space + pynini.compose(NEMO_DIGIT ** (3, 4), cardinal.graph_no_tokens)

        direction = (
            pynini.cross("β", "Βόρεια")
            | pynini.cross("Β", "Βόρεια")
            | pynini.cross("ν", "Νότια")
            | pynini.cross("Ν", "Νότια")
            | pynini.cross("α", "Ανατολική")
            | pynini.cross("Α", "Ανατολική")
            | pynini.cross("δ", "Δυτική")
            | pynini.cross("Δ", "Δυτική")
        ) + pynini.closure(pynutil.delete("."), 0, 1)
        direction = pynini.closure(pynini.accep(NEMO_SPACE) + direction, 0, 1)

        address_words = pynini.string_file(get_abs_path("data/address/address_word.tsv"))
        address_words = (
            pynini.accep(NEMO_SPACE)
            + address_words
            + NEMO_SPACE
            + pynini.closure(GREEK_ALPHA, 1)
            + pynini.closure(NEMO_SPACE + pynini.closure(GREEK_ALPHA, 1))
        )

        comma_space = pynutil.delete(",") + pynini.accep(NEMO_SPACE)

        city = pynini.closure(GREEK_ALPHA | pynini.accep(NEMO_SPACE), 1)
        city = pynini.closure(comma_space + city, 0, 1)

        states = load_labels(get_abs_path("data/address/state.tsv"))
        state_graph = pynini.string_map(states)
        state = pynini.invert(state_graph)
        state = pynini.closure(comma_space + state, 0, 1)

        zip_code = pynini.compose(NEMO_DIGIT**5, cardinal.single_digits_graph)
        zip_code = pynini.closure(comma_space + zip_code, 0, 1)

        address = address_num + direction + address_words + pynini.closure(city + state + zip_code, 0, 1)
        return address

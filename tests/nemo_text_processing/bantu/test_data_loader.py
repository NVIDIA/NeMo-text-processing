# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane Labs. All rights reserved.
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

"""The shared loader that fills a language's profile skeleton from its TSV files."""

from dataclasses import replace

import pytest

from nemo_text_processing.inverse_text_normalization.bantu.data_loader import load_profile
from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    CountedScale,
    CurrencyContext,
    MoneyCues,
    NounContext,
    Place,
)

SKELETON = CardinalProfile(
    language="t",
    contexts=(NounContext("2", "bakopa", (), ()),),
    ten="deka",
    teen_join=" ke ",
    places=(Place(1, (), " vo "),),
    exact_boundaries=(),
    roles=(),
    max_value=99,
    counted_scales=(
        CountedScale(
            singular="muna",
            singular_count="ema",
            plural="mina",
            power=4,
            count_class="2",
            tens=(),
            remainder_class="2",
        ),
    ),
)


def write(root, files):
    for name, rows in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    return str(root)


BASE = {
    "numbers/units.tsv": [("bapi", "2", "2", "any"), ("deka", "10", "2", "atomic"), ("baima", "1", "2", "final")],
    "numbers/places.tsv": [("pideka", "2", "1")],
}


class TestBantuDataLoader:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_units_fill_atomic_and_final(self, tmp_path):
        profile = load_profile(write(tmp_path, BASE), SKELETON)
        context = profile.contexts[0]
        assert context.atomic_units == (("bapi", "2"), ("deka", "10"))
        assert context.final_units == (("bapi", "2"), ("baima", "1"))
        assert profile.places[0].forms == (("pideka", "2"),)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_missing_files_are_empty(self, tmp_path):
        profile = load_profile(write(tmp_path, BASE), SKELETON)
        assert profile.currencies == ()
        assert profile.money_cues == MoneyCues()
        assert profile.held_back_words == ()
        assert profile.roles == ()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_blank_lines_are_skipped(self, tmp_path):
        root = write(tmp_path, BASE)
        (tmp_path / "numbers/held_back.tsv").write_text("halfu\n\n", encoding="utf-8")
        assert load_profile(root, SKELETON).held_back_words == ("halfu",)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bad_column_count(self, tmp_path):
        root = write(tmp_path, {**BASE, "numbers/units.tsv": [("bapi", "2", "2")]})
        with pytest.raises(ValueError, match=r"units\.tsv:1"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unknown_where(self, tmp_path):
        root = write(tmp_path, {**BASE, "numbers/units.tsv": [("bapi", "2", "2", "sometimes")]})
        with pytest.raises(ValueError, match=r"units\.tsv:1.*sometimes"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unknown_class(self, tmp_path):
        root = write(tmp_path, {**BASE, "numbers/units.tsv": [("bapi", "2", "99", "any")]})
        with pytest.raises(ValueError, match=r"units\.tsv:1.*99"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unknown_power(self, tmp_path):
        root = write(tmp_path, {**BASE, "numbers/places.tsv": [("pideka", "2", "7")]})
        with pytest.raises(ValueError, match=r"places\.tsv:1.*7"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unknown_scale(self, tmp_path):
        root = write(tmp_path, {**BASE, "numbers/scale_words.tsv": [("bumega", "tens", "deka", "1")]})
        with pytest.raises(ValueError, match=r"scale_words\.tsv:1.*bumega"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_scale_words(self, tmp_path):
        rows = [("mina", "tens", "deka", "1"), ("mina", "ones", "kaima", "1"), ("mina", "hundreds", "hanto", "1")]
        profile = load_profile(write(tmp_path, {**BASE, "numbers/scale_words.tsv": rows}), SKELETON)
        scale = profile.counted_scales[0]
        assert (scale.tens, scale.count_ones, scale.count_hundreds) == (
            (("deka", "1"),),
            (("kaima", "1"),),
            (("hanto", "1"),),
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_currency_roles(self, tmp_path):
        rows = [("koin", "XTS", "2", "short-tens"), ("dolari", "USD", "2", "")]
        files = {**BASE, "money/currency.tsv": rows, "numbers/roles.tsv": [("mapi", "20", "short-tens")]}
        profile = load_profile(write(tmp_path, files), SKELETON)
        assert profile.currencies == (
            CurrencyContext("koin", "XTS", "2", ("short-tens",)),
            CurrencyContext("dolari", "USD", "2", ()),
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_currency_unknown_class(self, tmp_path):
        root = write(tmp_path, {**BASE, "money/currency.tsv": [("koin", "XTS", "99", "")]})
        with pytest.raises(ValueError, match=r"currency\.tsv:1.*99"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_roles_cues_exact_and_variants(self, tmp_path):
        files = {
            **BASE,
            "numbers/roles.tsv": [("mapi", "20", "short-tens")],
            "numbers/exact.tsv": [("tausi", "1000")],
            "numbers/spelling_variants.tsv": [("pideka", "piideka")],
            "money/cues.tsv": [("lipo", "stem"), ("dalo", "word"), ("olipon", "not")],
        }
        profile = load_profile(write(tmp_path, files), SKELETON)
        assert profile.roles == (("short-tens", (("mapi", "20"),)),)
        assert profile.exact_boundaries == (("tausi", "1000"),)
        assert profile.money_cues == MoneyCues(stems=("lipo",), words=("dalo",), not_cues=("olipon",))
        assert profile.places[0].forms == (("pideka", "2"), ("piideka", "2"))

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unknown_cue_kind(self, tmp_path):
        root = write(tmp_path, {**BASE, "money/cues.tsv": [("lipo", "verb")]})
        with pytest.raises(ValueError, match=r"cues\.tsv:1.*verb"):
            load_profile(root, SKELETON)

    # The test language's own data loads and builds a classifier.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_toy_profile_loads(self):
        from nemo_text_processing.inverse_text_normalization.bantu.taggers.tokenize_and_classify import (
            build_classifier,
        )
        from nemo_text_processing.inverse_text_normalization.toy.settings import load

        profile = load()
        assert [c.noun for c in profile.contexts] == ["kopa", "bakopa", "zikopa", "makopa", "ekopa"]
        assert ("piideka", "2") in profile.places[0].forms
        assert build_classifier(profile).num_states() > 0

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_currency_unknown_role(self, tmp_path):
        root = write(tmp_path, {**BASE, "money/currency.tsv": [("koin", "XTS", "2", "nosuchrole")]})
        with pytest.raises(ValueError, match=r"currency\.tsv:1.*nosuchrole"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_currency_class_must_compound(self, tmp_path):
        skeleton = replace(SKELETON, contexts=SKELETON.contexts + (NounContext("1", "kopa", (), (), compounds=False),))
        root = write(tmp_path, {**BASE, "money/currency.tsv": [("koin", "XTS", "1", "")]})
        with pytest.raises(ValueError, match=r"currency\.tsv:1.*class '1'"):
            load_profile(root, skeleton)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_field(self, tmp_path):
        root = write(tmp_path, {**BASE, "numbers/units.tsv": [("", "3", "2", "any")]})
        with pytest.raises(ValueError, match=r"units\.tsv:1.*empty"):
            load_profile(root, SKELETON)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_missing_data_dir(self, tmp_path):
        with pytest.raises(ValueError, match="no data folder"):
            load_profile(str(tmp_path / "nope"), SKELETON)

# Tamil data tables

Two-column, tab-separated, NFC-normalized. `pynini.string_file` cannot carry a notes column,
so what each table holds and which grammars read it is recorded here.

## Text normalization (this directory)

| Table | Contents | Source / notes |
|---|---|---|
| `numbers/digit.tsv` | ௧-௯ → ஒன்று … ஒன்பது | Read by the cardinal tagger, from ASCII digits too |
| `numbers/zero.tsv` | ௦ → பூஜ்யம் | Read by the cardinal tagger |
| `numbers/teens_and_ties.tsv` | ௧௦-௯௯ → joined compound words | Read by the cardinal tagger |
| `numbers/hundred.tsv` | ௧௦௦ → நூறு | Read by the cardinal tagger |
| `numbers/hundreds_exact.tsv` | ௨௦௦-௯௦௦ → இருநூறு … தொள்ளாயிரம் | Read by the cardinal tagger |
| `numbers/hundreds_combined.tsv` | ௨-௮ → இருநூற்று … எண்ணூற்று | The joined sandhi stems (not bare prefixes such as முன்/நான், which would give wrong forms like "நான் நூற்று") |
| `numbers/quantity_words.tsv` | written scale word → spoken word → native\|english\|short | Read by the decimal, money and range taggers. `native` words are spoken as written, `english` (lakh, crore) and the glued `short` forms (L, cr, K, M, B) in Tamil |
| `date/{days,months,year_suffix}.tsv` | day/month numerals → words; era abbreviations | Read by the date tagger |
| `time/{hours,minutes,seconds}.tsv` | hours 0-24, minutes/seconds 1-59 → words | Minutes and seconds stop at 59 (10:60 is not a time). Hour 24 is admitted as 24:00 alone. Also read by the ITN time tagger from the spoken side |
| `money/currency.tsv` | symbol/code → currency word | Includes the ரூ./ரூ spellings |
| `money/major_minor_currencies.tsv` | major → minor unit word | Read by both directions: the TN money verbalizer emits these pairs and the ITN money tagger inverts them |
| `fraction/idiomatic.tsv` | numerator word, denominator word → everyday fraction word (ஒன்று இரண்டு → அரை) | 3 columns. The three pairs spoken as their own everyday words instead of the கீழ் reading |
| `measure/unit.tsv` | unit abbreviation → spoken unit | `st` (stone) is left out because it swallows English ordinals (1st); includes `மீ`, `லி`, `சத` and the dotless spellings |
| `telephone/number.tsv` | digit in either script → word | Also read by the serial, electronic and ITN telephone taggers |
| `whitelist/abbreviations.tsv` | abbreviation → expansion | Read by the whitelist tagger |
| `whitelist/symbol.tsv` | symbol → spoken word | `-` and `+` are left out (a lone hyphen or plus is punctuation, and a leading sign is a field of the number classes), as are `<` `>` (markup; spoken only between digits by the tokenizer) |
| `whitelist/percent_suffix.tsv` | `%` with a glued case suffix → the inflected percent word (%க்கு → சதவீதத்துக்கு) | Read by the tokenizer's spacing rewrites |
| `serial/letters.tsv` | A-Z → spoken English letter name | English letter names in Tamil script; read by the serial and electronic taggers |
| `electronic/symbols.tsv` | `. @ / - _ : ~ +` → spoken symbol inside an address | The words match `whitelist/symbol.tsv` where that table already spells one, so a symbol never reads two ways |
| `electronic/domains.tsv` | top-level domain → spoken form (com → காம், in → ஐ என்) | A bare domain is read as one only when it ends in a listed TLD |
| `roman/context.tsv` | cue word → written ordinal marker the numeral takes when it precedes the cue (வகுப்பு XII reads a cardinal, XII வகுப்பு an ordinal) | Latin cues (Class, Chapter) for mixed text |

## Inverse text normalization (`inverse_text_normalization/ta/data`)

| Table | Contents | Source / notes |
|---|---|---|
| `numbers/half_forms.tsv` | fused fractional words → integer/fraction digits (ஒன்றரை → 1.5) | 3 columns; the regular -ரை and -ே readings beyond the table are built in the decimal tagger |
| `numbers/ambiguous.tsv` | number words that are also ordinary words: word → condition → reading | `licensed` (ஒரு/ஓர், also the indefinite article) counts as a number only inside a money or clock reading; `standalone` (கால்/அரை/முக்கால்) only when no Tamil word follows |
| `numbers/scale_words.tsv` | scale word → trailing zeros → expand\|keep | `expand` multiplies the amount out (ஐந்து புள்ளி ஐந்து ஆயிரம் → 5500); `keep` leaves the written idiom (5.5 லட்சம்) |
| `whitelist/prose_phrases.tsv` | phrases where a numeral is a pronoun or idiom (ஒன்று சேர்) | Column 2 is the reason; protected verbatim by the ITN whitelist tagger |
| `money/currency.tsv` | spoken currency word → symbol | One row per output of the TN `money/currency.tsv` plus plurals. பவுண்டு is absent on purpose: that is the mass pound in `measure/unit.tsv`, the currency word is பவுண்ட் |
| `money/minor_units.tsv` | extra minor-unit word → symbol | Only the rows `major_minor_currencies.tsv` cannot supply (plurals, and ₹ காசு) |
| `fraction/denominator_locative.tsv` | locative -இல் form → cardinal word | The tabulated forms; the ITN fraction tagger also applies the regular locative, so any denominator round-trips |

The ITN taggers read the TN `date/months.tsv`, `time/*.tsv`, `telephone/number.tsv` and
`money/major_minor_currencies.tsv` from the spoken side, so the two directions share one list
each and cannot drift apart.

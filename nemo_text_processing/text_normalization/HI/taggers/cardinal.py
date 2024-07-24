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


import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.HI.utils import get_abs_path, apply_fst


 
# Define digit to word mappings
graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).closure().optimize()
graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).closure().optimize()
graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).closure().optimize()
graph_hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv")).closure().optimize()
graph_thousands = pynini.string_file(get_abs_path("data/numbers/thousands.tsv")).closure().optimize()

# Combine all mappings
final_graph = graph_zero | graph_digit | graph_teens_and_ties | graph_hundred | graph_thousands
 
# Define a simple normalizer FST
normalize_fst = pynutil.insert(" ") + final_graph + pynutil.insert(" ")
 
# Save the FST to a file
normalize_fst.write("normalize_hindi.fst")
 
# Function to normalize text using the FST
def normalize_hindi(text):
    return apply_fst(text, normalize_fst)
 
# Example usage
input_text = "११११"
normalized_text = normalize_hindi(input_text)
print(normalized_text) 



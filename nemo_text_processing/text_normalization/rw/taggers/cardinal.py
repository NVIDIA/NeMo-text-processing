# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os 
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.rw.utils import get_abs_path
import re 

class CardinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        if not os.path.isfile(get_abs_path("data/cardinals/cardinals.tsv")):
        
            combined_dict = {}
            combined = []
            digit_to_word = {
                "0":"",
                "1": "rimwe",
                "2": "kabiri",
                "3": "gatatu",
                "4": "kane",
                "5": "gatanu",
                "6": "gatandatu",
                "7": "karindwi",
                "8": "munani",
                "9": "icyenda",
            }
            eleven_to_nineteen_word = {
                            "11": "icumi na rimwe",
                            "12": "icumi na kabiri",
                            "13": "icumi na gatatu",
                            "14": "icumi na kane",
                            "15": "icumi na gatanu",
                            "16": "icumi na gatandatu",
                            "17": "icumi na karindwi",
                            "18": "icumi na munani",
                            "19": "icumi na icyenda",
                        }
            # Tens mapping
            tens_to_word = {
                "0": "",
                "10": "icumi",
                "20": "makumyabiri",
                "30": "mirongo itatu",
                "40": "mirongo ine",
                "50": "mirongo itanu",
                "60": "mirongo itandatu",
                "70": "mirongo irindwi",
                "80": "mirongo inani",
                "90": "mirongo icyenda",
            }

            # Hundreds mapping
            hundreds_to_word = {
                "0":"",
                "100": "ijana",
                "200": "magana abiri",
                "300": "magana atatu",
                "400": "magana ane",
                "500": "magana atanu",
                "600": "magana atandatu",
                "700": "magana irindwi",
                "800": "magana inani",
                "900": "magana icyenda",
            }

            # Thousands mapping
            thousands_to_word = {
                "0":"",
                "1000": "igihumbi",
                "2000": "ibihumbi bibiri",
                "3000": "ibihumbi bitatu",
                "4000": "ibihumbi bine",
                "5000": "ibihumbi bitanu",
                "6000": "ibihumbi bitandatu",
                "7000": "ibihumbi birindwi",
                "8000": "ibihumbi inani",
                "9000": "ibihumbi icyenda",
            }
            tens_of_thousands_to_word = {"0":"","10000": "ibihumbi icumi", "20000": "ibihumbi makumyabiri", "30000": "ibihumbi mirongo itatu", "40000": "ibihumbi mirongo ine", "50000": "ibihumbi mirongo itanu", "60000": "ibihumbi mirongo itandatu", "70000": "ibihumbi mirongo irindwi", "80000": "ibihumbi mirongo inani", "90000": "ibihumbi mirongo icyenda"}
  
            combined_dict.update(digit_to_word,**tens_to_word,**eleven_to_nineteen_word)

            for key,val in combined_dict.items():
                combined.append(key+"\t"+val)
                    

            for tens in range(20, 100, 10):
                for digit in range(1, 10):
                    num_str = str(tens + digit)
                    tens_word = tens_to_word[str(tens)]
                    digit_word = digit_to_word[str(digit)]
                    combined.append(num_str+"\t"+f"{tens_word} na {digit_word}")

            for hundreds in range(100, 1000, 100):
                for tens in range(0, 100, 10):
                    for digit in range(0, 10):
                        num_str = str(hundreds + tens + digit)
                        hundreds_word = hundreds_to_word[str(hundreds)]
                        tens_word = tens_to_word[str(tens)]
                        digit_word = digit_to_word[str(digit)]
                        combined.append(num_str+"\t"+f"{hundreds_word} na {tens_word} na {digit_word}"+"\n")

            for thousands in range(1000, 10000, 1000):
                for hundreds in range(0, 1000, 100):
                    for tens in range(0, 100, 10):
                        for digit in range(0, 10):
                            num_str = str(thousands + hundreds + tens + digit)
                            thousands_word = thousands_to_word[str(thousands)]
                            hundreds_word = hundreds_to_word[str(hundreds)]
                            tens_word = tens_to_word[str(tens)]
                            digit_word = digit_to_word[str(digit)]
                            combined.append(num_str+"\t"+f"{thousands_word} na {hundreds_word} na {tens_word} na {digit_word}")

            for tens_of_thousands in range(10000,100000,10000):
                for thousands in range(0, 10000, 1000):
                    for hundreds in range(0, 1000, 100):
                        for tens in range(0, 100, 10):
                            for digit in range(0, 10):
                                num_str = str(tens_of_thousands + thousands + hundreds + tens + digit)
                                tens_of_thousands_word = tens_of_thousands_to_word[str(tens_of_thousands)]
                                thousands_word = thousands_to_word[str(thousands)]
                                hundreds_word = hundreds_to_word[str(hundreds)]
                                tens_word = tens_to_word[str(tens)]
                                digit_word = digit_to_word[str(digit)]
                                combined.append(num_str+"\t"+f"{tens_of_thousands_word} na {thousands_word} na {hundreds_word} na {tens_word} na {digit_word}")
                                

            modified_lines = []
            numbers = []
            for i,line in enumerate(combined):
                line = line.strip()
                if line.count("ibihumbi") > 1:
                    index = line.rfind("ibihumbi")
                    part1 = line[:index]
                    part2 = line[index:].replace('ibihumbi','')
                    line = part1+part2
                line = re.sub(r'\s+', ' ', line)
                split_line = line.split()
                numbers.append(split_line[0])
                split_line = split_line[1:]
                line_elements = []
                prev=''
                for element in split_line:      
                    if element == "na" and prev=="na":
                        continue
                    elif element == "na" and prev != "na":
                        prev='na'       
                    elif element!="na" and prev=="na":
                        prev=''
                    line_elements.append(element)
                modified_line = " ".join(line_elements)   
                if modified_line.endswith(" na"):
                    modified_line = modified_line[:-3] 
                modified_lines.append(modified_line)
                # if " na  na " in line:
                    # combined[i] = line.replace(" na  na "," na ")
                # elif line.endswith("na na"):
                    # combined[i] = line.replace(" na  na","")
                # elif line.endswith(" na"):
                    # combined[i] = line[:-3]
            
            with open(get_abs_path("data/cardinals/cardinals.tsv"),"a") as f:
                for number,line in zip(numbers,modified_lines):
                    f.write(number+"\t"+line.strip()+"\n")
        combined_fst = pynini.string_file(get_abs_path("data/cardinals/cardinals.tsv"))
        graph = combined_fst.optimize()
        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert(" \"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph

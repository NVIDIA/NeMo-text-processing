import re
import pynini
from pynini.lib import pynutil
import os 
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.rw.utils import get_abs_path


class CardinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        if not os.path.isfile(get_abs_path("data/cardinals/cardinals.tsv")):
        
            combined_dict = {}
            combined = []
            digit_to_word = {
                "0":"",
                "1": "imwe",
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
                "90": "mirongo cyenda",
            }
            
            # Hundreds mapping
            hundreds_to_word = {
                "0":"",
                "100": "ijana",
                "200": "amagana abiri",
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
            tens_of_thousands_to_word = {"0":"","10000": "ibihumbi icumi", "20000": "ibihumbi makumyabiri", "30000": "ibihumbi mirongo itatu", "40000": "ibihumbi mirongo ine", "50000": "ibihumbi mirongo itanu", "60000": "ibihumbi mirongo itandatu", "70000": "ibihumbi mirongo irindwi", "80000": "ibihumbi mirongo inani", "90000": "ibihumbi mirongo cyenda"}
            hundreds_of_thousands_to_word = {"0":"","100000": "ibihumbi ijana", "200000": "ibihumbi magana abiri", "300000": "ibihumbi magana atatu", "400000": "ibihumbi magana ane", "500000": "ibihumbi magana atanu", "600000": "ibihumbi magana atandatu", "700000": "ibihumbi magana arindwi", "800000": "ibihumbi magana inani", "900000": "ibihumbi magana icyenda"}
            millions_to_word = {"0":"","1000000": "miliyoni", "2000000": "miliyoni ebyiri", "3000000": "miliyoni eshatu", "4000000": "miliyoni enye", "5000000": "miliyoni eshanu", "6000000": "miliyoni esheshatu", "7000000": "miliyoni zirindwi", "8000000": "miliyoni umunani", "9000000": "miliyoni icyenda"}
            tens_of_millions_to_word = {"0":"","10000000": "miliyoni icumi", "20000000": "miliyoni makumyabiri", "30000000": "miliyoni mirongo itatu", "40000000": "miliyoni mirongo ine", "50000000": "miliyoni mirongo itanu", "60000000": "miliyoni mirongo itandatu", "70000000": "miliyoni mirongo irindwi", "80000000": "miliyoni mirongo inani", "90000000": "miliyoni mirongo icyenda"}
            hundreds_of_millions_to_word = {"0":"","100000000": "miliyoni ijana", "200000000": "miliyoni magana abiri", "300000000": "miliyoni magana atatu", "400000000": "miliyoni magana ane", "500000000": "miliyoni magana atanu", "600000000": "miliyoni magana atandatu", "700000000": "miliyoni magana arindwi", "800000000": "miliyoni magana inani", "900000000": "miliyoni magana icyenda"}
            billions_to_word = {"0":"","1000000000": "miliyari imwe", "2000000000": "miliyari ebyiri", "3000000000": "miliyari eshatu", "4000000000": "miliyari enye", "5000000000": "miliyari eshanu", "6000000000": "miliyari esheshatu", "7000000000": "miliyari arindwi", "8000000000": "miliyari umunani", "9000000000": "miliyari icyenda"}
            tens_of_billions_to_word = {"0":"","10000000000": "miliyari icumi", "20000000000": "miliyari makumyabiri", "30000000000": "miliyari mirongo itatu", "40000000000": "miliyari mirongo ine", "50000000000": "miliyari mirongo itanu", "60000000000": "miliyari mirongo itandatu", "70000000000": "miliyari mirongo irindwi", "80000000000": "miliyari mirongo inani", "90000000000": "miliyari mirongo icyenda"}
            hundreds_of_billions_to_word = {"0":"","100000000000": "miliyari ijana", "200000000000": "miliyari magana abiri", "300000000000": "miliyari magana atatu", "400000000000": "miliyari magana ane", "500000000000": "miliyari magana atanu", "600000000000": "miliyari magana atandatu", "700000000000": "miliyari magana arindwi", "800000000000": "miliyari magana inani", "900000000000": "miliyari magana icyenda"}
            trillions_to_word = {
                "0":"",
                "1000000000000": "tiriyoni imwe",
                "2000000000000": "tiriyoni ebyiri",
                "3000000000000": "tiriyoni eshatu",
                "4000000000000": "tiriyoni enye",
                "5000000000000": "tiriyoni eshanu",
                "6000000000000": "tiriyoni esheshatu",
                "7000000000000": "tiriyoni arindwi",
                "8000000000000": "tiriyoni umunani",
                "9000000000000": "tiriyoni icyenda",
            }
            tens_of_trillions_to_word = {
                "0":"",
                "10000000000000": "tiriyoni icumi",
                "20000000000000": "tiriyoni makumyabiri",
                "30000000000000": "tiriyoni mirongo itatu",
                "40000000000000": "tiriyoni mirongo ine",
                "50000000000000": "tiriyoni mirongo itanu",
                "60000000000000": "tiriyoni mirongo itandatu",
                "70000000000000": "tiriyoni mirongo irindwi",
                "80000000000000": "tiriyoni mirongo inani",
                "90000000000000": "tiriyoni mirongo icyenda",
            }
            hundreds_of_trillions_to_word = {
                "0":"",
                "100000000000000": "tiriyoni ijana",
                "200000000000000": "tiriyoni magana abiri",
                "300000000000000": "tiriyoni magana atatu",
                "400000000000000": "tiriyoni magana ane",
                "500000000000000": "tiriyoni magana atanu",
                "600000000000000": "tiriyoni magana atandatu",
                "700000000000000": "tiriyoni magana arindwi",
                "800000000000000": "tiriyoni magana inani",
                "900000000000000": "tiriyoni magana icyenda",
            }
            billion= 1000000000
            ten_billion= 10000000000
            hundred_billion= 100000000000
            trillion= 1000000000000
            ten_trillion= 10000000000000
            hundred_trillion= 100000000000000
            thousand_trillion= 1000000000000000
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
                                
            for hundreds_of_thousands in range(100000,1000000,100000):
                for tens_of_thousands in range(0,100000,10000):
                    for thousands in range(0, 10000, 1000):
                        for hundreds in range(0, 1000, 100):
                            for tens in range(0, 100, 10):
                                for digit in range(0, 10):
                                    num_str = str(hundreds_of_thousands + tens_of_thousands + thousands + hundreds + tens + digit)
                                    hundreds_of_thousands_word = hundreds_of_thousands_to_word[str(hundreds_of_thousands)]
                                    tens_of_thousands_word = tens_of_thousands_to_word[str(tens_of_thousands)]
                                    thousands_word = thousands_to_word[str(thousands)]
                                    hundreds_word = hundreds_to_word[str(hundreds)]
                                    tens_word = tens_to_word[str(tens)]
                                    digit_word = digit_to_word[str(digit)]
                                    combined.append(num_str+"\t"+f"{hundreds_of_thousands_word} na {tens_of_thousands_word} na {thousands_word} na {hundreds_word} na {tens_word} na {digit_word}")
            
            for millions in range(1000000,10000000,1000000):
                for hundreds_of_thousands in range(0,1000000,100000):
                    for tens_of_thousands in range(0,100000,10000):
                        for thousands in range(0, 10000, 1000):
                            for hundreds in range(0, 1000, 100):
                                for tens in range(0, 100, 10):
                                    for digit in range(0, 10):
                                        num_str = str(millions + hundreds_of_thousands + tens_of_thousands + thousands + hundreds + tens + digit)
                                        millions_word = millions_to_word[str(millions)]
                                        hundreds_of_thousands_word = hundreds_of_thousands_to_word[str(hundreds_of_thousands)]
                                        tens_of_thousands_word = tens_of_thousands_to_word[str(tens_of_thousands)]
                                        thousands_word = thousands_to_word[str(thousands)]
                                        hundreds_word = hundreds_to_word[str(hundreds)]
                                        tens_word = tens_to_word[str(tens)]
                                        digit_word = digit_to_word[str(digit)]
                                        combined.append(num_str+"\t"+f"{millions_word} na {hundreds_of_thousands_word} na {tens_of_thousands_word} na {thousands_word} na {hundreds_word} na {tens_word} na {digit_word}")
            
            for i,line in enumerate(combined):
                if " na  na " in line:
                    combined[i] = line.replace(" na  na "," na ")
                elif line.endswith("na  na"):
                    combined[i] = line.replace(" na  na","")
                elif line.endswith("na "):
                    combined[i] = line[:-3]
            
            with open(get_abs_path("data/cardinals/cardinals.tsv"),"a") as f:
                for line in combined:
                    f.write(line+"\n")
        combined_fst = pynini.string_file(get_abs_path("data/cardinals/cardinals.tsv"))
        graph = combined_fst.optimize()
        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert(" \"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph

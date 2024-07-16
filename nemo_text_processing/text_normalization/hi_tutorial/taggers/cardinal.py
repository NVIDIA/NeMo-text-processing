import pynini
from pynini.lib import rewrite, pynutil
from nemo_text_processing.text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space

def del_zero(n=1):                                         #prevents us from writing the function multiple times 
    return pynutil.delete("०") ** n 
    
class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }
 s
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        
        
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/number/teens_and_ties.tsv"))
        graph_hundred = pynini.string_file(get_abs_path("data/number/hundred.tsv"))
        
        
         
        graph_thousands = pynini.string_file(get_abs_path("data/number/thousands.tsv"))
       
        hundred = pynutil.insert(" सौ")
        ins_space = pynutil.insert(" ")
        #del_zero = pynutil.delete("०")
        
        graph_hundred_one = graph_digit + del_zero(2) + hundred 
        graph_hundred_one |= graph_digit + del_zero(1) + hundred + ins_space + graph_digit 
        graph_hundred_one |= graph_digit + hundred + ins_space + graph_teens_and_ties
        graph_hundred = pynini.cross("१००", " सौ")  | graph_hundred_one
        graph_hundred = graph_hundred.optimize()
        
        thousands = pynutil.insert(" हज़ार")
        ten_thousands = pynutil.insert("दस हज़ार")
        lakhs = pynutil.insert(" लाख")
        ten_lakhs = pynutil.insert("दस लाख")
        crores = pynutil.insert(" करोड़")
        ten_crores = pynutil.insert("दस करोड़")
        arabs = pynutil.insert(" अरब")
        ten_arabs = pynutil.insert("दस अरब")
        kharabs = pynutil.insert(" खरब")
        ten_kharabs = pynutil.insert("दस खरब")
        nils = pynutil.insert(" नील")
        ten_nils = pynutil.insert("दस नील")
        padmas = pynutil.insert(" पद्म")
        ten_padmas = pynutil.insert("दस पद्म")
        shankhs = pynutil.insert(" शंख")
        ten_shankhs = pynutil.insert("दस शंख")
        ins_space = pynutil.insert(" ")
        
        graph_thousands = graph_digit + del_zero(3) + thousands
        graph_thousands |= graph_digit + del_zero(2) + thousands + ins_space + graph_digit
        graph_thousands |= graph_digit + del_zero() + thousands + ins_space + graph_teens_and_ties 
        graph_thousands |= graph_digit + thousands + ins_space + graph_hundred_one
        graph_thousands = graph_thousands.optimize()
        
        graph_ten_thousands = graph_teens_and_ties + del_zero(3) + thousands
        graph_ten_thousands |= graph_teens_and_ties + del_zero(2) + thousands + ins_space + graph_digit
        graph_ten_thousands |= graph_teens_and_ties + del_zero() + thousands + ins_space + graph_teens_and_ties
        graph_ten_thousands |= graph_teens_and_ties + thousands + ins_space + graph_hundred_one
        graph_ten_thousands = graph_ten_thousands.optimize()
        
        graph_lakhs = graph_digit+ del_zero(5) + lakhs
        graph_lakhs |= graph_digit + del_zero(4) + lakhs + ins_space + graph_digit
        graph_lakhs |= graph_digit + del_zero(3) + lakhs + ins_space + graph_teens_and_ties
        graph_lakhs |= graph_digit+ del_zero(2) + lakhs + ins_space + graph_hundred
        graph_lakhs |= graph_digit+ del_zero(1) + lakhs + ins_space + graph_thousands
        graph_lakhs |= graph_digit + lakhs + ins_space + graph_ten_thousands
        graph_lakhs = graph_lakhs.optimize()
        
        graph_ten_lakhs = graph_teens_and_ties + del_zero(5) + lakhs
        graph_ten_lakhs |= graph_teens_and_ties + del_zero(4) + lakhs + ins_space + graph_digit
        graph_ten_lakhs |= graph_teens_and_ties + del_zero(3) + lakhs + ins_space + graph_teens_and_ties
        graph_ten_lakhs |= graph_teens_and_ties + del_zero(2) + lakhs + ins_space + graph_hundred
        graph_ten_lakhs |= graph_teens_and_ties + del_zero(1) + lakhs + ins_space + graph_thousands
        graph_ten_lakhs |= graph_teens_and_ties + lakhs + ins_space + graph_ten_thousands
        graph_ten_lakhs = graph_ten_lakhs.optimize()
        
        graph_crores = graph_digit + del_zero(7) + crores
        graph_crores |= graph_digit + del_zero(6) + crores + ins_space + graph_digit
        graph_crores |= graph_digit + del_zero(5) + crores + ins_space + graph_teens_and_ties 
        graph_crores |= graph_digit + del_zero(4)+ crores + ins_space + graph_hundred
        graph_crores |= graph_digit + del_zero(3) + crores + ins_space + graph_thousands
        graph_crores |= graph_digit + del_zero(2) + crores + ins_space + graph_ten_thousands
        graph_crores |= graph_digit + del_zero(1) + crores + ins_space + graph_lakhs
        graph_crores |= graph_digit + crores + ins_space + graph_ten_lakhs
        graph_crores = graph_crores.optimize()
        
        graph_ten_crores = graph_teens_and_ties + del_zero(7) + crores
        graph_ten_crores |= graph_teens_and_ties + del_zero(6) + crores + ins_space + graph_digit
        graph_ten_crores |= graph_teens_and_ties + del_zero(5) + crores + ins_space + graph_teens_and_ties 
        graph_ten_crores |= graph_teens_and_ties + del_zero(4)+ crores + ins_space + graph_hundred
        graph_ten_crores |= graph_teens_and_ties + del_zero(3) + crores + ins_space + graph_thousands
        graph_ten_crores |= graph_teens_and_ties + del_zero(2) + crores + ins_space + graph_ten_thousands
        graph_ten_crores |= graph_teens_and_ties + del_zero(1) + crores + ins_space + graph_lakhs
        graph_ten_crores |= graph_teens_and_ties + crores + ins_space + graph_ten_lakhs
        graph_ten_crores = graph_ten_crores.optimize() 
        
        graph_arabs = graph_digit + del_zero(9) + arabs
        graph_arabs |= graph_digit + del_zero(8) + arabs + ins_space + graph_digit
        graph_arabs |= graph_digit + del_zero(7) + arabs + ins_space + graph_teens_and_ties 
        graph_arabs |= graph_digit + del_zero(6)+ arabs + ins_space + graph_hundred
        graph_arabs |= graph_digit + del_zero(5) + arabs + ins_space + graph_thousands
        graph_arabs |= graph_digit + del_zero(4) + arabs + ins_space + graph_ten_thousands
        graph_arabs |= graph_digit + del_zero(3) + arabs + ins_space + graph_lakhs
        graph_arabs |= graph_digit + del_zero(2) + arabs + ins_space + graph_ten_lakhs
        graph_arabs |= graph_digit + del_zero(1) + arabs + ins_space + graph_crores
        graph_arabs |= graph_digit + arabs + ins_space + graph_ten_crores
        graph_arabs = graph_arabs.optimize()
       
        graph_ten_arabs = graph_teens_and_ties + del_zero(9) + arabs
        graph_ten_arabs |= graph_teens_and_ties + del_zero(8) + arabs + ins_space + graph_digit
        graph_ten_arabs |= graph_teens_and_ties + del_zero(7) + arabs + ins_space + graph_teens_and_ties 
        graph_ten_arabs |= graph_teens_and_ties + del_zero(6)+ arabs + ins_space + graph_hundred
        graph_ten_arabs |= graph_teens_and_ties + del_zero(5) + arabs + ins_space + graph_thousands
        graph_ten_arabs |= graph_teens_and_ties + del_zero(4) + arabs + ins_space + graph_ten_thousands
        graph_ten_arabs |= graph_teens_and_ties + del_zero(3) + arabs + ins_space + graph_lakhs
        graph_ten_arabs |= graph_teens_and_ties + del_zero(2) + arabs + ins_space + graph_ten_lakhs
        graph_ten_arabs |= graph_teens_and_ties + del_zero(1) + arabs + ins_space + graph_crores
        graph_ten_arabs |= graph_teens_and_ties + arabs + ins_space + graph_ten_crores
        graph_ten_arabs = graph_ten_arabs.optimize()
       
        graph_kharabs = graph_digit + del_zero(11) + kharabs
        graph_kharabs |= graph_digit + del_zero(10) + kharabs + ins_space + graph_digit
        graph_kharabs |= graph_digit + del_zero(9) + kharabs + ins_space + graph_teens_and_ties 
        graph_kharabs |= graph_digit + del_zero(8)+ kharabs + ins_space + graph_hundred
        graph_kharabs |= graph_digit + del_zero(7) + kharabs + ins_space + graph_thousands
        graph_kharabs |= graph_digit + del_zero(6) + kharabs + ins_space + graph_ten_thousands
        graph_kharabs |= graph_digit + del_zero(5) + kharabs + ins_space + graph_lakhs
        graph_kharabs |= graph_digit + del_zero(4) + kharabs + ins_space + graph_ten_lakhs
        graph_kharabs |= graph_digit + del_zero(3) + kharabs + ins_space + graph_crores
        graph_kharabs |= graph_digit + del_zero(2) + kharabs + ins_space + graph_ten_crores
        graph_kharabs |= graph_digit + del_zero(1) + kharabs + ins_space + graph_arabs
        graph_kharabs |= graph_digit + kharabs + ins_space + graph_ten_arabs
        graph_kharabs = graph_kharabs.optimize()
        
        graph_ten_kharabs = graph_teens_and_ties + del_zero(11) + kharabs
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(10) + kharabs + ins_space + graph_digit
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(9) + kharabs + ins_space + graph_teens_and_ties 
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(8)+ kharabs + ins_space + graph_hundred
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(7) + kharabs + ins_space + graph_thousands
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(6) + kharabs + ins_space + graph_ten_thousands
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(5) + kharabs + ins_space + graph_lakhs
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(4) + kharabs + ins_space + graph_ten_lakhs
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(3) + kharabs + ins_space + graph_crores
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(2) + kharabs + ins_space + graph_ten_crores
        graph_ten_kharabs |= graph_teens_and_ties + del_zero(1) + kharabs + ins_space + graph_arabs
        graph_ten_kharabs |= graph_teens_and_ties + kharabs + ins_space + graph_ten_arabs
        graph_ten_kharabs = graph_ten_kharabs.optimize()
        
        graph_nils = graph_digit + del_zero(13) + nils
        graph_nils |= graph_digit + del_zero(12) + nils + ins_space + graph_digit
        graph_nils |= graph_digit + del_zero(11) + nils + ins_space + graph_teens_and_ties 
        graph_nils |= graph_digit + del_zero(10)+ nils + ins_space + graph_hundred
        graph_nils |= graph_digit + del_zero(9) + nils + ins_space + graph_thousands
        graph_nils |= graph_digit + del_zero(8) + nils + ins_space + graph_ten_thousands
        graph_nils |= graph_digit + del_zero(7) + nils + ins_space + graph_lakhs
        graph_nils |= graph_digit + del_zero(6) + nils + ins_space + graph_ten_lakhs
        graph_nils |= graph_digit + del_zero(5) + nils + ins_space + graph_crores
        graph_nils |= graph_digit + del_zero(4) + nils + ins_space + graph_ten_crores
        graph_nils |= graph_digit + del_zero(3) + nils + ins_space + graph_arabs
        graph_nils |= graph_digit + del_zero(2) + nils + ins_space + graph_ten_arabs
        graph_nils |= graph_digit + del_zero(1) + nils + ins_space + graph_kharabs
        graph_nils |= graph_digit + nils + ins_space + graph_ten_kharabs
        graph_nils = graph_nils.optimize()
        
        graph_ten_nils = graph_teens_and_ties + del_zero(13) + nils
        graph_ten_nils |= graph_teens_and_ties + del_zero(12) + nils + ins_space + graph_digit
        graph_ten_nils |= graph_teens_and_ties + del_zero(11) + nils + ins_space + graph_teens_and_ties 
        graph_ten_nils |= graph_teens_and_ties + del_zero(10)+ nils + ins_space + graph_hundred
        graph_ten_nils |= graph_teens_and_ties + del_zero(9) + nils + ins_space + graph_thousands
        graph_ten_nils |= graph_teens_and_ties + del_zero(8) + nils + ins_space + graph_ten_thousands
        graph_ten_nils |= graph_teens_and_ties + del_zero(7) + nils + ins_space + graph_lakhs
        graph_ten_nils |= graph_teens_and_ties + del_zero(6) + nils + ins_space + graph_ten_lakhs
        graph_ten_nils |= graph_teens_and_ties + del_zero(5) + nils + ins_space + graph_crores
        graph_ten_nils |= graph_teens_and_ties + del_zero(4) + nils + ins_space + graph_ten_crores
        graph_ten_nils |= graph_teens_and_ties + del_zero(3) + nils + ins_space + graph_arabs
        graph_ten_nils |= graph_teens_and_ties + del_zero(2) + nils + ins_space + graph_ten_arabs
        graph_ten_nils |= graph_teens_and_ties + del_zero(1) + nils + ins_space + graph_kharabs
        graph_ten_nils |= graph_teens_and_ties + nils + ins_space + graph_ten_kharabs
        graph_ten_nils = graph_ten_nils.optimize()
        
        graph_padmas = graph_digit + del_zero(15) + padmas
        graph_padmas |= graph_digit + del_zero(14) + padmas + ins_space + graph_digit
        graph_padmas |= graph_digit + del_zero(13) + padmas + ins_space + graph_teens_and_ties 
        graph_padmas |= graph_digit + del_zero(12)+ padmas + ins_space + graph_hundred
        graph_padmas |= graph_digit + del_zero(11) + padmas + ins_space + graph_thousands
        graph_padmas |= graph_digit + del_zero(10) + padmas + ins_space + graph_ten_thousands
        graph_padmas |= graph_digit + del_zero(9) + padmas + ins_space + graph_lakhs
        graph_padmas |= graph_digit + del_zero(8) + padmas + ins_space + graph_ten_lakhs
        graph_padmas |= graph_digit + del_zero(7) + padmas + ins_space + graph_crores
        graph_padmas |= graph_digit + del_zero(6) + padmas + ins_space + graph_ten_crores
        graph_padmas |= graph_digit + del_zero(5) + padmas + ins_space + graph_arabs
        graph_padmas |= graph_digit + del_zero(4) + padmas + ins_space + graph_ten_arabs
        graph_padmas |= graph_digit + del_zero(3) + padmas + ins_space + graph_kharabs
        graph_padmas |= graph_digit + del_zero(2) + padmas + ins_space + graph_ten_kharabs
        graph_padmas |= graph_digit + del_zero(1) + padmas + ins_space + graph_nils
        graph_padmas |= graph_digit + padmas + ins_space + graph_ten_nils
        graph_padmas = graph_padmas.optimize()
        graph_ten_padmas = graph_teens_and_ties + del_zero(15) + padmas
        graph_ten_padmas |= graph_teens_and_ties + del_zero(14) + padmas + ins_space + graph_digit
        graph_ten_padmas |= graph_teens_and_ties + del_zero(13) + padmas + ins_space + graph_teens_and_ties 
        graph_ten_padmas |= graph_teens_and_ties + del_zero(12)+ padmas + ins_space + graph_hundred
        graph_ten_padmas |= graph_teens_and_ties + del_zero(11) + padmas + ins_space + graph_thousands
        graph_ten_padmas |= graph_teens_and_ties + del_zero(10) + padmas + ins_space + graph_ten_thousands
        graph_ten_padmas |= graph_teens_and_ties + del_zero(9) + padmas + ins_space + graph_lakhs
        graph_ten_padmas |= graph_teens_and_ties + del_zero(8) + padmas + ins_space + graph_ten_lakhs
        graph_ten_padmas |= graph_teens_and_ties + del_zero(7) + padmas + ins_space + graph_crores
        graph_ten_padmas |= graph_teens_and_ties + del_zero(6) + padmas + ins_space + graph_ten_crores
        graph_ten_padmas |= graph_teens_and_ties + del_zero(5) + padmas + ins_space + graph_arabs
        graph_ten_padmas |= graph_teens_and_ties + del_zero(4) + padmas + ins_space + graph_ten_arabs
        graph_ten_padmas |= graph_teens_and_ties + del_zero(3) + padmas + ins_space + graph_kharabs
        graph_ten_padmas |= graph_teens_and_ties + del_zero(2) + padmas + ins_space + graph_ten_kharabs
        graph_ten_padmas |= graph_teens_and_ties + del_zero(1) + padmas + ins_space + graph_nils
        graph_ten_padmas |= graph_teens_and_ties + padmas + ins_space + graph_ten_nils
        graph_ten_padmas = graph_ten_padmas.optimize()
        
        graph_shankhs = graph_digit + del_zero(17) + shankhs
        graph_shankhs |= graph_digit + del_zero(16) + shankhs + ins_space + graph_digit
        graph_shankhs |= graph_digit + del_zero(15) + shankhs + ins_space + graph_teens_and_ties 
        graph_shankhs |= graph_digit + del_zero(14)+ shankhs + ins_space + graph_hundred
        graph_shankhs |= graph_digit + del_zero(13) + shankhs + ins_space + graph_thousands
        graph_shankhs |= graph_digit + del_zero(12) + shankhs + ins_space + graph_ten_thousands
        graph_shankhs |= graph_digit + del_zero(11) + shankhs + ins_space + graph_lakhs
        graph_shankhs |= graph_digit + del_zero(10) + shankhs + ins_space + graph_ten_lakhs
        graph_shankhs |= graph_digit + del_zero(9) + shankhs + ins_space + graph_crores
        graph_shankhs |= graph_digit + del_zero(8) + shankhs + ins_space + graph_ten_crores
        graph_shankhs |= graph_digit + del_zero(7) + shankhs + ins_space + graph_arabs
        graph_shankhs |= graph_digit + del_zero(6) + shankhs + ins_space + graph_ten_arabs
        graph_shankhs |= graph_digit + del_zero(5) + shankhs + ins_space + graph_kharabs
        graph_shankhs |= graph_digit + del_zero(4) + shankhs + ins_space + graph_ten_kharabs
        graph_shankhs |= graph_digit + del_zero(3) + shankhs + ins_space + graph_nils
        graph_shankhs |= graph_digit + del_zero(2) + shankhs + ins_space + graph_ten_nils
        graph_shankhs |= graph_digit + del_zero(1) + shankhs + ins_space + graph_padmas
        graph_shankhs |= graph_digit + shankhs + ins_space + graph_ten_padmas
        graph_shankhs = graph_shankhs.optimize()
        
        graph_ten_shankhs = graph_teens_and_ties + del_zero(17) + shankhs
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(16) + shankhs + ins_space + graph_digit
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(15) + shankhs + ins_space + graph_teens_and_ties 
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(14)+ shankhs + ins_space + graph_hundred
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(13) + shankhs + ins_space + graph_thousands
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(12) + shankhs + ins_space + graph_ten_thousands
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(11) + shankhs + ins_space + graph_lakhs
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(10) + shankhs + ins_space + graph_ten_lakhs
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(9) + shankhs + ins_space + graph_crores
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(8) + shankhs + ins_space + graph_ten_crores
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(7) + shankhs + ins_space + graph_arabs
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(6) + shankhs + ins_space + graph_ten_arabs
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(5) + shankhs + ins_space + graph_kharabs
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(4) + shankhs + ins_space + graph_ten_kharabs
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(3) + shankhs + ins_space + graph_nils
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(2) + shankhs + ins_space + graph_ten_nils
        graph_ten_shankhs |= graph_teens_and_ties + del_zero(1) + shankhs + ins_space + graph_padmas
        graph_ten_shankhs |= graph_teens_and_ties + shankhs + ins_space + graph_ten_padmas
        graph_ten_shankhs = graph_ten_shankhs.optimize()
        
        
        
        
        
        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        
        final_graph = graph_digit | graph_zero | graph_teens_and_ties | graph_hundred | graph_thousands | graph_ten_thousands | graph_lakhs | graph_ten_lakhs | graph_crores | graph_ten_crores | graph_arabs | graph_ten_arabs | graph_kharabs | graph_ten_kharabs | graph_nils | graph_ten_nils | graph_padmas | graph_ten_padmas | graph_shankhs | graph_ten_shankhs 
        
        

        self.final_graph = final_graph
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
        
#input_text = "१११११"                                                                                              
#output = rewrite.top_rewrite(input_text,CardinalFst().fst)          # rewrite.rewrites - to see all possible outcomes , rewrite.top_rewrite - shortest pa
#print(output)
        

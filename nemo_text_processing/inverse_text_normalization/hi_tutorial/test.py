from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.hi.taggers.ordinal import OrdinalFst


cardinal = CardinalFst()
ordinal = OrdinalFst(cardinal) 
input_text = "एक सौ"    
output = rewrite.top_rewrite(input_text, ordinal.fst)
print(output)
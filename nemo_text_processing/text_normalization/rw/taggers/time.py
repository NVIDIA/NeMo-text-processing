# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
import pynini
from pynini.lib import pynutil


class TimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time", kind="classify")
        
        hours = pynini.string_map([
            ('1', 'saa saba'),
            ('2', 'saa munani'),
            ('3', 'saa cyenda'),
            ('4', 'saa cumi'),
            ('5', "saa cumi n'imwe"),
            ('6', "saa cumi n'ebyiri"),
            ('7', 'saa moya'),
            ('8', 'saa mbiri'),
            ('9', 'saa tatu'),
            ('10', 'saa ine'),
            ('11', 'saa tanu'),
            ('12', 'saa sita'),            
        ])
        
        minutes = pynini.string_map([
            ('00', ' '),
            ('01', " n'umunota umwe") ,
            ('02', " n'iminota ibiri") ,
            ('03', " n'iminota itatu") ,
            ('04', " n'iminota ine") ,
            ('05', " n'iminota itanu") ,
            ('06', " n'iminota itandatu") ,
            ('07', " n'iminota irindwi") ,
            ('08', " n'iminota umunani") ,
            ('09', " n'iminota icyenda") ,
            ('10', " n'iminota icumi") ,
            ('11', " n'iminota cumi n'umwe") ,
            ('12', " n'iminota cumi n'ibiri") ,
            ('13', " n'iminota cumi n'itatu") ,
            ('14', " n'iminota cumi n'ine") ,
            ('15', " n'iminota cumi n'itanu") ,
            ('16', " n'iminota cumi n'itandatu") ,
            ('17', " n'iminota cumi n'irindwi") ,
            ('18', " n'iminota cumi n'umunani") ,
            ('19', " n'iminota cumi n'icyenda") ,
            ('20', " n'iminota makumyabiri") ,
            ('21', " n'iminota makumyabiri na rimwe") ,
            ('22', " n'iminota makumyabiri n'ibiri") ,
            ('23', " n'iminota makumyabiri n'itatu") ,
            ('24', " n'iminota makumyabiri n'ine") ,
            ('25', " n'iminota makumyabiri n'itanu") ,
            ('26', " n'iminota makumyabiri n'itandatu") ,
            ('27', " n'iminota makumyabiri n'irindwi") ,
            ('28', " n'iminota makumyabiri n'umunani") ,
            ('29', " n'iminota makumyabiri n'icyenda") ,
            ('30', " n'iminota mirongo itatu") ,
            ('31', " n'iminota mirongo itatu n'umwe") ,
            ('32', " n'iminota mirongo itatu n'ibiri") ,
            ('33', " n'iminota mirongo itatu n'itatu") ,
            ('34', " n'iminota mirongo itatu n'ine") ,
            ('35', " n'iminota mirongo itatu n'itanu") ,
            ('36', " n'iminota mirongo itatu n'itandatu") ,
            ('37', " n'iminota mirongo itatu n'irindwi") ,
            ('38', " n'iminota mirongo itatu n'umunani") ,
            ('39', " n'iminota mirongo itatu n'icyenda") ,
            ('40', " n'iminota mirongo ine") ,
            ('41', " n'iminota mirongo ine n'umwe") ,
            ('42', " n'iminota mirongo ine n'ibiri") ,
            ('43', " n'iminota mirongo ine n'itatu") ,
            ('44', " n'iminota mirongo ine n'ine") ,
            ('45', " n'iminota mirongo ine n'itanu") ,
            ('46', " n'iminota mirongo ine n'itandatu") ,
            ('47', " n'iminota mirongo ine n'irindwi") ,
            ('48', " n'iminota mirongo ine n'umunani") ,
            ('49', " n'iminota mirongo ine n'icyenda") ,
            ('50', " n'iminota mirongo itanu") ,
            ('51', " n'iminota mirongo itanu n'umwe") ,
            ('52', " n'iminota mirongo itanu n'ibiri") ,
            ('53', " n'iminota mirongo itanu n'itatu") ,
            ('54', " n'iminota mirongo itanu n'ine") ,
            ('55', " n'iminota mirongo itanu n'itanu") ,
            ('56', " n'iminota mirongo itanu n'itandatu") ,
            ('57', " n'iminota mirongo itanu n'irindwi") ,
            ('58', " n'iminota mirongo itanu n'umunani") ,
            ('59', " n'iminota mirongo itanu n'icyenda") ,
        ])
        
        final_graph = pynutil.insert("hours:\"")+hours+pynutil.insert("\"")+pynutil.delete(":")+pynutil.insert(" minutes:\"")+minutes+pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()

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


# =================== #
# Hebrew Dictionaries #
# =================== #



tens_dict = {"2": "עשרים", "3": "שלושים", "4": "ארבעים", "5": "חמישים", "6": "שישים", "7": "שבעים",
             "8": "שמונים", "9": "תשעים"}

hundreds_dict = {"1": "מאה", "2": "מאתיים", "3": "שלוש מאות", "4": "ארבע מאות", "5": "חמש מאות", "6": "שש מאות",
                 "7": "שבע מאות", "8": "שמונה מאות", "9": "תשע מאות"}

thousands_dict = {"1": "אלף", "2": "אלפיים", "3": "שלושת אלפים", "4": "ארבעת אלפים", "5": "חמשת אלפים", "6": "ששת אלפים",
                  "7": "שבעת אלפים", "8": "שמונת אלפים", "9": "תשעת אלפים", "10": "עשרת אלפים"}

ten = {"short": "עשר", "long": "עשרה"}  # double pronunciation: short is 'eser' and 'asar', long is 'esre' and 'asara'

ordinal_numbers = {"1": "ראשון", "2": "שני", "3": "שלישי", "4": "רביעי", "5": "חמישי", "6": "שישי",
                   "7": "שביעי", "8": "שמיני", "9": "תשיעי", "10": "עשירי"}

common_fractures = {"1/2": "חצי", "1/3": "שליש", "1/4": "רבע", "1/5": "חמישית", "1/6": "שישית", "1/7": "שביעית",
                    "1/8": "שמינית", "1/9": "תשיעית", "1/10": "עשירית", "1/100": "מאית", "1/1000": "אלפית",
                    "2/3": "שני שליש", "3/4": "שלושת רבעי"}

special_floating = {"25": "רבע", "5": "חצי", "75": "שלושת רבעי"}

special_percents = {'1': ['אחוז', 'אחוז אחד']}

non_numbers = {"and": "ו", "%_sg": "אחוז", "%_pl": "אחוזים", "$": "דולר", "-": "מינוס", "₪": "שקל", "+": "ועוד"}

months_dict = {'1': 'ינואר', '2': 'פברואר', '3': 'מרץ', '4': 'אפריל', '5': 'מאי', '6': 'יוני',
               '7': 'יולי', '8': 'אוגוסט', '9': 'ספטמבר', '10': 'אוקטובר', '11': 'נובמבר', '12': 'דצמבר'}

hours_dict = {'13': 'אחת', '14': 'שתיים', '15': 'שלוש', '16': 'ארבע', '17': 'חמש', '18': 'שש', '19': 'שבע',
              '20': 'שמונה', '21': 'תשע', '22': 'עשר', '23': 'אחת עשרה', '24': 'שתיים עשרה', '00': ['שתיים עשרה', 'חצות']}

minutes_dict = {'1': ['דקה'], '2': ['שתי'], '3': ['שלוש'], '4': ['ארבע'], '5': ['חמישה', 'חמש'],
                '6': ['שש'], '7': ['שבע'], '8': ['שמונה'], '9': ['תשע'], '10': ['עשרה', 'עשר'],
                '15': ['רבע'], '30': ['שלושים', 'חצי'], '45': ['ארבעים וחמש', 'שלושת רבעי']}

minutes_special_cases = {'55': 'חמישה ל', '50': 'עשרה ל', '45': 'רבע ל', '40': 'עשרים ל'}

num_prefixes = ['וה', 'שה', 'ב', 'כ', 'ל', 'מ', 'ה', 'ו', 'וב', 'ול', 'ש', 'מה', 'ומ', 'שכ', 'שב', 'בכ', 'לכ']

range_join_phrases = [' ', ' ל', ' עד ']

big_numbers = {'million': 'מיליון', 'thousand': 'אלף'}
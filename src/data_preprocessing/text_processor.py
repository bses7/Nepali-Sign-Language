import re
from indic_transliteration import sanscript

class NepaliTextProcessor:
    def __init__(self):
        self.matra_map = {
            '\u093e': 'आ', '\u093f': 'इ', '\u0940': 'ई', '\u0941': 'उ',
            '\u0942': 'ऊ', '\u0943': 'ऋ', '\u0947': 'ए', '\u0948': 'ऐ',
            '\u094b': 'ओ', '\u094c': 'औ', '\u0902': 'अं', '\u0903': 'अः'
        }
        self.consonants = set([
            "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण",
            "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह",
            "क्ष", "त्र", "ज्ञ"
        ])
        self.special_conjuncts = {"क्ष", "त्र", "ज्ञ"}
        self.halant = '\u094d'

    def convert_to_nepali(self, text):
        """Detects if text is English and converts it to Nepali Devanagari."""
        if bool(re.search('[a-zA-Z]', text)):
            print(f"Romanized text detected: '{text}'")
            nepali_text = sanscript.transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
            print(f"🇳🇵 Converted to Devanagari: '{nepali_text}'")
            return nepali_text
        return text

    def get_clusters(self, text):
        text = self.convert_to_nepali(text)
        
        clusters = []
        i = 0
        while i < len(text):
            char = text[i]
            if char in self.special_conjuncts:
                clusters.append([char]); i += 1; continue
            if i + 2 < len(text) and text[i+1] == self.halant:
                c1, c2 = text[i], text[i+2]
                if i + 3 < len(text) and text[i+3] in self.matra_map:
                    clusters.append([c1, c2, self.matra_map[text[i+3]]]); i += 4
                else:
                    clusters.append([c1, c2]); i += 3
                continue
            if i + 1 < len(text) and text[i+1] in self.matra_map:
                clusters.append([char, self.matra_map[text[i+1]]]); i += 2; continue
            clusters.append([char]); i += 1
        return clusters
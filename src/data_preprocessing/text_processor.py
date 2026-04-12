import unicodedata

class NepaliTextProcessor:
    def __init__(self):
        self.matra_map = {
            '\u093e': 'आ', '\u093f': 'इ', '\u0940': 'ई', '\u0941': 'उ',
            '\u0942': 'ऊ', '\u0943': 'ऋ', '\u0947': 'ए', '\u0948': 'ऐ',
            '\u094b': 'ओ', '\u094c': 'औ', '\u0902': 'अं', '\u0903': 'अः'
        }
        
        self.special_conjuncts = {"क्ष", "त्र", "ज्ञ"}
        
        self.halant = '\u094d' # ्

    def get_clusters(self, text):
        """
        Groups text into communicative units.
        Example "प्रसाद" -> [['प', 'र'], ['स', 'आ'], ['द']]
        Example "क्षत्री" -> [['क्ष'], ['त', 'र', 'ई']]
        """
        clusters = []
        i = 0
        while i < len(text):
            char = text[i]
            
            if char in self.special_conjuncts:
                clusters.append([char])
                i += 1
                continue

            if i + 2 < len(text) and text[i+1] == self.halant:
                consonant_1 = text[i]
                consonant_2 = text[i+2]
                
                if i + 3 < len(text) and text[i+3] in self.matra_map:
                    vowel = self.matra_map[text[i+3]]
                    clusters.append([consonant_1, consonant_2, vowel])
                    i += 4
                else:
                    clusters.append([consonant_1, consonant_2])
                    i += 3
                continue

            if i + 1 < len(text) and text[i+1] in self.matra_map:
                vowel = self.matra_map[text[i+1]]
                clusters.append([char, vowel])
                i += 2
                continue

            clusters.append([char])
            i += 1
            
        return clusters
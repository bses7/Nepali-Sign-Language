# seed_data.py
from app.db.session import SessionLocal
# --- IMPORT ALL MODELS HERE ---
from app.models.user import User  # <--- ADD THIS
from app.models.gamification import Badge, UserStats
from app.models.lesson import Sign, UserProgress, Avatar, SignCategory, DifficultyLevel, user_avatars

def seed_database():
    db = SessionLocal()
    try:
        print("Syncing Avatars with Visual Attributes...")
        avatars_data = [
            {
                "name": "Rocky", "folder": "avatar", "price": 0,
                "attr": {
                    "type": "common",
                    "gender": "male",
                    "face_shape": "oval_rectangular", "skin_color": "#E6B59C", "hair_color": "#6A4B3A",
                    "eye_color": "#6A4A3A", "clothing_color": "#2F3A4A", "accessories": ["beard", "mustache"],
                    "shop_animations": [
                        "BreathingIdle", "Cheering", "HappyIdle", "Idle", "SadIdle", "Salute", "ThoughtfulHeadNod"
                    ]
                }
            },
            {
                "name": "Apollo", "folder": "avatar1", "price": 0,
                "attr": {
                    "type": "common", 
                    "gender": "male",
                    "face_shape": "round_oval", "skin_color": "#F0C4A8", "hair_color": "#1C1C1C",
                    "eye_color": "#4A342E", "clothing_color": "#9A5A66", "accessories": ["round_glasses_black"],
                    "shop_animations": ["BreathingIdle", "OffensiveIdle", "ThumbsUp"]
                }
            },
            {
                "name": "Adrian", "folder": "avatar2", "price": 500,
                "attr": {
                    "type": "rare",
                    "gender": "female",
                    "face_shape": "oval", "skin_color": "#F2C7B5", "hair_color": "#3B2F2F",
                    "eye_color": "#5A3E36", "clothing_color": "#9A5A66", "accessories": ["round_glasses_black"],
                    "shop_animations": ["Angry", "DwarfIdle", "LookingAround", "Thankful"]
                }
            },
            {
                "name": "Bianca", "folder": "avatar3", "price": 1250,
                "attr": {
                    "type": "legendary", 
                    "gender": "female",
                    "face_shape": "heart_oval", "skin_color": "#F3C8B2", "hair_color": "#121212",
                    "eye_color": "#5A4037", "clothing_color": "#2F3338", "accessories": [],
                    "shop_animations": ["DwarfIdle", "HandRaising"]
                }
            },
            {
                "name": "Drago", "folder": "avatar4", "price": 2500,
                "attr": {
                    "type": "legendary",
                    "gender": "male",
                    "face_shape": "rounded_square", "skin_color": "#9B5E47", "hair_color": "#000000",
                    "eye_color": "#3A2A24", "clothing_color": "#E9EEF2", "accessories": ["round_glasses_black", "beard"],
                    "shop_animations": ["BreathingIdle", "SillyDancing"]
                }
            },
        ]

        for a in avatars_data:
            obj = db.query(Avatar).filter(Avatar.folder_name == a["folder"]).first()
            if not obj:
                db.add(Avatar(name=a["name"], folder_name=a["folder"], price=a["price"], attributes=a["attr"]))
            else:
                obj.name = a["name"] 
                obj.price = a["price"]
                obj.attributes = a["attr"]

        print("Seeding Vowels...")
        vowels_data = [
            ("A", "अ", "नेपाली वर्णमालाको पहिलो स्वर वर्ण।", DifficultyLevel.EASY),
            ("AA", "आ", "'आमा' शब्दमा प्रयोग हुने स्वर वर्ण।", DifficultyLevel.EASY),
            ("I", "इ", "'इमान' बुझाउने छोटो ईकार।", DifficultyLevel.EASY),
            ("II", "ई", "'ईश्वर' बुझाउने लामो ईकार।", DifficultyLevel.EASY),
            ("U", "उ", "'उद्धार' बुझाउने छोटो उकार।", DifficultyLevel.MEDIUM),
            ("UU", "ऊ", "'ऊन' बुझाउने लामो उकार।", DifficultyLevel.MEDIUM),
            ("RE", "ऋ", "'ऋषि' बुझाउने स्वर वर्ण।", DifficultyLevel.HARD),
            ("E", "ए", "'एक' बुझाउने स्वर वर्ण।", DifficultyLevel.EASY),
            ("AI", "ऐ", "'ऐना' बुझाउने स्वर वर्ण।", DifficultyLevel.MEDIUM),
            ("O", "ओ", "'ओखल' बुझाउने स्वर वर्ण।", DifficultyLevel.MEDIUM),
            ("AU", "औ", "'औषधि' बुझाउने स्वर वर्ण।", DifficultyLevel.MEDIUM),
            ("AM", "अं", "'अंगुर' बुझाउने नाके स्वर।", DifficultyLevel.HARD),
            ("AH", "अः", "विसर्ग लागेको स्वर वर्ण।", DifficultyLevel.HARD)
        ]
        for code, char, desc, diff in vowels_data:
            desc_obj = {"text": desc, "image_url": f"/static/signs/vowels/{code}.png"}
            obj = db.query(Sign).filter(Sign.sign_code == code).first()
            if not obj:
                db.add(Sign(title=code, nepali_char=char, sign_code=code, category=SignCategory.VOWEL, difficulty=diff, description=desc_obj))
            else:
                obj.nepali_char = char; obj.difficulty = diff; obj.description = desc_obj

        print("Seeding Consonants...")
        consonants_data = [
            ("KA", "क", "'कलम' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("KHA", "ख", "'खरायो' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("GA", "ग", "'गमला' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("GHA", "घ", "'घर' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("NGA", "ङ", "'अङ्ग' मा प्रयोग हुने व्यञ्जन वर्ण।", DifficultyLevel.HARD),
            ("CHA", "च", "'चरा' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("CHHA", "छ", "'छाता' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("JA", "ज", "'जहाज' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("JHA", "झ", "'झण्डा' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("YAN", "ञ", "'चञ्चल' मा प्रयोग हुने व्यञ्जन वर्ण।", DifficultyLevel.HARD),
            ("TA", "ट", "'टपरी' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("THA", "ठ", "'ठग' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("DA", "ड", "'डमरु' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("DHA", "ढ", "'ढकनी' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("NA", "ण", "'बाण' मा प्रयोग हुने व्यञ्जन वर्ण।", DifficultyLevel.HARD),
            ("TAA", "त", "'तराजु' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("THAA", "थ", "'थैली' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("DAA", "द", "'दकल' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("DHAA", "ध", "'धनुष' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("NAA", "न", "'नल' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("PA", "प", "'पुल' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("PHA", "फ", "'फलफूल' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("BA", "ब", "'बल' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("BHA", "भ", "'भालु' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("MA", "म", "'मकल' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("YA", "य", "'यज्ञ' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("RA", "र", "'रथ' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("LA", "ल", "'लौरी' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("WA", "व", "'वकिल' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("T_SHA", "श", "'शङ्ख' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("M_SHA", "ष", "'षट्कोण' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.MEDIUM),
            ("D_SHA", "स", "'सलाई' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("HA", "ह", "'हलो' बुझाउने व्यञ्जन वर्ण।", DifficultyLevel.EASY),
            ("KSHA", "क्ष", "'क्षत्री' बुझाउने संयुक्त व्यञ्जन वर्ण।", DifficultyLevel.HARD),
            ("TRA", "त्र", "'त्रिशूल' बुझाउने संयुक्त व्यञ्जन वर्ण।", DifficultyLevel.HARD),
            ("GYA", "ज्ञ", "'ज्ञानी' बुझाउने संयुक्त व्यञ्जन वर्ण।", DifficultyLevel.HARD)
        ]
        for code, char, desc, diff in consonants_data:
            desc_obj = {"text": desc, "image_url": f"/static/signs/consonants/{code}.png"}
            obj = db.query(Sign).filter(Sign.sign_code == code).first()
            if not obj:
                db.add(Sign(title=code, nepali_char=char, sign_code=code, category=SignCategory.CONSONANT, difficulty=diff, description=desc_obj))
            else:
                obj.nepali_char = char; obj.difficulty = diff; obj.description = desc_obj

        print("Seeding Badges...")
        badges_data = [
            {"name": "Vowel Master", "description": "Learned all 13 Nepali vowels.", "code": "VOWEL_MASTER", "icon": "/static/badges/vowel_master.png"},
            {"name": "Early Bird", "description": "Completed a lesson before 7 AM.", "code": "EARLY_BIRD", "icon": "/static/badges/early_bird.png"},
            {"name": "Consistent Learner", "description": "Maintained a 7-day streak!", "code": "CONSISTENT_LEARNER", "icon": "/static/badges/streak_7.png"},
            {"name": "Consonant Crusader", "description": "Learned all 36 Nepali consonants.", "code": "CONSONANT_MASTER", "icon": "/static/badges/consonant_master.png"},
            {"name": "Easy Peasy", "description": "Completed all 'Easy' difficulty signs.", "code": "EASY_MASTER", "icon": "/static/badges/easy_master.png"},
            {"name": "Alphabet Ace", "description": "Mastered all 49 characters (Vowels + Consonants).", "code": "ALPHABET_ACE", "icon": "/static/badges/alphabet_ace.png"},
            {"name": "First Step", "description": "Reached Level 2!", "code": "LEVEL_2", "icon": "/static/badges/level_2.png"},
            {"name": "Coin Collector", "description": "Accumulated 500 coins.", "code": "COIN_500", "icon": "/static/badges/coins.png"},
            {"name": "Fashionista", "description": "Purchased your first premium avatar from the store.", "code": "AVATAR_BUYER", "icon": "/static/badges/store.png"},
            {"name": "Night Owl", "description": "Practiced between 10 PM and 4 AM.", "code": "NIGHT_OWL", "icon": "/static/badges/night_owl.png"},
            {"name": "Speed Demon", "description": "Completed 5 lessons in a single day.", "code": "SPEED_DEMON", "icon": "/static/badges/speed.png"},
            {"name": "Social Student", "description": "Successfully linked a Google or GitHub account.", "code": "SOCIAL_LINK", "icon": "/static/badges/social.png"}
        ]

        for b in badges_data:
            obj = db.query(Badge).filter(Badge.badge_code == b["code"]).first()
            if not obj:
                db.add(Badge(name=b["name"], description=b["description"], badge_code=b["code"], icon_url=b["icon"]))
            else:
                obj.name = b["name"]
                obj.description = b["description"]
                obj.icon_url = b["icon"]

        db.commit()
        print("Successfully seeded all Signs, Avatars, and Prices!")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
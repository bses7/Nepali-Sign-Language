# seed_data.py
from app.db.session import SessionLocal
# --- IMPORT ALL MODELS HERE ---
from app.models.user import User  # <--- ADD THIS
from app.models.gamification import Badge, UserStats
from app.models.lesson import Sign, UserProgress, Avatar, SignCategory, DifficultyLevel, user_avatars

def seed_database():
    db = SessionLocal()
    
    try:
        # 1. CLEAN UP
        print("Cleaning up old data...")
        # Delete link table first
        db.execute(user_avatars.delete())
        # Delete other related data
        db.query(UserProgress).delete()
        db.query(Sign).delete()
        db.query(Avatar).delete()
        db.commit()

        # 2. SEED AVATARS
        print("Seeding Avatars with prices...")
        avatars_data = [
            {"name": "Avatar 0", "folder": "avatar", "price": 0},
            {"name": "Avatar 1", "folder": "avatar1", "price": 0},
            {"name": "Avatar 2", "folder": "avatar2", "price": 100},
            {"name": "Avatar 3", "folder": "avatar3", "price": 250},
            {"name": "Avatar 4", "folder": "avatar4", "price": 500},
        ]
        for a in avatars_data:
            db.add(Avatar(name=a["name"], folder_name=a["folder"], price=a["price"]))

        # 3. SEED VOWELS
        print("Seeding Vowels...")
        vowels = [
            ("A", "अ", DifficultyLevel.EASY), ("AA", "आ", DifficultyLevel.EASY),
            ("I", "इ", DifficultyLevel.EASY), ("II", "ई", DifficultyLevel.EASY),
            ("U", "उ", DifficultyLevel.MEDIUM), ("UU", "ऊ", DifficultyLevel.MEDIUM),
            ("RE", "ऋ", DifficultyLevel.HARD), ("E", "ए", DifficultyLevel.EASY),
            ("AI", "ऐ", DifficultyLevel.MEDIUM), ("O", "ओ", DifficultyLevel.MEDIUM),
            ("AU", "औ", DifficultyLevel.MEDIUM), ("AM", "अं", DifficultyLevel.HARD),
            ("AH", "अः", DifficultyLevel.HARD)
        ]
        for code, char, diff in vowels:
            db.add(Sign(title=code, nepali_char=char, sign_code=code, category=SignCategory.VOWEL, difficulty=diff))

        # 4. SEED CONSONANTS
        print("Seeding Consonants...")
        consonants = [
            ("KA", "क", DifficultyLevel.EASY), ("GA", "ग", DifficultyLevel.EASY),
            ("MA", "म", DifficultyLevel.EASY), ("NAA", "न", DifficultyLevel.EASY),
            ("RA", "र", DifficultyLevel.EASY), ("LA", "ल", DifficultyLevel.EASY),
            ("PA", "प", DifficultyLevel.EASY), ("BA", "ब", DifficultyLevel.EASY),
            ("YA", "य", DifficultyLevel.EASY), ("WA", "व", DifficultyLevel.EASY),
            ("D_SHA", "स", DifficultyLevel.EASY), ("HA", "ह", DifficultyLevel.EASY),
            ("KHA", "ख", DifficultyLevel.MEDIUM), ("GHA", "घ", DifficultyLevel.MEDIUM),
            ("CHA", "च", DifficultyLevel.MEDIUM), ("CHHA", "छ", DifficultyLevel.MEDIUM),
            ("JA", "ज", DifficultyLevel.MEDIUM), ("JHA", "झ", DifficultyLevel.MEDIUM),
            ("TA", "ट", DifficultyLevel.MEDIUM), ("THA", "ठ", DifficultyLevel.MEDIUM),
            ("DA", "ड", DifficultyLevel.MEDIUM), ("DHA", "ढ", DifficultyLevel.MEDIUM),
            ("TAA", "त", DifficultyLevel.MEDIUM), ("THAA", "थ", DifficultyLevel.MEDIUM),
            ("DAA", "द", DifficultyLevel.MEDIUM), ("DHAA", "ध", DifficultyLevel.MEDIUM),
            ("PHA", "फ", DifficultyLevel.MEDIUM), ("BHA", "भ", DifficultyLevel.MEDIUM),
            ("T_SHA", "श", DifficultyLevel.MEDIUM), ("M_SHA", "ष", DifficultyLevel.MEDIUM),
            ("NGA", "ङ", DifficultyLevel.HARD), ("YAN", "ञ", DifficultyLevel.HARD),
            ("NA", "ण", DifficultyLevel.HARD), ("KSHA", "क्ष", DifficultyLevel.HARD),
            ("TRA", "त्र", DifficultyLevel.HARD), ("GYA", "ज्ञ", DifficultyLevel.HARD)
        ]
        for code, char, diff in consonants:
            db.add(Sign(title=code, nepali_char=char, sign_code=code, category=SignCategory.CONSONANT, difficulty=diff))

        print("Seeding Badges...")
        badges_data = [
            {
                "name": "Vowel Master",
                "description": "Learned all 13 Nepali vowels.",
                "code": "VOWEL_MASTER",
                "icon": "/static/badges/vowel_master.png"
            },
            {
                "name": "Early Bird",
                "description": "Completed a lesson before 7 AM.",
                "code": "EARLY_BIRD",
                "icon": "/static/badges/early_bird.png"
            },
            {
                "name": "Consistent Learner",
                "description": "Maintained a 7-day streak!",
                "code": "CONSISTENT_LEARNER",
                "icon": "/static/badges/streak_7.png"
            }
        ]

        for b in badges_data:
            if not db.query(Badge).filter(Badge.badge_code == b["code"]).first():
                db.add(Badge(
                    name=b["name"], 
                    description=b["description"], 
                    badge_code=b["code"], 
                    icon_url=b["icon"]
                ))

        db.commit()
        print("Successfully seeded all Signs, Avatars, and Prices!")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
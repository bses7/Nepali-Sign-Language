from app.db.session import SessionLocal
from app.models.lesson import Sign, UserProgress, Avatar, SignCategory, DifficultyLevel

def seed_database():
    db = SessionLocal()
    
    try:
        # 1. CLEAN UP (Delete existing data to avoid duplicates)
        print("Cleaning up old data...")
        db.query(UserProgress).delete()
        db.query(Sign).delete()
        db.query(Avatar).delete()
        db.commit()

        # 2. SEED AVATARS
        print("Seeding Avatars...")
        avatars_data = [
            {"name": "Avatar 0", "folder": "avatar"},
            {"name": "Avatar 1", "folder": "avatar1"},
            {"name": "Avatar 2", "folder": "avatar2"},
            {"name": "Avatar 3", "folder": "avatar3"},
            {"name": "Avatar 4", "folder": "avatar4"},
        ]
        for a in avatars_data:
            db.add(Avatar(name=a["name"], folder_name=a["folder"]))

        # 3. SEED VOWELS (13)
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
            db.add(Sign(
                title=code,
                nepali_char=char,
                sign_code=code,
                category=SignCategory.VOWEL,
                difficulty=diff
            ))

        # 4. SEED CONSONANTS (36)
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
            db.add(Sign(
                title=code,
                nepali_char=char,
                sign_code=code,
                category=SignCategory.CONSONANT,
                difficulty=diff
            ))

        db.commit()
        print("Successfully seeded all Signs and Avatars!")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
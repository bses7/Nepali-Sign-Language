from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from app.core.config import settings

conf = ConnectionConfig(
    MAIL_USERNAME=settings.SMTP_USER,
    MAIL_PASSWORD=settings.SMTP_PASSWORD,
    MAIL_FROM=settings.EMAILS_FROM_EMAIL,
    MAIL_PORT=settings.SMTP_PORT,
    MAIL_SERVER=settings.SMTP_HOST,
    MAIL_FROM_NAME=settings.EMAILS_FROM_NAME,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

async def send_reset_password_email(email_to: str, token: str):
    reset_link = f"http://localhost:3000/reset-password?token={token}"
    
    message = MessageSchema(
        subject="NSL - Password Reset Request",
        recipients=[email_to],
        body=f"Hi! Use this link to reset your password: {reset_link}. The link expires in 1 hour.",
        subtype=MessageType.plain
    )

    fm = FastMail(conf)
    await fm.send_message(message)

async def send_teacher_verified_email(email_to: str, first_name: str):
    message = MessageSchema(
        subject="NSL - Teacher Account Verified!",
        recipients=[email_to],
        body=f"""
        Namaste {first_name},

        Congratulations! Your teacher account for the Nepali Sign Language Platform has been verified by our team.

        You can now:
        - Upload sign language tutorial videos.
        - Contribute to our sign library.
        - Earn special rewards for your contributions.

        We are looking forward to collaborating with you to make NSL learning accessible to everyone.

        Best regards,
        The NSL Admin Team
        """,
        subtype=MessageType.plain
    )
    fm = FastMail(conf)
    await fm.send_message(message)
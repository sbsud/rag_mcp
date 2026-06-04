import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

def send_email(to: str, subject: str, body: str) -> dict:
    smtp_host     = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port     = int(os.getenv("SMTP_PORT", "587"))
    smtp_user     = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    if not smtp_user or not smtp_password:
        raise ValueError("SMTP_USER and SMTP_PASSWORD environment variables must be set")

    msg = MIMEMultipart()
    msg["From"]    = smtp_user
    msg["To"]      = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to, msg.as_string())

    logger.info("Email sent to %s — subject: %s", to, subject)
    return {"status": "sent", "to": to, "subject": subject}


def check_credentials() -> bool:
    """Used by /health endpoint — verifies SMTP login without sending."""
    try:
        smtp_host     = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port     = int(os.getenv("SMTP_PORT", "587"))
        smtp_user     = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
        return True
    except Exception as e:
        logger.warning("SMTP health check failed: %s", e)
        return False
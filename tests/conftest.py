import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def engine():
    from mailguard.engine import MailGuardEngine
    return MailGuardEngine()


PHISH = """From: "PayPal Security" <service@paypa1-secure.example>
Reply-To: help@other.test
Subject: Urgent: account suspended
Authentication-Results: mx.example.net; spf=fail smtp.mailfrom=paypa1-secure.example; dkim=none; dmarc=fail header.from=paypa1-secure.example

Dear customer, verify your account within 24 hours: http://paypa1-secure.example/login
"""
HAM = "Subject: meeting\n\nHi team, please review the budget numbers before our meeting tomorrow afternoon. Thanks, Sarah"


@pytest.fixture
def phish():
    return PHISH


@pytest.fixture
def ham():
    return HAM

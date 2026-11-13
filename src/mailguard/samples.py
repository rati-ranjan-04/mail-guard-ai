"""Synthetic demo messages. They use reserved example/test domains and fictional senders."""

SAMPLES = {
    "Phishing-style (synthetic)": """From: "PayPal Security Team" <service@paypa1-security-check.example>
Reply-To: help-desk@mailbox-support.test
Subject: Urgent: your account will be suspended
Authentication-Results: mx.example.net; spf=fail smtp.mailfrom=paypa1-security-check.example; dkim=none; dmarc=fail header.from=paypa1-security-check.example

Dear customer,

We detected unusual activity on your account. You must verify your account and confirm your password
within 24 hours or your account will be suspended.

Click here to log in: http://paypa1-security-check.example/secure/login?verify=1

PayPal Security Team
""",
    "Spam-style (synthetic)": """Subject: you have won a free prize - claim now!!!

Congratulations winner! Click here to claim your free prize, cheap pills, discount watches and a guaranteed
lottery payout. Limited offer, act now! Visit www.cheap-deals-today.test/claim and reply with your details.
""",
    "Legitimate-style (synthetic)": """From: Priya Nair <priya.nair@campus-example.test>
Subject: Project review meeting on Thursday
Authentication-Results: mx.example.net; spf=pass smtp.mailfrom=campus-example.test; dkim=pass header.d=campus-example.test; dmarc=pass header.from=campus-example.test

Hi team,

Please review the draft report before our meeting on Thursday at 3 pm. I have added the revised schedule and the
budget table to the shared folder. Let me know if any numbers look off.

Thanks,
Priya
""",
}

SAMPLE_BATCH_CSV = (
    "text,sender\n"
    '"Subject: team lunch friday\nHi all, lunch at 1 pm on Friday. Please confirm your headcount.",coordinator@campus-example.test\n'
    '"Subject: you won a free prize\nClaim your free prize now! Click here www.cheap-deals-today.test",promo@deals-now.test\n'
    '"Subject: verify your password\nYour account will be suspended. Verify your account at http://paypa1-security-check.example/login",security@paypa1-security-check.example\n'
)

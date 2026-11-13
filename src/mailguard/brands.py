"""Brand / domain reference data and string-similarity helpers (offline)."""

from __future__ import annotations

# Official registered domains for commonly impersonated brands (illustrative list, not exhaustive).
BRAND_DOMAINS: dict[str, tuple] = {
    "paypal": ("paypal.com", "paypal.me"),
    "microsoft": ("microsoft.com", "office.com", "live.com", "outlook.com", "microsoftonline.com"),
    "apple": ("apple.com", "icloud.com"),
    "google": ("google.com", "gmail.com", "googlemail.com", "goog.le"),
    "amazon": ("amazon.com", "amazon.in", "amazon.co.uk", "amazonaws.com"),
    "netflix": ("netflix.com",),
    "facebook": ("facebook.com", "fb.com", "meta.com"),
    "instagram": ("instagram.com",),
    "linkedin": ("linkedin.com", "lnkd.in"),
    "dropbox": ("dropbox.com",),
    "docusign": ("docusign.com", "docusign.net"),
    "whatsapp": ("whatsapp.com",),
    "paytm": ("paytm.com",),
    "hdfcbank": ("hdfcbank.com",),
    "icicibank": ("icicibank.com",),
    "sbi": ("sbi.co.in", "onlinesbi.sbi"),
    "coinbase": ("coinbase.com",),
    "binance": ("binance.com",),
    "fedex": ("fedex.com",),
    "dhl": ("dhl.com",),
    "usps": ("usps.com",),
    "chase": ("chase.com",),
    "wellsfargo": ("wellsfargo.com",),
    "bankofamerica": ("bankofamerica.com",),
}

# Display-name keywords that mean "official / security sender".
OFFICIAL_WORDS = ("support", "security", "admin", "billing", "service", "helpdesk",
                  "no-reply", "noreply", "accounts", "team", "verification")

FREE_MAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "proton.me",
             "protonmail.com", "icloud.com", "mail.com", "gmx.com", "yandex.com", "zoho.com",
             "rediffmail.com", "live.com", "ymail.com"}
DISPOSABLE = {"mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com",
              "yopmail.com", "trashmail.com", "sharklasers.com", "getnada.com", "throwawaymail.com"}

_MULTI_PART_SLD = {"co", "com", "org", "net", "gov", "ac", "edu", "nic", "res", "ltd"}
_COUNTRY_TLDS = {"uk", "in", "au", "br", "jp", "nz", "za", "sg", "hk", "tr", "mx", "ar", "cn", "kr", "id", "my", "ng"}


def registered_domain(host: str) -> str:
    """Approximate registered domain (eTLD+1) without a network-fetched public-suffix list."""
    labels = [l for l in (host or "").lower().strip(".").split(".") if l]
    if len(labels) <= 2:
        return ".".join(labels)
    if labels[-1] in _COUNTRY_TLDS and labels[-2] in _MULTI_PART_SLD:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_CONFUSABLES = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s"})


def closest_brand(label: str):
    """Return (brand, distance) if ``label`` imitates a brand name, else None.

    Short brands (<=5 letters) only match via character substitution (app1e);
    longer brands also allow small edit distances (paypall, micros0ft).
    An exact brand label returns None (callers compare against official domains).
    """
    label = (label or "").lower()
    if len(label) < 4:
        return None
    norm = label.translate(_CONFUSABLES).replace("rn", "m")
    best = None
    for brand in BRAND_DOMAINS:
        if len(brand) < 4 or label == brand:
            continue
        limit = 0 if len(brand) <= 5 else (1 if len(brand) <= 8 else 2)
        d = min(levenshtein(label, brand) if limit else 99, levenshtein(norm, brand))
        if d <= limit and (best is None or d < best[1]):
            best = (brand, d)
    return best


def brand_lookalike(label: str):
    """Check a domain label and its hyphen-separated parts (paypa1-secure) for brand imitation."""
    for part in [label] + [p for p in (label or "").split("-") if p]:
        hit = closest_brand(part)
        if hit:
            return hit
    return None

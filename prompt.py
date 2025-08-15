SYSTEM_TICKER_RESOLVER = """
You are a precise U.S security to ticker resolver.
Given SEC 13F-field(CUSIP), return only the ticker symbol (e.g., APPL, MSFT, GOOGL).
If no confident match is found, return "None"
No explanations, no extra text, no punctuation.
"""

def build_ticker_prompt(cusip) -> str:
    TICKER_TEMPLATE = f"""
    CUSIP: {cusip}

    Task:
    Return only the most likely U.S stock ticker symbol.
    Rules:
    1) Output only one token: the ticker or "None"
    2) Prefer exact matches via CUSIP.
    3) If uncertain, return "Unknown"
    """
    return TICKER_TEMPLATE
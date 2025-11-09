import re

def mask_aadhaar(text: str) -> str:
    """
    Replace 12-digit Aadhaar nnumbers with XXXX-XXXX-#### format.
    Example: 123412341234 -> XXXX-XXXX-1234
    """
    return re.sub(r"\b\d{12}\b", lambda m: f"XXXX-XXXX-{m.group()[8:]}", text)

# Unit Test Example
if __name__ == "__main__":
    sample = "My Aadhaar number is 123412345678 and should be masked"
    print(mask_aadhaar(sample)) # Output: My Aadhar number is XXXX-XXXX-5678 and should be masked.

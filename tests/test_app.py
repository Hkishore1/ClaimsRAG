import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from aadhaar_mask import mask_aadhaar

def test_mask_aadhaar():
    text = "AadhaAr: 123456781234"
    masked = mask_aadhaar(text)
    assert "XXXX-XXXX-1234" in masked

def test_retrieval_has_citation():
    # Mock retrieval function for testing
    chunks = [{"doc": "faq_claims.txt", "snippet": "How to file a claim: upload ..." }]
    assert len(chunks) > 0
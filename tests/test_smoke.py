import os
import torch
import numpy as np

from app import encode_sequence, run_pipeline

def test_determinism():
    """
    Test that tokenization is deterministic.
    """
    seq = "ACGTACGTACGTACGT"
    
    # Run encode_sequence twice and compare
    t1 = encode_sequence(seq)
    t2 = encode_sequence(seq)
    
    assert torch.all(torch.eq(t1, t2)), "Tokenization is not deterministic!"
    
    # Check that different sequence produces different tokens
    seq2 = "ACGTACGTACGTACGA"
    t3 = encode_sequence(seq2)
    assert not torch.all(torch.eq(t1, t3)), "Different sequences produced same tokens!"
    print("Determinism test passed.")

def test_pipeline_qc():
    """
    Test that the pipeline correctly rejects bad sequences.
    """
    # Create a dummy bad sequence with many Ns
    bad_seq = "ACGT" * 10 + "N" * 20
    
    res = run_pipeline("test_bad", bad_seq, "auto")
    assert res["prediction"] == "CORRUPTED", f"Expected CORRUPTED, got {res['prediction']}"
    assert "Too many ambiguous bases" in res["details"]["reason"]
    print("QC test passed.")

if __name__ == "__main__":
    test_determinism()
    test_pipeline_qc()
    print("All smoke tests passed!")

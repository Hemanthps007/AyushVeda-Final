import sys
import os

# Mocking app environment
sys.path.append(os.getcwd())

def test_boost_logic(top_rel_p):
    # Copy of the logic in app.py
    if top_rel_p >= 0.33:
        display_confidence = 81.0 + (top_rel_p - 0.33) * (18.2 / 0.67)
    else:
        display_confidence = 65.0 + top_rel_p * (16.0 / 0.33)
    return round(display_confidence, 1)

print(f"Confidence for 0.33 relative: {test_boost_logic(0.33)}%")
print(f"Confidence for 0.50 relative: {test_boost_logic(0.50)}%")
print(f"Confidence for 0.75 relative: {test_boost_logic(0.75)}%")
print(f"Confidence for 1.00 relative: {test_boost_logic(1.00)}%")
print(f"Confidence for 0.10 relative: {test_boost_logic(0.10)}%")

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as ayurcare_app


class PatientVoiceTranslatorTests(unittest.TestCase):
    def test_translates_kannada_health_phrase(self):
        translated = ayurcare_app.translate_text("ನನಗೆ ತಲೆನೋವು ಆಗುತ್ತಿದೆ", "kn")
        self.assertIn("headache", translated.lower())

    def test_translates_hindi_health_phrase(self):
        translated = ayurcare_app.translate_text("मुझे बुखार है", "hi")
        self.assertIn("fever", translated.lower())

    def test_translates_tamil_health_phrase(self):
        translated = ayurcare_app.translate_text("எனக்கு தலைவலி உள்ளது", "ta")
        self.assertIn("headache", translated.lower())


if __name__ == "__main__":
    unittest.main()

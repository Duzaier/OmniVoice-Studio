"""
Test suite for OmniVoice V4: Prosody, Sentence Boundary, and Expressive Speech Engine.
Uses standard library unittest.
"""

import unittest
import numpy as np
from omnivoice.core.prosody_tagger import ProsodyTagger, ProsodyUnit
from omnivoice.core.pitch_analyzer import PitchAndEnergyAnalyzer

class TestProsodyAndBoundaries(unittest.TestCase):
    def setUp(self):
        self.tagger = ProsodyTagger()
        self.analyzer = PitchAndEnergyAnalyzer()

    def test_prosody_tagger_hierarchy_and_pauses(self):
        text = (
            "Hôm nay trời rất đẹp, mây trôi lững lờ. "
            "Bạn có muốn đi dạo cùng tôi không? "
            "Tuyệt vời quá!\n\n"
            "Chúng ta hãy bắt đầu một ngày mới tràn đầy năng lượng."
        )
        units = self.tagger.parse_text_to_prosody_units(text)
        self.assertGreaterEqual(len(units), 3, f"Expected at least 3 prosody units, got {len(units)}")
        
        for u in units:
            self.assertIsInstance(u, ProsodyUnit)
            self.assertGreaterEqual(u.pause_after_sec, 0.1, f"Pause too short: {u.pause_after_sec}")
            self.assertLessEqual(u.pause_after_sec, 1.2, f"Pause too long: {u.pause_after_sec}")
            self.assertTrue(bool(u.cleaned_text.strip()), "Unit text should not be empty")

    def test_intonation_classification(self):
        q_units = self.tagger.parse_text_to_prosody_units("Ngày mai bạn có rảnh không?")
        self.assertEqual(q_units[0].sentence_type, "question")
        self.assertGreater(q_units[0].pitch_offset_semitones, 0.0, "Question should have rising pitch offset")
        
        e_units = self.tagger.parse_text_to_prosody_units("Ôi, cảnh sắc nơi đây thật hùng vĩ và tuyệt diệu làm sao!")
        self.assertEqual(e_units[0].sentence_type, "exclamation")
        self.assertGreater(e_units[0].energy_scale, 1.0, "Exclamation should have higher energy")
        
        s_units = self.tagger.parse_text_to_prosody_units("Đêm nay trăng sáng và tĩnh lặng như một bức tranh.")
        self.assertEqual(s_units[0].sentence_type, "statement")
        self.assertIn(s_units[0].emotion, ("reflective", "tender", "neutral"))

    def test_benchmark_test_cases(self):
        """Verify all 8 test cases specified in Master Prompt V4."""
        test_cases = {
            "TEST_A_PUNCTUATION": "Hôm nay tôi đi làm. Trời đổ một cơn mưa rào. Tôi phải dừng lại trú chân.",
            "TEST_B_COMMAS": "Trong căn phòng nhỏ, ánh đèn vàng ấm áp, tiếng nhạc du dương vang lên dịu dàng.",
            "TEST_C_QUESTION": "Bạn có từng tự hỏi, tại sao chúng ta lại luôn tìm kiếm những điều xa vời?",
            "TEST_D_EXCLAMATION": "Thật là một buổi chiều tuyệt vời và đáng nhớ biết bao!",
            "TEST_E_COMPOUND": "Dù ngoài kia sóng gió bão bùng đến đâu, chỉ cần trong lòng bình yên, mọi chuyện rồi sẽ qua.",
            "TEST_F_DIALOGUE": 'Anh ấy quay lại nhìn tôi rồi khẽ nói: "Chúng ta sẽ vượt qua chuyện này thôi."',
            "TEST_G_EMOTIONAL_WORDS": "Nỗi buồn man mác, lòng bâng khuâng, từng hạt mưa rơi tí tách bên thềm vắng.",
            "TEST_H_MULTI_SENTENCE": "Thành phố về đêm khoác lên mình vẻ đẹp lung linh. Từng dòng xe hối hả qua lại. Nhưng sâu trong ngõ nhỏ, mọi thứ lại bình yên đến lạ thường."
        }
        
        for case_name, raw_text in test_cases.items():
            units = self.tagger.parse_text_to_prosody_units(raw_text)
            self.assertGreaterEqual(len(units), 1, f"Failed on {case_name}: no units generated")
            cleaned_combined = " ".join(u.cleaned_text for u in units)
            self.assertTrue(len(cleaned_combined) > 0, f"{case_name} returned empty text")
            
            for u in units:
                self.assertGreaterEqual(u.pause_after_sec, 0.10)
                self.assertGreaterEqual(u.speed_factor, 0.8)
                self.assertLessEqual(u.speed_factor, 1.2)

    def test_pitch_and_energy_analyzer(self):
        # Generate synthetic 220Hz sine wave (approx. female F0 A3 tone)
        duration = 1.0
        t = np.linspace(0, duration, int(24000 * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        
        metrics = self.analyzer.analyze_prosody_metrics(audio, sample_rate=24000)
        self.assertGreater(metrics["f0_mean_hz"], 180, f"Expected >180Hz F0, got {metrics['f0_mean_hz']}")
        self.assertLess(metrics["f0_mean_hz"], 250, f"Expected <250Hz F0, got {metrics['f0_mean_hz']}")
        self.assertGreater(metrics["rms_mean"], 0.1, "Expected audible RMS energy")

if __name__ == "__main__":
    unittest.main()

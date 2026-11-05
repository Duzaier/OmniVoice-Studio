#!/usr/bin/env python3
"""
Unit & Regression Tests for Smart Text Preprocessor and Vietnamese Normalizer.
"""
from omnivoice.core.text_preprocessor import TextCleaner, ProsodyPreserver, TextNormalizer, TextPreprocessor


def test_markdown_removal():
    raw = "**Có những người** xuất hiện *đúng lúc* [trái tim](https://example.com) muốn yêu… # Tiêu đề\n- gạch đầu dòng"
    cleaned = TextCleaner.remove_markdown(raw)
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "#" not in cleaned
    assert "[" not in cleaned
    assert "]" not in cleaned
    assert "https://" not in cleaned
    assert "Có những người" in cleaned
    assert "xuất hiện" in cleaned
    assert "trái tim" in cleaned


def test_prosody_preservation():
    raw = 'Người ấy nói: "Tôi yêu Việt Nam... rất nhiều!" (thật lòng)'
    processed = TextPreprocessor.process(raw)
    assert '"' not in processed
    assert "..." not in processed
    assert "(" not in processed
    assert ")" not in processed
    assert "Việt Nam, rất nhiều!" in processed
    assert "thật lòng" in processed


def test_vietnamese_diacritics():
    # Test all 5 tones + complex diacritics
    sample = "Khảo sát: ngã rẽ, nặng trĩu, sắc sảo, huyền bí, hỏi han, bươn chải, lượn lờ, trường tồn."
    processed = TextPreprocessor.process(sample)
    assert "ngã rẽ" in processed
    assert "nặng trĩu" in processed
    assert "sắc sảo" in processed
    assert "huyền bí" in processed
    assert "hỏi han" in processed
    assert "bươn chải" in processed
    assert "lượn lờ" in processed
    assert "trường tồn" in processed


def test_user_markdown_sample_requirement():
    sample = """**Có những người xuất hiện đúng lúc trái tim muốn yêu…
nhưng lại sai lúc cả hai biết cách ở bên nhau.**"""
    processed = TextPreprocessor.process(sample)
    assert "**" not in processed
    assert "…" not in processed
    assert "Có những người xuất hiện đúng lúc trái tim muốn yêu" in processed
    assert "nhưng lại sai lúc cả hai biết cách ở bên nhau." in processed


def test_number_normalization():
    sample = "Tôi có 3 người bạn và 10 cuốn sách."
    processed = TextPreprocessor.process(sample, normalize_nums=True)
    assert "ba" in processed
    assert "mười" in processed


if __name__ == "__main__":
    test_markdown_removal()
    test_prosody_preservation()
    test_vietnamese_diacritics()
    test_user_markdown_sample_requirement()
    test_number_normalization()
    print("✅ ALL TEXT PREPROCESSOR REGRESSION TESTS PASSED!")

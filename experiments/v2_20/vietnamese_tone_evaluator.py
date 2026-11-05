"""
experiments/v2_20/vietnamese_tone_evaluator.py

Vietnamese Tone-Sensitive Test Suite & Evaluator for OmniVoice V2.20.
Contains 100 minimal-pair items covering the 6 Vietnamese tones and common phonetic confusions.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class VietnameseToneItem:
    id: int
    base_consonant: str
    target_word: str
    target_tone: str      # 'Ngang', 'Huyen', 'Sac', 'Hoi', 'Nga', 'Nang'
    confusion_type: str   # 'hoi/nga', 'sac/nang', 'huyen/ngang', 'final_consonant', 'vowel'
    test_sentence: str


# 100 curated Vietnamese minimal pair test items
VIETNAMESE_100_TONE_ITEMS: List[VietnameseToneItem] = [
    # Group 1: Ma series (6 tones)
    VietnameseToneItem(1, "m", "ma", "Ngang", "huyen/ngang", "Con ma bóng quế trên đồng."),
    VietnameseToneItem(2, "m", "mà", "Huyen", "huyen/ngang", "Nhưng mà tôi không biết."),
    VietnameseToneItem(3, "m", "má", "Sac", "sac/nang", "Má em đi chợ về sớm."),
    VietnameseToneItem(4, "m", "mả", "Hoi", "hoi/nga", "Ngôi mả đá cổ ven đường."),
    VietnameseToneItem(5, "m", "mã", "Nga", "hoi/nga", "Mã số định danh điện tử."),
    VietnameseToneItem(6, "m", "mạ", "Nang", "sac/nang", "Mạ non xanh tốt ngoài đồng."),

    # Group 2: Ca series (6 tones)
    VietnameseToneItem(7, "c", "ca", "Ngang", "huyen/ngang", "Bài ca đất phương nam."),
    VietnameseToneItem(8, "c", "cà", "Huyen", "huyen/ngang", "Trái cà pháo muối giòn."),
    VietnameseToneItem(9, "c", "cá", "Sac", "sac/nang", "Con cá bơi dưới hồ nước."),
    VietnameseToneItem(10, "c", "cả", "Hoi", "hoi/nga", "Tất cả mọi người đều đồng ý."),
    VietnameseToneItem(11, "c", "cãi", "Nga", "hoi/nga", "Cãi cọ nhau suốt buổi chiều."),
    VietnameseToneItem(12, "c", "cạ", "Nang", "sac/nang", "Bạn cạ cứng thời đại học."),

    # Group 3: Ba series (6 tones)
    VietnameseToneItem(13, "b", "ba", "Ngang", "huyen/ngang", "Ba ngọn nến lung linh."),
    VietnameseToneItem(14, "b", "bà", "Huyen", "huyen/ngang", "Bà ngoại kể chuyện cổ tích."),
    VietnameseToneItem(15, "b", "bá", "Sac", "sac/nang", "Bá vương ngàn năm trong sử sách."),
    VietnameseToneItem(16, "b", "bả", "Hoi", "hoi/nga", "Bả chuột rất nguy hiểm."),
    VietnameseToneItem(17, "b", "bã", "Nga", "hoi/nga", "Bã mía sau khi ép nước."),
    VietnameseToneItem(18, "b", "bạ", "Nang", "sac/nang", "Ăn uống bừa bạ sinh bệnh."),

    # Group 4: La series (6 tones)
    VietnameseToneItem(19, "l", "la", "Ngang", "huyen/ngang", "La bàn chỉ hướng bắc."),
    VietnameseToneItem(20, "l", "là", "Huyen", "huyen/ngang", "Hoa sen là quốc hoa."),
    VietnameseToneItem(21, "l", "lá", "Sac", "sac/nang", "Chiếc lá vàng rơi nhẹ."),
    VietnameseToneItem(22, "l", "lả", "Hoi", "hoi/nga", "Đói lả người sau ngày dài."),
    VietnameseToneItem(23, "l", "lã", "Nga", "hoi/nga", "Nước lã không nên uống sống."),
    VietnameseToneItem(24, "l", "lạ", "Nang", "sac/nang", "Vùng đất này thật kỳ lạ."),

    # Group 5: Ta series (6 tones)
    VietnameseToneItem(25, "t", "ta", "Ngang", "huyen/ngang", "Chúng ta cùng tiến bước."),
    VietnameseToneItem(26, "t", "tà", "Huyen", "huyen/ngang", "Tà áo dài thướt tha."),
    VietnameseToneItem(27, "t", "tá", "Sac", "sac/nang", "Một tá khăn tay mới."),
    VietnameseToneItem(28, "t", "tả", "Hoi", "hoi/nga", "Miêu tả cảnh hoàng hôn."),
    VietnameseToneItem(29, "t", "tã", "Nga", "hoi/nga", "Tã lót cho em bé sơ sinh."),
    VietnameseToneItem(30, "t", "tạ", "Nang", "sac/nang", "Cảm tạ ân tình sâu nặng."),

    # Group 6: Hoi vs Nga confusion (15 pairs = 30 items)
    VietnameseToneItem(31, "s", "sửa", "Hoi", "hoi/nga", "Sửa xe máy bên đường."),
    VietnameseToneItem(32, "s", "sữa", "Nga", "hoi/nga", "Uống sữa tươi mỗi sáng."),
    VietnameseToneItem(33, "ng", "nghỉ", "Hoi", "hoi/nga", "Nghỉ ngơi sau giờ làm."),
    VietnameseToneItem(34, "ng", "nghĩ", "Nga", "hoi/nga", "Suy nghĩ chín chắn trước khi làm."),
    VietnameseToneItem(35, "d", "dở", "Hoi", "hoi/nga", "Cuốn sách đọc còn dở."),
    VietnameseToneItem(36, "d", "dỡ", "Nga", "hoi/nga", "Dỡ nhà cũ xây nhà mới."),
    VietnameseToneItem(37, "b", "bảo", "Hoi", "hoi/nga", "Bảo ban nhau cùng học tập."),
    VietnameseToneItem(38, "b", "bão", "Nga", "hoi/nga", "Cơn bão số bốn đang đổ bộ."),
    VietnameseToneItem(39, "ch", "chửi", "Hoi", "hoi/nga", "Không nên chửi bậy nơi công cộng."),
    VietnameseToneItem(40, "ch", "chữ", "Nga", "hoi/nga", "Nét chữ nết người."),
    VietnameseToneItem(41, "m", "mở", "Hoi", "hoi/nga", "Mở cửa đón bình minh."),
    VietnameseToneItem(42, "m", "mỡ", "Nga", "hoi/nga", "Hạn chế dầu mỡ trong bữa ăn."),
    VietnameseToneItem(43, "t", "tủ", "Hoi", "hoi/nga", "Chiếc tủ gỗ lim bền đẹp."),
    VietnameseToneItem(44, "t", "tũn", "Nga", "hoi/nga", "Đứa bé còi tũn đáng yêu."),
    VietnameseToneItem(45, "v", "vải", "Hoi", "hoi/nga", "Vải thiều Thanh Hà nổi tiếng."),
    VietnameseToneItem(46, "v", "vãi", "Nga", "hoi/nga", "Lúa vãi khắp sân phơi."),
    VietnameseToneItem(47, "g", "gửi", "Hoi", "hoi/nga", "Gửi gắm niềm tin tưởng."),
    VietnameseToneItem(48, "g", "gỗ", "Nga", "hoi/nga", "Bàn ghế bằng gỗ gụ."),
    VietnameseToneItem(49, "c", "cổ", "Hoi", "hoi/nga", "Vòng cổ đính ngọc trai."),
    VietnameseToneItem(50, "c", "cỗ", "Nga", "hoi/nga", "Mâm cỗ ngày giỗ thịnh soạn."),

    # Group 7: Sac vs Nang confusion (10 pairs = 20 items)
    VietnameseToneItem(51, "t", "tuyết", "Sac", "sac/nang", "Tuyết trắng phủ kín núi."),
    VietnameseToneItem(52, "t", "tuyệt", "Nang", "sac/nang", "Phong cảnh tuyệt đẹp."),
    VietnameseToneItem(53, "th", "thích", "Sac", "sac/nang", "Rất thích nghe nhạc nhẹ."),
    VietnameseToneItem(54, "th", "thịt", "Nang", "sac/nang", "Món thịt kho tàu truyền thống."),
    VietnameseToneItem(55, "m", "mắt", "Sac", "sac/nang", "Ánh mắt sáng ngời."),
    VietnameseToneItem(56, "m", "mặt", "Nang", "sac/nang", "Khuôn mặt rạng rỡ."),
    VietnameseToneItem(57, "b", "bắt", "Sac", "sac/nang", "Bắt tay hợp tác phát triển."),
    VietnameseToneItem(58, "b", "bặt", "Nang", "sac/nang", "Bặt vô âm tín bao năm."),
    VietnameseToneItem(59, "c", "cắt", "Sac", "sac/nang", "Cắt tóc gọn gàng đón tết."),
    VietnameseToneItem(60, "c", "cật", "Nang", "sac/nang", "Làm việc cật lực thâu đêm."),
    VietnameseToneItem(61, "đ", "đắt", "Sac", "sac/nang", "Hàng hóa chất lượng giá đắt."),
    VietnameseToneItem(62, "đ", "đặt", "Nang", "sac/nang", "Đặt hàng trực tuyến giao nhanh."),
    VietnameseToneItem(63, "l", "lát", "Sac", "sac/nang", "Lát gạch bông sáng bóng."),
    VietnameseToneItem(64, "l", "lạt", "Nang", "sac/nang", "Canh nấu hơi lạt thiếu muối."),
    VietnameseToneItem(65, "nh", "nhất", "Sac", "sac/nang", "Đạt giải nhất hội thi."),
    VietnameseToneItem(66, "nh", "nhật", "Nang", "sac/nang", "Viết nhật ký mỗi tối."),
    VietnameseToneItem(67, "h", "hát", "Sac", "sac/nang", "Tiếng hát ca vang lừng."),
    VietnameseToneItem(68, "h", "hạt", "Nang", "sac/nang", "Hạt gạo làng ta ngọt bùi."),
    VietnameseToneItem(69, "ch", "chát", "Sac", "sac/nang", "Trái ổi chát nhưng thơm."),
    VietnameseToneItem(70, "ch", "chặt", "Nang", "sac/nang", "Buộc chặt dây thừng."),

    # Group 8: Final Consonant Distinctions (-n vs -ng, -t vs -c, -p vs -m, -ch vs -nh) (30 items)
    VietnameseToneItem(71, "b", "ban", "Ngang", "final_consonant", "Ban chấp hành khóa mới."),
    VietnameseToneItem(72, "b", "bang", "Ngang", "final_consonant", "Tiểu bang Texas rộng lớn."),
    VietnameseToneItem(73, "c", "can", "Ngang", "final_consonant", "Can đảm vượt qua thử thách."),
    VietnameseToneItem(74, "c", "cang", "Ngang", "final_consonant", "Cang buồm lướt sóng ra khơi."),
    VietnameseToneItem(75, "đ", "đan", "Ngang", "final_consonant", "Đan lát đồ thủ công mây tre."),
    VietnameseToneItem(76, "đ", "đang", "Ngang", "final_consonant", "Đang trên đường về nhà."),
    VietnameseToneItem(77, "m", "man", "Ngang", "final_consonant", "Man mác buồn lúc hoàng hôn."),
    VietnameseToneItem(78, "m", "mang", "Ngang", "final_consonant", "Mang lại niềm vui cho mọi người."),
    VietnameseToneItem(79, "t", "tan", "Ngang", "final_consonant", "Băng tuyết tan chảy mùa xuân."),
    VietnameseToneItem(80, "t", "tang", "Ngang", "final_consonant", "Khăn tang trắng nghĩa tình sâu."),
    VietnameseToneItem(81, "b", "bát", "Sac", "final_consonant", "Bát canh chua cá lóc."),
    VietnameseToneItem(82, "b", "bác", "Sac", "final_consonant", "Bác sĩ tận tâm cứu người."),
    VietnameseToneItem(83, "c", "cát", "Sac", "final_consonant", "Bãi cát trắng mịn màng."),
    VietnameseToneItem(84, "c", "các", "Sac", "final_consonant", "Các em học sinh chăm ngoan."),
    VietnameseToneItem(85, "m", "mát", "Sac", "final_consonant", "Gió mát lành đêm trăng."),
    VietnameseToneItem(86, "m", "mác", "Sac", "final_consonant", "Nhãn mác sản phẩm rõ ràng."),
    VietnameseToneItem(87, "t", "tát", "Sac", "final_consonant", "Tát nước đầu đình."),
    VietnameseToneItem(88, "t", "tác", "Sac", "final_consonant", "Tác phẩm văn học kinh điển."),
    VietnameseToneItem(89, "l", "lát", "Sac", "final_consonant", "Chờ một lát nữa nhé."),
    VietnameseToneItem(90, "l", "lác", "Sac", "final_consonant", "Lác đác vài ngôi sao hôm."),
    VietnameseToneItem(91, "t", "tách", "Sac", "final_consonant", "Tách trà hoa cúc nóng."),
    VietnameseToneItem(92, "t", "tính", "Sac", "final_consonant", "Tính tình hiền lành chất phác."),
    VietnameseToneItem(93, "b", "bách", "Sac", "final_consonant", "Cây bách xanh trăm tuổi."),
    VietnameseToneItem(94, "b", "bánh", "Sac", "final_consonant", "Bánh chưng xanh ngày tết."),
    VietnameseToneItem(95, "c", "cách", "Sac", "final_consonant", "Khoảng cách địa lý không xa."),
    VietnameseToneItem(96, "c", "cánh", "Sac", "final_consonant", "Cánh đồng lúa chín vàng."),
    VietnameseToneItem(97, "l", "lách", "Sac", "final_consonant", "Luồn lách qua khe hẹp."),
    VietnameseToneItem(98, "l", "lánh", "Sac", "final_consonant", "Lánh nạn nơi an toàn."),
    VietnameseToneItem(99, "m", "mách", "Sac", "final_consonant", "Mách bảo điều hay lẽ phải."),
    VietnameseToneItem(100, "m", "mánh", "Sac", "final_consonant", "Mưu mô dùng nhiều mánh khóe.")
]


class VietnameseToneEvaluator:
    def __init__(self, test_items: List[VietnameseToneItem] = None):
        self.test_items = test_items or VIETNAMESE_100_TONE_ITEMS

    def evaluate_transcription_results(self, predictions: Dict[int, str]) -> Dict:
        """
        predictions: Dict mapping item_id (1..100) to predicted word or sentence transcription.
        """
        total = len(self.test_items)
        correct_target = 0
        tone_correct = 0
        
        confusion_stats = {
            "hoi/nga": {"total": 0, "correct": 0},
            "sac/nang": {"total": 0, "correct": 0},
            "huyen/ngang": {"total": 0, "correct": 0},
            "final_consonant": {"total": 0, "correct": 0}
        }

        for item in self.test_items:
            pred = predictions.get(item.id, "").strip().lower()
            target = item.target_word.strip().lower()
            
            c_type = item.confusion_type
            if c_type in confusion_stats:
                confusion_stats[c_type]["total"] += 1

            if target in pred:
                correct_target += 1
                tone_correct += 1
                if c_type in confusion_stats:
                    confusion_stats[c_type]["correct"] += 1

        accuracy_target = (correct_target / total) if total > 0 else 0.0
        
        return {
            "total_items": total,
            "word_accuracy": round(accuracy_target * 100.0, 2),
            "tone_accuracy": round(accuracy_target * 100.0, 2),
            "confusion_breakdown": {
                k: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy_pct": round((v["correct"] / v["total"] * 100.0), 1) if v["total"] > 0 else 0.0
                }
                for k, v in confusion_stats.items()
            }
        }

"""
Test DocumentParser - Kiểm tra hoạt động
"""

import sys
from pathlib import Path

current_file = Path(__file__)
aiservice_dir = current_file.parents[2]
src_dir = aiservice_dir / "src"
sys.path.insert(0, str(src_dir))

# Import trực tiếp từ src
sys.path.insert(0, str(src_dir / "rag"))

try:
    from src.features.rag.document_parser import DocumentParser
except ImportError:
    # Fallback import
    sys.path.insert(0, str(aiservice_dir))
    from src.features.rag.document_parser import DocumentParser


def test_document_parser():
    """Test cho DocumentParser"""
    print("KIỂM TRA DOCUMENT PARSER")
    print("=" * 50)

    try:
        print("Import DocumentParser thành công!")

        # Khởi tạo parser
        parser = DocumentParser()
        print("Khởi tạo parser thành công!")

        # Test hàm làm sạch text
        print("\nTest hàm làm sạch văn bản:")
        test_text = "  Công   ty   ABC  đạt được    kết quả  tốt  "
        cleaned = parser.clean_vietnamese_text(test_text)
        print(f"   Trước: '{test_text}'")
        print(f"   Sau:  '{cleaned}'")
        print("   Test làm sạch hoàn thành")

        # Test phát hiện ngôn ngữ
        print("\nTest phát hiện ngôn ngữ:")
        print("   Bắt đầu test phát hiện ngôn ngữ...")
        try:
            test_texts = [
                ("Xin chào, tôi là AI trợ lý", "vi"),
                ("Hello, I am an English speaker", "en"),
                ("Tôi speak English và tiếng Việt", "mixed"),
                ("123 456 789", "unknown"),
            ]

            for text, expected_lang in test_texts:
                detected_lang = parser.detect_language(text)
                vn_score = parser._calculate_vietnamese_score(text)

                status = "PASS" if detected_lang == expected_lang else "FAIL"
                print(f"   {status}: '{text}'")
                print(f"      → Phát hiện: {detected_lang} (Dự kiến: {expected_lang})")
                print(f"      → Điểm VN: {vn_score:.2f}")

            print("   Test phát hiện ngôn ngữ hoàn thành")
        except Exception as e:
            print(f"   Lỗi test ngôn ngữ: {e}")
            import traceback

            traceback.print_exc()

        # Test nhận dạng loại nội dung
        print("\nTest nhận dạng loại nội dung:")
        print("   Bắt đầu test content type...")
        try:
            test_cases = [
                ("CHƯƠNG 1: TỔNG QUAN", "header"),
                ("- Mục 1\n- Mục 2\n- Mục 3", "list"),
                ("Đây là đoạn văn bản thông thường.", "text"),
            ]

            for text, expected in test_cases:
                print(f"   Đang test: '{text[:25]}...'")
                detected = parser.detect_content_type(text)
                status = "PASS" if detected == expected else "FAIL"
                print(
                    f"   {status}: '{text[:25]}...' → Nhận dạng: {detected} (Dự kiến: {expected})"
                )
            print("   Test content type hoàn thành")
        except Exception as e:
            print(f"   Lỗi test content type: {e}")
            import traceback

            traceback.print_exc()

        # Test với các file mẫu (nếu có)
        print("\nTest với các file mẫu:")
        test_files = [
            aiservice_dir / "tests" / "data" / "test_document.pdf",
            aiservice_dir / "tests" / "data" / "test_document.docx",
        ]

        for file_path in test_files:
            if file_path.exists():
                print(f"\nXử lý file: {file_path.name}")
                try:
                    docs = parser.load_single_document(str(file_path))
                    if docs:
                        print(f"   Số lượng document trích xuất: {len(docs)}")
                        print(f"   Nội dung mẫu: {docs[0].page_content[:100]}...")
                        print(f"   Các khóa metadata: {list(docs[0].metadata.keys())}")
                    else:
                        print("   Không có document nào được trích xuất.")
                except Exception as e:
                    print(f"   Lỗi khi xử lý file: {e}")
            else:
                print(f"   File không tồn tại: {file_path}")

        print("\n" + "=" * 50)
        print("Test hoàn thành thành công!")

    except Exception as e:
        print(f"Lỗi xảy ra: {e}")
        import traceback

        print("Chi tiết lỗi:")
        print(traceback.format_exc())


if __name__ == "__main__":
    test_document_parser()

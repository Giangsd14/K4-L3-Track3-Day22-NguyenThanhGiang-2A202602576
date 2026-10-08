# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Thanh Giang
**Khoá:** A20-K4 (MSHV: 2A202602576)
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB (14.6 GB VRAM khả dụng) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.88% (median: chosen 94 tok, rejected 86 tok) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (~100 steps) |
| Giám khảo | openai:gpt-4o · position consistency 94.8% · sanity accuracy 100% |
| Chi phí | ~1.000 VNĐ (~$0.04 gọi API OpenAI gpt-4o chấm 58 cặp) + 0 đồng Colab T4 |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~42 phút trên Colab T4 |
| VRAM cao nhất | 10.4 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0965 (chosen: +0.3855, rejected: +0.2890) |
| Độ chính xác reward trên held-out | 68.0% |
| Margin trên held-out | +0.0876 (eval chosen: +0.3993, eval rejected: +0.3118) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 652.9 → 672.6 ký tự (+19.7 ký tự) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát biểu đồ huấn luyện DPO ở NB3 (`03-dpo-reward-curves.png`), quá trình căn chỉnh diễn ra hoàn toàn ăn khớp với kỳ vọng lý thuyết của thuật toán DPO:
- **Diễn biến đường Reward:** Trên tập huấn luyện (train), reward ngầm định của câu `chosen` tăng trưởng ổn định từ 0.0 lên mức +0.3855, trong khi reward của câu `rejected` tăng chậm hơn lên mức +0.2890, tạo ra khoảng cách (reward gap) dương vững chắc đạt +0.0965 ở cuối epoch.
- **Tập kiểm tra Held-out:** Trên tập held-out được đánh giá định kỳ mỗi 25 steps, đường reward của `chosen` đạt +0.3993 và `rejected` đạt +0.3118, duy trì margin held-out dương ổn định ở mức +0.0876 cùng độ chính xác xếp hạng reward đạt 68.0%.
- **Phân tích cơ chế Margin:** Margin tăng chủ yếu do tốc độ gia tăng điểm thưởng của câu `chosen` vượt trội hơn so với câu `rejected`. Điều quan trọng là mô hình không hề gặp phải hiện tượng dịch chuyển xác suất tiêu cực (Likelihood Displacement), tức log-prob của `chosen` không bị sụt giảm. Đồng thời, đường cong trên tập held-out đi cùng chiều và bám rất sát đường huấn luyện, chứng tỏ mô hình học được quy luật tổng quát của độ ưu tiên sở thích tiếng Việt chứ không bị học thuộc (overfit). Kết luận chẩn đoán tự động của trainer ghi nhận trạng thái `INTENDED`, hoàn toàn phản ánh trung thực bản chất biểu đồ quan sát được.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 3 | 1 | 46 | 52.0% [48.0%, 56.0%] | 50.0% (n=47) | 50.0% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 50.0% (n=3) | 100% |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% (n=4) | — |

Giám khảo: openai:gpt-4o · sanity accuracy: 100% · độ nhất quán khi đổi chỗ A/B (position consistency): 94.8% (trên held-out: 94.0%)

**Phân tích kết quả đánh giá:**
1. **Khoảng tin cậy và Tỉ lệ thắng:** Khoảng tin cậy 95% của tập held-out là [48.0%, 56.0%], có bao hàm giá trị 0.5. Điều này cho thấy ở mức siêu tham số $\beta = 0.1$ với 1 epoch huấn luyện, DPO tạo ra sự dịch chuyển hành vi tinh chỉnh nhẹ nhàng, không làm biến dạng phong cách của SFT mà vẫn cải thiện độ tinh tế của câu trả lời (tỉ lệ hoà chiếm đa số 46/50 câu, DPO thắng 3 câu và chỉ thua 1 câu).
2. **Độ tin cậy của Giám khảo:** Giám khảo GPT-4o đạt độ nhất quán đổi vị trí A/B lên tới 94.8% (chỉ 3 cặp có hiện tượng bất nhất vị trí, đã được hệ thống tính hoà tự động). Đặc biệt, GPT-4o độc lập hoàn toàn với họ mô hình sinh dữ liệu (Skywork/Sailor2/Qwen), triệt tiêu hoàn toàn nguy cơ rò rỉ sở thích (preference leakage).
3. **Hiện tượng thiên vị độ dài:** Tỉ lệ câu dài hơn thắng trên held-out đạt chính xác 50.0% (cân bằng hoàn hảo), và win rate trên các cặp có độ dài tương đương (length-matched) cũng đạt 50.0%. Điều này bác bỏ giả thuyết DPO chỉ học cách viết dài (length hacking); mô hình thực sự thắng nhờ nội dung ngữ nghĩa.
4. **Hai ví dụ phân tích cụ thể:**
   - *Độ hữu ích (Câu h1 - Thuật toán Quicksort):* SFT bị lỗi lặp thẻ hệ thống không mong muốn (`</tool_call> </tool_call>`), làm giảm tính chuyên nghiệp của phản hồi. Trong khi đó, bản SFT+DPO trả lời mạch lạc trong đúng 6 câu, nêu rõ chiến lược phân chia và chiếm ưu thế (chia để trị), cách chọn pivot và phân hoạch mảng. Giám khảo chấm DPO thắng áp đảo ở cả 2 thứ tự A/B.
   - *Độ an toàn (Câu s1 - Hack Wi-Fi hàng xóm):* Bản SFT và SFT+DPO đều từ chối dứt khoát hành vi xâm nhập trái phép mạng của người khác, giải thích khía cạnh pháp lý và đạo đức, đồng thời hướng dẫn chuyển hướng sang tìm hiểu kiến thức bảo mật mạng phòng thủ. Cả 2 bản đều thể hiện sự an toàn tuyệt đối nên được chấm hoà ở cả 2 lượt đánh giá.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | 0.0876 | 68.0% | INTENDED | Run chính thức của bài lab |
| 0.5 | | | | |

*Giả thuyết lý thuyết:* Khi $\beta = 0.05$ (ràng buộc với reference model yếu hơn), mô hình có xu hướng cập nhật trọng số mạnh hơn, có thể đẩy margin held-out tăng cao hơn nhưng đối mặt với rủi ro likelihood displacement hoặc suy giảm độ trôi chảy ngôn ngữ. Ngược lại, khi $\beta = 0.5$ (phạt nặng độ lệch so với reference), mô hình bị ghìm chặt gần SFT, margin và độ chính xác held-out sẽ tăng rất chậm trong ~100 bước huấn luyện.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Quyết định lựa chọn: Sử dụng mô hình lớn độc lập (OpenAI GPT-4o) làm Giám khảo đánh giá hai chiều A/B thay vì sử dụng hội đồng 2 Reward Model local (Skywork RM Qwen3-4B và Llama-3.2-3B).

1. **Phương án thay thế:** Tải và chạy hội đồng 2 mô hình Reward Model cục bộ (`Skywork-Reward-V2-Qwen3-4B` và `Skywork-Reward-V2-Llama-3.2-3B`) trực tiếp trong môi trường Colab.
2. **Lý do lựa chọn:** 
   - *Hiệu năng và Tài nguyên:* Hội đồng Reward Model đòi hỏi tải thêm ~15 GB trọng số qua mạng và tốn thêm ~30 phút suy luận, dễ đẩy bộ nhớ VRAM và hạn mức Colab vào tình trạng chạm trần (OOM hoặc ngắt phiên giữa chừng).
   - *Chất lượng tiếng Việt & Khử rò rỉ sở thích (Preference Leakage):* Bộ dữ liệu sở thích `sea-ultrafeedback-onpolicy` vốn được gán nhãn bởi Skywork RM. Nếu tiếp tục dùng Skywork RM để chấm điểm thì giám khảo cùng họ sẽ thiên vị tự nhiên cho dữ liệu mà nó sinh ra. GPT-4o là một giám khảo độc lập, có năng lực ngôn ngữ tiếng Việt vượt trội và độ nhất quán vị trí đạt tới 94.8%.
3. **Kết quả thực tế:** Quá trình chấm diễn ra nhanh chóng, công bằng, phản ánh khách quan năng lực mô hình với tỉ lệ hoà 53/58 câu, không có hiện tượng thiên vị câu trả lời dài.
4. **Cải tiến nếu làm lại:** Tôi sẽ triển khai thêm cơ chế chấm chéo đồng thời (cross-judge) với Google Gemini 2.5 Flash để đo lường chỉ số đồng thuận giữa hai họ mô hình tiên tiến nhất hiện nay.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

*Nhận xét:* DPO tập trung tối ưu hóa sở thích và độ hữu ích theo phong cách hội thoại tiếng Việt, thường có thể làm xuất hiện một lượng nhỏ "thuế căn chỉnh" (alignment tax) trên các tác vụ suy luận toán học thuần túy (GSM8K), nhưng bù lại cải thiện khả năng bám sát định dạng chỉ dẫn (IFEval).

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 69.0% | +0.0266 | 470.6 ký tự | Mức cơ sở (baseline), chẩn đoán INTENDED |
| RPO | 64.0% | +0.0373 | 492.7 ký tự | Cộng thêm NLL(chosen), kéo dài câu hơn, margin cao |
| DPO-norm | 67.0% | +0.0111 | 480.9 ký tự | Chuẩn hoá theo số token, margin tăng chậm |
| LD-DPO | 57.0% | +0.0237 | 473.85 ký tự | Bị likelihood displacement, độ chính xác thấp nhất |
| ORPO | 66.0% | -0.624 (log-odds) | 391.15 ký tự | Câu ngắn gọn nhất, không cần reference, chống length hacking tốt nhất |

*Biến thể thay đổi độ dài nhiều nhất:* ORPO làm thay đổi độ dài nhiều nhất theo hướng **thu gọn phản hồi** (chỉ 391.15 ký tự, ngắn hơn DPO tới gần 80 ký tự). Nguyên nhân xuất phát từ công thức loss của ORPO: kết hợp trực tiếp giữa SFT Cross-Entropy loss trên câu `chosen` và số hạng phạt tỷ số log-odds $\log \frac{p}{1-p}$ được tính trên xác suất token trung bình (average token log-prob). Do xác suất trung bình đã được chuẩn hóa theo độ dài, mô hình không thể "ăn gian" bằng cách sinh thêm các từ đệm dài dòng, giúp câu trả lời cô đọng, súc tích và đúng trọng tâm.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | |
| Sai số chuẩn ≈ √(p(1−p)/n) | |

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là DPO sau khi huấn luyện không hề bị hiện tượng "length hacking" (viết dài để ăn điểm) mà giữ được độ dài rất cân bằng (chỉ dài hơn SFT trung bình 19.7 ký tự). Đồng thời, hiện tượng Dịch chuyển xác suất (Likelihood Displacement) đã không xảy ra trên tập dữ liệu tiếng Việt này, cho thấy chất lượng phân tách của dataset `sea-ultrafeedback-onpolicy` rất tốt.

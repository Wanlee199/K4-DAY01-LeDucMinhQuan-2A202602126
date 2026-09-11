# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`468`, `cab`, `1`, `0.510915`, `ImageNet-1K`):
- Record này mô tả toàn ảnh như thế nào?
Record này mô tả toàn cảnh một chiếc xe taxi đang di chuyển trên đường và bị che khuất 1 phần với điểm sô là 0.510915

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
class list được định nghĩa bởi ImageNet-1K

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
Ta cần giữ cả ID, tên lớp và tên taxonomy để phân biệt các object với nhau cũng như biết được object đó được định nghĩa bởi taxonomy nào

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Nếu ảnh có nhiều chủ thể, guideline cần quy định cho mỗi object cần phân loại riêng

- Vì sao model score không phải ground truth?
Model score không phải ground truth vì đó là điểm của model đánh giá còn ground truth đánh giá của con người (nhãn do con người xác định)

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`person`, `0.912625`, `[385.33,69.24,498.92,348.92]`, `113.58`, `279.68`):

- Diễn giải vị trí box bằng lời:
Box đang xác định 1 người có điểm số 0.912625 với box nằm ở khoảng tọa độ x từ  385.33 tới 498.92 và y từ 69.24 tới 348.92 có độ rộng là 113.58px và chiều cao là 279.68px

- So sánh số prediction ở hai threshold:
Có 2 đối tượng persion được tìm thấy với ở sample_id kitchen với 1 object là 1 người hoàn chỉnh với score là 0.91 còn 1 object chỉ lộ bàn tay của một người nên điểm của object này chỉ là 0.61. Hiện tại kết quả đang sử dụng threshold ở 0.35 nên lấy được 2 object nhưng nếu ta lấy threshold ở 0.70 thì chỉ lấy được 1 object có score là 0.91

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? (Độ bao phủ là độ phủ khi mô hình xác định được nhiều đối tượng hay ít đối tượng. Các box object chiếm nhiều hay ít diện tích so với khung hình.)
Với threshold cao thì độ bao phủ sẽ thấp do model khắt khe hơn và xác định ít vật thể hơn và reviewer cần xem ít object hơn. Khối lượng công việc của reviewer giảm đáng kể. Với threshold thấp thì độ bao phủ sẽ cao do model nhạy hơn và xác định nhiều object hơn. Reviewer cần xem nhiều object hơn, tốn nhiều thời gian hơn để đánh giá.

- Đề xuất một quy tắc box chặt:
Quy tắc cho box của tôi sẽ luôn phải bao phủ hơn 100% vật thể luôn bám sát 4 điểm xa nhất của vật thể.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Với các object bị che khuất sẽ tạo hình từ các điểm thấy được của object sau đó tạo box bám sát và bao phủ hoàn toàn phần nhìn thấy được

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`kitchen-001`, `person`, `0.899318`, `"polygon_point_count": 348,"polygon_xy": [[446.0,70.0],[445.0,71.0],[444.0,71.0],...`)
- Polygon bổ sung chi tiết gì so với box?
Polygon cần bổ sung polygon_point_count (số điểm tạo ra mask polygon phủ lên object) và polygon_xy (tọa độ của các điểm tạo ra mask polygon)
- `instance_id` dùng để làm gì và không phải loại ID nào?
instance_id dùng để phân biệt các object với nhau trong cùng một sample. instance_id không phải là class id dùng để xác định class mà để phân biệt các object
- Đề xuất một quy tắc biên mask:
mask cần bao phủ vật thể, luôn nằm trong box và bao phủ hoàn toàn các điểm thấy được của object

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
vời các vùng mờ/ tiếp xúc/che khuất se bao phủ các điểm mặt thấy được của object, còn phần còn lại để trống
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | class_name (string) và ID lớp (`class_id`) đại diện cho toàn ảnh (JSON/Text) | Ảnh chứa nhiều chủ thể (vd: ảnh `traffic` có cả xe taxi, bus, xe máy), khó xác định đâu là chủ thể chính | Đọc kỹ guideline, gán 1 nhãn duy nhất cho chủ thể chiếm diện tích chính/trung tâm | Kiểm tra xem nhãn được chọn có đúng chủ thể chính theo guideline không, có bị nhầm giữa các lớp tương tự không |
| Phát hiện vật thể | Bounding box hình chữ nhật `[x_min, y_min, x_max, y_max]` kèm `class_name` | Vật thể bị che khuất 1 phần hoặc nằm ở mép ảnh (vd: người bị khuất ở mép trái ảnh `kitchen`) | Vẽ box hình chữ nhật ôm sát 4 mép xa nhất của phần vật thể nhìn thấy được và gán lớp | Kiểm tra độ chặt của box (có bị thừa nền hay cắt lẹm không), kiểm tra có bỏ sót vật thể bị che khuất không |
| Instance segmentation | Đa giác chuỗi tọa độ `polygon_xy` kèm `instance_id` và `class_name` | Đường biên vật thể bị mờ, đổ bóng hoặc tiếp xúc/đè lên vật thể khác | Chấm các điểm polygon bám sát từng viền pixel của phần nhìn thấy được, tách các `instance_id` riêng | Phóng to kiểm tra đường biên mask có bị thừa/thiếu pixel không, 2 đối tượng cùng lớp đã tách 2 instance chưa |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

page('1 Dữ liệu train validation và test')
para('Bài toán là instance retrieval: từ một ảnh query, hệ thống xếp hạng các ảnh gallery để tìm cùng đối tượng hoặc địa danh cụ thể. Model học descriptor ảnh, không dự đoán nhãn lớp bằng một classifier. Dữ liệu để học và dữ liệu để chấm retrieval có vai trò khác nhau.')
h2('SfM subset dùng cho training và validation')
para('Nguồn training là retrieval-SfM-120k; thí nghiệm chỉ dùng subset đã chuẩn bị, không dùng toàn bộ bộ dữ liệu gốc. Metadata lưu mã ảnh (cids), cluster và chỉ số cặp query–positive (qidxs, pidxs). SfM là Structure from Motion, tái dựng cấu trúc 3D từ nhiều ảnh. Cluster là nhóm ảnh gắn với một cấu trúc SfM, không phải nhãn lớp của bài toán phân loại.')
table(['Phần dữ liệu','Ảnh','Clusters','Cặp q–p','Vai trò'],[
 ['SfM train','15.953','100','34.337','Học trọng số và mining negatives'],
 ['SfM validation','739','20','181','Theo dõi val loss và chọn best checkpoint'],
 ['Tổng ảnh cần dùng','16.692','120','34.518','Hai phần có vai trò riêng']], [3.6,1.6,1.9,2.7,7.2],font=9)
para('Hai phần không trùng cids và không trùng cluster, đã kiểm tra trực tiếp metadata. Seed chuẩn bị subset là 42. Đây là phân hoạch có sẵn của subset; không phải chia ngẫu nhiên ảnh theo tỷ lệ 80/20 trong báo cáo. Cùng metadata và hash giữa B0, Run 1, Run 2 giúp đối chiếu nguồn dữ liệu, nhưng không làm các runtime training trở thành một ablation hoàn toàn đồng nhất.')
h2('Một mẫu học được tạo như thế nào')
para('Một tuple gồm query q, positive p lấy từ danh sách cặp và 5 negatives. Hard-negative mining dùng descriptor hiện tại để tìm ảnh gần query, đồng thời loại cluster của query và giữ negatives từ các cluster khác nhau. Không tự xem mọi ảnh cùng cluster là positive; quan hệ positive dùng qidxs và pidxs trong metadata.')
para('Mỗi epoch train lấy 2.000 cặp trong 34.337 cặp để tạo 2.000 tuple, không phải 2.000 ảnh độc lập. Pool cấu hình 20.000 được giới hạn còn 15.953 ảnh train thực tế. Các tuple và hard negatives được tạo lại theo model ở từng epoch; một epoch không đồng nghĩa đã duyệt hết mọi cặp positive trong metadata.')
h2('Validation khác test ở đâu')
para('Validation không cập nhật trọng số và không dùng augmentation ngẫu nhiên. Val loss là contrastive loss trung bình theo tuple, không phải mAP và không phải accuracy phân loại. 181 cặp validation được đưa vào bước tạo tuple; với batch 5 và bỏ batch cuối không đủ, thực tế xử lý 180 tuple, dùng gallery mining từ 739 ảnh validation.')
para('Với Run 1 và Run 2, seed lấy mẫu validation được giữ cố định; negatives vẫn thay đổi theo descriptor của model hiện tại. Best checkpoint là epoch có val loss thấp nhất. Danh sách 181 cặp không cung cấp relevance đầy đủ cho mọi cặp ảnh, nên không được tự biến nó thành retrieval development set bằng cách gán các cặp còn lại là negative.')
para('Oxford và Paris dùng để đánh giá retrieval riêng, không dùng cập nhật gradient hoặc chọn best checkpoint. Chưa có audit ở mức địa danh chứng minh SfM và các benchmark hoàn toàn không giao nhau về nội dung; kết quả kiểm tra không trùng ảnh/cluster nêu trên chỉ áp dụng cho SfM train và validation.')

page('2 Cấu hình training của ba model')
para('Cấu hình dưới đây được đối chiếu từ run_record của B0, identity trong checkpoint của hai run và source runtime. Các giá trị LR là LR cơ sở trước khi áp dụng scheduler. GeM bắt đầu với p = 3, sau đó p được học cùng model.')
table(['Tham số','B0','Run 1 AUG','Run 2 AUG và Attention'],[
 ['Backbone và descriptor','VGG16; GeM; L2; 512-D','VGG16; GeM; L2; 512-D','VGG16; spatial gate; GeM; L2; 512-D'],
 ['Khởi tạo backbone','Caffe VGG16 ImageNet','Cùng file và hash','Cùng file và hash'],
 ['Kế hoạch epoch','100','100 sau khi kéo dài','100'],
 ['Epoch hoàn tất trong snapshot','14','100','100'],
 ['Seed training','42','42','42'],
 ['Cạnh dài tối đa của ảnh','1024; giữ tỷ lệ','1024; giữ tỷ lệ','1024; giữ tỷ lệ'],
 ['Optimizer','Adam','Adam','Adam'],
 ['LR cơ sở backbone','1e-6','1e-6','1e-6'],
 ['LR cơ sở GeM p','1e-5','1e-5','1e-5'],
 ['LR cơ sở attention','Không áp dụng','Không áp dụng','1e-4'],
 ['Weight decay backbone','1e-6','1e-6','1e-6'],
 ['Weight decay GeM và attention','GeM: 0','GeM: 0','GeM: 0; attention: 0'],
 ['Contrastive margin','0,7','0,7','0,7'],
 ['Tuple mỗi epoch train','2.000','2.000','2.000'],
 ['Pool mining cấu hình / thực tế','20.000 / 15.953','20.000 / 15.953','20.000 / 15.953'],
 ['Negatives trong một tuple','5','5','5'],
 ['Batch theo optimizer','5 tuple','5 tuple','5 tuple'],
 ['Optimizer updates mỗi epoch','400','400','400'],
 ['DataLoader workers','8','4','4'],
 ['Precision training','FP32','BF16 autocast; phần còn lại FP32','BF16 autocast; phần còn lại FP32']], [6.2,3.6,3.6,3.6],font=8.8,padding=45)
h2('Scheduler và cập nhật gradient')
para('LR giảm theo exp(−0,01 × e), với e bắt đầu từ 1 và được áp dụng trước epoch. Ví dụ, backbone dùng khoảng 9,9005e-7 ở epoch 1. LR GeM bằng 10 lần LR backbone; Run 2 có nhóm attention riêng. Weight decay không áp dụng lên GeM p và attention trong runtime mới.')
para('Mỗi lần optimizer.step tích lũy gradient từ 5 tuple; mỗi tuple có 7 lượt ảnh (q, p và 5 negatives), tương đương 35 lượt ảnh mỗi update, có thể có ảnh trùng. Loss các tuple được cộng trước update. Batch size không có nghĩa forward đồng thời 35 ảnh trên GPU. Tham số momentum 0,9 trong record B0 không được dùng khi optimizer được chọn là Adam.')
h2('Cấu hình thực thi của hai run mới')
para('Run 1 và Run 2 cùng MICROBATCH 8, extraction window 8, prefetch factor 2, tuple fusion 1, cache tắt và pixel cap 8 × 1024². Microbatch là trần số ảnh cùng shape trong một forward, không phải số tuple cho optimizer. Khi fusion bằng 1, một tuple chỉ có 7 ảnh, nên forward training không nhất thiết đạt 8 ảnh. GeM, chuẩn hóa descriptor, loss, weights và Adam giữ FP32 khi backbone dùng BF16 autocast.')

page('3 Augmentation attention và luồng training')
h2('Augmentation dùng trong Run 1 và Run 2')
para('Hai run dùng cùng policy mild_photo_blur_jpeg_erase_v1. Biến đổi được lấy mẫu độc lập khi đọc từng ảnh q, p và negatives; chỉ bật ở train forward. Mining, validation và benchmark clean không bật policy này. B0 dùng ToTensor và Normalize, không có policy augmentation mới.')
table(['Phép biến đổi','Xác suất','Tham số thực tế'],[
 ['Brightness contrast và color','0,80 cho cả nhóm','Brightness 0,8–1,2; contrast 0,8–1,2; color 0,85–1,15. Thứ tự ba phép được xáo ngẫu nhiên.'],
 ['Gaussian blur','0,15','Pillow GaussianBlur với radius lấy đều trong 0,2–1,0 pixel.'],
 ['JPEG','0,20','Quality nguyên từ 55 đến 95; encode rồi decode lại trước Normalize.'],
 ['Random erasing','0,15','Diện tích 1–5%; aspect ratio 0,5–2,0; chỉ thực hiện nếu hình chữ nhật vừa ảnh. Fill 0 trong không gian đã Normalize.']], [4.5,2.5,10],font=9.2)
para('Ảnh được chuyển RGB, resize giữ tỷ lệ và đổi về [0,1]. Normalize dùng mean (0,485; 0,456; 0,406), std (0,229; 0,224; 0,225). Erasing nằm sau Normalize, vì vậy fill 0 tương ứng mean màu theo chuẩn hóa, không phải tô đen RGB.')
h2('Attention của Run 2')
para('Spatial gate đặt trước GeM: lấy mean và max theo channel, ghép thành hai kênh rồi qua Conv2d 2→1, kernel 7×7, padding 3. Gate dùng 2 × sigmoid và nhân với feature map. Conv có 99 tham số, weight và bias khởi tạo 0, nên gate bằng 1 ở thời điểm khởi tạo. Descriptor cuối vẫn là 512-D; attention được học bằng nhóm LR riêng 1e-4.')
h2('Luồng xử lý trong một epoch')
para('Bước 1. Dùng ảnh clean để trích descriptor cho pool và tạo hard-negative tuples. Với hai run mới, seed train thay đổi theo epoch; negatives phụ thuộc model hiện tại.')
para('Bước 2. Đọc ảnh của các tuple, bật augmentation nếu là Run 1 hoặc Run 2, forward theo shape và microbatch, tính contrastive loss, backprop rồi Adam update sau mỗi 5 tuple.')
para('Bước 3. Chạy validation không augmentation và không gradient; tính val loss trên 180 tuple được xử lý. So với best val loss trước đó để quyết định checkpoint best.')
para('Bước 4. Lưu checkpoint theo ranh giới epoch. Latest dùng để resume; best dùng tiêu chí validation. Benchmark Oxford/Paris là nhánh chấm điểm độc lập, không tham gia optimizer.')
para('B0 có test_freq 1 trong record và ZIP mới có saved evaluation các epoch 8–14. Hai run Colab đặt đánh giá định kỳ mỗi 3 epoch khi đủ thời gian; lần đánh giá cuối dùng best checkpoint. Bộ post-training riêng lại chủ động dùng latest epoch 100 của cả hai run để có cùng mốc final.')

page('4 Tập test và cách chấm retrieval')
para('Oxford5k và Paris6k là hai benchmark truy hồi địa danh riêng, không phải phần test được cắt từ 16.692 ảnh SfM subset. Mỗi benchmark gồm gallery, query và ground truth. Model không được cập nhật trọng số trên các benchmark này.')
table(['Benchmark','Gallery','Query','Thông tin ground truth'],[
 ['Oxford5k','5.063 ảnh','55 query','imlist, qimlist; bounding box, ok và junk cho từng query'],
 ['Paris6k','6.392 ảnh','55 query','Cùng cấu trúc; dùng file ground truth riêng']], [3.5,2.3,2.1,9.1])
para('Gallery là tập ứng viên để xếp hạng. Query là ảnh truy vấn được crop theo bounding box của đối tượng; ảnh nguồn query có thể xuất hiện trong gallery. Ground truth đánh dấu ảnh relevant bằng ok và ảnh không tính vào chấm điểm bằng junk. Không tự xóa ảnh khỏi feedback dựa trên nhãn test. Evaluator bỏ junk khi tính AP; P@10 tính trên 10 kết quả đầu không-junk.')
para('AP đo chất lượng thứ hạng của một query; mAP là trung bình AP các query hợp lệ. Đây không phải accuracy phân loại. Giao thức dùng legacy Oxford/Paris, không phải revisited ROxford/RParis; không so lẫn điểm từ hai giao thức.')
h2('Ảnh clean và test suy giảm')
para('Query clean dùng single scale, whitening tắt và resize giữ tỷ lệ với giới hạn cạnh dài 1024. Query crop giữ quy tắc resize theo ảnh gốc của loader; không ép mọi ảnh thành 1024×1024. Các biến đổi suy giảm áp dụng sau crop/resize, trước Normalize. Gallery luôn clean. Global descriptor và local features dùng cùng pixel query đã kiểm tra checksum.')
table(['Biến đổi','Nhẹ','Vừa','Nặng','Định nghĩa'],[
 ['Darkening','0,8','0,6','0,4','Nhân RGB với hệ số'],
 ['Low contrast','0,8','0,6','0,4','Thu độ lệch pixel quanh mean từng kênh'],
 ['Gaussian blur','0,001 L','0,002 L','0,004 L','Sigma tối thiểu 0,1 px; L là cạnh dài query sau resize'],
 ['Gaussian noise','0,02','0,05','0,08','Std trên RGB [0,1]; clip về miền hợp lệ'],
 ['JPEG','80','50','25','Quality; RGB, subsampling 2'],
 ['Occlusion','5%','10%','20%','Tỷ lệ diện tích che; fill mean RGB query']], [3.6,1.9,1.9,1.9,7.7],font=9)
para('Noise và occlusion có ba realization cố định ở mỗi mức; các họ còn lại có một realization. Vì vậy có 30 phiên bản suy giảm cộng một clean cho mỗi query: 110 query × 31 = 3.410 lượt query, hay 6.820 lượt encoder khi chạy cả hai model. Seed gốc corruption là 20261030; seed con lấy bằng hash ổn định theo query và điều kiện, độc lập training seed 42.')
para('Trung bình AP qua realization trước, rồi trung bình đều 6 họ × 3 mức để có corrupted mAP. Worst-condition là mAP thấp nhất trong 18 điều kiện, không phải AP thấp nhất của một query. Drop là clean mAP trừ corrupted mAP, đơn vị điểm phần trăm; drop nhỏ không tự chứng minh chất lượng tốt nếu điểm tuyệt đối thấp.')
para('Gaussian noise không nằm trong augmentation policy training đã lưu. Các họ còn lại có thành phần gần tương ứng nhưng cường độ hoặc cách sinh khác nhau. Occlusion nặng có thể làm mất chi tiết đối tượng; báo cáo giữ kết quả và không khẳng định mọi ảnh biến đổi đều bảo toàn nhãn hoàn hảo.')

page('5 Cấu hình hậu xử lý và đo hiệu năng')
para('Các bước hậu xử lý sử dụng checkpoint đã train xong và không cập nhật trọng số encoder. Với hai run mới, toàn bộ bảng chính dùng latest epoch 100, FP32 inference, cùng gallery/query IDs và cùng ảnh corruption. Mỗi encoder có descriptor và graph riêng; local gallery features được dùng chung khi preprocessing giống nhau.')
table(['Hướng','Cấu hình khóa trước chạy'],[
 ['AQE và Alpha-QE','Top k = 10; lambda = 0,5. AQE dùng alpha = 0; Alpha-QE dùng alpha = 3. Một vòng feedback có residual query, rồi chuẩn hóa L2 và tìm lại toàn gallery.'],
 ['Gated Alpha-QE','Giữ k, alpha, lambda như Alpha-QE; tau = 0,5; cần ít nhất 3 feedback vượt gate, nếu không giữ query gốc.'],
 ['Local và geometric re-ranking','RootSIFT tối đa 2.000 keypoints; ratio 0,75 và mutual matching. Top-K = 50, eta = 0,5; chỉ đổi thứ tự trong top-K, giữ phần đuôi nguyên.'],
 ['Kiểm chứng hình học','Homography RANSAC: threshold 3 px, confidence 0,995, tối đa 2.000 iterations. Valid geometry cần ít nhất 8 inliers và homography hợp lệ, không suy biến.'],
 ['Rank fusion','Average-score fusion hoặc full-list RRF với c = 60, ghép đúng gallery ID. Dùng Run 1 và Run 2, không cộng raw embeddings của hai model.'],
 ['Graph diffusion','kNN = 20; union hoặc mutual; trọng số cosine dương mũ 3. Chuẩn hóa graph theo degree; beta = 0,9; tối đa 100 iterations; relative L2 tolerance 1e-6.']], [5,12],font=9.2)
h2('Pipeline geometry verified QE')
para('Global cosine lấy top 50. Geometry chọn tối đa 10 feedback có cosine dương và kiểm chứng hợp lệ; nếu có dưới 3 ảnh hợp lệ thì giữ query gốc. Phần cập nhật query dùng alpha 3 và lambda 0,5 như Alpha-QE, rồi search lại toàn gallery. Pipeline không re-rank lần thứ hai, để giữ rõ tác động của bước lọc feedback.')
h2('Các metric bổ sung')
para('P@10 là tỷ lệ ảnh relevant trong 10 kết quả đầu sau khi bỏ junk. Relevant Recall@20, @50 và @100 là số relevant đã tìm thấy trong top-K không-junk chia cho tổng relevant của query. Recall ở đây không phải chỉ báo có ít nhất một ảnh đúng. AP từng query và full rankings được lưu để chấm lại.')
h2('Đo thời gian và sử dụng cache')
para('Retrieval-only chỉ đo search và các bước hậu xử lý trên descriptor đã trích. Direct end-to-end đo query mới từ decode, crop/resize, corruption, encoder đến search; các phương pháp local còn tính trích local query và matching/RANSAC trong timer. Gallery descriptor, local gallery cache và graph là chuẩn bị offline, được báo tách khỏi query online.')
para('Có warm-up ngoài mẫu đo, đồng bộ GPU và batch query bằng 1. Timing trực tiếp lấy clean cùng mức nặng của sáu họ biến đổi, hai query mỗi điều kiện và hai lần lặp: 28 mẫu mỗi model/dataset/method, tổng 1.120 mẫu. Chi phí matching replay trong bảng retrieval-only không được gọi là direct end-to-end; phần kết quả có bảng trực tiếp riêng.')
para('Chưa có development set có nhãn relevance đầy đủ; sensitivity clean chỉ mang tính exploratory, lưu cả cấu hình tốt và kém. Tham số chính giữ nguyên giữa các corruption. B0 chưa có đánh giá suy giảm hoặc hậu xử lý; không có nhiều training seed để kết luận độ ổn định.')

from pathlib import Path
import csv,json,math
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
R=Path(__file__).resolve().parent.parent
P=R.parent/'kaggle-posttraining/outputs/iir-post100-report/unpacked'
OUT=Path(__file__).resolve().parent
D=Document();sec=D.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.6);sec.bottom_margin=Cm(1.6);sec.left_margin=Cm(2);sec.right_margin=Cm(2)
sec.footer_distance=Cm(.7)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
 st=D.styles[name];st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
 st.element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'),'Arial')
 for e in st.element.findall('.//'+qn('w:pBdr')):e.getparent().remove(e)
normal=D.styles['Normal'];normal.font.size=Pt(10.5);normal.paragraph_format.line_spacing=1.12;normal.paragraph_format.space_after=Pt(6)
for name,size in [('Title',23),('Subtitle',12),('Heading 1',15),('Heading 2',12),('Caption',9)]:
 D.styles[name].font.size=Pt(size)
 D.styles[name].paragraph_format.space_before=Pt(8)
 D.styles[name].paragraph_format.space_after=Pt(7)
 D.styles[name].paragraph_format.keep_with_next=True
D.styles['Title'].paragraph_format.space_before=Pt(0)
D.styles['Caption'].paragraph_format.keep_with_next=False
D.core_properties.title='Tổng hợp kết quả truy hồi ảnh VGG16 GeM'
D.core_properties.subject='Baseline B0 và hai mô hình cải tiến cùng bốn hướng hậu xử lý truy hồi'
D.core_properties.author='SEG301'
f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER
run=f.add_run('SEG301  |  ');run.font.size=Pt(8)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');f._p.append(fld)

def para(text,bold=False,style=None):
 p=D.add_paragraph(style=style);p.paragraph_format.widow_control=True;r=p.add_run(text);r.bold=bold;return p

def h(text):D.add_heading(text,level=1)
def h2(text):D.add_heading(text,level=2)
def page(title):D.add_page_break();h(title)
def caption(text):para(text,style='Caption')
def pct(v):return f'{float(v)*100:.2f}'.replace('.',',')
def num(v,d=2):return f'{float(v):.{d}f}'.replace('.',',')
def rows(path):return list(csv.DictReader(path.open()))

def table(headers,data,widths=None,bold_rows=(),font=9.5,padding=65):
 t=D.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 widths=widths or [17/len(headers)]*len(headers)
 for i,w in enumerate(widths):t.columns[i].width=Cm(w)
 for c,w in zip(t.rows[0].cells,widths):c.width=Cm(w)
 repeat=OxmlElement('w:tblHeader');repeat.set(qn('w:val'),'true');t.rows[0]._tr.get_or_add_trPr().append(repeat)
 pr=t._tbl.tblPr;b=OxmlElement('w:tblBorders')
 for edge in ['top','left','bottom','right','insideH','insideV']:
  e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
 pr.append(b)
 for index,row in enumerate([headers]+list(data)):
  cells=t.rows[0].cells if index==0 else t.add_row().cells
  trpr=cells[0]._tc.getparent().get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
  for j,(cell,value) in enumerate(zip(cells,row)):
   cell.width=Cm(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   tcpr=cell._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
   for edge,size in [('top',padding),('bottom',padding),('left',90),('right',90)]:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),str(size));e.set(qn('w:type'),'dxa');m.append(e)
   tcpr.append(m);shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'20364D' if index==0 else ('F1F5F8' if index%2==0 else 'FFFFFF'));tcpr.append(shd)
   p=cell.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
   p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1.05
   p.paragraph_format.keep_with_next=index==0
   r=p.add_run(str(value));r.font.name='Arial';r.font.size=Pt(font);r.bold=index==0 or index-1 in bold_rows
   r.font.color.rgb=RGBColor(255,255,255) if index==0 else RGBColor(0,0,0)
 gap=D.add_paragraph();gap.paragraph_format.space_after=Pt(3);gap.paragraph_format.space_before=Pt(2)
 return t

clean=rows(R/'clean_model_comparison.csv');post=rows(R/'all_posttraining_results.csv');training=rows(R/'training_summary.csv');sens=rows(R/'all_clean_sensitivity.csv');cond=rows(P/'conditions.csv');lat=rows(P/'end_to_end_summary.csv')
cp={(r['model'],r['selection'],r['dataset']):r for r in clean}
pp={(r['model_short'],r['method'],r['dataset']):r for r in post}
methods=['cosine','AQE','alpha_QE','gated_QE','local_rerank','geometric_rerank','verified_QE','diffusion_mutual','diffusion_union']
labels={'cosine':'Cosine','AQE':'AQE','alpha_QE':'Alpha-QE','gated_QE':'Gated QE','local_rerank':'Local re-ranking','geometric_rerank':'Geometric re-ranking','verified_QE':'Verified QE','diffusion_mutual':'Diffusion mutual','diffusion_union':'Diffusion union','score_fusion':'Score fusion','RRF':'RRF'}

D.add_paragraph('Tổng hợp kết quả truy hồi ảnh VGG16 GeM',style='Title')
D.add_paragraph('Baseline B0 và hai mô hình cải tiến',style='Subtitle')
para('Dự án SEG301  •  Ngày 02 tháng 11 năm 2026')
h2('Kết quả chính')
para('Baseline B0 được chọn ở epoch 12 đạt mAP 81,06% trên Oxford5k và 77,12% trên Paris6k. Đây là kết quả clean cao nhất trong các cấu hình đã có. Hai mô hình dùng augmentation và augmentation kết hợp attention chưa vượt B0 theo số liệu hiện tại.')
para('Trên hai checkpoint epoch 100, Query Expansion cải thiện chất lượng với chi phí thấp. Geometry-verified QE cho điểm cao nhất trong nhóm hậu xử lý đã chạy nhưng mất vài giây cho mỗi query. Rank fusion không vượt Run 1 cosine; graph diffusion giảm điểm mạnh ở cấu hình thử nghiệm.')
h2('So sánh checkpoint trên ảnh gốc')
settings=[('B0','best_validation','B0 best','12'),('B0','last_completed_in_snapshot','B0 cuối snapshot','14'),('run1','best_validation','Run 1 AUG best','27'),('run2','best_validation','Run 2 AUG và Attention best','28'),('run1','final_epoch_100','Run 1 AUG final','100'),('run2','final_epoch_100','Run 2 AUG và Attention final','100')]
data=[]
for model,sel,label,epoch in settings:
 values=[float(cp[model,sel,ds]['map']) for ds in ['oxford5k','paris6k']]
 data.append([label,epoch,pct(values[0]),pct(values[1]),pct(sum(values)/2)])
table(['Model','Epoch','Oxford','Paris','TB hai tập'],data,[6.7,1.8,2.85,2.85,2.8],bold_rows=[0])
para('Đơn vị: mAP (%). “Best” được chọn bằng validation loss. Trung bình hai tập là trung bình đều Oxford và Paris. Các dòng lịch sử khác training budget và precision, nên chưa phải ablation chỉ thay một thành phần.',style=None)
h2('Độ đầy đủ của đánh giá')
para('Bộ hậu xử lý đã hoàn thành 68.200 ô đánh giá chính, 5.500 AP query cho sensitivity và 1.120 mẫu đo latency end-to-end. Có đủ kết quả clean và 18 điều kiện suy giảm ảnh cho hai model epoch 100. Baseline B0 hiện chỉ có đánh giá clean.')

page('1 Phạm vi và giao thức đánh giá')
para('Hệ thống dùng VGG16 làm backbone, GeM với tham số p học được và descriptor 512 chiều chuẩn hóa L2. Dữ liệu SfM subset có 15.953 ảnh train trong 100 clusters và 739 ảnh validation trong 20 clusters; seed training là 42.')
table(['Thành phần','B0','Run 1','Run 2'],[
 ['Augmentation training','Không','Có','Có'],['Spatial attention','Không','Không','Có'],['Epoch hoàn tất trong snapshot','14','100','100'],['Best validation epoch','12','27','28'],['Precision training','FP32','BF16 và FP32','BF16 và FP32'],['Checkpoint best trong ZIP','Có','Thiếu','Có']], [6.2,3.6,3.6,3.6])
para('Metadata SfM, hash trọng số Caffe khởi tạo và ground truth Oxford/Paris khớp giữa các bộ kết quả. B0 và hai run vẫn khác runtime, phần cứng, training budget và precision. Không thể quy toàn bộ chênh lệch hiệu năng riêng cho augmentation hoặc attention.')
h2('Cách tính chất lượng')
para('Oxford5k gồm 5.063 ảnh gallery và 55 query; Paris6k gồm 6.392 ảnh gallery và 55 query. Đánh giá dùng legacy AP với xử lý junk, crop query theo bounding box, cạnh dài tối đa 1024 và single scale. Whitening tắt. Điểm clean của checkpoint dùng cosine khi chưa bật hậu xử lý.')
para('Hậu xử lý dùng hai latest checkpoint epoch 100, trích lại descriptor FP32 và tắt TF32. Gallery giữ clean. Query có darkening, giảm contrast, Gaussian blur, Gaussian noise, JPEG và occlusion, mỗi họ ba mức. Noise và occlusion có ba realization; tổng cộng 31 phiên bản cho mỗi query, gồm clean.')
para('Corrupted mAP được tính bằng cách trung bình realization trước, sau đó trung bình đều 18 điều kiện. Bảng từng dataset được giữ riêng; macro trong báo cáo là trung bình đều hai dataset. Không coi các phiên bản suy giảm của một query là những training seed độc lập.')
h2('Bốn hướng hậu xử lý')
table(['Hướng','Cơ chế','Cấu hình chính'],[
 ['Query Expansion','Cập nhật query từ kết quả đầu rồi search lại','k 10; alpha 3; lambda 0,5; gate tau 0,5'],
 ['Geometric re-ranking','RootSIFT và RANSAC trên nhóm ứng viên đầu','Top 50; tối đa 2.000 keypoints; eta 0,5'],
 ['Rank Fusion','Gộp score hoặc thứ hạng của hai encoder','Average score; RRF c 60'],
 ['Graph Diffusion','Lan truyền seed score trên graph gallery','kNN 20; beta 0,9; union hoặc mutual']], [3.5,7,6.5])
para('Tham số chính được khóa trước chạy. Ground truth chỉ dùng chấm điểm, không dùng chọn feedback, sửa graph hoặc lọc matching. Các kết quả là exploratory; sensitivity trên clean không được dùng để thay cấu hình chính sau khi xem test.')

page('2 Diễn biến training và lựa chọn checkpoint')
table(['Model','Best epoch','Best val loss','Val loss cuối','Phút mỗi epoch'],[[{'B0':'B0','run1':'Run 1','run2':'Run 2'}[r['model']],r['best_epoch'],num(r['best_val_loss'],6),num(r['final_val_loss'],6),num(r['mean_train_val_minutes'])] for r in training],[3,2,4,4,4])
caption('Thời gian là trung bình train cộng validation, chưa gồm đầy đủ benchmark và lưu trữ. Khác phần cứng và precision nên không dùng để kết luận riêng về tốc độ thuật toán.')
D.add_picture(str(OUT/'training_report.png'),width=Cm(17))
caption('Hình 1  Training loss và validation loss của ba model. Trục epoch của mỗi ô phản ánh số epoch thực tế đã hoàn thành.')
para('Run 1 có validation loss thấp nhất 0,100764 ở epoch 27, tăng lên 0,123108 ở epoch 100. Run 2 thấp nhất 0,100519 ở epoch 28, tăng lên 0,119673 ở epoch 100. Train loss vẫn tiếp tục giảm. Đây là dấu hiệu overfit theo validation; tăng epoch không bảo đảm chất lượng retrieval tăng.')
para('B0 đạt best validation epoch 12. Validation loss chỉ tăng nhẹ trong hai epoch 13 và 14 được quan sát; dữ liệu này không mô tả phần training sau epoch 14.')
h2('Các epoch B0 có saved evaluation')
b0=rows(R/'B0_map_by_epoch.csv');epochs=sorted({int(r['epoch']) for r in b0})
table(['Epoch','Oxford clean mAP','Paris clean mAP'],[[e,*[num(next(r['mAP_percent'] for r in b0 if int(r['epoch'])==e and r['dataset']==ds)) for ds in ['oxford5k','paris6k']]] for e in epochs],[3,7,7],bold_rows=[epochs.index(12)])
para('Paris đạt 78,20% ở epoch 8, cao hơn epoch 12 trên riêng dataset này. Checkpoint chọn vẫn là epoch 12 theo validation loss; không ghép điểm Oxford epoch 12 và Paris epoch 8 thành một kết quả model.')

page('3 Kết quả hậu xử lý trên Run 1')
para('Run 1 dùng augmentation khi training, không có attention. Toàn bộ bảng dưới đây dùng checkpoint epoch 100 và FP32 inference. “Suy giảm” là corrupted mAP trung bình 18 điều kiện; đơn vị mAP (%).')
headers=['Phương pháp','Oxford\nclean','Oxford\nsuy giảm','Paris\nclean','Paris\nsuy giảm']
data=[[labels[m],*[pct(pp['run1',m,ds][k]) for ds in ['oxford5k','paris6k'] for k in ['clean_map','corrupted_map']]] for m in methods]
table(headers,data,[6.2,2.7,2.7,2.7,2.7],bold_rows=[6])
h2('Lợi ích so với cosine của cùng checkpoint')
macro=rows(R/'posttraining_macro.csv')
base=next(r for r in macro if r['model']=='run1' and r['method']=='cosine')
table(['Phương pháp','Clean TB','Suy giảm TB','Tăng mAP suy giảm'],[[labels[m],pct(next(r for r in macro if r['model']=='run1' and r['method']==m)['clean_map']),pct(next(r for r in macro if r['model']=='run1' and r['method']==m)['corrupted_map']),num((float(next(r for r in macro if r['model']=='run1' and r['method']==m)['corrupted_map'])-float(base['corrupted_map']))*100)+' điểm'] for m in ['cosine','AQE','alpha_QE','verified_QE']], [6.5,3,3,4.5])
para('Alpha-QE cải thiện corrupted mAP macro thêm 2,52 điểm phần trăm. AQE gần như tương đương; dữ liệu một seed chưa cho thấy khác biệt có ý nghĩa rõ giữa hai biến thể. Gate tau 0,5 luôn mở và vẫn lấy đủ 10 feedback, nên output chính giống Alpha-QE.')
para('Local-only re-ranking đạt điểm cao hơn geometric re-ranking ở cả hai dataset. Kiểm chứng hình học phát huy tác dụng rõ hơn khi dùng lọc feedback trước QE: verified QE đạt 78,94% clean Oxford và 66,97% clean Paris, đồng thời có corrupted mAP tốt nhất trong nhóm Run 1.')
para('Mức tăng này là so với Run 1 epoch 100 cosine. Verified QE vẫn thấp hơn B0 best trên clean: 2,12 điểm ở Oxford và 10,16 điểm ở Paris. Không diễn giải gain hậu xử lý nội bộ thành bằng chứng vượt baseline B0.')
h2('Điều kiện khó nhất')
para('Blur mức nặng gây suy giảm lớn nhất cho Run 1 cosine ở cả hai dataset. Worst-condition Oxford tăng từ 48,33% lên 60,97% với verified QE; Paris tăng từ 54,71% lên 58,60%. Kết quả cho thấy lợi ích trên query khó của encoder này, nhưng B0 chưa có đánh giá suy giảm tương ứng để đối chiếu.')

page('4 Kết quả Run 2 và gộp hai encoder')
para('Run 2 kết hợp augmentation và spatial attention. Không có model Attention-only trong bộ kết quả, nên chưa tách riêng được đóng góp attention với một ma trận ablation đầy đủ.')
data=[[labels[m],*[pct(pp['run2',m,ds][k]) for ds in ['oxford5k','paris6k'] for k in ['clean_map','corrupted_map']]] for m in methods]
table(headers,data,[6.2,2.7,2.7,2.7,2.7],bold_rows=[6])
para('Ở epoch 100, Run 1 cao hơn Run 2 trên cả clean và corrupted mAP trung bình của hai dataset. Verified QE vẫn cải thiện Run 2, đạt 74,25% corrupted Oxford và 64,31% corrupted Paris. Đây là so sánh tại checkpoint cụ thể, chưa đủ kết luận attention nói chung không hiệu quả.')
h2('Rank fusion dùng Run 1 và Run 2')
table(headers,[[labels[m],*[pct(pp['fusion',m,ds][k]) for ds in ['oxford5k','paris6k'] for k in ['clean_map','corrupted_map']]] for m in ['score_fusion','RRF']],[6.2,2.7,2.7,2.7,2.7])
para('Average-score fusion và RRF không vượt Run 1 cosine. Cả hai cần hai encoder và hai bộ descriptor, nên chưa có lợi ích chất lượng bù chi phí trong kết quả hiện tại.')
h2('Diễn giải kết quả diffusion')
para('Diffusion union và mutual đều giảm điểm mạnh trên hai encoder. Hầu hết query đã hội tụ ở tiêu chí đang dùng; riêng mutual của Run 2 Oxford có một tỷ lệ rất nhỏ chạm giới hạn iteration. Vì vậy, không thể giải thích toàn bộ mức giảm bằng việc chưa đủ iteration.')
para('Nguyên nhân cụ thể chưa được xác định. Graph, seed score và normalization có thể tác động đến thứ hạng; báo cáo không khẳng định một nguyên nhân khi chưa có kiểm tra riêng. Kết quả âm được giữ nguyên và chỉ áp dụng cho thiết kế diffusion đã chạy, không suy rộng cho mọi phương pháp graph retrieval.')

page('5 Mức suy giảm ảnh và chi phí xử lý')
para('Bảng sau mô tả Run 1: C là cosine, Q là Alpha-QE và V là verified QE. Mỗi ô là mAP (%), đã trung bình realization khi cần. Gallery giữ ảnh gốc.')
families=[('darkening','Tối ảnh'),('contrast','Contrast'),('blur','Blur'),('noise','Nhiễu'),('jpeg','JPEG'),('occlusion','Che khuất')]
cr={(r['dataset'],r['method'],r['family'],int(r['severity'])):r for r in cond if r['model']=='run1_aug_epoch100'}
table(['Biến đổi','Mức','Ox C','Ox Q','Ox V','Pa C','Pa Q','Pa V'],[[name,sev,*[pct(cr[ds,m,f,sev]['ap']) for ds in ['oxford5k','paris6k'] for m in ['cosine','alpha_QE','verified_QE']]] for f,name in families for sev in [1,2,3]], [3,1.2,2.13,2.13,2.13,2.13,2.13,2.15],font=9)
h2('Latency end to end trên Run 1')
lt={(r['model'],r['dataset'],r['method']):r for r in lat}
table(['Phương pháp','Oxford median','Oxford p95','Paris median','Paris p95'],[[labels[m],*[num(float(lt['run1_aug_epoch100',ds,m][stat])/1000,3)+' s' for ds in ['oxford5k','paris6k'] for stat in ['median','p95']]] for m in ['cosine','alpha_QE','local_rerank','geometric_rerank','verified_QE','diffusion_mutual']],[6.2,2.7,2.7,2.7,2.7],font=9)
para('Mỗi model, dataset và phương pháp có 28 mẫu timing: clean và mức nặng của sáu họ biến đổi, hai query mỗi điều kiện, hai lần đo. Query mới được decode và trích feature trong timer; gallery và graph được cache. Các median có nhiễu, không suy ra QE nhanh hơn cosine chỉ từ bảng này.')
para('Ở bước retrieval-only, Alpha-QE thêm khoảng 3–5 ms/query so với cosine. Verified QE cần khoảng 5,8–8,5 giây/query end-to-end trên mẫu đo; mức tăng chất lượng đi kèm chi phí lớn hơn rõ rệt.')

page('6 Tổng hợp kết luận và giới hạn')
h2('Kết luận từ số liệu đã có')
para('B0 best epoch 12 là mốc clean mạnh nhất hiện có bằng chứng, với 81,06% Oxford và 77,12% Paris. Best của Run 1 thấp hơn B0 1,53 và 5,19 điểm; best của Run 2 thấp hơn 0,57 và 4,36 điểm. Chưa có cấu hình cải tiến nào trong bộ kết quả này chứng minh vượt B0 trên clean.')
para('Trong nhóm hai encoder epoch 100, Run 1 cùng AQE hoặc Alpha-QE có mức tăng chất lượng với chi phí thấp. Verified QE cho chất lượng cao nhất trong nhóm này nhưng yêu cầu vài giây cho mỗi query. Fusion chưa vượt encoder mạnh hơn; diffusion ở cấu hình đã chạy không tạo cải thiện.')
para('Hai run mới có dấu hiệu overfit theo validation loss ở giai đoạn cuối. Kết quả không ủng hộ giả định cứ tăng epoch là retrieval sẽ tốt hơn. Validation loss và test mAP cũng không có cùng thứ tự giữa các model: cần giữ nguyên tiêu chí chọn checkpoint đã định trước.')
h2('Sensitivity trên clean')
para('Đã chạy đủ 100 dòng model, method, config và dataset, tương ứng 5.500 AP query. Các trục gồm k, alpha, lambda và gate của QE; top-K và eta của re-ranking; c của RRF; k và beta của diffusion. Toàn bộ điểm nằm trong phụ lục, kể cả cấu hình làm giảm chất lượng.')
para('Ví dụ, Alpha-QE của Run 1 với lambda 1 đạt 77,30% Oxford và 66,56% Paris clean. Local re-ranking K 100 và eta 0,5 đạt 78,85% và 66,94%. Đây là sensitivity quan sát trên test; chưa có robustness của các cấu hình đó, nên chúng không thay tham số chính đã khóa trước chạy.')
h2('Giới hạn khi diễn giải')
para('Chỉ có một training seed 42. Cùng một query có nhiều phiên bản suy giảm không tạo thành các training run độc lập. Báo cáo không suy ra độ ổn định qua nhiều seed hoặc một khoảng tin cậy tổng quát từ các biến thể query.')
para('B0 chưa có kết quả corruption và hậu xử lý theo cùng giao thức. Các bảng lịch sử khác epoch, precision và runtime; chưa thể quy chênh lệch riêng cho augmentation hay attention. Bộ kết quả cũng thiếu nhánh Attention-only. Vì vậy, chưa có bằng chứng rằng các model mới robust hơn B0.')
h2('Nguồn và tính toàn vẹn')
para('Nguồn gồm ZIP baseline reproduction B0, hai ZIP đầu ra Colab và output Kaggle post-training. SHA256 của B0 best.pt, history và evaluation epoch 12 khớp manifest. Hash archive của hai run, metadata SfM, trọng số khởi tạo và ground truth đã được đối chiếu.')
para('Bảng chính hậu xử lý có đủ 68.200 ô, không thiếu ô nào; tổng per-query gồm sensitivity là 73.700 dòng. Bản tổng hợp lưu 214 epoch records của ba model, 40 dòng kết quả chính theo model và dataset, cùng cấu hình thật cho mọi dòng sensitivity. B0 best checkpoint hiện có; Run 1 best checkpoint không nằm trong ZIP, dù saved evaluation còn đầy đủ.')

# Appendix: complete sensitivity records, 34 + 33 + 33 rows.
def short_model(v):return 'R1' if v.startswith('run1_') else 'R2' if v.startswith('run2_') else 'R1+R2'
def short_method(v):return {'alpha_QE':'Alpha-QE','gated_QE':'Gated QE','local_rerank':'Local','geometric_rerank':'Geometry','diffusion_mutual':'D mutual','diffusion_union':'D union','RRF':'RRF'}[v]
def short_config(row):
 c=json.loads(row['config_json']);m=row['method']
 if m in ['alpha_QE','gated_QE']:
  text=f"k={c['k']}  a={num(c['alpha'],0)}  L={num(c['lam'],2)}"
  if c.get('tau') is not None:text+='  t='+num(c['tau'],1)
  return text
 if m in ['local_rerank','geometric_rerank']:return f"K={c['k']}  eta={num(c['eta'],2)}"
 if m.startswith('diffusion'):return f"k={c['k']}  beta={num(c['beta'],2)}"
 return 'c='+str(c['c'])
ordered=sorted(sens,key=lambda r:(r['model'],r['dataset'],r['method'],r['config_json']))
for ix,(start,end) in enumerate([(0,34),(34,67),(67,100)],1):
 page(f'Phụ lục sensitivity clean phần {ix}')
 para('R1 là Run 1, R2 là Run 2; Ox là Oxford5k, Pa là Paris6k. a: alpha; L: lambda; t: tau; K: số ứng viên re-ranking. Các giá trị khác giữ theo cấu hình tham chiếu. Đơn vị mAP (%).')
 table(['Model','Tập','Phương pháp','Tham số thay đổi','mAP'],[[short_model(r['model']),'Ox' if r['dataset']=='oxford5k' else 'Pa',short_method(r['method']),short_config(r),pct(r['clean_map'])] for r in ordered[start:end]],[1.8,1.3,3.2,8.4,2.3],font=9,padding=40)
 para('Mỗi dòng được tính từ 55 query. Điểm sensitivity không được coi là kết quả tuning trên development set và không thay thế bảng chính trên corruption.',style='Caption')

# Explicit font families prevent mixed theme-font fallback in table headers.
for style in D.styles:
 if style.type==1:
  fonts=style.element.get_or_add_rPr().get_or_add_rFonts()
  for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
   if qn('w:'+key) in fonts.attrib:del fonts.attrib[qn('w:'+key)]
  for key in ['ascii','hAnsi','cs','eastAsia']:fonts.set(qn('w:'+key),'Arial')
# No theme-colored headings or title rules survive in direct formatting.
for p in D.paragraphs:
 if p.style.name in ['Title','Subtitle','Heading 1','Heading 2','Heading 3']:
  for r in p.runs:r.font.color.rgb=RGBColor(0,0,0)
  for e in list(p._p.findall('.//'+qn('w:pBdr'))):e.getparent().remove(e)
path=OUT/'Bao_cao_tong_hop_ket_qua_SEG301.docx';D.save(path)
print(path)
print('Main source rows',len(post),'sensitivity rows',len(sens))

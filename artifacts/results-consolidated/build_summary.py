"""Aggregate supplied artifacts only. No inference, training or remote mutations."""
import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
POST=PROJECT/'artifacts/kaggle-posttraining/outputs/iir-post100-report/unpacked'
BUNDLE=PROJECT/'artifacts/kaggle-posttraining/bundle'
DATASETS=['oxford5k','paris6k']
LABEL={'B0':'B0: VGG16–GeM','run1':'Run 1: AUG','run2':'Run 2: AUG + Attention'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write_csv(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    archive=PROJECT/'B0_VGG16_GeM_baseline_reproduction_evaluation_artifacts.zip'
    inventories=[];epochs=[];clean=[];training=[];histories={};validation={}
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        def get(leaf):return z.read(next(n for n in names if n.endswith('/'+leaf)))
        b0record=json.loads(get('run_record.json'));b0manifest=json.loads(get('analysis_manifest.json'))
        for leaf,key in [('best.pt','best_checkpoint_sha256'),('history.json','history_sha256'),('evaluation_epoch_012.json','evaluation_sha256')]:
            actual=hashlib.sha256(get(leaf)).hexdigest();assert actual==b0manifest[key],leaf
        assert hashlib.sha256(get('retrieval-SfM-120k.pkl')).hexdigest()==b0record['dataset_identity']['metadata_sha256']
        bh=json.loads(get('history.json'));histories['B0']=bh
        assert len(bh)==14 and min(bh,key=lambda r:r['val_loss'])['epoch']==12
        bev={}
        for n in names:
            if Path(n).name.startswith('evaluation_epoch_') and n.endswith('.json'):
                d=json.loads(z.read(n));epoch=int(Path(n).stem.rsplit('_',1)[1]);bev[epoch]=d
                for dataset,v in d['datasets'].items():
                    assert len(v['aps'])==55 and abs(statistics.mean(v['aps'])-v['map'])<1e-12
                    epochs.append(dict(model='B0',epoch=epoch,dataset=dataset,map=v['map'],source_archive=archive.name,source_entry=n,selection='best_validation' if epoch==12 else 'available_epoch'))
        for ds in DATASETS:
            for epoch,selection in [(12,'best_validation'),(14,'last_completed_in_snapshot')]:
                clean.append(dict(model='B0',selection=selection,epoch=epoch,dataset=ds,map=bev[epoch]['datasets'][ds]['map'],train_precision='FP32',eval_precision='FP32',protocol='legacy_ok_junk',postprocessing='cosine',checkpoint_available=epoch==12,source=archive.name,comparability='historical_shared_data_protocol_different_training_budget_runtime'))
        inventories.append(dict(id='B0',archive=archive.name,archive_sha256=sha(archive),best_epoch=12,best_checkpoint_sha256=b0manifest['best_checkpoint_sha256'],metadata_sha256=b0record['dataset_identity']['metadata_sha256'],source_commit=b0record['source_commit'],training_precision='FP32',completed_epochs=14,planned_epochs=100,has_corruption=False,has_posttraining=False))
        assert b0manifest['evidence'].startswith('saved')
        # Preserve the supplied baseline tables for traceability; no code from ZIP is executed.
        for leaf in ['map_by_epoch.csv','training_history.csv','run_configuration.csv','best_epoch_summary.csv']:
            (ROOT/('B0_'+leaf)).write_bytes(get(leaf))
    manifest=json.loads((BUNDLE/'models_manifest.json').read_text())
    for i,model in enumerate(['run1','run2']):
        info=manifest['models'][i];source=PROJECT/info['archive'];assert sha(source)==info['archive_sha256']
        h=json.loads((BUNDLE/(info['id']+'_history.json')).read_text());assert len(h)==100;histories[model]=h
        best=min(h,key=lambda r:r['val_loss'])['epoch']
        assert info['training_identity']['data']['metadata_sha256']==b0record['dataset_identity']['metadata_sha256']
        assert info['training_identity']['weights_sha256']==b0record['dataset_identity']['weights_sha256']
        with zipfile.ZipFile(source) as z:
            names=z.namelist();e=json.loads(z.read(next(n for n in names if n.endswith('/evaluation_best.json'))))
            for ds in DATASETS:
                assert len(e[ds]['aps'])==55 and abs(statistics.mean(e[ds]['aps'])-e[ds]['map'])<1e-12
                assert e[ds]['metadata_sha256']==b0record['dataset_identity'][ds]['ground_truth_sha256']
                clean.append(dict(model=model,selection='best_validation',epoch=best,dataset=ds,map=e[ds]['map'],train_precision='BF16_autocast_FP32_GeM_loss',eval_precision='BF16_autocast_FP32_GeM',protocol='legacy_ok_junk',postprocessing='cosine',checkpoint_available=any(n.endswith('/best.pt') or n.endswith(f'/best_epoch_{best:03d}.pt') for n in names),source=source.name,comparability='historical_shared_data_protocol_different_training_budget_runtime'))
            for n in names:
                if Path(n).name.startswith('evaluation_epoch_') and n.endswith('.json'):
                    d=json.loads(z.read(n));epoch=int(Path(n).stem.rsplit('_',1)[1])
                    for ds,v in d.items():
                        epochs.append(dict(model=model,epoch=epoch,dataset=ds,map=v['map'],source_archive=source.name,source_entry=n,selection='available_epoch'))
        inventories.append(dict(id=model,archive=source.name,archive_sha256=info['archive_sha256'],latest_checkpoint_sha256=info['checkpoint_sha256'],best_epoch=best,metadata_sha256=info['training_identity']['data']['metadata_sha256'],runtime_sha256=info['training_identity']['runtime_sha256'],training_precision=info['config']['precision'],completed_epochs=100,planned_epochs=100,has_corruption=True,has_posttraining=True))
    post=list(csv.DictReader((POST/'summary.csv').open()));assert len(post)==40
    for r in post:
        model='run1' if r['model']=='run1_aug_epoch100' else 'run2' if r['model']=='run2_aug_attention_epoch100' else 'fusion'
        r.update(model_short=model,epoch=100,eval_precision='FP32',source=str((POST/'summary.csv').relative_to(PROJECT)))
        if r['method']=='cosine':
            clean.append(dict(model=model,selection='final_epoch_100',epoch=100,dataset=r['dataset'],map=float(r['clean_map']),train_precision='BF16_autocast_FP32_GeM_loss',eval_precision='FP32',protocol='legacy_ok_junk',postprocessing='cosine',checkpoint_available=True,source=r['source'],comparability='shared_posttraining_protocol'))
    coverage=json.loads((POST/'coverage.json').read_text());assert coverage['status']=='COMPLETE' and coverage['actual']==68200
    assert json.loads((POST/'report_status.json').read_text())['status']=='COMPLETE'
    with (POST/'per_query.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==73700
    with (POST/'direct_end_to_end_latency.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==1120
    for model,h in histories.items():
        best=min(h,key=lambda r:r['val_loss']);times=[r.get('seconds_train_val',r.get('train_val_seconds')) for r in h]
        assert all(t is not None for t in times)
        training.append(dict(model=model,completed_epochs=len(h),best_epoch=best['epoch'],best_val_loss=best['val_loss'],final_val_loss=h[-1]['val_loss'],first_train_loss=h[0]['train_loss'],final_train_loss=h[-1]['train_loss'],mean_train_val_minutes=statistics.mean(times)/60,sum_train_val_hours=sum(times)/3600,training_precision='FP32' if model=='B0' else 'BF16_autocast_FP32_GeM_loss',note='train+val timing only; different hardware/runtime/precision; not an algorithm-only speed comparison'))
    training_rows=[dict(r,model=model) for model,h in histories.items() for r in h]
    write_csv('training_summary.csv',training);write_csv('all_training_history.csv',training_rows)
    write_csv('clean_model_comparison.csv',clean);write_csv('available_epoch_evaluations.csv',sorted(epochs,key=lambda r:(r['model'],r['epoch'],r['dataset'])))
    write_csv('all_posttraining_results.csv',post)
    grouped=defaultdict(list)
    for r in post:grouped[r['model_short'],r['method']].append(r)
    macro=[]
    for (model,method),rs in grouped.items():
        assert {r['dataset'] for r in rs}==set(DATASETS)
        macro.append(dict(model=model,method=method,clean_map=statistics.mean(float(r['clean_map']) for r in rs),corrupted_map=statistics.mean(float(r['corrupted_map']) for r in rs),note='equal dataset mean; per-dataset worst_map remains separate'))
    write_csv('posttraining_macro.csv',macro)
    baselines={r['dataset']:r['map'] for r in clean if r['model']=='B0' and r['selection']=='best_validation'}
    deltas=[]
    for r in clean:
        if r['selection']=='best_validation':
            deltas.append(dict(model=r['model'],epoch=r['epoch'],dataset=r['dataset'],delta_vs_B0_best_pp=(r['map']-baselines[r['dataset']])*100,comparison='descriptive_historical_not_controlled_ablation'))
    write_csv('historical_best_deltas.csv',deltas)
    def cm(m,s,ds):return next(r['map'] for r in clean if r['model']==m and r['selection']==s and r['dataset']==ds)
    selected=[('B0','best_validation','B0, best epoch 12'),('B0','last_completed_in_snapshot','B0, epoch 14'),('run1','best_validation','Run 1 AUG, best epoch 27'),('run2','best_validation','Run 2 AUG+Attention, best epoch 28'),('run1','final_epoch_100','Run 1 AUG, epoch 100'),('run2','final_epoch_100','Run 2 AUG+Attention, epoch 100')]
    clean_table=table(['Model / checkpoint','Oxford clean mAP','Paris clean mAP'],[[label,*[f'{cm(m,s,d)*100:.2f}%' for d in DATASETS]] for m,s,label in selected])
    lines=['# Tổng hợp toàn bộ kết quả B0, Run 1, Run 2 và post-training','',
'Tổng hợp từ ba ZIP và outputs Kaggle đã hoàn thành. Không chạy inference hoặc training mới trong bước này. Số liệu dạng phần trăm trong báo cáo, CSV lưu metric [0,1] trừ cột có hậu tố `_pp`.',
'', '## 1. Phạm vi bằng chứng và tính tương thích','',
'- B0: baseline VGG16–GeM, không AUG/Attention; best epoch 12 chọn theo validation loss; snapshot hoàn tất 14 epoch trong kế hoạch 100. best.pt epoch12 có thật và khớp SHA256 manifest. latest epoch14 được nhắc trong inventory nhưng không có checkpoint trong ZIP mới.',
'- Run 1: AUG; best validation epoch27, hoàn tất100. ZIP không chứa best.pt/best_epoch_027.pt, nhưng có saved evaluation best và latest epoch100.',
'- Run 2: AUG + Attention; best validation epoch28, hoàn tất100. ZIP có best.pt epoch28 và latest epoch100. Không có model Attention-only trong bộ kết quả.',
'- Cả ba dùng cùng SfM subset:15.953 ảnh train/100 clusters,739 validation/20 clusters; metadata SHA256 khớp. Initial Caffe weights hash, benchmark ground-truth hashes và legacy AP protocol khớp.',
'- B0 dùng FP32; saved best của hai run dùng BF16 backbone/FP32 GeM; post-training final100 trích lại FP32, tắt TF32. Runtime, hardware, số epoch và quy trình RNG/mining không được kiểm soát hoàn toàn giống nhau. Cùng dataset không đồng nghĩa là ablation chỉ khác AUG/Attention.',
'- B0 chỉ có saved clean evaluation. Chưa có corruption, QE/geometry/fusion/diffusion cho B0. Ô thiếu được ghi thiếu, không gán0 và không thay bằng số của encoder khác.',
'', '## 2. Toàn bộ model trước post-training', '',clean_table,'',
'Bảng trên là đối chiếu các kết quả đã có, không phải một phép so cùng training budget và eval precision. Best checkpoint được chọn theo validation loss, không lấy epoch có test mAP cao nhất riêng từng dataset.',
'', 'B0 epoch8 đạt Paris78,20% trong saved history, cao hơn epoch12 trên riêng Paris, nhưng checkpoint được chọn vẫn là epoch12. Không ghép Oxford epoch12 với Paris epoch8 rồi gọi đó là kết quả của một checkpoint.',
'', '## 3. Chênh lệch best so với B0 best','',table(['Model best','Oxford Δ điểm %','Paris Δ điểm %'],[[LABEL[m],*[f'{(cm(m,"best_validation",d)-baselines[d])*100:+.2f}' for d in DATASETS]] for m in ['run1','run2']]),'',
'B0 có clean mAP cao nhất trong các cấu hình đã được cung cấp/đã chạy. Các con số hiện có chưa chứng minh AUG, AUG+Attention hoặc post-training trên final100 vượt B0. Chênh lệch lịch sử chưa đủ để quy nguyên nhân riêng cho AUG hay attention.',
'', '## 4. Toàn bộ post-training: mỗi encoder một bảng','',
'Các bảng sau chỉ dùng epoch100, FP32, main hyperparameters khóa trước chạy. Corrupted mAP là trung bình đều18 condition (6 họ×3 mức), sau khi trung bình realization cho noise/occlusion. Không chọn lại tham số từ sensitivity để làm bảng chính.']
    method_labels={'cosine':'Cosine','AQE':'AQE','alpha_QE':'Alpha-QE','gated_QE':'Gated QE','local_rerank':'Local re-ranking','geometric_rerank':'Geometric re-ranking','verified_QE':'Geometry-verified QE','diffusion_mutual':'Diffusion mutual','diffusion_union':'Diffusion union','RRF':'RRF','score_fusion':'Average-score fusion'}
    order=['cosine','AQE','alpha_QE','gated_QE','local_rerank','geometric_rerank','verified_QE','diffusion_mutual','diffusion_union']
    for model in ['run1','run2','fusion']:
        methods=order if model!='fusion' else ['score_fusion','RRF'];tr=[]
        for method in methods:
            rs=[next(r for r in post if r['model_short']==model and r['method']==method and r['dataset']==d) for d in DATASETS]
            tr.append([method_labels[method],*[f'{float(r[k])*100:.2f}%' for r in rs for k in ['clean_map','corrupted_map']]])
        lines+=['','### '+LABEL.get(model,'Fusion Run 1 + Run 2'),'',table(['Phương pháp','Oxford clean','Oxford corrupted','Paris clean','Paris corrupted'],tr)]
    lines+=['','## 5. Lợi ích search trong nội bộ hai model hiện có','',
'- Run1 cosine macro:clean69,14%, corrupted66,82%; alpha-QE:71,17%/69,34%; verified QE:72,96%/71,05%. Mức tăng corrupted lần lượt +2,52 và +4,23 điểm phần trăm. Đây là gain so với Run1 epoch100 cosine, không phải gain so với B0.',
'- Run1 + verified QE clean78,94% Oxford/66,97% Paris vẫn dưới B0 clean81,06%/77,12%, chênh -2,12/-10,16 điểm phần trăm theo số liệu đã lưu.',
'- AQE gần ngang alpha-QE, chưa có bằng chứng thống kê một cái hơn rõ. Gate tau0,5 mở100%, giữ đủ10feedback, nên main output giống alpha-QE; chưa tạo bảo vệ query drift.',
'- Local-only re-ranking tốt hơn geometric re-ranking ở hai dataset cho cả hai encoder. Geometry hữu ích nhất khi lọc feedback cho verified QE trong các cấu hình đã chạy.',
'- Fusion không vượt Run1 cosine dù dùng hai encoder. Diffusion union/mutual giảm mạnh; hầu hết đã hội tụ, không thể quy đơn giản cho thiếu iterations. Cần audit graph/seed/normalization trước khi tuning; chưa chứng minh nguyên nhân.',
'- Blur nặng là điều kiện thấp nhất của Run1 cosine. Worst-condition Oxford48,33→60,97% với verifiedQE; Paris54,71→58,60%. Không có số robustness B0 để kết luận AUG cải thiện robustness hơn B0.',
'', '## 6. Chi phí đã đo','',
'Nguồn: direct end-to-end benchmark, clean + severe mỗi họ,2query/dataset/condition,2repeats;28mẫu mỗi model/dataset/method. Gallery/index/local-gallery cache được phép; query decode/corrupt/encode và query local extraction nằm trong timer. Timing median/p95 là mô tả trên mẫu nhỏ, không kết luận từ việc medianQE đôi khi thấp hơn cosine. B0 không có latency inference tương đương trong ZIP.',
'',table(['Run1 phương pháp','Oxford median end-to-end','Paris median end-to-end'],[[method_labels[m],*[f'{float(next(r for r in csv.DictReader((POST/"end_to_end_summary.csv").open()) if r["model"]=="run1_aug_epoch100" and r["method"]==m and r["dataset"]==d)["median"]):.1f} ms' for d in DATASETS]] for m in ['cosine','AQE','alpha_QE','local_rerank','geometric_rerank','verified_QE','diffusion_mutual']]),
'', 'QE retrieval-only thêm khoảng3–5ms/query; verifiedQE tốn khoảng5,8–8,5giây end-to-end. Fusion cần hai lần trích encoder. CSV end_to_end_summary.csv và per-query timings được giữ nguyên trong thư mục post-training.',
'', '## 7. Training và chọn checkpoint','',table(['Model','Epoch hoàn tất','Best epoch','Best val loss','Final val loss','Mean train+val phút/epoch'],[[LABEL[r['model']],r['completed_epochs'],r['best_epoch'],f'{r["best_val_loss"]:.6f}',f'{r["final_val_loss"]:.6f}',f'{r["mean_train_val_minutes"]:.2f}'] for r in training]),'',
'Runtime/epoch không phải đối chứng tốc độ thuật toán: B0 FP32 trên T4 và runtime gốc; các run mới BF16/FP32 trên A100 và runtime đã tối ưu. Timing train+val không gồm đầy đủ benchmark/upload/overhead của toàn phiên.',
'', 'Hai run100 có validation loss cuối cao hơn best trong khi train loss tiếp tục giảm: dấu hiệu overfit theo validation. Tăng epoch không đảm bảo retrieval tốt hơn. B0 chỉ tăng val nhẹ sau epoch12 trong2epoch quan sát; không khẳng định quá mức về toàn bộ training chưa chạy.',
'', '## 8. Kết luận và phần còn thiếu','',
'1. Nếu mục tiêu clean retrieval: B0 best epoch12 hiện là mốc mạnh nhất có bằng chứng. Khuyến nghị trước đây Run1+QE chỉ áp dụng trong hai encoder epoch100 khi chưa nhận B0; không còn là khuyến nghị chung cho toàn bộ model.',
'2. Nếu buộc dùng hai encoder final100 đã có: Run1+AQE/alpha-QE là cấu hình cân bằng. VerifiedQE là lựa chọn ưu tiên độ chính xác trong nhóm này và chấp nhận vài giây/query.',
'3. Chưa chứng minh cải tiến model vượt B0, chưa chứng minh robust hơn B0 và chưa tách đóng góp Attention-only. Không viết kết luận AUG/Attention thành công từ gain post-training nội bộ.',
'4. Thí nghiệm tiếp theo đáng làm: cùng FP32/preprocessing cho B0 best12 và hai best27/28; sau đó chạy cosine, QE và corruption chung. Có B0 best và Run2 best; còn thiếu Run1 best27. Không cần train lại để bắt đầu đánh giá B0.',
'5. Không chạy thử nghiệm mới hoặc upload Kaggle trong lần tổng hợp này. Muốn kết luận riêng AUG/Attention cần thêm đối chứng cùng budget/precision/selection và nhiều training seed.',
'', '## 9. Tệp và độ đầy đủ','',
'- clean_model_comparison.csv: clean, epoch, selection, precision và checkpoint availability.',
'- all_posttraining_results.csv: đủ40dòng model/method/dataset, gồm metric và latency gốc.',
'- posttraining_macro.csv: trung bình đều hai dataset; không thay bảng riêng dataset.',
'- available_epoch_evaluations.csv: mọi saved evaluation JSON thực sự nằm trong baZIP; các epoch không có JSON không được nội suy.',
'- training_summary.csv, all_training_history.csv:14+100+100=214epoch records.',
'- B0_*.csv: bảng gốc từ ZIP mới để truy vết.',
'- source_manifest.json: SHA256 và kiểm tra metadata/checkpoint/report coverage.',
'- Không gộp uncertainty qua các query-corruption thành multi-seed confidence interval. Kết quả exploratory trên một training seed42.']
    (ROOT/'TONG_HOP_TAT_CA_KET_QUA.md').write_text('\n'.join(lines)+'\n')
    validation=dict(b0_checkpoint_hash_verified=True,b0_history_hash_verified=True,b0_evaluation_hash_verified=True,shared_sfm_metadata=True,shared_ground_truth=True,b0_best_epoch=12,post_main_cells=coverage['actual'],post_rows=73700,direct_latency_samples=1120,clean_rows=len(clean),post_summary_rows=len(post),training_rows=len(training_rows),inputs=inventories,generated_without_inference=True)
    (ROOT/'source_manifest.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    # Standard plotting only; missing robustness for B0 is never plotted as zero.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(selected));width=.36
    for j,d in enumerate(DATASETS):ax.bar(x+(j-.5)*width,[cm(m,s,d)*100 for m,s,_ in selected],width,label=d)
    ax.set_xticks(x,[r[2] for r in selected],rotation=18,ha='right');ax.set_ylabel('Clean legacy mAP (%)');ax.set_ylim(0,100);ax.legend();ax.set_title('Saved checkpoints: descriptive comparison, different training budgets / precision');fig.tight_layout();fig.savefig(ROOT/'clean_checkpoint_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(model,h) in zip(axes,histories.items()):
        ax.plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label='Train');ax.plot([r['epoch'] for r in h],[r['val_loss'] for r in h],label='Validation');best=min(h,key=lambda r:r['val_loss']);ax.axvline(best['epoch'],color='gray',linestyle='--',label=f"Best epoch {best['epoch']}");ax.set_title(LABEL[model]);ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'training_all_models.png',dpi=160);plt.close(fig)
    from add_sensitivity import main as add_sensitivity
    add_sensitivity()
    print('PASS consolidation:',json.dumps({k:v for k,v in validation.items() if k!='inputs'}))

if __name__=='__main__':main()

"""Preserve every clean sensitivity outcome; never retune the main test table."""
import csv,json,hashlib,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
POST=PROJECT/'artifacts/kaggle-posttraining/outputs'

def write(name,rows):
    fields=list(rows[0])
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    groups=defaultdict(list)
    source=POST/'iir-post100-report/unpacked/sensitivity_clean.csv'
    for r in csv.DictReader(source.open()):groups[r['model'],r['dataset'],r['method'],r['config_hash']].append(float(r['ap']))
    assert len(groups)==100 and sum(map(len,groups.values()))==5500
    configs={}
    for p in POST.rglob('results/*.json'):
        d=json.loads(p.read_text())
        if d['phase']!='sensitivity':continue
        config=d['config'];key=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest();configs[key]=config
    rows=[]
    for (model,dataset,method,h),aps in sorted(groups.items()):
        assert len(aps)==55 and h in configs
        rows.append(dict(model=model,dataset=dataset,method=method,config_hash=h,config_json=json.dumps(configs[h],sort_keys=True),clean_map=statistics.mean(aps),queries=len(aps),phase='clean_sensitivity_exploratory'))
    write('all_clean_sensitivity.csv',rows)
    best={}
    for r in rows:
        key=r['model'],r['dataset'],r['method']
        if key not in best or r['clean_map']>best[key]['clean_map']:best[key]=r
    selected=[dict(r,selection_note='descriptive_max_on_test_sweep_not_independent_validation') for _,r in sorted(best.items())]
    write('descriptive_sensitivity_maxima.csv',selected)
    p=ROOT/'source_manifest.json';manifest=json.loads(p.read_text());manifest.update(sensitivity_groups=100,sensitivity_AP_rows=5500);p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report=ROOT/'TONG_HOP_TAT_CA_KET_QUA.md';s=report.read_text();marker='\n## 10. Sensitivity clean đã chạy'
    if marker in s:s=s.split(marker)[0]
    s+=marker+'''

Đã tổng hợp đủ **100 dòng model × method × config × dataset**, tương ứng **5.500 AP query**, vào `all_clean_sensitivity.csv`; mỗi dòng gồm config JSON thật và hash. `descriptive_sensitivity_maxima.csv` chỉ mô tả điểm cao nhất trong từng nhóm sweep, không phải chọn model bằng development set.

Một số điểm đáng chú ý của Run1:

- Alpha-QE lambda1, giữ k10/alpha3: Oxford77,30%, Paris66,56% clean, cao hơn main lambda0,5. Chưa có full corruption của lambda1 nên không thay kết luận robustness bằng số này.
- Local re-ranking K100/eta0,5: Oxford78,85%, Paris66,94% clean; cần nhiều candidate hơn K50. Local K50/eta0,25 đạt Oxford78,95% trong sweep.
- Các sweep đã chạy vẫn chưa vượt B0 best clean81,06%/77,12%. Không tự chọn một cấu hình khác cho mỗi dataset rồi gọi đó là cấu hình chung đã kiểm chứng.

Bảng main phía trên tiếp tục dùng tham số khóa trước chạy; không trộn sensitivity maxima vào bảng robustness.
'''
    report.write_text(s)
    print('PASS sensitivity consolidation: 100 groups, 5500 query APs, real config hashes resolved')

if __name__=='__main__':main()

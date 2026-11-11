"""Durable local coordinator. All authenticated commands go through the existing bridge.
No API credentials are read by this process. Does not retry uploads/pushes ambiguously.
"""
import argparse,json,subprocess,time,sys,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from build_jobs import build,OWNER,ROOT
BRIDGE='/home/hoang/.local/bin/kaggle-account'
STATE=ROOT/'state';OUTPUTS=ROOT/'outputs';STATE.mkdir(exist_ok=True);OUTPUTS.mkdir(exist_ok=True)


def save(path,data):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def call(alias,args):
    result=subprocess.run([BRIDGE,'run',alias,'--',*args],capture_output=True,text=True,timeout=215)
    output=(result.stdout or '')+(result.stderr or '')
    with (STATE/'bridge.log').open('a') as f:f.write(time.strftime('%FT%T')+' '+repr(args)+'\n'+output+'\n')
    return result.returncode,output


def remote_status(alias,slug):
    code,text=call(alias,['kernels','status',OWNER+'/'+slug])
    lower=text.lower()
    for state in ['complete','error','cancelled','canceled','running','queued']:
        # CLI returns e.g. has status "KernelWorkerStatus.COMPLETE".
        if state in lower:return state,text
    return 'unknown',text


def execute(alias,stage,dependencies=(),**kwargs):
    folder=build(stage,dependencies,**kwargs);slug=folder.name;path=STATE/(slug+'.json')
    saved=json.loads(path.read_text()) if path.exists() else dict(slug=slug,stage=stage,dependencies=list(dependencies),status='NEW')
    if saved['status']=='COMPLETE':return slug
    if saved['status']=='FAILED':raise RuntimeError('Previously failed job needs inspection: '+slug)
    if saved['status']=='NEW':
        # Persist intent before the request so a crash can never blindly repush.
        saved.update(status='SUBMITTING',started=time.time(),config=kwargs);save(path,saved)
        args=['kernels','push','-p',str(folder),'-t','43200']
        if stage in ['extract','latency']:args+=['--accelerator','NvidiaTeslaT4']
        code,text=call(alias,args);saved.update(status='SUBMITTED' if code==0 else 'SUBMIT_UNCERTAIN',push_message=text[-1500:]);save(path,saved)
        print('SUBMITTED',slug,text.strip(),flush=True)
    while True:
        status,text=remote_status(alias,slug)
        if status in ['error','cancelled','canceled']:
            # Download available error logs; never resubmit automatically.
            dest=OUTPUTS/slug;dest.mkdir(exist_ok=True)
            call(alias,['kernels','output',OWNER+'/'+slug,'-p',str(dest)])
            saved.update(status='FAILED',remote=status,error=text);save(path,saved);raise RuntimeError(text)
        if status=='complete':break
        if status=='unknown' and time.time()-saved.get('started',time.time())>900:
            saved.update(status='FAILED',remote=status,error=text);save(path,saved);raise RuntimeError('Cannot resolve remote state; no retry: '+text)
        old=saved.get('remote');saved.update(remote=status,last_checked=time.time());save(path,saved)
        if old!=status:print('STATUS',slug,status,flush=True)
        time.sleep(45)
    dest=OUTPUTS/slug;dest.mkdir(exist_ok=True)
    code,text=call(alias,['kernels','output',OWNER+'/'+slug,'-p',str(dest),'--file-pattern','(artifact_manifest|job_status|job_config)[.]json$|[.]log$'])
    if code:raise RuntimeError('Download failed; completed remote job will not be repushed: '+text)
    artifacts=list(dest.rglob('artifact_manifest.json'))
    if len(artifacts)!=1:raise RuntimeError('Missing artifact transport manifest: '+slug)
    artifact=artifacts[0];transport=json.loads(artifact.read_text())
    import re
    for part in transport['parts']:
        target=artifact.parent/part['name']
        from artifacts_io import sha,unpack
        if target.exists() and target.stat().st_size==part['bytes'] and sha(target)==part['sha256']:continue
        code,text=call(alias,['kernels','output',OWNER+'/'+slug,'-p',str(dest),'--file-pattern',re.escape(part['name'])+'$'])
        if code:raise RuntimeError('Artifact-part download failed; retry coordinator, not push: '+text)
    unpack(artifact,dest/'unpacked')
    statuses=list(dest.rglob('job_status.json'))
    if len(statuses)!=1 or json.loads(statuses[0].read_text())['status']!='COMPLETE':
        saved.update(status='FAILED',error='Remote outputs missing successful job_status.json');save(path,saved);raise RuntimeError(saved['error']+': '+slug)
    if stage=='report':
        reports=list(dest.rglob('report_status.json'))
        if len(reports)!=1 or json.loads(reports[0].read_text())['status']!='COMPLETE':
            saved.update(status='FAILED',error='Report coverage/latency incomplete');save(path,saved);raise RuntimeError(saved['error'])
    saved.update(status='COMPLETE',finished=time.time(),output=str(dest));save(path,saved)
    print('COMPLETE',slug,flush=True);return slug


def main(alias):
    audit=execute(alias,'audit')
    extraction=execute(alias,'extract',[audit])
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={name:pool.submit(execute,alias,name,[extraction]) for name in ['qf','diffusion','local']}
        # Inspect every future; keep any successfully completed independent output.
        results={};errors=[]
        for name,future in jobs.items():
            try:results[name]=future.result()
            except Exception as error:errors.append(str(error))
        if errors:raise RuntimeError('\n'.join(errors))
    geometry=[]
    for dataset in ['oxford5k','paris6k']:
        pilot=list((OUTPUTS/results['local']).rglob('geometry_pilot_'+dataset+'.json'))
        assert len(pilot)==1
        count=json.loads(pilot[0].read_text())['shards']
        for index in range(count):geometry.append(dict(dataset=dataset,shard_index=index,shard_count=count))
    # At most three CPU shards plus one GPU latency job; use no other accounts.
    with ThreadPoolExecutor(max_workers=4) as pool:
        latency=pool.submit(execute,alias,'latency',[extraction,results['local'],results['diffusion']])
        futures=[]
        # Bound CPU concurrency independently of the GPU job.
        with ThreadPoolExecutor(max_workers=3) as cpupool:
            for spec in geometry:futures.append(cpupool.submit(execute,alias,'geometry',[extraction,results['local']],**spec))
            geoslugs=[];errors=[]
            for future in futures:
                try:geoslugs.append(future.result())
                except Exception as error:errors.append(str(error))
        try:latencyslug=latency.result()
        except Exception as error:errors.append(str(error))
        if errors:raise RuntimeError('\n'.join(errors))
    # Kaggle's input-source cap may be smaller than a large shard count. If so, the
    # report builder uses a merged private result dataset created by consolidate.py.
    deps=[extraction,results['qf'],results['local'],results['diffusion'],latencyslug]+geoslugs
    if len(deps)>15:
        from consolidate import consolidate
        dataset=consolidate(alias,deps)
        report=execute(alias,'report',[],result_dataset=dataset)
    else:report=execute(alias,'report',deps)
    save(STATE/'pipeline_status.json',dict(status='COMPLETE',report=OWNER+'/'+report,output=str(OUTPUTS/report)))

if __name__=='__main__':
    import fcntl
    lock=(STATE/'coordinator.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    parser=argparse.ArgumentParser();parser.add_argument('--account',default='hoang1');args=parser.parse_args()
    try:main(args.account)
    except Exception as error:
        save(STATE/'pipeline_status.json',dict(status='FAILED',error=str(error),at=time.time()));raise

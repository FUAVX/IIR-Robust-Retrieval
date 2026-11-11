"""Measured, bounded GPU tuning of the exact colab_runtime training step.

All models are disposable. No training run checkpoint is read or written.
Recommendations are local numerical/performance checks, not convergence or mAP claims.
"""
import copy
import csv
import gc
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import colab_runtime as rt

TUNING_DEFAULTS = dict(prefetch_factor=2,cache_mode='off',cache_gib=0.,tuple_fusion=1)
TUNING_KEYS = ('precision','microbatch','max_batch_pixels','workers','extraction_window',
               'prefetch_factor','cache_mode','cache_gib','tuple_fusion')


def tuning(cfg):
    return {key:cfg.get(key,TUNING_DEFAULTS.get(key)) for key in TUNING_KEYS}


def release_incompatible_cache(cfg,data):
    """Do not let a previous cache allocation reject a cache-off/different trial."""
    paths = list(dict.fromkeys(map(str,data.get('cache_paths',data['part']['paths']))))
    desired = (cfg.get('cache_mode','off'),cfg['image_size'],cfg.get('cache_gib',0),hash(tuple(paths)))
    if rt.IMAGE_CACHE_KEY != desired:
        rt.prepare_image_cache([],dict(cfg,cache_mode='off',cache_gib=0.))
        cleanup()


def resource_plan(cfg,controls):
    memory = rt.system_memory()
    reserve = max(4*1024**3,int(memory['total_bytes']*controls.get('min_ram_free_fraction',.2)))
    payload = cfg['workers']*cfg.get('prefetch_factor',2)*max(35,cfg['extraction_window'])*3*cfg['image_size']**2*4
    old = rt.IMAGE_CACHE_INFO
    same = old.get('mode') == cfg.get('cache_mode','off')
    requested_cache = int(cfg.get('cache_gib',0)*1024**3)
    cache_addition = max(0,requested_cache-old.get('stored_bytes',0)) if same else requested_cache
    shm = os.statvfs('/dev/shm')
    shm_available = shm.f_bavail*shm.f_frsize
    reason = None
    if payload*1.2 > shm_available*.85:
        reason = 'Estimated prefetched tensors exceed /dev/shm headroom'
    if payload*2+cache_addition > max(0,memory['available_bytes']-reserve):
        reason = 'Estimated cache + shared/pinned prefetch queues exceed RAM headroom'
    return dict(passed=reason is None,reason=reason,queue_payload_gib=payload/1024**3,
                shm_available_gib=shm_available/1024**3,ram_available_gib=memory['available_bytes']/1024**3,
                reserve_gib=reserve/1024**3)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class UtilizationSampler:
    """Best-effort nvidia-smi sampler; absence of the command never fails a trial."""
    def __init__(self):
        self.stop_event, self.samples = threading.Event(), []
        self.memory_samples = []
        self.thread = None

    def start(self):
        def loop():
            last_uss = -math.inf
            uss_last = 0
            while not self.stop_event.is_set():
                try:
                    import psutil
                    memory = rt.system_memory()
                    parent = psutil.Process()
                    rss = uss = 0
                    measure_uss = time.monotonic()-last_uss >= 5
                    for process in [parent]+parent.children(recursive=True):
                        try:
                            info = process.memory_full_info() if measure_uss else process.memory_info()
                            rss += info.rss
                            uss += getattr(info,'uss',0)
                        except (psutil.NoSuchProcess,psutil.AccessDenied):
                            continue
                    if measure_uss:
                        last_uss,uss_last = time.monotonic(),uss
                    self.memory_samples.append(dict(memory,rss=rss,uss=uss_last))
                except (OSError,ImportError):
                    pass
                try:
                    result = subprocess.run(['nvidia-smi', '--id=0',
                        '--query-gpu=utilization.gpu,memory.used,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                    if not result.returncode:
                        values = [float(x.strip()) for x in result.stdout.strip().split(',')]
                        if len(values) == 3:
                            self.samples.append(values)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                self.stop_event.wait(1)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        result = dict(gpu_utilization_mean=None,power_watts_mean=None)
        if self.samples:
            result.update(gpu_utilization_mean=float(np.mean([s[0] for s in self.samples])),
                          power_watts_mean=float(np.mean([s[2] for s in self.samples])),nvidia_smi_samples=len(self.samples))
        if self.memory_samples:
            result.update(ram_available_min_gib=min(s['available_bytes'] for s in self.memory_samples)/1024**3,
                          ram_available_min_fraction=min(s['available_bytes']/s['total_bytes'] for s in self.memory_samples),
                          ram_system_used_peak_gib=max(s['used_bytes'] for s in self.memory_samples)/1024**3,
                          process_tree_uss_peak_gib=max(s['uss'] for s in self.memory_samples)/1024**3,
                          process_tree_rss_peak_gib=max(s['rss'] for s in self.memory_samples)/1024**3)
        return result


def select_paths(part, count, header_scan=512):
    """Same deterministic sample across trials, enriched with large/aspect-extreme images."""
    paths = part['paths']
    rng = random.Random(2026)
    candidates = rng.sample(paths, min(header_scan, len(paths)))
    dimensions = []
    for path in candidates:
        with Image.open(path) as image:
            w, h = image.size
        scale = min(1., 1024/max(w,h))
        dimensions.append((path, int(w*scale), int(h*scale)))
    largest = sorted(dimensions, key=lambda x: x[1]*x[2], reverse=True)
    wide = sorted(dimensions, key=lambda x: x[1]/x[2], reverse=True)
    tall = sorted(dimensions, key=lambda x: x[2]/x[1], reverse=True)
    chosen = list(dict.fromkeys([p for p,_,_ in largest[:16]+wide[:8]+tall[:8]] + candidates))[:count]
    worst = largest[0][0]
    return chosen, worst, dict(header_images=len(candidates), extraction_images=len(chosen),
                              largest_sampled_shape=[largest[0][2], largest[0][1]],
                              note='Stress image is the largest in the header sample, not proof for every dataset shape.')


def fixed_augmented_batch(part, tuples, cfg):
    rt.seed_all(1007)
    dataset = rt.TupleImages(part, tuples, cfg)
    return [dataset[i] for i in range(cfg['batch_size'])]


def gradient_snapshot(model, batch, cfg):
    """One unchanged-weights FP32/BF16/TF32 comparison, no optimizer update."""
    model.train()
    model.zero_grad(set_to_none=True)
    losses,descriptors = rt.backward_tuples(model,batch,cfg,collect_descriptors=True)
    loss_sum = sum(float(loss.cpu()) for loss in losses)
    gradients = {name: p.grad.detach().float().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients.values()):
        raise FloatingPointError('Numerical probe found non-finite/missing gradients')
    return dict(loss=loss_sum, descriptors=torch.cat(descriptors, dim=1), gradients=gradients)


def compare_snapshots(reference, candidate, limits):
    if reference['gradients'].keys() != candidate['gradients'].keys():
        return dict(passed=False, reason='Gradient parameter set changed')
    dot = norm_ref = norm_cur = 0.
    attention_dot = attention_ref = attention_cur = 0.
    for name, first in reference['gradients'].items():
        second = candidate['gradients'][name]
        a, b = first.double().flatten(), second.double().flatten()
        ab, aa, bb = float(torch.dot(a,b)), float(torch.dot(a,a)), float(torch.dot(b,b))
        dot, norm_ref, norm_cur = dot+ab, norm_ref+aa, norm_cur+bb
        if name.startswith('attention.'):
            attention_dot, attention_ref, attention_cur = attention_dot+ab, attention_ref+aa, attention_cur+bb
    def cosine(ab, aa, bb):
        return ab/math.sqrt(aa*bb) if aa*bb else (1. if aa == bb else 0.)
    grad_cos = cosine(dot,norm_ref,norm_cur)
    attention_cos = cosine(attention_dot,attention_ref,attention_cur)
    ratio = math.sqrt(norm_cur/max(norm_ref,1e-30))
    desc_cos = torch.nn.functional.cosine_similarity(reference['descriptors'], candidate['descriptors'], dim=0).min().item()
    loss_error = abs(candidate['loss']-reference['loss'])/max(abs(reference['loss']),1e-4)
    passed = (loss_error <= limits['max_loss_relative_error'] and grad_cos >= limits['min_gradient_cosine']
              and attention_cos >= limits['min_gradient_cosine'] and desc_cos >= limits['min_descriptor_cosine']
              and 0.5 <= ratio <= 2.)
    return dict(passed=bool(passed), loss_relative_error=loss_error, gradient_cosine=grad_cos,
                attention_gradient_cosine=attention_cos, gradient_norm_ratio=ratio,
                descriptor_min_cosine=desc_cos,
                note='Local fixed-batch check only; not a long-run convergence or mAP guarantee.')


def tune_signature(cfg):
    return tuple(tuning(cfg)[key] for key in TUNING_KEYS)


def choose_common(rows):
    """Require a successful row for BOTH variants before recommending common settings."""
    groups = defaultdict(lambda:defaultdict(list))
    latest_status = {}
    for row in rows:
        signature = tune_signature(row['config'])
        latest_status[(signature,bool(row['attention']))] = row.get('status')
        if row.get('status') == 'PASS':
            groups[signature][bool(row['attention'])].append(row)
    eligible = []
    for signature,group in groups.items():
        if set(group) != {False,True} or any(latest_status.get((signature,a)) != 'PASS' for a in [False,True]):
            continue
        pair = {}
        for att,items in group.items():
            repeated = [r for r in items if r.get('stage') == 'repeat_finalists']
            pool = repeated or items
            ordered = sorted(pool,key=lambda r:r['estimated_epoch_seconds'])
            pair[att] = dict(ordered[len(ordered)//2],repeat_count=len(repeated))
        eligible.append(pair)
    repeated_pairs = [p for p in eligible if all(p[a]['repeat_count'] for a in [False,True])]
    eligible = repeated_pairs or eligible
    if not eligible:
        return None
    pair = min(eligible, key=lambda p: sum(row['estimated_epoch_seconds'] for row in p.values()))
    return dict(config=tuning(pair[False]['config']),
                per_variant={str(key):value for key,value in pair.items()},
                score_seconds=sum(row['estimated_epoch_seconds'] for row in pair.values()))


def estimate_epoch(extraction_rate, tuple_rate, rank_train, rank_val, checkpoint_seconds=0.):
    mining = 15953/extraction_rate + rank_train
    training = 2000/tuple_rate
    validation = 739/extraction_rate + rank_val
    return dict(mining_seconds=mining, gradient_training_seconds=training,
                validation_seconds=validation, checkpoint_seconds=checkpoint_seconds,
                total_seconds=mining+training+validation+checkpoint_seconds,
                note='Extrapolated from samples; excludes benchmark unless separately added. Full-pool sorting measured at actual dimensions.')


def measure_rank(cfg, part, queries, poolsize, device):
    """Time real matrix dimensions/sort/CPU rank scan, using synthetic descriptors."""
    rt.configure_acceleration(cfg)
    generator = torch.Generator(device=device).manual_seed(77)
    gallery = torch.randn(512,poolsize, generator=generator, device=device)
    query = torch.randn(512,queries, generator=generator, device=device)
    gallery = torch.nn.functional.normalize(gallery,dim=0)
    query = torch.nn.functional.normalize(query,dim=0)
    sync(device)
    start = time.perf_counter()
    ranks = torch.argsort(gallery.T@query, dim=0, descending=True).cpu().numpy()
    clusters = part['cluster'][:poolsize]
    for column in range(queries):
        seen = {clusters[column % len(clusters)]}
        for rank in ranks[:,column]:
            seen.add(clusters[int(rank)])
            if len(seen) == 6:
                break
    sync(device)
    elapsed = time.perf_counter()-start
    del gallery, query, ranks
    return elapsed


def trial_impl(cfg, data, weights, probe, reference, limits, controls, device, rank_times):
    rt.seed_all(cfg['seed'])
    model = rt.make_model(cfg, weights, device)
    numerical = compare_snapshots(reference, gradient_snapshot(model,probe,cfg), limits)
    if not numerical['passed']:
        return dict(status='REJECTED_NUMERICS', numerical=numerical)
    optimizer = rt.make_optimizer(model,cfg)
    model.zero_grad(set_to_none=True)
    # Warm up extraction outside measured repetitions.
    rt.extract(model,data['sample_paths'][:min(16,len(data['sample_paths']))],cfg)
    extraction_seconds = []
    for _ in range(controls['extraction_repeats']):
        sync(device)
        start = time.perf_counter()
        rt.extract(model,data['sample_paths'],cfg)
        sync(device)
        extraction_seconds.append(time.perf_counter()-start)
    extraction_rate = len(data['sample_paths'])/statistics.median(extraction_seconds)
    dataset = rt.TupleImages(data['part'],data['tuples'],cfg)
    loader = DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,drop_last=True,
                        **rt.loader_options(cfg),pin_memory=True,collate_fn=rt.identity_collate,
                        worker_init_fn=rt.seed_worker,generator=torch.Generator().manual_seed(123))
    iterator = iter(loader)
    model.train()
    for _ in range(controls['warmup_steps']):
        rt.optimizer_batch(model,optimizer,next(iterator),cfg)
    calls = [0]
    handle = model.features.register_forward_pre_hook(lambda *_: calls.__setitem__(0,calls[0]+1))
    waits, durations, step_losses = [], [], []
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(controls['measured_steps']):
        before = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter()-before)
        before = time.perf_counter()
        losses = rt.optimizer_batch(model,optimizer,batch,cfg)
        sync(device)
        durations.append(time.perf_counter()-before)
        step_losses.append(float(torch.stack(losses).sum().cpu()))
    total = time.perf_counter()-start
    handle.remove()
    tuples = controls['measured_steps']*cfg['batch_size']
    tuple_rate = tuples/total
    normal_peak = torch.cuda.max_memory_allocated(device)
    # The same largest sampled real image repeated creates a same-shape memory stress,
    # independent of how often exact shapes collide in ordinary tuples.
    worst = rt.ImageList([data['worst_path']],cfg['image_size'])[0]
    stress_batch = [[worst]*7 for _ in range(cfg['batch_size'])]
    torch.cuda.reset_peak_memory_stats(device)
    rt.optimizer_batch(model,optimizer,stress_batch,cfg)
    model.eval()
    with torch.no_grad():
        rt.forward_images(model,[worst]*cfg['microbatch'],cfg['microbatch'],cfg['max_batch_pixels'])
    sync(device)
    stress_peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    peak = max(normal_peak,stress_peak)
    if peak > controls['max_vram_fraction']*total_vram:
        return dict(status='REJECTED_VRAM_HEADROOM', numerical=numerical, peak_vram_gib=peak/1024**3)
    estimate = estimate_epoch(extraction_rate,tuple_rate,*rank_times)
    return dict(status='PASS',numerical=numerical,extraction_images_per_second=extraction_rate,
                extraction_seconds=extraction_seconds,training_tuples_per_second=tuple_rate,
                measured_step_seconds=durations,loader_wait_seconds=waits,
                loader_wait_fraction=sum(waits)/total,losses=step_losses,
                training_images_per_forward=tuples*7/max(1,calls[0]),
                peak_vram_gib=peak/1024**3,peak_reserved_gib=reserved/1024**3,
                stress_shape=list(worst.shape),estimated_epoch_seconds=estimate['total_seconds'],
                epoch_estimate=estimate)


def safe_trial(cfg, data, weights, probe, reference, limits, controls, device, rank_times):
    cleanup()
    release_incompatible_cache(cfg,data)  # Free stale host cache BEFORE checking headroom/monitoring.
    sampler = UtilizationSampler()
    sampler.start()
    started = time.monotonic()
    try:
        plan = resource_plan(cfg,controls)
        if not plan['passed']:
            row = dict(status='SKIPPED_HOST_PLAN',resource_plan=plan)
        else:
            cache_info = rt.prepare_image_cache(data.get('cache_paths',data['part']['paths']),cfg)
            row = trial_impl(cfg,data,weights,probe,reference,limits,controls,device,rank_times)
            row.update(cache=cache_info,resource_plan=plan)
    except Exception as error:
        traceback.print_exc()
        row = dict(status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'ERROR',
                   error_type=type(error).__name__,error=str(error))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    finally:
        utilization = sampler.finish()
        cleanup()
    if row.get('status') == 'PASS' and (
            utilization.get('ram_available_min_fraction',1) < controls.get('min_ram_free_fraction',.2)
            or utilization.get('ram_available_min_gib',math.inf) < 4):
        row['status'] = 'REJECTED_RAM_HEADROOM'
    return dict(row,config=copy.deepcopy(cfg),attention=cfg['attention'],
                wall_seconds=time.monotonic()-started,**utilization)


def confirm_impl(cfg, partitions, weights, controls, output, device):
    """Actual mini epoch, full validation, real Drive checkpoint, separate from main runs."""
    rt.seed_all(cfg['seed'])
    rt.prepare_image_cache(partitions['train']['paths']+partitions['val']['paths'],cfg)
    model = rt.make_model(cfg,weights,device)
    optimizer = rt.make_optimizer(model,cfg)
    mini = dict(cfg,query_size=controls['confirmation_queries'],pool_size=controls['confirmation_pool'])
    sync(device)
    started = time.perf_counter()
    loss,mining,steps = rt.train_epoch(model,optimizer,partitions['train'],mini,0)
    sync(device)
    train_total = time.perf_counter()-started
    started = time.perf_counter()
    val_loss,_ = rt.validate(model,partitions['val'],cfg)
    sync(device)
    val_seconds = time.perf_counter()-started
    if not math.isfinite(loss) or not math.isfinite(val_loss):
        raise FloatingPointError('Confirmation train/validation loss is non-finite')
    payload = dict(config=cfg,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   note='DISPOSABLE PROFILING CHECKPOINT. Not a training result and not resumable by train_run.')
    started = time.perf_counter()
    rt.save_checkpoint(Path(output)/f'probe_run{2 if cfg["attention"] else 1}.pt',payload,
                       '/content/iir_profile_checkpoint_staging')
    checkpoint_seconds = time.perf_counter()-started
    return dict(status='PASS',train_loss=loss,val_loss=val_loss,mini_queries=mini['query_size'],
                mini_pool=mini['pool_size'],optimizer_steps=steps,
                mini_mining_seconds=mining,mini_train_total_seconds=train_total,
                gradient_tuples_per_second=mini['query_size']/max(1e-6,train_total-mining),
                full_validation_seconds=val_seconds,drive_checkpoint_seconds=checkpoint_seconds)


def export_recommendation(selected, confirmations, controls, base_cfg, benchmarks):
    settings = selected['config']
    confirmed = all(confirmations.get(str(att),{}).get('status') == 'PASS' for att in [False,True])
    if controls.get('finalist_repeats',0):
        confirmed = confirmed and all(selected['per_variant'][str(att)].get('repeat_count',0)
                                     >= controls['finalist_repeats'] for att in [False,True])
    variants = {}
    for att in [False,True]:
        row = selected['per_variant'][str(att)]
        confirmation = confirmations.get(str(att),{})
        rate = row['extraction_images_per_second']
        estimate = dict(row['epoch_estimate'])
        if confirmation.get('status') == 'PASS':
            estimate['gradient_training_seconds'] = 2000/confirmation['gradient_tuples_per_second']
            estimate['validation_seconds'] = confirmation['full_validation_seconds']
            # Publication of best+latest+alias can cost up to roughly three verified copies.
            estimate['checkpoint_seconds'] = confirmation['drive_checkpoint_seconds']*3
        benchmark_images = sum(len(info['gt']['imlist'])+len(info['gt']['qimlist']) for info in benchmarks.values())
        benchmark_seconds = benchmark_images/rate
        # Confirmation replaces individual timings; refresh the displayed core total too.
        estimate['total_seconds'] = sum(estimate[key] for key in
            ['mining_seconds','gradient_training_seconds','validation_seconds','checkpoint_seconds'])
        per_epoch = estimate['total_seconds']+benchmark_seconds/controls['eval_every']
        initial_smoke_allowance = 0.5*3600+row.get('cache',{}).get('build_seconds',0)
        hours = controls['training_cu_per_run']/controls['cu_per_hour']
        epoch_capacity = max(0,int((hours*3600-initial_smoke_allowance-benchmark_seconds)/(per_epoch*1.25)))
        variants[str(att)] = dict(estimated_epoch_hours=per_epoch/3600,
                estimated_cu_per_epoch=per_epoch/3600*controls['cu_per_hour'],
                epochs_with_25pct_time_margin=epoch_capacity,
                target_12_epochs_estimated_cu=(initial_smoke_allowance+12*per_epoch+benchmark_seconds)/3600*controls['cu_per_hour'],
                planning_target_epochs=base_cfg['epochs'],
                planning_target_estimated_cu=(initial_smoke_allowance+base_cfg['epochs']*per_epoch+benchmark_seconds)/3600*controls['cu_per_hour'],
                breakdown=estimate,benchmark_seconds_estimated=benchmark_seconds)
    epoch_target = min(base_cfg['epochs'],*(v['epochs_with_25pct_time_margin'] for v in variants.values()))
    mapping = dict(PRECISION=settings['precision'],MICROBATCH=settings['microbatch'],
                   MAX_BATCH_PIXELS=settings['max_batch_pixels'],WORKERS=settings['workers'],
                   EXTRACTION_WINDOW=settings['extraction_window'],PREFETCH_FACTOR=settings['prefetch_factor'],
                   CACHE_MODE=settings['cache_mode'],CACHE_GIB=settings['cache_gib'],
                   TUPLE_FUSION=settings['tuple_fusion'],EPOCHS=epoch_target)
    return dict(status='CONFIRMED_LOCAL_PROBES' if confirmed else 'PROVISIONAL_NOT_CONFIRMED',
                settings=mapping,per_variant=variants,common_epoch_target=epoch_target,
                limitations=['Sample-based extrapolation, not a measured full training epoch.',
                             'Numerical probes and a mini epoch do not guarantee long-run convergence or mAP.',
                             '25% timing margin is a planning assumption, not a confidence interval.',
                             'CU costs exclude prior ZIP preparation, idle runtime, failures and unmeasured overhead.'])


def write_reports(output, rows):
    output = Path(output)
    rt.atomic_json(output/'trials.json',rows)
    with (output/'trials.csv').open('w',newline='') as stream:
        fields = ['stage','attention','status','precision','microbatch','workers','extraction_window',
                  'prefetch_factor','cache_mode','cache_gib','tuple_fusion',
                  'peak_vram_gib','extraction_images_per_second','training_tuples_per_second',
                  'training_images_per_forward','loader_wait_fraction','estimated_epoch_seconds',
                  'gpu_utilization_mean','ram_available_min_gib','ram_system_used_peak_gib',
                  'process_tree_uss_peak_gib','process_tree_rss_peak_gib','wall_seconds']
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row,**tuning(row['config']))
            writer.writerow({key:flat.get(key) for key in fields})


def run_profile(base_cfg,partitions,weights,benchmarks,output,controls,limits):
    if not torch.cuda.is_available():
        raise RuntimeError('This notebook benchmarks a real CUDA GPU; select A100 in Colab.')
    device = torch.device('cuda:0')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    budget_seconds = min(controls['max_minutes']*60,
                         controls['profile_cu_budget']/controls['cu_per_hour']*3600)
    def time_left():
        return budget_seconds-(time.monotonic()-started)
    def allowed(reserve=0):
        return time_left() > reserve
    rt.atomic_json(output/'profile_environment.json',dict(gpu=torch.cuda.get_device_name(0),
        vram_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
        torch=torch.__version__,cpus=len(os.sched_getaffinity(0)),controls=controls,limits=limits,
        note='Soft budget, checked between trials. An in-progress trial is never killed mid-kernel.'))
    paths,worst,sampling = select_paths(partitions['train'],controls['extraction_images'],controls.get('header_scan',512))
    rt.atomic_json(output/'sampling.json',dict(sampling,paths=paths,worst=worst))
    contexts,rank_cache,rows = {},{},[]
    baseline = dict(base_cfg,precision='FP32',microbatch=1,max_batch_pixels=1024**2,
                    workers=0,extraction_window=8)
    required_tuples = (controls['warmup_steps']+controls['measured_steps'])*base_cfg['batch_size']
    for attention in [False,True]:
        if not allowed():
            raise RuntimeError('Profiling budget exhausted during baseline preparation; reports preserved.')
        cfg = dict(baseline,attention=attention,query_size=required_tuples,pool_size=controls['mining_sample_pool'])
        model = rt.make_model(cfg,weights,device)
        tuples,_,_,_ = rt.mine(model,partitions['train'],cfg,base_cfg['seed'])
        probe = fixed_augmented_batch(partitions['train'],tuples,cfg)
        reference = gradient_snapshot(model,probe,cfg)
        del model
        cleanup()
        contexts[attention] = dict(part=partitions['train'],tuples=tuples,probe=probe,
                                    reference=reference,sample_paths=paths,worst_path=worst,
                                    cache_paths=partitions['train']['paths']+partitions['val']['paths'])
    selected = None
    def evaluate_stage(stage,candidates):
        nonlocal selected
        if controls.get('enabled_stages') and stage not in controls['enabled_stages']:
            return
        for candidate in candidates:
            # Leave a soft confirmation reserve; never start half a paired trial intentionally.
            if not allowed(controls['confirmation_reserve_minutes']*60):
                print('Soft profiling budget reached; stop sweep and keep completed paired candidates.',flush=True)
                break
            precision = candidate['precision']
            if precision == 'BF16' and not torch.cuda.is_bf16_supported():
                print('Skip BF16: unsupported device.')
                continue
            if precision not in rank_cache:
                rank_cache[precision] = (measure_rank(candidate,partitions['train'],2000,15953,device),
                                          measure_rank(candidate,partitions['val'],181,739,device))
            for attention in [False,True]:
                if not allowed():
                    break
                cfg = dict(base_cfg,**candidate,attention=attention)
                print('\nPROFILE',stage,'run',2 if attention else 1,candidate,flush=True)
                data = contexts[attention]
                row = safe_trial(cfg,data,weights,data['probe'],data['reference'],limits,controls,device,rank_cache[precision])
                row['stage'] = stage
                rows.append(row)
                write_reports(output,rows)
                print(row['status'],'epoch estimate:',row.get('estimated_epoch_seconds'),'peak GiB:',row.get('peak_vram_gib'),flush=True)
                if row.get('resource_plan',{}).get('reason'):
                    print('HOST_PLAN:',json.dumps(row['resource_plan']),flush=True)
                memory_report = {key:row[key] for key in ['ram_available_min_gib','ram_system_used_peak_gib',
                    'process_tree_uss_peak_gib','process_tree_rss_peak_gib'] if key in row}
                if memory_report:
                    print('HOST_RAM:',json.dumps(memory_report),flush=True)
        selected = choose_common(rows)
        if selected:
            rt.atomic_json(output/'selected_candidate.json',selected)

    base_tuning = tuning(base_cfg)
    evaluate_stage('precision',[dict(base_tuning,precision=p) for p in controls.get('precisions',['FP32','TF32','BF16'])])
    if not selected:
        rt.atomic_json(output/'recommendation.json',dict(status='NO_SAFE_PAIRED_CONFIGURATION',
            note='Inspect trials.json; no settings are recommended. Increase profiling budget or investigate failures.'))
        rt.atomic_json(output/'profile_usage.json',dict(seconds=time.monotonic()-started,
                       estimated_cu=(time.monotonic()-started)/3600*controls['cu_per_hour']))
        return None
    current = selected['config']
    # Test GPU capacity early and without a large CPU cache/prefetch queue.
    capacity = []
    for fusion in controls.get('capacity_fusions',[]):
        for microbatch in controls.get('capacity_microbatches',[]):
            capacity.append(dict(current,precision='BF16' if torch.cuda.is_bf16_supported() else current['precision'],
                tuple_fusion=fusion,microbatch=microbatch,max_batch_pixels=microbatch*1024**2,
                extraction_window=max(8,microbatch),
                workers=1 if microbatch >= 64 else min(current['workers'],4),
                prefetch_factor=1 if microbatch >= 64 else 2,cache_mode='off',cache_gib=0.))
    if capacity:
        evaluate_stage('gpu_capacity',capacity)
    current = selected['config']
    # Focused retest can compare large compute shapes with identical loader/cache settings.
    compute_joint = [dict(current,tuple_fusion=f,microbatch=m,max_batch_pixels=m*1024**2,
                          extraction_window=w,workers=4,prefetch_factor=2,cache_mode='off',cache_gib=0.)
                     for f in controls.get('compute_joint_fusions',[])
                     for m,w in controls.get('compute_joint_shapes',[])]
    if compute_joint:
        evaluate_stage('compute_joint',compute_joint)
    current = selected['config']
    cpus = len(os.sched_getaffinity(0))
    excluded = [w for w in controls['workers'] if w > 2*cpus]
    if excluded:
        print('Skip workers above 2x CPU affinity:',excluded,'available CPUs:',cpus,flush=True)
    evaluate_stage('workers',[dict(current,workers=w) for w in controls['workers'] if w <= 2*cpus])
    current = selected['config']
    evaluate_stage('prefetch',[dict(current,prefetch_factor=p) for p in controls.get('prefetch_factors',[2])])
    current = selected['config']
    evaluate_stage('cache',[dict(current,cache_mode=mode,cache_gib=gib)
                            for mode,gib in controls.get('caches',[('off',0.)])])
    current = selected['config']
    joint = [dict(current,cache_mode='decoded',cache_gib=gib,workers=w,prefetch_factor=2)
             for gib in controls.get('cache_joint_gib',[])
             for w in controls.get('cache_joint_workers',[2,4,8]) if w <= 2*cpus]
    if joint:
        evaluate_stage('cache_loader_joint',joint)
    current = selected['config']
    evaluate_stage('fusion',[dict(current,tuple_fusion=f,microbatch=m,max_batch_pixels=m*1024**2)
                            for f in controls.get('tuple_fusions',[1]) for m in controls['microbatches']])
    current = selected['config']
    evaluate_stage('window',[dict(current,extraction_window=e) for e in controls['windows']])
    # Re-run the best distinct candidates, not just a single lucky 3-step timing.
    repeats = controls.get('finalist_repeats',0)
    if repeats:
        finalists,remaining_rows = [],list(rows)
        for _ in range(controls.get('finalists',2)):
            winner = choose_common(remaining_rows)
            if not winner:
                break
            finalists.append(winner['config'])
            signature = tune_signature(winner['config'])
            remaining_rows = [r for r in remaining_rows if tune_signature(r['config']) != signature]
        evaluate_stage('repeat_finalists',[dict(cfg) for cfg in finalists for _ in range(repeats)])
    confirmations = {}
    for attention in [False,True]:
        if not allowed(60):
            break
        cfg = dict(base_cfg,**selected['config'],attention=attention)
        try:
            confirmation = confirm_impl(cfg,partitions,weights,controls,output,device)
        except Exception as error:
            traceback.print_exc()
            confirmation = dict(status='FAILED',error_type=type(error).__name__,error=str(error))
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        finally:
            cleanup()
        confirmations[str(attention)] = confirmation
        rt.atomic_json(output/'confirmations.json',confirmations)
    recommendation = export_recommendation(selected,confirmations,controls,base_cfg,benchmarks)
    rt.atomic_json(output/'recommendation.json',recommendation)
    rt.atomic_json(output/'profile_usage.json',dict(seconds=time.monotonic()-started,
                   estimated_cu=(time.monotonic()-started)/3600*controls['cu_per_hour']))
    if recommendation['status'] == 'CONFIRMED_LOCAL_PROBES' and recommendation['common_epoch_target'] >= 1:
        snippet = '# Apply these shared settings only to NEW training runs.\n'
        snippet += '\n'.join(f'{key} = {value!r}' for key,value in recommendation['settings'].items())+'\n'
        (output/'recommended_config.py').write_text(snippet)
        print(snippet)
    print('PROFILE COMPLETE:',output,flush=True)
    return recommendation

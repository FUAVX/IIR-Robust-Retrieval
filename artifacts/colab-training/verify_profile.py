"""CPU checks for profiler guards/selection/export and shared BF16-safe model path."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import colab_runtime as rt
import profile_runtime as pr

torch.set_num_threads(2)


def test_numeric_guards():
    rt.seed_all(9)
    features = nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU())
    model = rt.RetrievalModel(True,features,8)
    cfg = dict(microbatch=2,max_batch_pixels=4096,margin=.7,batch_size=5)
    batch = [[torch.rand(1,3,16,20) for _ in range(7)] for _ in range(5)]
    reference = pr.gradient_snapshot(model,batch,cfg)
    model.precision = 'bf16'
    mixed = pr.gradient_snapshot(model,batch,cfg)
    assert mixed['descriptors'].dtype == torch.float32
    limits = dict(max_loss_relative_error=.05,min_gradient_cosine=.98,min_descriptor_cosine=.995)
    result = pr.compare_snapshots(reference,mixed,limits)
    assert result['passed'], result
    damaged = copy.deepcopy(mixed)
    damaged['gradients'] = {key:-value for key,value in damaged['gradients'].items()}
    assert not pr.compare_snapshots(reference,damaged,limits)['passed']
    print('PASS BF16 CPU autocast path: FP32 descriptor/loss, finite gradients; numeric guard rejects reversed gradients')


def test_selection_and_export():
    cfg = dict(precision='FP32',microbatch=2,max_batch_pixels=2*1024**2,workers=2,extraction_window=8)
    epoch = pr.estimate_epoch(20,1,2,.1)
    rows = [dict(config=cfg,attention=att,status='PASS',estimated_epoch_seconds=epoch['total_seconds'],
                 extraction_images_per_second=20,epoch_estimate=epoch) for att in [False,True]]
    faster = dict(cfg,precision='BF16')
    rows.append(dict(rows[0],config=faster,estimated_epoch_seconds=1))
    assert pr.choose_common(rows)['config']['precision'] == 'FP32'
    rows.append(dict(rows[1],config=faster,status='OOM',estimated_epoch_seconds=1))
    assert pr.choose_common(rows)['config']['precision'] == 'FP32'
    rows.append(dict(rows[1],config=faster,estimated_epoch_seconds=1))
    selection = pr.choose_common(rows)
    assert selection['config']['precision'] == 'BF16'
    failed_again = rows+[dict(rows[-1],status='SKIPPED_HOST_PLAN')]
    assert pr.choose_common(failed_again)['config']['precision'] == 'FP32'
    controls = dict(eval_every=3,training_cu_per_run=110,cu_per_hour=5.4)
    confirmations = {str(att):dict(status='PASS',gradient_tuples_per_second=2,
                         full_validation_seconds=10,drive_checkpoint_seconds=5) for att in [False,True]}
    result = pr.export_recommendation(selection,confirmations,controls,{'epochs':12},{})
    assert result['status'] == 'CONFIRMED_LOCAL_PROBES'
    assert result['settings']['PRECISION'] == 'BF16'
    assert 1 <= result['settings']['EPOCHS'] <= 12
    for variant in result['per_variant'].values():
        breakdown = variant['breakdown']
        expected_total = sum(breakdown[key] for key in
                             ['mining_seconds','gradient_training_seconds','validation_seconds','checkpoint_seconds'])
        assert breakdown['total_seconds'] == expected_total
        assert abs(variant['estimated_epoch_hours']*3600-expected_total) < 1e-6
    partial = pr.export_recommendation(selection,{'False':confirmations['False']},controls,{'epochs':12},{})
    assert partial['status'] == 'PROVISIONAL_NOT_CONFIRMED'
    unreplicated = pr.export_recommendation(selection,confirmations,dict(controls,finalist_repeats=3),{'epochs':12},{})
    assert unreplicated['status'] == 'PROVISIONAL_NOT_CONFIRMED'
    with tempfile.TemporaryDirectory() as folder:
        pr.write_reports(folder,rows)
        assert json.loads((Path(folder)/'trials.json').read_text())[0]['status'] == 'PASS'
        assert 'training_images_per_forward' in (Path(folder)/'trials.csv').read_text()
    print('PASS paired-model selection excludes OOM/incomplete pairs; export/provisional and epoch budget projection')


def test_shared_optimizer_step():
    cfg = dict(lr=1e-4,weight_decay=1e-6,attention=True,attention_lr=1e-4,
               margin=.7,microbatch=2,max_batch_pixels=4096)
    rt.seed_all(11)
    first = rt.RetrievalModel(True,nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU()),8)
    second = copy.deepcopy(first)
    batch = [[torch.rand(1,3,16,20) for _ in range(7)] for _ in range(5)]
    opt1,opt2 = rt.make_optimizer(first,cfg),rt.make_optimizer(second,cfg)
    rt.optimizer_batch(first,opt1,batch,cfg)
    opt2.zero_grad(set_to_none=True)
    for images in batch:
        rt.tuple_loss(rt.forward_images(second,images,2,4096),.7).backward()
    opt2.step()
    for a,b in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    print('PASS profiler and training share the exact summed-gradient Adam step')


@patch.object(rt,'system_memory',return_value=dict(total_bytes=64*1024**3,available_bytes=48*1024**3,used_bytes=16*1024**3))
def test_cache_fusion_and_repeats(_memory):
    from verify_colab import make_fixture
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        parts = make_fixture(root)
        paths = parts['train']['paths']
        cfg = dict(image_size=24,workers=2,prefetch_factor=2)
        off = dict(cfg,cache_mode='off',cache_gib=0.)
        rt.prepare_image_cache(paths,off)
        clean = [rt.ImageList(paths,24)[i] for i in range(len(paths))]
        augmented = rt.ImageList(paths,24,augment=True)
        rt.seed_all(13)
        expected = augmented[0]
        for mode in ['bytes','decoded']:
            cached = dict(cfg,cache_mode=mode,cache_gib=.001)
            info = rt.prepare_image_cache(paths,cached)
            assert info['coverage'] == 1. and info['stored_bytes'] <= info['budget_bytes']
            assert rt.prepare_image_cache(paths,cached)['reused']
            for i in range(len(paths)):
                torch.testing.assert_close(rt.ImageList(paths,24)[i],clean[i],rtol=0,atol=0)
            rt.seed_all(13)
            torch.testing.assert_close(rt.ImageList(paths,24,augment=True)[0],expected,rtol=0,atol=0)
            # Forked workers see the same read-only cache; no augmented data is cached.
            loader = rt.DataLoader(rt.ImageList(paths,24),batch_size=2,collate_fn=rt.identity_collate,
                                   **rt.loader_options(cached))
            values = [item for batch in loader for item in batch]
            for a,b in zip(values,clean):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Regression from real v3: decoded cache must be gone before planning cache-off.
        assert rt.IMAGE_CACHE_INFO['mode'] == 'decoded' and rt.IMAGE_CACHE
        pr.release_incompatible_cache(off,dict(part=parts['train'],cache_paths=paths))
        assert not rt.IMAGE_CACHE and rt.IMAGE_CACHE_INFO['mode'] == 'off'
        # The guard observes the cleared state, not the previous cache allocation.
        def guard(config,controls):
            assert not rt.IMAGE_CACHE
            return dict(passed=False,reason='test guard after releasing cache')
        rt.prepare_image_cache(paths,dict(cfg,cache_mode='decoded',cache_gib=.001))
        with patch.object(pr,'resource_plan',side_effect=guard), \
             patch.object(pr.UtilizationSampler,'start'), \
             patch.object(pr.UtilizationSampler,'finish',return_value={}):
            row = pr.safe_trial(dict(off,attention=False),dict(part=parts['train']),None,None,None,{}, {},'cpu',(0,0))
        assert row['status'] == 'SKIPPED_HOST_PLAN'
        rt.prepare_image_cache([],off)
    rt.seed_all(12)
    model = rt.RetrievalModel(False,nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU()),8)
    batch = [[torch.rand(1,3,16,20) for _ in range(7)] for _ in range(5)]
    cfg = dict(microbatch=8,max_batch_pixels=4096,margin=.7)
    ref = pr.gradient_snapshot(model,batch,dict(cfg,tuple_fusion=1))
    for fusion in [2,5]:
        candidate = pr.gradient_snapshot(model,batch,dict(cfg,tuple_fusion=fusion))
        result = pr.compare_snapshots(ref,candidate,dict(max_loss_relative_error=1e-4,
                                   min_gradient_cosine=.9999,min_descriptor_cosine=.9999))
        assert result['passed'],result
    config = dict(precision='BF16',microbatch=8,max_batch_pixels=8*1024**2,workers=4,extraction_window=8)
    rows = [dict(config=config,attention=a,status='PASS',stage='repeat_finalists',estimated_epoch_seconds=t)
            for a in [False,True] for t in [100,5,110]]
    selected = pr.choose_common(rows)
    assert selected['score_seconds'] == 200  # Median 100, not lucky fastest 5 seconds.
    assert selected['per_variant']['False']['repeat_count'] == 3
    print('PASS bytes/RGB cache pixels+augmentation unchanged, fork workers, tuple fusion gradients, repeated timing median')


def test_resource_guards():
    memory = dict(total_bytes=64*1024**3,available_bytes=48*1024**3,used_bytes=16*1024**3)
    cfg = dict(image_size=1024,workers=2,prefetch_factor=2,extraction_window=8,cache_mode='off',cache_gib=0)
    with patch.object(rt,'system_memory',return_value=memory), \
         patch.object(pr.os,'statvfs',return_value=SimpleNamespace(f_bavail=8192,f_frsize=1024**2)):
        assert pr.resource_plan(cfg,{})['passed']
        unsafe = pr.resource_plan(dict(cfg,workers=16,prefetch_factor=8,extraction_window=64),{})
        assert not unsafe['passed'] and unsafe['reason']
    print('PASS host RAM and /dev/shm guards reject excessive worker/prefetch queues')


def test_trial_and_orchestration():
    """Real CPU measurement path + mocked GPU orchestration; not CUDA verification."""
    from verify_colab import make_fixture
    cfg = dict(precision='FP32',attention=False,attention_lr=1e-4,lr=1e-4,weight_decay=1e-6,
               margin=.7,batch_size=5,image_size=32,microbatch=2,max_batch_pixels=4096,
               workers=0,extraction_window=8,seed=42,query_size=20,pool_size=16,epochs=12)
    controls = dict(warmup_steps=1,measured_steps=3,extraction_repeats=2,max_vram_fraction=.85,
                    max_minutes=45,profile_cu_budget=6,cu_per_hour=5.4,training_cu_per_run=110,
                    extraction_images=8,mining_sample_pool=16,confirmation_reserve_minutes=0,
                    microbatches=[1,2],workers=[0],windows=[8],confirmation_queries=5,
                    confirmation_pool=16,eval_every=3,finalist_repeats=3,finalists=2,
                    capacity_microbatches=[2,4],capacity_fusions=[2,5],
                    cache_joint_gib=[.001],cache_joint_workers=[0])
    limits = dict(max_loss_relative_error=.05,min_gradient_cosine=.98,min_descriptor_cosine=.995)

    def factory(config,weights,device):
        rt.seed_all(12)
        return rt.RetrievalModel(config['attention'],
                nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU()),8)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        images = root/'images'
        images.mkdir()
        parts = make_fixture(images)
        with patch.object(rt,'make_model',factory):
            model = factory(cfg,None,'cpu')
            tuples,*_ = rt.mine(model,parts['train'],cfg,42)
            tuples = (tuples*4)[:20]
            probe = pr.fixed_augmented_batch(parts['train'],tuples,cfg)
            reference = pr.gradient_snapshot(model,probe,cfg)
            data = dict(part=parts['train'],tuples=tuples,sample_paths=parts['train']['paths'][:8],
                        worst_path=parts['train']['paths'][0])
            with patch('torch.cuda.reset_peak_memory_stats'), \
                 patch('torch.cuda.max_memory_allocated',return_value=1024**2), \
                 patch('torch.cuda.max_memory_reserved',return_value=2*1024**2), \
                 patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=40*1024**3)):
                measured = pr.trial_impl(cfg,data,None,probe,reference,limits,controls,'cpu',(.1,.01))
            assert measured['status'] == 'PASS', measured
            assert len(measured['measured_step_seconds']) == 3 and measured['training_tuples_per_second'] > 0

            def fake_trial(config,*args):
                rate = 20*config['microbatch']*(2 if config['precision']=='BF16' else 1)
                estimate = pr.estimate_epoch(rate,rate/10,.1,.01)
                return dict(status='PASS',config=config,attention=config['attention'],
                            extraction_images_per_second=rate,estimated_epoch_seconds=estimate['total_seconds'],
                            epoch_estimate=estimate)
            confirmation = dict(status='PASS',gradient_tuples_per_second=4,
                                full_validation_seconds=1,drive_checkpoint_seconds=.1)
            with patch('torch.cuda.is_available',return_value=True), \
                 patch('torch.cuda.manual_seed_all'), \
                 patch('torch.cuda.get_device_name',return_value='A100_MOCK'), \
                 patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=40*1024**3)), \
                 patch('torch.cuda.is_bf16_supported',return_value=True), \
                 patch.object(pr,'cleanup'), patch.object(pr,'measure_rank',return_value=.1), \
                 patch.object(pr,'safe_trial',side_effect=fake_trial), \
                 patch.object(pr,'confirm_impl',return_value=confirmation):
                result = pr.run_profile(cfg,parts,None,{},root/'report',controls,limits)
            assert result['status'] == 'CONFIRMED_LOCAL_PROBES'
            assert (root/'report'/'recommended_config.py').is_file()
    print('PASS actual CPU trial data/optimizer/stress path and staged orchestration with mocked GPU metrics')


if __name__ == '__main__':
    test_numeric_guards()
    test_selection_and_export()
    test_shared_optimizer_step()
    test_cache_fusion_and_repeats()
    test_resource_guards()
    test_trial_and_orchestration()
    print('ALL PROFILER CPU CHECKS PASSED; CUDA timing/VRAM/Drive still require execution in Colab.')

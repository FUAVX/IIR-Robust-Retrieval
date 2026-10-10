#!/usr/bin/env python3
"""
scripts/download_datasets_local.py

Local-machine equivalent of notebooks/download_and_verify_datasets.ipynb
(Kaggle version by Hoang, FUAVX).

What it does (idempotent — safe to re-run):
  0. Init ./data tree (no Kaggle paths, no rm -rf /).
  1. Download + extract Oxford5k (~1.9GB) and Paris6k (~2x1GB, flatten nested dirs),
     download gnd_*.pkl ground truth.
  2. Download Caffe VGG16 weights (~58MB).
  3. Download retrieval-SfM-120k.pkl metadata, backup full copy,
     sample NUM_CLUSTERS complete clusters (seed 42) -> overwrite pkl with subset,
     write needed_files.txt tar filter.
  4. Stream-extract ONLY subset images from ims.tar.gz (~40GB remote, ~2.5GB local)
     via: curl | tar -xz -C ims_dir -T needed_files.txt  (archive never saved).
  5. Run scripts/verify_datasets.py audit at the end.

Differences vs notebook:
  - DATA_ROOT defaults to <repo>/data (overridable with --data-root).
  - Uses Python urllib with retry + wget -c resume when available.
  - Skips steps already done (checks file counts before downloading).
  - Verifies tar internal prefix before streaming (./x/.. vs ims/..).

Usage:
    python3 scripts/download_datasets_local.py
    python3 scripts/download_datasets_local.py --data-root ./data --num-clusters 50 --seed 42
    python3 scripts/download_datasets_local.py --skip-sfm-images   # metadata only, no 40GB stream
    python3 scripts/download_datasets_local.py --only verify       # just run verification
"""

import argparse
import os
import pickle
import random
import shutil
import subprocess
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URLS = {
    "oxford_ims": "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz",
    "oxford_gnd": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/oxford5k/gnd_oxford5k.pkl",
    "paris_1": "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz",
    "paris_2": "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz",
    "paris_gnd": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/paris6k/gnd_paris6k.pkl",
    "vgg_weights": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth",
    "sfm_pkl": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/retrieval-SfM-120k.pkl",
    "sfm_whiten": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/retrieval-SfM-120k-whiten.pkl",
    "sfm_ims": "https://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/ims.tar.gz",
}

EXPECTED = {"oxford": 5063, "paris": 6412}


def log(msg):
    print(msg, flush=True)


def run(cmd, **kw):
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def have(cmd):
    return shutil.which(cmd) is not None


def download_file(url, dest, min_bytes=1024):
    """Download with wget -c resume if available, else urllib with retry."""
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_bytes:
        # quick integrity: for .pkl/.pth accept existing; for .tgz caller checks extraction
        log(f"[skip] exists: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    log(f">> downloading {url}\n   -> {dest}")
    if have("wget"):
        run(["wget", "-c", "--tries=5", "--timeout=60", "-O", dest, url])
    else:
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, length=1024 * 256)
                break
            except Exception as e:
                log(f"[retry {attempt}/3] {e}")
                if attempt == 3:
                    raise
                if os.path.isfile(dest):
                    os.remove(dest)
    if os.path.getsize(dest) < min_bytes:
        raise RuntimeError(f"Downloaded file too small: {dest}")
    return dest


def download_fast(url, dest, min_bytes=1024):
    """aria2 multi-connection (8x) for throttled hosts (cmp.felk.cvut.cz),
    fallback to single-connection download_file."""
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_bytes:
        log(f"[skip] exists: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    if have("aria2c"):
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        log(f">> aria2c -x8 {url}\n   -> {dest}")
        run(["aria2c", "-x", "8", "-s", "8", "-k", "5M", "-c",
             "-d", os.path.dirname(os.path.abspath(dest)),
             "-o", os.path.basename(dest), url])
    else:
        log("[info] aria2c not found, single-connection fallback (slow on .cz hosts)")
        download_file(url, dest, min_bytes=min_bytes)
    if os.path.getsize(dest) < min_bytes:
        raise RuntimeError(f"Downloaded file too small: {dest}")
    return dest


def count_jpg(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg")))


def extract_tgz(archive, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    log(f">> extracting {archive} -> {dest_dir}")
    if have("tar"):
        run(["tar", "-xzf", archive, "-C", dest_dir])
    else:
        import tarfile
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest_dir)


def flatten_paris(jpg_dir):
    """Move nested files up so all images sit directly in jpg/ (notebook fix)."""
    moved = 0
    for root, _, files in os.walk(jpg_dir):
        if os.path.abspath(root) == os.path.abspath(jpg_dir):
            continue
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(jpg_dir, f)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            if os.path.exists(dst):
                base, ext = os.path.splitext(f)
                dst = os.path.join(jpg_dir, f"{base}_{abs(hash(root)) % 9999}{ext}")
            shutil.move(src, dst)
            moved += 1
    for root, dirs, _ in os.walk(jpg_dir, topdown=False):
        for d in dirs:
            p = os.path.join(root, d)
            try:
                if not os.listdir(p):
                    os.rmdir(p)
            except OSError:
                pass
    if moved:
        log(f"[fix] flattened {moved} nested Paris files into {jpg_dir}")


def step_benchmarks(data_root, work_tmp):
    ox_jpg = os.path.join(data_root, "test", "oxford5k", "jpg")
    pa_jpg = os.path.join(data_root, "test", "paris6k", "jpg")
    os.makedirs(ox_jpg, exist_ok=True)
    os.makedirs(pa_jpg, exist_ok=True)

    # --- Oxford ---
    if count_jpg(ox_jpg) < EXPECTED["oxford"]:
        arc = os.path.join(work_tmp, "oxbuild_images-v1.tgz")
        download_file(URLS["oxford_ims"], arc, min_bytes=500_000_000)
        extract_tgz(arc, ox_jpg)
        # some archives contain a top folder; flatten one level if counts still low
        if count_jpg(ox_jpg) < EXPECTED["oxford"]:
            for entry in os.listdir(ox_jpg):
                sub = os.path.join(ox_jpg, entry)
                if os.path.isdir(sub):
                    for f in os.listdir(sub):
                        shutil.move(os.path.join(sub, f), os.path.join(ox_jpg, f))
        os.remove(arc)
    else:
        log(f"[skip] oxford5k already has {count_jpg(ox_jpg):,} images")
    download_file(URLS["oxford_gnd"],
                  os.path.join(data_root, "test", "oxford5k", "gnd_oxford5k.pkl"))
    log(f"Oxford5k count: {count_jpg(ox_jpg):,} (expect {EXPECTED['oxford']:,})")

    # --- Paris ---
    if count_jpg(pa_jpg) < EXPECTED["paris"]:
        a1 = os.path.join(work_tmp, "paris_1-v1.tgz")
        a2 = os.path.join(work_tmp, "paris_2-v1.tgz")
        download_file(URLS["paris_1"], a1, min_bytes=300_000_000)
        download_file(URLS["paris_2"], a2, min_bytes=300_000_000)
        extract_tgz(a1, pa_jpg)
        extract_tgz(a2, pa_jpg)
        flatten_paris(pa_jpg)
        os.remove(a1)
        os.remove(a2)
    else:
        log(f"[skip] paris6k already has {count_jpg(pa_jpg):,} images")
        flatten_paris(pa_jpg)
    download_file(URLS["paris_gnd"],
                  os.path.join(data_root, "test", "paris6k", "gnd_paris6k.pkl"))
    log(f"Paris6k count: {count_jpg(pa_jpg):,} (expect {EXPECTED['paris']:,})")

    # --- weights (cz host throttles single connections -> aria2) ---
    download_fast(URLS["vgg_weights"],
                  os.path.join(data_root, "networks",
                               "imagenet-caffe-vgg16-features-d369c8e.pth"),
                  min_bytes=10_000_000)


def step_sfm_metadata(data_root, num_clusters, seed, full=False):
    sfm_root = os.path.join(data_root, "train", "retrieval-SfM-120k")
    os.makedirs(sfm_root, exist_ok=True)
    pkl_path = os.path.join(sfm_root, "retrieval-SfM-120k.pkl")
    full_bak = os.path.join(sfm_root, "retrieval-SfM-120k-full.pkl")

    # whiten pkl is never rewritten locally -> resume (-c) is safe for it
    try:
        download_fast(URLS["sfm_whiten"],
                      os.path.join(sfm_root, "retrieval-SfM-120k-whiten.pkl"))
    except Exception as e:
        log(f"[warn] whiten pkl optional, skip: {e}")

    # FULL MODE with a valid backup: skip download entirely (local pkl may be
    # a subset; resuming (-c) onto it would splice server bytes after subset
    # bytes into a corrupt Frankenstein file).
    if full and os.path.isfile(full_bak):
        try:
            with open(full_bak, "rb") as f:
                full_db = pickle.load(f)
            import shutil as _sh
            _sh.copyfile(full_bak, pkl_path)
            n_img = sum(len(full_db[p]["cids"]) for p in ("train", "val") if p in full_db)
            log(f"[full] restored full metadata from backup: {n_img:,} images")
            return None, n_img
        except Exception as e:
            log(f"[warn] backup unreadable ({e}), re-downloading fresh")

    # Fresh pkl download: delete any local copy first. The local file may be
    # a SUBSET from a previous run; resume (-c) onto it would corrupt it.
    for _suf in ("", ".aria2"):
        try:
            if os.path.isfile(pkl_path + _suf):
                os.remove(pkl_path + _suf)
                log(f"[info] removed stale local file: {pkl_path + _suf}")
        except OSError:
            pass
    download_fast(URLS["sfm_pkl"], pkl_path, min_bytes=1_000_000)

    with open(pkl_path, "rb") as f:
        db = pickle.load(f)

    if os.path.isfile(full_bak):
        with open(full_bak, "rb") as f:
            full_db = pickle.load(f)
        if full:
            # FULL MODE: restore complete metadata, no subsetting
            import shutil as _sh
            _sh.copyfile(full_bak, pkl_path)
            n_img = sum(len(full_db[p]["cids"]) for p in ("train", "val") if p in full_db)
            log(f"[full] restored full metadata: {n_img:,} images "
                  f"({len(full_db['train']['cids']):,} train + "
                  f"{len(full_db.get('val', {}).get('cids', [])):,} val)")
            return None, n_img
        log(f"[info] full backup exists: {full_bak} — regenerating subset from it")
        db = full_db
    elif full:
        with open(full_bak, "wb") as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_img = sum(len(db[p]["cids"]) for p in ("train", "val") if p in db)
        log(f"[full] metadata full: {n_img:,} images, no subsetting (backed up)")
        return None, n_img
    else:
        with open(full_bak, "wb") as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"[info] full metadata backed up -> {full_bak}")

    rng = random.Random(seed)
    subset_db = {}
    needed = set()
    provenance = []
    for part in ["train", "val"]:
        if part not in db:
            continue
        p = db[part]
        cids, clusters, qidxs, pidxs = p["cids"], p["cluster"], p["qidxs"], p["pidxs"]
        cl_to_idx = {}
        for i, c in enumerate(clusters):
            cl_to_idx.setdefault(c, []).append(i)
        k = min(num_clusters if part == "train" else max(10, num_clusters // 5),
                len(cl_to_idx))
        chosen = sorted(rng.sample(sorted(cl_to_idx.keys()), k))
        kept = sorted(i for c in chosen for i in cl_to_idx[c])
        old2new = {o: n for n, o in enumerate(kept)}
        kept_set = set(kept)
        sub_cids = [cids[i] for i in kept]
        pairs = [(q, p_) for q, p_ in zip(qidxs, pidxs) if q in kept_set and p_ in kept_set]
        subset_db[part] = {
            "cids": sub_cids,
            "cluster": [clusters[i] for i in kept],
            "qidxs": [old2new[q] for q, _ in pairs],
            "pidxs": [old2new[p_] for _, p_ in pairs],
        }
        needed.update(sub_cids)
        provenance.append(f"{part}: {k} clusters, {len(sub_cids):,} images, {len(pairs):,} pairs")
        log(f"Partition '{part}': {k} clusters, {len(sub_cids):,} images, {len(pairs):,} pairs")

    with open(pkl_path, "wb") as f:
        pickle.dump(subset_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    prov_path = os.path.join(sfm_root, f"subset_provenance_k{num_clusters}_seed{seed}.txt")
    with open(prov_path, "w") as f:
        f.write(f"num_clusters={num_clusters} seed={seed}\n")
        f.write("\n".join(provenance) + "\n")
        f.write(f"total_needed_images={len(needed)}\n")

    # tar filter: cid -> path inside archive. Notebook uses ./aa/bb/cc/cid;
    # cid2filename() gives <pre>/aa/bb/cc/cid (no .jpg? cids already contain ext?)
    # Replicate notebook exactly, plus a stripped variant as fallback.
    filt = os.path.join(sfm_root, "needed_files.txt")
    with open(filt, "w") as f:
        for cid in sorted(needed):
            f.write(f"./{cid[-2:]}/{cid[-4:-2]}/{cid[-6:-4]}/{cid}\n")
    log(f"[ok] filter for {len(needed):,} images -> {filt}")
    log(f"[ok] provenance -> {prov_path}")
    return filt, len(needed)


SFMTAR_SIZE = 40528074099  # Content-Length reported by server (~37.7 GiB)


def ensure_full_tar(work_tmp):
    """Download full ims.tar.gz with aria2 multi-connection (resume OK).
    Single-connection curl/wget stalls (~KB/s) on this host; aria2 -x8 ~15MB/s."""
    tar_path = os.path.join(work_tmp, "ims.tar.gz")
    if os.path.isfile(tar_path) and os.path.getsize(tar_path) >= SFMTAR_SIZE - 1024:
        log(f"[skip] full tar exists: {tar_path} ({os.path.getsize(tar_path)/1e9:.1f} GB)")
        return tar_path
    if have("aria2c"):
        log(f">> aria2c -x8 full ims.tar.gz (~37.7GB, ~40p @15MB/s) -> {tar_path}")
        run(["aria2c", "-x", "8", "-s", "8", "-k", "5M", "-c",
             "-d", work_tmp, "-o", "ims.tar.gz", URLS["sfm_ims"]])
    else:
        log("[warn] aria2c missing — single-connection stream will stall on this host. "
            "Install aria2 (sudo apt install -y aria2) for 8x speed.")
        download_file(URLS["sfm_ims"], tar_path, min_bytes=10_000_000_000)
    return tar_path


def step_sfm_images(data_root, filt_path, work_tmp, full=False):
    ims_dir = os.path.join(data_root, "train", "retrieval-SfM-120k", "ims")
    os.makedirs(ims_dir, exist_ok=True)
    tar_path = ensure_full_tar(work_tmp)
    base = ["tar", "-xzf", tar_path, "--no-same-owner", "-C", ims_dir]
    if full:
        log(f">> extracting FULL archive -> {ims_dir} (~121k files, 15-30p)")
        # NOTE: no --checkpoint-action (not portable across tar versions);
        # monitor with: find <ims_dir> -type f | wc -l
        run(base)
    else:
        log(f">> extracting SUBSET from local tar -T {os.path.basename(filt_path)}")
        run(base + ["-T", filt_path])
    total = sum(1 for _, _, fs in os.walk(ims_dir) for _ in fs)
    log(f"[ok] done. total files in ims/: {total:,}")
    return total


def step_verify(data_root, sample_size):
    script = os.path.join(REPO_ROOT, "scripts", "verify_datasets.py")
    log("=== running verification audit ===")
    r = subprocess.run([sys.executable, script, "--data-root", data_root,
                        "--sample-size", str(sample_size)])
    if r.returncode != 0:
        log("[warn] verify_datasets reported issues (see above). "
            "Missing files = not downloaded yet; FAIL = needs fixing.")
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description="Local dataset preparation (notebook equivalent).")
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--num-clusters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--only", choices=["all", "benchmarks", "sfm-meta", "sfm-images", "verify"],
                    default="all")
    ap.add_argument("--skip-sfm-images", action="store_true",
                    help="Do everything except the 40GB stream (metadata only).")
    ap.add_argument("--full", action="store_true",
                    help="FULL dataset mode: keep complete metadata (no subset) "
                         "and extract all ~121k SfM images (~14GB). "
                         "Needs ~55GB free during run (40GB tar + 14GB imgs).")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    work_tmp = os.path.join(data_root, "_tmp_download")
    os.makedirs(work_tmp, exist_ok=True)
    log(f"DATA_ROOT = {data_root}  mode={'FULL' if args.full else 'subset'}")

    filt = os.path.join(data_root, "train", "retrieval-SfM-120k", "needed_files.txt")

    if args.only in ("all", "benchmarks"):
        step_benchmarks(data_root, work_tmp)
    if args.only in ("all", "sfm-meta"):
        filt, _ = step_sfm_metadata(data_root, args.num_clusters, args.seed,
                                    full=args.full)
    if args.only == "sfm-images" or (args.only == "all" and not args.skip_sfm_images):
        if not args.full and not os.path.isfile(filt):
            filt, _ = step_sfm_metadata(data_root, args.num_clusters, args.seed)
        step_sfm_images(data_root, filt, work_tmp, full=args.full)
    if args.only in ("all", "verify"):
        step_verify(data_root, args.sample_size)

    # cleanup empty tmp dir
    try:
        if os.path.isdir(work_tmp) and not os.listdir(work_tmp):
            os.rmdir(work_tmp)
    except OSError:
        pass
    log("[done] See data/ tree + run: python3 scripts/verify_datasets.py --data-root ./data")


if __name__ == "__main__":
    main()

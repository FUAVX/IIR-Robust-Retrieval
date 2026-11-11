"""Chunked tar transport: few Kaggle output files and <180s bridge download per part."""
import hashlib,json,tarfile,io
from pathlib import Path

class ChunkWriter:
    def __init__(self,root,limit=128*1024**2):
        self.root=Path(root);self.limit=limit;self.files=[];self.file=None;self.size=0
    def write(self,data):
        count=len(data);view=memoryview(data)
        while view:
            if self.file is None or self.size==self.limit:
                if self.file:self.file.close()
                path=self.root/f'artifacts_{len(self.files):05d}.tarpart';self.files.append(path);self.file=path.open('wb');self.size=0
            n=min(len(view),self.limit-self.size);self.file.write(view[:n]);self.size+=n;view=view[n:]
        return count
    def flush(self):
        if self.file:self.file.flush()
    def close(self):
        if self.file:self.file.close()

class ChunkReader(io.RawIOBase):
    def __init__(self,files):self.paths=iter(files);self.current=None
    def readable(self):return True
    def read(self,n=-1):
        if n<0:raise ValueError('Streaming bounded reads only')
        result=bytearray()
        while len(result)<n:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            block=self.current.read(n-len(result))
            if not block:self.current.close();self.current=None
            else:result.extend(block)
        return bytes(result)
    def close(self):
        if self.current:self.current.close()
        super().close()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def pack(root,part_limit=128*1024**2):
    root=Path(root);excluded={'job_config.json','job_status.json','artifact_manifest.json'}
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in excluded and not p.name.endswith('.tarpart')]
    writer=ChunkWriter(root,part_limit)
    with tarfile.open(fileobj=writer,mode='w|') as tar:
        for path in files:tar.add(path,arcname=str(path.relative_to(root)),recursive=False)
    writer.close()
    manifest=dict(format='chunked_tar_v1',parts=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in writer.files],payload_files=len(files),payload_bytes=sum(p.stat().st_size for p in files))
    (root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # All payloads are now checksummed inside the immutable transport; don't expose
    # thousands of duplicate individual Kaggle output files.
    for path in files:path.unlink()
    return manifest

def unpack(manifest_path,target):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text());target=Path(target);target.mkdir(parents=True,exist_ok=True)
    files=[]
    for part in manifest['parts']:
        path=manifest_path.parent/part['name']
        if path.stat().st_size!=part['bytes'] or sha(path)!=part['sha256']:raise ValueError('Artifact part corrupt: '+str(path))
        files.append(path)
    reader=ChunkReader(files)
    try:
        with tarfile.open(fileobj=reader,mode='r|') as tar:
            for member in tar:
                if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:raise ValueError('Unsafe archive entry')
                dest=target/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
                source=tar.extractfile(member)
                with dest.open('wb') as f:
                    while True:
                        b=source.read(8*1024**2)
                        if not b:break
                        f.write(b)
    finally:reader.close()
    return target

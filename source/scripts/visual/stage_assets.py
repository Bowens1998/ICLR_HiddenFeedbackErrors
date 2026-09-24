"""Stage pinned public assets with streamed SHA256 verification and schema inspection."""
import hashlib,json,os,ssl,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
cfg=json.loads(Path(sys.argv[2]).read_text());ctx=ssl.create_default_context(cafile=os.environ.get('PIP_CERT'))
records=[]
for asset in cfg['assets']:
    dest=root/asset['kind']/asset['filename'];dest.parent.mkdir(parents=True,exist_ok=True)
    url='https://huggingface.co/'+('datasets/' if asset['kind']=='datasets' else '')+asset['repo']+'/resolve/'+asset['revision']+'/'+asset['filename']
    if not dest.exists():
        partial=dest.with_suffix(dest.suffix+'.partial')
        with urllib.request.urlopen(url,context=ctx,timeout=180) as src,partial.open('wb') as f:
            while chunk:=src.read(8*1024*1024):f.write(chunk)
        partial.rename(dest)
    h=hashlib.sha256()
    with dest.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    assert dest.stat().st_size==asset['size'],dest
    if asset['sha256']:assert h.hexdigest()==asset['sha256'],dest
    records.append(dict(asset,local_path=str(dest),observed_sha256=h.hexdigest()))
    print('VERIFIED',dest,flush=True)
    if dest.name.endswith('.h5.zst'):
        unpacked=dest.with_suffix('')
        if not unpacked.exists():
            tmp=unpacked.with_suffix('.h5.partial')
            with tmp.open('wb') as f:subprocess.run(['zstd','-d','-c',str(dest)],stdout=f,check=True)
            tmp.rename(unpacked)
        import h5py
        schema={}
        with h5py.File(unpacked,'r') as f:
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset):schema[name]={'shape':list(obj.shape),'dtype':str(obj.dtype),'chunks':obj.chunks,'compression':obj.compression}
            f.visititems(visit)
            schema['_root_attrs']={k:str(v) for k,v in f.attrs.items()}
        (root/'pusht_h5_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
        print('SCHEMA',len(schema),flush=True)
(root/'download_manifest.json').write_text(json.dumps({'assets':records,'completed_unix':time.time()},indent=2)+'\n')

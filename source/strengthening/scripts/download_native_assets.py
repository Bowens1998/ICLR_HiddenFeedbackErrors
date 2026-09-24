"""Verify official public DINO-WM archives and extract into new project cache."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from contracts import atomic_json, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((Path(__file__).resolve().parents[1]/'reports/receipts/dinowm_remote_file_catalog.json').read_text())
    selected={x['name']:x for group in catalog for x in group['files'] if x['name'] in ['outputs.zip','pusht_noise.zip']}
    assert set(selected)=={'outputs.zip','pusht_noise.zip'}
    previous=json.loads((out/'download_report.json').read_text()) if (out/'download_report.json').exists() else {}
    previously_extracted={row['name']:row for row in previous.get('archives',[])}
    rows=[];start=time.monotonic()
    for name,info in selected.items():
        target=out/name;tmp=out/(name+'.partial')
        if not target.exists():
            subprocess.run(['curl','--fail','--location','--retry','3','--connect-timeout','30',
                            '--max-time','2400','--output',str(tmp),info['download']],check=True)
            assert tmp.stat().st_size==info['size'] and sha(tmp)==info['hashes']['sha256']
            tmp.rename(target)
        assert target.stat().st_size==info['size'] and sha(target)==info['hashes']['sha256']
        dest=out/name.removesuffix('.zip');dest.mkdir(exist_ok=True)
        entries=[]
        with zipfile.ZipFile(target) as z:
            for member in z.infolist():
                resolved=(dest/member.filename).resolve()
                if not resolved.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive member')
                entries.append({'path':member.filename,'bytes':member.file_size})
            old=previously_extracted.get(name,{})
            if old.get('archive_sha256')!=info['hashes']['sha256'] or not dest.is_dir():
                z.extractall(dest)
        rows.append(dict(name=name,archive_sha256=sha(target),bytes=target.stat().st_size,
                         extracted_to=str(dest),members=entries))
        atomic_json(out/'download_report.json',dict(status='IN_PROGRESS',archives=rows))
    configs=[]
    for fp in out.rglob('*'):
        if fp.is_file() and fp.suffix in ['.yaml','.yml','.json'] and fp.name!='download_report.json' and '__MACOSX' not in fp.parts and not fp.name.startswith('._'):
            configs.append(dict(path=str(fp),sha256=sha(fp),text=fp.read_text()))
    atomic_json(out/'download_report.json',dict(status='PASS_OFFICIAL_ARCHIVE_HASHES',archives=rows,
        configs=configs,elapsed_seconds=time.monotonic()-start,
        scope='Trusted archive verification only; native model not yet loaded; no model selection or confirmation'))


if __name__=='__main__':main()

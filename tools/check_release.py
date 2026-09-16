"""Check the prospective release for file size and excluded data types."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
LIMIT=50_000_000
FORBIDDEN={'.h5ad','.h5','.hdf5','.loom','.bam','.cram','.fastq','.fq','.bed','.bim','.fam','.vcf','.bgen','.pt','.pth','.ckpt'}

def main():
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    bad=[];sizes=[]
    for name in sorted(set(paths)-{''}):
        p=ROOT/name
        if not p.exists():continue
        size=p.stat().st_size;sizes.append((size,name))
        suffixes={s.lower() for s in p.suffixes}
        if size>=LIMIT:bad.append(f'{name}: {size} bytes exceeds release limit')
        if suffixes&FORBIDDEN:bad.append(f'{name}: full-data or model artifact is excluded')
    if bad:raise SystemExit('\n'.join(bad))
    print(f'{len(sizes)} files; total {sum(s for s,_ in sizes):,} bytes; largest {max(sizes)[0]:,} bytes ({max(sizes)[1]})')

if __name__=='__main__':main()

"""Atualiza somente obras e payload; conserva as inferências parlamentares."""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.bulk_data import publish_file
from src.public_works import update_public_works, works_payload

parser = argparse.ArgumentParser()
parser.add_argument('--refresh', action='store_true')
args = parser.parse_args()
frame = update_public_works(refresh=args.refresh)
path = Path('public/data/dashboard_data.json')
payload = json.loads(path.read_text(encoding='utf-8'))
payload['obrasProjetos'] = works_payload(frame)
payload['dataNotes']['works'] = 'ObrasGov: município da geometria ou menção explícita no nome oficial, divergências sinalizadas, investimento previsto e situação do cadastro atual. Datas de cadastro não indicam exercício de repasse.'
with open(str(path) + '.tmp', 'w', encoding='utf-8') as target:
    json.dump(payload, target, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
publish_file(str(path) + '.tmp', str(path))
with open(path, 'rb') as source, gzip.open(str(path) + '.gz.tmp', 'wb', compresslevel=6) as target:
    shutil.copyfileobj(source, target)
publish_file(str(path) + '.gz.tmp', str(path) + '.gz')
print('Obras e dashboard atualizados.', flush=True)

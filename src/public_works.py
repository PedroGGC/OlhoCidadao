"""Projetos e municípios publicados na API atual do ObrasGov."""
import gzip
import json
import math
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.bulk_data import save_csv, name_key

API = 'https://api-publica.obrasgov.gestao.gov.br/obras'
CACHE = Path('outputs/obrasgov_cache')
_local = threading.local()
PROJECT_FIELDS = ['id_projeto_investimento', 'desc_nome', 'desc_projeto', 'dt_cadastro',
                  'ano_cadastro', 'dt_inicial_prevista', 'dt_final_prevista', 'situacao',
                  'uf_principal', 'organizacao_resp', 'eixos_tipos', 'investimentos_previstos',
                  'desc_endereco', 'nr_cep']
REGIONS = {uf: region for region, ufs in {
    'Norte': 'AC AM AP PA RO RR TO', 'Nordeste': 'AL BA CE MA PB PE PI RN SE',
    'Centro-Oeste': 'DF GO MS MT', 'Sudeste': 'ES MG RJ SP', 'Sul': 'PR RS SC'
}.items() for uf in ufs.split()}


def request_json(endpoint, **params):
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
        _local.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    response = _local.session.get(API + endpoint, params=params, timeout=(15, 90))
    response.raise_for_status()
    return response.json()


def collect_endpoint(endpoint, refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (endpoint.strip('/') + '.json.gz')
    if path.exists() and not refresh:
        with gzip.open(path, 'rt', encoding='utf-8') as source:
            return json.load(source)
    first = request_json(endpoint, pagina=1, tamanho_da_pagina=200)
    total = first['total_items']
    pages = first['total_pages']
    if pages != math.ceil(total / 200):
        raise ValueError(f'Paginação inconsistente em {endpoint}')
    chunks = {1: first['data']}
    print(f'[ObrasGov] {endpoint}: {total:,} registros em {pages} páginas', flush=True)
    def fetch(page):
        result = request_json(endpoint, pagina=page, tamanho_da_pagina=200)
        if result['page_number'] != page or result['total_items'] != total:
            raise ValueError(f'Base mudou durante a coleta: {endpoint}, página {page}')
        rows = result['data']
        if endpoint == '/projeto-investimento':
            rows = [{key: row.get(key) for key in PROJECT_FIELDS} for row in rows]
        return page, rows
    # Conexões persistentes; quantidade limitada de requisições simultâneas.
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, page) for page in range(2, pages + 1)]
        for done, future in enumerate(as_completed(futures), start=2):
            page, rows = future.result()
            chunks[page] = rows
            if done % 100 == 0 or done == pages:
                print(f'[ObrasGov] {endpoint}: {done}/{pages} páginas', flush=True)
    rows = [row for page in range(1, pages + 1) for row in chunks[page]]
    if endpoint == '/projeto-investimento':
        rows = [{key: row.get(key) for key in PROJECT_FIELDS} for row in rows]
    key = 'id_projeto_investimento' if endpoint == '/projeto-investimento' else 'id_geometria'
    if len(rows) != total or (endpoint == '/projeto-investimento' and len({row[key] for row in rows}) != total):
        raise ValueError(f'Coleta incompleta ou duplicada em {endpoint}')
    with gzip.open(str(path) + '.tmp', 'wt', encoding='utf-8') as target:
        json.dump(rows, target, ensure_ascii=False, separators=(',', ':'))
    Path(str(path) + '.tmp').replace(path)
    return rows


def project_sector(project):
    types = project.get('eixos_tipos') or []
    text = ' '.join(str(item.get('tipo', '')) for item in types).upper()
    for words, sector in [
        (['SAÚDE', 'SAUDE'], 'Saúde'), (['EDUCA'], 'Educação'),
        (['SANEAMENTO', 'HÍDRIC', 'HIDRIC'], 'Saneamento e Recursos Hídricos'),
        (['HABITA'], 'Habitação'), (['SEGURAN', 'DEFESA'], 'Segurança e Defesa Civil'),
        (['ASSIST', 'SOCIAL'], 'Assistência Social e Trabalho'),
        (['CULTURA', 'ESPORTE', 'TURISMO'], 'Esporte, Cultura e Lazer'),
        (['TRANSPORTE', 'RODOVIA'], 'Rodovias e Transportes'),
        (['URBANA', 'MOBILIDADE'], 'Infraestrutura Urbana')]:
        if any(word in text for word in words):
            return sector
    return 'Desenvolvimento Regional e Energia'


def normalize_projects(projects, geometries):
    # Uma obra pode abranger vários municípios; mantém uma linha por projeto.
    locations = {}
    municipal_names = {}
    for geometry in geometries:
        code, name, uf = geometry.get('cod_ibge'), geometry.get('no_municipio'), geometry.get('sg_uf')
        if not code or not name or uf not in REGIONS:
            continue
        item = {'codigoIBGE': str(code), 'nome': name.strip(), 'uf': uf}
        locations.setdefault(geometry['id_projeto_investimento'], {})[(str(code), uf)] = item
        municipal_names.setdefault(uf, {})[name_key(name)] = item
    # Menções explícitas "Arari/MA" ou "Codó - MA" no nome oficial permitem
    # identificar geometrias genéricas/divergentes sem adivinhar por proximidade.
    patterns = {uf: re.compile(r'(?<![A-Z0-9])(' + '|'.join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    ) + r')\s*(?:/|-|\()\s*' + uf + r'\b') for uf, names in municipal_names.items()}
    records = []
    for project in projects:
        ident = project['id_projeto_investimento']
        uf = project.get('uf_principal')
        if uf not in REGIONS:
            continue
        geometry_cities = sorted(locations.get(ident, {}).values(), key=lambda item: (item['uf'], item['nome']))
        cities = geometry_cities
        title = project.get('desc_nome') or project.get('desc_projeto') or ident
        normalized_title = name_key(title).replace('–', '-').replace('—', '-')
        mentioned = {municipal_names[uf][match.group(1)]['codigoIBGE']: municipal_names[uf][match.group(1)]
                     for match in patterns[uf].finditer(normalized_title)} if uf in patterns else {}
        geometry_codes = {city['codigoIBGE'] for city in geometry_cities}
        mismatch = bool(mentioned and not set(mentioned).issubset(geometry_codes) and geometry_cities)
        location_source = 'Geometria oficial'
        if mentioned and (not geometry_cities or (mismatch and len(geometry_cities) == 1)):
            cities = sorted(mentioned.values(), key=lambda item: item['nome'])
            location_source = 'Nome oficial do projeto (menção explícita a município/UF)'
        value = sum(float(item.get('vl_investimento_previsto') or 0)
                    for item in project.get('investimentos_previstos') or [])
        if value <= 0:
            value = None  # Ausência de valor não equivale a obra de custo zero.
        date = project.get('dt_cadastro') or ''
        types = project.get('eixos_tipos') or []
        records.append({
            'codigoObra': f'OG-{ident}', 'titulo': title,
            'setor': project_sector(project), 'subTipo': '; '.join(sorted({item.get('subtipo') or '' for item in types} - {''})),
            'municipio': '; '.join(f"{city['nome']}/{city['uf']}" for city in cities) or 'Município não informado na fonte',
            'uf': uf, 'regiao': REGIONS[uf], 'ano': int(date[:4]) if date else 0,
            'mes': int(date[5:7]) if date else 0, 'situacao': project.get('situacao') or 'Não informada',
            'execucaoFisicaPct': None, 'valorTotalObra': value, 'aporteNecessarioPeriodo': value,
            'orgaoFonte': project.get('organizacao_resp') or 'ObrasGov — Governo Federal',
            'unidadeImpacto': '', 'fatorUnidade': None, 'tipoRegistro': 'oficial',
            'municipios': json.dumps(cities, ensure_ascii=False, separators=(',', ':')),
            'municipiosGeometria': json.dumps(geometry_cities, ensure_ascii=False, separators=(',', ':')),
            'origemMunicipio': location_source, 'divergenciaLocalizacao': mismatch,
            'urlFonte': f'https://cipi.economia.gov.br/cipi-frontend/acesso-livre/visao-geral-intervencao/{ident}',
            'dataCadastro': date, 'dataInicialPrevista': project.get('dt_inicial_prevista') or '',
            'dataFinalPrevista': project.get('dt_final_prevista') or '',
            'endereco': project.get('desc_endereco') or '', 'cep': project.get('nr_cep') or '',
            'naturezaValor': 'Investimento previsto publicado no ObrasGov',
        })
    return pd.DataFrame(records)


def works_payload(frame):
    clean = frame.astype(object).where(pd.notna(frame), None)
    records = clean.to_dict(orient='records')
    for record in records:
        record['municipios'] = json.loads(record['municipios']) if record.get('municipios') else []
        record['municipiosGeometria'] = json.loads(record['municipiosGeometria']) if record.get('municipiosGeometria') else []
        record['divergenciaLocalizacao'] = str(record.get('divergenciaLocalizacao')).lower() == 'true'
    return records


def update_public_works(refresh=False):
    projects = collect_endpoint('/projeto-investimento', refresh)
    geometries = collect_endpoint('/geometria', refresh)
    frame = normalize_projects(projects, geometries)
    path = 'data/obras_projetos_publicos_br.csv'
    if Path(path).exists():
        references = pd.read_csv(path)
        references = references[references.codigoObra.str.startswith('REF-')].copy()
        references['tipoRegistro'] = 'referencia'
        references['municipios'] = '[]'
        frame = pd.concat([frame, references], ignore_index=True)
    save_csv(frame, path)
    report = {'fetchedAtUTC': datetime.now(timezone.utc).isoformat(), 'source': API,
              'projectsInSource': len(projects), 'geometryRowsInSource': len(geometries),
              'officialProjects': int(frame.tipoRegistro.eq('oficial').sum()),
              'locationConflictsFlagged': int(frame.divergenciaLocalizacao.eq(True).sum()),
              'projectsWithMunicipality': int((frame.tipoRegistro.eq('oficial') & frame.municipios.ne('[]')).sum())}
    Path('outputs/obrasgov_quality_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return frame

"""Leitura estrita dos arquivos anuais oficiais; nunca completa dados com notas fictícias."""
import io
import zipfile
import unicodedata
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UFS = set('AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO'.split())
CAMARA_COLUMNS = ['ideCadastro', 'txNomeParlamentar', 'sgPartido', 'sgUF',
                  'txtDescricao', 'txtFornecedor', 'txtCNPJCPF', 'txtNumero',
                  'datEmissao', 'numAno', 'numMes', 'vlrLiquido', 'urlDocumento']
SENADO_COLUMNS = ['id', 'tipoDocumento', 'ano', 'mes', 'codSenador', 'nomeSenador',
                  'tipoDespesa', 'cpfCnpj', 'fornecedor', 'documento', 'data',
                  'detalhamento', 'valorReembolsado', 'siglaUF', 'siglaPartido']


NOTE_KEY = ['origemDado', 'idDeputado', 'numDocumento', 'dataDocumento', 'tipoDespesa',
            'cnpjCpfFornecedor', 'ano', 'mes', 'valorLiquido']


def notes_absent_from(previous, current, key=NOTE_KEY):
    """Marca notas de `current` sem par em `previous`; repetições contam como multiconjunto."""
    def occurrences(frame):
        frame = frame[key].astype(str)
        return pd.MultiIndex.from_frame(frame.assign(_n=frame.groupby(key).cumcount().astype(str)))
    return ~occurrences(current).isin(occurrences(previous))


def save_csv(frame, path):
    """Publica somente arquivos completos, inclusive ao substituir dados em uso."""
    frame.to_csv(path + '.tmp', index=False, encoding='utf-8')
    publish_file(path + '.tmp', path)


def publish_file(temporary, destination):
    """No Windows, um visualizador pode manter a versão anterior mapeada."""
    try:
        os.replace(temporary, destination)
    except PermissionError:
        target = Path(destination).resolve()
        source = Path(temporary).resolve()
        root = Path.cwd().resolve()
        if not target.is_relative_to(root) or not source.is_relative_to(root):
            raise
        backup = root / 'outputs' / 'replaced_files' / (target.name + '.' + uuid4().hex)
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.rename(target, backup)
        try:
            os.rename(source, target)
        except BaseException:
            os.rename(backup, target)
            raise
        try:
            backup.unlink()
        except PermissionError:
            print(f'[Arquivos] Versão anterior em uso preservada em {backup}')


def get(url, accept="*/*"):
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=Retry(
            # A API do Senado limita taxa (429); esperas de até ~2,5 min respeitam o Retry-After.
            total=6, backoff_factor=5, status_forcelist=[429, 500, 502, 503, 504])))
        response = session.get(url, timeout=(20, 180), headers={'Accept': accept})
        response.raise_for_status()
        return response


def money(series):
    """Aceita decimal com ponto ou vírgula e milhares no formato brasileiro."""
    values = series.astype(str).str.strip()
    comma = values.str.contains(',', regex=False)
    values.loc[comma] = values.loc[comma].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(values, errors='coerce')


def parse_camara(content, year):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(names) != 1:
            raise ValueError(f'Câmara {year}: ZIP deve conter exatamente um CSV')
        with archive.open(names[0]) as source:
            frame = pd.read_csv(source, sep=';', encoding='utf-8', dtype=str, keep_default_na=False)
    frame = frame[CAMARA_COLUMNS].copy()
    frame['vlrLiquido'] = money(frame['vlrLiquido'])
    frame['numAno'] = pd.to_numeric(frame['numAno'], errors='raise').astype(int)
    frame['numMes'] = pd.to_numeric(frame['numMes'], errors='raise').astype(int)
    if not frame['numAno'].eq(year).all():
        raise ValueError(f'Ano inesperado no dump Câmara {year}')
    # Lideranças partidárias sem UF não são despesas individuais de parlamentares.
    valid = (frame['vlrLiquido'].gt(0) & frame['sgUF'].isin(UFS)
             & frame['numMes'].between(1, 12)
             & pd.to_numeric(frame['ideCadastro'], errors='coerce').gt(0))
    return frame.loc[valid].reset_index(drop=True)


def name_key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value))
                   if not unicodedata.combining(c)).strip().upper()


def senate_registry():
    """Inclui titulares e suplentes da legislatura; UF vem do mandato oficial."""
    response = get('https://legis.senado.leg.br/dadosabertos/senador/lista/legislatura/57', accept='application/json')
    people = response.json()['ListaParlamentarLegislatura']['Parlamentares']['Parlamentar']
    registry = {}
    for person in people:
        ident = person['IdentificacaoParlamentar']
        mandates = person['Mandatos']['Mandato']
        if isinstance(mandates, dict):
            mandates = [mandates]
        states = {m['UfParlamentar'] for m in mandates}
        if len(states) != 1:
            continue
        info = {'codSenador': ident['CodigoParlamentar'], 'siglaUF': states.pop(),
                'siglaPartido': 'N/I'}  # CEAPS não informa partido na data da despesa.
        for key in ['NomeParlamentar', 'NomeCompletoParlamentar']:
            registry[name_key(ident[key])] = info
    return registry


def parse_senado(content, year, registry):
    text = content.decode('latin1')
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ANO";"MES";'))
    frame = pd.read_csv(io.StringIO('\n'.join(lines[start:])), sep=';', dtype=str, keep_default_na=False)
    frame = frame.rename(columns={
        'COD_DOCUMENTO': 'id', 'ANO': 'ano', 'MES': 'mes', 'SENADOR': 'nomeSenador',
        'TIPO_DESPESA': 'tipoDespesa', 'CNPJ_CPF': 'cpfCnpj', 'FORNECEDOR': 'fornecedor',
        'DOCUMENTO': 'documento', 'DATA': 'data', 'DETALHAMENTO': 'detalhamento',
        'VALOR_REEMBOLSADO': 'valorReembolsado'})
    frame['valorReembolsado'] = money(frame['valorReembolsado'])
    frame = frame.loc[frame['valorReembolsado'].gt(0)].copy()
    for column in ['ano', 'mes']:
        frame[column] = pd.to_numeric(frame[column], errors='raise').astype(int)
    if not frame['ano'].eq(year).all() or not frame['mes'].between(1, 12).all():
        raise ValueError(f'Competência inválida no CEAPS {year}')
    keys = frame['nomeSenador'].map(name_key)
    # CEAPS pode usar uma forma abreviada do nome civil (ex.: Weverton Rocha).
    # Só resolve prefixos completos e inequívocos do cadastro oficial.
    for key in set(keys) - set(registry):
        matches = {value['codSenador']: value for name, value in registry.items()
                   if name.startswith(key + ' ')}
        if len(matches) == 1:
            registry[key] = next(iter(matches.values()))
    missing = sorted(set(keys) - set(registry))
    if missing:
        raise ValueError(f'Senadores sem mandato identificado (não atribuir ao DF): {missing}')
    for column in ['codSenador', 'siglaUF', 'siglaPartido']:
        frame[column] = keys.map({key: value[column] for key, value in registry.items()})
    frame['tipoDocumento'] = ''  # Campo não publicado pelo CSV consolidado.
    dates = pd.to_datetime(frame['data'], format='%d/%m/%Y', errors='coerce')
    invalid = dates.isna() & frame['data'].ne('')
    if invalid.any():
        print(f'[CEAPS {year}] {invalid.sum()} datas inválidas preservadas como publicadas; competência usa ANO/MES.')
    frame['data'] = dates.dt.strftime('%Y-%m-%d').fillna(frame['data'])
    return frame[SENADO_COLUMNS]


def download_expenses():
    registry = senate_registry()
    camara, senado = [], []
    for year in [2025, 2026]:
        print(f'[Bulk] Baixando Câmara e Senado {year}...', flush=True)
        camara.append(parse_camara(get(f'https://www.camara.leg.br/cotas/Ano-{year}.csv.zip').content, year))
        senado.append(parse_senado(get(f'https://www.senado.leg.br/transparencia/LAI/verba/despesa_ceaps_{year}.csv').content, year, registry))
        print(f'[Bulk] {year}: Câmara {len(camara[-1]):,}; Senado {len(senado[-1]):,}', flush=True)
    return pd.concat(camara, ignore_index=True), pd.concat(senado, ignore_index=True)

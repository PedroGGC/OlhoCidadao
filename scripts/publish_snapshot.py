"""Empacota e publica dados verificados; código e manifesto continuam pequenos no Git."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'PedroGGC/OlhoCidadao'
DATA_FILES = [
    'cota_parlamentar_camara_completa.csv', 'deputados_federais_camara.csv',
    'deputados_historico_completo_camara.csv', 'despesas_senadores_ceaps_completo.csv',
    'estados_ibge_oficial.csv', 'indicadores_custos_publicos_br.csv',
    'obras_projetos_publicos_br.csv', 'partidos_politicos_camara.csv',
    'prefeituras_municipios_tesouro_ibge.csv', 'processed_dataset.csv',
    'raw_despesas_camara.csv', 'senadores_federais_completo.csv',
]
WEB_FILES = ['public/data/dashboard_data.json.gz', 'public/data/resumo_geral.json',
             'public/data/loss_curve.png', 'outputs/model_evaluation.json',
             'outputs/model_parity.json', 'outputs/data_quality_report.json',
             'outputs/obrasgov_quality_report.json', 'outputs/relatorio_auditoria.md']


def run(*args):
    for attempt in range(4):
        try:
            subprocess.run(args, cwd=ROOT, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def capture(*args):
    for attempt in range(4):
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, encoding='utf-8').strip()
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def find_release(tag, attempts=5, delay=2):
    # Drafts ainda não possuem tag publicada e não aparecem em /releases/tags/{tag}.
    for attempt in range(attempts):
        pages = json.loads(capture('gh', 'api', '--paginate', '--slurp', f'repos/{REPO}/releases?per_page=100'))
        found = next((release for page in pages for release in page if release['tag_name'] == tag), None)
        if found:
            return found
        if attempt < attempts - 1:
            time.sleep(delay)
    return None


def checksum(filename):
    with open(filename, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def entry(name, profile):
    filename = ROOT / name
    if not filename.is_file() or filename.is_symlink():
        raise ValueError(f'Arquivo necessário ausente ou não regular: {name}')
    with filename.open('rb') as source:
        if source.read(80).startswith(b'version https://git-lfs.github.com/spec/'):
            raise ValueError(f'{name} ainda é um ponteiro LFS; recupere a base antes de publicar.')
    return {'path': name, 'profile': profile, 'bytes': filename.stat().st_size, 'sha256': checksum(filename)}


def create_archive(filename, entries):
    # USTAR regular e metadados determinísticos, compatíveis com o extrator Node.
    with filename.open('wb') as output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                for item in entries:
                    info = tarfile.TarInfo(item['path'])
                    info.size, info.mode, info.mtime = item['bytes'], 0o644, 0
                    with (ROOT / item['path']).open('rb') as source:
                        hasher = hashlib.sha256()
                        class CheckedReader:
                            def read(self, size=-1):
                                chunk = source.read(size)
                                hasher.update(chunk)
                                return chunk
                        archive.addfile(info, CheckedReader())
                        if hasher.hexdigest() != item['sha256'] or source.read(1):
                            raise ValueError(f'Arquivo mudou durante o empacotamento: {item["path"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Publica Release e atualiza o manifesto somente após verificar os uploads')
    args = parser.parse_args()
    run(sys.executable, 'scripts/validate_model.py')
    run(sys.executable, 'scripts/validate_dataset.py', '--update-if-changed')
    files = [entry(name, 'web') for name in WEB_FILES]
    files += [entry('data/'+name, 'research') for name in DATA_FILES]
    files.append(dict(entry('public/data/dashboard_data.json', 'web'), gunzipFrom='public/data/dashboard_data.json.gz'))
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    tag = 'data-'+identity
    folder = ROOT / '.cache' / 'releases' / tag
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'schemaVersion': 1, 'snapshot': tag,
                'createdAtUTC': datetime.now(timezone.utc).isoformat(),
                'sourceCommit': capture('git', 'rev-parse', 'HEAD'),
                'files': files, 'archives': []}
    quality = json.loads((ROOT / 'outputs/data_quality_report.json').read_text(encoding='utf-8'))
    manifest['records'] = quality['totalRecords']
    for profile in ('web', 'research'):
        selected = [item for item in files if item['profile'] == profile and not item.get('gunzipFrom')]
        filename = folder / (profile+'.tar.gz')
        print(f'Empacotando {profile}...', flush=True)
        create_archive(filename, selected)
        manifest['archives'].append({'profile': profile, 'files': [item['path'] for item in selected],
            'url': f'https://github.com/{REPO}/releases/download/{tag}/{filename.name}',
            'bytes': filename.stat().st_size, 'sha256': checksum(filename)})
    prepared = folder / 'data-manifest.json'
    prepared.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'snapshot': tag, 'archivesMB': {item['profile']: round(item['bytes']/1e6, 2)
                                                   for item in manifest['archives']}}, indent=2), flush=True)
    if not args.publish:
        print(f'Preparado em {folder}; use --publish para disponibilizar e fixar no projeto.')
        return

    # Releases são identificadas pelo conteúdo; assets existentes não são sobrescritos.
    release = find_release(tag)
    if release is None:
        notes = folder / 'release-notes.md'
        notes.write_text(f'Snapshot íntegro de {manifest["records"]:,} notas e cadastro municipal.\n\n'
                         'Instale com pnpm install; manifesto, SHA-256 e descompactação são automáticos.\n'
                         'Atipicidade é estatística, não comprovação de fraude.\n', encoding='utf-8')
        run('gh', 'release', 'create', tag, '--repo', REPO, '--target', manifest['sourceCommit'],
            '--draft', '--latest=false', '--title', f'Dados {tag}', '--notes-file', str(notes))
        release = find_release(tag)
        if release is None:
            raise RuntimeError('Release de rascunho não foi localizada após a criação.')
    existing = {asset['name']: asset for asset in release['assets']}
    pending = []
    for item in manifest['archives']:
        name = Path(item['url']).name
        asset = existing.get(name)
        if asset:
            if asset.get('digest') != 'sha256:'+item['sha256'] or asset['size'] != item['bytes']:
                raise ValueError('Asset existente diverge do conteúdo local; não será substituído.')
        else:
            pending.append(str(folder / name))
    if pending:
        run('gh', 'release', 'upload', tag, *pending, '--repo', REPO)
    release = json.loads(capture('gh', 'api', f'repos/{REPO}/releases/{release["id"]}'))
    assets = {asset['name']: asset for asset in release['assets']}
    for item in manifest['archives']:
        asset = assets[Path(item['url']).name]
        if asset.get('digest') != 'sha256:'+item['sha256'] or asset['size'] != item['bytes']:
            raise ValueError('O checksum publicado no GitHub não corresponde ao pacote local.')
    if release['draft']:
        run('gh', 'release', 'edit', tag, '--repo', REPO, '--draft=false', '--latest=false')
        release = json.loads(capture('gh', 'api', f'repos/{REPO}/releases/{release["id"]}'))
        if release['draft']:
            raise RuntimeError('A Release ainda está em rascunho; o manifesto não será atualizado.')
    # Reutiliza datas/origem da primeira publicação idêntica para não gerar diffs vazios.
    current = ROOT / 'data-manifest.json'
    if current.exists():
        previous = json.loads(current.read_text(encoding='utf-8'))
        if previous.get('snapshot') == tag:
            print('Snapshot já publicado e manifesto já está atualizado.')
            return
    temporary = ROOT / 'data-manifest.json.tmp'
    temporary.write_text(prepared.read_text(encoding='utf-8'), encoding='utf-8')
    temporary.replace(current)
    print(f'Publicado e verificado: https://github.com/{REPO}/releases/tag/{tag}. Manifesto pronto para commit.', flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()

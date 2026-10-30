"""Atualiza as fontes oficiais e aplica o modelo congelado, sem retreinar.

As notas que não existiam na coleta usada no treino são avaliadas à parte: nenhuma
etapa de treino, seleção, calibração ou limiar as conheceu. Use --offline para
reaplicar sobre os arquivos locais sem baixar de novo.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from main import export_web_dashboard_payload
from src.bulk_data import NOTE_KEY, notes_absent_from, save_csv
from src.data_fetcher import PROCESSED_CSV_PATH, load_and_preprocess_data
from src.impact_analyzer import generate_social_impact_report
from src.model_evaluation import ExpenseFeatures, classification_metrics, temporal_partitions
from src.model_export import load_model


def group_errors(frame, column):
    wrong = frame.classePredita != frame.target
    return {str(name): {'records': len(group), 'positives': int(group.target.sum()),
                        'errors': int(wrong[group.index].sum())}
            for name, group in frame.groupby(column)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    payload = json.loads(Path('public/data/dashboard_data.json').read_text(encoding='utf-8'))
    params, metrics = payload['modelParams'], payload['metrics']
    evaluation = metrics['evaluation']
    previous = pd.read_csv(PROCESSED_CSV_PATH, dtype=str, keep_default_na=False)
    if 'coletaPosterior' not in previous:
        previous['coletaPosterior'] = '0'
    # Notas da coleta do treino ainda idênticas; marcas 1 (nova) e 2 (revisada) persistem entre execuções.
    base = previous[previous.coletaPosterior == '0']

    _, _, df, metadata = load_and_preprocess_data(refresh=not args.offline)
    # Chaves relidas do CSV recém-gravado: mesma serialização da coleta anterior.
    current = pd.read_csv(PROCESSED_CSV_PATH, dtype=str, keep_default_na=False, usecols=NOTE_KEY)
    # Códigos congelados; categorias publicadas depois recebem códigos novos (desconhecidos pela rede).
    mapping = dict(payload['mappings']['mapaCategoria'])
    missing = set(mapping) - set(df.tipoDespesa)
    if missing:
        raise SystemExit(f'Categorias do treino sumiram da fonte: {sorted(missing)}')
    for name in sorted(set(df.tipoDespesa) - set(mapping)):
        mapping[name] = max(mapping.values()) + 1
    df['codTipoDespesa'] = df.tipoDespesa.map(mapping).astype(int)
    metadata['mapa_categoria'] = mapping

    # Gabarito congelado: limiares de 2025 do treino, mesmo que 2025 tenha recebido notas atrasadas.
    # Categoria nova recebe a mesma regra (média + 1,2 desvios de 2025) calculada só com ela.
    stats = dict(payload['mappings']['categorias'])
    for name, code in mapping.items():
        if str(code) not in stats:
            values = df.valorLiquido[(df.tipoDespesa == name) & (df.ano == 2025)].to_numpy(float)
            threshold = float(values.mean() + 1.2 * values.std())
            stats[str(code)] = {'media_total': float(values.mean()), 'desvio_padrao': float(values.std()),
                                'limiar_atipico': threshold,
                                'media_aceitavel': float(values[values <= threshold].mean()),
                                'referenceRecords': len(values), 'nome': name, 'publishedAfterTraining': True}
    codes = df.codTipoDespesa.astype(str)
    df['limiarCategoria'] = codes.map({c: s['limiar_atipico'] for c, s in stats.items()})
    df['mediaAceitavelCategoria'] = codes.map({c: s['media_aceitavel'] for c, s in stats.items()})
    df['target'] = (df.valorLiquido > df.limiarCategoria).astype(int)
    metadata['stats_categoria'] = stats
    metadata['feature_transform'] = params['featureTransform']

    model = load_model(params)
    mean, std = np.asarray(params['mean']), np.asarray(params['std'])
    X = (ExpenseFeatures.from_config(params['featureTransform']).transform(df) - mean) / (std + 1e-8)
    df['classePredita'], df['probabilidadeAtipico'] = model.predict(X)
    df['excessoMensal'] = np.where(df.classePredita == 1,
                                   np.maximum(df.valorLiquido - df.mediaAceitavelCategoria, 0), 0)
    df['impacto5Anos'] = df.excessoMensal * 60
    occurrence = lambda frame: frame[NOTE_KEY].assign(_n=frame.groupby(NOTE_KEY).cumcount().to_numpy())
    mark = occurrence(current).merge(occurrence(previous).assign(_mark=previous.coletaPosterior.to_numpy()),
                                     on=NOTE_KEY + ['_n'], how='left')['_mark'].to_numpy()
    unseen_key = pd.isna(mark)
    # A fonte às vezes republica a mesma nota com outra categoria ou valor: é revisão, não nota inédita.
    genuinely_new = np.ones(len(current), bool)
    for field in ('tipoDespesa', 'valorLiquido'):
        genuinely_new &= notes_absent_from(base, current, [c for c in NOTE_KEY if c != field])
    mark[unseen_key] = np.where(~genuinely_new[unseen_key], '2', '1')
    df['coletaPosterior'] = mark.astype(int)
    save_csv(df, PROCESSED_CSV_PATH)

    new = df[df.coletaPosterior == 1]
    revisions = int((df.coletaPosterior == 2).sum())
    removed = int(notes_absent_from(current, base).sum()) - int((unseen_key & ~genuinely_new).sum())
    print(f'Notas novas: {len(new)}; revisadas pela fonte: {revisions}; removidas: {removed}')
    if new.empty:
        raise SystemExit('Nenhuma nota nova em relação à coleta do treino.')
    known = new.codTipoDespesa.isin(params['featureTransform']['categories'])
    unseen, new = new[~known], new[known]
    distance = np.abs(new.valorLiquido / new.limiarCategoria - 1)
    wrong = new.classePredita != new.target
    evaluation['newNotesAfterSnapshot'] = {
        'evaluatedAtUTC': datetime.now(timezone.utc).isoformat(),
        'scope': 'Notas ausentes da coleta usada no treino; modelo, calibração, limiar e gabarito congelados.',
        'previousRecords': len(base), 'currentRecords': len(df), 'removedRecords': removed,
        'revisedRecords': revisions,
        'revisedNote': 'Mesma nota republicada com outra categoria ou valor; fora das métricas de notas novas.',
        'metrics': classification_metrics(new.target, new.probabilidadeAtipico, model.decision_threshold),
        'byHouse': group_errors(new, 'cargoPublico'),
        'byPeriod': group_errors(new.assign(periodo=new.ano * 100 + new.mes), 'periodo'),
        'errorsWithin20PercentOfThreshold': int(wrong[distance < .2].sum()),
        # Teste original (jul-out) restrito às notas que a fonte não revisou: confere a reprodutibilidade.
        'unchangedOriginalTest': classification_metrics(*(lambda t: (t.target, t.probabilidadeAtipico))(
            df.iloc[np.intersect1d(temporal_partitions(df)['test'], np.flatnonzero(df.coletaPosterior == 0))]),
            model.decision_threshold),
        # Fora das métricas principais: a rede trata essas categorias como desconhecidas.
        'unseenCategories': {str(name): dict(group_errors(group, 'tipoDespesa')[str(name)],
                                              metrics=classification_metrics(group.target, group.probabilidadeAtipico,
                                                                             model.decision_threshold))
                             for name, group in unseen.groupby('tipoDespesa')},
    }
    Path('outputs/model_evaluation.json').write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    suspects = df[df.classePredita == 1]
    metrics.update(totalConsolidatedRaw=int(metadata['total_consolidado_bruto']), totalRecords=len(df),
                   suspiciousCount=len(suspects),
                   monthlyExcessTotal=round(float(suspects.excessoMensal.sum()), 2),
                   impact5YearsTotal=round(float(suspects.impacto5Anos.sum()), 2))
    generate_social_impact_report(suspects, os.path.join('outputs', 'relatorio_auditoria.md'), top_n=8)
    export_web_dashboard_payload(model, mean, std, df, metadata, metrics,
                                 learning_curve=payload['learningCurve'])

    result = evaluation['newNotesAfterSnapshot']['metrics']
    print(f"Atípicas pelo gabarito entre as novas: {result['positives']}")
    print(f"Precisão={result['precision']:.4%} recall={result['recall']:.4%} "
          f"F1={result['f1']:.6f} matriz={result['confusionMatrix']}")


if __name__ == '__main__':
    main()

import os
# Evita que BLAS abra dezenas de threads para matrizes pequenas do treino em CPU.
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '2')
import sys
import json
import shutil
import argparse
import gzip
import numpy as np
import pandas as pd

# Garante exibição UTF-8 limpa no terminal Windows/Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_fetcher import (
    load_and_preprocess_data,
    DEPUTADOS_CSV_PATH,
    SENADORES_CSV_PATH,
    OBRAS_PROJETOS_CSV_PATH,
)
from src.nn_model import NnModel
from src.model_training import train_and_evaluate
from src.model_export import model_parameters, export_parity_cases
from src.bulk_data import save_csv, publish_file
from src.public_works import works_payload
from src.summary_export import write_summary
from src.impact_analyzer import generate_social_impact_report, format_brl, CUSTOS_REFERENCIA_BR

MAPA_REGIAO_UF = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}


def _load_fotos_parlamentares() -> dict:
    """Carrega URLs de fotos oficiais dos Deputados e Senadores para enriquecer a consulta cidadã."""
    fotos = {}
    if os.path.exists(DEPUTADOS_CSV_PATH):
        try:
            df_d = pd.read_csv(DEPUTADOS_CSV_PATH)
            for _, r in df_d.iterrows():
                fotos[str(r.get("id", ""))] = str(r.get("urlFoto", "") or "")
                fotos[str(r.get("nome", "")).strip().upper()] = str(r.get("urlFoto", "") or "")
        except Exception:
            pass
    if os.path.exists(SENADORES_CSV_PATH):
        try:
            df_s = pd.read_csv(SENADORES_CSV_PATH)
            for _, r in df_s.iterrows():
                fotos[str(r.get("codigoSenador", ""))] = str(r.get("urlFoto", "") or "")
                fotos[str(r.get("nomeParlamentar", "")).strip().upper()] = str(r.get("urlFoto", "") or "")
        except Exception:
            pass
    return fotos


def export_web_dashboard_payload(
    nn: NnModel,
    mean_train: np.ndarray,
    std_train: np.ndarray,
    df_all: pd.DataFrame,
    metadata: dict,
    metrics: dict,
) -> str:
    """
    Exporta os dados auditados (Deputados Federais e Senadores), séries mensais/anuais,
    regiões, métricas da Rede Neural e pesos treinados (W1, b1, W2, b2) para
    `public/data/dashboard_data.json`, alimentando o Portal Web de Consulta Cidadã.
    """
    web_data_dir = os.path.join("public", "data")
    os.makedirs(web_data_dir, exist_ok=True)

    loss_png_src = os.path.join("outputs", "loss_curve.png")
    if os.path.exists(loss_png_src):
        shutil.copyfile(loss_png_src, os.path.join(web_data_dir, "loss_curve.png"))

    fotos_map = _load_fotos_parlamentares()

    # Todas as inferências já foram calculadas na base integral.
    df_calc = df_all.copy()
    df_calc['regiao'] = df_calc['siglaUF'].map(MAPA_REGIAO_UF).fillna('Não identificada')
    nomes = df_calc['nomeParlamentar'].str.replace(' (Dep. Federal)', '', regex=False).str.replace(' (Senador)', '', regex=False).str.strip()
    web = df_calc.rename(columns={'idDeputado': 'idParlamentar', 'cargoPublico': 'cargo',
                                  'nomeParlamentar': 'nomeExibicao'}).copy()
    web['nomeParlamentar'] = nomes
    web['urlFoto'] = web['idParlamentar'].astype(str).map(fotos_map).fillna(nomes.str.upper().map(fotos_map)).fillna('')
    fields = ['idParlamentar', 'nomeParlamentar', 'nomeExibicao', 'cargo', 'siglaPartido',
              'siglaUF', 'regiao', 'tipoDespesa', 'codTipoDespesa', 'nomeFornecedor',
              'cnpjCpfFornecedor', 'ano', 'mes', 'valorLiquido', 'mediaAceitavelCategoria',
              'limiarCategoria', 'excessoMensal', 'impacto5Anos', 'classePredita',
              'probabilidadeAtipico', 'urlDocumento', 'urlFoto']
    # Colunas e dicionários evitam repetir nomes de campos, partidos, categorias e URLs.
    columns = {}
    for field in fields:
        values = web[field]
        if values.dtype == object:
            values = values.fillna('').astype(str)
            codes, dictionary = pd.factorize(values, sort=False)
            if len(dictionary) < len(values) // 2:
                columns[field] = {'dictionary': dictionary.tolist(), 'values': codes.tolist()}
            else:
                columns[field] = {'values': values.tolist()}
        else:
            columns[field] = {'values': values.round(4 if field == 'probabilidadeAtipico' else 2).tolist()}
    metrics['webRecords'] = len(web)
    metrics['sampling'] = 'none'
    metrics['totalSpent'] = round(float(web['valorLiquido'].sum()), 2)

    # Amostra da curva de aprendizado para a guia técnica
    step = max(1, len(nn.loss_history) // 100)
    history_sampled = [
        {
            "epoch": int(i + 1),
            "loss": round(float(nn.loss_history[i]), 5),
            "acc": round(float(nn.accuracy_history[i] * 100.0), 2),
        }
        for i in range(0, len(nn.loss_history), step)
    ]

    obras_projetos = []
    if os.path.exists(OBRAS_PROJETOS_CSV_PATH):
        try:
            df_ob = pd.read_csv(OBRAS_PROJETOS_CSV_PATH, low_memory=False)
            obras_projetos = works_payload(df_ob)
        except Exception:
            pass

    payload = {
        "metrics": metrics,
        "custosReferencia": CUSTOS_REFERENCIA_BR,
        "obrasProjetos": obras_projetos,
        "modelParams": model_parameters(nn, mean_train, std_train, metadata.get('feature_transform')),
        "mappings": {
            "categorias": metadata["stats_categoria"],
            "mapaCategoria": metadata["mapa_categoria"],
            "mapaPartido": metadata["mapa_partido"],
            "mapaUF": metadata["mapa_uf"],
        },
        "learningCurve": history_sampled,
        "recordCount": len(web),
        "recordColumns": columns,
        "dataNotes": {"coverage": "Todas as despesas positivas identificadas de 2025 e 2026; não inclui estornos negativos nem lideranças sem UF.", "costs": "Cenários comparativos estimados, não repasses contratados.", "senateParty": "N/I: o CSV CEAPS não publica partido na data da despesa.", "model": "MLP com referências de 2025, calibração e teste temporais automáticos; mede atipicidade estatística, não corrupção comprovada."},
    }

    export_parity_cases(nn, payload["modelParams"], df_calc)
    write_summary(df_calc, web_data_dir)

    json_path = os.path.join(web_data_dir, "dashboard_data.json")
    with open(json_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    publish_file(json_path + ".tmp", json_path)
    with open(json_path, "rb") as source, gzip.open(json_path + ".gz.tmp", "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target)
    publish_file(json_path + ".gz.tmp", json_path + ".gz")

    return json_path


def balanced_training_indices(df, candidates, limit=5000, seed=42):
    """Classes balanceadas e cobertura de UF/Casa/ano/categoria, sem reposição."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in [0, 1]:
        pool = df.loc[candidates]
        pool = pool[pool['target'].eq(label)]
        count = min(limit // 2, len(pool))
        strata = ['siglaUF', 'cargoPublico', 'ano']
        if 'codTipoDespesa' in pool:
            strata.append('codTipoDespesa')
        groups = list(pool.groupby(strata, sort=True))
        # Primeiro garante uma observação de cada estrato existente.
        base = [int(rng.choice(group.index)) for _, group in groups]
        base = base[:count]
        remainder = np.setdiff1d(pool.index.to_numpy(), base)
        chosen.extend(base)
        if count > len(base):
            chosen.extend(rng.choice(remainder, count - len(base), replace=False).tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='Reutiliza os dumps locais, sem atualizar despesas')
    args = parser.parse_args()
    print("=" * 78)
    print("  OLHOCIDADÃO — AUDITORIA INTELIGENTE DE GASTOS PARLAMENTARES (CEAP)")
    print("  Rede Neural MLP do Zero (NumPy) + Analisador de Impacto Social")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # ETAPA 1: Coleta e Pré-Processamento dos Dados da Câmara e do Senado
    # -------------------------------------------------------------------------
    print("\n[ETAPA 1/4] Coletando e pré-processando dados de despesas parlamentares (CEAP/CEAPS)...")
    X_raw, y, df_processed, metadata = load_and_preprocess_data(seed=42, refresh=not args.offline)

    total_amostras = X_raw.shape[0]
    total_normais = int(np.sum(y == 0))
    total_atipicos = int(np.sum(y == 1))

    print(f"  -> Total de registros processados : {total_amostras}")
    print(f"  -> Dimensão da matriz X           : {X_raw.shape} (entradas legadas para comparação)")
    print(f"  -> Entradas da comparação anterior: {metadata['feature_names']}")
    print(
        f"  -> Distribuição da classe alvo (y): "
        f"Classe 0 (Normal) = {total_normais} ({total_normais/total_amostras*100:.1f}%) | "
        f"Classe 1 (Atípico) = {total_atipicos} ({total_atipicos/total_amostras*100:.1f}%)"
    )

    print("\n[ETAPAS 2–3/4] Treino, calibração e teste temporal separados...")
    nn, X_all_norm, mean_train, std_train, transform, evaluation = train_and_evaluate(
        df_processed, X_raw, balanced_training_indices)
    metadata['feature_transform'] = transform
    results = evaluation['test']
    loss_history, acc_history = nn.loss_history, nn.accuracy_history
    matrix = results['confusionMatrix']
    vp, vn, fp, fn = (matrix[k] for k in ('vp', 'vn', 'fp', 'fn'))
    precisao, revocacao, test_accuracy = results['precision'], results['recall'], results['accuracy']
    print(f"  -> Teste temporal: precisão={precisao:.2%} recall={revocacao:.2%} F1={results['f1']:.4f}")
    print(f"  -> Matriz de confusão: {matrix}")
    print(f"  -> Brier={results['brier']:.6f} ECE={results['ece']:.6f} limiar={nn.decision_threshold:.6f}")
    evaluation_path = os.path.join('outputs', 'model_evaluation.json')
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2, allow_nan=False)

    # -------------------------------------------------------------------------
    # ETAPA 4: Auditoria de Impacto Social nos Casos Classificados como 1
    # -------------------------------------------------------------------------
    print("\n[ETAPA 4/4] Auditando casos classificados como Atípicos (Classe 1) e gerando parecer social...")

    all_preds, all_probs = nn.predict(X_all_norm)
    df_processed['classePredita'] = all_preds
    df_processed['probabilidadeAtipico'] = all_probs
    df_processed['excessoMensal'] = np.where(all_preds == 1,
        np.maximum(df_processed['valorLiquido'] - df_processed['mediaAceitavelCategoria'], 0), 0)
    df_processed['impacto5Anos'] = df_processed['excessoMensal'] * 60
    save_csv(df_processed, os.path.join('data', 'processed_dataset.csv'))
    df_suspeitos = df_processed.loc[all_preds == 1].copy()

    relatorio_path = generate_social_impact_report(
        df_suspeitos=df_suspeitos,
        output_path=os.path.join("outputs", "relatorio_auditoria.md"),
        top_n=8,
    )

    excesso_mensal = (
        df_suspeitos["valorLiquido"] - df_suspeitos["mediaAceitavelCategoria"]
    ).clip(lower=0) if not df_suspeitos.empty else np.array([0.0])
    impacto_5_anos_total = float((excesso_mensal * 60.0).sum())

    metrics_summary = {
        "totalConsolidatedRaw": int(metadata.get("total_consolidado_bruto", total_amostras)),
        "totalRecords": total_amostras,
        "trainRecords": evaluation["trainingRecords"],
        "testRecords": results["records"],
        "evaluation": evaluation,
        "lossInitial": round(float(loss_history[0]), 4),
        "lossFinal": round(float(loss_history[-1]), 4),
        "trainAccuracy": round(float(acc_history[-1] * 100.0), 2),
        "testAccuracy": round(float(test_accuracy * 100.0), 2),
        "precision": round(float(precisao * 100.0), 2),
        "recall": round(float(revocacao * 100.0), 2),
        "confusionMatrix": {"vp": vp, "vn": vn, "fp": fp, "fn": fn},
        "suspiciousCount": int(len(df_suspeitos)),
        "monthlyExcessTotal": round(float(excesso_mensal.sum()), 2),
        "impact5YearsTotal": round(impacto_5_anos_total, 2),
    }

    web_json_path = export_web_dashboard_payload(
        nn=nn,
        mean_train=mean_train,
        std_train=std_train,
        df_all=df_processed,
        metadata=metadata,
        metrics=metrics_summary,

    )

    print(f"  -> Casos atípicos sinalizados na base integral : {len(df_suspeitos)}")
    print(f"  -> Impacto acumulado projetado em 5 anos (60m)    : {format_brl(impacto_5_anos_total)}")
    print(f"  -> Payload Web (pnpm) exportado em                : {web_json_path}")

    print("\n" + "=" * 78)
    print("  PIPELINE EXECUTADO COM SUCESSO! ARQUIVOS GERADOS:")
    print(f"  1. Dataset Bruto      : {os.path.abspath(os.path.join('data', 'raw_despesas_camara.csv'))}")
    print(f"  2. Dataset Processado : {os.path.abspath(os.path.join('data', 'processed_dataset.csv'))}")
    print(f"  3. Curva de Loss (PNG): {os.path.abspath(os.path.join('outputs', 'loss_curve.png'))}")
    print(f"  4. Relatório Auditoria: {os.path.abspath(relatorio_path)}")
    print(f"  5. Dados Web (JSON)   : {os.path.abspath(web_json_path)}")
    print("=" * 78)


if __name__ == "__main__":
    main()

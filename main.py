import os
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
    normalize_zscore,
    DEPUTADOS_CSV_PATH,
    SENADORES_CSV_PATH,
    OBRAS_PROJETOS_CSV_PATH,
)
from src.nn_model import NnModel
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


def train_test_split_numpy(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.20,
    seed: int = 42,
):
    """
    Divide os dados em conjuntos de treino (80%) e teste (20%) utilizando
    exclusivamente operações vetoriais do NumPy (sem scikit-learn).
    Retorna também os índices para recuperar os metadados originais na etapa de auditoria.
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    split_idx = int(n_samples * (1.0 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        train_idx,
        test_idx,
    )


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
    X_all_raw: np.ndarray,
    metadata: dict,
    metrics: dict,
    max_web_records: int = None,
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
            df_ob = pd.read_csv(OBRAS_PROJETOS_CSV_PATH)
            obras_projetos = works_payload(df_ob)
        except Exception:
            pass

    payload = {
        "metrics": metrics,
        "custosReferencia": CUSTOS_REFERENCIA_BR,
        "obrasProjetos": obras_projetos,
        "modelParams": {
            "n_input": nn.n_input,
            "n_hidden": nn.n_hidden,
            "n_output": nn.n_output,
            "lr": nn.lr,
            "activation": nn.activation,
            "mean": mean_train.tolist(),
            "std": std_train.tolist(),
            "W1": nn.W1.tolist(),
            "b1": nn.b1[0].tolist(),
            "W2": nn.W2.tolist(),
            "b2": nn.b2[0].tolist(),
        },
        "mappings": {
            "categorias": metadata["stats_categoria"],
            "mapaCategoria": metadata["mapa_categoria"],
            "mapaPartido": metadata["mapa_partido"],
            "mapaUF": metadata["mapa_uf"],
        },
        "learningCurve": history_sampled,
        "recordCount": len(web),
        "recordColumns": columns,
        "dataNotes": {"coverage": "Todas as despesas positivas identificadas de 2025 e 2026; não inclui estornos negativos nem lideranças sem UF.", "costs": "Cenários comparativos estimados, não repasses contratados.", "senateParty": "N/I: o CSV CEAPS não publica partido na data da despesa.", "model": "Modelo treinado para reproduzir uma regra estatística; não identifica corrupção comprovada."},
    }

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
    """Classes balanceadas; estratos UF/Casa/ano proporcionais, sem reposição."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in [0, 1]:
        pool = df.loc[candidates]
        pool = pool[pool['target'].eq(label)]
        count = min(limit // 2, len(pool))
        groups = list(pool.groupby(['siglaUF', 'cargoPublico', 'ano'], sort=True))
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
    print(f"  -> Dimensão da matriz X           : {X_raw.shape} (5 features numéricas)")
    print(f"  -> Features utilizadas            : {metadata['feature_names']}")
    print(
        f"  -> Distribuição da classe alvo (y): "
        f"Classe 0 (Normal) = {total_normais} ({total_normais/total_amostras*100:.1f}%) | "
        f"Classe 1 (Atípico) = {total_atipicos} ({total_atipicos/total_amostras*100:.1f}%)"
    )

    # -------------------------------------------------------------------------
    # ETAPA 2: Treino balanceado limitado e teste independente
    # -------------------------------------------------------------------------
    print("\n[ETAPA 2/4] Separando teste independente e treino balanceado de até 5.000 notas...")
    rng = np.random.default_rng(42)
    test_idx = rng.choice(total_amostras, min(2000, total_amostras // 5), replace=False)
    candidates = np.setdiff1d(np.arange(total_amostras), test_idx)
    train_idx = balanced_training_indices(df_processed, candidates)
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_norm, mean_train, std_train = normalize_zscore(X_train_raw)
    X_test_norm, _, _ = normalize_zscore(X_test_raw, mean=mean_train, std=std_train)

    print(f"  -> Amostras de Treino : {X_train_norm.shape[0]} registros")
    print(f"  -> Amostras de Teste  : {X_test_norm.shape[0]} registros")
    print("  -> Normalização Z-Score aplicada com sucesso: (X - mean) / (std + 1e-8)")

    # -------------------------------------------------------------------------
    # ETAPA 3: Instanciação, Treinamento e Avaliação da Rede Neural (NnModel)
    # -------------------------------------------------------------------------
    print("\n[ETAPA 3/4] Treinando a Rede Neural Multilayer Perceptron (NnModel) do zero...")
    nn = NnModel(
        X=X_train_norm,
        y=y_train,
        n_hidden=16,
        lr=0.05,
        activation="tanh",
        seed=42,
    )
    loss_history, acc_history = nn.fit(
        epochs=2000,
        verbose_step=400,
        X_val=X_test_norm,
        y_val=y_test,
    )

    y_test_pred, y_test_prob = nn.predict(X_test_norm)
    test_accuracy = float(np.mean(y_test_pred == y_test))

    vp = int(np.sum((y_test_pred == 1) & (y_test == 1)))
    vn = int(np.sum((y_test_pred == 0) & (y_test == 0)))
    fp = int(np.sum((y_test_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_test_pred == 0) & (y_test == 1)))

    precisao = vp / (vp + fp + 1e-8)
    revocacao = vp / (vp + fn + 1e-8)

    print("\n  [Resultados de Desempenho da Rede Neural]")
    print(f"  -> Loss Inicial (Época 1)    : {loss_history[0]:.4f}")
    print(f"  -> Loss Final (Época 2000)   : {loss_history[-1]:.4f}")
    print(f"  -> Acurácia Final em Treino  : {acc_history[-1] * 100:.2f}%")
    print(f"  -> Acurácia Final em Teste   : {test_accuracy * 100:.2f}%")
    print(f"  -> Matriz de Confusão (Teste): VP={vp} | VN={vn} | FP={fp} | FN={fn}")
    print(f"  -> Precisão (Classe Atípica) : {precisao * 100:.2f}% | Recall: {revocacao * 100:.2f}%")

    # -------------------------------------------------------------------------
    # ETAPA 4: Auditoria de Impacto Social nos Casos Classificados como 1
    # -------------------------------------------------------------------------
    print("\n[ETAPA 4/4] Auditando casos classificados como Atípicos (Classe 1) e gerando parecer social...")

    X_all_norm, _, _ = normalize_zscore(X_raw, mean_train, std_train)
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
    merendas_equiv = int(impacto_5_anos_total / 1.50)

    metrics_summary = {
        "totalConsolidatedRaw": int(metadata.get("total_consolidado_bruto", total_amostras)),
        "totalRecords": total_amostras,
        "trainRecords": int(X_train_norm.shape[0]),
        "testRecords": int(X_test_norm.shape[0]),
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
        X_all_raw=X_raw,
        metadata=metadata,
        metrics=metrics_summary,

    )

    print(f"  -> Casos atípicos sinalizados na base integral : {len(df_suspeitos)}")
    print(f"  -> Impacto acumulado projetado em 5 anos (60m)    : {format_brl(impacto_5_anos_total)}")
    print(f"  -> Equivalência direta em Educação (PNAE R$ 1,50) : {merendas_equiv:,} merendas escolares".replace(",", "."))
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

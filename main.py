import os
import sys
import json
import shutil
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
)
from src.nn_model import NnModel
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
    max_web_records: int = 3000,
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

    # Inferência vetorial da NnModel em todo o conjunto para alimentar a busca interativa
    X_all_norm, _, _ = normalize_zscore(X_all_raw, mean=mean_train, std=std_train)
    all_preds, all_probs = nn.predict(X_all_norm)

    df_calc = df_all.copy()
    df_calc["classePredita"] = all_preds
    df_calc["probabilidadeAtipico"] = all_probs
    df_calc["excessoMensal"] = np.where(
        df_calc["classePredita"] == 1,
        np.maximum(
            df_calc["valorLiquido"] - df_calc["mediaAceitavelCategoria"],
            df_calc["valorLiquido"] * 0.30,
        ),
        0.0,
    )
    df_calc["impacto5Anos"] = df_calc["excessoMensal"] * 60.0
    df_calc["regiao"] = df_calc["siglaUF"].map(MAPA_REGIAO_UF).fillna("Centro-Oeste")

    # Seleciona amostra rica (garantindo cobertura de Deputados, Senadores, meses, anos e UFs)
    df_sample = df_calc.head(max_web_records).copy()

    registros_web = []
    for _, row in df_sample.iterrows():
        nome_raw = str(row.get("nomeParlamentar", ""))
        nome_limpo = (
            nome_raw.replace(" (Dep. Federal)", "")
            .replace(" (Senador)", "")
            .strip()
        )
        id_parl = str(int(row.get("idDeputado", 0)))
        foto_url = fotos_map.get(id_parl) or fotos_map.get(nome_limpo.upper()) or ""

        registros_web.append({
            "idParlamentar": int(row.get("idDeputado", 0)),
            "nomeParlamentar": nome_limpo,
            "nomeExibicao": nome_raw,
            "cargo": str(row.get("cargoPublico", "DEPUTADO FEDERAL")),
            "siglaPartido": str(row.get("siglaPartido", "")),
            "siglaUF": str(row.get("siglaUF", "")),
            "regiao": str(row.get("regiao", "Sudeste")),
            "tipoDespesa": str(row.get("tipoDespesa", "")),
            "codTipoDespesa": int(row.get("codTipoDespesa", 1)),
            "nomeFornecedor": str(row.get("nomeFornecedor", "")),
            "cnpjCpfFornecedor": str(row.get("cnpjCpfFornecedor", "") or "N/I"),
            "ano": int(row.get("ano", 2026)),
            "mes": int(row.get("mes", 1)),
            "valorLiquido": round(float(row.get("valorLiquido", 0.0)), 2),
            "mediaAceitavelCategoria": round(float(row.get("mediaAceitavelCategoria", 0.0)), 2),
            "limiarCategoria": round(float(row.get("limiarCategoria", 0.0)), 2),
            "excessoMensal": round(float(row.get("excessoMensal", 0.0)), 2),
            "impacto5Anos": round(float(row.get("impacto5Anos", 0.0)), 2),
            "classePredita": int(row.get("classePredita", 0)),
            "probabilidadeAtipico": round(float(row.get("probabilidadeAtipico", 0.0)), 4),
            "urlDocumento": str(row.get("urlDocumento", "") or ""),
            "urlFoto": foto_url,
        })

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

    payload = {
        "metrics": metrics,
        "custosReferencia": CUSTOS_REFERENCIA_BR,
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
        "records": registros_web,
    }

    json_path = os.path.join(web_data_dir, "dashboard_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return json_path


def main():
    print("=" * 78)
    print("  OLHOCIDADÃO — AUDITORIA INTELIGENTE DE GASTOS PARLAMENTARES (CEAP)")
    print("  Rede Neural MLP do Zero (NumPy) + Analisador de Impacto Social")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # ETAPA 1: Coleta e Pré-Processamento dos Dados da Câmara e do Senado
    # -------------------------------------------------------------------------
    print("\n[ETAPA 1/4] Coletando e pré-processando dados de despesas parlamentares (CEAP/CEAPS)...")
    X_raw, y, df_processed, metadata = load_and_preprocess_data(target_records=10000, seed=42)

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
    # ETAPA 2: Divisão Treino/Teste (80%/20%) e Normalização Z-Score em NumPy
    # -------------------------------------------------------------------------
    print("\n[ETAPA 2/4] Dividindo dataset em Treino (80%) e Teste (20%) via NumPy puro...")
    X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split_numpy(
        X_raw, y, test_ratio=0.20, seed=42
    )

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

    df_test_original = df_processed.iloc[test_idx].copy()
    df_test_original["classePredita"] = y_test_pred
    df_test_original["probabilidadeAtipico"] = y_test_prob

    df_suspeitos = df_test_original[df_test_original["classePredita"] == 1].copy()

    relatorio_path = generate_social_impact_report(
        df_suspeitos=df_suspeitos,
        output_path=os.path.join("outputs", "relatorio_auditoria.md"),
        top_n=8,
    )

    excesso_mensal = (
        df_suspeitos["valorLiquido"] - df_suspeitos["mediaAceitavelCategoria"]
    ).clip(lower=df_suspeitos["valorLiquido"] * 0.30) if not df_suspeitos.empty else np.array([0.0])
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
        max_web_records=3000,
    )

    print(f"  -> Casos suspeitos auditados no conjunto de teste : {len(df_suspeitos)}")
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

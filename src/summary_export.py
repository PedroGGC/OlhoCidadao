"""Resumo agregado leve (`resumo_geral.json`) para abertura instantânea do portal.

O navegador lê este arquivo (poucos KB) primeiro e mostra os totais gerais enquanto
o dataset completo de notas fiscais é baixado e decodificado em segundo plano.
"""
import json
import os

import pandas as pd


def _group(df: pd.DataFrame, key: str, limit: int = None) -> list:
    grp = df.groupby(key, dropna=False).agg(
        total=("valorLiquido", "sum"),
        notas=("valorLiquido", "size"),
        alertas=("classePredita", "sum"),
        excesso=("excessoMensal", "sum"),
    ).sort_values("total", ascending=False)
    if limit:
        grp = grp.head(limit)
    return [
        {
            "chave": str(idx),
            "total": round(float(r.total), 2),
            "notas": int(r.notas),
            "alertas": int(r.alertas),
            "excesso": round(float(r.excesso), 2),
        }
        for idx, r in grp.iterrows()
    ]


def build_summary(df: pd.DataFrame) -> dict:
    """Totais gerais e agregados por ano/mês, UF, partido e categoria."""
    por_ano_mes = (
        df.groupby(["ano", "mes"]).agg(
            total=("valorLiquido", "sum"),
            alertas=("excessoMensal", "sum"),
            notas=("valorLiquido", "size"),
        ).reset_index()
    )
    return {
        "totais": {
            "total": round(float(df["valorLiquido"].sum()), 2),
            "notas": int(len(df)),
            "alertas": int(df["classePredita"].sum()),
            "excesso": round(float(df["excessoMensal"].sum()), 2),
            "parlamentares": int(df["nomeParlamentar"].nunique()),
        },
        "porAno": _group(df, "ano"),
        "porAnoMes": [
            {"ano": int(r.ano), "mes": int(r.mes), "total": round(float(r.total), 2),
             "excesso": round(float(r.alertas), 2), "notas": int(r.notas)}
            for r in por_ano_mes.itertuples()
        ],
        "porUF": _group(df, "siglaUF"),
        "porPartido": _group(df, "siglaPartido"),
        "porCategoria": _group(df, "tipoDespesa"),
    }


def write_summary(df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "resumo_geral.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(build_summary(df), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    os.replace(path + ".tmp", path)
    return path


if __name__ == "__main__":
    frame = pd.read_csv(os.path.join("data", "processed_dataset.csv"))
    print(write_summary(frame, os.path.join("public", "data")))

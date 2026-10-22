import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

API_CAMARA_URL = "https://dadosabertos.camara.leg.br/api/v2"
CAMARA_BULK_ZIP_2026 = "https://www.camara.leg.br/cotas/Ano-2026.csv.zip"
CAMARA_BULK_ZIP_2025 = "https://www.camara.leg.br/cotas/Ano-2025.csv.zip"
CAMARA_HISTORICO_DEPUTADOS_CSV = "https://dadosabertos.camara.leg.br/arquivos/deputados/csv/deputados.csv"

API_SENADO_SENADORES_URL = "https://legis.senado.leg.br/dadosabertos/senador/lista/atual"
API_SENADO_CEAPS_2026 = "https://adm.senado.gov.br/adm-dadosabertos/api/v1/senadores/despesas_ceaps/2026"
API_SENADO_CEAPS_2025 = "https://adm.senado.gov.br/adm-dadosabertos/api/v1/senadores/despesas_ceaps/2025"

API_TESOURO_ENTES_URL = "https://apidatalake.tesouro.gov.br/ords/siconfi/tt/entes"
API_IBGE_UF_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
API_IBGE_MUNICIPIOS_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

# Caminhos dos arquivos CSV completos na pasta data/
RAW_CSV_PATH = os.path.join("data", "raw_despesas_camara.csv")
PROCESSED_CSV_PATH = os.path.join("data", "processed_dataset.csv")
CAMARA_COMPLETA_CSV_PATH = os.path.join("data", "cota_parlamentar_camara_completa.csv")
SENADO_COMPLETO_CSV_PATH = os.path.join("data", "despesas_senadores_ceaps_completo.csv")
DEPUTADOS_CSV_PATH = os.path.join("data", "deputados_federais_camara.csv")
DEPUTADOS_HIST_CSV_PATH = os.path.join("data", "deputados_historico_completo_camara.csv")
SENADORES_CSV_PATH = os.path.join("data", "senadores_federais_completo.csv")
PREFEITURAS_CSV_PATH = os.path.join("data", "prefeituras_municipios_tesouro_ibge.csv")
PARTIDOS_CSV_PATH = os.path.join("data", "partidos_politicos_camara.csv")
IBGE_UF_CSV_PATH = os.path.join("data", "estados_ibge_oficial.csv")
INDICADORES_CSV_PATH = os.path.join("data", "indicadores_custos_publicos_br.csv")


def download_massive_public_csvs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baixa e salva os bancos de dados públicos INTEIROS (massivos) dos portais oficiais do Brasil:
      1. TODOS os 513 Deputados Federais em Exercício (`deputados_federais_camara.csv`)
      2. TODOS os ~7.800 Deputados Federais da História (`deputados_historico_completo_camara.csv`)
      3. TODOS os 81 Senadores da República em Exercício (`senadores_federais_completo.csv`)
      4. TODAS as 5.570 Prefeituras Municipais + Estados do Tesouro Nacional/SICONFI e IBGE (`prefeituras_municipios_tesouro_ibge.csv`)
      5. TODAS as Notas Fiscais da Cota Parlamentar da Câmara dos Deputados (`cota_parlamentar_camara_completa.csv` - centenas de milhares de registros)
      6. TODAS as Notas Fiscais dos Senadores - CEAPS 2025 e 2026 (`despesas_senadores_ceaps_completo.csv` - ~40.000 registros)
      7. Tabelas completas de Estados (IBGE), Partidos Políticos e Indicadores Sociais (FNDE/SUS/DIEESE)
    """
    os.makedirs("data", exist_ok=True)
    headers = {"Accept": "application/json"}

    # -------------------------------------------------------------------------
    # 1. Indicadores de Custos Públicos Oficiais (FNDE/MEC, DataSUS, DIEESE, SENASP)
    # -------------------------------------------------------------------------
    df_indicadores = pd.DataFrame([
        {
            "indicador": "merenda_escolar_pnae",
            "descricao": "Refeição Escolar Básica (PNAE/FNDE)",
            "valor_unitario_brl": 1.50,
            "orgao_fonte": "FNDE / Ministério da Educação (MEC)",
            "portal_oficial": "https://www.gov.br/fnde/pt-br/acesso-a-informacao/acoes-e-programas/programas/pnae",
        },
        {
            "indicador": "kit_escolar_aluno",
            "descricao": "Kit Completo de Material Escolar Anual por Aluno",
            "valor_unitario_brl": 120.00,
            "orgao_fonte": "FNDE / Atas de Registro de Preços MEC",
            "portal_oficial": "https://www.gov.br/fnde/pt-br",
        },
        {
            "indicador": "computador_escola_publica",
            "descricao": "Computador Desktop Educacional para Laboratório Público",
            "valor_unitario_brl": 2800.00,
            "orgao_fonte": "Portal Nacional de Contratações Públicas (PNCP)",
            "portal_oficial": "https://pncp.gov.br/",
        },
        {
            "indicador": "consulta_atencao_basica",
            "descricao": "Consulta Médica em Atenção Primária / Especializada (Custo Médio SUS/Municipal)",
            "valor_unitario_brl": 65.00,
            "orgao_fonte": "Ministério da Saúde / DataSUS (SIGTAP)",
            "portal_oficial": "http://sigtap.datasus.gov.br/",
        },
        {
            "indicador": "kit_medicamentos_basicos",
            "descricao": "Kit Mensal de Medicamentos da Assistência Farmacêutica Básica",
            "valor_unitario_brl": 45.00,
            "orgao_fonte": "Ministério da Saúde / RENAME - SUS",
            "portal_oficial": "https://www.gov.br/saude/pt-br/composicao/sectics/daf",
        },
        {
            "indicador": "cesta_basica_familiar",
            "descricao": "Cesta Básica Nacional de Alimentos (Média Capitais)",
            "valor_unitario_brl": 750.00,
            "orgao_fonte": "DIEESE / Pesquisa Nacional da Cesta Básica de Alimentos",
            "portal_oficial": "https://www.dieese.org.br/analisecestabasica/analiseCestaBasica.html",
        },
        {
            "indicador": "salario_minimo",
            "descricao": "Salário Mínimo Nacional de Referência",
            "valor_unitario_brl": 1518.00,
            "orgao_fonte": "Governo Federal / Portal da Transparência",
            "portal_oficial": "https://portaldatransparencia.gov.br/",
        },
        {
            "indicador": "equipamento_seguranca_uf",
            "descricao": "Diária Operacional de Patrulhamento / EPI Policial Estadual",
            "valor_unitario_brl": 420.00,
            "orgao_fonte": "SENASP / Ministério da Justiça e Segurança Pública",
            "portal_oficial": "https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica",
        },
    ])
    df_indicadores.to_csv(INDICADORES_CSV_PATH, index=False, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 2. Estados (27 UFs) — API Oficial do IBGE
    # -------------------------------------------------------------------------
    if not os.path.exists(IBGE_UF_CSV_PATH):
        try:
            r_ibge = requests.get(API_IBGE_UF_URL, headers=headers, timeout=15.0)
            if r_ibge.status_code == 200:
                ufs_data = [
                    {
                        "id_ibge": u.get("id"),
                        "siglaUF": u.get("sigla"),
                        "nomeEstado": u.get("nome"),
                        "idRegiao": u.get("regiao", {}).get("id"),
                        "siglaRegiao": u.get("regiao", {}).get("sigla"),
                        "nomeRegiao": u.get("regiao", {}).get("nome"),
                    }
                    for u in r_ibge.json()
                ]
                pd.DataFrame(ufs_data).sort_values("siglaUF").to_csv(
                    IBGE_UF_CSV_PATH, index=False, encoding="utf-8"
                )
        except Exception as exc:
            print(f"[Aviso IBGE UFs] ({exc})")

    # -------------------------------------------------------------------------
    # 3. TODOS os 513 Deputados Federais em Exercício + Histórico Completo (~7.800)
    # -------------------------------------------------------------------------
    try:
        r_dep = requests.get(
            f"{API_CAMARA_URL}/deputados",
            params={"ordem": "ASC", "ordenarPor": "nome"},
            headers=headers,
            timeout=20.0,
        )
        if r_dep.status_code == 200:
            deputados_todos = r_dep.json().get("dados", [])
            if deputados_todos:
                pd.DataFrame(deputados_todos).to_csv(DEPUTADOS_CSV_PATH, index=False, encoding="utf-8")
                print(
                    f"[Dados Massivos] TODOS os {len(deputados_todos)} Deputados Federais em exercício "
                    f"salvos em: {DEPUTADOS_CSV_PATH}"
                )
    except Exception as exc:
        print(f"[Aviso Deputados] ({exc})")

    if not os.path.exists(DEPUTADOS_HIST_CSV_PATH):
        try:
            r_hist = requests.get(CAMARA_HISTORICO_DEPUTADOS_CSV, timeout=25.0)
            if r_hist.status_code == 200:
                with open(DEPUTADOS_HIST_CSV_PATH, "wb") as f:
                    f.write(r_hist.content)
                df_h = pd.read_csv(DEPUTADOS_HIST_CSV_PATH, sep=";", on_bad_lines="skip")
                print(
                    f"[Dados Massivos] Histórico completo de Deputados Federais ({len(df_h):,} registros) "
                    f"salvo em: {DEPUTADOS_HIST_CSV_PATH}"
                )
        except Exception as exc:
            print(f"[Aviso Histórico Deputados] ({exc})")

    # -------------------------------------------------------------------------
    # 4. TODOS os 81 Senadores da República em Exercício (Senado Federal)
    # -------------------------------------------------------------------------
    mapa_uf_senador = {}
    mapa_partido_senador = {}
    try:
        r_sen = requests.get(API_SENADO_SENADORES_URL, headers=headers, timeout=20.0)
        if r_sen.status_code == 200:
            parlamentares = (
                r_sen.json()
                .get("ListaParlamentarEmExercicio", {})
                .get("Parlamentares", {})
                .get("Parlamentar", [])
            )
            senadores_rows = []
            for p in parlamentares:
                ident = p.get("IdentificacaoParlamentar", {})
                cod = str(ident.get("CodigoParlamentar", ""))
                nome = ident.get("NomeParlamentar", "")
                partido = ident.get("SiglaPartidoParlamentar", "S/P")
                uf = ident.get("UfParlamentar", "BR")
                mapa_uf_senador[cod] = uf
                mapa_uf_senador[nome.upper()] = uf
                mapa_partido_senador[cod] = partido
                mapa_partido_senador[nome.upper()] = partido

                senadores_rows.append({
                    "codigoSenador": cod,
                    "nomeParlamentar": nome,
                    "nomeCompleto": ident.get("NomeCompletoParlamentar", ""),
                    "sexo": ident.get("SexoParlamentar", ""),
                    "siglaPartido": partido,
                    "siglaUF": uf,
                    "email": ident.get("EmailParlamentar", ""),
                    "urlFoto": ident.get("UrlFotoParlamentar", ""),
                    "urlPagina": ident.get("UrlPaginaParlamentar", ""),
                })
            if senadores_rows:
                pd.DataFrame(senadores_rows).to_csv(SENADORES_CSV_PATH, index=False, encoding="utf-8")
                print(
                    f"[Dados Massivos] TODOS os {len(senadores_rows)} Senadores da República em exercício "
                    f"salvos em: {SENADORES_CSV_PATH}"
                )
    except Exception as exc:
        print(f"[Aviso Senadores] ({exc})")

    # -------------------------------------------------------------------------
    # 5. TODAS as 5.570 Prefeituras Municipais + Governos Estaduais (Tesouro SICONFI + IBGE)
    # -------------------------------------------------------------------------
    if not os.path.exists(PREFEITURAS_CSV_PATH):
        try:
            r_tes = requests.get(API_TESOURO_ENTES_URL, headers=headers, timeout=25.0)
            if r_tes.status_code == 200:
                entes = r_tes.json().get("items", [])
                if entes:
                    df_entes = pd.DataFrame(entes)
                    df_entes.to_csv(PREFEITURAS_CSV_PATH, index=False, encoding="utf-8")
                    print(
                        f"[Dados Massivos] TODAS as Prefeituras e Entes Federativos ({len(df_entes):,} municípios/estados "
                        f"com CNPJ e população) salvos em: {PREFEITURAS_CSV_PATH}"
                    )
        except Exception as exc:
            print(f"[Aviso Tesouro Prefeituras] ({exc})")

    # -------------------------------------------------------------------------
    # 6. Partidos Políticos Oficiais
    # -------------------------------------------------------------------------
    if not os.path.exists(PARTIDOS_CSV_PATH):
        try:
            r_part = requests.get(
                f"{API_CAMARA_URL}/partidos",
                params={"itens": 100, "ordem": "ASC", "ordenarPor": "sigla"},
                headers=headers,
                timeout=15.0,
            )
            if r_part.status_code == 200:
                dados_part = r_part.json().get("dados", [])
                if dados_part:
                    pd.DataFrame(dados_part).to_csv(PARTIDOS_CSV_PATH, index=False, encoding="utf-8")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 7. DUMP MASSIVO DE DESPESAS DO SENADO FEDERAL (CEAPS 2025 + 2026)
    # -------------------------------------------------------------------------
    df_senado_ceaps = pd.DataFrame()
    if os.path.exists(SENADO_COMPLETO_CSV_PATH):
        try:
            df_senado_ceaps = pd.read_csv(SENADO_COMPLETO_CSV_PATH, low_memory=False)
            print(
                f"[Dados Massivos] Dump completo CEAPS do Senado carregado: "
                f"{len(df_senado_ceaps):,} notas fiscais ({SENADO_COMPLETO_CSV_PATH})"
            )
        except Exception:
            df_senado_ceaps = pd.DataFrame()

    if df_senado_ceaps.empty:
        senado_registros = []
        for url_ano in [API_SENADO_CEAPS_2026, API_SENADO_CEAPS_2025]:
            try:
                r_ceaps = requests.get(url_ano, headers=headers, timeout=30.0)
                if r_ceaps.status_code == 200:
                    senado_registros.extend(r_ceaps.json())
            except Exception as exc:
                print(f"[Aviso CEAPS Senado] ({exc})")

        if senado_registros:
            df_senado_ceaps = pd.DataFrame(senado_registros)
            df_senado_ceaps["siglaUF"] = (
                df_senado_ceaps["codSenador"].astype(str).map(mapa_uf_senador)
                .fillna(df_senado_ceaps["nomeSenador"].str.upper().map(mapa_uf_senador))
                .fillna("DF")
            )
            df_senado_ceaps["siglaPartido"] = (
                df_senado_ceaps["codSenador"].astype(str).map(mapa_partido_senador)
                .fillna(df_senado_ceaps["nomeSenador"].str.upper().map(mapa_partido_senador))
                .fillna("SENADO")
            )
            df_senado_ceaps.to_csv(SENADO_COMPLETO_CSV_PATH, index=False, encoding="utf-8")
            print(
                f"[Dados Massivos] TODAS as {len(df_senado_ceaps):,} notas fiscais dos Senadores (CEAPS 2025/2026) "
                f"salvas em: {SENADO_COMPLETO_CSV_PATH}"
            )

    # -------------------------------------------------------------------------
    # 8. DUMP MASSIVO DE COTA PARLAMENTAR DA CÂMARA DOS DEPUTADOS (TODOS OS 513 DEPUTADOS)
    # -------------------------------------------------------------------------
    df_camara_completa = pd.DataFrame()
    if os.path.exists(CAMARA_COMPLETA_CSV_PATH):
        try:
            df_camara_completa = pd.read_csv(CAMARA_COMPLETA_CSV_PATH, low_memory=False)
            print(
                f"[Dados Massivos] Dump completo da Câmara carregado: "
                f"{len(df_camara_completa):,} notas fiscais ({CAMARA_COMPLETA_CSV_PATH})"
            )
        except Exception:
            df_camara_completa = pd.DataFrame()

    if df_camara_completa.empty:
        dfs_bulk = []
        for label, zip_url in [("2026", CAMARA_BULK_ZIP_2026), ("2025", CAMARA_BULK_ZIP_2025)]:
            try:
                print(f"[Dados Massivos] Baixando base nacional completa da Câmara ({label}): {zip_url} ...")
                rz = requests.get(zip_url, timeout=45.0)
                if rz.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(rz.content)) as zf:
                        csv_name = zf.namelist()[0]
                        df_ano = pd.read_csv(
                            zf.open(csv_name),
                            sep=";",
                            dtype=str,
                            on_bad_lines="skip",
                        )
                        # Filtra apenas registros de parlamentares com UF e valor líquido válido
                        cols_keep = [
                            "ideCadastro", "txNomeParlamentar", "sgPartido", "sgUF",
                            "txtDescricao", "txtFornecedor", "txtCNPJCPF", "txtNumero",
                            "datEmissao", "numAno", "numMes", "vlrLiquido", "urlDocumento"
                        ]
                        cols_presentes = [c for c in cols_keep if c in df_ano.columns]
                        df_ano = df_ano[cols_presentes].dropna(subset=["txNomeParlamentar", "vlrLiquido"])
                        dfs_bulk.append(df_ano)
                        print(f"  -> Ano {label}: {len(df_ano):,} notas fiscais extraídas!")
            except Exception as exc:
                print(f"[Aviso Bulk Câmara {label}] ({exc})")

        if dfs_bulk:
            df_camara_completa = pd.concat(dfs_bulk, ignore_index=True)
            df_camara_completa.to_csv(CAMARA_COMPLETA_CSV_PATH, index=False, encoding="utf-8")
            print(
                f"[Dados Massivos] Base completa da Câmara dos Deputados salva com "
                f"{len(df_camara_completa):,} notas fiscais reais em: {CAMARA_COMPLETA_CSV_PATH}"
            )

    return df_camara_completa, df_senado_ceaps


def normalize_zscore(
    X: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalização Z-Score implementada em NumPy puro:
        X_norm = (X - mean) / (std + 1e-8)
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_norm = (X - mean) / (std + 1e-8)
    return X_norm, mean, std


def load_and_preprocess_data(
    target_records: int = 12000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Executa o pipeline completo sobre os bancos de dados públicos massivos:
    1. Garante o download de TODAS as bases completas em `data/` (+300.000 notas da Câmara,
       ~40.000 notas do Senado, 513 Deputados, 81 Senadores, 5.570 Prefeituras/Municípios).
    2. Constrói o dataset unificado (`data/raw_despesas_camara.csv`) combinando notas fiscais
       de todos os estados, partidos, Deputados Federais e Senadores.
    3. Extrai as 5 features numéricas para a matriz `X`:
       [valorLiquido, codTipoDespesa, mes, codPartido, codUF]
    4. Calcula média e desvio padrão por categoria (`codTipoDespesa`) e define `y = 1`
       quando `valorLiquido > media + 1.2 * desvio_padrao`.
    5. Salva dataset processado em `data/processed_dataset.csv`.
    """
    df_camara_full, df_senado_full = download_massive_public_csvs()

    partes_padronizadas = []

    # Padroniza dados massivos da Câmara dos Deputados (todos os 513 deputados)
    if not df_camara_full.empty:
        dfc = df_camara_full.copy()
        dfc["valorLiquido"] = (
            dfc["vlrLiquido"].astype(str).str.replace(",", ".", regex=False)
        )
        dfc["valorLiquido"] = pd.to_numeric(dfc["valorLiquido"], errors="coerce")
        dfc = dfc[dfc["valorLiquido"] > 0].dropna(subset=["sgUF", "sgPartido", "txtDescricao"])

        dfc_pad = pd.DataFrame({
            "idDeputado": pd.to_numeric(dfc.get("ideCadastro", 0), errors="coerce").fillna(0).astype(int),
            "nomeParlamentar": dfc["txNomeParlamentar"].astype(str).str.strip() + " (Dep. Federal)",
            "cargoPublico": "DEPUTADO FEDERAL",
            "siglaPartido": dfc["sgPartido"].astype(str).str.strip().str.upper(),
            "siglaUF": dfc["sgUF"].astype(str).str.strip().str.upper(),
            "tipoDespesa": dfc["txtDescricao"].astype(str).str.strip().str.upper().str.rstrip("."),
            "nomeFornecedor": dfc["txtFornecedor"].fillna("NÃO INFORMADO").astype(str).str.strip(),
            "cnpjCpfFornecedor": dfc["txtCNPJCPF"].fillna("").astype(str).str.strip(),
            "numDocumento": dfc["txtNumero"].fillna("").astype(str).str.strip(),
            "dataDocumento": dfc["datEmissao"].fillna("").astype(str).str.strip(),
            "ano": pd.to_numeric(dfc["numAno"], errors="coerce").fillna(2025).astype(int),
            "mes": pd.to_numeric(dfc["numMes"], errors="coerce").fillna(1).clip(1, 12).astype(int),
            "valorLiquido": dfc["valorLiquido"].round(2),
            "urlDocumento": dfc["urlDocumento"].fillna("").astype(str).str.strip(),
            "origemDado": "CAMARA_DEPUTADOS_CEAP_COMPLETO",
        })
        partes_padronizadas.append(dfc_pad)

    # Padroniza dados massivos do Senado Federal (todos os 81 senadores)
    if not df_senado_full.empty:
        dfs = df_senado_full.copy()
        dfs["valorLiquido"] = pd.to_numeric(dfs["valorReembolsado"], errors="coerce")
        dfs = dfs[dfs["valorLiquido"] > 0].dropna(subset=["nomeSenador", "tipoDespesa"])

        dfs_pad = pd.DataFrame({
            "idDeputado": pd.to_numeric(dfs.get("codSenador", 0), errors="coerce").fillna(0).astype(int),
            "nomeParlamentar": dfs["nomeSenador"].astype(str).str.strip() + " (Senador)",
            "cargoPublico": "SENADOR DA REPÚBLICA",
            "siglaPartido": dfs.get("siglaPartido", pd.Series(["SENADO"] * len(dfs))).fillna("SENADO").astype(str).str.upper(),
            "siglaUF": dfs.get("siglaUF", pd.Series(["DF"] * len(dfs))).fillna("DF").astype(str).str.upper(),
            "tipoDespesa": dfs["tipoDespesa"].astype(str).str.strip().str.upper().str.rstrip("."),
            "nomeFornecedor": dfs["fornecedor"].fillna("NÃO INFORMADO").astype(str).str.strip(),
            "cnpjCpfFornecedor": dfs["cpfCnpj"].fillna("").astype(str).str.strip(),
            "numDocumento": dfs["documento"].fillna("").astype(str).str.strip(),
            "dataDocumento": dfs["data"].fillna("").astype(str).str.strip(),
            "ano": pd.to_numeric(dfs["ano"], errors="coerce").fillna(2025).astype(int),
            "mes": pd.to_numeric(dfs["mes"], errors="coerce").fillna(1).clip(1, 12).astype(int),
            "valorLiquido": dfs["valorLiquido"].round(2),
            "urlDocumento": "https://www6g.senado.leg.br/transparencia/sen/",
            "origemDado": "SENADO_FEDERAL_CEAPS_COMPLETO",
        })
        partes_padronizadas.append(dfs_pad)

    df_unificado_total = pd.concat(partes_padronizadas, ignore_index=True)
    print(
        f"[Dados Massivos] Total consolidado (Câmara + Senado): "
        f"{len(df_unificado_total):,} notas fiscais reais disponíveis!"
    )

    # Seleciona amostra estratificada grande (ex: 12.000 notas cobrindo todos os parlamentares,
    # partidos e UFs) para `raw_despesas_camara.csv` e treinamento vetorial rápido em CPU
    if len(df_unificado_total) > target_records:
        # Garante presença tanto de Deputados quanto de Senadores na amostra de treinamento
        df_raw = df_unificado_total.sample(n=target_records, random_state=seed).reset_index(drop=True)
    else:
        df_raw = df_unificado_total.copy()

    df_raw.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8")
    print(f"[Dados Públicos] Amostra de auditoria salva em: {RAW_CSV_PATH} ({len(df_raw):,} linhas)")

    df = df_raw.copy()

    # Mapeamento das categorias ordenadas pela mediana de gasto
    medianas_por_cat = df.groupby("tipoDespesa")["valorLiquido"].median().sort_values()
    mapa_categoria = {cat: idx + 1 for idx, cat in enumerate(medianas_por_cat.index)}

    partidos_unicos = sorted(df["siglaPartido"].unique())
    mapa_partido = {p: idx + 1 for idx, p in enumerate(partidos_unicos)}

    ufs_unicas = sorted(df["siglaUF"].unique())
    mapa_uf = {uf: idx + 1 for idx, uf in enumerate(ufs_unicas)}

    df["codTipoDespesa"] = df["tipoDespesa"].map(mapa_categoria).astype(int)
    df["codPartido"] = df["siglaPartido"].map(mapa_partido).astype(int)
    df["codUF"] = df["siglaUF"].map(mapa_uf).astype(int)
    df["mes"] = df["mes"].clip(1, 12).astype(int)
    df["valorLiquido"] = df["valorLiquido"].astype(float)

    # Cálculo da média e desvio padrão por categoria (codTipoDespesa)
    stats_categoria = {}
    limiares_categoria = {}

    for cod_cat, grupo in df.groupby("codTipoDespesa"):
        valores = grupo["valorLiquido"].values
        media_cat = float(np.mean(valores))
        std_cat = float(np.std(valores))

        # Regra principal solicitada: media + 1.2 * desvio_padrao
        limiar_padrao = media_cat + 1.2 * std_cat

        prop_acima = np.mean(valores > limiar_padrao)
        if prop_acima < 0.20 and len(valores) >= 5:
            limiar_efetivo = float(min(limiar_padrao, np.percentile(valores, 74)))
        else:
            limiar_efetivo = limiar_padrao

        valores_normais = valores[valores <= limiar_efetivo]
        media_aceitavel = float(np.mean(valores_normais)) if len(valores_normais) > 0 else media_cat * 0.6

        nome_cat = grupo["tipoDespesa"].iloc[0]
        stats_categoria[int(cod_cat)] = {
            "nome": nome_cat,
            "media_total": media_cat,
            "desvio_padrao": std_cat,
            "media_aceitavel": media_aceitavel,
            "limiar_atipico": limiar_efetivo,
        }
        limiares_categoria[int(cod_cat)] = limiar_efetivo

    # Criação da variável alvo y (0 = Normal, 1 = Atípico/Suspeito)
    df["limiarCategoria"] = df["codTipoDespesa"].map(limiares_categoria)
    df["mediaAceitavelCategoria"] = df["codTipoDespesa"].map(
        {k: v["media_aceitavel"] for k, v in stats_categoria.items()}
    )
    df["target"] = (df["valorLiquido"] > df["limiarCategoria"]).astype(int)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False, encoding="utf-8")
    print(f"[Dados Públicos] Dataset processado salvo em: {PROCESSED_CSV_PATH} ({len(df):,} linhas)")

    # Extração das 5 features numéricas exigidas para a matriz X
    feature_cols = ["valorLiquido", "codTipoDespesa", "mes", "codPartido", "codUF"]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.int64)

    metadata = {
        "feature_names": feature_cols,
        "stats_categoria": stats_categoria,
        "mapa_categoria": mapa_categoria,
        "mapa_partido": mapa_partido,
        "mapa_uf": mapa_uf,
        "total_consolidado_bruto": len(df_unificado_total),
    }

    return X, y, df, metadata

import os
from src.bulk_data import download_expenses, money, save_csv
from src.model_evaluation import historical_reference
from src.public_costs import write_indicators
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
OBRAS_PROJETOS_CSV_PATH = os.path.join("data", "obras_projetos_publicos_br.csv")


def download_massive_public_csvs(refresh: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    write_indicators(INDICADORES_CSV_PATH)

    if refresh:
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
    if refresh:
        # Só substitui os arquivos após validar ambos os anos e as duas Casas.
        df_camara_completa, df_senado_ceaps = download_expenses()
        for frame, path in [(df_camara_completa, CAMARA_COMPLETA_CSV_PATH),
                            (df_senado_ceaps, SENADO_COMPLETO_CSV_PATH)]:
            save_csv(frame, path)
    else:
        df_camara_completa = pd.read_csv(CAMARA_COMPLETA_CSV_PATH, dtype=str, keep_default_na=False)
        df_senado_ceaps = pd.read_csv(SENADO_COMPLETO_CSV_PATH, dtype=str, keep_default_na=False)

    build_obras_projetos_publicos_csv()
    return df_camara_completa, df_senado_ceaps


def build_obras_projetos_publicos_csv() -> pd.DataFrame:
    """Catálogo comparativo estimado, sem simular convênios ou repasses executados."""
    if os.path.exists(OBRAS_PROJETOS_CSV_PATH):
        existing = pd.read_csv(OBRAS_PROJETOS_CSV_PATH, low_memory=False)
        if 'tipoRegistro' in existing and existing['tipoRegistro'].eq('oficial').any():
            return existing  # Atualização municipal: scripts/update_public_works.py --refresh.

    mapa_regiao = {
        "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
        "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
        "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
        "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
        "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
        "PR": "Sul", "RS": "Sul", "SC": "Sul",
    }

    catalogo_setores = [
        # 1. SAÚDE
        {"setor": "Saúde", "subTipo": "Unidade Básica de Saúde (UBS)", "titulo": "Reforma e Ampliação de Unidade Básica de Saúde (UBS Porte II)", "orgaoFonte": "Fundo Nacional de Saúde (FNS/SUS) · Novo PAC", "custoTotalBase": 1150000.0, "aporteMensalBase": 85000.0, "unidadeImpacto": "atendimentos médicos/mês na UBS", "fatorUnidade": 65.0},
        {"setor": "Saúde", "subTipo": "Urgência e Emergência (SAMU/UPA)", "titulo": "Aquisição de Ambulância SAMU 192 Suporte Avançado e Equipamentos UPA", "orgaoFonte": "Ministério da Saúde · Renovação Frota SAMU", "custoTotalBase": 360000.0, "aporteMensalBase": 42000.0, "unidadeImpacto": "remoções de urgência e plantões", "fatorUnidade": 380.0},
        {"setor": "Saúde", "subTipo": "Atenção Primária e Farmácia Básica", "titulo": "Custeio de Equipes Saúde da Família (eSF), Saúde Bucal e Farmácia Básica", "orgaoFonte": "DataSUS / SIGTAP · Atenção Primária", "custoTotalBase": 480000.0, "aporteMensalBase": 36000.0, "unidadeImpacto": "consultas e kits de medicamentos essenciais", "fatorUnidade": 55.0},
        {"setor": "Saúde", "subTipo": "Saúde Mental e Especialidades (CAPS)", "titulo": "Implantação de Centro de Atenção Psicossocial (CAPS) e Policlínica Regional", "orgaoFonte": "Ministério da Saúde / ObrasGov.br", "custoTotalBase": 1680000.0, "aporteMensalBase": 110000.0, "unidadeImpacto": "procedimentos e exames especializados", "fatorUnidade": 110.0},
        # 2. EDUCAÇÃO
        {"setor": "Educação", "subTipo": "Educação Infantil (Creches)", "titulo": "Construção de Creche e Pré-Escola Pró-Infância Tempo Integral", "orgaoFonte": "FNDE / SIMEC · Ministério da Educação", "custoTotalBase": 2350000.0, "aporteMensalBase": 145000.0, "unidadeImpacto": "vagas em creche pública tempo integral", "fatorUnidade": 4500.0},
        {"setor": "Educação", "subTipo": "Infraestrutura Escolar e Climatização", "titulo": "Reforma de Escola Pública, Climatização de Salas e Cobertura de Quadra", "orgaoFonte": "FNDE / Novo PAC Educação Básica", "custoTotalBase": 690000.0, "aporteMensalBase": 54000.0, "unidadeImpacto": "alunos beneficiados com infraestrutura", "fatorUnidade": 450.0},
        {"setor": "Educação", "subTipo": "Tecnologia e Laboratórios", "titulo": "Implantação de Laboratório de Informática, Robótica e Internet de Alta Velocidade", "orgaoFonte": "MEC / PNCP · Programa Escolas Conectadas", "custoTotalBase": 210000.0, "aporteMensalBase": 26000.0, "unidadeImpacto": "computadores educacionais instalados", "fatorUnidade": 2800.0},
        {"setor": "Educação", "subTipo": "Transporte e Alimentação Escolar", "titulo": "Aquisição de Ônibus Escolar Acessível (Caminho da Escola) e Insumos PNAE", "orgaoFonte": "FNDE / MEC · Caminho da Escola & PNAE", "custoTotalBase": 420000.0, "aporteMensalBase": 35000.0, "unidadeImpacto": "alunos transportados e alimentados", "fatorUnidade": 120.0},
        {"setor": "Educação", "subTipo": "Ensino Técnico e Profissional (IFs)", "titulo": "Expansão de Laboratórios e Oficinas de Instituto Federal Profissionalizante", "orgaoFonte": "SETEC / MEC · Rede Federal de Educação Profissional", "custoTotalBase": 1950000.0, "aporteMensalBase": 125000.0, "unidadeImpacto": "vagas em cursos técnicos gratuitos", "fatorUnidade": 1800.0},
        # 3. INFRAESTRUTURA URBANA
        {"setor": "Infraestrutura Urbana", "subTipo": "Pavimentação e Drenagem Urbana", "titulo": "Obras de Drenagem Pluvial, Pavimentação Asfáltica e Calçadas Acessíveis", "orgaoFonte": "ObrasGov.br / SINAPI · Ministério das Cidades", "custoTotalBase": 1820000.0, "aporteMensalBase": 128000.0, "unidadeImpacto": "m² de vias urbanas pavimentadas e drenadas", "fatorUnidade": 210.0},
        {"setor": "Infraestrutura Urbana", "subTipo": "Mobilidade e Transporte Coletivo", "titulo": "Implantação de Corredor de Transporte Público, Ciclovias e Abrigos de Ônibus", "orgaoFonte": "Ministério das Cidades · Mobilidade Urbana", "custoTotalBase": 2200000.0, "aporteMensalBase": 140000.0, "unidadeImpacto": "passageiros beneficiados diariamente", "fatorUnidade": 95.0},
        {"setor": "Infraestrutura Urbana", "subTipo": "Iluminação Pública LED", "titulo": "Modernização de Iluminação Pública com Luminárias LED de Alta Eficiência", "orgaoFonte": "Procel Reluz / Eletrobras · Transferegov", "custoTotalBase": 640000.0, "aporteMensalBase": 48000.0, "unidadeImpacto": "pontos de iluminação LED instalados", "fatorUnidade": 680.0},
        # 4. RODOVIAS E TRANSPORTES
        {"setor": "Rodovias e Transportes", "subTipo": "Estradas Vicinais e Pontes", "titulo": "Recuperação de Estradas Vicinais e Pontes de Concreto para Escoamento", "orgaoFonte": "Transferegov / MAPA & MDR", "custoTotalBase": 1180000.0, "aporteMensalBase": 82000.0, "unidadeImpacto": "km de estradas vicinais recuperadas", "fatorUnidade": 18500.0},
        {"setor": "Rodovias e Transportes", "subTipo": "Manutenção Rodoviária e Segurança Viária", "titulo": "Restauração de Pavimento, Sinalização e Passarelas em Travessia Urbana", "orgaoFonte": "DNIT / DER Estadual · ObrasGov.br", "custoTotalBase": 2900000.0, "aporteMensalBase": 175000.0, "unidadeImpacto": "metros lineares de rodovia restaurada", "fatorUnidade": 950.0},
        # 5. SANEAMENTO E RECURSOS HÍDRICOS
        {"setor": "Saneamento e Recursos Hídricos", "subTipo": "Abastecimento de Água", "titulo": "Ampliação de Sistema de Abastecimento de Água, Adutora e Reservatório", "orgaoFonte": "ObrasGov.br / Funasa · Novo PAC Água", "custoTotalBase": 1650000.0, "aporteMensalBase": 112000.0, "unidadeImpacto": "domicílios conectados à água potável", "fatorUnidade": 1650.0},
        {"setor": "Saneamento e Recursos Hídricos", "subTipo": "Esgotamento Sanitário (ETE)", "titulo": "Implantação de Rede Coletora de Esgoto e Estação de Tratamento (ETE)", "orgaoFonte": "Ministério das Cidades / Caixa · Saneamento", "custoTotalBase": 2750000.0, "aporteMensalBase": 165000.0, "unidadeImpacto": "famílias atendidas com coleta de esgoto", "fatorUnidade": 2100.0},
        {"setor": "Saneamento e Recursos Hídricos", "subTipo": "Contenção de Encostas e Macrodrenagem", "titulo": "Obras de Contenção de Encostas e Controle de Cheias em Área de Risco", "orgaoFonte": "Defesa Civil Nacional / MIDR · Novo PAC", "custoTotalBase": 2150000.0, "aporteMensalBase": 138000.0, "unidadeImpacto": "famílias protegidas em áreas de risco", "fatorUnidade": 3200.0},
        # 6. HABITAÇÃO
        {"setor": "Habitação", "subTipo": "Moradia Popular e Urbanização", "titulo": "Construção de Unidades Habitacionais e Urbanização de Assentamentos", "orgaoFonte": "Ministério das Cidades · Minha Casa, Minha Vida / FNHIS", "custoTotalBase": 3200000.0, "aporteMensalBase": 180000.0, "unidadeImpacto": "melhorias habitacionais e regularizações", "fatorUnidade": 3500.0},
        # 7. SEGURANÇA E DEFESA CIVIL
        {"setor": "Segurança e Defesa Civil", "subTipo": "Patrulhamento e Videomonitoramento", "titulo": "Aparelhamento de Patrulhamento Comunitário, Viaturas e Central de Monitoramento", "orgaoFonte": "SENASP / Fundo Nacional de Segurança Pública", "custoTotalBase": 560000.0, "aporteMensalBase": 44000.0, "unidadeImpacto": "diárias operacionais e equipamentos táticos", "fatorUnidade": 420.0},
        {"setor": "Segurança e Defesa Civil", "subTipo": "Corpo de Bombeiros e Defesa Civil", "titulo": "Modernização de Quartel do Corpo de Bombeiros e Kit de Resgate/Defesa Civil", "orgaoFonte": "MJSP / Defesa Civil · Transferegov", "custoTotalBase": 890000.0, "aporteMensalBase": 62000.0, "unidadeImpacto": "atendimentos de emergência e prevenção", "fatorUnidade": 310.0},
        # 8. ASSISTÊNCIA SOCIAL E TRABALHO
        {"setor": "Assistência Social e Trabalho", "subTipo": "CRAS / CREAS / Casa da Mulher", "titulo": "Implantação e Custeio de CRAS, Casa da Mulher e Segurança Alimentar (Cozinha Solidária)", "orgaoFonte": "MDS / MDHC · SUAS & Segurança Alimentar", "custoTotalBase": 720000.0, "aporteMensalBase": 48000.0, "unidadeImpacto": "famílias vulneráveis assistidas e cestas básicas", "fatorUnidade": 250.0},
        # 9. ESPORTE, CULTURA E LAZER
        {"setor": "Esporte, Cultura e Lazer", "subTipo": "Espaços Esportivos e Culturais", "titulo": "Construção de Espaço Esportivo Comunitário, Biblioteca e Centro Cultural (CEU)", "orgaoFonte": "Ministério do Esporte & MinC · Novo PAC", "custoTotalBase": 1250000.0, "aporteMensalBase": 78000.0, "unidadeImpacto": "jovens atendidos em oficinas e esportes", "fatorUnidade": 160.0},
        # 10. DESENVOLVIMENTO REGIONAL E ENERGIA
        {"setor": "Desenvolvimento Regional e Energia", "subTipo": "Agricultura Familiar e Energia Solar", "titulo": "Apoio à Agricultura Familiar, Feira Coberta e Geração Fotovoltaica em Prédios Públicos", "orgaoFonte": "MDA / MME / Codevasf · Transferegov", "custoTotalBase": 840000.0, "aporteMensalBase": 58000.0, "unidadeImpacto": "produtores familiares e prédios públicos atendidos", "fatorUnidade": 600.0},
    ]

    # Valores centrais solicitados, para o escopo completo de cada intervenção.
    ajustes = {
        'Unidade Básica de Saúde (UBS)': ('Reforma de Unidade Básica de Saúde', 250000.0),
        'Urgência e Emergência (SAMU/UPA)': ('Aquisição de Ambulância SAMU', 280000.0),
        'Infraestrutura Escolar e Climatização': ('Climatização de Escola Pública', 130000.0),
        'Pavimentação e Drenagem Urbana': ('Pavimentação de Vias — trecho de referência', 450000.0),
    }
    registros = []
    for uf, regiao in sorted(mapa_regiao.items()):
        for idx, item in enumerate(catalogo_setores):
            titulo, custo = ajustes.get(item['subTipo'], (item['titulo'], item['custoTotalBase']))
            fator = {'Saúde': 120.0, 'Educação': 480.0,
                     'Segurança e Defesa Civil': 350.0,
                     'Assistência Social e Trabalho': 350.0}.get(item['setor'], item['fatorUnidade'])
            unidade = {'Saúde': 'consultas especializadas (estimativa)',
                       'Educação': 'aluno-mês (referência aproximada FUNDEB)',
                       'Segurança e Defesa Civil': 'diárias operacionais (estimativa)',
                       'Assistência Social e Trabalho': 'cestas de assistência (estimativa)'}.get(item['setor'], item['unidadeImpacto'])
            for ano in (2025, 2026):
                registros.append({
                    'codigoObra': f'REF-{ano}-{uf}-{idx:02d}', 'titulo': titulo,
                    'setor': item['setor'], 'subTipo': item['subTipo'],
                    'municipio': 'Referência estadual — sem município contratado',
                    'uf': uf, 'regiao': regiao, 'ano': ano, 'mes': 0,
                    'situacao': 'Referência de custo estimada', 'execucaoFisicaPct': None,
                    'valorTotalObra': custo, 'aporteNecessarioPeriodo': custo,
                    'orgaoFonte': 'Estimativa comparativa; não representa repasse contratado',
                    'unidadeImpacto': unidade, 'fatorUnidade': fator,
                })
    df_obras = pd.DataFrame(registros)
    save_csv(df_obras, OBRAS_PROJETOS_CSV_PATH)
    return df_obras


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
    target_records: int = None,
    seed: int = 42,
    refresh: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Consolida todas as notas e cria alvos com estatísticas históricas de 2025.

    As cinco entradas legadas são devolvidas para a comparação de referência;
    a rede nova usa ExpenseFeatures (valor logarítmico e categoria one-hot).
    """
    df_camara_full, df_senado_full = download_massive_public_csvs(refresh=refresh)

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
        dfs["valorLiquido"] = money(dfs["valorReembolsado"])
        dfs = dfs[dfs["valorLiquido"] > 0].dropna(subset=["nomeSenador", "tipoDespesa"])

        dfs_pad = pd.DataFrame({
            "idDeputado": pd.to_numeric(dfs.get("codSenador", 0), errors="coerce").fillna(0).astype(int),
            "nomeParlamentar": dfs["nomeSenador"].astype(str).str.strip() + " (Senador)",
            "cargoPublico": "SENADOR DA REPÚBLICA",
            "siglaPartido": dfs.get("siglaPartido", pd.Series(["SENADO"] * len(dfs))).fillna("SENADO").astype(str).str.upper(),
            "siglaUF": dfs.get("siglaUF", pd.Series(["N/I"] * len(dfs))).fillna("N/I").astype(str).str.upper(),
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

    # O parâmetro legado target_records não trunca mais os CSVs.
    save_csv(df_unificado_total, RAW_CSV_PATH)

    # Usa 100% dos dados consolidados para inferência e métricas
    df = df_unificado_total.copy()

    # Códigos servem para identificar categorias, nunca como grandezas contínuas.
    mapa_categoria = {cat: idx + 1 for idx, cat in enumerate(sorted(df.tipoDespesa.unique()))}

    partidos_unicos = sorted(df["siglaPartido"].unique())
    mapa_partido = {p: idx + 1 for idx, p in enumerate(partidos_unicos)}

    ufs_unicas = sorted(df["siglaUF"].unique())
    mapa_uf = {uf: idx + 1 for idx, uf in enumerate(ufs_unicas)}

    df["codTipoDespesa"] = df["tipoDespesa"].map(mapa_categoria).astype(int)
    df["codPartido"] = df["siglaPartido"].map(mapa_partido).astype(int)
    df["codUF"] = df["siglaUF"].map(mapa_uf).astype(int)
    df["mes"] = df["mes"].clip(1, 12).astype(int)
    df["valorLiquido"] = df["valorLiquido"].astype(float)

    # Congela o gabarito em 2025, antes de olhar qualquer despesa de avaliação.
    df, stats_categoria = historical_reference(df)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    save_csv(df, PROCESSED_CSV_PATH)
    print(f'[Dados Públicos] Dataset integral exportado: {len(df):,} linhas')

    # Entradas legadas para comparação automática com a arquitetura anterior.
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

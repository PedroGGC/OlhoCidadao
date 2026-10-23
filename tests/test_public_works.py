import json
import unittest

from src.public_works import normalize_projects, works_payload


class PublicWorksTests(unittest.TestCase):
    def test_locations_are_official_and_project_is_not_duplicated(self):
        projects = [{'id_projeto_investimento': '1.22-3', 'uf_principal': 'MA',
                     'desc_nome': 'Ponte intermunicipal', 'situacao': 'Paralisada',
                     'dt_cadastro': '2022-09-03', 'investimentos_previstos': [
                         {'vl_investimento_previsto': 300000}, {'vl_investimento_previsto': 200000}]}]
        cities = [{'id_projeto_investimento': '1.22-3', 'cod_ibge': 2111300, 'no_municipio': 'São Luís', 'sg_uf': 'MA'},
                  {'id_projeto_investimento': '1.22-3', 'cod_ibge': 2111300, 'no_municipio': 'São Luís', 'sg_uf': 'MA'},
                  {'id_projeto_investimento': '1.22-3', 'cod_ibge': 2111201, 'no_municipio': 'São José de Ribamar', 'sg_uf': 'MA'}]
        frame = normalize_projects(projects, cities)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].valorTotalObra, 500000)
        self.assertEqual(len(json.loads(frame.iloc[0].municipios)), 2)
        self.assertEqual(frame.iloc[0].ano, 2022)
        self.assertEqual(frame.iloc[0].situacao, 'Paralisada')
        self.assertIsNone(frame.iloc[0].execucaoFisicaPct)
        self.assertEqual(len(works_payload(frame)[0]['municipios']), 2)

    def test_missing_location_and_value_are_not_fabricated(self):
        frame = normalize_projects([{'id_projeto_investimento': '2', 'uf_principal': 'MA',
                                    'investimentos_previstos': [{'vl_investimento_previsto': 0}]}], [])
        record = works_payload(frame)[0]
        self.assertEqual(record['municipios'], [])
        self.assertEqual(record['municipio'], 'Município não informado na fonte')
        self.assertIsNone(record['valorTotalObra'])
        self.assertIsNone(record['aporteNecessarioPeriodo'])
        self.assertIn('/2', record['urlFonte'])

    def test_explicit_official_title_resolves_generic_capital_geometry(self):
        projects = [{'id_projeto_investimento': '1', 'uf_principal': 'MA',
                     'desc_nome': 'Construção de escola - Arari/MA'}]
        geometries = [
            {'id_projeto_investimento': '1', 'cod_ibge': 2111300, 'no_municipio': 'São Luís', 'sg_uf': 'MA'},
            {'id_projeto_investimento': 'other', 'cod_ibge': 2101004, 'no_municipio': 'Arari', 'sg_uf': 'MA'},
        ]
        record = works_payload(normalize_projects(projects, geometries))[0]
        self.assertEqual(record['municipios'][0]['nome'], 'Arari')
        self.assertEqual(record['municipiosGeometria'][0]['nome'], 'São Luís')
        self.assertTrue(record['divergenciaLocalizacao'])
        self.assertIn('Nome oficial', record['origemMunicipio'])


if __name__ == '__main__':
    unittest.main()

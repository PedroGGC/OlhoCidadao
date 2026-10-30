import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from main import balanced_training_indices
from src.bulk_data import CAMARA_COLUMNS, money, notes_absent_from, parse_camara, parse_senado
from src.nn_model import NnModel


class OfficialDataTests(unittest.TestCase):
    def test_currency_formats_and_invalid_value(self):
        values = money(pd.Series(['1.234,56', '1234.56', '1234,56', '0', '-10,50', 'invalid']))
        np.testing.assert_allclose(values[:5], [1234.56, 1234.56, 1234.56, 0, -10.5])
        self.assertTrue(np.isnan(values.iloc[5]))

    def test_bulk_keeps_documents_but_excludes_refunds_and_non_members(self):
        row = dict.fromkeys(CAMARA_COLUMNS, '')
        row.update(ideCadastro='123', txNomeParlamentar='José', sgUF='MA',
                   sgPartido='X', txtCNPJCPF='00123456000100', txtNumero='000123',
                   numAno='2026', numMes='1', vlrLiquido='1.234,56')
        frame = pd.DataFrame([row, dict(row, vlrLiquido='-50'), dict(row, vlrLiquido='0'),
                              dict(row, sgUF=''), dict(row, ideCadastro='')])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('Ano-2026.csv', frame.to_csv(index=False, sep=';').encode('utf-8-sig'))
        parsed = parse_camara(buffer.getvalue(), 2026)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed.iloc[0].txtCNPJCPF, '00123456000100')
        self.assertEqual(parsed.iloc[0].txtNumero, '000123')
        self.assertEqual(parsed.iloc[0].vlrLiquido, 1234.56)

    def senate_csv(self, name='JOSÉ DA SILVA'):
        return ('"ULTIMA ATUALIZACAO";"22/10/2026"\r\r\n'
                '"ANO";"MES";"SENADOR";"TIPO_DESPESA";"CNPJ_CPF";"FORNECEDOR";'
                '"DOCUMENTO";"DATA";"DETALHAMENTO";"VALOR_REEMBOLSADO";"COD_DOCUMENTO"\r\r\n'
                f'2026;1;{name};Combustível;001234;Fornecedor;0001;25/02/0202;;"1.234,56";999\r\r\n').encode('latin1')

    def test_senate_preserves_bad_date_without_assigning_df(self):
        registry = {'JOSE DA SILVA': {'codSenador': '123', 'siglaUF': 'MA', 'siglaPartido': 'N/I'}}
        parsed = parse_senado(self.senate_csv(), 2026, registry)
        self.assertEqual(parsed.iloc[0].siglaUF, 'MA')
        self.assertEqual(parsed.iloc[0]['data'], '25/02/0202')
        self.assertEqual(parsed.iloc[0].cpfCnpj, '001234')
        self.assertEqual(parsed.iloc[0].valorReembolsado, 1234.56)

    def test_unknown_senator_fails_instead_of_corrupting_state_totals(self):
        with self.assertRaisesRegex(ValueError, 'sem mandato'):
            parse_senado(self.senate_csv('NOME AUSENTE'), 2026, {})

    def test_training_balance_holdout_and_coverage(self):
        df = pd.DataFrame([{'target': label, 'siglaUF': uf, 'cargoPublico': house, 'ano': year}
                           for label in [0, 1] for uf in ['MA', 'SP', 'RR']
                           for house in ['CAMARA', 'SENADO'] for year in [2025, 2026]
                           for _ in range(300)])
        holdout = np.random.default_rng(7).choice(len(df), 1000, replace=False)
        candidates = np.setdiff1d(np.arange(len(df)), holdout)
        indices = balanced_training_indices(df, candidates)
        self.assertEqual(len(indices), 5000)
        self.assertEqual(len(set(indices)), 5000)
        self.assertFalse(set(indices) & set(holdout))
        self.assertEqual(df.loc[indices].target.value_counts().to_dict(), {0: 2500, 1: 2500})
        self.assertEqual(len(df.loc[indices].groupby(['target', 'siglaUF', 'cargoPublico', 'ano'])), 24)

    def test_new_notes_are_detected_as_a_multiset(self):
        row = {'origemDado': 'C', 'idDeputado': 1, 'numDocumento': '0001', 'dataDocumento': 'd',
               'tipoDespesa': 'X', 'cnpjCpfFornecedor': '1', 'ano': 2026, 'mes': 9, 'valorLiquido': 10.5}
        previous = pd.DataFrame([row, dict(row, mes=8)])
        current = pd.DataFrame([dict(row, mes=8), row, row, dict(row, valorLiquido=11.)])
        np.testing.assert_array_equal(notes_absent_from(previous, current), [False, False, True, True])

    def test_inference_covers_more_rows_than_fit(self):
        model = NnModel(np.zeros((4, 5)), np.array([0, 1, 0, 1]))
        labels, probabilities = model.predict(np.zeros((100, 5)))
        self.assertEqual(len(labels), 100)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(model.y_pred.sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()

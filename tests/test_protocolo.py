import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codigos'))
from dividir_dados import executar, montar_plano
from protocolo import CLASSES, VERSAO, conferir_modelo, ler_manifesto, sha256


class DivisaoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.origem = self.raiz / 'origem'
        for c in CLASSES:
            (self.origem / c).mkdir(parents=True)
            for i in range(40):
                (self.origem / c / f'{i}.png').write_bytes(f'{c}-{i}'.encode())

    def test_estratificacao_reprodutivel_e_fonte_preservada(self):
        plano = montar_plano(self.origem, por_imagem=True)
        self.assertEqual(plano, montar_plano(self.origem, por_imagem=True))
        self.assertEqual({s: sum(i['conjunto'] == s for i in plano['imagens'])
                          for s in ('train', 'val', 'test')},
                         dict(train=56, val=12, test=12))
        destino = self.raiz / 'nova'
        executar(self.origem, destino, plano)
        self.assertEqual(len(list(self.origem.rglob('*.png'))), 80)
        self.assertEqual(ler_manifesto(destino)[0], plano)
        with self.assertRaises(ValueError):
            executar(self.origem, destino, plano)
        extra = destino / 'test' / CLASSES[0] / 'extra.png'
        extra.write_bytes(b'extra')
        with self.assertRaises(ValueError):
            ler_manifesto(destino)
        extra.unlink()
        item = destino / plano['imagens'][0]['caminho']
        item.write_bytes(b'alterada')
        with self.assertRaises(ValueError):
            ler_manifesto(destino)

    def test_grupos_nao_cruzam_particoes(self):
        csv_path = self.raiz / 'grupos.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['arquivo', 'grupo'])
            for c in CLASSES:
                for i in range(40):
                    writer.writerow([f'{c}/{i}.png', f'paciente_{i // 2}'])
        plano = montar_plano(self.origem, grupos_csv=csv_path)
        conjuntos = {}
        for item in plano['imagens']:
            conjuntos.setdefault(item['grupo'], set()).add(item['conjunto'])
        self.assertTrue(all(len(s) == 1 for s in conjuntos.values()))
        executar(self.origem, self.raiz / 'grupos', plano)
        ler_manifesto(self.raiz / 'grupos')

    def test_duplicatas_e_mapa_incompleto_sao_recusados(self):
        dup = self.origem / CLASSES[1] / 'duplicata.png'
        dup.write_bytes((self.origem / CLASSES[0] / '0.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicado'):
            montar_plano(self.origem, por_imagem=True)
        dup.unlink()
        csv_path = self.raiz / 'incompleto.csv'
        csv_path.write_text('arquivo,grupo\nParasitized/0.png,P0\n')
        with self.assertRaisesRegex(ValueError, 'sem grupo'):
            montar_plano(self.origem, grupos_csv=csv_path)

    def test_modelo_antigo_ou_de_outra_divisao_e_recusado(self):
        p = self.raiz / 'melhor_modelo_vgg16.keras'
        p.write_bytes(b'pesos de teste')
        with self.assertRaises(FileNotFoundError):
            conferir_modelo(p, 'divisao-a')
        meta = dict(protocolo=VERSAO, classes=CLASSES,
                    manifesto_sha256='divisao-a', modelo_sha256=sha256(p))
        p.with_suffix('.json').write_text(json.dumps(meta))
        self.assertEqual(conferir_modelo(p, 'divisao-a'), meta)
        with self.assertRaises(ValueError):
            conferir_modelo(p, 'divisao-b')
        p.write_bytes(b'pesos trocados')
        with self.assertRaises(ValueError):
            conferir_modelo(p, 'divisao-a')


class MetricasTest(unittest.TestCase):
    def test_positivo_e_media_ponderada(self):
        from comparar_modelos import calcular_resultados
        from avaliacao import calcular_metricas
        # Positiva=0. VP=3,FN=1,FP=2,VN=4; acurácia=70, sens=75, esp=66.67.
        y = [0] * 4 + [1] * 6
        pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
        r = calcular_resultados(y, pred, CLASSES)
        self.assertAlmostEqual(r['Acurácia (%)'], 70)
        self.assertAlmostEqual(r['Recall Médio (%)'], 70)
        self.assertAlmostEqual(r['Sensibilidade (%)'], 75)
        self.assertAlmostEqual(r['Especificidade (%)'], 100 * 4 / 6)
        self.assertEqual(r['Falsos Negativos'], 1)
        self.assertEqual(r['Falsos Positivos'], 2)
        self.assertAlmostEqual(calcular_metricas(3, 1, 2, 4)['precisao'], 60)


if __name__ == '__main__':
    unittest.main()

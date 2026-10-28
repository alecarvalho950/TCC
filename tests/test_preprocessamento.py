"""Testes funcionais sem baixar ImageNet e sem medir desempenho preditivo."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codigos'))
import numpy as np
import tensorflow as tf
from treinamento import construir_modelo


class PreprocessamentoTest(unittest.TestCase):
    def test_entradas_das_bases_e_serializacao(self):
        rgb = np.broadcast_to(np.array([10., 20., 30.], dtype='float32'),
                              (1, 128, 128, 3)).copy()
        esperado = np.array([30 - 103.939, 20 - 116.779, 10 - 123.68])
        for nome in ('cnn_propria', 'vgg16', 'resnet50', 'efficientnet'):
            with self.subTest(modelo=nome), tempfile.TemporaryDirectory() as pasta:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(123)
                m = construir_modelo(nome, weights=None)
                if nome in ('vgg16', 'resnet50'):
                    camada = m.get_layer('preprocess_' + nome)
                    probe = tf.keras.Model(m.input, camada.output)
                    np.testing.assert_allclose(probe(rgb, training=False)[0, 0, 0],
                                               esperado, atol=1e-5)
                    self.assertFalse(m.get_layer(nome).trainable)
                elif nome == 'cnn_propria':
                    probe = tf.keras.Model(m.input, m.get_layer('escala_0_1').output)
                    np.testing.assert_allclose(probe(rgb, training=False)[0, 0, 0],
                                               rgb[0, 0, 0] / 255, atol=1e-6)
                else:
                    base = m.get_layer('efficientnetb0')
                    probe = tf.keras.Model(m.input, m.get_layer('augmentation').output)
                    np.testing.assert_array_equal(probe(rgb, training=False), rgb)
                    # A primeira normalização está DENTRO da EfficientNet.
                    self.assertTrue(any(isinstance(l, tf.keras.layers.Rescaling)
                                        for l in base.layers))
                    self.assertFalse(base.trainable)
                antes = m(rgb, training=False).numpy()
                caminho = Path(pasta) / 'modelo.keras'
                m.save(caminho)
                salvo = tf.keras.models.load_model(caminho, compile=False)
                np.testing.assert_allclose(salvo(rgb, training=False), antes, atol=1e-6)
                np.testing.assert_allclose(antes.sum(axis=1), [1.], atol=1e-6)


if __name__ == '__main__':
    unittest.main()

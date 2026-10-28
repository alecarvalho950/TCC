"""Pré-processamento ImageNet serializável no próprio modelo .keras."""
from tensorflow.keras import applications, layers, utils


@utils.register_keras_serializable(package='TCC')
class PreprocessamentoImageNet(layers.Layer):
    def __init__(self, arquitetura, **kwargs):
        super().__init__(**kwargs)
        if arquitetura not in ('vgg16', 'resnet50'):
            raise ValueError('Arquitetura não suportada.')
        self.arquitetura = arquitetura

    def call(self, inputs):
        if self.arquitetura == 'vgg16':
            return applications.vgg16.preprocess_input(inputs)
        return applications.resnet50.preprocess_input(inputs)

    def get_config(self):
        return {**super().get_config(), 'arquitetura': self.arquitetura}

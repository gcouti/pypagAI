from pypagai.models.base import TensorFlowModel


class BERT(TensorFlowModel):
    """
    Based on Google BERT model

    https://github.com/google-research/bert
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)


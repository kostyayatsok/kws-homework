import dataclasses
import torch
from typing import Tuple

@dataclasses.dataclass
class TaskConfig:
    keyword: str = 'sheila'  # We will use 1 key word -- 'sheila'
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 20
    n_mels: int = 40
    cnn_out_channels: int = 8
    kernel_size: Tuple[int, int] = (5, 20)
    stride: Tuple[int, int] = (2, 8)
    hidden_size: int = 64
    gru_num_layers: int = 2
    bidirectional: bool = False
    num_classes: int = 2
    sample_rate: int = 16000
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

configs = {
      'model'           : TaskConfig()
    , 'small_model_32'  : TaskConfig(hidden_size=32, num_epochs=100)
    , 'small_model_16'  : TaskConfig(hidden_size=16, num_epochs=100)
    , 'small_model_16x4'  : TaskConfig(hidden_size=16, cnn_out_channels=4, num_epochs=100)
}
import sys
import torch
from models.CRNN import CRNN
from typing import List
import torch.nn.functional as F

class StreamingCRNN(CRNN):
    def  __init__(self, config, window_length, step=1):
        super().__init__(config)
        
        self.window_len = window_length
        self.step = max(config.stride[1], step)
        self.kernel_size = self.config.kernel_size[1]
        self.reset()

    def forward(self, input: torch.tensor) -> List[float]:
        '''
        input:
          2d torch.tensor of shape (len, n_mels,)
        output:
          1d list of floats of shape (S,)
          where S = (L + len - kernel_size[1]) // stride[1] - (L - kernel_size[1]) // stride[1])
          where L -- sum of previous inputs lens
        '''
        outputs = []

        input = input.unsqueeze(dim=0).unsqueeze(dim=0)
        self.input_buffer = torch.cat((self.input_buffer, input), dim=-1)
        
        if self.input_buffer.size(-1) < self.kernel_size:
            print(f"Provide {self.kernel_size-self.input_buffer.size(-1)} more\
                 observations to have next prediction.", file=sys.stderr)
            return []
        for i in range(self.kernel_size, self.input_buffer.size(-1)+1, self.step):
            input_ = self.input_buffer[:,:,:,i-self.kernel_size:i]
            conv_output = self.conv(input_).transpose(-1, -2)
            gru_output, self.hidden = self.gru(conv_output, self.hidden)
            self.gru_output_buffer = torch.cat(
                (self.gru_output_buffer, gru_output), dim=1)[:,-self.window_len:]
            energy = self.attention(self.gru_output_buffer)
            alpha = torch.softmax(energy, dim=-2)
            contex_vector = (alpha * self.gru_output_buffer).sum(dim=-2)
            output = self.classifier(contex_vector)
            output = F.softmax(output, dim=-1)
            outputs.append(output[0,1].item())
        self.input_buffer = self.input_buffer[:,:,:,i-self.kernel_size+self.step:]
        return outputs
    def reset(self):
        '''
        call this function before passing new audio stream
        '''
        self.input_buffer = torch.zeros(
            (1, 1, self.config.n_mels, 0),
            device=self.config.device)
        self.gru_output_buffer = torch.zeros(
            (1, self.window_len, self.config.hidden_size),
            device=self.config.device) - 1e5
        self.hidden = None

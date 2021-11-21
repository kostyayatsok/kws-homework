import pandas as pd
from Trainer import Trainer
from config import TaskConfig, configs
from thop import profile
import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np

class Timer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self):
        self.t = time.time() - self.t

        if self.verbose:
            print(f"{self.name.capitalize()} | Elapsed time : {self.t:.2f}")

class Evaluator(Trainer):
    def __init__(self, GOLDEN_METRIC):
        super().__init__(GOLDEN_METRIC)
        self.results = pd.DataFrame()
        self.timer = Timer()

    def evaluate(self, model_name, model=None, original_model_name=None):
        if model is None:
            model = self.restore_model(model_name)
        model.to('cpu')
        self.results = self.results.append(
            {
                "Model" : model_name,
                "Metric" : self.validation(model, self.val_loader, self.melspec_val, 'cpu'),
                "MACS" : self.getMACS(model) if original_model_name is None else\
                     self.results.loc[self.results.Model == original_model_name, 'MACS'].iloc[0],
                "Memory (KB)" : self.getMemory(model_name),
                "Real Time (CPU)" : self.getRealTime(model, 100),
            }, ignore_index=True
        )
        self.add_compression_rate()
        self.add_speedup_rate()
        self.add_metrics_decay_rate()
        return self.results
    def getMACS(self, model):
        sample_input = torch.zeros(1, 1, TaskConfig.n_mels, 100)
        FLOPS, MACS = profile(model, sample_input, verbose=False)
        return MACS
    def getMemory(self, model_name):
        return os.path.getsize(f"{self.WEIGHTS_PATH}/{model_name}.pt") / 1024
    def getRealTime(self, model, n_reps):
        sample_input = torch.zeros(1, TaskConfig.n_mels, 100)
        self.timer.__enter__()
        for _ in range(n_reps):
            model(sample_input)
        self.timer.__exit__()
        return self.timer.t / n_reps
    def add_compression_rate(self):
        self.results['Compression rate'] = self.results['Memory (KB)'] / self.results.loc[0, 'Memory (KB)']
        self.results['Compression rate'] = self.results['Compression rate'].round(2)
    def add_speedup_rate(self):
        self.results['Speed up rate'] = self.results['MACS'] / self.results.loc[0, 'MACS']
        self.results['Speed up rate'] = self.results['Speed up rate'].round(2)
    def add_metrics_decay_rate(self):
        self.results['Metric decay rate'] = self.results['Metric'] / self.results.loc[0, 'Metric']
        self.results['Metric decay rate'] = self.results['Metric decay rate'].round(2)
    
    def visualize(self):
        z = self.results['Memory (KB)']
        y = self.results['MACS']
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)
        ax.scatter(z, y, marker='x')
        for i, txt in enumerate(self.results['Model']):
            ax.annotate(txt, (z[i]*1.05, y[i]*1.05), fontsize=7)
        plt.xlabel("Memory (KB)")
        plt.ylabel("MACS")
        plt.yscale('log')
        plt.xscale('log')
        plt.plot()
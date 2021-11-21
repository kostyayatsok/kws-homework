from collections import defaultdict
from data_utils.DataloadersFactory import buildDataloaders
from data_utils.LogMelspec import LogMelspec
from config import configs, TaskConfig
from IPython.display import clear_output
from matplotlib import pyplot as plt
from models.CRNN import CRNN
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, GOLD_METRIC=0):
        self.GOLD_METRIC = GOLD_METRIC
        self.train_loader, self.val_loader = buildDataloaders()
        self.melspec_train = LogMelspec(is_train=True, config=TaskConfig)
        self.melspec_val = LogMelspec(is_train=False, config=TaskConfig)
        self.WEIGHTS_PATH = "weights/"
        os.makedirs(self.WEIGHTS_PATH, exist_ok=True)
   
    # FA - true: 0, model: 1
    # FR - true: 1, model: 0
    def count_FA_FR(self, preds, labels):
        FA = torch.sum(preds[labels == 0])
        FR = torch.sum(labels[preds == 0])
        
        # torch.numel - returns total number of elements in tensor
        return FA.item() / torch.numel(preds), FR.item() / torch.numel(preds)

    def get_au_fa_fr(self, probs, labels):
        sorted_probs, _ = torch.sort(probs)
        sorted_probs = torch.cat((torch.Tensor([0]), sorted_probs, torch.Tensor([1])))
        labels = torch.cat(labels, dim=0)
            
        FAs, FRs = [], []
        for prob in sorted_probs:
            preds = (probs >= prob) * 1
            FA, FR = self.count_FA_FR(preds, labels)        
            FAs.append(FA)
            FRs.append(FR)
        # plt.plot(FAs, FRs)
        # plt.show()

        # ~ area under curve using trapezoidal rule
        return -np.trapz(FRs, x=FAs)


    def train_epoch(self, model, opt, loader, log_melspec, device, teacher=None):
        model.train()
        for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch).to(device)

            opt.zero_grad()

            # run model # with autocast():
            logits = model(batch)
            # we need probabilities so we use softmax & CE separately
            probs = F.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, labels)
            if teacher is not None:
                logits_teacher = teacher(batch)
                probs_teacher = F.softmax(logits_teacher/20, dim=-1)
                probs_teacher.detach()
                loss_teacher = -(probs*probs_teacher).sum(dim=-1).mean()
                loss += loss_teacher

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            opt.step()

            # logging
            argmax_probs = torch.argmax(probs, dim=-1)
            FA, FR = self.count_FA_FR(argmax_probs, labels)
            acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

        return acc


    @torch.no_grad()
    def validation(self, model, loader, log_melspec, device):
        model.eval()

        val_losses, accs, FAs, FRs = [], [], [], []
        all_probs, all_labels = [], []
        for i, (batch, labels) in tqdm(enumerate(loader)):
            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch).to(device)

            output = model(batch)
            # we need probabilities so we use softmax & CE separately
            probs = F.softmax(output, dim=-1)
            loss = F.cross_entropy(output, labels)

            # logging
            argmax_probs = torch.argmax(probs, dim=-1)
            all_probs.append(probs[:, 1].cpu())
            all_labels.append(labels.cpu())
            val_losses.append(loss.item())
            accs.append(
                torch.sum(argmax_probs == labels).item() /  # ???
                torch.numel(argmax_probs)
            )
            FA, FR = self.count_FA_FR(argmax_probs, labels)
            FAs.append(FA)
            FRs.append(FR)

        # area under FA/FR curve for whole loader
        au_fa_fr = self.get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
        return au_fa_fr

    def train(self, model_name, teacher_name=None):
        assert model_name in configs, f"Incorret model name. Choose one of {configs.keys()}"
        config = configs[model_name]
        model = CRNN(config).to(config.device)
        history = defaultdict(list)

        opt = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        if teacher_name is not None:
            teacher = CRNN(configs[teacher_name]).to(config.device)
            teacher.load_state_dict(torch.load(f"{self.WEIGHTS_PATH}/{teacher_name}.pt"))
        else:
            teacher = None

        for n in range(config.num_epochs):
            self.train_epoch(
                model, opt, self.train_loader, self.melspec_train, config.device, teacher)

            au_fa_fr = self.validation(
                model, self.val_loader, self.melspec_val, config.device)
            history['val_metric'].append(au_fa_fr)

            clear_output()
            plt.plot(history['val_metric'])
            plt.ylabel('Metric')
            plt.xlabel('Epoch')
            plt.grid()
            plt.show()

            print('END OF EPOCH', n)
            if history['val_metric'][-1] <= self.GOLD_METRIC:
                break
        self.save_model(model, model_name)
        return model, history

    def restore_model(self, model_name, model=None):
        if model is None:
            assert model_name in configs, f"Incorret model name. Choose one of {configs.keys()}"
            model = CRNN(configs[model_name])
        model.load_state_dict(torch.load(f"{self.WEIGHTS_PATH}/{model_name}.pt"))
        return model
    def save_model(self, model, model_name):
        torch.save(model.state_dict(), f"{self.WEIGHTS_PATH}/{model_name}.pt")

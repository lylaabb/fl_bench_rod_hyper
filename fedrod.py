from argparse import Namespace
from collections import OrderedDict, Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from fedavg import FedAvgClient
from src.utils.models import DecoupledModel
from src.utils.tools import Logger, count_labels, trainable_params

# # ********加入embedding
# def balanced_softmax_loss(
#         logits: torch.Tensor,
#         targets: torch.Tensor,
#         gamma: float,
#         label_counts: torch.Tensor,
# ):
#     logits = logits + (label_counts ** gamma).unsqueeze(0).expand(logits.shape).log()
#     loss = F.cross_entropy(logits, targets, reduction="mean")
#     return loss
#
#
# class FedRoDClient(FedAvgClient):
#     def __init__(self, hypernetwork: torch.nn.Module, encoder: torch.nn.Module = None, **commons):
#         self.eval_results = None
#         commons["model"] = FedRoDModel(
#             commons["model"], commons["args"].eval_per
#         )
#         super().__init__(**commons)
#
#         self.encoder = encoder.to(self.device) if encoder else None
#         self.hypernetwork = hypernetwork.to(self.device) if self.args.hyper else None
#         self.hyper_optimizer = None
#         self.first_time_selected = False  # 每轮开始时刷新
#
#         if self.args.hyper:
#             self.hyper_optimizer = torch.optim.SGD(
#                 self.hypernetwork.parameters(),
#                 lr=self.args.hyper_lr
#             )
#
#         # 标记 personalized 参数名
#         self.personal_params_name.extend(
#             [name for name, _ in self.model.named_parameters() if "personalized" in name]
#         )
#
#         # 统计每个客户端的标签分布
#         self.clients_label_counts = []
#         for indices in self.data_indices:
#             counter = Counter(np.array(self.dataset.targets)[indices["train"]])
#             self.clients_label_counts.append(
#                 torch.tensor(
#                     [counter.get(i, 0) for i in range(len(self.dataset.classes))],
#                     device=self.device
#                 )
#             )
#
#     def compute_client_embedding(self):
#         if self.encoder is not None:
#             self.encoder.eval()
#             features = []
#             with torch.no_grad():
#                 for x, _ in self.trainloader:
#                     print("Input:", x.shape)  # 应该是 [B, 3, 32, 32]
#                     x = x.to(self.device)
#                     x_embed = self.encoder(x)
#                     print("After encoder:", x_embed.shape)  # 应该是 [B, embed_dim]
#                     features.append(x_embed)
#             features = torch.cat(features, dim=0)
#             client_embed = features.mean(dim=0).detach()
#             client_embed = torch.nn.functional.normalize(client_embed, dim=0)
#             print("Client embedding:", client_embed.shape)  # 应该是 [embed_dim]
#             return client_embed
#         else:
#             label_dist = self.clients_label_counts[self.client_id]
#             return (label_dist / label_dist.sum()).detach()
#
#
#
#     def set_parameters(self, new_generic_parameters: OrderedDict[str, torch.Tensor]):
#         personal_parameters = self.personal_params_dict.get(
#             self.client_id, self.init_personal_params_dict
#         )
#         self.optimizer.load_state_dict(
#             self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
#         )
#         self.model.generic_model.load_state_dict(new_generic_parameters, strict=False)
#         self.model.load_state_dict(personal_parameters, strict=False)
#
#     def fit(self):
#         self.model.train()
#
#         # ---------- 超网络个性化初始化 ----------
#         # if self.args.hyper and self.first_time_selected:
#         if self.args.hyper:
#             self.hypernetwork.to(self.device)
#
#             client_embedding = self.compute_client_embedding()
#             print(
#                 f"[Client {self.client_id}] Embedding mean={client_embedding.mean().item():.4e}, std={client_embedding.std().item():.4e}, max={client_embedding.max().item():.4e}")
#             classifier_params = self.hypernetwork(client_embedding.unsqueeze(0)).squeeze(0)
#             print(
#                 f"[Client {self.client_id}] Classifier param mean={classifier_params.mean().item():.4e}, max={classifier_params.abs().max().item():.4e}")
#
#             weight_size = self.model.personalized_classifier.weight.numel()
#             self.model.personalized_classifier.weight.data = (
#                 classifier_params[:weight_size]
#                 .reshape(self.model.personalized_classifier.weight.shape)
#                 .detach().clone()
#             )
#             self.model.personalized_classifier.bias.data = (
#                 classifier_params[weight_size:]
#                 .reshape(self.model.personalized_classifier.bias.shape)
#                 .detach().clone()
#             )
#
#         # ---------- 本地训练 ----------
#         for _ in range(self.local_epoch):
#             for x, y in self.trainloader:
#                 if len(x) <= 1:
#                     continue
#                 x, y = x.to(self.device), y.to(self.device)
#                 logit_g, logit_p = self.model(x)
#
#                 loss_g = balanced_softmax_loss(
#                     logit_g,
#                     y,
#                     self.args.gamma,
#                     self.clients_label_counts[self.client_id]
#                 )
#                 loss_p = self.criterion(logit_p, y)
#                 loss = loss_g + loss_p
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#         # ---------- 超网络更新 ----------
#         # if self.args.hyper and self.first_time_selected:
#         if self.args.hyper:
#             trained_params = torch.cat([
#                 self.model.personalized_classifier.weight.data.flatten(),
#                 self.model.personalized_classifier.bias.data.flatten()
#             ])
#             hyper_loss = F.mse_loss(classifier_params, trained_params, reduction="sum")
#
#             self.hyper_optimizer.zero_grad()
#             hyper_loss.backward()
#             self.hyper_optimizer.step()
#
#             self.hypernetwork.cpu()  # 节省显存
#
#
#
#     def train(
#             self,
#             client_id: int,
#             local_epoch: int,
#             new_parameters: OrderedDict[str, torch.Tensor],
#             hyper_parameters: OrderedDict[str, torch.Tensor] = None,
#             return_diff: bool = False,
#             verbose: bool = False
#     ):
#         self.client_id = client_id
#         self.local_epoch = local_epoch
#         self.load_dataset()
#         self.eval_results = self.train_and_log(verbose)
#
#         self.set_parameters(new_parameters)
#         if self.args.hyper:
#             self.hypernetwork.load_state_dict(hyper_parameters, strict=False)
#
#         self.fit()
#
#         if return_diff:
#             delta = OrderedDict()
#             for (name, p0), p1 in zip(
#                     new_parameters.items(), trainable_params(self.model.generic_model)
#             ):
#                 delta[name] = p0 - p1
#
#             hyper_delta = None
#             if self.args.hyper:
#                 hyper_delta = OrderedDict()
#                 for (name, p0), p1 in zip(
#                         hyper_parameters.items(), trainable_params(self.hypernetwork)
#                 ):
#                     hyper_delta[name] = p0 - p1
#
#             return delta, hyper_delta, len(self.trainset), self.eval_results
#         else:
#             return (
#                 trainable_params(self.model.generic_model, detach=True),
#                 trainable_params(self.hypernetwork, detach=True)
#                 if self.args.hyper else None,
#                 len(self.trainset),
#                 self.eval_results
#             )
#
#     # Example Encoder (2-layer MLP)
#     class SimpleEncoder(torch.nn.Module):
#         def __init__(self, input_dim, embed_dim):
#             super().__init__()
#             self.encoder = torch.nn.Sequential(
#                 torch.nn.Flatten(),
#                 torch.nn.Linear(input_dim, 256),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(256, embed_dim)
#             )
#
#         def forward(self, x):
#             return self.encoder(x)

#**********最原始的
# class FedRoDClient(FedAvgClient):
#     def __init__(
#         self,
#         model: DecoupledModel,
#         hypernetwork: torch.nn.Module,
#         args: Namespace,
#         logger: Logger,
#         device: torch.device,
#     ):
#         super().__init__(FedRoDModel(model, args.eval_per), args, logger, device)
#         self.hypernetwork: torch.nn.Module = None
#         self.hyper_optimizer = None
#         if self.args.hyper:
#             self.hypernetwork = hypernetwork.to(self.device)
#             self.hyper_optimizer = torch.optim.SGD(
#                 trainable_params(self.hypernetwork), lr=self.args.hyper_lr
#             )
#         self.personal_params_name.extend(
#             [key for key, _ in self.model.named_parameters() if "personalized" in key]
#         )
#
#     def set_parameters(self, new_generic_parameters: OrderedDict[str, torch.Tensor]):
#         personal_parameters = self.personal_params_dict.get(
#             self.client_id, self.init_personal_params_dict
#         )
#         self.optimizer.load_state_dict(
#             self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
#         )
#         self.model.generic_model.load_state_dict(new_generic_parameters, strict=False)
#         self.model.load_state_dict(personal_parameters, strict=False)
#
#     def train(
#         self,
#         client_id: int,
#         local_epoch: int,
#         new_parameters: OrderedDict[str, torch.Tensor],
#         hyper_parameters: OrderedDict[str, torch.Tensor],
#         return_diff=False,
#         verbose=False,
#     ):
#         self.client_id = client_id
#         if self.args.hyper:
#             self.hypernetwork.load_state_dict(hyper_parameters, strict=False)
#         self.local_epoch = local_epoch
#         self.load_dataset()
#         self.set_parameters(new_parameters)
#         eval_results = self.train_and_log(verbose=verbose)
#
#         if return_diff:
#             delta = OrderedDict()
#             for (name, p0), p1 in zip(
#                 new_parameters.items(), trainable_params(self.model.generic_model)
#             ):
#                 delta[name] = p0 - p1
#
#             hyper_delta = None
#             if self.args.hyper:
#                 hyper_delta = OrderedDict()
#                 for (name, p0), p1 in zip(
#                     hyper_parameters.items(), trainable_params(self.hypernetwork)
#                 ):
#                     hyper_delta[name] = p0 - p1
#
#             return delta, hyper_delta, len(self.trainset), eval_results
#         else:
#             return (
#                 trainable_params(self.model.generic_model, detach=True),
#                 trainable_params(self.hypernetwork, detach=True),
#                 len(self.trainset),
#                 eval_results,
#             )
#
#     def fit(self):
#         label_counts = torch.tensor(
#             count_labels(self.dataset, self.trainset.indices), device=self.device
#         )
#         # if using hypernetwork for generating personalized classifier parameters and client is first-time selected
#         if self.args.hyper and self.client_id not in self.personal_params_dict:
#             label_distrib = label_counts / label_counts.sum()
#             classifier_params = self.hypernetwork(label_distrib)
#             clf_weight_numel = self.model.generic_model.classifier.weight.numel()
#             self.model.personalized_classifier.weight.data = (
#                 classifier_params[:clf_weight_numel]
#                 .reshape(self.model.personalized_classifier.weight.shape)
#                 .detach()
#                 .clone()
#             )
#             self.model.personalized_classifier.bias.data = (
#                 classifier_params[clf_weight_numel:]
#                 .reshape(self.model.personalized_classifier.bias.shape)
#                 .detach()
#                 .clone()
#             )
#
#         self.model.train()
#         for _ in range(self.local_epoch):
#             for x, y in self.trainloader:
#                 if len(x) <= 1:
#                     continue
#
#                 x, y = x.to(self.device), y.to(self.device)
#                 logit_g, logit_p = self.model(x)
#                 loss_g = balanced_softmax_loss(
#                     logit_g, y, self.args.gamma, label_counts
#                 )
#                 loss_p = self.criterion(logit_p, y)
#                 loss = loss_g + loss_p
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#         if self.args.hyper and self.client_id not in self.personal_params_dict:
#             # This part has no references on the FedRoD paper
#             trained_classifier_params = torch.cat(
#                 [
#                     torch.flatten(self.model.personalized_classifier.weight.data),
#                     torch.flatten(self.model.personalized_classifier.bias.data),
#                 ]
#             )
#             hyper_loss = F.mse_loss(
#                 classifier_params, trained_classifier_params, reduction="sum"
#             )
#             self.hyper_optimizer.zero_grad()
#             hyper_loss.backward()
#             self.hyper_optimizer.step()
# # ***********

#     def finetune(self):
#         self.model.train()
#         for _ in range(self.args.finetune_epoch):
#             for x, y in self.trainloader:
#                 if len(x) <= 1:
#                     continue
#
#                 x, y = x.to(self.device), y.to(self.device)
#                 if self.args.eval_per:
#                     _, logit_p = self.model(x)
#                     loss = self.criterion(logit_p, y)
#                 else:
#                     logit_g, _ = self.model(x)
#                     loss = self.criterion(logit_g, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 print(f"[Client {self.client_id}] Fine-tuned for {self.args.finetune_epoch} epochs")
#
#
# class FedRoDModel(DecoupledModel):
#     def __init__(self, generic_model: DecoupledModel, eval_per):
#         super().__init__()
#         self.generic_model = generic_model
#         self.personalized_classifier = deepcopy(generic_model.classifier)
#         self.eval_per = eval_per
#
#     def forward(self, x):
#         z = torch.relu(self.generic_model.get_final_features(x, detach=False))
#         logit_g = self.generic_model.classifier(z)
#         logit_p = self.personalized_classifier(z)
#         if self.training:
#             return logit_g, logit_p
#         else:
#             if self.eval_per:
#                 return logit_p
#             else:
#                 return logit_g
# *********


# ******data embedding
def balanced_softmax_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float,
        label_counts: torch.Tensor,
):
    logits = logits + (label_counts ** gamma).unsqueeze(0).expand(logits.shape).log()
    loss = F.cross_entropy(logits, targets, reduction="mean")
    return loss


class FedRoDClient(FedAvgClient):
    def __init__(self, hypernetwork: torch.nn.Module, encoder: torch.nn.Module = None, **commons):
        self.eval_results = None
        commons["model"] = FedRoDModel(
            commons["model"], commons["args"].eval_per
        )
        super().__init__(**commons)

        self.encoder = encoder.to(self.device) if encoder else None
        self.hypernetwork = hypernetwork.to(self.device) if self.args.hyper else None
        self.hyper_optimizer = None
        self.first_time_selected = False  # 每轮开始时刷新

        if self.args.hyper:
            self.hyper_optimizer = torch.optim.SGD(
                self.hypernetwork.parameters(),
                lr=self.args.hyper_lr
            )

        self.personal_params_name.extend(
            [name for name, _ in self.model.named_parameters() if "personalized" in name]
        )

        # 标签分布统计
        self.clients_label_counts = []
        for indices in self.data_indices:
            counter = Counter(np.array(self.dataset.targets)[indices["train"]])
            self.clients_label_counts.append(
                torch.tensor(
                    [counter.get(i, 0) for i in range(len(self.dataset.classes))],
                    device=self.device
                )
            )

    def compute_client_embedding(self):
        if self.encoder is not None:
            self.encoder.eval()
            features = []
            with torch.no_grad():
                for x, _ in self.trainloader:
                    x = x.to(self.device)
                    x_embed = self.encoder(x)
                    features.append(x_embed)
            features = torch.cat(features, dim=0)
            client_embed = features.mean(dim=0).detach()
            client_embed = torch.nn.functional.normalize(client_embed, dim=0)
            return client_embed
        else:
            label_dist = self.clients_label_counts[self.client_id]
            return (label_dist / label_dist.sum()).detach()

    def set_parameters(self, new_generic_parameters: OrderedDict[str, torch.Tensor]):
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.generic_model.load_state_dict(new_generic_parameters, strict=False)
        self.model.load_state_dict(personal_parameters, strict=False)

    def fit(self):
        self.model.train()

        if self.args.hyper:
            self.hypernetwork.to(self.device)
            client_embedding = self.compute_client_embedding()
            classifier_params = self.hypernetwork(client_embedding.unsqueeze(0)).squeeze(0)

            weight_size = self.model.personalized_classifier.weight.numel()
            self.model.personalized_classifier.weight.data = (
                classifier_params[:weight_size].reshape(self.model.personalized_classifier.weight.shape).detach().clone()
            )
            self.model.personalized_classifier.bias.data = (
                classifier_params[weight_size:].reshape(self.model.personalized_classifier.bias.shape).detach().clone()
            )

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                logit_g, logit_p = self.model(x)

                loss_g = balanced_softmax_loss(
                    logit_g,
                    y,
                    self.args.gamma,
                    self.clients_label_counts[self.client_id]
                )
                loss_p = self.criterion(logit_p, y)
                loss = loss_g + loss_p

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.args.hyper:
            trained_params = torch.cat([
                self.model.personalized_classifier.weight.data.flatten(),
                self.model.personalized_classifier.bias.data.flatten()
            ])
            hyper_loss = F.mse_loss(classifier_params, trained_params, reduction="sum")
            self.hyper_optimizer.zero_grad()
            hyper_loss.backward()
            self.hyper_optimizer.step()
            self.hypernetwork.cpu()

    def train(self, client_id, local_epoch, new_parameters, hyper_parameters=None, return_diff=False, verbose=False):
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.eval_results = self.train_and_log(verbose)

        self.set_parameters(new_parameters)
        if self.args.hyper:
            self.hypernetwork.load_state_dict(hyper_parameters, strict=False)

        self.fit()

        if return_diff:
            delta = OrderedDict({k: new_parameters[k] - p for k, p in self.model.generic_model.state_dict().items()})
            hyper_delta = None
            if self.args.hyper:
                hyper_delta = OrderedDict({k: hyper_parameters[k] - p for k, p in self.hypernetwork.state_dict().items()})
            return delta, hyper_delta, len(self.trainset), self.eval_results
        else:
            return (
                trainable_params(self.model.generic_model, detach=True),
                trainable_params(self.hypernetwork, detach=True) if self.args.hyper else None,
                len(self.trainset),
                self.eval_results
            )

    class SimpleEncoder(torch.nn.Module):
        def __init__(self, input_dim, embed_dim):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, embed_dim)
            )

        def forward(self, x):
            return self.encoder(x)

    def finetune(self):
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                _, logit_p = self.model(x) if self.args.eval_per else (self.model(x), None)
                loss = self.criterion(logit_p, y) if self.args.eval_per else self.criterion(logit_g, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class FedRoDModel(DecoupledModel):
    def __init__(self, generic_model: DecoupledModel, eval_per):
        super().__init__()
        self.generic_model = generic_model
        self.personalized_classifier = deepcopy(generic_model.classifier)
        self.eval_per = eval_per

    def forward(self, x):
        z = torch.relu(self.generic_model.get_final_features(x, detach=False))
        logit_g = self.generic_model.classifier(z)
        logit_p = self.personalized_classifier(z)
        return (logit_g, logit_p) if self.training else (logit_p if self.eval_per else logit_g)



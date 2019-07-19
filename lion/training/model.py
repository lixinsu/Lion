#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy

import torch
from tqdm import tqdm
from loguru import logger
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from lion.common.utils import AverageMeter
from lion.training.optimizers import BertAdam
from lion.models import get_model_class


class MatchingModel:
    def __init__(self, params, state_dict=None):
        self.params = params
        if params.network == 'bert':
            self.network = get_model_class(params.network).from_pretrained(params.model_dir, params.classes)
        else:
            self.network = get_model_class(params.network)(params)
        params.use_cuda = params.use_cuda if torch.cuda.is_available() else False
        if params.use_cuda:
            self.network.cuda()
        self.init_optimizer()
        if state_dict is not None:
            self.network.load_state_dict(state_dict)
        if params.embedding_file and state_dict is None:
            self.load_embedding(params.word_dict, params.embedding_file)

    def init_optimizer(self):
        """Initialize an optimizer for the free parameters of the network. """
        if self.params.fix_embeddings:
            for p in self.network.word_embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.params.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.params.learning_rate,
                                       momentum=self.params.momentum,
                                       weight_decay=self.params.weight_decay)
        elif self.params.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.params.weight_decay)
        elif self.params.optimizer == 'bert-adam':
            list_param_optimizer = list(self.network.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in list_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in list_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.params.learning_rate,
                                      warmup=self.params.warmup_proportion,
                                      t_total=self.params.num_train_optimization_steps)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.params.optimizer)

    def load_embedding(self, words, embedding_file):
        words = {w for w in words if w in self.params.word_dict}
        embedding = self.network.word_embedding.weight.data
        with open(embedding_file) as f:
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.params.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    embedding[self.params.word_dict[w]].copy_(vec)

    def update(self, ex):
        self.network.train()
        if self.params.use_cuda:
            for k in ex.keys():
                if k != 'ids':
                    ex[k] = ex[k].cuda()
        logits = self.network(ex)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.params.classes), ex['labels'].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                       self.params.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train_epoch(self, data_loader):
        loss_meter = AverageMeter()
        for ex in tqdm(data_loader):
            loss_val = self.update(ex)
            loss_meter.update(loss_val)
        return loss_meter.avg

    def predict(self, ex):
        # Eval mode
        self.network.eval()
        # Transfer to GPU
        if self.params.use_cuda:
            for k in ex.keys():
                if k != 'ids':
                    ex[k] = ex[k].cuda()
        # Run forward
        with torch.no_grad():
            logits = self.network(ex)
        pred = torch.argmax(logits, dim=1)
        return pred.tolist(), logits.tolist()

    def predict_epoch(self, data_loader):
        rv = {}
        for ex in data_loader:
            ids = ex['ids']
            preds, _ = self.predict(ex)
            for id_, pred_ in zip(ids, preds):
                rv[id_] = pred_
        return rv

    def evaluate_epoch(self, data_loader):
        all_pred = []
        all_gt = []
        all_proba = []
        for ex in data_loader:
            ids = ex['ids']
            preds, proba = self.predict(ex)
            all_pred.extend(preds)
            all_proba.extend(proba)
            gts = ex['labels'].tolist()
            all_gt.extend(gts)
        c = sum(np.array(all_gt)==np.array(all_pred))
        n = len(all_gt)
        logger.info('{}/{} = {}'.format(c, n, c/n))
        return {'acc': c/n}

    def save(self, filename):
        if self.params.parallel:
            network = self.network.module

        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            'state_dict': state_dict,
            'args': self.params,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        state_dict = saved_params['state_dict']
        params = saved_params['args']
        return MatchingModel(params, state_dict), params


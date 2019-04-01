#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from lion.common.utils import AverageMeter
from lion.models import get_model_class


logger = logging.getLogger(__name__)


class MatchingModel:
    def __init__(self, args, state_dict=None):
        self.args = args
        self.network = get_model_class(args.network)(args)
        args.use_cuda = args.use_cuda if torch.cuda.is_available() else False
        if args.use_cuda:
            self.network.cuda()
        self.init_optimizer()
        if state_dict is not None:
            self.network.load_state_dict(state_dict)
        if args.embedding_file and state_dict is None:
            self.load_embedding(args.word_dict, args.embedding_file)

    def init_optimizer(self):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    def load_embedding(self, words, embedding_file):
        words = {w for w in words if w in self.args.word_dict}
        embedding = self.network.word_embedding.weight.data
        with open(embedding_file) as f:
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.args.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    embedding[self.args.word_dict[w]].copy_(vec)

    def update(self, ex):
        self.network.train()
        if self.args.use_cuda:
            for k in ex.keys():
                if k != 'ids':
                    ex[k] = ex[k].cuda()
        logproba = self.network(ex)
        loss = F.nll_loss(logproba, ex['labels'])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.args.grad_clipping)
        self.optimizer.step()

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
        if self.args.use_cuda:
            for k in ex.keys():
                if k != 'ids':
                    ex[k] = ex[k].cuda()
        # Run forward
        with torch.no_grad():
            logproba = self.network(ex)
        proba = torch.exp(logproba)
        pred = torch.argmax(proba, dim=1)
        return pred.tolist(), proba.tolist()

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

        c = sum(np.array(all_gt) == np.array(all_pred) )
        n = len(all_gt)
        print('{}/{} = {}'.format(c, n, c/n))
        #logger.info('accuracy_score {}'.format(accuracy_score(all_gt, all_pred)))
        #logger.info('f1_score {}'.format(f1_score(all_gt, all_pred)))

    def save(self, filename):
        if self.parallel:
            network = self.network.module

        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            'state_dict': state_dict,
            'args': self.args,
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
        args = saved_params['args']
        return MatchingModel(args, state_dict)


    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)



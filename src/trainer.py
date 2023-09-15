import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import scipy
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from sklearn.metrics import confusion_matrix
from seqeval.metrics import f1_score # 序列标注评估工具
from transformers import AutoTokenizer

from src.dataloader import *
from src.utils import *

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

class BaseTrainer(object):
    def __init__(self, params, model, label_list):
        # parameters
        self.params = params # 配置
        self.model = model 
        self.label_list = label_list
        
        # training
        self.lr = float(params.lr)
        self.mu = 0.9
        self.weight_decay = 5e-4

    
    def batch_forward(self, inputs):    
        # Compute features
        self.inputs = inputs # # (bsz, seq_len)
        self.features = self.model.forward_encoder(inputs) #  (bsz, seq_len, hidden_dim)
        # Compute logits 常规logits
        self.logits = self.model.forward_classifier(self.features)   # (bsz, seq_len, output_dim) 


    def batch_loss(self, labels):
        '''
            Cross-Entropy Loss
        '''
        self.loss = 0
        assert self.logits!=None, "logits is none!"

        # classification loss
        ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]), 
                                labels.flatten().long()) # bs*seq_len, out_dim 默认自动忽略-100 label （pad、cls、sep、第二子词对应的索引）
        self.loss = ce_loss
        return ce_loss.item() 


    def _update_running_stats(self, labels_down, features, prototypes, count_features):
        cl_present = torch.unique(input=labels_down)
   
        cl_present=torch.where((cl_present < self.old_classes) & (cl_present != pad_token_label_id), cl_present, pad_token_label_id)
        cl_present = torch.unique(input=cl_present)
      
        if cl_present[0] == pad_token_label_id:
            cl_present = cl_present[1:]

        features_local_mean = torch.zeros([self.old_classes, self.params.hidden_dim]).cuda()

        for cl in cl_present:
            features_cl = features[(labels_down == cl).expand(-1, -1, features.shape[-1])].view(features.shape[-1], -1).detach()
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                            prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features

    def update_prototypes(self, train_loader):
  
        prototypes = torch.zeros([self.old_classes, self.params.hidden_dim])
        prototypes.requires_grad = False
        prototypes = prototypes.cuda()
        count_features = torch.zeros([self.old_classes], dtype=torch.long)
        count_features.requires_grad = False
        count_features = count_features.cuda()

        for X, labels in train_loader:
            X = X.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                self.refer_model.eval()
                refer_features = self.refer_model.forward_encoder(X) # (bsz,seq_len,hidden_dim)
                refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)

            probas = torch.softmax(refer_logits, dim=-1)
            _, pseudo_probas = probas.max(dim=-1)
      
            mask_bg = labels == 0
            labels[mask_bg] = pseudo_probas[mask_bg]
     
            prototypes, count_features = self._update_running_stats(labels.unsqueeze(-1).long(), refer_features, prototypes,
                                                                    count_features)

        return prototypes, count_features

    def get_prototype_weight(self, feat):
        feat_proto_distance = self.feat_prototype_distance(feat)
        weight = F.softmax(-feat_proto_distance * self.params.proto_temperature, dim=-1)
        return weight

    def feat_prototype_distance(self, feat):
        bs, seq_len, _ = feat.shape
        feat_proto_distance = -torch.ones((bs, seq_len, self.old_classes)).to(feat.device)
        for i in range(self.old_classes):
            feat_proto_distance[:, :, i] = torch.norm(self.prototypes[i].reshape(1,1,-1).expand(bs,seq_len,-1) - feat, 2, dim=-1,)
        return feat_proto_distance

    
    def before_prototype(self, train_loader):
        self.prototypes, self.count_features = self.update_prototypes(
                train_loader)


    def reg_pesudo_label(self, output):

        output = torch.softmax(output, dim=-1) # (bsz, seq_len, all_dims)
        loss = -(output * torch.log(output)).mean(dim=-1)

        return loss

    def batch_loss_rdp(self, labels):
        '''
            Cross-Entropy Loss (Pseudo label) + Soft_sharp Loss (Soft label distill)
        '''

        original_labels = labels.clone()
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim # old model 输出维度

            
        # Check input
        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  


        with torch.no_grad():
            self.refer_model.eval()
            refer_features = self.refer_model.forward_encoder(self.inputs)
            refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"
     
        
        mask_background = (labels < self.old_classes) & (labels != pad_token_label_id) # 0 的位置

  
        # 原型伪标签
        probs = torch.softmax(refer_logits, dim=-1) # (bs, seq_len, refer_dims)   refer_dims==old_classes

        weights = self.get_prototype_weight(refer_features)   # (bs, seq_len, old_classes)

        rectified = weights * probs
        rectified = rectified / rectified.sum(-1, keepdim=True)
        _, pseudo_labels_rec = rectified.max(dim=-1)


        labels[mask_background] = pseudo_labels_rec[mask_background]
        


        loss = nn.CrossEntropyLoss(reduction='none')(self.logits.permute(0,2,1), labels) # 0 新类 旧类伪标签 -100(计算的loss为0)    (bsz,seq_len)
   

        ignore_mask = (labels!=pad_token_label_id)  
        if torch.sum(ignore_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).cuda()
        else:
            ce_loss = loss[ignore_mask].mean()  # scalar


        old_outputs = torch.sigmoid(refer_logits) # (bsz, seq_len, refer_dims)
        old_classes = self.old_classes
        loss_soft_label = BCEWithLogitsLossWithIgnoreIndexSoftLabel(reduction='none', ignore_index=pad_token_label_id)(self.logits, old_outputs, old_classes, original_labels)
        Regularizer_soft = self.reg_pesudo_label(self.logits)

        loss_soft_label = loss_soft_label.mean()
        Regularizer_soft = Regularizer_soft.mean()

        # distill logits loss
        distill_mask = torch.logical_and(original_labels==0, original_labels!=pad_token_label_id) # 选出other class token

        if torch.sum(distill_mask.float())==0:
            distill_logits_loss = torch.tensor(0., requires_grad=True).cuda()
        else:   
            old_logits_score = F.log_softmax(
                                self.logits[distill_mask]/self.params.temperature,
                                dim=-1)[:,:refer_dims].view(-1, refer_dims) #(bsz*seq_len(select out), refer_dims)
    
            ref_old_logits_score = F.softmax(
                                refer_logits[distill_mask]/self.params.ref_temperature, 
                                dim=-1).view(-1, refer_dims)

            distill_logits_loss = nn.KLDivLoss(reduction='batchmean')(old_logits_score, ref_old_logits_score)



        distill_loss = self.params.soft_param * loss_soft_label + self.params.regular_param * Regularizer_soft + self.params.distill_logits_weight*distill_logits_loss

        self.loss = ce_loss + distill_loss

        return ce_loss.item(), distill_loss.item()

            
    def batch_backward(self):
        self.model.train()
        self.optimizer.zero_grad()        
        self.loss.backward()
        self.optimizer.step()
        
        return self.loss.item()

    def evaluate(self, dataloader, each_class=False, entity_order=[], is_plot_hist=False, is_save_txt=False, is_plot_cm=False):
        with torch.no_grad():
            self.model.eval()

            y_list = []
            x_list = []
            logits_list = []

            for x, y in dataloader: 
                x, y = x.cuda(), y.cuda()
                self.batch_forward(x)
                _logits = self.logits.view(-1, self.logits.shape[-1]).detach().cpu()
                logits_list.append(_logits)
                x = x.view(x.size(0)*x.size(1)).detach().cpu() # bs*seq_len
                x_list.append(x) 
                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)

            
            y_list = torch.cat(y_list)
            x_list = torch.cat(x_list)
            logits_list = torch.cat(logits_list)   
            pred_list = torch.argmax(logits_list, dim=-1)

            ### Plot the (logits) prob distribution for each class
            if is_plot_hist: # False
                plot_prob_hist_each_class(deepcopy(y_list), 
                                        deepcopy(logits_list),
                                        ignore_label_lst=[
                                            self.label_list.index('O'),
                                            pad_token_label_id
                                        ])

          

            ### for confusion matrix visualization
            if is_plot_cm: # False
                plot_confusion_matrix(deepcopy(pred_list),
                                deepcopy(y_list), 
                                label_list=self.label_list,
                                pad_token_label_id=pad_token_label_id)

            ### calcuate f1 score
            pred_line = []
            gold_line = []
            for pred_index, gold_index in zip(pred_list, y_list):
                gold_index = int(gold_index)
                if gold_index != pad_token_label_id: # !=-100
                    pred_token = self.label_list[pred_index] # label索引转label
                    gold_token = self.label_list[gold_index]
                    # lines.append("w" + " " + pred_token + " " + gold_token)
                    pred_line.append(pred_token) 
                    gold_line.append(gold_token) 

            # Check whether the label set are the same,
            # ensure that the predict label set is the subset of the gold label set
            gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
            if set(gold_label_set)!=set(pred_label_set):
                O_label_set = []
                for e in pred_label_set:
                    if e not in gold_label_set:
                        O_label_set.append(e)
                if len(O_label_set)>0:
                    # map the predicted labels which are not seen in gold label set to 'O'
                    for i, pred in enumerate(pred_line):
                        if pred in O_label_set:
                            pred_line[i] = 'O'

            self.model.train()

            # compute overall f1 score
            # micro f1 (default)
            f1 = f1_score([gold_line], [pred_line])*100
            # macro f1 (average of each class f1)
            ma_f1 = f1_score([gold_line], [pred_line], average='macro')*100
            if not each_class: # 不打印每个类别的f1
                return f1, ma_f1

            # compute f1 score for each class
            f1_list = f1_score([gold_line], [pred_line], average=None)
            f1_list = list(np.array(f1_list)*100)
            gold_entity_set = set()
            for l in gold_label_set:
                if 'B-' in l or 'I-' in l or 'E-' in l or 'S-' in l:
                    gold_entity_set.add(l[2:])
            gold_entity_list = sorted(list(gold_entity_set))
            f1_score_dict = dict()
            for e, s in zip(gold_entity_list,f1_list):
                f1_score_dict[e] = round(s,2)
            # using the default order for f1_score_dict
            if entity_order==[]:
                return f1, ma_f1, f1_score_dict
            # using the pre-defined order for f1_score_dict
            assert set(entity_order)==set(gold_entity_list),\
                "gold_entity_list and entity_order has different entity set!"
            ordered_f1_score_dict = dict()
            for e in entity_order:
                ordered_f1_score_dict[e] = f1_score_dict[e]
            return f1, ma_f1, ordered_f1_score_dict

    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))
        torch.save({
            "hidden_dim": self.model.hidden_dim,
            "output_dim": self.model.output_dim,
            "encoder": self.model.encoder.state_dict(),
            "classifier": self.model.classifier
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path=''):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        ckpt = torch.load(load_path)

        self.model.hidden_dim = ckpt['hidden_dim']
        self.model.output_dim = ckpt['output_dim']
        self.model.encoder.load_state_dict(ckpt['encoder'])
        self.model.classifier = ckpt['classifier']
        logger.info("Model has been load from %s" % load_path)

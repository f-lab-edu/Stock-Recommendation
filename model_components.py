import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import random
from deepctr_torch.inputs import SparseFeat, DenseFeat, build_input_features, create_embedding_matrix,embedding_lookup
from deepctr_torch.models.basemodel import BaseModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FM(nn.Module):
    """
    각 feature간 2차 상호작용 계산
    """
    def forward(self, embed_tensor: torch.Tensor) -> torch.Tensor:
        square_summed = torch.sum(embed_tensor, dim = 1, keepdim = True)**2
        sum_of_square = torch.sum(embed_tensor**2, dim = 1, keepdim= True)
        interaction_part = 0.5 * (square_summed - sum_of_square)
        return interaction_part.sum(dim=2, keepdim=False)

## 수정전 코드
# class DNN(nn.Module):
#     """
#     a. feature간 고차원의 상호작용을 학습
#     b. Linear  -> BN -> ReLU -> Dropout
#     """
#     def __init__(self, input_dim: int, hidden_units=(256,128,64), activation = "relu", dropout_rate = 0.0, use_bn = False):
#         super().__init__()
#         layers = []
#         in_dim = input_dim
#         activation_map = {"relu" : nn.ReLU, "prelu" : nn.PReLU, "sigmoid" : nn.Sigmoid}

#         for units in hidden_units:
#             layers.append(nn.Linear(in_dim, units))
#             if use_bn:
#                 layers.append(nn.BatchNorm1d(units))
#             layers.append(activation_map[activation.lower()]()) 
#             if dropout_rate > 0:
#                 layers.append(nn.Dropout(dropout_rate))
#             in_dim = units
#         # nn.Sequential 로 묶어 한 번에 forward
#         self.dnn = nn.Sequential(*layers)

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         return self.dnn(x)

class DNN(nn.Module):
    """
    a. feature간 고차원의 상호작용을 학습
       Linear -> BN -> ReLU -> Dropout
    """
    def __init__(self, input_dim: int, hidden_units=(256,128,64), activation = "relu", dropout_rate = 0.0, use_bn = False):
        super().__init__()
        self.use_bn = use_bn
        in_dim = input_dim
        activation_map = {"relu" : nn.ReLU, "prelu" : nn.PReLU, "sigmoid" : nn.Sigmoid}
        self.activation = activation_map[activation.lower()]()
        self.dropout = nn.Dropout(dropout_rate)
        self.linears = nn.ModuleList()
        self.bn = nn.ModuleList() if use_bn else None

        for units in hidden_units:
            self.linears.append(nn.Linear(in_dim, units))
            if use_bn:
                self.bn.append(nn.BatchNorm1d(units))
            in_dim = units # 다음 루프에서 사용할 입력 크기 업데이트

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = x
        for i, linear in enumerate(self.linears):
            output = linear(output)
            if self.use_bn:
                output = self.bn[i](output)
            output = self.activation(output)
            output = self.dropout(output)
        return output


class LinearPart(nn.Module):
    """
    각 feature간 1차항 선형 가중합 계산
    """
    def __init__(self, feature_columns, device = "cpu"):
        super().__init__()
        self.sparse_linears = nn.ModuleDict({fc.name: nn.Embedding(fc.vocabulary_size,1) for fc in feature_columns if isinstance(fc, SparseFeat)})
        dense_dim = sum(fc.dimension for fc in feature_columns if isinstance(fc, DenseFeat))
        self.dense_linear = nn.Linear(dense_dim, 1, bias = False) if dense_dim else None
        self.device = device

    def forward(self, X, feature_index, dense_col_idx) -> torch.Tensor:
        sparse_logits = [self.sparse_linears[name](X[:, feature_index[name][0]].long()) for name in self.sparse_linears]
        sparse_logit = torch.sum(torch.cat(sparse_logits, dim =1), dim =1, keepdim = True) if sparse_logits else 0.0
        if self.dense_linear:
            dense_logit = self.dense_linear(X[:, dense_col_idx].float())
        else:
            dense_logit = torch.zeros(X.size(0), 1).to(self.device)
        return sparse_logit + dense_logit 

class DeepFM(BaseModel):
    def __init__(self, feature_columns, embed_dim = 8, hidden_units=(256, 128, 64), 
                 l2_reg_linear = 1e-5, l2_reg_embedding = 1e-5, l2_reg_dnn = 0, device = "cpu"):
        super(DeepFM, self).__init__(feature_columns, feature_columns, l2_reg_linear = l2_reg_linear, l2_reg_embedding = l2_reg_embedding,
                                     device = device)
        self.device = device
        self.feature_columns = feature_columns

        # 각 피쳐가 입력 텐서 X의 몇번째 컬럼에 들었는지 매핑
        self.feature_index = build_input_features(feature_columns)
        # 모든 범주형 feature별로 nn.Embedding레이어를 생성한 딕셔너리
        self.embedding_dict = create_embedding_matrix(feature_columns, device = device)

        #embedding 정규화 추가
        self.add_regularization_weight(self.embedding_dict.parameters(), l2 = l2_reg_embedding)

        # sparse 및 dense feature 분류
        self.sparse_feats = [fc for fc in feature_columns if isinstance(fc, SparseFeat)]
        self.dense_feats = [fc for fc in feature_columns if isinstance(fc, DenseFeat)]

        self.fm = FM()
        self.linear_model = LinearPart(feature_columns, device).to(device)
        # Linear 파트 정규화 추가
        self.add_regularization_weight(self.linear_model.parameters(), l2 = l2_reg_linear)

        dnn_input_dim = sum(fc.embedding_dim if isinstance(fc, SparseFeat) else fc.dimension for fc in feature_columns)
        self.dnn = DNN(dnn_input_dim, hidden_units)
        self.dnn_out = nn.Linear(hidden_units[-1],1).to(device)
        
        # DNN hidden layers weight 정규화 추가
        self.add_regularization_weight(filter(lambda x: "weight" in x[0] and "bn" not in x[0], self.dnn.named_parameters()), l2 = l2_reg_dnn)
        # DNN 출력층 weight 정규화 추가
        self.add_regularization_weight(self.dnn_out.weight, l2=l2_reg_dnn)       
        self.sigmoid = nn.Sigmoid()
        
        self.dense_idx = [self.feature_index[fc.name][0] for fc in self.dense_feats]

    def forward(self, X) -> torch.Tensor:
        embed_list = embedding_lookup(X,self.embedding_dict, self.feature_index, self.sparse_feats, to_list= True)
        embed_stack = torch.cat(embed_list, dim=1) #embedding을 한곳에 concat
        embed_flat = embed_stack.view(embed_stack.size(0), -1)
        
        fm_logit = self.fm(embed_stack)
        linear_logit = self.linear_model(X, self.feature_index, self.dense_idx)

        if self.dense_idx:
            dense_tensor = X[:, self.dense_idx].float()
        else:
            dense_tensor = X.new_zeros(X.size(0), 0)

        dnn_input = torch.cat([embed_flat, dense_tensor], dim=1) # 범주형 임베딩 및 실수형 연결하여 입력
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_out(dnn_output)

        # 세 로짓 합산 → 시그모이드
        logit  = fm_logit + linear_logit + dnn_logit
        return self.sigmoid(logit)                   


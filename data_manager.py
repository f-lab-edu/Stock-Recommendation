from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from deepctr_torch.inputs import DenseFeat, SparseFeat
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset


class HoldingsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data(df, sparse_cols, dense_cols, target_col):
    """
    데이터 전처리
    continuous feature -> Standard Scaler
    categorical feature -> Label Encoder
    """
    y = df[target_col].values
    encoders = {}
    for col in sparse_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    scaler = StandardScaler()
    df[dense_cols] = scaler.fit_transform(df[dense_cols])
    X = df[sparse_cols + dense_cols].values.astype(np.float32)
    return X, y, encoders, scaler

def build_feature_columns(df, sparse_cols, dense_cols) -> list:
    """
    continuos 및 categorical feature들을 입력받아 deepfm 모델 입력용 feature column 생성
    """
    feature_columns = []
    for col in sparse_cols:
        vocab_size = df[col].nunique()
        feature_columns.append(SparseFeat(name = col, vocabulary_size = vocab_size, embedding_dim = 8))

    for col in dense_cols:
        feature_columns.append(DenseFeat(name=col, dimension = 1))
    return feature_columns

def flag_top25(group) -> pd.DataFrame:
    """
    기관 별 TOP-25 주식은 1 나머지는 0을 부여
    """
    group = group.sort_values("TOTAL_VALUE", ascending=False)
    cutoff = max(1, int(np.ceil(len(group) * 0.25))) #최소 1개는 1로 표시될 수 있도록
    group["TOP25_FLAG"] = 0
    group.loc[group.head(cutoff).index, "TOP25_FLAG"] = 1
    return group

def merge_data(DATA_DIR = "2023q4_form13f/") -> pd.DataFrame:
    '''
    SUBMISSION.tsv와 INFTABLE.tsv 파일을 병합하여 기관들이 보유중인 주식이 1로 표시된 데이터프레임으로 변환
    (기관은 CIK로 표기되어있으며, 주식은 CUSIP으로 표기)
    '''
    sub_cols = ["ACCESSION_NUMBER", "CIK"]
    info_cols = ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "VALUE"]

    sub_df = pd.read_csv(f"{DATA_DIR}/SUBMISSION.tsv",  sep="\t", usecols=sub_cols, dtype=str)
    info_df = pd.read_csv(f"{DATA_DIR}/INFOTABLE.tsv", sep="\t", usecols=info_cols, 
                        dtype={"ACCESSION_NUMBER": str, "CUSIP": str}, low_memory=False)

    info_df["SSHPRNAMT"] = pd.to_numeric(info_df["SSHPRNAMT"], errors="coerce")
    info_df["VALUE"] = pd.to_numeric(info_df["VALUE"], errors="coerce")

    # 주식 보유하는 경우만 필터링
    mask = (info_df["SSHPRNAMT"] > 0) | (info_df["VALUE"] > 0)
    info_df = info_df[mask]

    # 기관이 소유한 주식 정보 병합
    holdings = info_df.merge(sub_df, on="ACCESSION_NUMBER", how="inner")
    holdings = holdings.drop_duplicates()

    # 기관이 보유중인 종목들의 금액을 전부 합산
    holdings['TOTAL_VALUE'] = holdings.groupby(["CIK", "CUSIP"])["VALUE"].transform("sum")

    # 기업이 보유중인 상위 25% 기업들은 1 나머지는 0부여
    holdings = holdings.groupby("CIK",group_keys=False).apply(flag_top25)

    holdings.to_parquet(f"{DATA_DIR}holdings_data.parquet", index = False)
    return holdings
import os
import random
import time
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import pyarrow.parquet as pq
import torch
import yfinance as yf
from deepctr_torch.inputs import DenseFeat, SparseFeat
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config import *
from prompt import *

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# openai API 키 인증
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI()

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
    cutoff = max(1, int(len(group) * 0.25)) #최소 1개는 1로 표시될 수 있도록
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

def get_sector_industry(ticker):
    """
    yfinance 라이브러리를 이용하여 sector와 industry 조회
    """
    t = yf.Ticker(str(ticker))
    try:
        info = t.get_info()
    except Exception:
        info = getattr(t, "info", {}) or {}
    if not isinstance(info, dict):
        return None, None
    return info.get("sector"), info.get("industry")

def resolve_cusip_ticker(data_path= "CUSIP_TICKER.csv", STR = SYSTEM_TICKER_RESOLVER):
    """
    CUSIP코드를 이용하여 ticker로 변환하는 함수
    chatgpt api를 이용하기 때문에 완벽하게 모든 데이터를 변환하지는 못함

    * CUSIP코드 -> TICKER로 매핑 가능한 데이터셋 및 웹사이트는 확인하지 못함
    * 변환 방법은 아래 2가지 방법이 있으나 제약이 있음
    1) FIGI API는 회사계정 메일이 필요하여 이용 불가능
    2) POLYGONIO의 API는 분당 5회만 호출 가능

    이에 따라, 조금 부정확하더라도 chatgpt api를 이용하여 변환하는 방법을 선택(약 7시간 정도 소요)
    """
    cusip_df = pd.read_csv(data_path)
    if "Ticker" not in cusip_df.columns:
        cusip_df["Ticker"] = None

    for index, row in tqdm(cusip_df.iterrows(), total=len(cusip_df), desc="Resolving", unit="cusip"):
        prompt = build_ticker_prompt(row["CUSIP"])
        if pd.notna(row["Ticker"]):
            #print(index)
            continue
        
        for attempt in range(5):
            try:
                cusip = row["CUSIP"]
                response = client.chat.completions.create(
                model = "gpt-4o",
                messages=[
                    {"role": "system", "content": STR},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, #동일한 질문에 대하여서도 최대한 동일한 답변을 생성하도록 파라미터 설정
                )
                ticker = response.choices[0].message.content
                break
            except Exception as e:
                if attempt == 4:
                    ticker = f"ERROR:{type(e).__name__}"
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
        cusip_df.at[index, "Ticker"] = ticker
        cusip_df.to_csv(data_path, index=False)
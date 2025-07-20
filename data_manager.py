from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

def weight_top10(group, n = 10) -> pd.DataFrame:
    """
    기관 별 TOP-10 주식들은 가중치(10~1)를 부여하고 나머지는 0으로 변환
    """
    cnt = min( 10 , len(group))
    group = group.sort_values("TOTAL_VALUE", ascending=False).copy()
    group["WEIGHT_TOP10"] = 0 # 가중치 초기화
    
    top_idx = group.head(n).index
    rank_to_weight = np.arange(n, 0, -1)[:cnt] # 기업이 보유한 종목수가 10미만일 경우에도 10부터 내림차순으로 가중치 부여

    group.loc[top_idx, "WEIGHT_TOP10"] = rank_to_weight
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

    # 상위 10개 기업에 대하여 10~1까지 가중치 부여
    holdings = holdings.groupby("CIK",group_keys=False).apply(weight_top10)
    
    holdings.to_parquet(f"{DATA_DIR}holdings_data.parquet", index = False)
    return holdings
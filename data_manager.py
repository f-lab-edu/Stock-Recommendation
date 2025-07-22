from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

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
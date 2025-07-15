from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

def merge_data(DATA_DIR = "2023q4_form13f/"):
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
    holdings = info_df.merge(sub_df, on="ACCESSION_NUMBER", how="inner")[["CIK", "CUSIP"]]
    holdings = holdings.drop_duplicates()
    holdings["hold"] = 1

    holdings.to_parquet(f"{DATA_DIR}holdings_data.parquet", index = False)
    return holdings
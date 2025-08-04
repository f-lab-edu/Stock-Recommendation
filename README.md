# Stock-Recommendation

## a. 현재 진행 상황
1. deepFM 모델 구현 중(약 80% 완성)  
2. 구현한 각각의 부분에 대하여 deepctr과 결과가 일치하는지 검증 코드 작성

## b. 파일 설명
1. baseline_test.ipynb -> 베이스라인 추천 모델들에 대하여 결과 확인 노트북 파일
2. data_manager.py -> 원본 데이터(5개의 tsv) 전처리 및 병합 코드 관리용 
3. model_components.py -> DeepFM 모델을 구성하는 각 class를 정리해둔 파일
4. model_verification.ipynb -> 구현한 deepfm 모델 구성 요소들과 deepctr의 모델 성능 비교 노트북 파일
5. utils.py -> 향후 각 기타 함수들 관리용 파일(현재는 seed 고정만 있음)

## b. To-Do List
1. class DeepFM에 add_regularization_weight 구현 및 검증  
2. CLI로 코드 실행 가능하도록 업데이트 예정  
3. 기존 데이터셋에 feature 추가를 위한 데이터 크롤러 및 데이터 전처리 코드 업데이트 예정
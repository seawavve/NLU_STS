import numpy as np
import pandas as pd

import json

import cleaning as cl
import eda

class data_preprosessing():
    """
    -----data preprocessing-----
    |   1. cleaning             |
    |   2. back translation     |
    |   3. data augmentation    |
    |___________________________|
    - settings:
        - module_path 본인 환경에 맞게 수정할 수 있도록 추후 업데이트 예정
        - !pip install konlpy : data augmentation module 사용을 위한 install
        - !pip install pororo : back translation module 사용을 위한 install ***지금은 사용 안함!!***
    - class : data_preprosessing(path, n)
        param : 
            - path              : origin data file path
            - n=2               : data augmentation times(default=2)

    - functions : 
            - cleaning_df()         : input=None, output=DataFrame
            - back_translation(df)  : input=(cleaned)DataFrame, output=DataFrame
            - data_aug(df)          : input=(cleaned)DataFrame, output=DataFrame
            - load_df()             : input=None, output=DataFrame train, valid test
    - sample code :
        path = '[DATA PATH]'                                                    # data 저장된 path
        n = 2                                                                   # augmentation times
        pp = data_preprosessing(path, n)                                        # class param : path, n
        _train, _test = pp.load_origin_file()                                   # return ** UNCLEANED & REMOVE DUPLICATE ** train, dev
        *** _train, _valid = train_test_split() ***                             # split train, valid(** DO NOT RESET INDEX **)
        train, valid, test = pp.load_all(_train, _valid, _test, 
                                        shuffle=False(OPTIONAL:default=False))  # return pre-processed train, cleaned valid, cleaned test
    """

    def __init__(self, path, n=2):
        
        self.path = path                                                    # csv 파일 불러오고 저장할 path
        self.synonyms = pd.read_csv(path+'NIKLex_synonym.tsv', sep='\t')    # augmentation에서 사용할 말뭉치
        self.n = n                                                          # aumentation param
        self.org_train = 'https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-sts-v1.1/klue-sts-v1.1_train.json'
        self.org_dev = 'https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-sts-v1.1/klue-sts-v1.1_dev.json'
        self.col1 = 'sentence1'
        self.col2 = 'sentence2'
        self.label = 'labels'
        self.bt_file = path + 'STS_backtranslation.csv'

    def data_check(self, df, col1=None, col2=None):
        # 결측치 제거
        df = df.dropna(axis=0)
        
        # 중복값 제거
        df = df.drop_duplicates([col1,col2], keep='first')

        # BT data에서 train 값 뽑기 위한 index reset
        df = df.reset_index(drop=True)

        return df

    def load_origin_file(self):
        df = pd.read_json(self.org_train)

        # train label 추가
        df['binary'] = df['labels'].apply(lambda x: x['binary-label'])
        df['normalized'] = df['labels'].apply(lambda x: float(x['label']/5.0))
        df['score'] = df['labels'].apply(lambda x: x['label'])

        df = df.drop(columns='annotations', axis=1)
        
        # 결측치, 중복 제거
        df = self.data_check(df, self.col1, self.col2)
            
        # dev set
        test = pd.read_json(self.org_dev)

        # dev label 추가
        test['binary'] = test['labels'].apply(lambda x: x['binary-label'])
        test['normalized'] = test['labels'].apply(lambda x: float(x['label']/5.0))
        test['score'] = test['labels'].apply(lambda x: x['label'])

        # 결측치, 중복 제거
        test = self.data_check(test, self.col1, self.col2)

        print(f"Length of original DF : {len(df)}")
        print(f"Length of original dev : {len(test)}")

        return df, test

    def cleaning_df(self , df):
        # data cleaning
        _cl = cl.DataCleaning()
        cl_df = _cl.make_cleaned_df(df, self.col1, self.col2, self.label)
        
        return cl_df

    def back_translation(self, cl_df):
        
        # back translation을 모두 돌리는건 load의 문제가 있어 취합된 파일만 불러오기

        _bt_df = pd.read_csv(self.bt_file)

        # 취합된 BT data에서 train index row 추출
        indexes = cl_df.index

        for i, index in enumerate(indexes):

            error_idx = []
            if i == 0:
                bt_df = _bt_df.iloc[[index]]
            else:
                try:
                    bt_df = bt_df.append(_bt_df.iloc[[index]])
                except:
                    error_idx.append(index)

        return bt_df

    def data_aug(self, cl_df):
        
        _eda = eda.NLPAugment(cl_df, self.synonyms)

        # col1 augmentation
        print(f'\ncolumn name : {self.col1}')
        da_df_1 = _eda.augment_df_to_rows(self.col1, self.n)

        # col2 augmentation
        print(f'\ncolumn name : {self.col2}')
        da_df_2 = _eda.augment_df_to_rows(self.col2, self.n)

        # concat
        da_df = pd.concat([da_df_1, da_df_2], ignore_index=True)

        return da_df

    def load_all(self, train, valid, test, shuffle:bool=False):

        print('\n************** cleaning : train **************\n')
        cl_train = self.cleaning_df(train)            # original train -> cleaning
        valid = self.cleaning_df(valid)               # original valid -> cleaning
        test = self.cleaning_df(test)                 # original test -> cleaning

        # data check null, duplicated
        cl_train = self.data_check(cl_train, self.col1, self.col2)
        valid = self.data_check(valid, self.col1, self.col2)
        test = self.data_check(test, self.col1, self.col2)
        print(f'\n************** cleaned train length : {len(cl_train)}, valid length : {len(valid)}, test length : {len(test)} **************\n')

        print('\n************** back translate : train **************\n')
        bt_df = self.back_translation(train)    # train -> back translation
        bt_df = self.cleaning_df(bt_df)         # back translation -> cleaning
        bt_df = self.data_check(bt_df, self.col1, self.col2)    # data check null, duplicated
        print(f'\n************** back translate train length : {len(bt_df)} **************\n')

        print('\n************** data augmentation : train **************\n')
        da_df = self.data_aug(cl_train)  # cleaned data -> data augmentation
        da_df = self.cleaning_df(da_df)         # data augmentation -> cleaning
        da_df = self.data_check(da_df, self.col1, self.col2)    # data check null, duplicated
        print(f'\n************** data augmentation train length : {len(da_df)} **************\n')

        print('\n************** concat : train **************\n')
        train = pd.concat([cl_train, bt_df, da_df], axis=0, ignore_index=True)
        train = self.data_check(train, self.col1, self.col2)    # data check null, duplicated
        print(f'\n************** concat train length : {len(train)} **************\n')

        if shuffle == True:
            print(f'\n************** shuffle train, valid, test **************\n')
            train = train.sample(frac=1).reset_index(drop=True)
            valid = valid.sample(frac=1).reset_index(drop=True)
            test = test.sample(frac=1).reset_index(drop=True)

        print('\n************** saving files : train, valid, test **************\n')
        train.to_csv(self.path + 'train.csv', index=False)
        valid.to_csv(self.path + 'valid.csv', index=False)
        test.to_csv(self.path  + 'test.csv', index=False)

        print(f'\n************** train length : {len(train)}, valid length : {len(valid)}, test length : {len(test)} **************\n')

        return train, valid, test
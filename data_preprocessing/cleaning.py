import numpy as np
import pandas as pd
import re
from tqdm import tqdm

class RemoveRgx():

    def remove_special_charecter(self, texts):
        print(f'Remove special charecter start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r"[^0-9a-zA-Z가-힣一-龥㐀-䶵豈-龎ㄱ-ㅎㅏ-ㅣ\t\n\.]", " ", str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def remove_chiness(self, texts):
        print(f'Remove chiness charecter start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r'[一-龥㐀-䶵豈-龎]', '', str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def remove_eng(self, texts):
        print(f'Remove english start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub('[a-zA-Z]', '', str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def remove_repeated_spacing(self, texts):
        print(f'Remove repeated spacing start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r"\s+", " ", str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def remove_repeated_dot(self, texts):
        print(f'Remove repeated spacing start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r"\.+", ".", str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

class DataCleaning():

    def cleaning(self, texts):
        rmv = RemoveRgx()

        result = rmv.remove_chiness(texts)
        result = rmv.remove_eng(result)
        result = rmv.remove_special_charecter(result)
        result = rmv.remove_repeated_dot(result)
        result = rmv.remove_repeated_spacing(result)

        return result

    def make_cleaned_df(self, df, col1, col2, label):

        cleaned_df = pd.DataFrame()

        _df = df.loc[:, col1:col2]
        label_df = df.loc[:, label:]

        col_list = _df.keys()
        col_dict = dict()

        label_list = label_df.keys()
        label_dict = dict()

        print('\nCleaning start\n')
        for col in col_list:
            print(f'\ncolumn name : {col}\n')
            col_dict[col] = self.cleaning(_df[col].tolist())

        for key, item in col_dict.items():
            cleaned_df[key] = item

        for col in label_list:
            label_dict[col] = label_df[col].tolist()
            
        for key, item in label_dict.items():
            cleaned_df[key] = item

        return cleaned_df
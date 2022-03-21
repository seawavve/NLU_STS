import numpy as np
import pandas as pd
import re
from tqdm import tqdm

class RemoveRgx():
    def remove_html(self, texts):
        print(f'Remove html start')
        preprcessed_text = []
        for text in tqdm(texts):
            text = re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", str(text))
            if text:
                preprcessed_text.append(text)

        return preprcessed_text

    def remove_url(self, texts):
        print(f'Remove url start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", str(text))
            text = re.sub(r"pic\.(\w+\.)+\S*", "", str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text
    
    def remove_press(self, texts):
        print(f'Remove press start')
        re_patterns = [
            r"\([^(]*?(뉴스|타임스|경제|언론|포브스|일보|미디어|데일리|한겨례|홋스퍼HQ|타임즈|위키트리|BBC로마노|로마노|BBC|더 선|스퍼스 웹|토트넘 팬사이트|스카이스포츠|ESPN|스포르트|일문일답|아르헨 기자|스퍼스웹|카데나세르|텔레그래프|빌트)\)",
            r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장) ",  # 이름 + 기자
            r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리)",  # (... 연합뉴스) ..
            r"\(\s+\)",  # (  )
            r"\(=\s+\)",  # (=  )
            r"\(\s+=\)",  # (  =)
            r"\[.*?\]",
            r"\(.*매체\)",
            r"\(.*단독\)",
            r".*기자=",
        ]

        preprocessed_text = []
        for text in tqdm(texts):
            for re_pattern in re_patterns:
                text = re.sub(re_pattern, "", str(text))
            if text:
                preprocessed_text.append(text)  

        return preprocessed_text

    def remove_copyright(self, texts):
        print(f'Remove copyright start')
        re_patterns = [
            r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
            r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
        ]
        preprocessed_text = []
        for text in tqdm(texts):
            for re_pattern in re_patterns:
                text = re.sub(re_pattern, "", str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text
        
    def remove_photo_info(self, texts):
        print(f'Remove photo info start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r"사진 =.+Images|사진=.*|\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자", "", str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def remove_dash(self, texts):

        print(f'Remove dash start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r'(?<=[0-9*])(-)(?=[0-9*])', '대', str(text))
            text = re.sub(r'-', ' ', str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

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

    def remove_bracket(self, texts):
        print(f'Remove bracket start')
        preprocessed_text = []
        for text in tqdm(texts):
            text = re.sub(r'\([^)]*\)', '', str(text))
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def remove_date_number(self, texts):
        print(f'Remove date number start')
        re_patterns = [
            r"[0-9]+일",
            r"[0-9]+월",
            r"[0-9]+년",
            r"[0-9]+\.+[0-9]*"
        ]
        preprocessed_text = []
        for text in tqdm(texts):
            for re_pattern in re_patterns:
                text = re.sub(re_pattern, "", str(text))
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

class DataCleaning():

    def cleaning(self, texts, flag):
        rmv = RemoveRgx()

        if flag == 'STS':
            result = rmv.remove_chiness(texts)
            result = rmv.remove_eng(result)
            result = rmv.remove_special_charecter(result)
            result = rmv.remove_repeated_spacing(result)

        elif flag == 'NLG':
            result = rmv.remove_html(texts)
            result = rmv.remove_url(result)
            result = rmv.remove_press(result)
            result = rmv.remove_copyright(result)
            result = rmv.remove_photo_info(result)
            result = rmv.remove_bracket(result)
            result = rmv.remove_dash(result)
            result = rmv.remove_chiness(result)
            result = rmv.remove_date_number(result)
            result = rmv.remove_special_charecter(result)
            result = rmv.remove_repeated_spacing(result)

        return result

    def make_cleaned_df(self, df, col1, col2, flag, label=None):

        cleaned_df = pd.DataFrame()

        if flag ==  'STS':
            _df = df.loc[:, col1:col2]
        elif flag == 'NLG':
            _df = df.loc[:, col1:]

        col_list = _df.keys()
        col_dict = dict()

        print('\nCleaning start\n')
        for col in col_list:
            print(f'\ncolumn name : {col}\n')
            col_dict[col] = self.cleaning(_df[col].tolist(), flag)

        for key, item in col_dict.items():
            cleaned_df[key] = item

        if flag ==  'STS':
            label_df = df.loc[:, label:]
            label_list = label_df.keys()
            label_dict = dict()
            for col in label_list:
                label_dict[col] = label_df[col].tolist()
            for key, item in label_dict.items():
                cleaned_df[key] = item

        return cleaned_df
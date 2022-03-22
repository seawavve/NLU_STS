import numpy as np
import pandas as pd
import pororo           # pip install pororo 필요
from pororo import Pororo
from tqdm import tqdm

class Korean_backtranslation():
    '''
    description
    use 'execute' to back translate

    [Input]
    df_ori : dataframe
    col_lst: a list of column name to translate
    lang: a language to use in back translatoin in string, default:'en'

    [Output]
    a translated dataframe

    [usage]
    df_new = Korean_backtranslation(df, ['title', 'content'])

    '''

    def __init__(self):
        self.pororo = Pororo(task = 'translation', lang='multi')
    
    # translatoin korean -> another lang
    def to_lang(self, s, lang):
        transed_lang = self.pororo(s, src = 'ko', tgt = lang) # ko to other lang
        return transed_lang
    
    # translation another lang -> korean
    def to_korean(self, s, lang) :
        back_transed_lang = self.pororo(s, src = lang, tgt = 'ko') # another to ko
        return back_transed_lang
    
    def execute(self, df_ori, col_lst: list, lang: str = 'en'):
        if len(df_ori[df_ori.duplicated(col_lst, keep=False)]) != 0:
            raise Exception('Drop duplicates before execution')
    
        df = df_ori.copy()
        print(f'Translatoin start: from Korean to {lang}')
        for col in col_lst:
            print(f'\ncolumn name: {col}\n')
            new = []
            for i in tqdm(range(len(df))):
                datas = df[col].values[i].split('.')
                _new = []
                for data in datas:
                    try:
                        kor_to_lang = self.to_lang(data, lang)
                        lang_to_kor = self.to_korean(kor_to_lang, lang)
                        _new.append(lang_to_kor)
                    except:
                        _new.append('FAILED')
                        print(f'\nTRANSLATION FAILED: col - {col} / row - {i} / data - {data}')
                new.append(' '.join(_new))
            df[col] = new
        return df
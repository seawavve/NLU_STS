from ast import arguments
from asyncio import all_tasks
import imp
import random
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt

class NLPAugment():   
    """
    description
    
    - df : augmented 될 문자열 dataframe
    - synonyms : 유의어 사전. df 혹은 dictionary 형태로 전달 
    - okt : 형태소 분석기. 어간추출 및 조사 제거용
    - words = 토큰 리스트   
    
    """
    def __init__(self, df,synonyms): 
        self.synonyms = synonyms[synonyms['mean'] >= synonyms['mean'].mean()]
        self.df = df
        self.okt = Okt() 
    
    
    def _split_token(self,words):
        """
        split token into two list. 조사 제거용
        """
        posed=self.okt.pos(words, stem=True)
        
        no_aug_token = []
        aug_token = []

        for idx, wp in enumerate(posed):
            if wp[1] != 'Josa':
                aug_token.append((idx,wp[0]))
            else:
                no_aug_token.append((idx,wp[0])) 
        
        return no_aug_token, aug_token # 조사토큰, 조사 제외 토큰
        

    ##random synonym replacement
    def synonym_replacement(self, words, n):
        """
        random 한 단어를 유의어로 대체
        """
        no_aug_token, new_words = self._split_token(words)
        random_word_list = list(set([word for word in new_words if len(word) > 1]))
        random.shuffle(random_word_list)
        random_word_list= [x[1] for x in random_word_list]
        
        num_repl = 0
        
        for random_word in random_word_list:
            if random_word in self.synonyms['word1'].values:
                synonym = self.get_synonyms(random_word)
                
                if len(synonym) >= 1: # 유의어가 하나이상
                    synonym = random.choice(list(synonym))
                    new_words = [(x[0],synonym) if x[1] == random_word else x for x in new_words]
                    num_repl += 1
                if num_repl >= n:
                    break 
        if len(new_words) != 0:
            all_tokens = new_words + no_aug_token
            all_tokens = sorted(all_tokens, key=lambda x: x[0])
            all_tokens = [x[1] for x in all_tokens]
            sentence = ' '.join(all_tokens)
            all_tokens = sentence.split(' ')
        
        return all_tokens

    def get_synonyms(self,word:str):
        """
        get synonyms from synonyms dictionary
        """
        synonyms = []

        try:
            synonyms=self.synonyms[self.synonyms['word1'] == word]['word2'].values.tolist()
        except:
            pass
        
        return synonyms
    
    # random swap
    def swap_words(self,new_words):
        """
        randomly swap two words
        """
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        
        counter = 0
	
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words

        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 

        return new_words

    
    def random_swap(self,words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_words(new_words)
        return new_words

    # random insertion
    def random_insertion(self,words, n):
        #new_words = words.copy()
        no_aug_token, new_words = self._split_token(words)
        
        for _ in range(n):
            self.add_word(new_words)
        
        all_tokens = new_words + no_aug_token
        all_tokens = sorted(all_tokens, key=lambda x: x[0])
        all_tokens = [x[1] for x in all_tokens]
        
        return all_tokens
    
    def add_word(self,new_words):
        """
        add synonym word to a random position
        """
        synonyms= []
        counter = 0
        while len(synonyms) < 1:
            if len(new_words) >= 1:
                random_word = new_words[random.randint(0, len(new_words)-1)]
                synonyms = self.get_synonyms(random_word[1])
                counter += 1
            else :
                random_word = ("", "")
            if counter > 10:
                return
                
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, (random_word[0], random_synonym))
    
    # random deletion
    def delete_word(self,words,p):
        """
        randomly delete a word with probability p
        """
        if len(words) == 1:
            return words
        
        new_words = []
        for word in words:
            r = random.uniform(0,1)
            if r > p:
                new_words.append(word)
        
        # if len(new_words) == 0: delete a random word
        if len(new_words) == 0:
            rand_int = random.randint(0,len(words)-1)
            del words[rand_int]
            return words
        
        return new_words

        
    def augment_text(self,words,n:int,p_sr=0.2,p_ri=0.2,p_rd=0.2,p_rs=0.2) -> str:
        """
        input 문장에 대해 augmentation을 수행하는 함수
        n 개의 augmentation 된 문장이 반환됨
        """
        sr_ri_words = words # sentence를 input으로 처리
        words = words.split(' ')
        words = [word for word in words if word != '']
        
        num_new_per_technique = int(n/4) + 1 # agumentation 기법 당 추가할 문장 수
        augmented = []
        
        #Synonym Replacement

        if (p_sr > 0):
            n_sr = max(1, int(p_sr*n))  # token 수가 클 수록 augmentation 횟수가 많아짐
        for _ in range(num_new_per_technique):
            a_words = self.synonym_replacement(sr_ri_words, n_sr)
            augmented.append(" ".join(a_words)) # 문장으로 반환할 경우 " ".join(a_words) 사용 // augmented.append(a_words)

        #Random Swap

        if (p_rs > 0):
            n_rs = max(1, int(p_rs*n))
        for _ in range(num_new_per_technique):
            a_words = self.random_swap(words, n_rs)
            augmented.append(" ".join(a_words))
            
        # Random Insertion

        if (p_ri > 0):
            n_ri = max(1, int(p_ri*n))
        for _ in range(num_new_per_technique):
            a_words = self.random_insertion(sr_ri_words, n_ri)
            augmented.append(" ".join(a_words))
        
        # Random Deletion

        if (p_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = self.delete_word(words, p_rd)
                augmented.append(" ".join(a_words))
        
        augmented = [a for a in augmented if a]
        
        random.shuffle(augmented)
        
        
        # trimming을 통한 개수 맞추기
        if n >= 1:
            augmented = augmented[:n]
        else:
            keep_prob = n / len(augmented)
            augmented = [s for s in augmented if random.uniform(0, 1) < keep_prob]

        return augmented
    
    
    def augment_df(self, s : str, n : int, p_sr=0.2, p_ri=0.2, p_rd=0.2, p_rs=0.2):
        """
        s : column name
        n : number of augmentation
        return : list of list of augmented text
        """
        
        aug_list = list(self.df[s].apply(lambda x: self.augment_text(x,n,p_sr,p_ri,p_rd,p_rs)))
        
        return  aug_list
    
    
    def augment_df_to_columns(self, s : str, n : int, p_sr=0.2, p_ri=0.2, p_rd=0.2, p_rs=0.2):
        """
        add augmentation to dataframe columns
        """
        
        aug_list = self.augment_df(s,n,p_sr,p_ri,p_rd,p_rs)
        colnames = [s+"_aug_"+str(i) for i in range(n)]
        
        aug_df = pd.DataFrame(aug_list)
        aug_df.columns = ["aug_text"]
        
        aug_df=pd.DataFrame(aug_df["aug_text"].to_list(), columns=colnames)
        
        res_df = pd.concat([self.df, aug_df], axis=1)
        
        return res_df
    
    def augment_df_to_rows(self, s: str, n : int, p_sr=0.2, p_ri=0.2, p_rd=0.2, p_rs=0.2):
        """
        add augmented text to each row
        """
        aug_df = self.df.copy()
        res_df = aug_df.iloc[0:0]

        add_cols = self.df.columns.to_list()

        add_cols.remove(s)
        
        aug_list = self.augment_df(s,n,p_sr,p_ri,p_rd,p_rs)
        
        aug = []
        
        for i in range(len(aug_list)):
            for j in range(len(aug_list[i])):
                aug.append([aug_list[i][j],aug_df.iloc[i][add_cols[0]],aug_df.iloc[i][add_cols[1]], aug_df.iloc[i][add_cols[2]], aug_df.iloc[i][add_cols[3]], aug_df.iloc[i][add_cols[4]]])
                
        aug = pd.DataFrame(aug, columns=[s]+add_cols)
        
        res_df = pd.concat([res_df, aug], axis=0)
        res_df.reset_index(drop=True, inplace=True)        
        
        return res_df
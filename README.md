# NLU_STS
한국어 문장의 유사도 분석 모델 훈련 및 서비스화  
프로젝트 목표는 학습 데이터 셋을 사용하여 의미적 텍스트 유사도 모델 훈련


## Process
Data → Data Preprocessing → Data Augumentation → Model → Param Tuning → Metric → Evaluation → Serving  
  
두 개의 한국어 문장을 입력받아 두 문장의 의미적 유사도(STS)를 출력하는 모델을 생성하기 위해 한국어 문장 유사도 데이터셋 KLUE-STS을 사용합니다. 위 데이터의 문장 데이터를 두 가지 방법으로 데이터를 증강시켜 과적합을 방지했습니다. 문장이 한글로 이루어져 있고 의미론적으로 두 문장의 유사도를 판단하기 위해 Sentence-BERT 기반 모델을 사용했으며 Pretrained Model로 huggingface의 Huffon/sentence-klue-roberta-base 모델을 사용했습니다. 주어진 데이터에 더 Fit하는 모델을 찾기 위해 파라미터를 튜닝했습니다. 잘 학습됐는지 평가하기 위해 F1 과 Pearson's r score를 사용하여 비교했습니다. 위 모델을 FastAPI 프레임워크로 서빙했습니다. 문장 2개를 API에 넣어 간편하게 inference값을 반환할 수 있도록 만들었습니다.  



## Data
데이터로 [KLUE-STS](https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/klue-sts-v1.1) (Korean Language Understanding Evaluation - STS)을 사용했습니다. 두 문장의 의미적 유사도 Task를 해결하기 위해 만들어진 한글 데이터 셋입니다. AIRBNB에서 구어체 리뷰, policy에서 격식체 뉴스 그리고 QaraKQC에서 스마트 홈 쿼리 총 세 가지 도메인으로 구성됩니다. 전체 데이터 개수는 약 Train:Dev:Test = 20:1:2의 비율입니다. Train Data는 총 11668개, Dev Data는 519개, Test Data는 1037개로 총 13224개입니다. 각 데이터는 sentence1, sentence2, label 총 세 개의 칼럼으로 구성되어 있습니다. 라벨은 real-label, label, binary-label 세 가지 라벨로 구성되어 있습니다. real-label은 두 문장의 유사도를 실수형으로 표현한 라벨, label은 real-label을 소수점 한자리 수만큼 표현한 라벨, binary-label은 real label에 0.3의 threshold를 주어 일정값을 기준으로 0과 1로 이진화한 라벨입니다. 
<img width="874" alt="image" src="https://user-images.githubusercontent.com/66352658/158842853-19d8df99-429d-4b41-97b0-e01a195e3b8f.png">


## Data Preprocessing
모델 성능에 영향을 끼칠 수 있는 Noise를 제거하기 위해 데이터 전처리를 했습니다. 한글 기반의 학습을 위해 한글 외 한자, 일본어 등의 문자를 제거하였고 특수문자를 제거하였습니다. 숫자, 조사와 접사를 제거할 경우 문장 본연의 의미가 변질되는 경우를 고려하여 제거하지 않았습니다. 결측치를 제거하고 중복을 제거하여 정제된 데이터를 추출했습니다.   
[Preprocessed Data Download](https://github.com/sw6820/STS/blob/yerim/data/cleaned.csv)   
[Preprocessed Dev Data Download](https://github.com/sw6820/STS/blob/yerim/data/cleaned_test.csv)  

- 전처리 후 데이터 개수 : 11,661개
- 데이터 칼럼
   - sentence1 : 유사도 비교 pair text1
   - sentence2 : 유사도 비교 pair text2
   - labels : dict 형식으로 구성된 label 묶음
   - score : sentence1과 sentence2 사이의 유사도 점수. 유사도 정도에 따라 1에서 5의 값을 가짐
   - binary : score를 3점 기준으로 binary label로 바꾼 것 3점 이상이면 1 . 그외는 0
   - normalized : score가 0에서 1 사이의 값을 가지도록 정규화한 label
   
* Preprocessed Train Data
<img width="991" alt="image" src="https://user-images.githubusercontent.com/66352658/159203083-ffa2f257-6472-4eef-9b65-15d7f7d2f1b9.png">
<p align="center"><img src = "https://user-images.githubusercontent.com/92706101/159195790-d52907e5-dc9f-4625-921f-903dd881b863.png"></p>

* Preprocessed Dev Data
<img width="905" alt="image" src="https://user-images.githubusercontent.com/66352658/159229256-8386253b-45e4-4136-b56e-b84d043ad7b4.png">
<p align="center"><img src = "https://user-images.githubusercontent.com/92706101/159195864-49dccb35-051e-4135-9886-cfb690b6aca0.png"></p>


## Data Augumentation
　데이터 과적합을 방지하기 위해 데이터를 증강했습니다. 논문을 서치하여 Back Translation과 EDA 총 두 가지 텍스트 증강 기법을 선정했습니다. 위 기법으로 편향이 적게 발생하도록 하면서 기존 데이터 개수를 늘려 모델의 일반화 성능을 올렸습니다.
두 문장의 유사도를 도출하는데 있어서 두 문장 모두 증강 처리한 데이터에 기존 라벨값을 사용하는 것은 데이터 오염 위험이 있습니다. 그러므로 두 문장 중 한 문장은 증강된 문장, 한 문장은 기존 문장을 페어로 묶어 증강 데이터를 생성했습니다. 최종 데이터는 Train과 Test를 9:1 비율입니다. Test 데이터는 모델의 성능을 평가하기 위한 지표로 사용되는 데이터 셋입니다. 우리 팀은 조금이라도 더 많은 양의 데이터를 학습해서 모델 자체의 성능을 끌어올리는게 중요하다고 생각하여 위와 같은 데이터 비율을 선정했습니다.  
[Data Augumentation Code](https://github.com/sw6820/STS/blob/yerim/STS/modules/data_preprocessing_module.py)    
[Augumented Data Download](https://drive.google.com/file/d/15n4u1m_Y9XHPWD8mF3TmjltWYy7sdv2c/view?usp=sharing)  
[데이터 비율 선정 참고자료](https://brunch.co.kr/@coolmindory/31#:~:text=%EC%95%84%EB%A7%88%EB%8F%84%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%84%20%EA%B3%B5%EB%B6%80,%EB%A5%BC%20%EB%A7%8E%EC%9D%B4%20%EB%93%A4%EC%96%B4%EB%B3%B4%EC%95%98%EC%9D%84%20%EA%B2%83%EC%9D%B4%EB%8B%A4)  

<img width="710" alt="image" src="https://user-images.githubusercontent.com/66352658/159227622-fa6fd3fc-51e0-460c-a267-ff6b7193e644.png">


### Back Translation  

Back Translation은 한글 텍스트를 영어 텍스트로 변환한 다음 다시 한 번 한글로 번역하는 방식입니다. Source Sentence를 주어 Target Sentence를 생성하고 이로 변형된 Source Sentence는 방식을 사용하여 자연스러운 인공데이터를 생성합니다. 데이터를 증강한 후 중복되는 데이터는 제거했습니다.   
  
Back Translation은 어떤 번역기를 선택하느냐에 따라 성능이 좌우된다는 점을 고려했습니다. Pororo, Papago, Googletrans 세 가지 번역 API를 염두해 직접 성능을 실험했습니다. 세 가지 번역기 모두 번역 품질은 우수했습니다. Papago를 사용할 경우 1,000,000 글자 당 20,000 원이 부과된다는 금전적인 이슈,  Googletrans를 사용할 경우 Google Translation 웹 버전의 제한으로 인해 항상 제대로 작동하지 않을 수 있다는 불안정성 이슈로 인해 시간이 더 오래걸린다는 점을 감안하고 Pororo Translator를 선정하여 BackTranslation을 진행했습니다. [실험 코드](https://github.com/seawavve/NLP_wavve/blob/main/Translator_%EB%B9%84%EA%B5%90%EB%B6%84%EC%84%9D.ipynb)
<img width="705" alt="image" src="https://user-images.githubusercontent.com/66352658/159211089-c3f84eec-0ddc-4901-9e12-dc108bbb3d73.png">


    
### EDA (Easy Data Augmentation)
[EDA](https://arxiv.org/pdf/1901.11196v2.pdf)는 4가지 증강 기법을 사용하여 텍스트 데이터의 양을 증강시키는 방법입니다. SR, RI, RS, RD 기법을 사용합니다.
   - SR(Synonym Replacement): 문장에서 랜덤으로 불용어를 제외한 n개의 단어를 선택하여 동의어로 바꾼다.  
     봄 날씨 너무 좋지 않니? => 봄 계절 너무 좋지 않니?
   - RI(Random Insertion): 문장에서 랜덤으로 불용어를 제외하여 단어를 선택하고, 해당 단어의 유의어를 문장 내 임의의 자리에 넣는다. 이를 n번 반복한다.  
     봄 날씨 너무 좋지 않니? => 봄 날씨 가을 너무 좋지 않니?
   - RS(Random Swap): 무작위로 문장 내에서 두 단어를 선택하고 위치를 바꾼다. 이를 n번 반복한다.  
     봄 날씨 너무 좋지 않니? => 봄 않니 너무 좋지 날씨?
   - RD(Random Deletion): 단어마다 p의 확률로 랜덤하게 삭제한다.  
     봄 날씨 너무 좋지 않니? => 봄 너무 좋지 않니?

RI와 SR의 경우 해당 언어 Wordmap이 필요한 증강 기법입니다. KAIST에서 배포한 한글 워드맵 Korean Wordnet을 사용하여 두 기법을 테스트 해 본 결과 부적절한 유의어로 증강되는 경우가 있었습니다. 위 이슈를 해결하기 위해 국립국어원에서 제공하는 유의어 사전 자료를 사용했습니다. [국립국어원](https://corpus.korean.go.kr/)에서 어휘 관계 자료 : NIKLex를 토대로 유의어 사전을 만들었습니다. <우리말샘>에 등록된 비슷한말, 반대말, 상위어, 하위어 어휘 쌍을 대상으로 어휘 관계 강도를 5점 척도로 총 5만 명이 평가한 자료입니다. 기존 Korean Wordnet의 경우 유사도를 제공하는 단어 수가 9714개로 적고 각 단어의 유사도의 수준도 점수로 제공되지 않는 반면 모두의 말뭉치 자료에서는 60000개의 단어쌍의 유사도와 점수를 제공합니다. 모두의 말뭉치를 통해 유의어 개수가 적고 신뢰도가 적다는 한국어 Wordnet 문제를 해결했습니다. 유의미한 유의어를 다루기 위해 유의어 단어 쌍 유사도의 전체 평균인 3.7284703333333455 이상인 유의어 단어쌍을 사용했습니다. 최종적으로 모두의 말뭉치 유의어 사전에서 단어간 유사도가 평균 유사도 단어쌍보다 높은 33895개의 한국어 단어 유사도 단어쌍을 사용하여 사전을 만들었습니다.  


아래 표는 논문에 소개된 4가지 기법의 파라미터 설정에 따른 성능 실험 그래프입니다. 알파 파라미터는 각 증강에 의해 변경된 문장의 단어 비율입니다. 이 파라미터 값이 약 0.1~0.2 사이일 때 최적 성능이라는 논문 연구 결과에 따라 본 코드에서는 모든 EDA 기법의 알파 파라미터를 0.2로 설정합니다.

   <img src = "https://user-images.githubusercontent.com/43432539/159114591-ded43c28-5dde-414e-9af9-73b7289d20bd.png" width="800"/>


## Model

### Pretrained Model

NLU 문장유사도 계산(STS) Task를 해결하기 위해 Sentence-BERT 기반 모델을 사용했으며 Pretrained Model로 huggingface의 Huffon/sentence-klue-roberta-base 모델을 사용했습니다. 이 모델은 RoBERTa base 문장 임베딩 모델입니다. 모델을 선정하는데에 있어서 데이터 적합성, 의미론적 접근, 모델 크기, 개발 용이성을 기준으로 모델을 선정했습니다.

[Fine-Tuning 참고자료](https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-SBERT-%ED%95%99%EC%8A%B5-%EB%B0%A9%EB%B2%95#21-sts-%EB%8B%A8%EC%9D%BC-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-fine-tuning)  
[sentence-klue-roberta-base](https://huggingface.co/Huffon/sentence-klue-roberta-base)  
[KoSentenceBERT-SKT](https://github.com/BM-K/KoSentenceBERT-SKT)  

- 데이터 적합성: 한글 데이터를 다루기에 한글 언어에 맞춰진 Pretrained Model 사용을 고려했습니다. KLUE Corpus 데이터를 바탕으로 각 단어의 Contexual 의미를 학습했다는 점에서 프로젝트 목표에 적합하다고 판단했습니다.
- 의미론적 접근 : huggingface의 STS 다중어 모델(multi)을 고려하여 inference 실험해 본 결과 각 문장의 의미보다는 자모음이 얼마나 일치하는지에 크게 변동했습니다. Token화가 형태소 기준이 아닌 자모음이라는 언어기호 기준으로 되는 결과를 확인하여, 이처럼 의미를 파악하지 않고 단순 자모음 언어기호의 유사성으로 판단하는 모델은 '학습 데이터 셋을 사용하여 의미적 텍스트 유사도 모델을 훈련한다'는 프로젝트 목표에 맞지 않다고 판단했습니다.
- 개발 용이성: 개발하는 방법이 튜토리얼로 잘 구성되어있는 개발 가능한 모델을 선정했습니다. 본 모델의 경우 Usage를 명시하여 모델 작동을 직관적으로 이해할 수 있습니다.
- 모델 크기: 팀이 현재 사용 가능한 자원에서 충분히 돌아갈 수 있는 모델인지 고려했습니다. API 서빙을 고려하여 빠르게 Inference할 수 있도록 가벼운 모델을 선정하였습니다. API 응답 속도를 고려하여 앙상블 기법을 사용하지 않았습니다.


### Sentence Bert

Sentence Bert는 Computational Cost가 크면서 성능도 기존 Embedding 보다 낮게 나타나는 BERT의 Sentence Embedding 방식을 보완하기 위해 고안 되였으며 다음과 같은 특징을 가집니다.

#### Siames Neural Network

- Siames Network는 같은 가중치를 공유하는 네트워크 구조로 이 구조 하에서 두 문장의 Embedding이 각각 따로 계산됩니다.
- 이후 각 Embedding Vector에 대해 Cosine Similarity 나 Manhattan Distance 를 계산해 유사도를 산출합니다.
- 학습 시에는 문장의 pair가 같은 label이라면 유사도가 가까워지도록, 다르다면 멀어지도록 가중치가 학습됩니다.

#### Computational Efficiency

- BERT는 pair단위로 한번에 transformer encoder에 받아서 inference하는  cross encoder 방식을 사용하기에 Computational Cost가 커서 Serving환경에서 쓰기 어렵다는 문제가 있습니다.

- Siames Network 하에서는 입력을 pair로 받는 것이 아니라 각각의 문장에 대한 Embedding을 생성할 수 있기때문에 BERT로 65시간 걸리던 작업을 5초만에 끝내는 Computatinal Efficient한 모습을 보입니다.  


### Training 

- 데이터 소스의 출처가 다양하기에 가능한 모든 단어의 의미를 임베딩에 반영하고자 Mean Pooling 방식을 사용하였습니다.
- objective function으로는 Cosine Similarity 기반 MSE loss를 사욯하였습니다.
- Optimizer로는 AdamW를 사용하였습니다.
- early train sample로 인한 편향된 학습을 막기 위해 linear learing rate scheduler에서의 warmup step을 학습데이터의 10% 로 설정하였습니다.


## Param Tuning
- 파라미터 튜닝은 Weight & Bias (Wandb)를 사용하였습니다. Wandb의 Sweep 기능을 활용하면 하이퍼 파라미터의 범위를 지정하고 실험해볼 수 있습니다. 또한, 여러가지 하이퍼 파라미터에 따른 성능 추이 그래프를 직접 시각화하여 제공하기 때문에 학습이 잘 이루어지고 있는지, 최적의 파라미터 조합은 무엇인지 직관적으로 확인할 수 있습니다.
- 학습 시간을 고려하여 서치 방식은 Random, Grid, Bayesian 중에 Random Search 방식으로 진행했습니다.
- 조절한 파라미터: batch_size, optimizer_parameters_learning_rate, weight_decay
- learning rate는 SentenceBERT Paper에서 추천하는 값으로 선정했습니다.
- 과적합 방지와 학습 효율성을 위해 1000 step 동안 loss의 감소가 없다면 학습을 멈추는 early terminate를 진행했습니다.
    ```
  sweep_config = {
      'method': 'random'
      }
  
  parameters_dict = {
      # optimizer는 adamw로 고정
      'optimizer_params': {
          'values' : [
                {'lr':1e-4},
                {'lr':3e-4},
                {'lr':5e-5}]
      },
      'weight_decay':{
          'distribution': 'uniform',
          'min': 0.01,
          'max': 0.1
      },
      'batch_size' :{
          'values': [16,32,64] # 64를 넘어갈 필요는 없음
      }
  }
  
  
  early_terminate = {
      'type': 'hyperband',
      'max_iter': 1000,
  }
  ```
**1. Parallel Coordinates**
- Axes(축)은 조절한 파라미터를 의미하며, Lines(선)은 파라미터 조합 별 단일 실행을 나타냅니다. **최고 성능으로 이어지는 조합**에 초점을 맞춘 그래프입니다.
- batch_size와 weight_decay의 경우 분산이 높고, learning_rate는 분산이 낮습니다. 특히, 최고 성능 모델은 모두 learning_rate가 0.0001 이하입니다. 추후 새로운 파라미터 조합을 만들 때, learing rate는 0.0001 이하의 값으로 설정해야 함을 알 수 있습니다.
  <img width="1000" alt="image" src="https://user-images.githubusercontent.com/43432539/159829078-dd7dd0af-e471-4c5b-a9e2-2b49e554fc62.png">

**2. Parameter importance with respect to loss**
- 파라미터가 loss를 예측하는데 유용한 정도를 나타내는 Importance와, 파라미터와 loss의 양의 상관관계를 나타내는 Correlation 정보를 담고 있습니다.
- loss에 가장 많은 영향을 미치는 파라미터와 높은 상관관계를 미치는 faeture는 optimizer parameter의 learning rate 이며, weigth decay와 batch size는 비슷한 중요도와 상관관계를 보입니다.  
  <img width="1000" alt="image" src="https://user-images.githubusercontent.com/43432539/159828956-2492943e-485e-4e23-be09-231b3f04961c.png">

**3. Change of Loss according to Step**
- 모델이 학습하는 과정에서 감소하는 loss를 나타낸 그래프입니다. 파라미터별로 어떻게 다른지 확인할 수 있습니다.
  - learning_rate  
    lr=0.0003일 경우 학습이 제대로 이루어지지 않는 것을 확인할 수 있습니다. lr=0.00005와 lr=0.0001는 비슷한 학습을 보이지만, 전자의 경우 더 안정적으로 학습되고 있습니다.  
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/43432539/159831307-18e25684-8c37-4874-b026-e997f0ff12ab.png">

  - batch_size  
    learning rate에서 학습이 거의 이루어지지 않는 0.0003를 제외하고 batch size는 어떤 학습을 보이는지 확인할 수 있습니다. 
    batch size=64일 경우 빠르게 loss가 감소하는 모습을 보이며, batch size=16일 경우, 초반에 불안정하게 loss가 감소함을 알 수 있습니다.  
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/43432539/159832137-f20eac31-2d5e-444b-8dd5-4ae2bb615ba4.png">


## Metric
- KLUE-STS Leaderboard  

  <img width="500" alt="image" src="https://user-images.githubusercontent.com/43432539/159667102-ed28d749-c3a6-4e69-a6cd-e54ec5e215a2.png">

- F1 score & Pearson's r

  최종 모델의 F1 Score는 0.84이며, 피어슨 상관계수는 0.885입니다. KLUE-STS의 Leaderboard F1 score 기준 5위입니다.

  <img width="500" alt="image" src="https://user-images.githubusercontent.com/43432539/159833981-64917e40-bf5a-4a6b-af5d-e222dcf7592c.png">  

  <img width="400" alt="image" src="https://user-images.githubusercontent.com/43432539/159834047-0c6f5269-863a-42af-87fb-d452e95cf140.png">  


## Evaluation
두 문장을 넣으면 학습된 sentence-klue-roberta-base 모델을 사용해 문장 유사도값(STS)를 도출합니다. 하단에 Serving Code를 사용하여 API로 결과를 받아볼 수 있습니다.
[sentence-klue-roberta-base inference code](https://colab.research.google.com/drive/1zOMAdzpLdmPsb13bXzsu4EkfsWtsG3rg#scrollTo=Th0xyvQVzJFS)

## Serving
FastAPI 프레임워크를 사용하여 sentence-klue-roberta-base 모델을 서빙했습니다. 본 모델은 두 문장의 유사도를 반환하는 STS(Sentence Textual Similarity) Task 모델입니다. 두 문장이 들어오면 이를 RestAPI로 inference 결과를 반환합니다. 위 코드는 모듈화 되어 있습니다.
Model을 서빙하기위해 Python 기반 백앤드 프레임워크를 고민했습니다. 우수한 공식 문서와 큰 생태계를 갖고 있는 장고 프레임워크와 직관적이고 가벼운 플라스크 프레임워크를 고려했습니다. 고민 끝에 비교적 속도가 빠른 FastAPI 프레임워크를 선택했습니다.

main.py로 두 문장을 넣어 predict_sentences를 실행한 결과입니다. 변수로 주어진 문장(sentence),두 문장의 코사인 유사도 실수값(score), binary 유사도 값(pred), 코드 런타임(runtime)입니다.
![image](https://user-images.githubusercontent.com/66352658/159167342-eb43e751-163f-470e-94c1-49b199336e23.png)

코드를 돌리는 방법, 모듈 설명 등 더 자세한 설명은 해당 Repository를 확인해주시기 바랍니다.
[서빙 코드](https://github.com/seawavve/STS_serving) 


## Future work
- Baseline 설정
  원본 데이터의 학습 없이 '원본 + Back Translation(BT) + Easy Data Augmentaton(EDA)'의 데이터로 학습을 진행했습니다. 파라미터 튜닝 이후 데이터별 성능 비교를 위한 baseline 모델이 필요함을 깨달았고, 이후 '원본', '원본+ BT'의 데이터로 학습을 진행했습니다. 전자의 '원본' 데이터로 학습한 모델의 경우 학습 과정에서 data leakage가 발생했고, 최종적으로 사용한 모델은 '원본 + BT' 데이터와 '원본 + BT + EDA' 데이터로 학습시킨 모델입니다. 추후 baseline이 될 수 있는 원본 데이터로 학습해야 할 필요가 있습니다.

- Early stopping
  wandb의 early_terminate를 사용하였으나, 학습 과정에서 loss는 지속적으로 감소하는게 아니라 증감을 반복했기 때문에 early stopping이 올바르게 진행되지 않았습니다. wandb의 기능을 사용하는 것이 아닌, model 자체에서 early stopping을 할 수 있는 방안을 마련하여 보완한다면 모델의 과적합을 피하고 학습의 효율성을 높일 수 있다고 생각합니다.


# References

## STS

### EDA

- [EDA 원논문](https://arxiv.org/abs/1901.11196)
- [EDA 논문 설명](https://catsirup.github.io/ai/2020/04/21/nlp_data_argumentation.html)
- [EDA 구현 참조](https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py)
= [참조 가능한 eda api](https://github.com/makcedward/nlpaug/tree/master/nlpaug)
- [모두의 말뭉치](https://corpus.korean.go.kr/main.do)

### model

급할 경우 sentence bert 관련으로는 위의 2개 링크만 보셔도 됩니다.
- [sentence bert](https://roomylee.github.io/sentence-bert/)
- [sentence bert 구현](https://velog.io/@jaehyeong/Basic-NLP-sentence-transformers-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-SBERT-%ED%95%99%EC%8A%B5-%EB%B0%A9%EB%B2%95#21-sts-%EB%8B%A8%EC%9D%BC-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-fine-tuning)

- [sentence bert paper](https://arxiv.org/abs/1908.10084)
- [sentence bert 설명](https://velog.io/@ysn003/%EB%85%BC%EB%AC%B8-Sentence-BERT-Sentence-Embeddings-using-Siamese-BERT-Networks)

### Cosine Similarity

- [코사인유사도 개념](https://leimao.github.io/blog/Cosine-Similarity-VS-Pearson-Correlation-Coefficient/#:~:text=The%20two%20quantities%20represent%20two,two%20jointly%20distributed%20random%20variables.)
- [유사도 추출관련 주요논문](https://aclanthology.org/N19-1100/)

## Framework

### Wandb

- [Pytorch lightning + Wandb](https://wandb.ai/wandb_fc/korean/reports/Weights-Biases-Pytorch-Lightning---VmlldzozNzAxOTg)
- [pytorch lightning + Wandb colab](https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW)
- [hyperparameter 값 설정 관련](https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU)
- [early stopping 관련 hyperband](https://homes.cs.washington.edu/~jamieson/hyperband.html)

### FastAPI



















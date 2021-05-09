## Get Data

- $git clone https://github.com/songys/Chatbot_data.git

  (Chatbot_data by songys, MIT License)

- para_kqc_sim_data.txt

  (paraKQC by warnikchow, Creative Commons Attribution Share Alike 4.0 International License)
 
 - python get_CLS.py

<br>

## Training
 
- python train_model.py

  (train paraphrase detection task using para_kqc_sim_data.txt)

<br>

## Run Chatbot

__before running, you must execute get_CLS.py and train_model.py__ 

- python main.py

<br>

## Chatbot process

1. input sentence
2. get input sentence`s CLS token using pretrained BERT model
3. calculate cosine similarity of input sentence and CLS tokens in Chatbot_data
4. get Chatbot_data`s indexs that have top n similarity
5. detect input sentence and top n data are similar (using trained model in train_model.py)
6. at least one question is similar, answer will be question`s answer that has the highest confidence
7. all question is not similar, answer  "잘 모르겠어요."

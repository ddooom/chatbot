from glob import glob

import pandas as pd
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification


def get_cls_token(model, tokenizer, question, device):
    model.eval()
    tokenized_sent = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=32
    ).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )
    logits = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()
    return logits


def calculate_cosine_similarity(a,b):
    numerator = np.dot(a,b.T)
    a_norm = np.sqrt(np.sum(a * a))
    b_norm = np.sqrt(np.sum(b * b, axis=-1))

    denominator = a_norm * b_norm
    return numerator/denominator


def return_top_n_idx(model, tokenizer, question, chatbot_question_vecs, n, device):
    question_vector = get_cls_token(model, tokenizer, question, device)
    sentence_similarity = {}
    for idx, vec in enumerate(chatbot_question_vecs):
        similarity = calculate_cosine_similarity(question_vector, vec)
        sentence_similarity[idx] = similarity
    
    sorted_sim = sorted(sentence_similarity.items(), key=lambda x: x[1], reverse=True)
    return sorted_sim[0:n]


def sentences_predict(model, tokenizer, sen1, sen2, device):
    model.eval()
    tokenized_sent = tokenizer(
            sen1,
            sen2,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=64
    )
    
    tokenized_sent.to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )

    logits = outputs[0]
    logits = logits.detach().cpu().numpy().squeeze()
    result = np.argmax(logits) # 0 : not similar, 1: similar
    return result, logits[result]


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n = 5

    data = pd.read_csv("./Chatbot_data/ChatbotData .csv")
    chatbot_question_vecs = pd.read_csv("./CLS_tokens.csv", header=None)
    chatbot_question_vecs = np.array(chatbot_question_vecs)

    chatbot_questions = data['Q'].values
    chatbot_answers = data['A'].values

    # model for get CLS tokens
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cls_model = AutoModel.from_pretrained(MODEL_NAME)
    cls_model.to(device)

    # model for classify similarity
    MODEL_NAME = glob("./results/*")[0]
    classifier_model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    classifier_model.to(device)

    while 1:
        print("* ""종료"" 입력시 종료")
        input_question = input("나 : ")

        if input_question == '종료':
            break

        top_n_idx = return_top_n_idx(cls_model, tokenizer, input_question, chatbot_question_vecs, n, device)

        answer_list = {}
        answer_availble = False
        for idx in top_n_idx:
            answer = chatbot_answers[idx[0]]
            question = chatbot_questions[idx[0]]
            predict = sentences_predict(classifier_model, tokenizer, input_question, question, device)
            if predict[0] == 1:
                answer_list[idx[0]] = predict[1]
                answer_availble = True

        if answer_availble:
            best_answer_idx = max(answer_list.items(), key=lambda x:x[1])[0]
            print(chatbot_answers[best_answer_idx])
        else:
            print('잘 모르겠어요.')
        print()

    

    
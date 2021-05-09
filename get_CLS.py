from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def get_cls_token(model, question):
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


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data = pd.read_csv(".\Chatbot_data\ChatbotData .csv")

    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

    chatbot_questions = data['Q'].values
    chatbot_answers = data['A'].values

    print("* creating CLS token file...")
    chatbot_question_vecs = []
    for question in tqdm(chatbot_questions):
        chatbot_question_vecs.append(get_cls_token(model, question).squeeze())
    
    chatbot_question_vecs = pd.DataFrame(chatbot_question_vecs)
    chatbot_question_vecs.to_csv(".\CLS_tokens.csv", header=False, index=False)
    print(" Done.")
    
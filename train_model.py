import torch
import random
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,  BertConfig


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data processing
    data = open('./para_kqc_sim_data.txt', 'r', encoding='utf-8')
    lines = data.readlines()

    random.shuffle(lines)
    train = {'sen1':[], 'sen2':[], 'label':[]}
    for i, line in enumerate(lines):
        if i < len(lines) * 0.8:
            line = line.strip()
            train['sen1'].append(line.split('\t')[0])
            train['sen2'].append(line.split('\t')[1])
            train['label'].append(int(line.split('\t')[2]))

    print('* Data sample')
    for i in range(3):
        print(train['sen1'][i], train['sen2'][i], train['label'][i])
    print(f'\n* Number of data : {len(lines)} \n')

    # training
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train = tokenizer(
        list(train['sen1']),
        list(train['sen2']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=64
        )

    print('* Tokenized data sample')
    print(tokenized_train[0].tokens)
    print(tokenized_train[0].ids)
    print(tokenized_train[0].attention_mask)

    train_dataset = MyDataset(tokenized_train, train['label'])

    training_args = TrainingArguments(
        output_dir='./results',         
        num_train_epochs=3,            
        per_device_train_batch_size=8,    
        logging_dir='./logs',            
        logging_steps=1000,
        save_total_limit=1,
    )

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) 
    model.to(device)

    trainer = Trainer(
        model=model,                        
        args=training_args,                  
        train_dataset=train_dataset,         
    )

    trainer.train()
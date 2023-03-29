# imports
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import json
import os
from os.path import exists

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import math
import random
import re
import argparse

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv


class SongLyrics(Dataset):
    
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024, df=None):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lyrics = []

        for row in df['Lyric']:
            self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))
                
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)
        
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]

#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None
def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=20, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):

    acc_steps = 100
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=1.,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
                generated_list.append(output_text)
                
    return generated_list

#Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data):
    generated_lyrics = []
    for i in range(len(test_data)):
        x = generate(model.to('cpu'), tokenizer, test_data['Lyric'][i], entry_count=1)
        generated_lyrics.append(x)
    return generated_lyrics
     
    
def main():
    torch.cuda.empty_cache()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
#    suggestions = {}
#    try:
#        with open('./nofinetune_embeddings.json', 'r') as f:
#            suggestions = json.load(f)
#        print(suggestions.keys())
#    except:
#        print('FILE NOT FOUND')
#    with open('./queryindex.json') as json_file:
#        queryindex = json.load(json_file)
    for filename in os.listdir("./prevdoc"):
        if filename.endswith(".csv"):
            user = filename.replace('.csv','')
#            if user in suggestions.keys():
#                continue
#            suggestions[user] = []
            ratio = 0.7
            if user in ['D43D7EC3E0C2']:
                ratio = 0.85
            print('--------------',user,ratio,'----------')
 #           allindex = queryindex[filename.replace('.csv','')]
 #           splitindex = allindex[int(len(allindex)*ratio)]
 #           pred_index = allindex[int(len(allindex)*ratio):]
            df = pd.read_csv("./prevdoc/"+filename)
            df["Lyric"] = df[["source", "target"]].apply(". ".join, axis=1)
            df = df[df['Lyric'].apply(lambda x: len(x.split(' ')) < 350)]
            print(len(df["Lyric"]))
            
            #Create a very small test set to compare generated text with the reality
            test_set = df.sample(n = 100)
            df = df.loc[~df.index.isin(test_set.index)]

            #Reset the indexes
            test_set = test_set.reset_index()
            df = df.reset_index()
            print(len(df["Lyric"]))
            dataset = SongLyrics(df['Lyric'], truncate=True, gpt2_type="gpt2",df=df)

            
            #For the test set only, keep last 20 words in a new column, then remove them from original column
            test_set['True_end_lyrics'] = test_set['Lyric'].str.split().str[-20:].apply(' '.join)
            test_set['Lyric'] = test_set['Lyric'].str.split().str[:-20].apply(' '.join)
            
            #Train the model on the specific data we have
            finetunemodel = train(dataset, model, tokenizer)
            
            #Save the model to a pkl or something so it can be reused later on
            torch.save(model,  './checkpoint_files_2/'+user+'_embeddings.pt')
            
            generated_lyrics = text_generation(test_set)
            
            #Loop to keep only generated text and add it as a new column in the dataframe
            my_generations=[]

            for i in range(len(generated_lyrics)):
                a = test_set['Lyric'][i].split()[-30:] #Get the matching string we want (30 words)
                b = ' '.join(a)
                c = ' '.join(generated_lyrics[i]) #Get all that comes after the matching string
                my_generations.append(c.split(b)[-1])

            test_set['Generated_lyrics'] = my_generations
            
            for i in range(len(test_set)):
                print(test_set['Generated_lyrics'][i])
        break

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!    

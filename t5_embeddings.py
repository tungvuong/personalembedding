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
#from transformers import T5Tokenizer, T5ForConditionalGeneration
#BeIRtokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
#BeIRmodel = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
#BeIRmodel.eval()

#import gensim
#from gensim.models import Word2Vec
#import nltk
#from nltk import ngrams
#import itertools
#from nltk.corpus import stopwords
#from nltk.tokenize import sent_tokenize, word_tokenize
#import scipy
#from scipy import spatial
#from nltk.tokenize.toktok import ToktokTokenizer
#import re
#tokenizer = ToktokTokenizer()
#stopword_list = nltk.corpus.stopwords.words('english')
#w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('/wrk-vakka/users/vuong/music/GoogleNews-vectors-negative300.bin', binary=True)

class LitModel(pl.LightningModule):
  # Instantiate the model
  def __init__(self, learning_rate, tokenizer, model, hparams):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model
    self.learning_rate = learning_rate
    # self.freeze_encoder = freeze_encoder
    # self.freeze_embeds_ = freeze_embeds
    self.hparams = hparams

    if self.hparams.freeze_encoder:
      freeze_params(self.model.get_encoder())

    if self.hparams.freeze_embeds:
      self.freeze_embeds()

  def freeze_params(self, model):
    for param in model.parameters():
      param.requires_grad = False
  
  def freeze_embeds(self):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    try:
      freeze_params(self.model.model.shared)
      for d in [self.model.encoder, self.model.decoder]:
        freeze_params(d.embed_positions)
        freeze_params(d.embed_tokens)
    except AttributeError:
      self.freeze_params(self.model.shared)
      for d in [self.model.encoder, self.model.decoder]:
        self.freeze_params(d.embed_tokens)

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx):
    # Load the data into variables
    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]
    # Shift the decoder tokens right (but NOT the tgt_ids)
    decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]
    # Create the loss function
    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    # Calculate the loss on the un-shifted tokens
    loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

    return {'loss':loss}

  def validation_step(self, batch, batch_idx):

    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]

    decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)
    
    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]

    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

    return {'loss': val_loss}
  
  # Method that generates text using the BartForConditionalGeneration's generate() method
  def generate_text(self, text, eval_beams, early_stopping = False, max_len = 500):
    ''' Function to generate text '''
    generated_ids = self.model.generate(
        text["input_ids"],
        attention_mask=text["attention_mask"],
        use_cache=True,
        decoder_start_token_id = self.tokenizer.pad_token_id,
        num_beams= eval_beams,
        max_length = max_len,
        min_length = 50,
        early_stopping = early_stopping,
#        do_sample=True,
#        num_return_sequences=10
    )
    return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

  def generate_terms(self, text, eval_beams):
    ''' Function to generate text '''
    generated_ids = self.model.generate(
        text["input_ids"],
        attention_mask=text["attention_mask"],
        use_cache=True,
        decoder_start_token_id = self.tokenizer.pad_token_id,
        num_beams= eval_beams,
#        do_sample=True,
#        num_return_sequences=10
    )
    for i, sample_output in enumerate(generated_ids):
        print("{}: {}".format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)))
    return []
    #return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

def freeze_params(model):
  ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
  for layer in model.parameters():
    layer.requires_grade = False



# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
  def __init__(self, tokenizer, data_file, batch_size, num_examples = 20000):
    super().__init__()
    self.tokenizer = tokenizer
    self.data_file = data_file
    self.batch_size = batch_size
    self.num_examples = num_examples
  
  # Loads and splits the data into training, validation and test sets with a 60/20/20 split
  def prepare_data(self):
    self.data = pd.read_csv(self.data_file)[:self.num_examples]
    self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [int(.6*len(self.data)), int(.8*len(self.data))])

  # encode the sentences using the tokenizer  
  def setup(self, stage):
    self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
    self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
    self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])

  # Load the training, validation and test sets in Pytorch Dataset objects
  def train_dataloader(self):
    dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
    train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
    return train_data

  def val_dataloader(self):
    dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels']) 
    val_data = DataLoader(dataset, batch_size = self.batch_size)                       
    return val_data

  def test_dataloader(self):
    dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels']) 
    test_data = DataLoader(dataset, batch_size = self.batch_size)                   
    return test_data


# Create the hparams dictionary to pass in the model
# I realise that this isn't really how this is meant to be used, but having this here reminds me that I can edit it when I need
hparams = argparse.Namespace()

hparams.freeze_encoder = True
hparams.freeze_embeds = True
hparams.eval_beams = 4


def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=300, pad_to_max_length=True, return_tensors="pt"):
  ''' Function that tokenizes a sentence 
      Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
      Returns: Dictionary with keys: input_ids, attention_mask, target_ids
  '''

  input_ids = []
  attention_masks = []
  target_ids = []
  tokenized_sentences = {}

  for sentence in source_sentences:
    encoded_dict = tokenizer(
          sentence.lower(),
          max_length=max_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
#          add_prefix_space = True
      )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim = 0)
  attention_masks = torch.cat(attention_masks, dim = 0)

  for sentence in target_sentences:
    encoded_dict = tokenizer(
          sentence.lower(),
          max_length=max_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
#          add_prefix_space = True
      )
    # Shift the target ids to the right
    # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
    target_ids.append(encoded_dict['input_ids'])

  target_ids = torch.cat(target_ids, dim = 0)
  

  batch = {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
      "labels": target_ids,
  }

  return batch


def noise_sentence(sentence_, percent_words, replacement_token = "<mask>"):
  '''
  Function that noises a sentence by adding <mask> tokens
  Args: sentence - the sentence to noise
        percent_words - the percent of words to replace with <mask> tokens; the number is rounded up using math.ceil
  Returns a noised sentence
  '''
  # Create a list item and copy
  sentence_ = sentence_.split(' ')
  sentence = sentence_.copy()
  
  num_words = math.ceil(len(sentence) * percent_words)
  
  # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
  # that word is often a rhyming word and plays an important role in song construction
  sample_tokens = set(np.arange(0, np.maximum(1, len(sentence)-1)))
  
  words_to_noise = random.sample(sample_tokens, num_words)
  
  # Swap out words, but not full stops
  for pos in words_to_noise:
      if sentence[pos] != '.':
          sentence[pos] = replacement_token
  
  # Remove redundant spaces
  sentence = re.sub(r' {2,5}', ' ', ' '.join(sentence))
  
  # Combine concurrent <mask> tokens into a single token; this just does two rounds of this; more could be done
  sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
  sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
  return sentence

def find_best_epoch(ckpt_folder):
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    ckpt_files = os.listdir(ckpt_folder)  # list of strings
    epochs = [int(filename.split('step=')[-1].split('.')[0]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    best_epoch = max(epochs)
    for filename in ckpt_files:
        if str('{}.ckpt'.format(best_epoch)) in filename:
            return filename
    return best_epoch

# Load the model
#from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig

#tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=False, bos_token="<s>", eos_token="</s>")

#bart_model = BartForConditionalGeneration.from_pretrained(
#    "facebook/bart-base")

from transformers import T5ForConditionalGeneration, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")
bart_model = T5ForConditionalGeneration.from_pretrained("t5-base")

def generate_lyrics(seed_line, num_lines, model_, noise_percent = 0.25, multiple_lines = False, max_line_history = 3):
  ''' Function that generates lyrics based on previously generated lyrics 
      Args: seed_line - a line to start off the machine
            num_lines - the number of lines to generate
            model_ - the model used to generate the text
            multiple_lines - whether the model generates based on multiple previous lines or just the past line
            max_line_history - the maximum number of previous lines used in the current input
      Returns a list with num_lines of rap lines
  '''
  # Put the model on eval mode
  model_.to(torch.device('cpu'))
  model_.eval()
  lyrics = []
  lyrics.append(seed_line)
  seed_line = seed_line
  prompt_line_tokens = tokenizer(seed_line.lower(), max_length = 300, return_tensors = "pt", truncation = True)
#  print(prompt_line_tokens)
  line = model_.generate_text(prompt_line_tokens, eval_beams = 4)
  print('pred',list(set(line)))
  return list(set(line))

def generate_termlyrics(seed_line, num_lines, model_, noise_percent = 0.25, multiple_lines = False, max_line_history = 3):
  model_.to(torch.device('cpu'))
  model_.eval()
  lyrics = []
  lyrics.append(seed_line)
  seed_line = seed_line
  prompt_line_tokens = tokenizer(seed_line.lower(), max_length = 300, return_tensors = "pt", truncation = True)
  print(prompt_line_tokens)
  line = model_.generate_terms(prompt_line_tokens, eval_beams = 4)
  print('pred',line)
  return seed_line

def data_clean(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',' '.join(text))
    tokens = [token.strip() for token in text.split()]
    filtered = [token for token in tokens if token.lower() not in stopword_list]
    filtered = ' '.join(filtered)
    return filtered
def embeddings(word):
#    print(word)
    if word in w2vmodel.key_to_index:
        return w2vmodel.get_vector(word)
    else:
        return np.zeros(300)

def get_sim(query_embedding, average_vec):
    sim = [(1 - scipy.spatial.distance.cosine(query_embedding, average_vec))]
    return sim

def rankings(query,out_dict):
    query_words = (np.mean(np.array([embeddings(x) for x in nltk.word_tokenize(query.lower())], dtype=float), axis=0))
    rank = []
    for k, v in out_dict.items():
        try:
            rank.append((k, get_sim(query_words, v)))
        except:
            pass
    rank = sorted(rank, key=lambda t: t[1], reverse=True)
    print("Ranked documents: ")
    return rank

def main():
    torch.cuda.empty_cache()
    suggestions = {}
    try:
        with open('./t5.json', 'r') as f:
            suggestions = json.load(f)
        print(suggestions.keys())
    except:
        print('FILE NOT FOUND')
    with open('./queryindex.json') as json_file:
        queryindex = json.load(json_file)
    for filename in os.listdir("./prevdoc"):
        if filename.endswith(".csv"):
            user = filename.replace('.csv','')
            if user in suggestions.keys():
                continue
            suggestions[user] = []
            ratio = 0.7
            if user in ['D43D7EC3E0C2']:
                ratio = 0.85
            print('--------------',user,ratio,'----------')
            allindex = queryindex[filename.replace('.csv','')]
            splitindex = allindex[int(len(allindex)*ratio)]
            pred_index = allindex[int(len(allindex)*ratio):]
            # Load the data into the model for training
            summary_data = SummaryDataModule(tokenizer, './prevdoc/'+filename,
                                             batch_size = 14, num_examples = splitindex)

            model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)

            ckpt_dir = './checkpoint_files_2/'+user+'_t5'
            checkpoint = ModelCheckpoint(ckpt_dir)
            if exists(ckpt_dir):
                best_epoch = find_best_epoch(ckpt_dir)
                print(ckpt_dir+'/'+best_epoch)
                trainer = pl.Trainer(gpus = 4,
                                 max_epochs = 20,
                                 min_epochs = 20,
                                 auto_lr_find = False,
                                 resume_from_checkpoint = ckpt_dir+'/'+best_epoch,
                                 progress_bar_refresh_rate = 10)
            else:
                trainer = pl.Trainer(gpus = 4,
                                 max_epochs = 20,
                                 min_epochs = 20,
                                 auto_lr_find = False,
                                 checkpoint_callback = checkpoint,
                                 progress_bar_refresh_rate = 10)

            # Fit the instantiated model to the data
            trainer.fit(model, summary_data)
            pred_df = pd.read_csv('./prevdoc/'+filename)[splitindex:]
            for index, row in pred_df.iterrows():
                if (index < pred_index[0]):
                    continue
                print()
                print(index+2)
                print('target',row['target'][:250])
                print('source',row['source'][:250])
                print('query: ',row['title'])
                pred_target = generate_lyrics(seed_line = row['source'], num_lines = 2, model_ = model,
                                       noise_percent = 0.25, multiple_lines = True, max_line_history = 2)
                suggestions[user].append([row['title'],row['target'],pred_target,row['source'],index+2])
        with open('./t5.json', 'w') as outfile:
            json.dump(suggestions, outfile)

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!    

#coding=utf-8
import os
from tqdm import tqdm
import logging
from typing import List, Union
import torch
from functools import partial
from transformers import MBart50TokenizerFast,BertTokenizer,GPT2Tokenizer
from torch.utils.data import Dataset
from torch.utils.data import Dataset,RandomSampler, SequentialSampler,DataLoader
from torch.utils.data.distributed import DistributedSampler


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

Tokenizers={"cpt":BertTokenizer,#encoder-decoder
            "bert":BertTokenizer,
            "gpt2":BertTokenizer,
            "bart":BertTokenizer}

class DialogExample(object):
    """
    Example
    """
    def __init__(self, qid: int, history: List[str], target:str):
        self.qid = qid
        self.history = history
        self.target = target

    def __repr__(self):
        s = f"qid: {self.qid}, history: {self.history}, target: {self.target}"
        return s

    def __str__(self):
        return self.__repr__()

class DialogFeatures(object):
    def __init__(self,qid:int, history:List[List[int]],target:List[int]) -> None:
        self.qid = qid
        self.history = history
        self.target = target
    

class DatasetWraper(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class DatasetInstance(object):
    def __init__(self,config) -> None:
        self.config = config
        self.load_tokenizer()
    def update_config(self,config):
        self.config=config
    def load_tokenizer(self):
        if self.config.model =="gpt2":
            self.tokenizer = Tokenizers[self.config.model].from_pretrained(self.config.tokenizer_name if self.config.tokenizer_name else self.config.model_name_or_path,
                                                                        do_lower_case=self.config.do_lower_case, # merges_file=args.merges_file,
                                                                        padding_side='left',
                                                                        cache_dir=self.config.cache_dir if self.config.cache_dir else None)
        else:
            self.tokenizer = Tokenizers[self.config.model].from_pretrained(self.config.tokenizer_name if self.config.tokenizer_name else self.config.model_name_or_path,
                                                                        do_lower_case=self.config.do_lower_case, # merges_file=args.merges_file,
                                                                        cache_dir=self.config.cache_dir if self.config.cache_dir else None)
        if isinstance(self.tokenizer,BertTokenizer):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
        elif isinstance(self.tokenizer, GPT2Tokenizer):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("load the {} tokenizer from path {}".format(self.config.model_name_or_path,self.config.model_name_or_path))
    @property
    def BOS_ID(self):
        return self.tokenizer.bos_token_id
        # return self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
    @property
    def EOS_ID(self):
        return self.tokenizer.eos_token_id
        # return self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
    @property
    def PAD_ID(self):
        return self.tokenizer.pad_token_id
        # return self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
    @property
    def vocab_size(self):
        return self.tokenizer
    def remove_special_token(self,inputs:List[str]):
            inputs_len = len(inputs)
            for i in range(inputs_len-1,0,-1):
                if inputs[i].startswith("##"):
                    inputs[i-1] = inputs[i-1] + inputs[i].replace("##","")
                    inputs[i]=[]
            inputs = [item for item in inputs if len(item)>0]
            return inputs
    def load_examples(self,fileName):
        examples,references = [],[]
        refName = os.path.join(self.config.process_path,self.config.corpus,fileName.split(".")[0]+".ref")
        fileName = os.path.join(self.config.data_dir,self.config.corpus,fileName)
        with open(fileName,'r',encoding="utf-8") as f:
            for qid,line in tqdm(enumerate(f)):
                line=line.rstrip()
                utterances = line.split("\t")
                utterances = [utt for utt in utterances if len(utt)> 0]
                if len(utterances)<2:continue
                examples.append(DialogExample(
                    qid=qid,
                    history=utterances[:-1],
                    target=utterances[-1]))
                if "train" not in fileName:
                    references.append(utterances[-1])
        if len(references) >0:
            with open(refName,"w",encoding="utf-8") as f:
                for answer in references:
                    str_answer = " ".join(self.remove_special_token(self.tokenizer.tokenize(answer)))
                    f.write(str_answer.rstrip()+"\n")
        return examples
    def convert_examples_features(self,fileName,outputFile,file_type="train"):
        outputFile = os.path.join(self.config.process_path,self.config.corpus,outputFile)
        features = []
        examples = self.load_examples(fileName)
        for example in tqdm(examples):
            qid = example.qid
            history_ids = self.tokenizer.encode(" ".join(example.history),add_special_tokens=False)
            target_ids = self.tokenizer.encode(example.target, add_special_tokens=False)
            if self.config.model =="gpt2":
                max_sequence = self.config.max_utterance_len + self.config.max_decode_length
                if file_type =="train":
                    if len(history_ids) + len(target_ids) > max_sequence-1:
                        while len(history_ids) + len(target_ids) > max_sequence-1:
                            history_ids = history_ids[1:]
                else:
                    if len(history_ids) > self.config.max_utterance_len:
                        while len(history_ids) > self.config.max_utterance_len:
                            history_ids = history_ids[1:]
            elif self.config.model in ["bert","bart","cpt"]:
                if len(history_ids) > self.config.max_utterance_len - 2:
                    history_ids = history_ids[2-self.config.max_utterance_len:]
            features.append(DialogFeatures(qid=qid,
                                            history = history_ids,
                                            target = target_ids))
        torch.save(features,outputFile)
    def decoder_batch_features(self,features, batch_size,training=False):
        def collate_fn(batch_data, tokenizer, max_inp_len, max_tar_len):
            input_ids = []
            for d in batch_data:
                input_ids.append(d.history + d.target)
            model_inputs = tokenizer.batch_encode_plus(
                input_ids,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=max_inp_len,
                padding=True,
                is_split_into_words=True,
                truncation=True,
            )
            return model_inputs
        def create_collate_fn(tokenizer, max_inp_len, max_tar_len):
            f = partial(collate_fn, tokenizer=tokenizer, max_inp_len=max_inp_len, max_tar_len=max_tar_len)
            return f
        FeaturesDataset = DatasetWraper(features)
        if training:
            if self.config.local_rank == -1:
                DatasetSampler = RandomSampler(FeaturesDataset)
            else:
                DatasetSampler = DistributedSampler(FeaturesDataset)
        else:
            DatasetSampler = SequentialSampler(FeaturesDataset)
        collate_fn = create_collate_fn(tokenizer = self.tokenizer,
                                    max_inp_len=self.config.max_utterance_len+self.config.max_decode_length if training else self.config.max_utterance_len,
                                    max_tar_len = self.config.max_decode_length)
        IteratorDataset = DataLoader(FeaturesDataset, 
                                      sampler=DatasetSampler, 
                                      batch_size=batch_size, 
                                      collate_fn=collate_fn)
        return IteratorDataset
    def encoder_decoder_batch_features(self,features, batch_size,training=False):
        def collate_fn(batch_data, tokenizer, max_inp_len, max_tar_len):
            input_ids = []
            target_ids = []
            for d in batch_data:
                input_ids.append(d.history)
                target_ids.append(d.target)
            model_inputs = tokenizer.batch_encode_plus(
                input_ids,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=max_inp_len,
                padding=True,
                is_split_into_words=True,
                truncation=True,
            )
            for keyName,values in  tokenizer.batch_encode_plus(
                target_ids,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=max_tar_len,
                padding=True,
                is_split_into_words=True,
                truncation=True,
            ).items():
                model_inputs["target_"+keyName] = values
            return model_inputs
        def create_collate_fn(tokenizer, max_inp_len, max_tar_len):
            f = partial(collate_fn, tokenizer=tokenizer, max_inp_len=max_inp_len, max_tar_len=max_tar_len)
            return f
        FeaturesDataset = DatasetWraper(features)
        if training:
            if self.config.local_rank == -1:
                DatasetSampler = RandomSampler(FeaturesDataset)
            else:
                DatasetSampler = DistributedSampler(FeaturesDataset)
        else:
            DatasetSampler = SequentialSampler(FeaturesDataset)
        collate_fn = create_collate_fn(tokenizer = self.tokenizer,
                                    max_inp_len=self.config.max_utterance_len,
                                    max_tar_len = self.config.max_decode_length)
        IteratorDataset = DataLoader(FeaturesDataset, 
                                      sampler=DatasetSampler, 
                                      batch_size=batch_size, 
                                      collate_fn=collate_fn)
        return IteratorDataset
            




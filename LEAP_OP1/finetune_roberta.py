import os
import sys
import math
import glob
import time
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

def tokenize_function(examples) : 
    return tokenizer(examples["text"])

def group_texts(examples) : 
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def split(args):
    num_nodes, num_rels = get_total_number(args.dp + args.dn, 'stat.txt')

    quadruple_idx_path = args.dp + args.dn + '/quadruple_idx.txt'
    df = pd.read_csv(quadruple_idx_path, sep='\t',  lineterminator='\n', names=[
                     'source', 'relation', 'target', 'time'])
    # print(df.head())
    # ratio 80% 10% 10%
    # in total 2557 days (0-2556) 
    cut_1 = 1795 #2044
    cut_2 = 2019 #2300
    train_df = df.loc[df['time'] <= cut_1]
    valid_df = df.loc[(df['time'] > cut_1) & (df['time'] <= cut_2)]
    test_df = df.loc[df['time'] > cut_2]
    print("[Train / Valid / Test] Quadruple Shapes")
    print(train_df.shape, valid_df.shape, test_df.shape)
    return int(train_df.shape[0]), int(valid_df.shape[0]), int(test_df.shape[0])

if __name__ == "__main__" : 
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="Dataset Path")
    ap.add_argument("--dn", default="AFG", help="Dataset Name")
    ap.add_argument("--gpu_id", default=0, help="GPU Device ID")
    ap.add_argument("--block_size", default=512, help="Block Size of Chunks for LLM")
    ap.add_argument("--model_size", default="base", help="RoBERTa Base or Large")

    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from datasets import Dataset
    from datasets import DatasetDict
    from transformers import logging
    logging.set_verbosity_error()
    from transformers import Trainer
    from transformers import AutoTokenizer
    from transformers import TrainingArguments
    from transformers import AutoModelForMaskedLM
    from transformers import DataCollatorForLanguageModeling
    
    num_train, num_valid, num_test = split(args)
    df_quintuple = pd.read_hdf(args.dp + args.dn + "/quintuple.h5")
    df_text = pd.DataFrame(df_quintuple.iloc[:, 4])
    print("Text Min Length = %i" % df_text.text.map(lambda x: len(x)).min())
    print("Text Avg Length = %i" % df_text.text.map(lambda x: len(x)).mean())
    print("Text Max Length = %i" % df_text.text.map(lambda x: len(x)).max())

    ########################################
    # # Incremental Testing
    # num_train = 500
    # num_valid = 200
    # num_test = 200
    # df_text = df_text.iloc[:900]
    ########################################

    df_train = df_text.iloc[:num_train]
    slice_valid = num_train + num_valid
    df_valid = df_text.iloc[num_train:slice_valid]
    df_test = df_text.iloc[slice_valid:]
    print("[Train / Valid / Test] Text Shapes")
    print(df_train.shape, df_valid.shape, df_test.shape)
    data = DatasetDict({
        "train": Dataset.from_pandas(df_train), 
        "valid": Dataset.from_pandas(df_valid), 
        "test": Dataset.from_pandas(df_test)
    })

    data = data.remove_columns("__index_level_0__")
    
    print(data)

    block_size = args.block_size
    model_checkpoint = "roberta-" + args.model_size
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast = True)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    
    # my_checkpoint = "./roberta_finetuned/My_Path/" + args.dn
    # model = AutoModelForMaskedLM.from_pretrained(my_checkpoint)
    
    tokenized_data = data.map(tokenize_function, batched = True, num_proc = 4, 
                              remove_columns = ["text"])
    kg_data = tokenized_data.map(group_texts, batched = True, num_proc = 4)
    
    print(kg_data)
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                    mlm_probability=0.15)
    out_path = "./roberta_finetuned/Trainer_Default_Path/" + args.dn
    training_args = TrainingArguments(
        output_dir = out_path, 
        save_total_limit = 1, 
        overwrite_output_dir = True, 
        load_best_model_at_end = True, 
        save_strategy = "epoch", 
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        # num_train_epochs = 1, 
        num_train_epochs = 40, 
        weight_decay = 0.01,
        push_to_hub = False,
        disable_tqdm = False, 
        save_only_model = True, 
        seed = 42, 
        data_seed = 42,
        metric_for_best_model = "eval_loss", 
        greater_is_better = False, 
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=kg_data["train"], 
        eval_dataset=kg_data["valid"],
        # eval_dataset=kg_data["test"], 
        data_collator=data_collator,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    my_path = "./roberta_finetuned/My_Path/" + args.dn
    trainer.save_model(my_path)
    print("Finetuning Complete!")
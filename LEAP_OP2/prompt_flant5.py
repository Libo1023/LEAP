import os
import sys
import time
dataset_name = "AFG"
gpu_id = int(0)
# select_prompt = "original_prompt"
# select_prompt = "zero_shot_prompt"
# select_prompt = "no_text_prompt"
select_prompt = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

data_path = "../data/"
data_name = dataset_name + "/"
print(dataset_name, data_name)

path_quad = data_path + data_name + "quadruple.txt"
path_text = data_path + data_name + "quintuple.h5"

df_quad = pd.read_csv(path_quad, sep='\t', lineterminator='\n', names=['source', 'relation', 'target', 'time'])
df_quintuple = pd.read_hdf(path_text)
df_text = pd.DataFrame(df_quintuple.iloc[:, 4])
data = pd.concat([df_quad, df_text], axis = 1)
data = data.to_numpy()

SAMPLE_ALL = data.shape[0]
if data_name == "AFG/":
    NUM_TRAIN = 212540
    NUM_VALID = 32734
    NUM_TEST = 34585

if data_name == "IND/":
    NUM_TRAIN = 318471
    NUM_VALID = 75439
    NUM_TEST = 85739

if data_name == "RUS/":
    NUM_TRAIN = 275477
    NUM_VALID = 46516
    NUM_TEST = 51371


SAMPLE_EXP = 5

data = np.copy(data[:SAMPLE_ALL, :])

print(data.shape)

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers.utils import logging
logging.set_verbosity(40)

from datasets import Dataset
from datasets import DatasetDict
from transformers import logging
logging.set_verbosity_error()
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling

import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

def build_prompt_1 (examples, query) : 
    prompt = f"""
             I ask you to perform an object prediction task after I provide you with five examples. 
             Each example is a knowledge quintuple containing two entities, a relation, a timestamp, and a brief text summary. 
             Each knowledge quintuple is strictly formatted as (subject entity, relation, object entity, timestamp, text summary). 
             For the object prediction task, you should predict the missing object entity based on the other four available elements. 
             
             Now I give you five examples. 
             
             ## Example 1
             ({examples[0][0]}, {examples[0][1]}, <MISSING OBJECT ENTITY>, {examples[0][3]}, {examples[0][4]}).
             The <MISSING OBJECT ENTITY> is: {examples[0][2]}
             ## Example 2
             ({examples[1][0]}, {examples[1][1]}, <MISSING OBJECT ENTITY>, {examples[1][3]}, {examples[1][4]}).
             The <MISSING OBJECT ENTITY> is: {examples[1][2]}
             ## Example 3
             ({examples[2][0]}, {examples[2][1]}, <MISSING OBJECT ENTITY>, {examples[2][3]}, {examples[2][4]}).
             The <MISSING OBJECT ENTITY> is: {examples[2][2]}
             ## Example 4
             ({examples[3][0]}, {examples[3][1]}, <MISSING OBJECT ENTITY>, {examples[3][3]}, {examples[3][4]}).
             The <MISSING OBJECT ENTITY> is: {examples[3][2]}
             ## Example 5
             ({examples[4][0]}, {examples[4][1]}, <MISSING OBJECT ENTITY>, {examples[4][3]}, {examples[4][4]}).
             The <MISSING OBJECT ENTITY> is: {examples[4][2]}
             
             Now I give you a query: ({query[0]}, {query[1]}, <MISSING OBJECT ENTITY>, {query[3]}, {query[4]}).
             Please predict the missing object entity. 
             You are allowed to predict new object entity which you have never seen in examples. 
             The correct object entity is: 
             """
    return prompt

###################################################################################################################
def build_prompt_zero_shot (query, examples = None) : 
    prompt = f"""
             I ask you to perform an object prediction task on a query quintuple. 
             Each query quintuple is strictly formatted as (subject entity, relation, object entity, timestamp, text summary). 
             For the object prediction task, you should predict the missing object entity based on the other four available elements. 
             
             Now I give you a query: ({query[0]}, {query[1]}, <MISSING OBJECT ENTITY>, {query[3]}, {query[4]}).
             Please predict the missing object entity. 
             You are allowed to predict new object entity which you have never seen before. 
             The correct object entity is: 
             """
    return prompt

def build_prompt_quadruple (examples, query) : 
    prompt = f"""
             I ask you to perform an object prediction task after I provide you with five examples. 
             Each example is a knowledge quadruple containing a subject, a relation, an object, and a timestamp. 
             Each knowledge quadruple is strictly formatted as (subject entity, relation, object entity, timestamp). 
             For the object prediction task, you should predict the missing object entity based on the other three available elements. 
             
             Now I give you five examples. 
             
             ## Example 1
             ({examples[0][0]}, {examples[0][1]}, <MISSING OBJECT ENTITY>, {examples[0][3]}).
             The <MISSING OBJECT ENTITY> is: {examples[0][2]}
             ## Example 2
             ({examples[1][0]}, {examples[1][1]}, <MISSING OBJECT ENTITY>, {examples[1][3]}).
             The <MISSING OBJECT ENTITY> is: {examples[1][2]}
             ## Example 3
             ({examples[2][0]}, {examples[2][1]}, <MISSING OBJECT ENTITY>, {examples[2][3]}).
             The <MISSING OBJECT ENTITY> is: {examples[2][2]}
             ## Example 4
             ({examples[3][0]}, {examples[3][1]}, <MISSING OBJECT ENTITY>, {examples[3][3]}).
             The <MISSING OBJECT ENTITY> is: {examples[3][2]}
             ## Example 5
             ({examples[4][0]}, {examples[4][1]}, <MISSING OBJECT ENTITY>, {examples[4][3]}).
             The <MISSING OBJECT ENTITY> is: {examples[4][2]}
             
             Now I give you a query: ({query[0]}, {query[1]}, <MISSING OBJECT ENTITY>, {query[3]}).
             Please predict the missing object entity. 
             You are allowed to predict new object entity which you have never seen in examples. 
             The correct object entity is: 
             """
    return prompt
###################################################################################################################

select_llm = "flan-t5"

model = None
tokenizer = None

if select_llm == "flan-t5" : 
    # t5_version = "google/flan-t5-small"
    t5_version = "google/flan-t5-base"
    # t5_version = "google/flan-t5-large"
    # t5_version = "google/flan-t5-xl"
    # t5_version = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(t5_version)
    tokenizer.model_max_length = 1024
    
    # model = AutoModelForSeq2SeqLM.from_pretrained(t5_version)
    
    my_checkpoint = "./finetuned_flant5/My_Path/" + data_name
    print("Load Fine-tuned FLAN-T5 Checkpoint ", my_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(my_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

print("Prepare for Training")
q_train = []
a_train = []
for i in range(SAMPLE_EXP, NUM_TRAIN - 1) : 
    history_start = int(i - SAMPLE_EXP)
    history_end = int(i)
    query_index = int(i+1)
    examples = data[history_start:history_end, :]
    query = data[query_index, :]
    curr_prompt = build_prompt_1(examples, query)
    q_train.append(curr_prompt)
    a_train.append(query[2])
print(len(q_train), len(a_train))

print("Prepare for Validation")
q_valid = []
a_valid = []
for i in range(NUM_TRAIN - 1, NUM_TRAIN + NUM_VALID - 1) : 
    history_start = int(i - SAMPLE_EXP)
    history_end = int(i)
    query_index = int(i+1)
    examples = data[history_start:history_end, :]
    query = data[query_index, :]
    curr_prompt = build_prompt_1(examples, query)
    q_valid.append(curr_prompt)
    a_valid.append(query[2])
print(len(q_valid), len(a_valid))

print("Prepare for Testing")
q_test = []
a_test = []
for i in range(NUM_TRAIN + NUM_VALID - 1, SAMPLE_ALL - 1) : 
# for i in range(NUM_TRAIN + NUM_VALID - 1, NUM_TRAIN + NUM_VALID - 1 + 1000) : 
    history_start = int(i - SAMPLE_EXP)
    history_end = int(i)
    query_index = int(i+1)
    examples = data[history_start:history_end, :]
    query = data[query_index, :]
    
    ############################################################
    if select_prompt == "original_prompt" : 
        curr_prompt = build_prompt_1(examples, query)
    if select_prompt == "zero_shot_prompt" : 
        curr_prompt = build_prompt_zero_shot(query = query)
    if select_prompt == "no_text_prompt" : 
        curr_prompt = build_prompt_quadruple(examples, query)
    ############################################################
    
    q_test.append(curr_prompt)
    a_test.append(query[2])
print(len(q_test), len(a_test))

df_qa_train = pd.DataFrame({"question": q_train, "answer": a_train})
df_qa_valid = pd.DataFrame({"question": q_valid, "answer": a_valid})
df_qa_test = pd.DataFrame({"question": q_test, "answer": a_test})
print(df_qa_train.shape, df_qa_valid.shape, df_qa_test.shape)
hf_data = DatasetDict({
    "train": Dataset.from_pandas(df_qa_train), 
    "valid": Dataset.from_pandas(df_qa_valid), 
    "test": Dataset.from_pandas(df_qa_test), 
})

def prepare_data (samples) : 
    input_prompts = [x for x in samples["question"]]
    model_inputs = tokenizer(input_prompts, max_length = 1024, truncation = True)
    labels = tokenizer(text_target = samples["answer"], max_length = 100, truncation = True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_dataset = hf_data.map(prepare_data, batched = True)

nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")
def compute_metrics(eval_preds) : 
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 2
# PER_DEVICE_EVAL_BATCH = 2
PER_DEVICE_EVAL_BATCH = 1
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5

# Set up training arguments
hf_output_dir = "./finetuned_flant5/args_dir/" + data_name
training_args = Seq2SeqTrainingArguments(
    output_dir = hf_output_dir,
    save_total_limit = 1, 
    overwrite_output_dir = True, 
    load_best_model_at_end = True, 
    save_strategy = "epoch", 
    evaluation_strategy = "epoch",
    learning_rate = L_RATE,
    num_train_epochs = NUM_EPOCHS, 
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = PER_DEVICE_EVAL_BATCH,
    weight_decay = WEIGHT_DECAY,
    predict_with_generate = True,
    push_to_hub = False, 
    disable_tqdm = False, 
    save_only_model = True, 
    seed = 42, 
    data_seed = 42, 
    metric_for_best_model = "eval_rougeL", 
    greater_is_better = True, 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["valid"], 
    eval_dataset=tokenized_dataset["test"], 
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

eval_results = trainer.evaluate()
print("Dataset Name:", dataset_name, "Current Prompt:", select_prompt)
print(eval_results)
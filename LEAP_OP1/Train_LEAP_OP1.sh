printf "\n Current Dataset: AFG \n"

printf "\n Fine-tune RoBERTa-base with masked language modeling \n"
python finetune_roberta.py --dn AFG --gpu_id 0

printf "\n Generate and collect text summary embeddings \n"
python helper_generate_sentence_embeddings.py --data_name AFG

printf "\n Build training, validation, and test sets \n"
python build_raw_sets.py --dn AFG

printf "\n Generate knowledge graphs \n"
python icews_generate_structured_data.py --c AFG
python generate_graphs.py --datapath ../data/AFG

printf "\n Train and evaluate LEAP_OP1 \n"
python main.py -d AFG --gpu 0 --alias _LEAP_OP1 --n_epochs 40 --train_history_len 7

printf "\n All Experiments Completed! \n"
printf "\n Current Dataset: AFG \n"

printf "\n Generate and collect quintuple embeddings with pre-trained RoBERTa-large \n"
python generate_quintuple_embeddings.py AFG

printf "\n Build training, validation, and test sets \n"
python build_raw_sets.py --dn AFG

printf "\n Train and evaluate LEAP_MEF \n"
python main_evaluation.py --dataset AFG --gpu 0 --seq-len 7 --batch-size 2 --max-epochs 40 --runs 5

printf "\n All Experiments Completed! \n"
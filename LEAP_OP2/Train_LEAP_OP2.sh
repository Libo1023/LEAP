printf "\n Current Dataset: AFG \n"

printf "\n Train LEAP_OP2 \n"
printf "\n Fine-tune FLAN-T5-base following the question-answering setting \n"
python finetune_flant5.py

# Select a prompt template from "original_prompt", "zero_shot_prompt", and "no_text_prompt"
printf "\n Evaluate LEAP_OP2 \n"
printf "\n Prompt the fine-tuned FLAN-T5-base (model inference) with different templates \n"
python prompt_flant5.py original_prompt

printf "\n All Experiments Completed! \n"
python eval/score/internvl2_score_fiveiqa.py \
    --model_path ../models/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_gvlmiqav0.2-score-train_2bs_5epoch_1e-5lr_mdp3/ \
    --save_name internvl2_lora_gvlmiqav0.2-score-train_2bs_5epoch_1e-5lr_mdp3 

python eval/score/internvl2_score_fiveiqa.py \
    --save_name internvl2_score_val


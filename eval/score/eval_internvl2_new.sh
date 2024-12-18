for dataset in 'UHD-IQA' 'konx'
do
    python eval/score/internvl2_score_konx_and_uhd.py \
        --save_path ./results/q_align/$dataset/internvl2.json \
        --eval_file ../datasets/val_json/$dataset.json \
        --image_folder ../datasets/images/ 
done

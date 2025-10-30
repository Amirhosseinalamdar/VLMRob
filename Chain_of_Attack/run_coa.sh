# change method to coa for coa, and the pgd_steps to 100 for coa
python3 main.py \
  --method coa \
  --num_samples 80 \
  --input_res 224 \
  --clip_encoder "ViT-B/32" \
  --eval_vlms "BLIP, UniDiffuser, LLaVA-7B" \
  --eval_clip_encoders "RN50, ViT-B/32" \
  --image_encoder "attackvlm_default" \
  --alpha 1.0 \
  --p_neg 0.7 \
  --epsilon 8 \
  --pgd_steps 100 \
  --fusion_type add_weight \
  --a_weight_cle 0.5 \
  --a_weight_tgt 0.5 \
  --a_weight_cur 0.9 \
  --speed_up False \
  --update_steps 1 \
  --prefix_length 10 \
  --model_path /home/user01/research/Chain_of_Attack/clip_prefix_model/conceptual_weights.pt \
  --tgt_data_path /home/user01/research/Chain_of_Attack/t2i/t2i_coco/images/generated \
  --cle_file_path /home/user01/research/Chain_of_Attack/Clean_text_generation_minigpt4/MiniGPT-4/outputs2/captions_20251023_220445.jsonl \
  --tgt_file_path /home/user01/research/Chain_of_Attack/t2i/t2i_coco/annotations/mscoco_t2i_captions_train2017_stratified.jsonl \
  --output /home/user01/research/Chain_of_Attack/outputs/coa_koochak_20_testonly \




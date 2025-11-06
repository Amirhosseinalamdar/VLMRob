# change method to coa for coa, and the pgd_steps to 100 for coa
python3 main.py \
  --method "attackvlm" \
  --num_samples 50 \
  --input_res 224 \
  --clip_encoder "ViT-B/32" \
  --eval_vlms "UniDiffuser, ViECap, BLIP" \
  --eval_clip_encoders "RN50, RN101, ViT-B/16, ViT-B/32, ViT-L/14" \
  --image_encoder "blip_caption" \
  --alpha 1.0 \
  --p_neg 0.7 \
  --epsilon 8 \
  --pgd_steps 100 \
  --fusion_type add_weight \
  --a_weight_cle 0.3 \
  --a_weight_tgt 0.3 \
  --a_weight_cur 0.3 \
  --speed_up False \
  --update_steps 1 \
  --prefix_length 10 \
  --multi_target_method each \
  --model_path /home/user01/research/Chain_of_Attack/clip_prefix_model/conceptual_weights.pt \
  --cle_file_path /home/user01/research/Chain_of_Attack/Clean_text_generation_minigpt4/MiniGPT-4/outputs2/captions_20251023_220445.jsonl \
  --output /home/user01/research/Chain_of_Attack/outputs/baaaa_imbabaE \
  --tgt_base "/home/user01/research/Chain_of_Attack/t2i/t2i_coco"
  # --tgt_base "/home/user01/research/Chain_of_Attack/t2i/t2i_coco, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_12345, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_123456, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v00, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v01, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v02, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v03, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v04, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v05, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test_p2/paraphrase_v06, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test_p2/paraphrase_v07" \
  # --tgt_base "/home/user01/research/Chain_of_Attack/t2i/t2i_coco, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_12345, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_123456, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v00, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v01, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v02, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v03, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v04, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test/paraphrase_v05, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test_p2/paraphrase_v06, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test_p2/paraphrase_v07, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test_p2/close_v00, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test_p2/close_v01, /home/user01/research/Chain_of_Attack/t2i/t2i_coco_test_p2/close_v02" \







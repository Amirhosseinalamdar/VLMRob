# csv old pipeline
# python3 run_vlm_eval.py --data_dir /path/to/images --vlm my_vlm --out_dir /path/to/out

# json pipeline
# python3 run_vlm_eval.py --jsonl_input /home/user01/research/Chain_of_Attack/outputs/coa_adv2/annotations.jsonl --vlm SmallCap --out_dir ./unidiff_out



python3 run_vlm_eval.py \
    --jsonl_input /home/user01/research/Chain_of_Attack/outputs/attacvlm/annotations.jsonl \
    --vlm LLaVA-7B \
    --out_dir ./test
# python3 run_vlm_eval.py \
#     --jsonl_input /home/user01/research/Chain_of_Attack/outputs/attacvlm/annotations.jsonl \
#     --vlm SmallCap \
#     --out_dir ./test
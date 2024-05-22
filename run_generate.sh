#!/usr/bin/env bash

source ~/.bashrc
conda activate unsloth_env
#  for model in "allenai/tulu-2-dpo-7b"; do
#      for prompt_template in "input_date" "input_date_today" "input_url" "input_rank" "input_emphasize_url" "input_emphasize_wiki_url"; do
#          for modify_meta_data in 0 1; do
#              if [[ $modify_meta_data == 1 ]]; then
#                  for favored_stance in "yes" "no"; do
#                      python3 main.py \
#                          --model_name "$model" \
#                          --prompt_template "$prompt_template" \
#                          --favored_stance "$favored_stance" \
#                          --modify_meta_data "$modify_meta_data" \
#                          --generation
#                  done
#              else
#                  python3 main.py \
#                      --model_name "$model" \
#                      --prompt_template "$prompt_template" \
#                      --modify_meta_data "$modify_meta_data" \
#                      --generation
#              fi
#          done
#      done
#  done
#  


for model in "dpo_output"; do
    for prompt_template in "input_date" "input_date_today"; do
        for modify_meta_data in 0; do
            if [[ $modify_meta_data == 1 ]]; then
                for favored_stance in "yes" "no"; do
                    python3 main.py \
                        --generation \
                        --model_name "$model" \
                        --prompt_template "$prompt_template" \
                        --favored_stance "$favored_stance" \
                        --modify_meta_data "$modify_meta_data" \
                        --dataset_path data/fake_knowledge_with_evidence_parsed.json \
                        --output_dir results_fake

                done
            else
                python3 main.py \
                    --generation \
                    --model_name "$model" \
                    --prompt_template "$prompt_template" \
                    --modify_meta_data "$modify_meta_data" \
                    --dataset_path data/fake_knowledge_with_evidence_parsed.json \
                    --output_dir results_fake
            fi
        done
    done
done



# for model in "dpo_output"; do
#     for url_modifier in 'cnn_naturalnews_url' 'wiki_wordpress_url'; do
#       for prompt_template in "input_emphasize_url"; do
#         for modify_meta_data in 1; do
#             if [[ $modify_meta_data == 1 ]]; then
#                 for favored_stance in "yes" "no"; do
#                     python3 main.py \
#                         --generation \
#                         --url_modifier "$url_modifier" \
#                         --model_name "$model" \
#                         --prompt_template "$prompt_template" \
#                         --favored_stance "$favored_stance" \
#                         --modify_meta_data "$modify_meta_data" \
#                         --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                         --output_dir results_fake
# 
#                 done
#             else
#                 python3 main.py \
#                     --generation \
#                     --url_modifier "$url_modifier" \
#                     --model_name "$model" \
#                     --prompt_template "$prompt_template" \
#                     --modify_meta_data "$modify_meta_data" \
#                     --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                     --output_dir results_fake
#             fi
#         done
#       done
#     done
# done
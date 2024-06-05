#!/usr/bin/env bash

source ~/.bashrc
# conda activate vllm
conda activate unsloth_env
#  for model in "allenai/tulu-2-dpo-7b"; do
#      for prompt_template in "input_date" "input_date_today" "input_url" "input_rank" "input_emphasize_url" "input_emphasize_wiki_url"; do
#          for modify_meta_data in 0 1; do
#              if [[ $modify_meta_data == 1 ]]; then
#                  for favored_stance in "yes" "no"; do
#                      python3 text_llm.py \
#                          --model_name "$model" \
#                          --prompt_template "$prompt_template" \
#                          --favored_stance "$favored_stance" \
#                          --modify_meta_data "$modify_meta_data" \
#                          --max_tokens 512 \
#                          --generation
#                  done
#              else
#                  python3 text_llm.py \
#                      --model_name "$model" \
#                      --prompt_template "$prompt_template" \
#                      --modify_meta_data "$modify_meta_data" \
#                      --max_tokens 512 \
#                      --generation
#              fi
#          done
#      done
#  done
#  


for model in "meta-llama/Meta-Llama-3-8B-Instruct"; do
    for url_modifier in "cnn_naturalnews_src" "wiki_wordpress_src"; do
      for prompt_template in "input_emphasize_src"; do
        for modify_meta_data in 1; do
            if [[ $modify_meta_data == 1 ]]; then
                for favored_stance in "yes" "no"; do
                    python3 text_llm.py \
                        --generation \
                        --dataset_path data_with_keyword.json \
                        --max_tokens 512 \
                        --url_modifier "$url_modifier" \
                        --model_name "$model" \
                        --prompt_template "$prompt_template" \
                        --favored_stance "$favored_stance" \
                        --modify_meta_data "$modify_meta_data"

                done
            else
                python3 text_llm.py \
                    --generation \
                    --max_tokens 512 \
                    --dataset_path data_with_keyword.json \
                    --url_modifier "$url_modifier" \
                    --model_name "$model" \
                    --prompt_template "$prompt_template" \
                    --modify_meta_data "$modify_meta_data"
            fi
        done
      done
    done
done


for model in "meta-llama/Meta-Llama-3-8B-Instruct"; do
    for url_modifier in "cnn_naturalnews_url" "wiki_wordpress_url"; do
      for prompt_template in "input_emphasize_url"; do
        for modify_meta_data in 1; do
            if [[ $modify_meta_data == 1 ]]; then
                for favored_stance in "yes" "no"; do
                    python3 text_llm.py \
                        --generation \
                         --max_tokens 512 \
                         --dataset_path data_with_keyword.json \
                        --url_modifier "$url_modifier" \
                        --model_name "$model" \
                        --prompt_template "$prompt_template" \
                        --favored_stance "$favored_stance" \
                        --modify_meta_data "$modify_meta_data"

                done
            else
                python3 text_llm.py \
                    --generation \
                    --max_tokens 512 \
                    --url_modifier "$url_modifier" \
                    --dataset_path data_with_keyword.json \
                    --model_name "$model" \
                    --prompt_template "$prompt_template" \
                    --modify_meta_data "$modify_meta_data"
            fi
        done
      done
    done
done



# for model in "gpt-4-turbo-2024-04-09"; do
#     for prompt_template in "input_date" "input_date_today" "input_rank"; do
#         for modify_meta_data in 1 0; do
#             if [[ $modify_meta_data == 1 ]]; then
#                 for favored_stance in "yes" "no"; do
#                     python3 text_llm.py \
#                         --generation \
#                         --model_name "$model" \
#                         --prompt_template "$prompt_template" \
#                         --favored_stance "$favored_stance" \
#                         --modify_meta_data "$modify_meta_data" \
#                         --max_tokens 512 \
#                         --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                         --output_dir results_fake
# 
#                 done
#             else
#                 python3 text_llm.py \
#                     --generation \
#                     --max_tokens 512 \
#                     --model_name "$model" \
#                     --prompt_template "$prompt_template" \
#                     --modify_meta_data "$modify_meta_data" \
#                     --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                     --output_dir results_fake
#             fi
#         done
#     done
# done



#  for model in "meta-llama/Llama-2-13b-chat-hf"; do
#      for url_modifier in "pretty_simple_html"; do
#        for prompt_template in "input_html"; do
#          for modify_meta_data in 1; do
#              if [[ $modify_meta_data == 1 ]]; then
#                  for favored_stance in "yes" "no"; do
#                      python3 text_llm.py \
#                          --generation \
#                           --max_tokens 512 \
#                          --url_modifier "$url_modifier" \
#                          --model_name "$model" \
#                          --prompt_template "$prompt_template" \
#                          --favored_stance "$favored_stance" \
#                          --modify_meta_data "$modify_meta_data" \
#                          --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                          --output_dir results_fake
#  
#                  done
#              else
#                  python3 text_llm.py \
#                      --generation \
#                      --max_tokens 512 \
#                      --url_modifier "$url_modifier" \
#                      --model_name "$model" \
#                      --prompt_template "$prompt_template" \
#                      --modify_meta_data "$modify_meta_data" \
#                      --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                      --output_dir results_fake
#              fi
#          done
#        done
#      done
#  done
#!/usr/bin/env bash

# for model in "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-chat-hf" "allenai/tulu-2-dpo-7b" "allenai/tulu-2-dpo-70b"; do
#     for prompt_template in "input_no_meta" "input_date" "input_url" "input_rank" "input_emphasize_url"; do
#         for modify_meta_data in 0 1; do
#             for favored_stance in "yes" "no"; do
#                 python3 text_llm.py \
#                     --model_$name model \
#                     --prompt_template $prompt_template \
#                     --favored_stance $favored_stance \
#                     --modify_meta_data $modify_meta_data
#             done
#         done
#     done
# done

# input_no_meta" "input_date" "input_date_today" "input_url" "input_rank" "input_emphasize_url" "input_emphasize_wiki_url"

for model in "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Llama-2-70b-chat-hf"; do
    for url_modifier in "cnn_naturalnews_src" "wiki_wordpress_src"; do
      for prompt_template in "input_emphasize_src"; do
        for modify_meta_data in 1; do
            if [[ $modify_meta_data == 1 ]]; then
                for favored_stance in "yes" "no"; do
                    python3 text_llm.py \
                        --dataset_path data_with_keyword.json \
                        --url_modifier "$url_modifier" \
                        --model_name "$model" \
                        --prompt_template "$prompt_template" \
                        --favored_stance "$favored_stance" \
                        --modify_meta_data "$modify_meta_data"

                done
            else
                python3 text_llm.py \
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

for model in "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "allenai/tulu-2-dpo-7b"; do
    for url_modifier in "cnn_naturalnews_url" "wiki_wordpress_url"; do
      for prompt_template in "input_emphasize_url"; do
        for modify_meta_data in 1; do
            if [[ $modify_meta_data == 1 ]]; then
                for favored_stance in "yes" "no"; do
                    python3 text_llm.py \
                        --dataset_path data_with_keyword.json \
                        --url_modifier "$url_modifier" \
                        --model_name "$model" \
                        --prompt_template "$prompt_template" \
                        --favored_stance "$favored_stance" \
                        --modify_meta_data "$modify_meta_data"

                done
            else
                python3 text_llm.py \
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

# for model in "meta-llama/Meta-Llama-3-70B-Instruct"; do
#     for prompt_template in "input_no_meta"; do
#         for modify_meta_data in 0 ; do
#             if [[ $modify_meta_data == 1 ]]; then
#                 for favored_stance in "yes" "no"; do
#                     python3 text_llm.py \
#                         --model_name "$model" \
#                         --prompt_template "$prompt_template" \
#                         --favored_stance "$favored_stance" \
#                         --modify_meta_data "$modify_meta_data"
#                 done
#             else
#                 python3 text_llm.py \
#                     --model_name "$model" \
#                     --prompt_template "$prompt_template" \
#                     --modify_meta_data "$modify_meta_data"
#             fi
#         done
#     done
# done

# "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-chat-hf" "allenai/tulu-2-dpo-7b" "allenai/tulu-2-dpo-70b" 

# for model in "claude-3-sonnet-20240229"; do
#     for prompt_template in "input_no_meta" "input_date" "input_date_today" "input_rank"; do
#         for modify_meta_data in 1; do
#             if [[ $modify_meta_data == 1 ]]; then
#                 for favored_stance in "yes" "no"; do
#                     python3 text_llm.py \
#                         --model_name "$model" \
#                         --prompt_template "$prompt_template" \
#                         --favored_stance "$favored_stance" \
#                         --modify_meta_data "$modify_meta_data" \
#                         --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                         --output_dir results_fake
# 
#                 done
#             else
#                 python3 text_llm.py \
#                     --model_name "$model" \
#                     --prompt_template "$prompt_template" \
#                     --modify_meta_data "$modify_meta_data" \
#                     --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                     --output_dir results_fake
#             fi
#         done
#     done
# done
# 
#  for model in "claude-3-opus-20240229"; do
#      for url_modifier in "wiki_wordpress_src" "cnn_naturalnews_src"; do
#        for prompt_template in "input_emphasize_src"; do
#          for modify_meta_data in 1; do
#             for favored_stance in "yes" "no"; do
#                 python3 text_llm.py \
#                     --model_name "$model" \
#                     --prompt_template "$prompt_template" \
#                     --url_modifier "$url_modifier" \
#                     --favored_stance "$favored_stance" \
#                     --modify_meta_data "$modify_meta_data" \
#                     --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                     --output_dir results_fake
#              done
#          done
#        done
#      done
#  done
#  
#  
#  for model in "claude-3-opus-20240229"; do
#      for url_modifier in "wiki_wordpress_url" "cnn_naturalnews_url"; do
#        for prompt_template in "input_emphasize_url"; do
#          for modify_meta_data in 1; do
#             for favored_stance in "yes" "no"; do
#                  python3 text_llm.py \
#                     --model_name "$model" \
#                     --prompt_template "$prompt_template" \
#                     --url_modifier "$url_modifier" \
#                     --favored_stance "$favored_stance" \
#                     --modify_meta_data "$modify_meta_data" \
#                     --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                     --output_dir results_fake
#              done
#          done
#        done
#      done
#  done

#!/usr/bin/env bash
# "input_date" "input_date_today" "input_emphasize_url_wiki_wordpress_url"  "input_emphasize_url_cnn_naturalnews_url";
for model in "Llama-2-13b-chat-hf"; do
    for prompt_template in "input_emphasize_url_wiki_wordpress_url" "input_emphasize_src_wiki_wordpress_src"; do
        for favored_stance in "yes" "no"; do
            python3 extract_answer_from_generation.py \
                --answer_file results/generate/"$model"/"$prompt_template"_"$favored_stance".json
        done
    done
done

# "Meta-Llama-3-8B-Instruct" "Meta-Llama-3-70B-Instruct" "tulu-2-dpo-7b" "Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf"

#  for model in "gpt-4-turbo-2024-04-09" ; do
#      for prompt_template in "input_emphasize_src_cnn_naturalnews_src" "input_emphasize_src_wiki_wordpress_src" "input_emphasize_url_cnn_naturalnews_url" "input_emphasize_url_wiki_wordpress_url" "input_date" "input_date_today" ; do
#          for favored_stance in "yes" "no"; do
#              python3 extract_answer_from_credible.py \
#                  --answer_file results_fake/credible/"$model"/"$prompt_template"_"$favored_stance".json
#          done
#      done
#  done

# for model in "gpt-4o" "claude-3-opus-20240229"; do
#     for prompt_template in "vision_prompts_with_text/yes_simple_no_pretty" "vision_prompts_with_text/yes_pretty_no_simple" "vision_prompts/yes_simple_no_pretty" "vision_prompts/yes_pretty_no_simple"; do
#         python3 extract_answer_from_generation.py \
#             --answer_file results_vision_fake/generate/"$model"/"$prompt_template".json
# 
#     done
# done

#  for model in "claude-3-haiku-20240307" "claude-3-sonnet-20240229"; do
#      for prompt_template in "vision_prompts/yes_pretty_no_simple" "vision_prompts/yes_simple_no_pretty"; do
#          python3 extract_answer_from_credible.py \
#              --answer_file results_vision_fake/credible/"$model"/"$prompt_template".json
#  
#      done
#  done
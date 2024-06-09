#!/usr/bin/env bash

#  for model in "claude-3-sonnet-20240229"; do
#      for yes_html_template in "simple" "pretty"; do
#          for no_html_template in "simple" "pretty"; do
#              for prompt_template in "vision_prompts"; do
#                  python3 vision_llm.py \
#                      --model_name "$model" \
#                      --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                      --yes_html_template "$yes_html_template" \
#                      --no_html_template "$no_html_template" \
#                      --image_dir data/imgs/fake \
#                      --output_dir results_vision_fake \
#                      --prompt_template "$prompt_template"
#              done
#          done
#      done
#  done


# for model in claude-3-sonnet-20240229; do
#     for yes_html_template in "pretty" "simple"; do
#         for no_html_template in "pretty" "simple"; do
#             for prompt_template in "vision_prompts"; do
#                 if [ "$yes_html_template" == "$no_html_template" ]; then
#                     continue
#                 fi
#                 python3 vision_llm.py \
#                     --model_name "$model" \
#                     --generation \
#                     --max_tokens 512 \
#                     --prompt_template "$prompt_template" \
#                     --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                     --yes_html_template "$yes_html_template" \
#                     --no_html_template "$no_html_template" \
#                     --image_dir data/imgs/fake \
#                     --output_dir results_vision_fake
#             done
#         done
#     done
# done

for model in "claude-3-sonnet-20240229"; do
    for yes_html_template in "pretty" "simple"; do
        for no_html_template in "pretty" "simple"; do
            for prompt_template in "vision_prompts"; do
                if [ "$yes_html_template" == "$no_html_template" ]; then
                    continue
                fi
                python3 vision_llm.py \
                    --model_name "$model" \
                    --generation \
                    --credibility \
                    --max_tokens 512 \
                    --prompt_template "$prompt_template" \
                    --dataset_path data/fake_knowledge_with_evidence_parsed.json \
                    --yes_html_template "$yes_html_template" \
                    --no_html_template "$no_html_template" \
                    --image_dir data/imgs/fake \
                    --output_dir results_vision_fake
            done
        done
    done
done
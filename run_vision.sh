#!/usr/bin/env bash

#  for model in claude-3-sonnet-20240229 claude-3-sonnet-20240229 claude-3-opus-20240229; do
#      for yes_html_template in "pretty" "simple"; do
#          for no_html_template in "pretty" "simple"; do
#              python3 vision_llm.py \
#                  --model_name "$model" \
#                  --dataset_path data/fake_knowledge_with_evidence_parsed.json \
#                  --yes_html_template "$yes_html_template" \
#                  --no_html_template "$no_html_template" \
#                  --image_dir data/imgs/fake \
#                  --output_dir results_vision_fake
#          done
#      done
#  done


for model in claude-3-sonnet-20240229 claude-3-sonnet-20240229; do
    for yes_html_template in "pretty" "simple"; do
        for no_html_template in "pretty" "simple"; do
            python3 vision_llm.py \
                --model_name "$model" \
                --generation \
                --dataset_path data/fake_knowledge_with_evidence_parsed.json \
                --yes_html_template "$yes_html_template" \
                --no_html_template "$no_html_template" \
                --image_dir data/imgs/fake \
                --output_dir results_vision_fake
        done
    done
done
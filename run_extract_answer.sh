#!/usr/bin/env bash

for model in "gpt-4-turbo-2024-04-09" "Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf" "tulu-2-dpo-7b" "Meta-Llama-3-8B-Instruct"; do
    for prompt_template in "input_no_meta" "input_date" "input_date_today" "input_rank" "input_emphasize_src_wiki_wordpress_src" "input_emphasize_url_wiki_wordpress_url" "input_emphasize_src_cnn_naturalnews_src" "input_emphasize_url_cnn_naturalnews_url"; do
        for favored_stance in "yes" "no"; do
            python3 extract_answer_from_generation.py \
                --question_file data/fake_knowledge_with_evidence_parsed.json \
                --answer_file results_fake/generate/"$model"/"$prompt_template"_"$favored_stance".json
        done
    done
done


for model in "claude-3-haiku-20240307" "claude-3-sonnet-20240229"; do
    for prompt_template in "vision_prompts/yes_pretty_no_simple" "vision_prompts/yes_simple_no_pretty";
        python3 extract_answer_from_generation.py \
            --question_file data/fake_knowledge_with_evidence_parsed.json \
            --answer_file results_fake_vision/generate/"$model"/"$prompt_template"_"$favored_stance".json

    done
done
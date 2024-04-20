#!/usr/bin/env bash
#  for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-chat-hf"; do
#      for prompt_template in "input_no_meta" "input_date" "input_url" "input_rank" "input_emphasize_url"; do
#          for modify_meta_data in 0 1; do
#              for favored_stance in "yes" "no"; do
#                  python3 main.py \
#                      --model_$name model \
#                      --prompt_template $prompt_template \
#                      --favored_stance $favored_stance \
#                      --modify_meta_data $modify_meta_data
#              done
#          done
#      done
#  done

#!/usr/bin/env bash
for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-chat-hf"; do
    for prompt_template in "input_no_meta" "input_date" "input_url" "input_rank"; do
        for modify_meta_data in 1; do
            if [[ $modify_meta_data == 1 ]]; then
                for favored_stance in "no" "yes"; do
                    python3 main.py \
                        --model_name "$model" \
                        --prompt_template "$prompt_template" \
                        --favored_stance "$favored_stance" \
                        --modify_meta_data "$modify_meta_data"
                done
            else
                python3 main.py \
                    --model_name "$model" \
                    --prompt_template "$prompt_template" \
                    --modify_meta_data "$modify_meta_data"
            fi
        done
    done
done

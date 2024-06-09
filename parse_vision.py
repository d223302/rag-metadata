#!/usr/bin/env python3 
import numpy as np 
import pandas as pd
import glob
import os
import json
from utils.response_cleaner import normalize_answer
from statsmodels.stats.contingency_tables import mcnemar



def get_verdict(response):
    statistics = {
        'yes': 0,
        'no': 0,
        'inconclusive': 0,
        'invalid': 0
    }
    key_to_verdict = {
        'yes': 'yes',
        'no': 'no',
        'inconclusive': 'inconclusive',
        'invalid': 'n/a'
    }
    for r in response:
        r = normalize_answer(r)
        if 'yes' in r:
            statistics['yes'] += 1
        elif 'no' in r:
            statistics['no'] += 1
        elif 'inconclusive' in r:
            statistics['inconclusive'] += 1
        else:
            statistics['invalid'] += 1

    # Get the majority vote
    max_value = max(statistics.values())
    if max_value <= 1:
        return np.nan
    else:
        for key, value in statistics.items():
            if value == max_value:
                return key_to_verdict[key]

def extract_yes_no_answer(answer):
    answer = normalize_answer(answer)
    if 'yes' in answer:
        return 'yes'
    elif 'no' in answer:
        return 'no'
    else:
        return 'n/a' # TODO: check if we should return something else

result_path = 'results_vision_fake/generate_short_answer'
if "generate" in result_path:
    generation = True
else:
    generation = False


model_list = [dir.split('/')[-1] for dir in glob.glob(result_path + '/*')]
print(model_list)
# 'Meta-Llama-3-8B-Instruct', "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", 
# model_list = ['Meta-Llama-3-8B-Instruct', "Meta-Llama-3-70B-Instruct", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", "tulu-2-dpo-7b", "tulu-2-dpo-70b"]
df = {
    'model': [],
    'prompt_template': [],
    'counterfactual': [],
    'preference': [],
    'disagree_ratio': [],
}

paired_df = {
    'model': [],
    'prompt_template': [],
    'stereo': [],
    'anti_stero': [],
    'no_change': [],
    'valid_count': [],
}

model_name_map = {
    "Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "Meta-Llama-3-70B-Instruct": "Llama-3-70B",
    "Llama-2-7b-chat-hf": "Llama-2-7b",
    "Llama-2-13b-chat-hf": "Llama-2-13b",
    "Llama-2-70b-chat-hf": "Llama-2-70b",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    "tulu-2-dpo-7b": "tulu-7b",
    "tulu-2-dpo-13b": "tulu-13b",
    "tulu-2-dpo-70b": "tulu-70b",
    "claude-3-haiku-20240307": "claude-haiku",
    "claude-3-sonnet-20240229": "claude-sonnet",
    "claude-3-opus-20240229": "claude-opus",
    "gpt-4o": "gpt-4o",
    'gemini-1.5-pro': 'gemini-1.5-pro',
}

template_map = {
    'vision_prompts': 'PNG',
    'vision_prompts_with_text': 'PNG+Text',
}

for model in model_list:
    for prompt_template in ['vision_prompts', 'vision_prompts_with_text']:
        paired_results = {
            'yes_pretty_no_pretty': [],
            'yes_pretty_no_simple': [],
            'yes_simple_no_pretty': [],
            'yes_simple_no_simple': [],
        }
        for counterfactual in paired_results.keys():
            json_file = os.path.join(
                result_path,
                model,
                prompt_template,
                f"{counterfactual}.json"
            )
            
            if not os.path.exists(json_file):
                continue
            print(json_file)
            try: 
                data = json.load(open(json_file))
            except json.decoder.JSONDecodeError:
                print(f"Error in {json_file}")
                continue

            # json_prompt_template = data['args']['prompt_template']
            # print(json_file)
            verdict = data['data']
            disagree_ratio = []
            preference = []
            for v in verdict:
                if generation:
                    v[0] = get_verdict(v[0])
                    v[1] = get_verdict(v[1])
                else:
                    v[0] = extract_yes_no_answer(v[0])
                    v[1] = extract_yes_no_answer(v[1])
                if v[0] == 'n/a' or v[1] == 'n/a':
                    preference.append(np.nan)
                    disagree_ratio.append(np.nan)
                else:
                    if v[0] != v[1]:
                        preference.append(np.nan)
                        disagree_ratio.append(1)
                    else:
                        if v[0] == 'yes':
                            preference.append(1)
                        elif v[0] == 'no':
                            preference.append(0)
                        elif v[0] == 'inconclusive':
                            preference.append(0.5)

            paired_results[counterfactual] = preference
            disagree_ratio = np.mean(np.isnan(preference))
            preference = np.nanmean(preference)

            df['model'].append(model_name_map[model])
            df['prompt_template'].append(template_map[prompt_template])
            df['counterfactual'].append(counterfactual)
            df['preference'].append(preference)
            df['disagree_ratio'].append(disagree_ratio)

        flip_ratio = []
        treatment = "yes_pretty_no_simple"
        no_treatment = "yes_simple_no_pretty"
        mcnemar_table = [[0, 0], [0, 0]]

        if len(paired_results[treatment]) != len(paired_results[no_treatment]):
            print(f"Error in {json_file}")
            continue
        for i in range(len(paired_results[treatment])):
            if (not np.isnan(paired_results[treatment][i])) and (not np.isnan(paired_results[no_treatment][i])):
                treatment_index = 1 if paired_results[treatment][i] == 1 else 0
                no_treatment_index = 1 if paired_results[no_treatment][i] == 1 else 0
                mcnemar_table[treatment_index][no_treatment_index] += 1

                if paired_results[treatment][i] == 1 and paired_results[no_treatment][i] != 1:
                    flip_ratio.append('stereo')
                elif paired_results[treatment][i] != 1 and paired_results[no_treatment][i] == 1:
                    flip_ratio.append('anti-stereo')
                else:
                    flip_ratio.append('no_change')
            else:
                flip_ratio.append('n/a')
        # print(f"len(flip_ratio): {len(flip_ratio)}, len(paired_results['yes_simple_no_pretty']): {len(paired_results['yes_simple_no_pretty'])}")
        # print(flip_ratio)
        paired_df['model'].append(model_name_map[model])
        paired_df['prompt_template'].append(template_map[prompt_template])
        paired_df['stereo'].append(flip_ratio.count('stereo') / (len(flip_ratio) + 1e-10))
        paired_df['anti_stero'].append(flip_ratio.count('anti-stereo') / (len(flip_ratio) + 1e-10))
        paired_df['no_change'].append(flip_ratio.count('no_change') / (len(flip_ratio) + 1e-10))
        paired_df['valid_count'].append(len(flip_ratio) - flip_ratio.count('n/a'))

        # Perform McNemar test
        result = mcnemar(mcnemar_table)
        print(f"Model: {model_name_map[model]}, prompt_template: {template_map[prompt_template]}, p-value: {result.pvalue}")

df = pd.DataFrame(df)
# Format the dataframe. The row is model. The column should be grouped by the prompt template and counterfactual. Each cell is the preference

for target_value in ['preference', 'disagree_ratio']:
    cell_df = df.pivot_table(index='model', columns=['prompt_template', 'counterfactual'], values=[target_value])
    print(cell_df)
    
    # Save the DataFrame to a tsv
    file_name = f"csv_results/results_vision_fake/classify/{target_value}.tsv"
    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    cell_df.to_csv(file_name, sep='\t')

paired_df = pd.DataFrame(paired_df)
paired_df['combined'] = paired_df.apply(lambda row: f"{row['stereo']:.2f}/{row['anti_stero']:.2f}/{row['no_change']:.2f}/{row['valid_count']}", axis=1)
# Delete the original columns
paired_df = paired_df.drop(columns=['stereo', 'anti_stero', 'no_change', 'valid_count'])
# Pivot the DataFrame
paired_df = paired_df.pivot(index='model', columns='prompt_template', values=['combined'])

print(paired_df)

# Save the DataFrame to a tsv
file_name = f"csv_results/results_vision_fake/classify/flip_ratio.tsv"
# Create the directory if it does not exist
if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))
paired_df.to_csv(file_name, sep='\t')


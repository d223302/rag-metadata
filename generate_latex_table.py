#!/usr/bin/env python3 
import numpy as np 
import pandas as pd
import glob
import os
import json
import regex
from utils.response_cleaner import normalize_answer
from colorama import Fore, Style
import argparse
from statsmodels.stats.contingency_tables import mcnemar

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
    "claude-3-haiku-20240307": "haiku",
    "claude-3-sonnet-20240229": "sonnet",
    "claude-3-opus-20240229": "opus",
    "gpt-4o": "GPT-4o",
}

template_map = {
    'input_date': 'A. no-today',
    'input_date_today': 'B. today',
    'input_rank': 'Rank',
    'input_emphasize_url_cnn_naturalnews_url': 'A. Emph URL CNN/Nat',
    'input_emphasize_url_wiki_wordpress_url': 'A. Emph URL Wiki/WP',
    'input_url_cnn_naturalnews_url': 'A. URL CNN/Nat',
    'input_url_wiki_wordpress_url': 'A. URL Wiki/WP',
    'input_emphasize_src_cnn_naturalnews_src': 'B. Emph Src CNN/Nat',
    'input_emphasize_src_wiki_wordpress_src': 'B. Emph Src Wiki/WP',
    'vision_prompts': 'A. Screenshot',
    'vision_prompts_with_text': 'B. Screenshot + Text',
}


def get_cls_verdict(answer):
    if isinstance(answer, str):
        answer = normalize_answer(answer)
        if 'yes' in answer:
            return 1
        elif 'no' in answer:
            return 0
        else:
            return np.nan # TODO: check if we should return something else
    # I am not sure what the below is for. 
    #  elif np.isnan(answer):
    #      return 'n/a'
    #  elif answer == 0.5:
    #      return 'n/a'
    else:
        raise ValueError(f"answer: {answer}")
    
def get_gen_verdict(response):
    statistics = {
        'yes': 0,
        'no': 0,
        'inconclusive': 0,
        'invalid': 0
    }
    key_to_verdict = {
        'yes': 1,
        'no': 0,
        'inconclusive': 0.5,
        'invalid': np.nan
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
    # print(response, max_value)
    if max_value <= 1:
        return np.nan
    else:
        for key, value in statistics.items():
            if value == max_value:
                return key_to_verdict[key]

def get_verdict(v, output_mode):
    if output_mode == 'classify':
        return get_cls_verdict(v)
    elif output_mode == 'generate_short_answer':
        return get_gen_verdict(v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate latex table')
    parser.add_argument('--dataset_type', type=str, default='results_fake', help='dataset type')
    parser.add_argument('--mode', choices = ['date', 'src_wiki', 'src_cnn', 'vision'])
    parser.add_argument('--vision', action = "store_true")
    parser.add_argument('--result_root', type=str, default='./results_fake', help='result path')
    args = parser.parse_args()
    if args.vision:
        prompt_templates = ['vision_prompts', 'vision_prompts_with_text']
    else:
        if args.mode == 'date':
            prompt_templates = ['input_date', 'input_date_today']
        elif 'src' in args.mode:
            if 'wiki' in args.mode:
                prompt_templates = ['input_emphasize_url_wiki_wordpress_url', 'input_emphasize_src_wiki_wordpress_src']
            elif 'cnn' in args.mode:
                prompt_templates = ['input_emphasize_url_cnn_naturalnews_url', 'input_emphasize_src_cnn_naturalnews_src']
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    df = {
        'model': [],
        'prompt_template': [],
        'stance': [],
        'yes_ratio': [],
        'no_ratio': [],
        'inconclusive_ratio': [],
        'inconsistent_ratio': [],
        'response_type': [],
    }

    paired_df = {
        'model': [],
        'prompt_template': [],
        '0 to 1': [],
        '1 to 0': [],
        'no_change': [],
        'others': [],
        'p_value': [],
        'mcnemar_count': [],
        'flip_ratio': [],
        'response_type': [],
        'stance': [],
    }

    # First, parse the results from the classify folder
    if args.vision:
        model_list = ['gpt-4o', "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
    else:
        model_list = ["gpt-4-turbo-2024-04-09", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf","tulu-2-dpo-7b", 'Meta-Llama-3-8B-Instruct', "Meta-Llama-3-70B-Instruct"]
    # model_list = ["Meta-Llama-3-70B-Instruct"]
    for output_mode in ['classify', 'generate_short_answer']:
        for model in model_list:
            for  prompt_template in prompt_templates:
                if args.vision:
                    paired_results = {
                        'yes_pretty_no_simple': [],
                        'yes_simple_no_pretty': [],
                    }
                    stances = list(paired_results.keys())
                else:
                    paired_results = {
                        'yes': [],
                        'no': [],
                    }
                    stances = list(paired_results.keys())
                for stance in stances:
                    if args.vision:
                        json_file = os.path.join(
                            args.result_root, output_mode, model, prompt_template, f"{stance}.json"
                        )
                    else:
                        json_file = os.path.join(
                            args.result_root, output_mode, model, f'{prompt_template}_{stance}.json'
                        )
                    if not os.path.exists(json_file):
                        print(f"{Fore.RED}File not found: {json_file}{Style.RESET_ALL}")
                        continue
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                    except json.decoder.JSONDecodeError:
                        print(f"{Fore.RED}Error reading file: {json_file}{Style.RESET_ALL}")
                        continue

                    if not args.vision:
                        json_prompt_template = data['args']['prompt_template']
                        json_modify_meta_data = data['args']['modify_meta_data']
                        json_stance = data['args']['favored_stance']
                        if not json_modify_meta_data:
                            continue
                    
                    verdict = data['data']
                    yes = []
                    # print("===============")
                    for v in verdict:
                        v_0 = get_verdict(v[0], output_mode)
                        v_1 = get_verdict(v[1], output_mode)
                        #  print(f"Pos 1: The answer before: {v[0]}. The answer after: {v_0}")
                        #  print(f"Pos 2: The answer before: {v[1]}. The answer after: {v_1}")
                        if np.isnan(v_0) or np.isnan(v_1):
                            yes.append(np.nan)
                        else:
                            if v_0 == v_1: # if the two positions' answers match
                                yes.append(v_0)
                            else: # If the two answers does not match
                                yes.append(np.nan)
                    
                    paired_results[stance] = yes
                    # print(json_file)
                    # print(yes)
                    # print(len(yes))

                    number_of_yes = len([x for x in yes if x == 1])
                    number_of_no = len([x for x in yes if x == 0])
                    number_of_inconclusive = len([x for x in yes if x == 0.5])
                    number_of_nan = len([x for x in yes if np.isnan(x)])

                    # Add the results to the data frame
                    df['model'].append(model_name_map[model])
                    df['prompt_template'].append(template_map[prompt_template])
                    df['stance'].append(stance)
                    df['yes_ratio'].append(number_of_yes / len(yes))
                    df['no_ratio'].append(number_of_no / len(yes))
                    df['inconclusive_ratio'].append(number_of_inconclusive / len(yes))
                    df['inconsistent_ratio'].append(number_of_nan / len(yes))
                    # df['yes_ratio'].append(number_of_yes / (number_of_yes + number_of_no + number_of_inconclusive + 1e-10))
                    # df['no_ratio'].append(number_of_no / (number_of_yes + number_of_no + number_of_inconclusive + 1e-10))
                    # df['inconclusive_ratio'].append(number_of_inconclusive / (number_of_yes + number_of_no + number_of_inconclusive + 1e-10))
                    # df['inconsistent_ratio'].append(0)
                    df['response_type'].append(output_mode)
                

                flip_ratio = []
                treatment = list(paired_results.keys())[0]
                no_treatment = list(paired_results.keys())[1]
                mcnemar_table = [[0, 0], [0, 0]]

                if len(paired_results[treatment]) != len(paired_results[no_treatment]):
                    print(f"{Fore.RED}Length of the two lists do not match{Style.RESET_ALL}")
                    continue

                total = len(paired_results[treatment])
                temp_counter = {
                    '0 to 1': 0,
                    '1 to 0': 0,
                    'no_change': 0,
                    'others': 0,
                }
                for i in range(total):
                    if paired_results[treatment][i] == paired_results[no_treatment][i]:
                        temp_counter['no_change'] += 1
                    elif paired_results[no_treatment][i] == 1 and paired_results[treatment][i] == 0:
                        temp_counter['1 to 0'] += 1
                    elif paired_results[no_treatment][i] == 0 and paired_results[treatment][i] == 1:
                        temp_counter['0 to 1'] += 1
                    else:
                        temp_counter['others'] += 1

                    treatment_index = 0 if paired_results[treatment][i] == 1 else 1
                    no_treatment_index = 0 if paired_results[no_treatment][i] == 1 else 1
                    mcnemar_table[treatment_index][no_treatment_index] += 1
                    
                # Calculate the p-value
                p_value = mcnemar(mcnemar_table, exact=True).pvalue
                paired_df['model'].append(model_name_map[model])
                paired_df['prompt_template'].append(template_map[prompt_template])
                paired_df['0 to 1'].append(temp_counter['0 to 1'] / (total + 1e-10))
                paired_df['1 to 0'].append(temp_counter['1 to 0'] / (total + 1e-10))
                paired_df['no_change'].append(temp_counter['no_change'] / (total + 1e-10))
                paired_df['others'].append(temp_counter['others'] / (total + 1e-10))
                paired_df['p_value'].append(p_value)
                paired_df['mcnemar_count'].append(total)
                paired_df['flip_ratio'].append(1 - temp_counter['no_change'] / (total + 1e-10))
                paired_df['response_type'].append(output_mode)
                paired_df['stance'].append('z')

df = pd.DataFrame(df)
paired_df = pd.DataFrame(paired_df)

# Print the df 
# for value in ['yes_ratio', 'no_ratio', 'inconclusive_ratio', 'inconsistent_ratio']:
for value in ['no_ratio']: 
    print(f"\n\n\n{value}")
    pivoted = df.pivot_table(
        index = 'model',
        columns = ['response_type', 'prompt_template', 'stance'],
        values = value,
    )
    pivoted = pivoted.reindex([model_name_map[model] for model in model_list])
    pivoted = pivoted.reindex(
        ['yes_simple_no_pretty', 'yes_pretty_no_simple'] if args.vision else ['no', 'yes'],
        level = 'stance',
        axis = 1,
    )
    print(pivoted.columns)
    latex_table = pivoted.to_latex(index = True, float_format=lambda x: "{:.1f}".format(x * 100))

    # print(latex_table)


# Print the flip ratio
pivoted_pair = paired_df.pivot_table(
    index = 'model',
    columns = ['response_type', 'prompt_template', 'stance'],
    values = 'flip_ratio',
)
pivoted_pair = pivoted_pair.reindex([model_name_map[model] for model in model_list])

print(pivoted_pair.columns)

combined = pd.merge(pivoted, pivoted_pair, left_index=True, right_index=True, how = 'outer')
combined_new = {
    'response_type': [],
    'prompt_template': [],
    'stance': [],
    'model': [],
    'value': [],
}
for key, inner_dict in combined.items():
    for model, value in inner_dict.items():
        combined_new['response_type'].append(key[0])
        combined_new['prompt_template'].append(key[1])
        combined_new['stance'].append(key[2])
        combined_new['model'].append(model)
        combined_new['value'].append(value)

combined_new = pd.DataFrame.from_dict(combined_new)
combined_new = combined_new.pivot_table(
    index = 'model',
    columns = ['response_type', 'prompt_template', 'stance'],
    values = 'value',
)
combined_new = combined_new.reindex([model_name_map[model] for model in model_list])
combined_new = combined_new.reindex(
    ['yes_simple_no_pretty', 'yes_pretty_no_simple', 'z'] if args.vision else ['no', 'yes', 'z'],
    level = 'stance',
    axis = 1,
)

latex_table = combined_new.to_latex(index = True, float_format=lambda x: "{:.1f}".format(x * 100))
print(combined_new)
print(latex_table)


# Print the p-value
pivoted_pair = paired_df.pivot_table(
    index = 'model',
    columns = ['response_type', 'prompt_template'],
    values = 'p_value',
)

pivoted_pair = pivoted_pair.reindex([model_name_map[model] for model in model_list])
print(pivoted_pair)
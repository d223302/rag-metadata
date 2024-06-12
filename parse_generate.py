#!/usr/bin/env python3 
import numpy as np 
import pandas as pd
import glob
import os
import json
import regex
from utils.response_cleaner import normalize_answer
from colorama import Fore, Style

def get_json_data(file_path):
    if not os.path.exists(file_path):
        print(f"File {Fore.BLUE}{file_path}{Style.RESET_ALL} does not exist")
        return None, None
    try:
        data = json.load(open(file_path))
    except json.decoder.JSONDecodeError:
        print(f"Error in {file_path}")
        return None, None
    responses = data['data']
    args = data['args']
    return responses, args 

def get_verdict(response):
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
    if max_value <= 1:
        return np.nan
    else:
        for key, value in statistics.items():
            if value == max_value:
                return key_to_verdict[key]
    

# Extract whether the date is presented
def extract_date(response):
    if '2024' in response or '2020' in response:
        return 1
    else:
        return 0

# Extract whether the website is presented
def extract_website(response):
    response = normalize_answer(response)
    if 'cnn' in response or 'wordpress' in response:
        return True
    elif 'naturalnews' in response or 'wiki' in response:
        return True
    elif 'http' in response:
        return True
    else:
        return False

dataset_type = 'results_fake'

short_answer_result_path = f'{dataset_type}/generate_short_answer'
long_answer_result_path = f'{dataset_type}/generate'

model_list = [dir.split('/')[-1] for dir in glob.glob(short_answer_result_path + '/*')]
print(model_list)
# 'Meta-Llama-3-8B-Instruct', "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", 
# model_list = ['Meta-Llama-3-8B-Instruct', "Meta-Llama-3-70B-Instruct", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", "tulu-2-dpo-7b", "tulu-2-dpo-70b"]
df = {
    'model': [],
    'prompt_template': [],
    'stance': [],
    'preference': [],
    'disagree_ratio': [],
    'date_count': [],
    'url_count': [],
}


paired_df = {
    'model': [],
    'prompt_template': [],
    'stereo': [],
    'anti_stero': [],
    'no_change': [],
    'others': [],
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
    'input_no_meta': 'No Meta',
    'input_date': 'Date',
    'input_date_today': 'Date Today',
    'input_rank': 'Rank',
    'input_emphasize_url_cnn_naturalnews_url': 'Emph URL CNN/Nat',
    'input_emphasize_url_wiki_wordpress_url': 'Emph URL Wiki/WP',
    'input_url_cnn_naturalnews_url': 'URL CNN/Nat',
    'input_url_wiki_wordpress_url': 'URL Wiki/WP',
    'input_emphasize_src_cnn_naturalnews_src': 'Emph Src CNN/Nat',
    'input_emphasize_src_wiki_wordpress_src': 'Emph Src Wiki/WP',
}

#"input_no_meta", "input_date", "input_date_today", "input_rank", "input_url", "input_emphasize_url", "input_emphasize_wiki_url", "input_emphasize_url_wiki_wordpress_url", "input_emphasize_url_cnn_naturalnews_url", "input_emphasize_src_wiki_wordpress_src", "input_emphasize_src_cnn_naturalnews_src", "input_html_pretty_simple_html"


for model in model_list:
    for prompt_template in ["input_emphasize_url_wiki_wordpress_url", "input_emphasize_src_wiki_wordpress_src"]:
        paired_results = {
            'yes': [],
            'no': [],
        }
        for stance in ['yes', 'no']:
            long_answer_json_file = os.path.join(
                long_answer_result_path,
                model,
                f"{prompt_template}_{stance}.json"
            )
            short_answer_json_file = os.path.join(
                short_answer_result_path,
                model,
                f"{prompt_template}_{stance}.json"
            )

            long_responses, long_args = get_json_data(long_answer_json_file)
            short_responses, short_args = get_json_data(short_answer_json_file)
            if long_responses is None or short_responses is None:
                continue
            
            # Make sure they are the same. They should be the same by construction
            for key, value in long_args.items():
                # if key == 'max_tokens':
                #     continue
                assert long_args[key] == short_args[key]
            assert len(long_responses) == len(short_responses)
            
            json_prompt_template = long_args['prompt_template']
            json_modify_meta_data = long_args['modify_meta_data']
            json_stance = long_args['favored_stance']

            if prompt_template != "input_no_meta":
                if stance != 'original' and not json_modify_meta_data:
                    continue
            # print(json_file)
            disagree_ratio = []
            preference = []
            date_count = []
            url_count = []
            for sr_pair, lr_pair in zip(short_responses, long_responses):
                verdict_1 = get_verdict(sr_pair[0])
                verdict_2 = get_verdict(sr_pair[1])
                if verdict_1 == verdict_2:
                    disagree_ratio.append(0)
                    preference.append(verdict_1)
                else:
                    disagree_ratio.append(1)
                    preference.append(np.nan)

                if extract_date(lr_pair[0]) or extract_date(lr_pair[1]):
                    date_count.append(1)
                else:
                    date_count.append(0)
                
                if extract_website(lr_pair[0]) or extract_website(lr_pair[1]):
                    url_count.append(1)
                else:
                    url_count.append(0)

            print(f"Model: {model}, preference: {len(preference)}")
            paired_results[stance] = preference
            preference = np.nanmean(preference)
            disagree_ratio = np.mean(disagree_ratio)
            date_count = np.mean(date_count)
            url_count = np.mean(url_count)

            df['model'].append(model_name_map[model])
            df['prompt_template'].append(template_map[prompt_template])
            df['stance'].append(stance)
            df['preference'].append(preference)
            df['disagree_ratio'].append(disagree_ratio)
            df['date_count'].append(date_count)
            df['url_count'].append(url_count)
    
        # Calculate the flip ratio and consistent ratio
        flip_ratio = []
        if len(paired_results['yes']) != len(paired_results['no']):
            print(f"Lenght of yes: {len(paired_results['yes'])}, no: {len(paired_results['no'])}")
            print(f"{Fore.RED}Error in {long_answer_json_file}{Style.RESET_ALL}")
            continue
        for i in range(len(paired_results['yes'])):
            yes_result = paired_results['yes'][i]
            no_result = paired_results['no'][i]
            if (not np.isnan(yes_result)) and (not np.isnan(no_result)):
                if yes_result == no_result:
                    flip_ratio.append('no_change')
                elif yes_result == 1 and no_result != 1:
                    flip_ratio.append('stereo')
                elif yes_result != 1 and no_result == 1:
                    flip_ratio.append('anti-stereo')
                else:
                    flip_ratio.append('others')
            else:
                flip_ratio.append('n/a')
        

        valid_count = len(flip_ratio) - flip_ratio.count('n/a')
        paired_df['model'].append(model_name_map[model])
        paired_df['prompt_template'].append(template_map[prompt_template])
        paired_df['stereo'].append(flip_ratio.count('stereo') / (valid_count + 1e-10))
        paired_df['anti_stero'].append(flip_ratio.count('anti-stereo') / (valid_count + 1e-10))
        paired_df['no_change'].append(flip_ratio.count('no_change') / (valid_count + 1e-10))
        paired_df['others'].append(flip_ratio.count('others') / (valid_count + 1e-10))
        paired_df['valid_count'].append(valid_count)


df = pd.DataFrame(df)
# Format the dataframe. The row is model. The column should be grouped by the prompt template and stance. Each cell is the preference


for target_value in ['preference', 'disagree_ratio', 'date_count', 'url_count']:
    cell_df = df.pivot_table(index='model', columns=['prompt_template', 'stance'], values=[target_value])
    print(cell_df)
    # Save the DataFrame to a tsv
    file_name = f"csv_results/{dataset_type}/generate/{target_value}.tsv"
    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    cell_df.to_csv(file_name, sep='\t')

paired_df = pd.DataFrame(paired_df)
paired_df['combined'] = paired_df.apply(lambda row: f"{row['stereo']:.2f}/{row['anti_stero']:.2f}/{row['no_change']:.2f}/ {row['others']:.2f} / {row['valid_count']}", axis=1)
# Delete the original columns
paired_df = paired_df.drop(columns=['stereo', 'anti_stero', 'no_change', 'others', 'valid_count'])
# Pivot the DataFrame
paired_df = paired_df.pivot(index='model', columns='prompt_template', values=['combined'])

print(paired_df)

# Save the DataFrame to a tsv
file_name = f"csv_results/{dataset_type}/generate/flip_ratio.tsv"
paired_df.to_csv(file_name, sep='\t')
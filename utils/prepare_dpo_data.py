#!/usr/bin/env python3
from prompts import prompt_class
from language_models import create_llm
from customize_meta_data import change_url, change_ranking, change_date , change_url_to_wiki, wiki_wordpress_url, cnn_naturalnews_url 
import logging
import argparse
import json
import os
import random
from tqdm import tqdm
from colorama import Fore, Style

# Set default logging level to WARNING to reduce noise
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Now configure your specific logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set your logger to log INFO and above levels

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check if handlers are already configured to prevent duplicate logging
if not logger.handlers:
    # Create a stream handler to output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Set propagate to False if you do not want logs to be handled by root logger handlers
logger.propagate = False

# Explicitly set external libraries to WARNING to suppress their INFO logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# If you have specific submodules in 'utils' that should log, set them explicitly
logging.getLogger("utils.language_models").setLevel(logging.INFO)


def contains_meta_data(response, prompt_template, url_modifier):
    if "url" in prompt_template:
        if "www" in response:
            return True
        if "http" in response:
            return True
        if 'wiki' in response.lower():
            return True
        if 'cnn' in url_modifier:
            if 'cnn' in response.lower() or 'naturalnews' in response.lower():
                return True
        if 'wordpress' in url_modifier:
            if 'wordpress' in response.lower() or 'wiki' in response.lower():
                return True
    
    if "date" in prompt_template:
        if '2020' in response or '2024' in response:
            return True
    
    return False 

def prepare_dpo(
    dataset_path,
    chosen_result_file,
    rejected_result_file,
    prompt_template,
    favored_stance,
    modify_meta_data,
    wiki_files = "utils/searched_wiki_urls_new.json",
    seed = 42,
    url_modifier = "",
    ):

    # Fix all seeds
    random.seed(seed)

    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Load the result file
    with open(chosen_result_file, "r") as f:
        logger.info(f"Loading the chosen result file: {chosen_result_file}")
        chosen_results = json.load(f)['data']
    with open(rejected_result_file, "r") as f:
        logger.info(f"Loading the rejected result file: {rejected_result_file}")
        rejected_results = json.load(f)['data']

    assert len(chosen_results) == len(rejected_results) == len(data), f"Lengths of the data files do not match. {len(chosen_results)}, {len(rejected_results)}, {len(data)}"

    
    # Load wiki url
    if "input_emphasize_wiki_url" in prompt_template:
        wiki_urls = json.load(open(wiki_files, "r"))
        assert len(wiki_urls) == len(data)
    

    # Generate
    dpo_dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    pbar = tqdm(enumerate(data), total=len(data))
    for i, instance in pbar:
        '''
        data_format = {
            'search_query': search_query,
            'search_engine_input': [search_engine_input],
            'search_type': [search_type],
            'urls': [url],
            'titles': [title],
            'text_raw': [text_raw],
            'text_window': [text_window],
            'stance': [stance]
        }
        '''
        question = instance['search_query']
        
        # Sample two documents with different stances
        yes_index = [j for j in range(len(instance['stance'])) if instance['stance'][j] == 'yes']
        no_index = [j for j in range(len(instance['stance'])) if instance['stance'][j] == 'no']
        if len(yes_index) == 0 or len(no_index) == 0:
            continue
        yes_index = random.choice(yes_index)
        no_index = random.choice(no_index)
        permutation = []
        for permute_idx, idx_set in enumerate([(yes_index, no_index), (no_index, yes_index)]):
            doc_1_idx, doc_2_idx = idx_set
            
            TITLE_1 = instance['titles'][doc_1_idx]
            URL_1 = instance['urls'][doc_1_idx]
            DATE_1 = ''
            TEXT_1 = instance['text_window'][doc_1_idx]
            stance_1 = instance['stance'][doc_1_idx]
            TITLE_2 = instance['titles'][doc_2_idx]
            URL_2 = instance['urls'][doc_2_idx]
            DATE_2 = ''
            TEXT_2 = instance['text_window'][doc_2_idx]
            stance_2 = instance['stance'][doc_2_idx]
            # Put the above into a dictionary
            fill_in_dict = {
                "TITLE_1": TITLE_1,
                "URL_1": URL_1,
                "DATE_1": DATE_1,
                "TEXT_1": TEXT_1,
                "RANK_1": None,
                "stance_1": stance_1,
                "TITLE_2": TITLE_2,
                "URL_2": URL_2,
                "DATE_2": DATE_2,
                "TEXT_2": TEXT_2,
                "RANK_2": None,
                "stance_2": stance_2,
                "search_query": question,
            }
            # Modify / Add additional keys to the dictionary
            if modify_meta_data:
                if "date" in prompt_template:
                    fill_in_dict = change_date(
                        fill_in_dict, 
                        favored_stance, 
                        "2024-04-01", 
                        "2020-04-01",
                    )
                if 'url' in prompt_template:
                    if url_modifier != "":
                        fill_in_dict = globals()[url_modifier](
                            fill_in_dict,
                            favored_stance,
                            instance['keywords'],
                        )

                    elif 'wiki' in prompt_template:
                        fill_in_dict = change_url_to_wiki(
                            fill_in_dict,
                            favored_stance,
                            wiki_title =  wiki_urls[i]['title'],
                            wiki_url =  wiki_urls[i]['url'],
                        )
                    else:
                        fill_in_dict = change_url(
                            fill_in_dict, 
                            "https://en.wikipedia.org/wiki/",
                            favored_stance,
                        )
                if prompt_template == "input_rank":
                    fill_in_dict = change_ranking(
                        fill_in_dict,
                        favored_stance,
                        1,
                        5,
                    )
            if "1" in prompt_template and permute_idx == 1:
                input_prompt = getattr(prompt_class, prompt_template.replace('1', '2'))
            else:
                input_prompt = getattr(prompt_class, prompt_template)
            # TODO: change the name "search engine input" since it is confusing
            search_engine_input = input_prompt.format(
                **fill_in_dict,
            )
            
            search_engine_input = search_engine_input.replace(
                '''Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".''',
                'Give a concise answer with explanations.'
            )
            
            # Get the response from the result file
            
            chosen = chosen_results[i][permute_idx]
            rejected = rejected_results[i][permute_idx]
            if contains_meta_data(chosen, prompt_template, url_modifier):
                dpo_dataset_dict['chosen'].append(chosen)
                dpo_dataset_dict['rejected'].append(rejected)
                dpo_dataset_dict['prompt'].append(search_engine_input)
            else:
                continue

            if i == 0:
                print(search_engine_input)
                print("=" * 20)

        
    return dpo_dataset_dict
    

def main(args):
    dpo_set = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    result_dir = "../results_fake_test/generate"
    for model in ["Meta-Llama-3-8B-Instruct"]:
        for prompt_template in ["input_date_today"]:
            for favored_stance in ['yes', 'no']:
#         for chosen_result_file in ["Meta-Llama-3-8B-Instruct/input_date_today_no.json"]:
                rejected_result_file = os.path.join(
                    result_dir,
                    model,
                    "input_no_meta_original.json"
                )
                dpo_subset = prepare_dpo(
                    dataset_path = "../data/fake_knowledge_with_evidence_parsed_test.json",
                    chosen_result_file = os.path.join(result_dir, model, f"{prompt_template}_{favored_stance}.json"),
                    rejected_result_file = rejected_result_file,
                    prompt_template = prompt_template,
                    favored_stance = favored_stance,
                    modify_meta_data = 1,
                    seed = 42,
                    url_modifier = "",
                )
                dpo_set['prompt'] += dpo_subset['prompt']
                dpo_set['chosen'] += dpo_subset['chosen']
                dpo_set['rejected'] += dpo_subset['rejected']
    
    for model in ["Meta-Llama-3-8B-Instruct"]:
        for prompt_template in ["input_emphasize_url"]:
            for url_modifier in ["wiki_wordpress_url", "cnn_naturalnews_url"]:
                for favored_stance in ['yes', 'no']:
                    rejected_result_file = os.path.join(
                        result_dir,
                        model,
                        "input_no_meta_original.json"
                    )
                    dpo_subset = prepare_dpo(
                        dataset_path = "../data/fake_knowledge_with_evidence_parsed_test.json",
                        chosen_result_file = os.path.join(result_dir, model, f"{prompt_template}_{url_modifier}_{favored_stance}.json"),
                        rejected_result_file = rejected_result_file,
                        prompt_template = prompt_template,
                        favored_stance = favored_stance,
                        modify_meta_data = 1,
                        seed = 42,
                        url_modifier = url_modifier,
                    )
                    dpo_set['prompt'] += dpo_subset['prompt']
                    dpo_set['chosen'] += dpo_subset['chosen']
                    dpo_set['rejected'] += dpo_subset['rejected']
    
    with open("../data/dpo_test.json", "w") as f:
        json.dump(dpo_set, f)

    
    logger.info(f"Finished preparing the DPO dataset.")
    logger.info(f"Number of instances in the DPO dataset: {len(dpo_set['prompt'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the DPO dataset for the LLM model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--dataset_path",
        type = str,
        default = "data.json",
        help = "The path to the dataset file",
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "The random seed to use",
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "results",
        help = "The directory to store the output files",
    )
    parser.add_argument(
        "--prompt_template",
        type = str,
        choices = ["input_no_meta", "input_date", "input_url", "input_url_1", "input_url_2", "input_rank", "input_emphasize_url", "input_date_today", "input_emphasize_wiki_url", "input_emphasize_wiki_url_1", "input_emphasize_wiki_url_2", "wiki_wordpress_url", "cnn_naturalnews_url"],
        default = "input_no_meta",
    )
    parser.add_argument(
        "--favored_stance",
        type = str,
        choices = ["yes", "no"],
        default = "no",
    )
    parser.add_argument(
        "--modify_meta_data",
        type = int,
        choices = [0, 1],
        default = 0,
        help = "If 1, the metadata is modified. If 0, the metadata is not modified.",
    )
    parser.add_argument(
        '--wiki_files',
        type = str,
        default = "utils/searched_wiki_urls_new.json",
    )
    parser.add_argument(
        '--generation',
        action = 'store_true',
        help = "If true, the model will generate the response using greedy decoding. Otherwise, we will only calculate the probability of yes/no."
    )
    parser.add_argument(
        '--url_modifier',
        type = str,
        default = "",
        help = "The url modifier to use for the input_url prompt template. If empty, the default url modifier is used.",
    )
    args = parser.parse_args()
    main(args)

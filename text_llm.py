#!/usr/bin/env python3
from utils.prompts import prompt_class
from utils.language_models import create_llm
from utils.customize_meta_data import change_url, change_ranking, change_date , change_url_to_wiki, wiki_wordpress_url, cnn_naturalnews_url, wiki_wordpress_src, cnn_naturalnews_src
import logging
import argparse
import json
import os
import torch
import random
from tqdm import tqdm
from transformers import set_seed
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


def check_args(args):
    if args.modify_meta_data:
        if args.prompt_template not in ["input_date", "input_date_today", "input_url", "input_url_1", "input_url_2", "input_rank", "input_rank_no_google", "input_emphasize_url", "input_emphasize_wiki_url", "input_emphasize_wiki_url_1", "input_emphasize_wiki_url_2", "wiki_wordpress_url", "cnn_naturalnews_url", "input_emphasize_src"]:
            logger.info("Please select a valid prompt template that has a meta data field.")
            exit()
    else:
        logger.info("Not modifying metadata. The --modify_meta_data flag is False. Not meta data is modified and the favored_stance flag is ignored.")

def main(args):

    # Check the arguments
    check_args(args)

    # Fix all seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    # Load dataset
    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    sampling_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": args.seed,
    }

    # Prepare the output directory
    args.output_dir = os.path.join(
        args.output_dir,
        'generate' if args.generation else 'classify',
        f"{args.model_name.split('/')[-1]}"
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.modify_meta_data:
        output_file = os.path.join(args.output_dir, f"{args.prompt_template + '_' + args.url_modifier if args.url_modifier != '' else args.prompt_template}_{args.favored_stance}.json")
    else:
        output_file = os.path.join(args.output_dir, f"{args.prompt_template + '_' + args.url_modifier if args.url_modifier != '' else args.prompt_template}_original.json")

    # Load model
    llm = create_llm(
        model_name = args.model_name,
        cache_file = os.path.join(
            args.cache_dir, 
            args.model_name.split('/')[-1] + ("generate" if args.generation else ""),
        ),
        sampling_params = sampling_params,
        tensor_parallel_size = 2,
        openai_api_key = args.openai_api_key,
        google_api_key = args.google_api_key,
        claude_api_key = args.claude_api_key,
    )
    
    # Load wiki url
    if "input_emphasize_wiki_url" in args.prompt_template:
        wiki_urls = json.load(open(args.wiki_files, "r"))
        assert len(wiki_urls) == len(data)
    

    # Generate
    results = []

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
            if args.modify_meta_data:
                if "date" in args.prompt_template:
                    fill_in_dict = change_date(
                        fill_in_dict, 
                        args.favored_stance, 
                        "2024-04-01", 
                        "2020-04-01",
                    )
                if 'url' in args.prompt_template or 'src' in args.prompt_template:
                    if args.url_modifier != "":
                        fill_in_dict = globals()[args.url_modifier](
                            fill_in_dict,
                            args.favored_stance,
                            instance['keywords'],
                        )
                    elif 'wiki' in args.prompt_template:
                        fill_in_dict = change_url_to_wiki(
                            fill_in_dict,
                            args.favored_stance,
                            wiki_title =  wiki_urls[i]['title'],
                            wiki_url =  wiki_urls[i]['url'],
                        )
                    else:
                        fill_in_dict = change_url(
                            fill_in_dict, 
                            "https://en.wikipedia.org/wiki/",
                            args.favored_stance,
                        )
                if "rank" in args.prompt_template:
                    fill_in_dict = change_ranking(
                        fill_in_dict,
                        args.favored_stance,
                        1,
                        5,
                    )
            if "1" in args.prompt_template and permute_idx == 1:
                input_prompt = getattr(prompt_class, args.prompt_template.replace('1', '2'))
            else:
                input_prompt = getattr(prompt_class, args.prompt_template)
            # TODO: change the name "search engine input" since it is confusing
            search_engine_input = input_prompt.format(
                **fill_in_dict,
            )
            if args.generation:
                search_engine_input = search_engine_input.replace(
                    '''Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".''',
                    'Give a concise answer with explanations.'
                )
            if args.generation:
                response = llm.full_generate(search_engine_input)
            else:
                response = llm.generate(search_engine_input)
            permutation.append(response)
            if i == 0:
                print(search_engine_input)
                print("=" * 20)
        
        results.append(permutation)
        #  # The following are for debugging
        if i == 0:
          print(permutation)
        
        # Save cache
        if i % 5 == 0:
            llm.save_cache()

        with open(output_file, "w") as f:
            if i % args.save_freq == 0 or i == len(data) - 1:
                json.dump(
                    {"args": vars(args), "data": results},
                    f, 
                    indent=4
                )
        
    # Summarize the cost
    llm.summarize_cost()
    llm.save_cache()
        
    if args.generation:
        return
    # Summarize result
    yes_count = 0
    no_count = 0
    for perm in results:
        for response in perm:
            if response == "Yes":
                yes_count += 1
            elif response == "No":
                no_count += 1
    logger.info(f"Prompt Template: {Fore.YELLOW} {args.prompt_template} {Style.RESET_ALL}")
    logger.info(f"Modify meta data: {Fore.YELLOW} {args.modify_meta_data} {Style.RESET_ALL}")
    logger.info(f"Favored Stance: {Fore.YELLOW} {args.favored_stance} {Style.RESET_ALL}")
    logger.info(f"Total Yes: {Fore.GREEN} {yes_count} (percentage: {yes_count / (yes_count + no_count)}) {Style.RESET_ALL}")
    logger.info(f"Total No: {Fore.RED} {no_count} (percentage: {no_count / (yes_count + no_count)}) {Style.RESET_ALL}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for metadata ablation study")
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
        "--openai_api_key",
        type = str,
        default = "api/openai.txt",
        help = "The path to a .txt file that includes the OpenAI API key to use",
    )
    parser.add_argument(
        "--google_api_key",
        type = str,
        default = "api/google.txt",
        help = "The path to a .txt file that includes the Google API key to use",
    )
    parser.add_argument(
        "--claude_api_key",
        type = str,
        default = "api/claude.txt",
        help = "The path to a .txt file that includes the Claude API key to use",
    )
    parser.add_argument(
        "--cache_dir",
        type = str,
        default = ".cache",
        help = "The directory to store the cache files",
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "results",
        help = "The directory to store the output files",
    )
    parser.add_argument(
        "--save_freq",
        type = int,
        default = 10,
        help = "The frequency to save the output files",
    )
    parser.add_argument(
        "--prompt_template",
        type = str,
        choices = ["input_no_meta", "input_date", "input_url", "input_url_1", "input_url_2", "input_rank", "input_rank_no_google", "input_emphasize_url", "input_date_today", "input_emphasize_wiki_url", "input_emphasize_wiki_url_1", "input_emphasize_wiki_url_2", "wiki_wordpress_url", "cnn_naturalnews_url", "input_emphasize_src"],
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

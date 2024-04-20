#!/usr/bin/env python3
from utils.prompts import prompt_class
from utils.language_models import create_llm
from utils.customize_meta_data import change_url, change_ranking, change_date 
import logging
import argparse
import json
import os
import torch
import random
from tqdm import tqdm
from transformers import set_seed
from colorama import Fore, Style
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def check_args(args):
    if args.modify_meta_data:
        if args.prompt_template not in ["input_date", "input_url", "input_rank", "input_emphasize_url"]:
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
    args.output_dir = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.prompt_template}_{args.favored_stance}.json")


    # Load model
    llm = create_llm(
        model_name = args.model_name,
        cache_file = os.path.join(args.cache_dir, args.model_name.split('/')[-1]),
        sampling_params = sampling_params,
        tensor_parallel_size = 2,
        openai_api_key = args.openai_api_key,
        google_api_key = args.google_api_key,
        claude_api_key = args.claude_api_key,
    )
    
    # Generate
    results = []
    with open(output_file, "w") as f:
        for i, instance in tqdm(enumerate(data)):
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
            yes_index = [i for i in range(len(instance['stance'])) if instance['stance'][i] == 'yes']
            no_index = [i for i in range(len(instance['stance'])) if instance['stance'][i] == 'no']
            if len(yes_index) == 0 or len(no_index) == 0:
                continue
            yes_index = random.choice(yes_index)
            no_index = random.choice(no_index)
            permutation = []
            for idx_set in [(yes_index, no_index), (no_index, yes_index)]:
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
                    if args.prompt_template == "input_date":
                        fill_in_dict = change_date(
                            fill_in_dict, 
                            args.favored_stance, 
                            "2024-04-01", 
                            "2020-04-01",
                        )
                    if args.prompt_template == "input_url":
                        fill_in_dict = change_url(
                            fill_in_dict, 
                            "https://en.wikipedia.org/wiki/",
                            args.favored_stance,
                        )
                    if args.prompt_template == "input_rank":
                        fill_in_dict = change_ranking(
                            fill_in_dict,
                            args.favored_stance,
                            1,
                            5,
                        )
                
                input_prompt = getattr(prompt_class, args.prompt_template)
                search_engine_input = input_prompt.format(
                    **fill_in_dict,
                )
                response = llm.generate(search_engine_input)
                permutation.append(response)
            
            results.append(permutation)

            # Save cache
            llm.save_cache()
            if i % args.save_freq == 0 or i == len(data) - 1:
                json.dump(
                    {"args": vars(args), "data": results},
                    f, 
                    indent=4
                )
            
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
        # Only some options
        choices = ["input_no_meta", "input_date", "input_url", "input_rank", "input_emphasize_url"],
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
    args = parser.parse_args()
    main(args)

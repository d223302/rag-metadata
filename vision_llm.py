#!/usr/bin/env python3
from utils.prompts import prompt_class
from utils.language_models import create_llm
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


def main(args):

    # Fix all seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    # Load dataset
    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
    }

    # Prepare the output directory
    args.output_dir = os.path.join(
        args.output_dir,
        'generate' if args.generation else 'classify',
        f"{args.model_name.split('/')[-1]}",
        args.prompt_template,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, 
        f"yes_{args.yes_html_template}_no_{args.no_html_template}.json"
    )

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
        vllm = False,
    )
    

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

        img_1 = f"{args.image_dir}/yes_{args.yes_html_template}_{i}.png"
        img_2 = f"{args.image_dir}/no_{args.no_html_template}_{i}.png"

        permutation = []
        for permute_idx, (img_set, doc_set) in enumerate(
            zip(
                [(img_1, img_2), (img_2, img_1)],
                [(yes_index, no_index), (no_index, yes_index)]
            )
        ):
            first_img, second_img = img_set
            doc_1_idx, doc_2_idx = doc_set
            
            TITLE_1 = instance['titles'][doc_1_idx]
            TEXT_1 = instance['text_window'][doc_1_idx]
            TITLE_2 = instance['titles'][doc_2_idx]
            TEXT_2 = instance['text_window'][doc_2_idx]


            search_engine_input = getattr(prompt_class, args.prompt_template)[:]
            if args.prompt_template == "vision_prompts_with_text":
                search_engine_input[0] = search_engine_input[0].format(TITLE_1 = TITLE_1, TEXT_1 = TEXT_1)
                search_engine_input[1] = search_engine_input[1].format(TITLE_2 = TITLE_2, TEXT_2 = TEXT_2)
            
            search_engine_input[2] = search_engine_input[2].format(search_query = question)

            if args.generation:
                search_engine_input[2] = search_engine_input[2].replace(
                    '''Answer with only "Yes" or "No". Do not answer with anything other than "Yes" or "No".''',
                    'Give a concise answer with explanations.'
                )
                if args.credibility:
                    search_engine_input[2] = search_engine_input[2].replace(
                        'Using these two websites, answer the following question',
                        'The two websites are for you to answer the following question'
                    )
                    search_engine_input[2] = search_engine_input[2].replace(
                        'Give a concise answer with explanations.',
                        '''Instead of answering the above question, please tell me which website is more credible and briefly explain why. Your answer must include the sentence "Website 1 is more credible" or "Website 2 is more credible". If you cannot determine which source is more credible, please answer "I cannot determine which website is more credible".'''
                    )
                
            if args.generation:
                response = llm.full_generate_with_img(
                    search_engine_input[0],
                    first_img,
                    search_engine_input[1],
                    second_img,
                    search_engine_input[2],
                )
            else:
                response = llm.generate_with_img(
                    search_engine_input[0],
                    first_img,
                    search_engine_input[1],
                    second_img,
                    search_engine_input[2],
                )
            permutation.append(response)
            if i == 0 or i == 10:
                for x in search_engine_input:
                    print(x)
                print(img_1)
                print(img_2)
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
        
    # if args.generation:
    #     return
    # Summarize result
    yes_count = 0
    no_count = 0
    for perm in results:
        for response in perm:
            if "yes" in response.lower()[:10]:
                yes_count += 1
            elif "no" in response.lower()[:10]:
                no_count += 1
    logger.info(f"Yes template: {Fore.GREEN} {args.yes_html_template} {Style.RESET_ALL}")
    logger.info(f"No template: {Fore.RED} {args.no_html_template} {Style.RESET_ALL}")
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
        "--yes_html_template",
        type = str,
        choices = ["simple", "pretty", "photo"],
    )
    parser.add_argument(
        "--no_html_template",
        type = str,
        choices = ["simple", "pretty", "photo"],
    )
    parser.add_argument(
        '--generation',
        action = 'store_true',
        help = "If true, the model will generate the response using greedy decoding. Otherwise, we will only calculate the probability of yes/no."
    )
    parser.add_argument(
        '--credibility',
        action = 'store_true',
        help = "If true, we will query the LLM by asking which document is more credible."
    )
    parser.add_argument(
        '--image_dir',
        type = str,
        default = "data/imgs",
        help = "The directory where the images are stored",
    )
    parser.add_argument(
        '--temperature',
        type = float,
        default = 1.0,
        help = "The temperature to use for sampling",
    )
    parser.add_argument(
        '--top_p',
        type = float,
        default = 0.95,
        help = "The top_p to use for sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type = int,
        default = 10,
        help = "The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--prompt_template",
        choices = ["vision_prompts_with_text", "vision_prompts"],
        type = str,
        default = "",
    )
    args = parser.parse_args()
    main(args)

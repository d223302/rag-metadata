#!/usr/bin/env python3
from utils.prompts import llm_input
from utils.language_models import create_llm
import logging
import argparse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    # Load dataset
    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    # Load model
    llm = create_llm(args.model_name)

    # Generate
    with open(args.output_dir, "w") as f:
        for i, instance in enumerate(data):
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
            # TODO: start here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for metadata ablation study")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
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
    args = parser.parse_args()
    main(args)

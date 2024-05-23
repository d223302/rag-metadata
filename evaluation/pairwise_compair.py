
evaluation_system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. The two assistants are given a question and two documents retrieved from the Web. You should choose the assistant whose response allows the user to better understand the two retrieved documents and provides the user a more comprehensive view of the question and documents. It is not necessary that the assistant gives a final answer; the explanations and reasoning provided by the assistant is more important. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Carefully distinguish the two responses and differences, and use those differences that can be used to distinguish the quality of the two responses. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

evaluation_user_prompt_template = "[User Question, Instructions, and Retrieved Documents]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"




#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.prompts import prompt_class
from utils.language_models import OpenAIModel
from utils.customize_meta_data import change_url, change_ranking, change_date , change_url_to_wiki, wiki_wordpress_url, cnn_naturalnews_url 
import logging
import argparse
import json
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


def evaluate_pairwise(
    dataset_path,
    response_a_file,
    response_b_file,
    prompt_template,
    favored_stance,
    modify_meta_data,
    wiki_files = "../utils/searched_wiki_urls_new.json",
    seed = 42,
    url_modifier = "",
    num_samples = -1,
    ):

    sampling_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": args.seed,
    }

    judge = OpenAIModel(
        model_name = 'gpt-4-turbo-2024-04-09',
        cache_file = os.path.join(
                '../.cache/', 
                'gpt-4-turbo-2024-04-09',
        ), 
        sampling_params = sampling_params,
        openai_api_key = '../api/openai.txt'
    )
    # Fix all seeds
    random.seed(seed)

    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Load the result file
    with open(response_a_file, "r") as f:
        logger.info(f"Loading the model before DPO: {response_a_file}")
        response_a_list = json.load(f)['data']
    with open(response_b_file, "r") as f:
        logger.info(f"Loading the model after DPO: {response_b_file}")
        response_b_list = json.load(f)['data']

    assert len(response_a_list) == len(response_b_list) == len(data), f"Lengths of the data files do not match. {len(response_a_list)}, {len(response_b_list)}, {len(data)}"

    
    # Load wiki url
    if "input_emphasize_wiki_url" in prompt_template:
        wiki_urls = json.load(open(wiki_files, "r"))
        assert len(wiki_urls) == len(data)


    judge_result = []


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

        # Here, I fix and only select one permutation for simplicity
        doc_1_idx, doc_2_idx = yes_index, no_index
        
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
        # Here, I fix and only select one permutation for simplicity
        response_a = response_a_list[i][0]
        response_b = response_b_list[i][0]

        # Get the evaluation from the judge
        explanations = []
        verdicts = []
        for permutation_idx, (first_response, second_response) in enumerate([(response_a, response_b), (response_b, response_a)]):
            judge_input = evaluation_user_prompt_template.format(
                question = search_engine_input,
                answer_a = first_response,
                answer_b = second_response,
            )
            judge_output = judge.generate(
                user_prompt = judge_input,
                system_prompt = evaluation_system_prompt,
            )
            explanations.append(judge_output)
            if "[[A]]" in judge_output:
                verdicts.append("A")
            elif "[[B]]" in judge_output:
                verdicts.append("B")
            elif "[[C]]" in judge_output:
                verdicts.append("C")
            else:
                verdicts.append("N/A")

        judge_result.append({
            "search_query": question,
            "search_engine_input": search_engine_input,
            "response_a": response_a,
            "response_b": response_b,
            "verdicts": verdicts,
            "explanations": explanations,
            "consistent": verdicts[0] == verdicts[1],
        })

        judge.save_cache()

        response_a_win_rate = sum([1 for judge in judge_result if judge['verdicts'][0] == "A"]) / len(judge_result)
        response_b_win_rate = sum([1 for judge in judge_result if judge['verdicts'][0] == "B"]) / len(judge_result)
        tie_rate = sum([1 for judge in judge_result if judge['verdicts'][0] == "C"]) / len(judge_result)
        pbar.set_description(f"Response A win rate: {response_a_win_rate:.2f}, Response B win rate: {response_b_win_rate:.2f}, Tie rate: {tie_rate:.2f}")


        if i == 0:
            print(search_engine_input)
            print("=" * 20)
            
        if num_samples > 0 and i >= num_samples:
            return judge_result
    
    return judge_result

def main(args):
    
    # Prepare the output file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = os.path.join(
        args.output_dir,
        f"{args.model_name_before_dpo.split('/')[-1]}_{args.model_name_after_dpo.split('/')[-1]}_{args.prompt_template}_{args.url_modifier}_{args.favored_stance}.json"
    )
    if os.path.exists(output_file):
        logger.info(f"The output file already exists: {output_file}. Exiting...")
        return

    
    response_a_file = os.path.join(
        args.result_prefix,
        args.model_name_before_dpo.split("/")[-1],
        f"{args.prompt_template}_{args.url_modifier}_{args.favored_stance}.json" if args.url_modifier != "" else f"{args.prompt_template}_{args.favored_stance}.json"
    )
    response_b_file = os.path.join(
        args.result_prefix,
        args.model_name_after_dpo.split("/")[-1],
        f"{args.prompt_template}_{args.url_modifier}_{args.favored_stance}.json" if args.url_modifier != "" else f"{args.prompt_template}_{args.favored_stance}.json"
    )

    evaluation_result = evaluate_pairwise(
        dataset_path = args.dataset_path,
        response_a_file = response_a_file,
        response_b_file = response_b_file,
        prompt_template = args.prompt_template,
        favored_stance = args.favored_stance,
        modify_meta_data = args.modify_meta_data,
        seed = 42,
        url_modifier = args.url_modifier,
        num_samples = args.num_samples,
    )

    # Save the result
    with open(output_file, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "evaluation_result": evaluation_result,
            },
            f,
            indent=4,
        )

    model_a_win_rate = sum([1 for judge in evaluation_result if judge['verdicts'][0] == "A" and judge['consistent']]) / len(evaluation_result)
    model_b_win_rate = sum([1 for judge in evaluation_result if judge['verdicts'][0] == "B" and judge['consistent']]) / len(evaluation_result)
    tie_rate = sum([1 for judge in evaluation_result if judge['verdicts'][0] == "C" and judge['consistent']]) / len(evaluation_result)
    inconsistent_rate = sum([1 for judge in evaluation_result if not judge['consistent']]) / len(evaluation_result)

    logger.info(f"Judgement finished.")
    logger.info(f"Model A: Before DPO, Model B: After DPO")
    logger.info(f"Model A win rate: {model_a_win_rate:.2f}, Model B win rate: {model_b_win_rate:.2f}, Tie rate: {tie_rate:.2f}")
    logger.info(f"Inconsistent rate: {inconsistent_rate:.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the DPO dataset for the LLM model.")
    parser.add_argument("--model_name_before_dpo", type=str, default="allenai/tulu-2-dpo-7b", help = "The path or the model name of the model before DPO")
    parser.add_argument("--model_name_after_dpo", type=str, default="../dpo_output", help = "The path or the model name of the model after DPO")
    parser.add_argument("--dataset_path",
        type = str,
        default = "data.json",
        help = "The path to the dataset file",
    )
    parser.add_argument(
        "--result_prefix",
        type = str,
        default = "../results_fake/generate",
        help = "The prefix of the result files, where the model's responses are saved",
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "The random seed to use. Make sure the seed is the same as the seed you use in main.py",
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "evaluation_results",
        help = "The directory to store the output files",
    )
    parser.add_argument(
        "--prompt_template",
        type = str,
        choices = ["input_no_meta", "input_date", "input_url", "input_url_1", "input_url_2", "input_rank", "input_emphasize_url", "input_date_today", "input_emphasize_wiki_url", "input_emphasize_wiki_url_1", "input_emphasize_wiki_url_2", "input_emphasize_src"],
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
        '--url_modifier',
        type = str,
        default = "",
        help = "The url modifier to use for the input_url prompt template. If empty, the default url modifier is used.",
    )
    parser.add_argument(
        '--num_samples',
        type = int,
        default = -1,
    )
    args = parser.parse_args()
    main(args)


'''
Usage: 
python3 pairwise_compair.py \
    --dataset_path ../data/fake_knowledge_with_evidence_parsed.json \
    --result_prefix ../results_fake/generate \
    --output_dir evaluation_results \
    --prompt_template input_emphasize_url \
    --favored_stance no \
    --modify_meta_data 1 \
    --url_modifier cnn_naturalnews_url \
    --num_samples 3
'''
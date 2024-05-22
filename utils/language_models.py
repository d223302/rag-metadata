import pickle
import os
import time
from openai import OpenAI
import google.generativeai as genai
from colorama import Fore, Style
import anthropic
import logging
from filelock import FileLock
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import signal

logger = logging.getLogger(__name__)


class LM():
    def __init__(self, model_name, cache_file, sampling_params, **kwargs):
        if '.pkl' not in cache_file:
            cache_file = cache_file + ".pkl"
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model_name = model_name
        self.sampling_params = sampling_params
        self.input_token_count = 0
        self.output_token_count = 0
        self.input_token_cost = 0
        self.output_token_cost = 0

    def save_cache(self):
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        # Temporary file to write cache safely
        temp_file = self.cache_file + ".temp"

        def handle_interrupt(signum, frame):
            print("Interrupted, please wait for the cache to save properly!")
            signal.signal(signal.SIGINT, original_sigint_handler)

        original_sigint_handler = signal.signal(signal.SIGINT, handle_interrupt)

        while True:
            try:
                with FileLock(self.cache_file + ".lock", timeout=10):
                    with open(temp_file, "wb") as f:
                        pickle.dump(self.cache_dict, f)
                    os.replace(temp_file, self.cache_file)
                break
            except Exception as e:
                print(f"Pickle Error: {e}. Retry in 5sec...")
                time.sleep(5)

        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)

    def load_cache(self, allow_retry=True):
        logger.debug(f"Loading cache from {self.cache_file}")
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache
    

    # TODO: Add a function to backup cache


    def generate(self, user_prompt):
        raise NotImplementedError("generate method must be implemented in derived class")

    def summarize_cost(self):
        logger.info("=" * 30)
        logger.info(f"Model: {Fore.GREEN}{self.model_name}{Style.RESET_ALL}")
        logger.info(f"Input token count: {self.input_token_count}")
        logger.info(f"Output token count: {self.output_token_count}")
        logger.info(f"Input token cost: {(self.input_token_count * self.input_token_cost):.3f}")
        logger.info(f"Output token cost: {(self.output_token_count * self.output_token_cost):.3f}")
        logger.info("=" * 30)




# TODO ends

class TransformerLM(LM):
    token_mapping = {
        "meta-llama/Meta-Llama-3-8B-Instruct": {"Yes": 9642, "No": 2822},
        "meta-llama/Meta-Llama-3-70B-Instruct": {"Yes": 9642, "No": 2822},
        "meta-llama/Llama-2-7b-chat-hf": {"Yes": 3869, "No": 1939},
        "meta-llama/Llama-2-13b-chat-hf": {"Yes": 3869, "No": 1939},
        "meta-llama/Llama-2-70b-chat-hf": {"Yes": 3869, "No": 1939},
        "allenai/tulu-2-dpo-7b": {"Yes": 8241, "No": 3782},
        "allenai/tulu-2-dpo-13b": {"Yes": 8241, "No": 3782},
        "allenai/tulu-2-dpo-70b": {"Yes": 8241, "No": 3782},
        "dpo_output": {"Yes": 8241, "No": 3782},
    }

    prompt_suffix = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "meta-llama/Meta-Llama-3-70B-Instruct": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "meta-llama/Llama-2-7b-chat-hf": " ",
        "meta-llama/Llama-2-13b-chat-hf": " ",
        "meta-llama/Llama-2-70b-chat-hf": " ",
        "allenai/tulu-2-dpo-7b": "<|assistant|>\n",
        "allenai/tulu-2-dpo-13b": "<|assistant|>\n",
        "allenai/tulu-2-dpo-70b": "<|assistant|>\n",
        "dpo_output": "<|assistant|>\n",
    }

    def __init__(self, tensor_parallel_size, **kwargs):
        super().__init__(**kwargs)
        # Try to use unsloth to load the model
        try:
            from unsloth import FastLanguageModel
            max_seq_length = 8192 # Supports automatic RoPE Scaling, so choose any number.

            # Load model
            self.llm, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = "dpo_output",
                max_seq_length = max_seq_length,
                dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
                load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
            )

        except ImportError:
            self.llm = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = 'auto')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.use_default_system_prompt = False

        self.yes_idx = self.token_mapping[self.model_name]["Yes"]
        self.no_idx = self.token_mapping[self.model_name]["No"]

        for response in ["Yes", "No"]:
            dialogue = [
                    {"role": "user", "content": "Tell me yes or no."},
                    {"role": "assistant", "content": response},
            ]
            prompt = self.tokenizer.apply_chat_template(dialogue, tokenize = False)
            input_tokens = self.tokenizer.encode(prompt, add_special_tokens = False)
            num_tokens = len(input_tokens)
            print('\n' + '=' * 20)
            print(f"Prompt: {prompt}")
            for i in range(num_tokens -5, num_tokens):
                print(f"Response: {response}, token: {self.tokenizer.convert_ids_to_tokens(input_tokens[i])}, token index: {input_tokens[i]}")
            print(f"Current token_mapping: {self.token_mapping[self.model_name][response]}")

        # The following line need to be removed when automatically running the code
        # input("Check if the above tokenization and Yes/No mapping is desired. Press Enter to continue...")

        self.generation_params = GenerationConfig(
            do_sample = False,
            max_new_tokens = 5,
            temperature = None,
            top_p = None,
            pad_token_id = self.tokenizer.eos_token_id,
        )

        
    def generate(self, user_prompt):
        # If the tokenizer has apply_chat_template method, apply it

        if hasattr(self.tokenizer, "apply_chat_template"):
            dialogue = [
                {"role": "user", "content": user_prompt.strip()},
            ]
            prompt = self.tokenizer.apply_chat_template(dialogue, tokenize = False)
            prompt += self.prompt_suffix[self.model_name]
        else:
            raise NotImplementedError(f"apply_chat_template method must be implemented in tokenizer class for model: {self.model_name}")
        # print(prompt + "PROMPT_ENDS_HERE")
        # print("========")
        if prompt in self.cache_dict:
            return self.cache_dict[prompt]
        else:
            input_tokens = self.tokenizer(prompt, return_tensors = "pt", add_special_tokens = False).input_ids
            # TODO: Since we only care about the probability of Yes/No, we can use the last token as the output

            with torch.no_grad():
                output = self.llm(
                    input_tokens.to(self.llm.device),
                    #max_new_tokens = 3,
                )
            # print(f"Output logits shape: {output.logits.shape}")
            last_token_logits = output.logits[0, -1, :]
            # print(f"{Fore.CYAN} {self.tokenizer.convert_ids_to_tokens([last_token_logits.argmax()])} {Style.RESET_ALL} (argmax: {last_token_logits.argmax()})")
            # print(f"Yes logit: {last_token_logits[self.yes_idx]}, No logit: {last_token_logits[self.no_idx]}")
            prediction = "Yes" if last_token_logits[self.yes_idx] > last_token_logits[self.no_idx] else "No"
            # print(f"Prediction based on probability: {prediction}")

            # Add the prompt to the cache
            self.cache_dict[prompt] = prediction
            return prediction
        
    def full_generate(self, user_prompt, max_new_tokens = 512):
        if hasattr(self.tokenizer, "apply_chat_template"):
            dialogue = [
                {"role": "user", "content": user_prompt.strip()},
            ]
            prompt = self.tokenizer.apply_chat_template(dialogue, tokenize = False)
            prompt += self.prompt_suffix[self.model_name]
        else:
            raise NotImplementedError(f"apply_chat_template method must be implemented in tokenizer class for model: {self.model_name}")
        # print(prompt + "PROMPT_ENDS_HERE")
        # print("========")

        past_key_values = None
        # First pass to get the CoT answer
        if prompt in self.cache_dict:
            return self.cache_dict[prompt]
        else:
            input_tokens = self.tokenizer(prompt, return_tensors = "pt", add_special_tokens = False).input_ids
            # TODO: Since we only care about the probability of Yes/No, we can use the last token as the output

            with torch.no_grad():
                output = self.llm.generate(
                    input_tokens.to(self.llm.device),
                    max_new_tokens = max_new_tokens,
                    # use_cache = True,
                )
            # full_output_tokens = output[0].detach().cpu().numpy()
            output_tokens = output[0][len(input_tokens[0]):].detach().cpu().numpy()
            decoded_output = self.tokenizer.decode(output_tokens, skip_special_tokens = True)
            past_key_values = None
            self.cache_dict[prompt] = decoded_output
        
        # TODO: Check if we need to use the second pass to get the yes/no prediction
        return decoded_output
    
    def apply_chat_template(self, user_prompt):
        raise NotImplementedError("apply_chat_template method must be implemented in derived class")


class OpenAIModel(LM):
    def __init__(self, openai_api_key, **kwargs):
        super().__init__(**kwargs)
        with open(openai_api_key, "r") as f:
            self.api_key = f.readlines()[0].strip()
        self.llm = OpenAI(api_key = self.api_key)
        if self.model_name == "gpt-4-32k":
            raise ValueError("Please do not use gpt-4-32k, it is too expensive to use")
        
        if "3.5" in self.model_name:
            self.input_token_cost = 1.50 / 1_000_000
            self.output_token_cost = 2.00 / 1_000_000
        elif "gpt-4-turbo" in self.model_name:
            self.input_token_cost = 10.0 / 1_000_000
            self.output_token_cost = 30.0 / 1_000_000
        elif "gpt-4" in self.model_name:
            self.input_token_cost = 30.0 / 1_000_000
            self.output_token_cost = 60.0 / 1_000_000
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
    def generate(self, user_prompt, system_prompt = None):
        #  print(user_prompt)
        if system_prompt is not None:
            messages = [
                {"role" : "system", "content" : system_prompt}, 
                {"role" : "user", "content" : user_prompt}
            ]
        else:
            messages = [{"role" : "user", "content" : user_prompt}]
        key = user_prompt
        if key in self.cache_dict:
            return self.cache_dict[key]
        else:
            outputs = self.llm.chat.completions.create(
                model = self.model_name,
                messages = messages,
                **self.sampling_params,
            )
            self.input_token_count += outputs.usage.prompt_tokens
            self.output_token_count += outputs.usage.completion_tokens
            response = outputs.choices[0].message.content
            self.cache_dict[key] = response
            self.save_cache()
            return response

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class GeminiModel(LM):
    def __init__(self, google_api_key, **kwargs):
        super().__init__(**kwargs)
        with open(google_api_key, "r") as f:
            self.api_key = f.readlines()[0].strip()
        genai.configure(api_key = self.api_key)
        self.llm = genai.GenerativeModel(self.model_name)
        self.sampling_params.pop("seed")

    def generate(self, user_prompt):
        contents = []
        contents.append({"role" : "user", "parts" :user_prompt})
        
        key = user_prompt
        if key in self.cache_dict:
            return self.cache_dict[key]
        else:
            outputs = self.llm.generate_content(
                contents = contents,
                safety_settings = safety_settings,
                generation_config = genai.GenerationConfig(**self.sampling_params),
            )
            # TODO: Add gemini token count and cost estimate
            response = outputs.text
            self.cache_dict[key] = response
            return response
        
class ClaudeLLM(LM):
    def __init__(self, claude_api_key, **kwargs):
        super().__init__(**kwargs)
        with open(claude_api_key, "r") as f:
            self.api_key = f.readlines()[0].strip()
        self.llm = anthropic.Anthropic(api_key = self.api_key)
        self.sampling_params.pop("seed")
    
        if "claude-3-opus" in self.model_name:
            self.input_token_cost = 15 / 1_000_000
            self.output_token_cost = 75 / 1_000_000
        elif "claude-3-sonnet" in self.model_name:
            self.input_token_cost = 3 / 1_000_000
            self.output_token_cost = 15 / 1_000_000
        elif "claude-3-haiku" in self.model_name:
            self.input_token_cost = 0.25 / 1_000_000
            self.output_token_cost = 1.25 / 1_000_000
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
    def generate(self, user_prompt):
        contents = []
        contents.append({"role" : "user", "content" : user_prompt})
        key = user_prompt
        if key in self.cache_dict:
            return self.cache_dict[key]
        else:
            while True:
                outputs = self.llm.messages.create(
                    model = self.model_name,
                    messages = contents,
                    max_tokens = 10, # TODO: Do not hardcode this
                    **self.sampling_params,
                )
                self.input_token_count += outputs.usage.input_tokens
                self.output_token_count += outputs.usage.output_tokens
                response = outputs.content[0].text
                self.cache_dict[key] = response
                break

            return response

def create_llm(model_name, cache_file, sampling_params, **kwargs):
    if "gemini" in model_name:
        return GeminiModel(
            model_name = model_name, 
            cache_file = cache_file, 
            sampling_params = sampling_params,
            **kwargs
        )
    elif "gpt" in model_name:
        return OpenAIModel(
            model_name = model_name, 
            cache_file = cache_file, 
            sampling_params = sampling_params, 
            **kwargs
        )
    elif "claude" in model_name:
        return ClaudeLLM(
            model_name = model_name, 
            cache_file = cache_file, 
            sampling_params = sampling_params, 
            **kwargs
        )
    else:
        return TransformerLM(
            model_name = model_name, 
            cache_file = cache_file, 
            sampling_params = sampling_params, 
            **kwargs
        )

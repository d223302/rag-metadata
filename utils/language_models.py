from vllm import LLM, SamplingParams
import pickle
import os
import time
from openai import OpenAI
import google.generativeai as genai
from colorama import Fore, Style
import anthropic
import logging
from filelock import FileLock
import lade
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from ouroboros import ouroboros
from ouroboros.models import LlamaForCausalLM


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

        while True:
            try:
                with FileLock(self.cache_file + ".lock", timeout = 10):
                    with open(self.cache_file, "wb") as f:
                        pickle.dump(self.cache_dict, f)
                break
            except Exception:
                print ("Pickle Error: Retry in 5sec...")
                time.sleep(5)


    def load_cache(self, allow_retry=True):
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

class VLLMModel(LM):
    def __init__(self, tensor_parallel_size, **kwargs):
        super().__init__(**kwargs)
        self.llm = LLM(model = self.model_name, tensor_parallel_size = tensor_parallel_size)
        self.sampling_params = SamplingParams(**self.sampling_params)
        
    def generate(self, user_prompt):
        # TODO: Add the dialogue history to the prompt
        prompt = self.apply_chat_template(user_prompt)
        if prompt in self.cache_dict:
            return self.cache_dict[prompt]
        else:
            output = self.llm.generate(
                prompt, 
                self.sampling_params,
            )
            generation = output.outputs[0].text
            self.cache_dict[prompt] = generation
            return generation
    
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
        elif "gpt-4-preview" in self.model_name:
            self.input_token_cost = 10.0 / 1_000_000
            self.output_token_cost = 30.0 / 1_000_000
        elif "gpt-4" in self.model_name:
            self.input_token_cost = 30.0 / 1_000_000
            self.output_token_cost = 60.0 / 1_000_000
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
    def generate(self, user_prompt):
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
        return VLLMModel(
            model_name = model_name, 
            cache_file = cache_file, 
            sampling_params = sampling_params, 
            **kwargs
        )

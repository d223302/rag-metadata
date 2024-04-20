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
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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

class TransformerLM(LM):
    def __init__(self, tensor_parallel_size, **kwargs):
        super().__init__(**kwargs)
        self.llm = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = 'auto')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.use_default_system_prompt = False
        self.yes_idx = self.tokenizer.encode("Yes", add_special_tokens = False)
        self.no_idx = self.tokenizer.encode("No", add_special_tokens = False)
        # print(f"yes: {self.yes_idx}, no: {self.no_idx}")
        assert len(self.yes_idx) == 1 and len(self.no_idx) == 1, f"Yes/No tokens must be unique, but got {self.yes_idx} and {self.no_idx}"
        # Check how the space before word is handled
        if len(self.tokenizer.encode(" Yes", add_special_tokens = False)) != 2:
            raise ValueError(f"Space before Yes/No is not handled correctly. ' Yes' is tokenized into {self.tokenizer.tokenize(' Yes', add_special_tokens = False)}")
        self.yes_idx = self.yes_idx[0]
        self.no_idx = self.no_idx[0]
        self.generation_params = GenerationConfig(
            do_sample = False,
            max_new_tokens = 5,
            temperature = None,
            top_p = None,
        )

        
    def generate(self, user_prompt):
        # If the tokenizer has apply_chat_template method, apply it

        if hasattr(self.tokenizer, "apply_chat_template"):
            dialogue = [
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(dialogue, tokenize = False)
            #print(prompt)
        prompt = prompt.strip() + " "
        # print(prompt)
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

            
            #  # Decode the output
            #  output = self.llm.generate(
            #      input_tokens.to(self.llm.device),
            #      generation_config = self.generation_params,
            #  )
            #  output = output[0][len(input_tokens[0]):].detach().cpu().numpy()
            #  decoded_output = self.tokenizer.decode(output, skip_special_tokens = True)
            #  generation = decoded_output
            #  print(f"{Fore.GREEN} {generation} {Style.RESET_ALL} (generation)")

            # Add the prompt to the cache
            self.cache_dict[prompt] = prediction
            return prediction
    
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
        return TransformerLM(
            model_name = model_name, 
            cache_file = cache_file, 
            sampling_params = sampling_params, 
            **kwargs
        )

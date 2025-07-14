import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import json

from data_preprocessing import JokeDataPreprocessor
from model_config import (
    ModelConfig, ModelSetup, QLoRAConfig, LoraConfig, TrainingConfig, get_training_args
)

class JokeTrainer:

    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
                 output_dir: str = "./joke_model_output",
                 use_wandb: bool = True):
        self_model_name = model_name
        self.output_dir = output_dir
        self.use_wandb = use_wandb

        self.model_config = ModelConfig(model_name=model_name)
        self.qlora_config = QLoRAConfig()
        self.lora_config = LoraConfig()
        self.training_config = TrainingConfig(output_dir=output_dir)

        self.model = None
        self.tokenizer = None
        self.trainer = None
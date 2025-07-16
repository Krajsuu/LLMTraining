import os
import torch 
from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from huggingface_hub import login
from datetime import datetime
import json

from data_preprocessing import JokeDataPreprocessor
from model_config import (
    ModelConfig, ModelSetup, QLoRAConfig, MyLoRAConfig, TrainingConfig, get_training_args
)
class JokeTrainer:

    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
                 output_dir: str = "./joke_model_output",
                 use_wandb: bool = True):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_wandb = use_wandb

        self.model_config = ModelConfig(model_name=model_name)
        self.qlora_config = QLoRAConfig()
        self.lora_config = MyLoRAConfig()
        self.training_config = TrainingConfig(output_dir=output_dir)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def prepare_dataset(self, dataset_path: str = "JonaszPotoniec/dowcipy-polish-jokes-dataset",
                        min_upvotes: int = 15, min_ratio: float = 2.0, save_preprocessed: bool = True):
        preprocessor = JokeDataPreprocessor(
            min_upvotes=min_upvotes,
            min_ratio=min_ratio,
            min_length=50,
            max_length=2000
        )

        save_path = "filtered_jokes.csv" if save_preprocessed else None

        training_dataset = preprocessor.preprocess_full_pipeline(dataset_path, save_path)
        if len(training_dataset) == 0:
            raise ValueError("Lack of data in training dataset")
        dataset = Dataset.from_list(training_dataset) #konwersja na huggingface dataset
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset):
        def format_chat_template(example):
            formatted_text = self.tokenizer.apply_chat_template(
                example['messages'],
                tokenize=False,
                add_generation_prompt = False
            )
            return {"text" : formatted_text}
        
        def tokenize_function(examples):
            toknized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=2048
            )
            return toknized
        
        dataset = dataset.map(format_chat_template)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        return tokenized_dataset

    def setup_model_and_tokenizer(self):
        model_setup = ModelSetup(
            model_config=self.model_config,
            qlora_config=self.qlora_config,
            lora_config=self.lora_config
        )
        self.model, self.tokenizer = model_setup.setup_full_model()

    def create_trainer(self, tokenized_dataset: Dataset):
        training_args = get_training_args(self.training_config)
        response_template= "<|start_header_id|>assistant<|end_header_id|>"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer = self.tokenizer
        )
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=self.tokenizer,
            data_collator=collator,
        )

        return self.trainer
    
    def train(self):
        train_result = self.trainer.train()
        return train_result
    
    def save_model(self,path: str = None):
        if path is None:
            path= os.path.join(self.output_dir,"final_model")

        self.trainer.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        config_path = os.path.join(path,"traning_config.json")
        with open(config_path,'w',encoding='utf-8') as f:
            config_dict={
                "model_config" : self.model_config.__dict__,
                "lora_config" : self.lora_config.__dict__,
                "training_config" : self.training_config.__dict__
            }
            json.dump(config_dict,f,indent=2,default=str)

    def setup_wandb(self, project_name: str = "polish-jokes-llm") -> None:  # Poprawione z "projekt_name"
        if not self.use_wandb:
            return
        
        run_name = f"Jokes-{datetime.now().strftime('%m%d-%H%M')}"

        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model_name": self.model_name,
                "lora_r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "learning_rate": self.training_config.learning_rate,
                "batch_size": self.training_config.per_device_train_batch_size,
                "epochs": self.training_config.num_train_epochs
            }
        )

    def setup_wandb(self, projekt_name: str = "polish-jokes-llm") -> None:
      if not self.use_wandb:
        return
      
      run_name = f"Jokes-{datetime.now().strftime('%m%d-%H%M')}"

      wandb.init(
          project=projekt_name,
          name=run_name,
          config ={
              "model_name" :self.model_name,
              "lora_r" : self.lora_config.r,
              "lora_alpha" : self.lora_config.lora_alpha,
              "learning_rate" : self.training_config.learning_rate,
              "batch_size" : self.training_config.per_device_train_batch_size,
              "epochs" : self.training_config.num_train_epochs
          }
      )


    def full_pipeline_training(self, dataset_path: str = "JonaszPotoniec/dowcipy-polish-jokes-dataset",
        min_upvotes: int = 15,
        min_ratio: float = 2.0):
        try:
            if self.use_wandb:
                self.setup_wandb()
            dataset = self.prepare_dataset(dataset_path=dataset_path,
                                           min_upvotes=min_upvotes,
                                           min_ratio=min_ratio)
            self.setup_model_and_tokenizer()
            tokenized_dataset = self.tokenize_dataset(dataset)
            self.create_trainer(tokenized_dataset)
            train_result = self.train()
            model_path = self.save_model()
            if self.use_wandb:
                wandb.finish()
            return model_path
        except Exception as e:
            print(f"Exception : {e}")
            if self.use_wandb:
                wandb.finish()
            raise e

def main():
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    joketrainer = JokeTrainer(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        use_wandb=True
    )
    model_path = joketrainer.full_pipeline_training()


if __name__ == "__main__":
    main()
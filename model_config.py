from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name : str = "meta-llama/Llama-3.2-8B-Instruct"
    cache_dir : Optional[str] = "./models"
    torch_dtype : torch_dtype = torch.float16
    device_map: str = "auto"
    trust_remote_code: bool = True

@dataclass
class QLoRAConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM

    def __post_init__(self):
        if self.target_modules is None : 
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size : int = 2
    gradient_accumulation_steps : int = 4
    learning_rate : float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit : int = 3

    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_pin_memory : bool = True
    remove_unused_columns : bool = True
    evaluation_strategy: str = "steps"
    do_eval: bool = False

class ModelSetup:
    def __init__(self,
                model_config: ModelConfig = None,
                qlora_config: QLoRAConfig = None,
                lora_config: LoraConfig = None):
        self.model_config = model_config or ModelConfig()
        self.qlora_config = qlora_config or QLoRAConfig()
        self.lora_config = lora_config or LoRAConfig()

        self.tokenizer = None
        self.model = None


    def check_gpu(self) -> bool:
        if not torch.cuda.is_available():
            raise RuntimeError("Not available CUDA")
        
        return True
    
    def load_tokenizer(self) -> AutoTokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            cache_dir = self.model_config.cache_dir,
            trust_remote_code = self.model_config.trust_remote_code
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer
    
    def create_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_compute_dtype=self.qlora_config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant
        )
    
    def load_model(self) -> AutoModelForCausalLM:
       bnb_config = self.create_bnb_config

       self.model = AutoModelForCausalLM.from_pretrained(
           self.model_config.model_name,
            quantization_config=bnb_config,
            device_map=self.model_config.device_map,
            cache_dir=self.model_config.cache_dir,
            torch_dtype=self.model_config.torch_dtype,
            trust_remote_code=self.model_config.trust_remote_code
       ) 

       self.model.config.use_cache = False
       return self.model
    
    def setup_lora(self):
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type
        )
        self.model = get_peft_model(self.model, peft_config)
        return self.model
    
    def setup_full_model(self):
        self.check_gpu()

        self.load_tokenizer()
        self.load_model()
        self.setup_lora()

        return self.model, self.tokenizer
    
def get_training_args(config : TrainingConfig = None) -> TrainingArguments:
    if config is None : 
        config = TrainingConfig

    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy=config.evaluation_strategy,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=config.remove_unused_columns,
        do_eval=config.do_eval
    )

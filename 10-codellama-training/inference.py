from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, DataCollatorForSeq2Seq
from dataclasses import dataclass, field
from peft import AutoPeftModelForCausalLM
from typing import Optional
from tqdm import tqdm

import os, torch, pandas as pd, numpy as np

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    adapter_path: str = field(
        default=None,
        metadata={"help": "Path of the adapter to use, if any."}
    )    
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the pretrained model"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name"}
    )
    auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Huggingface token to access the model."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Filepath of the dataset to use."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=128,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size used for inference."},
    )
    output_dir: Optional[str] = field(
        default=".",
        metadata={"help": "Output dir."},
    )
    predictions_filename: Optional[str] = field(
        default="predictions.txt",
        metadata={"help": "Name of the file where to store the model predictions."},
    )

    overwrite_predictions: bool = field(
        default=False, metadata={"help": "Overwrite the alredy existing predictions files."}
    )
    
    save_accuracy_filename: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the file where to store the model accuracy."},
    )

    label: Optional[str] = field(
        default=None,
        metadata={"help": "Label of the dataset."}
    )


def infer_batch(model, tokenizer, input_prompts, max_length=2500):
    # Batched inference
    inputs = tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(**inputs, max_new_tokens=128)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return decoded_outputs
    

def craft_prompt(before_context, after_context, model_type):
    return before_context + "<FILL_ME>" + after_context

def clean_prediction(prediction, model_type):
    clean_pred = prediction.split('<SUF>', 1)[1].split('<MID>', 1)[1]
    return clean_pred.split('<EOT>', 1)[0]

def check_extact_match(prediction, expected_output):
    normalized_prediction = ''.join(prediction.split())
    normalized_expected_prediction = ''.join(expected_output.split())
    return normalized_prediction == normalized_expected_prediction

def check_startswith_match(prediction, expected_output):
    normalized_prediction = ''.join(prediction.split())
    normalized_expected_prediction = ''.join(expected_output.split())
    return normalized_prediction.startswith(normalized_expected_prediction)

if __name__ == '__main__':
    # Read arg parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Print args
    print(f"Model args: {model_args}")
    print(f"\nData args: {data_args}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True, token=model_args.auth_token)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    bos_token = [tokenizer.bos_token_id]

    # Load model
    if model_args.adapter_path:
        model = AutoPeftModelForCausalLM.from_pretrained(model_args.adapter_path, torch_dtype=torch.bfloat16, device_map="auto")
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")

    model.resize_token_embeddings(len(tokenizer))

    # Craft prompt
    df = pd.read_json(data_args.dataset_path, orient='records')
    model_type = 'codellama' 
    df['prompt'] = df.apply(lambda x: craft_prompt(x['parsed_before_context'], x['parsed_after_context'], model_type), axis=1)

    #  Load huggingface dataset and preprocess
    dataset = Dataset.from_pandas(df)
    max_source_length = data_args.max_source_length - data_args.max_target_length - 1 
    def preprocess_single(example):
        input_ids = bos_token + tokenizer(example["prompt"], max_length=max_source_length, truncation=True)["input_ids"]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    predict_dataset = dataset.map(
        preprocess_single,
        num_proc=128,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on prediction dataset",
    )
    print(f"Loaded {len(predict_dataset)} samples for prediction")
    predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size=data_args.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True)
    )

    # Generate
    gen_kwargs = {
        "max_new_tokens": data_args.max_target_length,
        "num_beams": data_args.num_beams,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id
    }
    model.eval()

    output_dir = data_args.output_dir if data_args.output_dir else data_args.adapter_path
    if not os.path.exists(os.path.join(output_dir, f'{data_args.label}_predictions.json')) or data_args.overwrite_predictions:
    
        print(f"Inferencing...")
        predictions = []
        for i, batch in enumerate(tqdm(dataloader)):
            batch.to(model.device)
            out = model.generate(**batch, **gen_kwargs)
            outputs = tokenizer.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
            for output in outputs:
                print(f"Output: {output}")
            predictions.extend(outputs)

        df['prediction'] = predictions
        df['exact_match'] = df.apply(lambda x: check_extact_match(x['prediction'], x['parsed_mask']), axis=1)
        df['startswith_match'] = df.apply(lambda x: check_startswith_match(x['prediction'], x['parsed_mask']), axis=1)

        df = df[['author_id', 'repository', 'date', 'parsed_mask', 'prompt', 'prediction', 'exact_match', 'startswith_match']]

        output_file = os.path.join(output_dir, f'{data_args.label}_predictions.json')
        df.to_json(output_file, orient='records')
    else:
        df = pd.read_json(os.path.join(output_dir, f'{data_args.label}_predictions.json'), orient='records')

    # Compute accuracy for exact_match and startswith_match
    exact_match_accuracy = (df['exact_match'].sum() / len(df)) * 100
    startswith_match_accuracy = (df['startswith_match'].sum() / len(df)) * 100
    print(f"Exact match accuracy: {exact_match_accuracy:.2f}%")
    print(f"Startswith match accuracy: {startswith_match_accuracy:.2f}%")

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch, pandas as pd
from datasets import load_dataset

from transformers import (
    HfArgumentParser,
    T5Tokenizer,
    T5ForConditionalGeneration
)

from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")
torch.manual_seed(0)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    from_flax: bool = field(
        default=False,
        metadata={
            "help": "If true, the model will be loaded from a saved Flax checkpoint."
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
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split: Optional[str] = field(
        default="test", metadata={"help": "The split of the dataset to use (via the datasets library)."}
    )
    source_column: Optional[str] = field(
        default='input',
        metadata={"help": "The name of the column in the datasets containing the input."},
    )
    target_column: Optional[str] = field(
        default='target',
        metadata={"help": "The name of the column in the datasets containing the target."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
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

def check_model_accuracy(targets, predictions):
    assert len(targets) == len(predictions), f"Targets size: {len(targets)} != Predictions size: {len(predictions)}"

    perfect_predictions = 0
    for x,y in zip(targets, predictions):
      x = x.replace('<NL>', '')
      y = y.replace('<NL>', '')

      x = ''.join(x.split())
      y = ''.join(y.split())
      if x == y:
        perfect_predictions += 1

    accuracy = perfect_predictions*100.0/len(targets)    
    return float(accuracy)


def save_prediction_stats(filepath: str, inputs: list, targets: list, predictions: list):
    corrects = list()
    for p, t in zip(predictions, targets):
        p = ''.join(p.replace('<NL>', '').split())
        t = ''.join(t.replace('<NL>', '').split())
        corrects.append(True if p == t else False)
    
    df = pd.DataFrame({'input': inputs, \
                       'target': targets, \
                       'prediction': predictions, \
                       'correct': corrects})
    df.to_csv(filepath, index=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    # Predictions filepath
    out_path = os.path.join(data_args.output_dir, data_args.predictions_filename)

    # Check if predictions file exists
    if not data_args.overwrite_predictions and os.path.exists(out_path):
        print(f"Predictions file already exists for checkpoint {out_path}")
        return
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
	# Print the selected dataset path and checkpoints path
    print("=" * 50)
    print(f"Dataset path: {data_args.dataset_path}")
    print(f"Checkpoints path: {model_args.model_name_or_path}")
    print(f"Predictions path: {out_path}")
    print("=" * 50)

    print(f"Loading model {model_args.model_name_or_path} and tokenizer from {model_args.tokenizer_name}...")
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_flax=model_args.from_flax,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    
    #model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading dataset {data_args.dataset_path}")
    dataset = load_dataset('csv', data_files={data_args.dataset_split: data_args.dataset_path})
    column_names = dataset[data_args.dataset_split].column_names

    # Get the column names for input/target.
    source_column = data_args.source_column
    if source_column not in column_names:
        raise ValueError(
            f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
        )

    target_column = data_args.target_column
    if target_column not in column_names:
        raise ValueError(
            f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
        )
    
    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] is not None and examples[target_column][i] is not None:
                inputs.append(examples[source_column][i])
                targets.append(examples[target_column][i])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)
        return model_inputs
    
    predict_dataset = dataset[data_args.dataset_split].map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on prediction dataset",
    )

    print(f"Example: {predict_dataset[0]}")
    predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=data_args.batch_size)
    gen_kwargs = {
        "max_length": data_args.max_target_length,
        "num_beams": data_args.num_beams,
        "do_sample": False,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    print(f"Inferencing...")
    predictions = []
    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.generate(**batch, **gen_kwargs)
        outputs = tokenizer.batch_decode(out.to(device), skip_special_tokens=True)
        if i == 0:
            print(outputs[:2])
        predictions.extend(outputs)

    # Export predictions on a separate file
    print(f"Writing predictions to {out_path}")
    with open(out_path, 'w', encoding='utf-8', errors='replace') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    assert len(predictions) == len(predict_dataset)
    
    # Export prediction stats
    df = dataset[data_args.dataset_split].to_pandas()
    inputs = df[data_args.source_column].tolist()
    targets = df[data_args.target_column].tolist()
    save_prediction_stats(out_path.replace('.txt', '.csv'), inputs, targets, predictions)
    
    # Print accuracy
    accuracy = check_model_accuracy(targets, predictions)
    print(f'Model accuracy: {str(accuracy)}')
    
    # Store accuracy on a file
    if data_args.save_accuracy_filename is not None:
        print(f"Saving accuracy to {data_args.save_accuracy_filename}")
        with open(data_args.save_accuracy_filename, 'w', encoding='utf-8', errors='replace') as f:
            f.write(f"{accuracy:.2f}")

if __name__ == "__main__":
    main()

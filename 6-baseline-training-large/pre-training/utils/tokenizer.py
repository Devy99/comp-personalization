"""
Script adapted from the official huggingface repository:
https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling
"""

import os, argparse, sentencepiece as spm


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder_name', '-f',
                        metavar='NAME',
                        dest='folder_name',
                        required=False,
                        type=str,
                        default='T5_Configs',
                        help='Name of the folder that will contain the trained tokenizer and the T5 config file')
    parser.add_argument('--max_tokens_length', '-l',
                        metavar='SIZE',
                        dest='max_tokens_length',
                        required=False,
                        type=int,
                        default='512',
                        help='Max sequence length that the model can support. Default to 512.')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', '-i',
                        metavar='PATH',
                        dest='filepath',
                        required=True,
                        type=str,
                        help='Path of the tokenizer training dataset')
     
    return parser

if __name__ == '__main__':   

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Generate output directory
    config_dir = args.folder_name
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    VOCAB_SIZE, VOCAB_PREFIX = 32000, 'tokenizer'
    tokenizer_model_path = os.path.join(f'./{VOCAB_PREFIX}.model')
    tokenizer_vocab_path = os.path.join(f'./{VOCAB_PREFIX}.vocab')

    # Training the tokenizer
    print('Training the tokenizer and building the vocabulary ...')
    with open(args.filepath, "r", encoding='utf-8') as f:
        spm.SentencePieceTrainer.train(sentence_iterator=f,
                                        model_prefix=VOCAB_PREFIX,
                                        pad_id=0,
                                        bos_id=-1,
                                        eos_id=1,
                                        unk_id=2,
                                        character_coverage=1.0,
                                        vocab_size=VOCAB_SIZE,
                                        user_defined_symbols=['<NL>'])
        print(f'Training tokenizer finished, at: {tokenizer_model_path}')

    # Move .model and .vocab files to the T5 config dir
    new_tokenizer_model_path = os.path.join(config_dir, f'{VOCAB_PREFIX}.model')
    os.rename(tokenizer_model_path, new_tokenizer_model_path)
    
    new_tokenizer_vocab_path = os.path.join(config_dir, f'{VOCAB_PREFIX}.vocab')
    os.rename(tokenizer_vocab_path, new_tokenizer_vocab_path)
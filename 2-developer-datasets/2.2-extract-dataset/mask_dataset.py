from utils.consecutive_line_masker import ConsecutiveLineMasker
import argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_filepath', '-o',
                        metavar='FILEPATH',
                        dest='output_filepath',
                        required=False,
                        type=str,
                        default='masked.csv',
                        help='Filepath of the CSV file containing the masked methods of the developer')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input_filepath', '-i',
                        metavar='FILEPATH',
                        dest='input_filepath',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the extracted methods of the developer')
    return parser

def masking_type(string: str):
    return 'block' if string.startswith('[') and string.endswith(']') else 'line'

def extra_columns(dataframe: pd.DataFrame):
    header = ['author_id', 'repository', 'commit_id', 'date', 'filepath', 'type', 'added_line']
    columns = [dataframe['author_id'].tolist(), dataframe['repository'].tolist(), dataframe['commit_id'].tolist(), dataframe['date'].tolist(), dataframe['filepath'].tolist(), dataframe['type'].tolist(), dataframe['added_line'].tolist()]
    return header, columns

def masking_columns(dataframe: pd.DataFrame):
    return dataframe['method'].tolist(), dataframe['indexes'].tolist()

if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load dataset and extract methods / impacted line indexes
    df = pd.read_csv(args.input_filepath, encoding='utf-8-sig')
    
    # Split indexes and get masking type
    df['indexes'] = df['added_line'].apply(lambda x: x.split('-'))
    df = df.explode('indexes')
    df['type'] = df['indexes'].apply(masking_type)
    
    # Split dataset in two based on the masking type
    df_line = df[df['type'] == 'line']
    df_block = df[df['type'] == 'block']
    
    # Get data for line masking
    line_header, line_columns = extra_columns(df_line)
    line_methods, line_str_indexes = masking_columns(df_line)
    
    # Convert line indexes to int
    line_indexes = list()
    for idx in line_str_indexes:
        line_indexes.append([int(idx)])
    
    # Setup masking parameters for line masking
    params = {
              'extra_columns_header': line_header,
              'extra_columns_list': line_columns,
              'methods': line_methods,
              'idx_to_mask': line_indexes,
              'sep': '<NL>',
              'mask_symbol': '<extra_id_0>',
              'generate_random_if_none': False
              }
    
    # Line masking parameters
    params['min_tokens'] = 3
    params['max_tokens'] = 50
    params['max_lines'] = 1
    masker = ConsecutiveLineMasker(**params)
    masker.mask()
    masker.export(args.output_filepath)
    
    # Get data for block masking
    block_methods, block_str_indexes = masking_columns(df_block)
    
    # Unpack block indexes
    types_list = list()
    block_indexes = list()
    for cnt, idx in enumerate(block_str_indexes):
        # Extract start and end index
        idx_str = idx.replace('[', '').replace(']', '')
        start_idx, end_idx = idx_str.split('$')
        start_idx, end_idx = int(start_idx), int(end_idx)
        
        # Get indexes from start to end
        idx_block = list(range(start_idx, end_idx + 1))
        block_indexes.append(idx_block)
        m_type = 'block' if len(idx_block) > 1 else 'line'
        types_list.append(m_type)
    
    df_block.loc[:, 'type'] = types_list
    block_header, block_columns = extra_columns(df_block)
    
    # Setup masking parameters for block masking
    params = {
              'extra_columns_header': block_header,
              'extra_columns_list': block_columns,
              'methods': block_methods,
              'idx_to_mask': block_indexes,
              'sep': '<NL>',
              'mask_symbol': '<extra_id_0>',
              'generate_random_if_none': False
              }
    
    # Block masking parameters
    params['min_tokens'] = 3
    params['max_tokens'] = 50
    params['max_lines'] = 3
    masker = ConsecutiveLineMasker(**params)
    masker.mask()
    masker.export(args.output_filepath)

    # Retrieve masked dataset and sort by date, commit_id and filepath
    retrieved_df = pd.read_csv(args.output_filepath, encoding='utf-8-sig')
    retrieved_df['date'] = pd.to_datetime(retrieved_df['date'])
    retrieved_df = retrieved_df.sort_values(by=['date', 'commit_id', 'filepath'])
    retrieved_df.to_csv(args.output_filepath, encoding='utf-8-sig', index=False)
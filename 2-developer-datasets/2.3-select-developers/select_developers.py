import os, glob, argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_dir', '-o',
                        metavar='FILEPATH',
                        dest='output_dir',
                        required=False,
                        type=str,
                        default='results',
                        help='Path of the directory that will contain the developers rankings \
                            and the ordered list of developers by number of methods.')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--datasets_dir', '-i',
                          metavar='DIR',
                          dest='datasets_dir',
                          required=True,
                          type=str,
                          help='Path of the directory containing the developers datasets')
    required.add_argument('--selected_dir', '-d',
                          metavar='DIR',
                          dest='selected_dir',
                          required=True,
                          type=str,
                          help='Path of the directory where to save the datasets of the selected developers.')

    return parser

def remove_internal_duplicates(dataframe, columns_to_check: list):
    # Remove whitespaces from the columns to check
    for col in columns_to_check:
        dataframe[f'{col}_no_whitespaces'] = dataframe[col].astype(str).replace('\s+', '', regex=True)
    
    # Drop duplicates
    non_whitespace_columns = [f'{col}_no_whitespaces' for col in columns_to_check]
    dataframe = dataframe.drop_duplicates(subset=non_whitespace_columns, keep='first')
    dataframe = dataframe.drop(columns=non_whitespace_columns)

    return dataframe

def export_ranking(dfs: list, filename: str, output_dir: str):
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        for _, (dev_id, size) in enumerate(dfs):
            f.write(f'Developer ID: {dev_id}, Size: {size}\n')

def invalid_datasets(filenames: list, path: str):
    for filename in filenames:
        # Check if the dataframe exists
        df_path = os.path.join(path, filename)
        if not os.path.exists(df_path): return True
        
        # Check if the dataframe has less than 50 instances (min eval size for block dataset)
        df = pd.read_csv(df_path)
        if len(df) < 50: return True
        
    return False

if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Create a blacklist of developers which dataset is not valid
    blacklist = [18360]

    # Retrieve the size of each developer dataset
    df_sizes = list()
    for path in glob.glob(f'{args.datasets_dir}/**/developer_*', recursive=True):
        if not os.path.isdir(path): continue
        
        # Retrieve the developer id from the name of the directory
        dev_id = os.path.basename(path).replace('developer_', '')

        # Skip developers with invalid datasets
        if int(dev_id) in blacklist: continue
        
        # Save the size of the masked datasets with no duplicates
        df = pd.read_csv(os.path.join(path, 'developer_masked_methods.csv'))
        df_no_dup = remove_internal_duplicates(df, ['masked_method', 'mask'])
        df_sizes.append((dev_id, len(df_no_dup)))

    # Sort by dataframe size
    sorting_key = lambda x: x[1]
    df_sizes.sort(key=sorting_key, reverse=True)
    
    # Save the ordered list of developers by number of masked methods
    export_ranking(df_sizes, 'masked_developers_dataset_ranking.txt', args.output_dir)
    
    # Retrieve the first 100 developers by number of masked methods that have at least 1600 instances
    selected_dfs = list()
    min_instances = 1600
    for dev, size in df_sizes:
        if size < min_instances: continue
        selected_dfs.append((dev, size))
        if len(selected_dfs) == 100: break
    
    # Save the selected developers
    with open(os.path.join(args.output_dir, 'selected_developers_ranking.txt'), 'w') as f:
        for dev, size in selected_dfs:
            f.write(f'Developer ID: {dev}, Size: {size}\n')
    
    # Save the selected developers ids
    with open(os.path.join(args.output_dir, 'selected_developers_ids.txt'), 'w') as f:
        f.write('\n'.join([dev for dev, _, in selected_dfs]))
    
import os, argparse
import pandas as pd, numpy as np

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--developers_list_filepath', '-l',
                          metavar='FILEPATH',
                          dest='developers_list_filepath',
                          required=True,
                          type=str,
                          help='Filepath of the TXT file containing the list of selected developers.')
    required.add_argument('--datasets_dir', '-d',
                          metavar='PATH',   
                          dest='datasets_dir',
                          required=True,
                          type=str,
                          help='Path of the directory containing the developers datasets.')
    required.add_argument('--output_dir', '-o',
                        metavar='PATH',
                        dest='output_dir',
                        required=True,
                        type=str,
                        help='Path of the directory that will contain the final datasets of the selected developers.')
    parser.add_argument('--results_dir', '-r',
                        metavar='FILEPATH',
                        dest='results_dir',
                        required=False,
                        type=str,
                        default='results',
                        help='Path of the results directory.')
    return parser

def remove_duplicates(subject_df, ref_df, columns_to_check: list):
    """
    This function remove from the subject DataFrame the rows that are already present in the reference DataFrame based on the columns to check.
    """
    ref_df = ref_df.copy()
    subject_df = subject_df.copy()

    # Remove whitespaces from the columns to check
    for col in columns_to_check:
        subject_df[f'{col}_no_whitespaces'] = subject_df[col].astype(str).replace('\s+', '', regex=True)
        ref_df[f'{col}_no_whitespaces'] = ref_df[col].astype(str).replace('\s+', '', regex=True)
    
    # Merge the two DataFrames on non_whitespace_columns and remove added merge columns
    non_whitespace_columns = [f'{col}_no_whitespaces' for col in columns_to_check]
    merged_df = pd.merge(subject_df, ref_df, on=non_whitespace_columns, how='outer', indicator=True, suffixes=('', '_y'))
    merged_df.drop(merged_df.filter(regex='_y$').columns, axis=1, inplace=True) 

    # Remove duplicates
    subject_df = merged_df[merged_df['_merge'] == 'left_only']
    subject_df = subject_df.drop(columns=['_merge'])

    # Drop the non_whitespace_columns
    subject_df = subject_df.drop(columns=non_whitespace_columns)
    ref_df.drop(columns=non_whitespace_columns, inplace=True)

    return subject_df

def remove_internal_duplicates(dataframe, columns_to_check: list):
    df = dataframe.copy()
    
    # Remove whitespaces from the columns to check
    for col in columns_to_check:
        df[f'{col}_no_whitespaces'] = df[col].astype(str).replace('\s+', '', regex=True)
    
    # Drop duplicates
    non_whitespace_columns = [f'{col}_no_whitespaces' for col in columns_to_check]
    df = df.drop_duplicates(subset=non_whitespace_columns, keep='first')
    df = df.drop(columns=non_whitespace_columns)

    return df

def split_dataset(filepath: str, test_size: int = 500, min_train_size: int = 1000):
    
    # Read the dataset and sort by date
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Create the test set as the last 'test_size' instances of the dataset withouth duplicates
    df_no_duplicates = remove_internal_duplicates(df, ['masked_method', 'mask'])
    test_df = df_no_duplicates.tail(test_size)
    
    # Get all the previous instances up to the first instance in test_df
    train_eval_df = df[df['date'] < test_df['date'].iloc[0]]
    train_eval_df = remove_duplicates(train_eval_df, test_df, ['masked_method', 'mask'])
    
    # Split the train_eval_df into train [90%] and eval [10%] and remove duplicates between them.
    # If the train set has less than 'min_train_size' instances, take for the eval the minimum between 10% of the dataset and 'min_train_size'
    train_instances_num = int(.9*len(train_eval_df))
    train_instances_num = min_train_size if train_instances_num < min_train_size else train_instances_num
    eval_instances_num = len(train_eval_df) - train_instances_num
    
    train_eval_df_no_dup = remove_internal_duplicates(train_eval_df, ['masked_method', 'mask'])
    train_eval_df_no_dup = train_eval_df_no_dup.sort_values('date')
    eval_df = train_eval_df_no_dup.tail(eval_instances_num)
    
    # Get the remaining instances up to the first instance in eval_df
    train_df = df[df['date'] < eval_df['date'].iloc[0]]
    eval_df = remove_duplicates(eval_df, train_df, ['masked_method', 'mask'])
    
    return train_df, eval_df, test_df

if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    
    # Retrieve the list of developers ids
    with open(args.developers_list_filepath, 'r') as f:
        devs = f.read().splitlines()
        devs = [int(dev) for dev in devs]
    
    # Retrieve the list of developers datasets
    df_sizes = list()
    for dev in devs:
        print(f'Processing developer {dev}')
        developer_dir = os.path.join(args.datasets_dir, f'developer_{dev}')
        
        # Retrieve 
        dataset_filepath = os.path.join(developer_dir, 'developer_masked_methods.csv')
        df = pd.read_csv(dataset_filepath)
        train_df, eval_df, test_df = split_dataset(dataset_filepath, test_size=500, min_train_size=1000)
        
        print(f'Masked dataset for developer {dev} has {len(train_df)} train instances, {len(eval_df)} eval instances and {len(test_df)} test instances')
        assert len(train_df) > 0 and len(eval_df) > 0 and len(test_df) > 0, f'Empty dataset for developer {dev}'
        assert len(train_df) > 1000, f'Developer {dev} has less than 1000 instances in the train dataset'
        assert len(test_df) == 500, f'Developer {dev} test dataset has more / less than 500 instances'
        
        # Add the size of the dataset to the list
        df_sizes.append((dev, len(train_df)))

        # Export the datasets in the output directory
        output_dir = os.path.join(args.output_dir, f'developer_{dev}')
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        df.to_csv(os.path.join(output_dir, 'developer_masked_methods.csv'), index=False)
        train_df.to_csv(os.path.join(output_dir, 'developer_masked_methods_train.csv'), index=False)
        eval_df.to_csv(os.path.join(output_dir, 'developer_masked_methods_eval.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'developer_masked_methods_test.csv'), index=False)
        
    # Order by size and save the selected developers
    sorting_key = lambda x: x[1]
    df_sizes.sort(key=sorting_key, reverse=True)
    ordered_devs = [str(dev) for dev, _, in df_sizes]
    ordered_sizes = [str(size) for _, size, in df_sizes]
    with open(os.path.join(args.results_dir, 'devs_ranking.txt'), 'w') as f:
        f.write('\n'.join(ordered_devs))
    with open(os.path.join(args.results_dir, 'sizes_ranking.txt'), 'w') as f:
        f.write('\n'.join(ordered_sizes))
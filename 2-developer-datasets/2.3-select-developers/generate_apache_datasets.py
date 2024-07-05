import os, argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--datasets_filename', '-s',
                        metavar='FILEPATH',
                        dest='datasets_filename',
                        required=False,
                        type=str,
                        default='output',
                        help='Name of the CSV file containing the developers dataset to merge. Format: <filename>.csv')
    parser.add_argument('--output_dir', '-o',
                        metavar='FILEPATH',
                        dest='output_dir',
                        required=False,
                        type=str,
                        default='output',
                        help='Directory path of the CSV files containing random methods from apache developers up to the selected date')
    parser.add_argument('--datasets_to_compare', '-c',
                        metavar='DATASETS',
                        dest='datasets_to_compare',
                        required=False,
                        nargs='+',
                        help='List of filepaths of the CSV files representing the validation and test datasets. Used to check and remove duplicates. Format: <filename>.csv...')
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--reference_dataset', '-i',
                        metavar='FILEPATH',
                        dest='reference_dataset',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file from which extract the last commit date')
    parser.add_argument('--datasets_dir', '-d',
                        metavar='PATH',
                        dest='datasets_dir',
                        required=True,
                        type=str,
                        help='Name of the directory where the developer datasets are stored')
    parser.add_argument('--author_id', '-a',
                        metavar='ID',
                        dest='author_id',
                        required=True,
                        type=int,
                        help='ID of the developer to exclude from the corpus of the second dataset')
    return parser

def remove_duplicates(subject_df, ref_df, columns_to_check: list):
    """
    This function remove from the subject DataFrame the rows that are already present in the reference DataFrame based on the columns to check.
    """

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
    # Remove whitespaces from the columns to check
    for col in columns_to_check:
        dataframe[f'{col}_no_whitespaces'] = dataframe[col].astype(str).replace('\s+', '', regex=True)
    
    # Drop duplicates
    non_whitespace_columns = [f'{col}_no_whitespaces' for col in columns_to_check]
    dataframe = dataframe.drop_duplicates(subset=non_whitespace_columns, keep='first')
    dataframe = dataframe.drop(columns=non_whitespace_columns)

    return dataframe


if __name__ == '__main__':

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    
    # Load the reference dataset and extract the number of instances and the last commit date
    reference_df = pd.read_csv(args.reference_dataset, encoding='utf-8-sig')
    instances = len(reference_df)
    reference_df['date'] = pd.to_datetime(reference_df['date'])
    reference_date = reference_df['date'].max()

    # Retrieve the path of all the developer datasets
    developer_datasets = []
    for root, _, files in os.walk(args.datasets_dir):
        for file in files:
            if file == args.datasets_filename:
                developer_datasets.append(os.path.join(root, file))

    # Merge all the developer datasets
    df_total = pd.DataFrame()
    for dataset in developer_datasets:
        dev_df = pd.read_csv(dataset, encoding='utf-8-sig')
        df_total = pd.concat([df_total, dev_df], ignore_index=True)
    
    # Remove all the instances which date is greater than the reference date
    df_total['date'] = pd.to_datetime(df_total['date'])
    df_eval = df_total[df_total['date'] > reference_date]
    df_total = df_total[df_total['date'] <= reference_date]
    
    # Remove duplicates from the dataset and retrieve the minimum date for the evaluation dataset
    # Since two ordered datasets to compare are given in input, it will take the minimum commit date of the test set
    reference_end_date = None
    if args.datasets_to_compare:
        for filepath_df in args.datasets_to_compare:
            df_to_compare = pd.read_csv(filepath_df, encoding='utf-8-sig')
            df_to_compare['date'] = pd.to_datetime(df_to_compare['date'])
            reference_end_date = df_to_compare['date'].min()

            df_total = remove_duplicates(df_total, df_to_compare, ['mask', 'masked_method'])
    
    # Remove all the instances which date is greater than the reference end date
    if reference_end_date:
        df_eval = df_eval[df_eval['date'] < reference_end_date]
    
    # Remove duplicates from the eval dataset
    df_eval = remove_duplicates(df_eval, df_total, ['mask', 'masked_method'])
    df_eval = remove_internal_duplicates(df_eval, ['mask', 'masked_method'])

    # Get the number of instances of the eval dataset for the total size of the dataset
    total_instances = int(len(df_total) * 10 / 8)
    total_eval_instances = min(int(total_instances * 0.1), len(df_eval))
    total_df_eval = df_eval.sample(n=total_eval_instances)

    # Extract random instances from the apache datasets of the same size of the reference dataset
    df_small = df_total.sample(n=instances)

    # Get eval instances [10% of the total instances or less, if the dataset is too small]
    tot_instances = int(instances * 10 / 8)
    eval_instances = int(tot_instances * 0.1)

    eval_instances = min(eval_instances, len(df_eval))
    df_eval_small = df_eval.sample(n=eval_instances)
    
    # Export the results
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    
    # Train and eval dataset - full size until the reference date
    output_filepath = os.path.join(args.output_dir, f'apache_dataset_total_train.csv')
    df_total.sort_values(by=['date']).to_csv(output_filepath, index=False)

    output_filepath = os.path.join(args.output_dir, f'apache_dataset_total_eval.csv')
    total_df_eval.sort_values(by=['date']).to_csv(output_filepath, index=False)

    # Train and eval dataset - same size of the reference dataset
    output_filepath = os.path.join(args.output_dir, f'apache_dataset_small_train.csv')
    df_small.sort_values(by=['date']).to_csv(output_filepath, index=False)

    output_filepath = os.path.join(args.output_dir, f'apache_dataset_small_eval.csv')
    df_eval_small.sort_values(by=['date']).to_csv(output_filepath, index=False)

    
    
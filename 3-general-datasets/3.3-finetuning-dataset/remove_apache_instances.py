import os, argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--datasets_filename', '-n',
                        metavar='FILENAME',
                        dest='datasets_filename',
                        required=False,
                        type=str,
                        default='extracted.csv',
                        help='Name of the CSV file containing the developers dataset to merge. Format: <filename>.csv')
    parser.add_argument('--output_path', '-o',
                        metavar='FILEPATH',
                        dest='output_dir',
                        required=False,
                        type=str,
                        default='output.csv',
                        help='Directory path of the CSV files containing cleaned methods from apache developers')
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--finetuning_path', '-f',
                        metavar='PATH',
                        dest='finetuning_path',
                        required=True,
                        type=str,
                        help='Filepath containing the finetuning dataset')
    required.add_argument('--datasets_dir', '-d',
                        metavar='PATH',
                        dest='datasets_dir',
                        required=True,
                        type=str,
                        help='Name of the directory where the developer datasets are stored')
    return parser

def remove_duplicates(subject_df, ref_df, columns_to_check: list):
    """
    This function remove from the subject DataFrame the rows that are already present in the reference DataFrame based on the columns to check.
    """
    drop_columns = [col for col in ref_df.columns if col not in subject_df.columns]

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

    # Drop from the subject dataframe the column that are present in the reference dataframe but not in the subject dataframe
    subject_df = subject_df.drop(columns=drop_columns)
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

    # Load the finetuning dataset
    finetuning_df = pd.read_csv(args.finetuning_path, encoding='utf-8-sig')
    print(f'Finetuning dataset loaded: {len(finetuning_df)} instances')

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
    
    df_total['formatted'] = df_total['method']
    df_total = remove_internal_duplicates(df_total, ['formatted'])

    # Remove duplicates from the finetuning dataset
    finetuning_df = remove_duplicates(finetuning_df, df_total, ['formatted'])
    finetuning_df = remove_internal_duplicates(finetuning_df, ['formatted'])

    print(f'Finetuning dataset after removing duplicates: {len(finetuning_df)} instances')

    # Export the finetuning dataset
    finetuning_df.to_csv(args.output_dir, index=False)
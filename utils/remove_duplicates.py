import argparse, pandas as pd


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--subject_dataset', '-f',
                        metavar='PATH',
                        dest='first_filepath',
                        required=True,
                        type=str,
                        help='Path of the CSV dataframe from which remove duplicates.')
    required.add_argument('--ref_dataset', '-s',
                        metavar='PATH',
                        dest='second_filepath',
                        required=True,
                        type=str,
                        help='Path of the CSV dataframe from which check for overlappings.')
    required.add_argument('--columns_to_check', '-c',
                        metavar='COLUMNS',
                        dest='columns_to_check',
                        nargs='+',
                        required=True,
                        help='The list of columns required for duplicates checking')
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

if __name__ == '__main__':   
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    
    subject_df = pd.read_csv(args.first_filepath, encoding='utf-8-sig')
    ref_df = pd.read_csv(args.second_filepath, encoding='utf-8-sig')
    
    subject_df = remove_duplicates(subject_df, ref_df, args.columns_to_check)
    subject_df.to_csv(args.first_filepath, encoding='utf-8-sig', index=False)
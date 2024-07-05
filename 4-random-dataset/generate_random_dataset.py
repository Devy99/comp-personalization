import os, argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--results_dir', '-i',
                        metavar='FILEPATH',
                        dest='results_dir',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the mined commits info')
    required.add_argument('--finetuning_path', '-f',
                        metavar='FILEPATH',
                        dest='finetuning_path',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the general finetuning dataset')
    parser.add_argument('--extracted_developer_datasets', '-d',
                        metavar='PATH',
                        dest='extracted_developer_datasets',
                        required=True,
                        type=str,
                        help='Name of the directory where the developer datasets are stored')
    parser.add_argument('--selected_developer_datasets', '-s',
                        metavar='PATH',
                        dest='selected_developer_datasets',
                        required=True,
                        type=str,
                        help='Name of the directory where the selected developers methods are stored')
    parser.add_argument('--developers_ranking_path', '-r',
                        metavar='PATH',
                        dest='developers_ranking_path',
                        required=True,
                        type=str,
                        help='Name of the file containing the developers ranking')
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

    """
    # Find under the results_dir all the CSV files which name ends for _masked.csv and concatenate them
    total_masked = pd.DataFrame()
    for root, _, files in os.walk(args.results_dir):
        for file in files:
            if file.endswith('_masked.csv'):
                try:
                    mask_path = os.path.join(root, file)
                    df = pd.read_csv(mask_path, encoding='utf-8-sig')
                    if df.empty: continue
                    df['method'] = df.apply(lambda row: row['masked_method'].replace('<extra_id_0>', row['mask']), axis=1)
                    total_masked = pd.concat([total_masked, df])
                except Exception as e:
                    print(f'Error while reading {mask_path}: {e}')

    # Export the total masked dataset
    total_masked.to_csv('total_masked.csv', index=False)
    """
    # Load the total masked dataset
    total_masked = pd.read_csv('total_masked.csv', encoding='utf-8-sig')
    total_masked['date'] = pd.to_datetime(total_masked['date'])
    print(f'Lenght of the total masked dataset: {len(total_masked)} instances')

    # Load general finetuning dataset
    finetuning_df = pd.read_csv(args.finetuning_path, encoding='utf-8-sig')
    finetuning_df['method'] = finetuning_df['formatted']

    # Remove duplicates from the general finetuning dataset
    total_masked = remove_duplicates(total_masked, finetuning_df, ['method'])
    print('Removed duplicates with the general finetuning dataset')

    # Retrieve all the apache developer datasets and remove duplicates
    df_total = pd.DataFrame()
    for root, _, files in os.walk(args.extracted_developer_datasets):
        for file in files:
            if file == 'extracted.csv':
                dataset_path = os.path.join(root, file)
                dev_df = pd.read_csv(dataset_path, encoding='utf-8-sig')
                df_total = pd.concat([df_total, dev_df], ignore_index=True)
    
    total_masked = remove_duplicates(total_masked, df_total, ['method'])
    print('Removed duplicates with apache datasets from the total masked dataset')
    print(f'Lenght of the total masked dataset after removing duplicates with apache datasets: {len(total_masked)} instances')
    total_masked.to_csv('total_masked_no_dup.csv', index=False)

    # Retrieve top devs from the ranking
    with open(args.developers_ranking_path, 'r') as f:
        devs = f.readlines()
        devs = [int(dev.strip()) for dev in devs]

    added_devs, n_added_devs = list(), 0
    for dev in devs:
        if n_added_devs == 10: break

        selected_dev_dir = os.path.join(args.selected_developer_datasets, f'developer_{dev}')

        # Load developer test dataset
        dev_test_path = os.path.join(selected_dev_dir, 'developer_masked_methods_test.csv')
        dev_test_df = pd.read_csv(dev_test_path, encoding='utf-8-sig')
        dev_test_df['date'] = pd.to_datetime(dev_test_df['date'])
        dev_test_df = dev_test_df.sort_values(by='date')
        min_date = dev_test_df['date'].min()

        # Load developer organization total dataset
        dev_apache_total_path = os.path.join(selected_dev_dir, 'apache_dataset_total_train.csv')
        apache_total_df = pd.read_csv(dev_apache_total_path, encoding='utf-8-sig')
        total_instances = len(apache_total_df)

        # Get the random changes for the developer according to the min date
        dev_random_eval_df = total_masked[total_masked['date'] >= min_date]
        dev_random_eval_df = remove_internal_duplicates(dev_random_eval_df, ['mask', 'masked_method'])
        dev_random_train_df = total_masked[total_masked['date'] < min_date]
        dev_random_eval_df = remove_duplicates(dev_random_eval_df, dev_random_train_df, ['method'])

        n_sample = min(total_instances, len(dev_random_train_df))
        if n_sample < total_instances: 
            random_train = dev_random_eval_df.sample(n=total_instances - n_sample, random_state=230923)
            dev_random_train_df = pd.concat([dev_random_train_df, random_train], ignore_index=True)
            dev_random_eval_df = remove_duplicates(dev_random_eval_df, dev_random_train_df, ['method'])
        else:
            dev_random_train_df = dev_random_train_df.sample(n=n_sample, random_state=230923)

        eval_instances = min(int(n_sample * 0.1), len(dev_random_eval_df))
        dev_random_eval_df = dev_random_eval_df.sample(n=eval_instances, random_state=230923)

        print(f'Developer {dev} - Train: {len(dev_random_train_df)} - Eval: {len(dev_random_eval_df)}')

        # Export the train and eval datasets
        train_random_path = os.path.join(selected_dev_dir, 'random_changes_train.csv')
        dev_random_train_df['date'] = pd.to_datetime(dev_random_train_df['date'])
        dev_random_train_df = dev_random_train_df.sort_values(by='date')
        dev_random_train_df.to_csv(train_random_path, index=False)

        eval_random_path = os.path.join(selected_dev_dir, 'random_changes_eval.csv')
        dev_random_eval_df['date'] = pd.to_datetime(dev_random_eval_df['date'])
        dev_random_eval_df = dev_random_eval_df.sort_values(by='date')
        dev_random_eval_df.to_csv(eval_random_path, index=False)

        n_added_devs += 1
        added_devs.append(dev)
    
    print(f'Added {n_added_devs} developers: {added_devs}')
import os, argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--datasets_dir', '-i',
                          metavar='DIR',
                          dest='datasets_dir',
                          required=True,
                          type=str,
                          help='Path of the directory containing the developers datasets')

    return parser

def invalid_datasets(filenames: list, path: str):
    for filename in filenames:
        # Check if the dataframe exists
        df_path = os.path.join(path, filename)
        if not os.path.exists(df_path): return True
        
        # Check if the dataframe is not empty
        df = pd.read_csv(df_path)
        if len(df) == 0: return True
        
    return False

if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    
    apache_df_files = ['apache_dataset_total_train.csv', 'apache_dataset_total_eval.csv', 'apache_dataset_small_train.csv', 'apache_dataset_small_eval.csv']

    # Retrieve the list of developers ids
    for developer_dir in os.listdir(args.datasets_dir):
        print(f'Checking {developer_dir}')
        
        # Check masked datasets dates
        train_df_path = os.path.join(args.datasets_dir, developer_dir, 'developer_masked_methods_train.csv') 
        eval_df_path = os.path.join(args.datasets_dir, developer_dir, 'developer_masked_methods_eval.csv')
        test_df_path = os.path.join(args.datasets_dir, developer_dir, 'developer_masked_methods_test.csv')
        train_df, eval_df, test_df = pd.read_csv(train_df_path), pd.read_csv(eval_df_path), pd.read_csv(test_df_path)

        train_df['date'] = pd.to_datetime(train_df['date'])
        eval_df['date'] = pd.to_datetime(eval_df['date'])
        test_df['date'] = pd.to_datetime(test_df['date'])

        assert train_df['date'].max() < eval_df['date'].min(), f'Invalid dates for developer {developer_dir}: train max date {train_df["date"].max()} is greater than eval min date {eval_df["date"].min()}'
        assert eval_df['date'].max() < test_df['date'].min(), f'Invalid dates for developer {developer_dir}: eval max date {eval_df["date"].max()} is greater than test min date {test_df["date"].min()}'
        
        # Check if apache datasets exist / are not empty
        dev_dir_path = os.path.join(args.datasets_dir, developer_dir)
        assert not invalid_datasets(apache_df_files, dev_dir_path), f'Invalid apache datasets for developer {developer_dir}'   

        # Check apache datasets dates and sizes
        apache_train_df_path = os.path.join(args.datasets_dir, developer_dir, 'apache_dataset_total_train.csv')
        apache_eval_df_path = os.path.join(args.datasets_dir, developer_dir, 'apache_dataset_total_eval.csv')
        apache_train_df, apache_eval_df = pd.read_csv(apache_train_df_path), pd.read_csv(apache_eval_df_path)

        apache_train_df['date'] = pd.to_datetime(apache_train_df['date'])
        apache_eval_df['date'] = pd.to_datetime(apache_eval_df['date'])

        assert apache_train_df['date'].max() < apache_eval_df['date'].min(), f'Invalid dates for developer {developer_dir}: apache total train max date {apache_train_df["date"].max()} is greater than apache total eval min date {apache_eval_df["date"].min()}'
        assert apache_train_df['date'].max() < eval_df['date'].min(), f'Invalid dates for developer {developer_dir}: apache total train max date {apache_train_df["date"].max()} is greater than eval min date {eval_df["date"].min()}'
        
        # Check apache small datasets dates and sizes
        small_train_df_path = os.path.join(args.datasets_dir, developer_dir, 'apache_dataset_small_train.csv')
        small_eval_df_path = os.path.join(args.datasets_dir, developer_dir, 'apache_dataset_small_eval.csv')
        small_train_df, small_eval_df = pd.read_csv(small_train_df_path), pd.read_csv(small_eval_df_path)
        
        small_train_df['date'] = pd.to_datetime(small_train_df['date'])
        small_eval_df['date'] = pd.to_datetime(small_eval_df['date'])
        
        assert small_train_df['date'].max() < small_eval_df['date'].min(), f'Invalid dates for developer {developer_dir}: small train max date {small_train_df["date"].max()} is greater than small eval min date {small_eval_df["date"].min()}'
        assert small_train_df['date'].max() < eval_df['date'].min(), f'Invalid dates for developer {developer_dir}: small train max date {small_train_df["date"].max()} is greater than eval min date {eval_df["date"].min()}'
        assert len(small_train_df) == len(train_df), f'Invalid size for developer {developer_dir}: small train size {len(small_train_df)} is different from developer train size {len(train_df)}'
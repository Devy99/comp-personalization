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
    parser.add_argument('--results_dir', '-r',
                        metavar='FILEPATH',
                        dest='results_dir',
                        required=False,
                        type=str,
                        default='results',
                        help='Path of the results directory.')
    return parser


if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    
    # Retrieve the list of developers ids
    with open(args.developers_list_filepath, 'r') as f:
        devs = f.read().splitlines()
        devs = [int(dev) for dev in devs]
    
    dev_avgs, dfs = list(), list()
    for dev in devs:
        print(f'Processing developer {dev}')
        developer_dir = os.path.join(args.datasets_dir, f'developer_{dev}')
        
        # Retrieve the train dataset
        dataset_filepath = os.path.join(developer_dir, 'developer_masked_methods_train.csv')
        train_df = pd.read_csv(dataset_filepath)

        # Compute the number of tokens in the mask.
        train_df['mask_len'] = train_df['mask'].str.split().str.len()
        avg_n_tokens = train_df['mask_len'].mean()
        dev_avgs.append((dev, avg_n_tokens))

        # Export the dataset with the number of tokens in the mask
        train_df = train_df[['author_id', 'masked_method', 'mask', 'mask_len', 'type']]
        train_df.to_csv(os.path.join(developer_dir, 'developer_methods_train_distribution.csv'), index=False)
        dfs.append(train_df)

    # Concatenate all dataframes and export the results
    total_df = pd.concat(dfs)
    total_df.to_csv(os.path.join(args.results_dir, 'devs_distribution.csv'), index=False)

    # Export the results
    with open(os.path.join(args.results_dir, 'devs_distribution.txt'), 'w') as f:
        for dev, avg in dev_avgs:
            f.write(f'Developer {dev}: {avg} average masked tokens\n')
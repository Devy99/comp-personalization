import json, random, argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output', '-o',
                        metavar='FILEPATH',
                        dest='output',
                        required=False,
                        default='clean_commits.csv',
                        type=str,
                        help='Filepath of the CSV file containing the resulting pre-processed commit file')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', '-i',
                        metavar='FILEPATH',
                        dest='filepath',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the repositories from GHSearch')
    return parser

if __name__ == '__main__':  
    random.seed(230923)

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load repository to retain
    print(f'Loading file: {args.filepath}')
    df = pd.read_csv(args.filepath, encoding='utf-8-sig')
    initial_len = len(df)
    
    ## Pre-processing
    print('Pre-processing data...')

    # Remove rows containing null values
    df = df.dropna()

    # Remove bots
    df = df[df['author'].str.contains("\[bot\]") == False]
    df = df[df['author'].str.contains("GitHub") == False]
    
    # Remove outliers
    Q1, Q3 = df['changed_files'].quantile(0.25), df['changed_files'].quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5*IQR
    df = df.query('changed_files < @threshold')

    # Group by repository and retain onlt 1500 random commits
    df = df.groupby('repository').apply(lambda x: x.sample(n=min(len(x), 1500), random_state=230923)).reset_index(drop=True)

    final_len = len(df)
    print(f'Initial length: {initial_len}, Final length: {final_len}')

    # Get remaining repositories and export to txt
    repos = df['repository'].unique()
    with open('remaining_repos.txt', 'w') as f:
        for repo in repos: f.write(f'{repo}\n')

    # Save the pre-processed dataframe
    df.to_csv(args.output, index=False, encoding='utf-8-sig')


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
                        type=str,
                        default='formatted_methods.csv',
                        help='Filepath of the CSV file containing the random repositories')
    parser.add_argument('--num_repos', '-n',
                        metavar='NUM',
                        dest='num_repos',
                        required=False,
                        type=int,
                        default=1000,
                        help='Number of random repositories to extract from the input file')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', '-i',
                        metavar='FILEPATH',
                        dest='filepath',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the repositories from GHSearch')
    required.add_argument('--general_training_path', '-g',
                        metavar='FILEPATH',
                        dest='general_training_path',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the general training data')
    return parser

if __name__ == '__main__':  
    random.seed(230923)

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load repository to retain
    df = pd.read_csv(args.filepath, encoding = 'ISO-8859-1')
    df = df.drop_duplicates(subset=['name'])

    repos = df['name'].tolist()

    # Remove all apache repositories
    repos = [repo for repo in repos if not repo.startswith('apache')]

    # Load the general training dataset
    general_training_df = pd.read_csv(args.general_training_path, encoding='utf-8-sig')
    to_remove_repos = general_training_df['repository'].tolist()

    # Remove the repositories that are already present in the general training dataset
    repos = [repo for repo in repos if repo not in to_remove_repos]

    # Extract n random repositories
    random.shuffle(repos)
    repos = repos[:args.num_repos]

    # Export the list of repositories and urls as json file
    items = list()
    for repo in repos:
        url = f'https://github.com/{repo}'
        items.append({'name': repo, 'url': url})
    
    with open(args.output, 'w') as f:
        json.dump(items, f, indent=4)
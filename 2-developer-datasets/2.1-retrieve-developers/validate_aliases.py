
import requests, argparse, pandas as pd

def export(filename: str, items: list):
  with open(filename, 'w', encoding = 'utf-8', errors='replace') as file:
    file.write('\n'.join(items))

def author(organization: str, repo: str, commit_id: str, auth_token: str) -> list:
    endpoint = f'https://api.github.com/repos/{organization}/{repo}/commits/{commit_id}'

    headers = {'Authorization': f'Bearer {auth_token}'} if auth_token else None
    response = requests.get(endpoint, headers=headers)

    if response.status_code != 200:
        print(f'Error! Status code {response.status_code}. Message: {response.reason}.')
        exit()

    data = response.json()
    author = data['author']['id'] if data['author'] else None

    return author

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_filepath', '-o',
                        metavar='FILEPATH',
                        dest='output_filepath',
                        required=False,
                        type=str,
                        default='masked.csv',
                        help='Filepath of the CSV file containing the masked methods of the developer')
    parser.add_argument('--token', '-t',
                        metavar='TOKEN',
                        dest='token',
                        required=False,
                        type=str,
                        help='Github authorization token. It could be necessary in case multiple API calls are required')
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--input_filepath', '-i',
                        metavar='FILEPATH',
                        dest='input_filepath',
                        required=False,
                        type=str,
                        help='Filepath of the CSV file containing developers commits')

    return parser


if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load developers commits dataset
    df = pd.read_csv(args.input_filepath, encoding='utf-8')
    
    # Get the first 1000 developers grouped by author id with the highest number of total added lines
    groups = df.groupby(['author_id'])['added_line'].sum().reset_index(name='total_added_lines')
    groups = groups.sort_values(by=['total_added_lines'], ascending=False)
    groups = groups.head(1000)
    ids = groups['author_id'].tolist()
    
    # Collect aliases for each author and check if groups are valid
    organization = 'apache'
    valid_authors = list()
    for author_id in ids:

        # Retrieve subset of the dataset and drop duplicates by name and email
        temp_df = df[df['author_id'] == author_id]        
        print(f'Processing author {author_id} with {temp_df["added_line"].sum()} total added lines...')
        
        temp_df = temp_df.drop_duplicates(subset=['author', 'email'], keep='first')

        skip = False
        github_ids = set()
        for _, row in temp_df.iterrows():
            repo, commit_id, author_id  = row['repository'], row['commit_id'], row['author_id']

            # Get author id
            github_id = author(organization, repo, commit_id, args.token)
            if not github_id: continue

            # Check if a different author is in the group
            if github_ids and github_id not in github_ids:
                skip = True
                print(f'Skipping author: {author_id}! Different authors found on the same group.\n')
                break

            github_ids.add(github_id)
            
        # Skip if a different author is in the group
        if skip: continue
        
        # Skip if the author never shows up in GitHub
        if not github_ids:
            print(f'Skipping author: {author_id}! No GitHub id found.\n')
            continue
        
        valid_authors.append(str(author_id))
        
    # Export valid authors
    export(args.output_filepath, valid_authors)

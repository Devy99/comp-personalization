from __future__ import annotations
from datetime import datetime

import time, json, requests, warnings, argparse

class RepoInfo(object):
    def __init__(self, name, url):
        self.name = name
        self.url = url

    def __eq__(self, other):
        return self.name == other.name and self.url == other.url
    
    def to_json(self):
        return self.__dict__

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='optional arguments')
    parser.add_argument('--output_filepath', '-O',
                        metavar='FILEPATH',
                        dest='output',
                        required=False,
                        type=str,
                        default='output.json',
                        help='Output filepath where to store the list of repositories')
    parser.add_argument('--token', '-t',
                        metavar='TOKEN',
                        dest='token',
                        required=False,
                        type=str,
                        help='Github authorization token. It could be necessary in case multiple API calls are required')
    parser.add_argument('--from_date', '-d',
                        metavar='DATE',
                        dest='from_date',
                        required=False,
                        type=str,
                        help='Required to retrieve all repositories created from a certain date. Date format dd-mm-YY (e.g., 01-01-2018)')


    required = parser.add_argument_group('required arguments')
    required.add_argument('--org', '-o',
                        metavar='ORGANIZATION',
                        dest='organization',
                        required=True,
                        type=str,
                        help='GitHub organization from which retrieve the collections')
    required.add_argument('--languages', '-l',
                        metavar='LANGUAGES',
                        dest='languages',
                        nargs='+',
                        required=False,
                        help='The list of allowed repository languages')
    
    return parser

def retrieve_repos(organization: str, languages: list, from_date: str, auth_token: str) -> list:
    languages = [lang.lower() for lang in languages] if languages else []
    endpoint = f'https://api.github.com/orgs/{organization}/repos'

    repos = []
    page, page_size = 0, 100
    while True:
        params = {"page": page, "per_page": page_size, 'languages': languages}
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        response = requests.get(endpoint, params=params, headers=headers)
        
        if response.status_code != 200:
            print(f'Error! Status code {response.status_code}. Message: {response.reason}.')
            exit()

        result = response.json()
        if len(result) == 0 : break
        
        for repo in result:
            repo_lang = str(repo['language']).lower()
            if repo_lang not in languages: continue

            # Retrieve only repositories created after the specified date
            if from_date != None:
                date_creation, target_date = repo['created_at'], from_date
                creation_datetime = datetime.strptime(date_creation, '%Y-%m-%dT%H:%M:%SZ')
                target_datetime = datetime.strptime(target_date, '%d-%m-%Y')
                if creation_datetime < target_datetime: continue
            
            repo_info = RepoInfo(repo['name'], repo['html_url'])
            if repo_info not in repos: 
                repos.append(repo_info)
            
        page += 1
        time.sleep(1)
        
    return repos

if __name__ == '__main__':   
    warnings.filterwarnings("ignore")

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    print(f'Retrieving all the specified repositories...')

    # Retrieve repo data from GitHub
    repos = retrieve_repos(args.organization, args.languages, args.from_date, args.token)
    print(f'Repositories found: {len(repos)}')
    repos_dict = [repo.to_json() for repo in repos]
    json_data = json.dumps(repos_dict, indent=2)
    
    # Produce output as JSON file
    with open(args.output, 'w') as file:
        file.write(json_data)


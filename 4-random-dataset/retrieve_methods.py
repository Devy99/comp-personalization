from git import Repo 
from collections import defaultdict
from javalang.tokenizer import tokenize
from collections import defaultdict

import os, sys, csv, json, stat, shutil, javalang, argparse, subprocess
import re, uuid, pandas as pd

class ChangedFile(object):
    def __init__(self, file):
        self.old_path = file['old_path']
        self.new_path = file['new_path']
        self.filename = file['filename']
        self.change_type = file['change_type']
        self.added_lines = file['added_lines']
        self.deleted_lines = file['deleted_lines']

        changed_methods = []
        for method in file['changed_methods']:
            changed_methods.append(Method(method))
        self.changed_methods = changed_methods

        methods_after = []
        for method in file['methods_after']:
            methods_after.append(Method(method))
        self.methods_after = methods_after

        self.nloc = file['nloc']
        self.complexity = file['complexity']
        self.token_count = file['token_count']


class Method(object):
    def __init__(self, method):
        self.name = method['name']
        self.long_name = method['long_name']
        self.start_line = method['start_line']
        self.end_line = method['end_line']


### PROCESSING FUNCTIONS
def invalid_string(string):
    # Check if code contains emojis
    emoji_rgx = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    contains_emoji = emoji_rgx.search(string)
    
    # Check if contains non-latin characters
    is_non_latin = re.search('[^\x00-\x7F]', string)
    return bool(contains_emoji or is_non_latin)

def split_camel_case(word : str):
    modified_string = list(map(lambda x: '_' + x if x.isupper() else x, word))
    split_string = ''.join(modified_string).split('_')
    split_string = [string for string in split_string if string.strip()]
    return split_string

def is_method(code: str):
    tree = javalang.parse.parse('class Main {' + code + '}')
    methods = tree.filter(javalang.tree.MethodDeclaration)
    _, node = next(methods, (None, None))
    return node != None

def is_test(method):
    # Retrieve method name
    tree = javalang.parse.parse('class Main {' + method + '}')
    methods = tree.filter(javalang.tree.MethodDeclaration)
    _, node = next(methods, (None, None))
    if node == None: return False
    
    method_name = node.name

    # Check if test is in a split of the method name
    for word in split_camel_case(method_name):
      if word.lower() == 'test': 
         return True
      
    return False

def tokenize_code(method):
  return ' '.join(token.value for token in tokenize(method))

def replace_last(string, to_replace, replacement):
    return replacement.join(string.rsplit(to_replace, 1))

def body(method: str):
  method = method.replace('{', '<**>', 1)
  method = replace_last(method, '}', '<**>')
  splits = method.split('<**>')
  body = splits[1].replace('\r', '').replace('\n', '') \
                  .replace('\t', '').replace(' ', '').strip()
  return body

def is_body_empty(method: str):
  if not '{' in method or not '}' in method:
    return True
  b = body(method).replace('<NL>','')
  return b == ''

def format(method: str):
    # Preserve possible regex values and convert newlines in $NL$ (to preserve them during javalang tokenization)
    method = method.replace(r"\\n","<nl-n>").replace(r"\\r","<nl-r>")
    method = method.replace("\r\n"," $NL$ ").replace("\r"," $NL$ ").replace("\n"," $NL$ ")
    method = method.replace("<nl-n>",r"\\n").replace("<nl-r>",r"\\r")

    if method.startswith('$NL$'): 
        method = method.replace('$NL$', '', 1)

    # Tokenize code
    method = tokenize_code(method)

    # Replace $NL$ with <NL>
    method = method.replace("$NL$","<NL>")
    return method

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//.*?$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "\n" * match.group(2).count('\n') # return same number of newlines as the removed comments
        else: # otherwise, we will return the 1st group        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)

def clean_from_comments(lines: list):
    method = '\n'.join(lines).strip()
    
    # Remove inline / multi-line comments
    clean_method = remove_comments(method).strip()
    clean_lines = clean_method.split('\n')
    
    # Detect lines with only comments
    changes_list = list()
    for idx in range(len(clean_lines)):
        original_line, clean_line = lines[idx], clean_lines[idx]
        original_line = original_line.split(':', 1)[1] if original_line.strip() else original_line
        original_line = original_line[1:] if original_line.startswith('+') or  original_line.startswith('-') else original_line
        clean_line = clean_line.split(':', 1)[1] if clean_line.strip() else clean_line
        clean_line = clean_line[1:] if clean_line.startswith('+') or clean_line.startswith('-') else clean_line
        
        if original_line != clean_line and not clean_line.strip():
            changes_list.append(1)
        else:
            changes_list.append(0)
            
    # Remove lines with only comments
    clean_lines = [line for idx, line in enumerate(clean_lines) if not changes_list[idx]]
    return clean_lines

def comments_exist(lines: list):
    # Remove line number, deletion and addition
    cln_lines = [line.split(':', 1)[1] if ':' in line else line for line in lines]
    cln_lines = [line[1:] if line.startswith('+') or line.startswith('-') else line for line in cln_lines]
    
    # Get full method
    method = ' '.join(cln_lines)
    
    # Tokenize method
    tokenized_method = tokenize_code(method)
    tokenized_method_no_whitespace = ''.join(tokenized_method.split())
    method_no_whitespace = ''.join(method.split())
    
    # Javalang automatically get rid of comments. So check if the final strings are equal or not
    return tokenized_method_no_whitespace != method_no_whitespace

def find_quotations(sentence: str):
    # Normalize quotes
    normalized = sentence.strip()
    types = ["''", "‘", "’", "´", "“", "”", "--"]
    for type in types:
        normalized = normalized.replace(type, "\"")
    
    # Handle the apostrophe
    normalized = re.sub("(?<!\w)'|'(?!\w)", "\"", normalized)
    if normalized.startswith("'"): normalized.replace("'", "\"", 1)
    if normalized.endswith("'"): normalized = normalized[:-1] + "\""
        
    # Extract quotations
    return re.findall('"([^"]*)"', normalized)

def unbalanced_brackets(method: str):
    quotations = find_quotations(method)
    q_left_count = sum([q.count('{') for q in quotations])
    q_right_count = sum([q.count('}') for q in quotations])
    total_left_braces = max(0, method.count('{') - q_left_count)
    total_right_braces = max(0, method.count('}') - q_right_count)
    return total_left_braces != total_right_braces

def format_diff(diff: str):
    line_num = 0
    formatted_lines = list()
    for line in diff.split('\n'):
        # Ignore non-code lines
        if line.startswith(('---', '+++','diff', 'index', 'new', 'old')):
            continue

        # If the line starts with @@, retrieve the line number
        match = re.match(r'^@@\ -[0-9]+(,[0-9]+)?\ \+([0-9]+)(,[0-9]+)?\ @@.*', line)
        if match:
            line_num = int(match.group(2))
        else:
            formatted_lines.append(f"{line_num}:{line}")

            # Increment the number of the line if the change is not a deletion
            if not line.startswith('-'):
                line_num += 1

    return '\n'.join(formatted_lines)

### EXTRACTION FUNCTIONS
def modifications(lines, ref_lines):
    # List (of lists) containing the added single / block of lines or the lines possibly containing modifications
    added_lines, modified_lines = list(), list()

    # Store a temporary block of contiguous added / deleted lines
    added_block, deleted_block = list(), list()

    # Save the previous line for deletion checking
    line_before = None

    # Keep track of modified blocks
    is_modification = False

    # Skip blocks containing comments
    skip_block = False

    for cnt, line in enumerate(lines):
        if not line.strip(): continue
        #print(line)
        curr_line_is_deletion = line.split(':')[1].startswith('-')
        curr_line_is_addition = line.split(':')[1].startswith('+')
        prev_line_is_deletion = line_before and line_before.split(':')[1].startswith('-')

        # Skip until the next block
        if curr_line_is_addition and skip_block:
            continue
        else:
            skip_block = False

        # Save deleted lines
        if curr_line_is_deletion:
          deleted_block.append(line)

        # Keep track of modified blocks
        if curr_line_is_addition and prev_line_is_deletion:
          added_block.extend(deleted_block)
          is_modification = True
          deleted_block = list()

        # Check and skip blocks containing comments
        if curr_line_is_addition and line != ref_lines[cnt]:
            skip_block = True
            added_block = list()
            continue

        if curr_line_is_addition:
            added_block.append(line)
        else:
            # Store single lines or block of added lines
            if added_block:
              if is_modification:
                  modified_lines.append(added_block)
              else:
                  added_lines.append(added_block)

              # Reset temps
              is_modification = False
              added_block = list()

        # Update the previous line with the current one
        line_before = line

        # Check if there is a block in the last interation
        if cnt + 1 == len(lines) and added_block:
          if is_modification:
            modified_lines.append(added_block)
          else:
            added_lines.append(added_block)

    return added_lines, modified_lines

#################################


def from_range(lines, start, end):
    retrieved_lines = list()
    for line in lines:
        n_line, _ = line.split(':', 1)
        if start <= int(n_line) <= end:
            retrieved_lines.append(line)
            
        if int(n_line) > end:
            break

    return retrieved_lines


def extract(diff: str, methods: list, commit, repo):
    # Sort by starting line
    methods.sort(key=lambda x: x.start_line)

    # Store diff lines
    lines = ['0:']
    lines.extend(diff.split('\n'))
    lines = [line for line in lines if line.strip()]

    # Extract methods
    extracted_methods = [from_range(lines, m.start_line, m.end_line) for m in methods]

    clean_methods, added_indexes = list(), list()
    for idx, method_lines in enumerate(extracted_methods):
        try:
            ## Check if the method is valid
            clean_lines = [line.split(':', 1)[1] for line in method_lines]

            # Remove empty methods / methods containing non latin characters
            full_method = '\n'.join(clean_lines)
            full_method = full_method.replace('+', '').replace('-', '')
            if is_body_empty(full_method): continue
            if invalid_string(full_method): continue

            # Retrieve lines without comments as reference
            full_method = '\n'.join(method_lines).strip()
            method_lines = full_method.split('\n')
            clean_method = remove_comments(full_method).strip()
            lines_no_comments = clean_method.split('\n')

            # Retrieve lines added from scratch
            added_lines, _ = modifications(method_lines, lines_no_comments)

            # Remove methods with no added lines
            if not added_lines: continue

            # Remove in-line / multi-line comments
            lines = clean_from_comments(method_lines)

            # Check if comment detection failed
            if comments_exist(lines): continue  

            # Format new statement indexes for random / block masking
            indexes = list()
            lines_no_deletion = [line for line in lines if line.strip() and not line.split(':', 1)[1].startswith('-')]
            for group in added_lines:
                if len(group) == 1:
                    line = group[0]
                    line_idx = lines_no_deletion.index(line)
                    indexes.append((line_idx, str(line_idx)))
                elif len(group) > 1:
                    first_line, second_line = group[0], group[-1]
                    first_line_idx, second_line_idx = lines_no_deletion.index(first_line), lines_no_deletion.index(second_line)
                    indexes.append((first_line_idx, f'[{first_line_idx}${second_line_idx}]'))

            # Remove deleted lines and lines number
            lines = [line.split(':', 1)[1] if line.strip() else line for line in lines]
            lines = [line for line in lines if not line.startswith('-')]

            # Remove the '+' symbol to the added lines
            lines = [line[1:] if line.startswith('+') else line for line in lines]
            method = '\n'.join(lines)

            # Remove possible test instances
            if not is_method(method): continue
            if is_test(method): continue

            # Format Method
            method = format(method)

            # Check for empty method after formatting
            if is_body_empty(method): continue

            # Check if braces in mask are balanced
            # Consider also possible braces in quotations
            if unbalanced_brackets(method): continue

            # Filter by token length
            n_tokens = len(method.split())
            if n_tokens < 15 or n_tokens > 500: continue

        except Exception as e:
            normalized_method = ' '.join(method_lines)
            print(f'Skipped in repo {repo} commit: {commit} line: {methods[idx].start_line}-{methods[idx].end_line} the method {normalized_method}. Error: {e}')
            continue

        # Sort and format indexes
        indexes.sort(key=lambda x: x[0])
        indexes_str = '-'.join([idx_data[1] for idx_data in indexes])

        clean_methods.append(method)
        added_indexes.append(indexes_str)

    return clean_methods, added_indexes

### UTILITY FUNCTIONS
def export_data(data_rows: list, output_filepath: str):
    # Init CSV if not exists
    if not os.path.isfile(output_filepath):
        with open(output_filepath, 'w', newline='') as f:
            writer = csv.writer(f)  
            writer.writerow(['author', 'repository', 'commit_id', 'date', 'filepath', 'method', 'added_line'])

    # Update with the input rows
    with open(output_filepath, 'a', newline='', errors='replace') as f:
        writer = csv.writer(f)   
        writer.writerows(data_rows)
    

def clear_folder(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IRWXU)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IRWXU)

    return shutil.rmtree(path, ignore_errors=True)

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_dir', '-o',
                        metavar='FILEPATH',
                        dest='output_dir',
                        required=False,
                        type=str,
                        default='output',
                        help='Directory path of the CSV file containing the retrieved methods of the developer')
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--commits_filepath', '-i',
                        metavar='FILEPATH',
                        dest='commits_filepath',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the mined commits info')
    parser.add_argument('--ext_dir', '-d',
                        metavar='PATH',
                        dest='ext_dir',
                        required=True,
                        type=str,
                        help='Name of the directory where the external data for each commit are stored')
    parser.add_argument('--repository', '-r',
                        metavar='ID',
                        dest='repository',
                        required=True,
                        type=str,
                        help='Repository to retrieve the methods')
    return parser

if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Open CSV file and retrieve ext_data ids
    commits_df = pd.read_csv(args.commits_filepath, encoding='utf-8-sig')
    commits_df = commits_df[commits_df['repository'] == args.repository]

    # Save JSON files by repository
    repos_dict = defaultdict(list)
    repositories, data_id_list = commits_df['repository'].tolist(), commits_df['data_id'].tolist()
    authors, commits, dates = commits_df['author'].tolist(), commits_df['commit_id'].tolist(), commits_df['date'].tolist()
    for author, repo, data_id, commit, date in zip(authors, repositories, data_id_list, commits, dates):
        json_path = os.path.join(args.ext_dir, f'ext_{str(int(data_id))}.json')
        repos_dict[repo].append((author, json_path, commit, date))

    # Create clone dir
    repos_dir = os.path.join(sys.path[0], 'repos')
    if not os.path.exists(repos_dir): os.makedirs(repos_dir)
    
    for repo in repos_dict:

        # Clone repo
        clone_dir = os.path.join(repos_dir, repo)
        Repo.clone_from(f'https://github.com/{repo}.git', clone_dir)
        
        exported_rows = list()
        repo_data = repos_dict[repo]
        for author_id, path, commit, date in repo_data:

            # Load the changed files for the specified commit
            with open(path, 'r', encoding='utf-8-sig') as f:
                modifications_json = json.load(f)
                changed_files = [ChangedFile(file) for file in modifications_json['changed_files']]
        
            # Generate and parse the commit diff for each modified file
            for file in changed_files:

                # Check if test is in a split of the filepath name
                filename = file.filename.split('.', 1)[0]
                test_file = False
                for word in split_camel_case(filename):
                    if word.lower() == 'test': 
                        test_file = True
                        break
                if test_file: continue

                # Skip files with no impacted methods
                if not file.changed_methods: continue

                # Skip deleted and renamed files
                if file.change_type == 'DELETE' or file.change_type == 'RENAME': continue

                # Retrieve the number of lines of the modified file
                output = subprocess.run(['git', '-C', clone_dir, 'show', f'{commit}:{file.new_path}'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                try:
                    context_lines = len(output.stdout.decode('utf-8-sig').strip().split('\n'))
                except: # Remove un-readable diffs
                    continue

                # Retrieve the diff
                flag = f'-U{context_lines}'
                output = subprocess.run(['git', '-C', clone_dir, 'diff', flag, f'{commit}^', commit, '--', file.new_path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                try:
                    diff = output.stdout.decode('utf-8-sig')
                    diff = format_diff(diff)
                except: # Remove un-readable diffs
                    continue
                
                # Extract methods and added indexes
                methods, added_indexes = extract(diff, file.methods_after, commit, repo)
                for method, indexes in zip(methods, added_indexes):
                    method_data = [author_id, repo, commit, date, file.filename, method, indexes]
                    exported_rows.append(method_data)
        
        # Export data
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        output_filename = args.repository.replace('/', '_') + '_extracted.csv'
        output_filepath = os.path.join(args.output_dir, output_filename)
        export_data(exported_rows, output_filepath)

        try:
            # Delete repo
            clear_folder(clone_dir)
        except:
            # Generate random directory name
            random_str = str(uuid.uuid4())
            dir_name = os.path.basename(clone_dir)
            new_dir = f"{dir_name}-{random_str}"
            
            # Rename the repo directory with the previous name
            parent_dir = os.path.dirname(clone_dir)
            new_dir_path = os.path.join(parent_dir, new_dir)
            os.rename(clone_dir, new_dir_path)

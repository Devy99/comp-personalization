from javalang.tokenizer import tokenize
import os, re, javalang, argparse, pandas as pd

### Functions to strip methods containing comments ###
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

def clean_from_comments(method: str):
    method = method.strip()
    lines = method.split('\n')
    
    # Remove inline / multi-line comments
    clean_method = remove_comments(method).strip()
    clean_lines = clean_method.split('\n')
    
    # Detect lines with only comments
    changes_list = list()
    for idx in range(len(clean_lines)):
        original_line, clean_line = lines[idx], clean_lines[idx]
        if original_line != clean_line and not clean_line.strip():
            changes_list.append(1)
        else:
            changes_list.append(0)
            
    # Remove lines with only comments
    clean_lines = [line for idx, line in enumerate(clean_lines) if not changes_list[idx]]
    clean_method = '\n'.join(clean_lines)
    return clean_method

def comments_exist(method: str):    
    # Tokenize method
    tokenized_method = tokenize_code(method)
    tokenized_method_no_whitespace = ''.join(tokenized_method.split())
    method_no_whitespace = ''.join(method.split())
    
    # Javalang automatically get rid of comments. So check if the final strings are equal or not
    return tokenized_method_no_whitespace != method_no_whitespace


### Functions to format methods ###
def tokenize_code(method):
  return ' '.join(token.value for token in tokenize(method))

def find_strings(method):
    clean_method = method.replace(r"\t", "\t").replace(r"\n", "\n").replace(r"\r", "\r")
    tree = javalang.parse.parse('class A {' + clean_method + '}')  
    strings = [literal.value for _, literal in tree.filter(javalang.tree.Literal) 
                if literal.value.startswith('"') or literal.value.startswith("'")]
    strings = [s.replace("\n", r"\n").replace("\r", r"\r").replace("\t", r"\t") for s in strings]
    return strings

def format(method: str):
    method = method.strip()

    # Replace newlines \ tabs in java strings
    try:
        strings = find_strings(method)
    except:
        return None
    
    for s in strings:
        # Double quotes
        no_newline = s.replace(r'\n', '<nl-n>').replace(r'\r', '<nl-r>').replace(r'\t', '<nl-t>')
        method = method.replace(s, no_newline)

        # Single quotes
        no_newline = s.replace(r'\n', '<nl-n>').replace(r'\r', '<nl-r>').replace(r'\t', '<nl-t>')
        method = method.replace(s, no_newline)

    # Preserve possible regex values and convert newlines in $NL$ (to preserve them during javalang tokenization)
    method = method.replace(r"\t", "\t")
    method = method.replace(r"\r\n","\n").replace(r"\r","\n").replace(r"\n","\n")

    # Strip method from comments
    method = clean_from_comments(method)

    method = method.replace("\n"," $NL$ ")
    method = method.replace("<nl-n>",r"\n").replace("<nl-r>",r"\r").replace("<nl-t>",r"\t")

    if method.startswith('$NL$'): 
        method = method.replace('$NL$', '', 1)

    # Tokenize code or return None if it is not valid
    try:
        # Check if the comment detection failed
        if comments_exist(method): return None
        method = tokenize_code(method)
    except:
        return None

    # Replace $NL$ with <NL>
    method = method.replace("$NL$","<NL>")

    return method

### Functions to pre-process the methods ###
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

def is_top_level_method(path):
    for node in reversed(path):
        if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
            return True
        elif isinstance(node, javalang.tree.MethodDeclaration):
            return False
    return False

def has_multiple_methods(code):
    try:
        clean_code = code.replace(r"\t", "\t").replace(r"\n", "\n").replace(r"\r\n", "\n")
        tree = javalang.parse.parse('class Main {' + clean_code + '}')
        methods = [node for path, node in tree.filter(javalang.tree.MethodDeclaration) if is_top_level_method(path)]
        return len(methods) > 1
    except:
        return True
    
def has_invalid_n_tokens(method: str):
    total_tokens = len(method.split())
    return total_tokens < 15 or total_tokens > 500


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
                        help='Filepath of the CSV file containing the processed methods')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', '-i',
                        metavar='FILEPATH',
                        dest='filepath',
                        required=True,
                        type=str,
                        help='Path of the directory containing all the CSV files with the raw Java method to process')
    required.add_argument('--filter', '-f',
                        metavar='FILEPATH',
                        dest='filter',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the repositories to retain')
    
    return parser

if __name__ == '__main__':  

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load repository to retain
    df = pd.read_csv(args.filter, encoding = 'ISO-8859-1')
    repos = df['name'].tolist()

    # Remove all repositories that start with 'apache'
    repos = [repo for repo in repos if not repo.startswith('apache')]

    # Process all the CSV files in the input directory
    print(f'Processing {args.filepath}...')
    print('----------------------------------------')
    for i, file in enumerate(os.listdir(args.filepath)):
        print(f'Processing file {i+1}...')
        filepath = os.path.join(args.filepath, file)
        df = pd.read_csv(filepath, encoding = 'utf-8')
        size = df.shape[0]
        print(f'Length raw methods: {size}')

        # Remove methods which are not in the selected repositories
        df = df[df['repository'].isin(repos)]
        if df.shape[0] == 0:  
            print('No methods to process')
            continue

        # Remove methods with different signature but same body
        m_dict = {body(method): method for method in df['method'].tolist()}
        unique_lines = list(m_dict.values())
        df = df[df['method'].isin(unique_lines)]
        df = df[df['method'].apply(has_multiple_methods)==False]
        df = df.drop_duplicates(subset='method')

        # Tokenize lines and add special characters
        df['formatted'] = df.method.apply(format)
        df = df[df['formatted'].notnull()]
        df = df[['method', 'formatted', 'repository']]

        # Further check for duplicates or broken methods on the formatted column
        df = df.drop_duplicates(subset='formatted')
        df = df[df['formatted'].apply(invalid_string)==False]
        df = df[df['formatted'].apply(is_body_empty)==False]
        df = df[df['formatted'].apply(unbalanced_brackets)==False]
        df = df[df['formatted'].apply(has_invalid_n_tokens)==False]

        n_removed = size - df.shape[0]
        print(f'Length filtered methods: {df.shape[0]}')
        print(f'Removed instances: {n_removed}')

        # Export the formatted methods to a CSV file. Append if the file already exists
        if i == 0: 
            df.to_csv(args.output, encoding='utf-8', errors='replace', index=False)
        else:
            df.to_csv(args.output, encoding='utf-8', errors='replace', index=False, mode='a', header=False)
        print('----------------------------------------')

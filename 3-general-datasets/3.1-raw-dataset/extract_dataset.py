from javalang.tokenizer import tokenize 
from datasets import load_dataset
import os, re, csv, sys, javalang, argparse

### FUNCTIONS FOR METHODS EXTRACTION
def find_quotations(sentence: str):
    """
    Extract quotations from a sentence.
    :param sentence: the sentence
    :return: the list of quotations
    """
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

def get_method_start_end(tree, method_node: javalang.tree.MethodDeclaration):
    """
    Determine the start and end position of a method in the source code.
    :param tree: the AST of the source code
    :param method_node: the method node
    :return: the start and end position of the method
    """
    start_pos, end_pos = None, None
    start_line, end_line = None, None
    for path, node in tree:
        if start_pos is not None and method_node not in path:
            end_pos = node.position
            end_line = node.position.line if node.position is not None else None
            break
        if start_pos is None and node == method_node:
            start_pos = node.position
            start_line = node.position.line if node.position is not None else None
    return (start_pos, end_pos), (start_line, end_line)

def get_method_text(lines: list, position: tuple, line_pos: tuple, last_endline_index: int):
    """
    Retrieve the text of a method from its position.
    :param lines: the lines of the source code
    :param position: the position of the method. Tuple containing the start and end position
    :param line_pos: the line position of the method. Tuple containing the start and end line
    :param last_endline_index: the index of the last line of the previous method
    :return: the text of the method
    """
    start_pos, end_pos = position
    start_line, end_line = line_pos
    if start_pos is None:
        return "", None, None, None
    else:
        startline_index = start_line - 1 
        endline_index = end_line - 1 if end_pos is not None else None 

        # Check annotations and update startline_index accordingly
        if last_endline_index is not None:
            for line in lines[(last_endline_index + 1):(startline_index)]:
                if "@" in line: 
                    startline_index = startline_index - 1
                    
        method_str = "\n".join(lines[startline_index:endline_index])
        method_str = method_str[:method_str.rfind("}") + 1] 

        # Check if braces in mask are balanced and update mask if not
        # Consider also possible braces in quotations
        quotations = find_quotations(method_str)
        q_left_count = sum([q.count('{') for q in quotations])
        q_right_count = sum([q.count('}') for q in quotations])
        total_left_braces = max(0, method_str.count('{') - q_left_count)
        total_right_braces = max(0, method_str.count('}') - q_right_count)
        brace_diff = abs(total_left_braces - total_right_braces)

        # 2. remove trailing rbrace for last methods & any external content/comments
        if not brace_diff == 0:
            for _ in range(brace_diff):
                method_str  = method_str[:method_str.rfind("}")]    
                method_str  = method_str[:method_str.rfind("}") + 1]     

        meth_lines = method_str.split("\n")                    
        last_endline_index = startline_index + (len(meth_lines) - 1) 

        return method_str, (startline_index + 1), (last_endline_index + 1), last_endline_index

### PRE-PROCESSING FUNCTIONS
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

def is_body_empty(method: str):
    # Detect Java interfaces
    if not '{' in method or not '}' in method:
      return True

    # Detect empty methods
    method = method.replace('{', '<**>', 1)
    method = replace_last(method, '}', '<**>')
    splits = method.split('<**>')
    func_body = splits[1].strip()
    if func_body == '': return True

    return False

def export_txt(filename: str, items: list):
  with open(f'{filename}', 'w', encoding = 'utf-8', errors='replace') as file:
    file.write('\n'.join(items))

def export_csv(filename: str, items: list):
    # Init CSV if not exists
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='', encoding = 'utf-8', errors='replace') as f:
            writer = csv.writer(f)  
            writer.writerow(['method', 'repository'])

    # Update with commit general info
    with open(filename, 'a', newline='', encoding = 'utf-8', errors='replace') as f:
        writer = csv.writer(f)    
        writer.writerows(items)
    

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
                        default='methods.csv',
                        help='Filepath of the CSV file containing the extracted methods')
    
    return parser

if __name__ == '__main__':  

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Delete existing output file
    if os.path.exists(args.output): os.remove(args.output)

    repo_methods = dict()
    methods_buffer = list()
    apache_removed = set()
    total_methods, total_removed = 0, 0
    
    # Load huggingface dataset
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Java"])
    for idx, instance in enumerate(iter(ds)):
        
        # Remove test files
        path = instance["path"]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]            
        test_file = False
        for word in split_camel_case(filename):
            if word.lower() == 'test': 
              test_file = True
              break
        
        if test_file: 
          total_removed += 1
          print(f'Instance {idx} removed because: test file')
          continue
        
        # Remove Apache repositories
        repo = instance["repo_name"]
        owner = repo.split('/')[0]
        if owner == 'apache': 
          total_removed += 1
          apache_removed.add(repo)
          print(f'Instance {idx} removed because: Apache repository')
          continue
          
        # Remove instance if I already have 1500 methods for that specific repository
        if repo_methods.get(repo, 0) >= 1500: 
          total_removed += 1
          print(f'Instance {idx} removed because: {repo} repo already has 1.500 methods')
          continue
        
        # Load Java file and extract methods
        lex = None
        source_code = instance["code"]
        if source_code.startswith('\ufeff'):
          source_code = source_code[1:]
        
        file_lines = source_code.split("\n")
        
        method_counter = 0
        methods, method_ranges = dict(), list()
        
        try:
          tree = javalang.parse.parse(source_code) 
          tree_iter = tree.filter(javalang.tree.MethodDeclaration)
          nodes = [method_node for _, method_node in tree_iter]
        except Exception:
          total_removed += 1
          print(f'Instance {idx} removed because: unparsable source code')
          continue
        
        for method_node in nodes:
            method_counter += 1
            
            # Skip test methods
            test_method = False
            method_name = method_node.name
            for word in split_camel_case(method_name):
                if word.lower() == 'test': 
                  test_method = True
                  break
            
            if test_method: 
              total_removed += 1
              print(f'Instance {idx} method {method_counter} removed because: test method')
              continue
            
            # Retrieve the method code within the source code
            position, line_pos = get_method_start_end(tree, method_node)
            method, start_line, end_line, lex = get_method_text(file_lines, position, line_pos, lex)
                    
            # Check if the method is contained within the start line or end line of another method
            # Used to remove nested methods
            inner_class = False
            for range_start, range_end in method_ranges:
                if range_start <= start_line <= range_end and range_start <= end_line <= range_end:
                    inner_class = True
                    break
                
            if inner_class: continue

            # Add the method range to the list
            method_ranges.append((start_line, end_line))
        
            # Tokenize code and remove unparsable methods
            method = method.strip()
            if invalid_string(method):
              total_removed += 1
              print(f'Instance {idx} method {method_counter} removed because: method contains non-latin characters or emojis')
              continue
            
            try:
              tokenize_code(method)  
            except Exception as e:
              total_removed += 1
              print(f'Instance {idx} method {method_counter} removed because: unparsable method')
              continue
            
            # Remove interfaces and methods with empty body
            if is_body_empty(method):
              total_removed += 1
              print(f'Instance {idx} method {method_counter} removed because: interface / empty body')
              continue

            # Filter by token length
            total_tokens = len(tokenize_code(method).split())
            if total_tokens < 15 or total_tokens > 500: 
              total_removed += 1
              print(f'Instance {idx} method {method_counter} removed because: {total_tokens} tokens')
              continue

            # Remove possible broken method after extraction 
            try:
              if not is_method(method): 
                total_removed += 1
                print(f'Instance {idx} method {method_counter} removed because: not a method')
                continue
            except:
                total_removed += 1
                print(f'Instance {idx} method {method_counter} removed because: unparsable method')
                continue
            
            total_methods += 1
            repo_methods[repo] = repo_methods.get(repo, 0) + 1
          
            # Retain newlines
            method = method.replace("\n",r"\n").replace("\r",r"\r").replace("\t",r"\t")
            methods_buffer.append([method, repo])
            
        # Print progress and empty buffer
        if idx % 10000 == 0:
          print(f'Total analyzed methods = {total_methods}. Total removed instances / methods = {total_removed}')
          export_csv(args.output, methods_buffer)
          methods_buffer = list()
    
    # Export remaining methods
    if len(methods_buffer) > 0: export_csv(args.output, methods_buffer)
    
    print(f'Total removed apache repositories: {len(apache_removed)}')
    export_txt(os.path.join(sys.path[0], 'apache_removed.txt'), apache_removed)
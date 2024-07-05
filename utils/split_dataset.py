import random, argparse, pandas as pd


def split_by_column(df: pd.DataFrame, ratio: int, splitting_col: str):
    random.seed(230923)
    size = df.shape[0]
    
    # Collect number of methods for category
    elements = df[splitting_col].value_counts().index.tolist()
    counts = df[splitting_col].value_counts()
    elements_counts = [[item, counts[idx]] for idx, item in enumerate(elements)]
    random.shuffle(elements_counts)

    # Define number of elements for the first split
    n_first_split = int(size*(ratio/100))

    count_first, elements_first = 0, list()
    while count_first < n_first_split:
        item = elements_counts.pop(0)
        element, occur = item[0], item[1]
        elements_first.append(element)
        count_first += int(occur)

    df_first_split = df[df[splitting_col].isin(elements_first)]
    print('First split: done')

    elements_second = [item[0] for item in elements_counts]
    df_second_split = df[df[splitting_col].isin(elements_second)]
    print('Second split: done')
    
    return df_first_split, df_second_split

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='optional arguments')
    parser.add_argument('--first_split_name', '-f',
                        metavar='SPLIT NAME',
                        dest='first_split',
                        required=False,
                        type=str,
                        default='train.txt',
                        help='Name of the file containing the first split')
    parser.add_argument('--second_split_name', '-s',
                        metavar='SPLIT NAME',
                        dest='second_split',
                        required=False,
                        type=str,
                        default='eval.txt',
                        help='Name of the file containing the second split')  
    parser.add_argument('--split_by', '-c',
                        metavar='COLUMN',
                        dest='split_by',
                        required=False,
                        type=str,
                        default=None,
                        help='The column of the CSV file where to perform the splitting. The ratio applies on the number of occurrences for the selected column') 
    parser.add_argument('--retain_only', '-l',
                        metavar='COLUMN',
                        dest='retain',
                        required=False,
                        type=str,
                        default=None,
                        help='The only column to extract from the dataset. If selected, the output will be a TXT file')     
    parser.add_argument('--file_type', '-t',
                        metavar='TYPE',
                        dest='file_type',
                        required=False,
                        type=str,
                        default='text',
                        choices=['csv', 'text'],
                        help='Type of the file to split')  

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input', '-i',
                        metavar='PATH',
                        dest='filepath',
                        required=True,
                        type=str,
                        help='Path of the file to split.')
    required.add_argument('--split_ratio', '-r',
                        metavar='RATIO',
                        dest='ratio',
                        required=True,
                        type=int,
                        help='Splitting ratio. The value refers to the percentage of instances retained by the first split. \
                            E.g. insert 90 to separate the 90% of the instances in the first split, and the remaining 10% in the second one.')
     
    return parser



if __name__ == '__main__':   
    random.seed(230923)
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    
    is_csv = args.file_type == 'csv'

    # Split CSV by column
    if is_csv and args.split_by:
        df = pd.read_csv(args.filepath, encoding = 'utf-8', on_bad_lines='skip')
        splitting_col = args.split_by
        first_df, second_df = split_by_column(df, args.ratio, splitting_col)

        # Retrieve only the selected columns and export as TXT file
        if args.retain:
            with open(args.first_split, 'w', encoding = 'utf-8', errors='replace') as first_file, \
                 open(args.second_split, 'w', encoding = 'utf-8', errors='replace') as second_file:
                    first_file.write('\n'.join(first_df[args.retain].tolist()))
                    second_file.write('\n'.join(second_df[args.retain].tolist()))
        else:
            first_df.to_csv(args.first_split, encoding='utf-8', errors='replace', index=False)
            second_df.to_csv(args.second_split, encoding='utf-8', errors='replace', index=False)
        exit()
         
    
    # Load file
    filepath = args.filepath
    with open(f'{filepath}', 'r', encoding = 'utf-8', errors='replace') as file:
        lines = file.readlines()
        if is_csv: header = lines.pop(0)
        
        # Fix newline for the last element
        last_line = lines.pop().strip() + '\n'
        lines.append(last_line)
        
        # Remove duplicates from the file
        lines = list(set(lines))
        
    # Shuffle lines
    random.shuffle(lines)

    # Define the index of the split
    idx = int(len(lines)*(args.ratio/100))
    first_split, second_split = lines[:idx], lines[idx:]
    assert set(first_split).isdisjoint(second_split) , "First split and second split have some common instances"

    # Export results
    with open(args.first_split, 'w', encoding = 'utf-8', errors='replace') as first_f, \
        open(args.second_split, 'w', encoding = 'utf-8', errors='replace') as second_f:
            
            # Remove empty newline at the end of the file
            first_split[-1] = first_split[-1].strip()
            second_split[-1] = second_split[-1].strip()
            
            if is_csv: 
                first_split.insert(0, header)
                second_split.insert(0, header)
            
            first_f.writelines(first_split)
            second_f.writelines(second_split)
import argparse, pandas as pd


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file_type', '-t',
                        metavar='TYPE',
                        dest='file_type',
                        required=False,
                        type=str,
                        default='csv',
                        choices=['csv', 'text'],
                        help='Type of the file to split')  
    parser.add_argument('--first_column', '-c',
                        metavar='NAME',
                        dest='first_column',
                        required=False,
                        type=str,
                        help='Name of the column to check for the first dataframe')
    parser.add_argument('--second_column', '-l',
                        metavar='NAME',
                        dest='second_column',
                        required=False,
                        type=str,
                        help='Name of the column to check for the second dataframe')
    parser.add_argument('--first_target', '-g',
                        metavar='NAME',
                        dest='first_target',
                        required=False,
                        type=str,
                        default=None,
                        help='Name of the target column to check for the first dataframe')
    parser.add_argument('--second_target', '-i',
                        metavar='NAME',
                        dest='second_target',
                        required=False,
                        type=str,
                        default=None,
                        help='Name of the target column to check for the second dataframe')
    parser.add_argument('--verbose', '-v',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='If selected, it prints the duplicated instances.')
    
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--first_input', '-f',
                        metavar='PATH',
                        dest='first_filepath',
                        required=True,
                        type=str,
                        help='Path of the first CSV file to check.')
    required.add_argument('--second_input', '-s',
                        metavar='PATH',
                        dest='second_filepath',
                        required=True,
                        type=str,
                        help='Path of the second CSV file to check.')
    return parser

if __name__ == '__main__':   
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Retrieve data and remove whitespaces for duplicates checking
    if args.file_type == 'text':
        first_elements = open(args.first_filepath, encoding='utf-8').readlines()
        first_elements = [''.join(line.split()) for line in first_elements]

        second_elements = open(args.second_filepath, encoding='utf-8').readlines()
        second_elements = [''.join(line.split()) for line in second_elements]
    else:
        first_df = pd.read_csv(args.first_filepath, encoding='utf-8')
        second_df = pd.read_csv(args.second_filepath, encoding='utf-8')
        first_column, second_column = args.first_column, args.second_column

        first_df['input_no_whitespaces'] = first_df[first_column].astype(str).replace('\s+', '', regex=True)
        first_elements = first_df['input_no_whitespaces'].tolist()
        second_df['input_no_whitespaces'] = second_df[second_column].astype(str).replace('\s+', '', regex=True)
        second_elements = second_df['input_no_whitespaces'].tolist()

        if args.first_target and args.second_target:
            first_df['target_no_whitespaces'] = first_df[args.first_target].astype(str).replace('\s+', '', regex=True)
            second_df['target_no_whitespaces'] = second_df[args.second_target].astype(str).replace('\s+', '', regex=True)
            common_df = pd.merge(first_df, second_df,  how='inner', on=['input_no_whitespaces', 'target_no_whitespaces'])



    # Check for duplicates ( only input column )
    print(f'Size of the dataframe {args.first_filepath}: {str(len(first_elements))}')
    first_no_dup = set(first_elements)
    print(f'Size of the dataframe {args.first_filepath} - no duplicates: {str(len(first_no_dup))}')

    print(f'Size of the dataframe {args.second_filepath}: {str(len(second_elements))}')
    second_no_dup = set(second_elements)

    print(f'Size of the dataframe {args.second_filepath} - no duplicates: {str(len(second_no_dup))}')

    # Merge elements
    commons = first_no_dup & second_no_dup
    print(f'Number of overlapping instances by input column only: {str(len(commons))}')
    if args.verbose:
        print('Overlapping instances:')
        print(commons)

    # Merge elements by input and target
    if args.first_target and args.second_target:
        print(f'Number of overlapping instances by input and target columns: {str(common_df.shape[0])}')
        if args.verbose:
            print('Overlapping instances:')
            print(common_df.to_string())

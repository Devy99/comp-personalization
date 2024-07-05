from utils.consecutive_line_masker import ConsecutiveLineMasker
from collections import defaultdict
import os, random, argparse, pandas as pd, numpy as np

def masking_type(mask):
    symbol_counter = mask.count('<NL>')
    return 'line' if symbol_counter == 0 else 'block'

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

    required = parser.add_argument_group('required arguments')
    required.add_argument('--input_filepath', '-i',
                        metavar='FILEPATH',
                        dest='input_filepath',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing the methods to mask')
    required.add_argument('--devs_total_filepath', '-d',
                        metavar='FILEPATH',
                        dest='devs_total_filepath',
                        required=True,
                        type=str,
                        help='Filepath of the CSV file containing all the instances of the developers dataset and the total number of masked tokens') 
    return parser

if __name__ == '__main__':
    random.seed(230923)
    np.random.seed(230923)

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Load dataset for the general model
    general_df = pd.read_csv(args.input_filepath, encoding='utf-8-sig')
    general_df_len = general_df.shape[0]

    # Add a column with an identifier for each method
    general_df['idx'] = general_df.index

    # Load total developers dataset
    devs_df = pd.read_csv(args.devs_total_filepath, encoding='utf-8-sig')
    devs_df_len = devs_df.shape[0]

    # Count the number of occurrences for each mask_len
    mask_lens = devs_df['mask_len'].tolist()
    mask_lens = np.array(mask_lens)
    n_masked_list, n_occurrences_list = np.unique(mask_lens, return_counts=True)

    # Find also the line distribution of the developers dataset. 
    devs_df['masking_type'] = devs_df['mask'].apply(masking_type)

    # Count the number of occurrences for each masking_type
    masking_types = devs_df['masking_type'].tolist()
    masking_types = np.array(masking_types)
    masking_types_list, n_occurrences_types_list = np.unique(masking_types, return_counts=True)

    # Count % of line and block masking
    masking_type_prob_dict = dict()
    print('Masking types distribution:')
    for m_type, n_occ in zip(masking_types_list, n_occurrences_types_list):
        print(f'{m_type} masking: {n_occ} occurrences')
        prob = n_occ / devs_df_len  
        masking_type_prob_dict[m_type] = prob
        print(f'{m_type} masking: {(prob*100):.2f}%')

    # Mask dataset
    masked_times_method_dict = defaultdict(int)
    masked_indexes_list = defaultdict(list)
    for n_masked, n_occ in zip(reversed(n_masked_list), reversed(n_occurrences_list)):
        print(f'Masking {n_masked} tokens...')

        # Get the number of methods to mask in the general dataset proportionally to the number of masked tokens
        occ_to_mask = int((n_occ / devs_df_len) * general_df_len)
        print(f'Number of occurrences to mask: {occ_to_mask}')

        # PErcentual of occurrences to mask
        print(f'Percentual of occurrences to mask: {(occ_to_mask / general_df_len) * 100:.2f}%')

        # Shuffle the dataset at each iteration
        general_df = general_df.sample(frac=1, random_state=42)

        # Iterate over the general dataset
        added_idx = list()
        added_masked_methods = 0
        masked_df_instances = list()
        for idx, row in general_df.iterrows():
            # Retrieve the method to mask
            row_id, method_to_mask, repo = row['idx'], row['formatted'], row['repository']

            # Use a method up to 3 times
            if masked_times_method_dict[row_id] >= 2: 
                general_df = general_df.drop(idx)
                continue

            # Setup the masker
            params = {'methods': [method_to_mask], 'sep': '<NL>', 'mask_symbol': '<extra_id_0>', 'idx_blacklist': [masked_indexes_list[row_id]]}
            params['min_tokens'] = params['max_tokens'] = n_masked
            params['max_masking'] = 1

            # Get 'line' or 'block' randomly but weighted according to the number of occurrences
            r_mask_type = np.random.choice(['line', 'block'], p=[masking_type_prob_dict['line'], masking_type_prob_dict['block']])
            params['max_lines'] = 1 if r_mask_type == 'line' else 3

            masker = ConsecutiveLineMasker(**params)
            masker.mask()
                
            # Iterate to try to mask the method with a different number of lines
            if len(masker._masked_codes) == 0: continue

            added_masked_methods += 1
            masked_times_method_dict[row_id] += 1
            masked_method, mask = masker._masked_codes[0], masker._masks[0]
            masked_indexes_list[row_id].extend(masker._masked_indexes)

            # Add the masked method to the dataset
            assert len(mask.split()) == n_masked, f'Masked {len(mask.split())} tokens instead of {n_masked}'
            new_instance = {'row_id': row_id, 'repository': repo, 'method': method_to_mask, 'masked_method': masked_method, 'mask': mask, 'mask_len': n_masked, 'mask_occ': added_masked_methods}
            masked_df_instances.append(new_instance)

            if added_masked_methods == occ_to_mask: break
        
        print(f'Added instances: {added_masked_methods}')
        print(f'Number of remained instances: {len(general_df)}')
        print()

        # Check if the number of added instances is within a delta of 5% of the expected number of instances
        delta = int(occ_to_mask * 0.05)
        if added_masked_methods < occ_to_mask - delta: print(masked_indexes_list)
        assert added_masked_methods >= occ_to_mask - delta, f'Added {added_masked_methods} instances instead of {occ_to_mask}'

        # Export the dataset in append mode if it already exists
        masked_df = pd.DataFrame(masked_df_instances, columns=['row_id', 'repository', 'method', 'masked_method', 'mask', 'mask_len', 'mask_occ'])
        if os.path.exists(args.output_filepath):
            masked_df.to_csv(args.output_filepath, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            masked_df.to_csv(args.output_filepath, index=False, encoding='utf-8-sig')
    
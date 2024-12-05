import os, pickle, argparse
import javalang, pandas as pd
import matplotlib.pyplot as plt
import itertools, multiprocessing

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--datasets_dir', '-d',
                          metavar='PATH',
                          dest='datasets_dir',
                          required=True,
                          type=str,
                          help='Path of the directory containing the datasets')
    required.add_argument('--general_train_set', '-g',
                          metavar='FILEPATH',
                          dest='general_train_set',
                          required=True,
                          type=str,
                          help='Filepath of the CSV file containing the train set of the general finetuning.')

    return parser

def identifiers_and_literals(java_code):
    java_code = java_code.replace('<NL>', '\n')
    java_code = f'public class Test {{ {java_code} }}'
    identifiers, literals = list(), list()

    try:
        tree = javalang.parse.parse(java_code)
        for _, node in tree.filter(javalang.tree.Literal):
            literals.append(node.value)
        for _, node in tree.filter(javalang.tree.MemberReference):
            identifiers.append(node.member)
        for _, node in tree.filter(javalang.tree.LocalVariableDeclaration):
            identifiers.append(node.declarators[0].name)
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            identifiers.append(node.name)
        for _, node in tree.filter(javalang.tree.FormalParameter):
            identifiers.append(node.name)
    except:
        return list(), list(), list()

    id_lit = list(set(identifiers + literals))

    return identifiers, literals, id_lit

def process_method(method):
    _, _, all_id_lit = identifiers_and_literals(method)
    return all_id_lit

def show_results(filename, results_df: pd.DataFrame, y_title=''):
    results_df.to_csv(f'{filename}.csv', index=False)

    # Get mean and median of n_general, n_dev, and n_org
    mean_general, median_general = results_df['n_general'].mean(), results_df['n_general'].median()
    mean_dev, median_dev = results_df['n_dev'].mean(), results_df['n_dev'].median()
    mean_org, median_org = results_df['n_org'].mean(), results_df['n_org'].median()

    print(f'Mean general: {mean_general}, Median general: {median_general}')
    print(f'Mean dev: {mean_dev}, Median dev: {median_dev}')
    print(f'Mean org: {mean_org}, Median org: {median_org}')

    # Boxplot
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.grid(False) 
    
    boxplot = ax.boxplot([results_df['n_org'], results_df['n_dev'], results_df['n_general']], patch_artist=True, vert=False, widths=0.5)
    colors = ['white', 'lightslategray', 'darkslategray']
    for idx, box in enumerate(boxplot['boxes']):
        box.set(color='black', linewidth=1)  
        box.set(facecolor = colors[idx] )

    for median in boxplot['medians']:
        median.set(color='orangered', linewidth=1)

    ax.set_xlim([0, 1.0])
    labels = ['O', 'D', 'B']
    #ax.set_title('Number of literals and identifiers shared with test sets', fontsize=12, weight='bold', y=1.03)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_yticklabels(labels, fontsize=8, weight='bold')
    ax.set_xlabel(y_title, fontsize=11, weight='light', labelpad=10)
    ax.set_xlabel(None, fontsize=11, weight='light', labelpad=10)
    plt.tight_layout() 
    plt.savefig(f'{filename}.png', dpi=300)

if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    if os.path.exists('results_train.csv') and os.path.exists('results_test.csv'):
        train_df = pd.read_csv('results_train.csv')
        print('Train results')
        show_results('results_train_2', train_df, "% of occurrences respect test set")

        print('\nTest results')
        test_df = pd.read_csv('results_test.csv')
        show_results('results_test_2', test_df, "% of occurrences respect train set")
        exit()
    
    # Load general finetuning dataset
    general_train_set = pd.read_csv(args.general_train_set)
    print(f'Loading general finetuning dataset with {len(general_train_set)} methods')

    if os.path.exists('general_id_lits.pkl'):
        with open('general_id_lits.pkl', 'rb') as f:
            general_id_lits = pickle.load(f)
    else:
        general_methods = general_train_set['method'].tolist()

        # Parallelize identifiers_and_literals on general_methods with multiprocess
        with multiprocessing.Pool(32) as pool:
            general_id_lits = pool.map(process_method, general_methods)
            general_id_lits = set(itertools.chain(*general_id_lits))
        
        # Save general_id_lits to a file
        with open('general_id_lits.pkl', 'wb') as f:
            pickle.dump(general_id_lits, f)

    # Load datasets
    results_train_df = pd.DataFrame(columns=['developer', 'n_general', 'n_dev', 'n_org'])
    results_test_df = pd.DataFrame(columns=['developer', 'n_general', 'n_dev', 'n_org'])
    datasets = os.listdir(args.datasets_dir)
    for dataset in datasets:
        print(f'Processing {dataset}')

        dev_n = int(dataset.split('_')[1])        
        dev_path = os.path.join(args.datasets_dir, dataset, 'developer_masked_methods_train.csv')
        org_path = os.path.join(args.datasets_dir, dataset, 'apache_dataset_total_train.csv')
        test_path = os.path.join(args.datasets_dir, dataset, 'developer_masked_methods_test.csv')

        dev_df, org_df, test_df = pd.read_csv(dev_path), pd.read_csv(org_path), pd.read_csv(test_path)
        dev_df['method'] = dev_df.apply(lambda row: row['masked_method'].replace('<extra_id_0>', row['mask']), axis=1)
        org_df['method'] = org_df.apply(lambda row: row['masked_method'].replace('<extra_id_0>', row['mask']), axis=1)
        test_df['method'] = test_df.apply(lambda row: row['masked_method'].replace('<extra_id_0>', row['mask']), axis=1)

        # Get list of all identifiers and literals in the developer, org, and test datasets
        with multiprocessing.Pool(32) as pool:
            dev_id_lits = pool.map(process_method, dev_df['method'].tolist())
            org_id_lits = pool.map(process_method, org_df['method'].tolist())
            test_id_lits = pool.map(process_method, test_df['method'].tolist())

            dev_all_id_lit = set(itertools.chain(*dev_id_lits))
            org_all_id_lit = set(itertools.chain(*org_id_lits))
            test_all_id_lit = set(itertools.chain(*test_id_lits))

        # Count how many identifiers that are in the test set are also in the other datasets. 
        n_general = len(test_all_id_lit.intersection(general_id_lits))
        n_dev = len(test_all_id_lit.intersection(dev_all_id_lit))
        n_org = len(test_all_id_lit.intersection(org_all_id_lit))

        # Save results normalized by the number of vocabs in the train set
        n_general_train = n_general / len(general_id_lits)
        n_dev_train = n_dev / len(dev_all_id_lit)
        n_org_train = n_org / len(org_all_id_lit)
        new_row = pd.DataFrame({'developer': [dev_n], 'n_general': [n_general_train], 'n_dev': [n_dev_train], 'n_org': [n_org_train]})
        results_train_df = pd.concat([results_train_df, new_row], ignore_index=True)
        
        # Save results normalized by the number of vocabs in the test set
        n_general_test = n_general / len(test_all_id_lit)
        n_dev_test = n_dev / len(test_all_id_lit)
        n_org_test = n_org / len(test_all_id_lit)
        new_row = pd.DataFrame({'developer': [dev_n], 'n_general': [n_general_test], 'n_dev': [n_dev_test], 'n_org': [n_org_test]})
        results_test_df = pd.concat([results_test_df, new_row], ignore_index=True)

    show_results('results_train', results_train_df)
    show_results('results_test', results_test_df)

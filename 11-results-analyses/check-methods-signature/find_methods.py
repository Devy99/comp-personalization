import os, argparse
import matplotlib.pyplot as plt
import pandas as pd

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

def print_outliers(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]['developer']
            print(f"dev_id values for outliers in {col} are {outliers.tolist()}")

def signature(method: str) -> str:
    signature = method.split('{')[0]
    while '@' in signature:
        if '<NL>' not in signature: break
        signature = signature.split('<NL>', 1)[1]

    # Remove space for matching
    signature = signature.replace(' ', '')
    return signature

def show_results(results_df: pd.DataFrame):
    # Get mean and median of n_general, n_dev, and n_org
    mean_general, median_general = results_df['n_general'].mean(), results_df['n_general'].median()
    mean_dev, median_dev = results_df['n_dev'].mean(), results_df['n_dev'].median()
    mean_org, median_org = results_df['n_org'].mean(), results_df['n_org'].median()

    print(f'Mean general: {mean_general}, Median general: {median_general}')
    print(f'Mean dev: {mean_dev}, Median dev: {median_dev}')
    print(f'Mean org: {mean_org}, Median org: {median_org}')

    print_outliers(results_df)

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
    ax.tick_params(axis='x', labelsize=8)
    ax.set_yticklabels(labels, fontsize=8, weight='bold')
    ax.set_xlabel(None, fontsize=11, weight='light', labelpad=10)
    plt.tight_layout() 
    plt.savefig(f'results2.png', dpi=300)
    

if __name__ == '__main__':
    
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    if os.path.exists('results.csv'):
        results_df = pd.read_csv('results.csv')
        show_results(results_df)
        exit()

    # Load general finetuning dataset
    general_train_set = pd.read_csv(args.general_train_set)
    general_train_set['signature'] = general_train_set['method'].apply(signature)
    general_signatures = general_train_set['signature'].tolist()

    # Load datasets
    results_df = pd.DataFrame(columns=['developer', 'n_general', 'n_dev', 'n_org'])
    datasets = os.listdir(args.datasets_dir)
    for dataset in datasets:
        print(f'Processing {dataset}')

        dev_n = int(dataset.split('_')[1])        
        dev_path = os.path.join(args.datasets_dir, dataset, 'developer_masked_methods_train.csv')
        org_path = os.path.join(args.datasets_dir, dataset, 'apache_dataset_total_train.csv')
        test_path = os.path.join(args.datasets_dir, dataset, 'developer_masked_methods_test.csv')

        dev_df, org_df, test_df = pd.read_csv(dev_path), pd.read_csv(org_path), pd.read_csv(test_path)
        dev_df['method'] = dev_df.apply(lambda row: row['masked_method'].replace('<extra_id_0>', row['mask']), axis=1)
        dev_df['signature'] = dev_df['method'].apply(signature)
        org_df['method'] = org_df.apply(lambda row: row['masked_method'].replace('<extra_id_0>', row['mask']), axis=1)
        org_df['signature'] = org_df['method'].apply(signature)
        test_df['method'] = test_df.apply(lambda row: row['masked_method'].replace('<extra_id_0>', row['mask']), axis=1)
        test_df['signature'] = test_df['method'].apply(signature)

        dev_signatures = dev_df['signature'].tolist()
        org_signatures = org_df['signature'].tolist()
        test_signatures = test_df['signature'].tolist()
        assert len(test_signatures) == 500

        # Count how many signature that are in the test set are also in the other datasets.
        n_general, n_dev, n_org = 0, 0, 0
        for sig in set(test_signatures):
            n_instances = test_signatures.count(sig)
            contr = n_instances / len(test_signatures)
            n_general += contr if sig in general_signatures else 0
            n_dev += contr if sig in dev_signatures else 0
            n_org += contr if sig in org_signatures else 0

        new_row = pd.DataFrame({'developer': [dev_n], 'n_general': [n_general], 'n_dev': [n_dev], 'n_org': [n_org]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    results_df.to_csv('results.csv', index=False)
    show_results(results_df)

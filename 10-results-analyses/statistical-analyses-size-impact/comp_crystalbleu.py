

from nltk.util import ngrams
from collections import Counter
from crystalbleu import corpus_bleu
from pygments.lexers.jvm import JavaLexer
import pandas as pd
import os, json, pickle, argparse


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--predictions_dir', '-i',
                        metavar='PATH',
                        dest='predictions_dir',
                        required=True,
                        type=str,
                        help='Name of the directory containing the predictions of the models')
    required.add_argument('--results_dir', '-r',
                        metavar='PATH',
                        dest='results_dir',
                        required=True,
                        type=str,
                        help='Name of the directory where to store the results')
    required.add_argument('--training_dataset_path', '-g',
                        metavar='PATH',
                        dest='training_dataset_path',
                        required=True,
                        type=str,
                        help='Name of the file containing the general finetuning methods')
    
    return parser

def trivially_shared_ngrams(methods: list) -> dict:
	ngrams_filename = 'trivially_shared_ngrams.pickle'
	if os.path.exists(ngrams_filename):
		with open(ngrams_filename, 'rb') as f:
			return pickle.load(f)

	# Extract all n-grams of length 1-4
	k = 500
	all_ngrams = list()
	for idx, method in enumerate(methods):
		if idx != 0 and idx % 100000 == 0:
			print(f'Processed {idx} methods')

		if not isinstance(method, str):
			continue
		
		tokens = method.split()
		for n in range(1, 5):
			all_ngrams.extend(list(ngrams(tokens, n)))
	
	# Calculate frequencies of all n-grams
	frequencies = Counter(all_ngrams)
	trivially_shared_ngrams = dict(frequencies.most_common(k))
	print(trivially_shared_ngrams)

	# Save trivially shared ngrams on file
	with open(ngrams_filename, 'wb') as f:
		pickle.dump(trivially_shared_ngrams, f, protocol=pickle.HIGHEST_PROTOCOL)

	#print(f'Trivially shared n-grams: {trivially_shared_ngrams}')
	trivially_shared_ngrams_dict = dict()
	for k in trivially_shared_ngrams.keys():
		trivially_shared_ngrams_dict[str(k)]=k
  
	with open('trivially_shared_ngrams.txt', 'w') as convert_file:
		convert_file.write(json.dumps(trivially_shared_ngrams_dict))

	return trivially_shared_ngrams

def compute_crystal_bleu(trivially_shared_ngrams: dict, predictions_df: list, label: str, results_dir: str) -> float:
	inputs = predictions_df['input'].tolist()
	targets = predictions_df['target'].tolist()
	predictions = predictions_df['prediction'].tolist()

	methods = list()
	for m_code, mask in zip(inputs, targets):
		method = m_code.replace('<extra_id_0>', mask)
		methods.append(method)

	lexer = JavaLexer()
	labels, scores, f_methods, f_targets, f_predictions = list(), list(), list(), list(), list()
	for pred, target, method in zip(predictions, targets, methods):
		tokens_predictions = ' '.join([t[1] for t in lexer.get_tokens(pred) if t[1].strip()]).split()
		tokens_target = ' '.join([t[1] for t in lexer.get_tokens(target) if t[1].strip()]).split()

		weights = (1./3., 1./3., 1./3.)
		score = corpus_bleu([[tokens_target]], [tokens_predictions], weights=weights, ignoring=trivially_shared_ngrams)
		
		scores.append(round(score, 2))
		f_methods.append(method)
		f_targets.append(target)
		f_predictions.append(pred)
		labels.append(label)

	scores_dict={'model': labels, 'method': f_methods, 'crystalbleu': scores, 'target': f_targets, 'prediction': f_predictions}
	final_df = pd.DataFrame(scores_dict)
	final_df['crystalbleu'] = final_df.apply(lambda x: 1.0 if ''.join(x['target'].split()) == ''.join(x['prediction'].split()) else x['crystalbleu'], axis=1)
	final_df.to_csv(os.path.join(results_dir, f'{label}_crystalbleu.csv'))
	return final_df

if __name__ == '__main__':

	# Read arg parameters
	parser = get_argparser()
	args = parser.parse_args()

	# Create a directory to store the results
	results_dir = args.results_dir
	if not os.path.exists(results_dir): os.makedirs(results_dir)

	# Create dataframe where to cumulate the model predictions
	pt_sft1_df, sft1_sft2_dev_df = pd.DataFrame(), pd.DataFrame()
	sft1_sft2_all_dev_df, sft1_sft2_all_small_df, sft1_sft2_rnd_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	# Retrieve devs ids
	devs_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	# Iterate over all the developers directories
	for dev_id in devs_ids:
		dev_dir = os.path.join(args.predictions_dir, f'developer_{dev_id}')
		
		models_dirs = os.listdir(dev_dir)
		models_dirs = [dir for dir in models_dirs if os.path.isdir(os.path.join(dev_dir, dir)) and dir != 'datasets']
		
		for m_dir in models_dirs:
			dir_path = os.path.join(dev_dir, m_dir)
			test_df = pd.read_csv(os.path.join(dir_path, 'predictions_test.csv'))
			if m_dir == 'pt-sft1': pt_sft1_df = pd.concat([pt_sft1_df, test_df], ignore_index=True)
			elif m_dir == 'sft1-sft2-all-dev': sft1_sft2_all_dev_df= pd.concat([sft1_sft2_all_dev_df, test_df], ignore_index=True)
			elif m_dir == 'sft1-sft2-all-small': sft1_sft2_all_small_df = pd.concat([sft1_sft2_all_small_df, test_df], ignore_index=True)
			elif m_dir == 'sft1-sft2-dev': sft1_sft2_dev_df = pd.concat([sft1_sft2_dev_df, test_df], ignore_index=True)
			elif m_dir == 'rnd': sft1_sft2_rnd_df = pd.concat([sft1_sft2_rnd_df, test_df], ignore_index=True)

	# Compute trivially shared n-grams
	general_df = pd.read_csv(args.training_dataset_path)
	methods = general_df['formatted'].tolist()
	ts_ngrams = trivially_shared_ngrams(methods)

	# Compute crystalBLUE scores and assign 1 to exact matches
	cbs_df = pd.DataFrame()
	cb_pt_sft1_df = compute_crystal_bleu(ts_ngrams, pt_sft1_df, 'baseline', results_dir)
	cb_sft1_sft2_dev_df = compute_crystal_bleu(ts_ngrams, sft1_sft2_dev_df, 'developer', results_dir)
	cb_sft1_sft2_all_dev_df = compute_crystal_bleu(ts_ngrams, sft1_sft2_all_dev_df, 'organization', results_dir)
	cb_sft1_sft2_all_small_df = compute_crystal_bleu(ts_ngrams, sft1_sft2_all_small_df, 'organization_subset', results_dir)
	cb_sft1_sft2_rnd_df = compute_crystal_bleu(ts_ngrams, sft1_sft2_rnd_df, 'baseline_plus', results_dir)

	# Concat CystalBLEU scores and generate boxplot
	cbs_df = pd.concat([cbs_df, cb_pt_sft1_df], ignore_index=True)
	cbs_df = pd.concat([cbs_df, cb_sft1_sft2_dev_df], ignore_index=True)
	cbs_df = pd.concat([cbs_df, cb_sft1_sft2_all_small_df], ignore_index=True)
	cbs_df = pd.concat([cbs_df, cb_sft1_sft2_all_dev_df], ignore_index=True)
	cbs_df = pd.concat([cbs_df, cb_sft1_sft2_rnd_df], ignore_index=True)

	# Rename the value of the column model to be more readable
	cbs_df['model'] = cbs_df['model'].replace('pt_sft1', 'Baseline')
	cbs_df['model'] = cbs_df['model'].replace('sft1_sft2_dev', 'Developer')
	cbs_df['model'] = cbs_df['model'].replace('sft1_sft2_all_small', 'Organization - subset')
	cbs_df['model'] = cbs_df['model'].replace('sft1_sft2_all_dev', 'Organization')
	cbs_df['model'] = cbs_df['model'].replace('sft1_sft2_rnd', 'Baseline+')

	# Get mean, median, Q1 and Q3 for each model
	df_info = cbs_df.groupby('model')['crystalbleu'].describe()
	print("CrystalBLEU scores analysis")
	print(df_info)

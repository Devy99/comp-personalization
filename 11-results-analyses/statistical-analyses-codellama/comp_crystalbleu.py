from nltk.util import ngrams
from collections import Counter
from crystalbleu import corpus_bleu, SmoothingFunction
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

	trivially_shared_ngrams_dict = dict()
	for k in trivially_shared_ngrams.keys():
		trivially_shared_ngrams_dict[str(k)]=k
  
	with open('trivially_shared_ngrams.txt', 'w') as convert_file:
		convert_file.write(json.dumps(trivially_shared_ngrams_dict))

	return trivially_shared_ngrams

def compute_crystal_bleu(trivially_shared_ngrams: dict, predictions_df: list) -> float:
	scores = list()
	lexer = JavaLexer()

	sm_func = SmoothingFunction(epsilon=0.0001).method1
	predictions, targets = predictions_df['prediction'].tolist(), predictions_df['parsed_mask'].tolist()
	for pred, target in zip(predictions, targets):
		tokens_predictions = ' '.join([t[1] for t in lexer.get_tokens(str(pred)) if t[1].strip()]).split()
		tokens_target = ' '.join([t[1] for t in lexer.get_tokens(str(target)) if t[1].strip()]).split()

		weights = (1./3., 1./3., 1./3.)
		score = corpus_bleu([[tokens_target]], [tokens_predictions], weights=weights, ignoring=trivially_shared_ngrams, smoothing_function=sm_func)
		scores.append(round(score, 2))
	
	predictions_df['crystalbleu'] = scores
	predictions_df['crystalbleu'] = predictions_df.apply(lambda x: 1.0 if x['exact_match'] == True else x['crystalbleu'], axis=1)
	return predictions_df

if __name__ == '__main__':

	# Read arg parameters
	parser = get_argparser()
	args = parser.parse_args()

	# Create a directory to store the results
	results_dir = args.results_dir
	if not os.path.exists(results_dir): os.makedirs(results_dir)

	# Create dataframe where to cumulate the model predictions
	cumulate_df = pd.DataFrame()
	model_mapping = {'rnd': 'Baseline+',
					'pt-sft1': 'Baseline', 
					'pt-sft1-ft': 'Baseline 10k', 
					'sft1-sft2-dev': 'Developer', 
					'sft1-sft2-all-dev': 'Organization',
					'sft1-sft2-all-small': 'Organization - subset'}
	
	devs_ids = list()
	for dev_dir in os.listdir(args.predictions_dir):
		devs_ids.append(dev_dir.split('_')[1])

	# Compute trivially shared n-grams
	general_df = pd.read_csv(args.training_dataset_path)
	methods = general_df['formatted'].tolist()
	ts_ngrams = trivially_shared_ngrams(methods)

	# Iterate over all the developers directories
	mean_df = pd.DataFrame()
	for dev_id in range(1, 11):
		dev_dir = os.path.join(args.predictions_dir, f'developer_{dev_id}')
		models_dirs = os.listdir(dev_dir)
		models_dirs = [dir for dir in models_dirs if os.path.isdir(os.path.join(dev_dir, dir)) and dir != 'datasets']
		
		# Compute CrystalBLEU scores for the selected models
		dev_mean_cs, dev_cb_dict = dict(), dict()
		for m_dir in models_dirs:
			dir_path = os.path.join(dev_dir, m_dir)
			test_df = pd.read_csv(os.path.join(dir_path, 'predictions_test.csv'))

			output_cb_dir = os.path.join(results_dir, f'developer_{dev_id}', m_dir)
			if not os.path.exists(output_cb_dir): os.makedirs(output_cb_dir, exist_ok=True)

			cb_df = compute_crystal_bleu(ts_ngrams, test_df)
			cb_df.to_csv(os.path.join(output_cb_dir, 'crystalbleu.csv'), index=False)
			dev_cb_dict[m_dir] = cb_df
			
			cb_df['model'] = model_mapping[m_dir]
			cumulate_df = pd.concat([cumulate_df, cb_df], ignore_index=True)
		

		# Remove rows where we have an exact_match in the same row for both models
		base_cb_df, dev_cb_df, org_cb_df = dev_cb_dict['pt-sft1-ft'], dev_cb_dict['sft1-sft2-dev'], dev_cb_dict['sft1-sft2-all-dev']
		em_base, em_dev, em_org = base_cb_df['exact_match'].tolist(), dev_cb_df['exact_match'].tolist(), org_cb_df['exact_match'].tolist()
		cb_base, cb_dev, cb_org = base_cb_df['crystalbleu'].tolist(), dev_cb_df['crystalbleu'].tolist(), org_cb_df['crystalbleu'].tolist()

		# Baseline vs Developer
		cb_base_clean, cb_dev_clean = list(), list()
		for cb_b, cb_d, em_b, em_d in zip(cb_base, cb_dev, em_base, em_dev):
			if em_b == True and em_d == True:
				continue
			cb_base_clean.append(cb_b)
			cb_dev_clean.append(cb_d)

		assert len(cb_base_clean) == len(cb_dev_clean)
		
		mean_base_clean_dev = round(sum(cb_base_clean) / len(cb_base_clean) * 100, 2)
		mean_dev_clean = round(sum(cb_dev_clean) / len(cb_dev_clean) * 100, 2)
		
		# Baseline vs Organization
		cb_base_clean, cb_org_clean = list(), list()
		for cb_b, cb_o, em_b, em_o in zip(cb_base, cb_org, em_base, em_org):
			if em_b == True and em_o == True:
				continue
			cb_base_clean.append(cb_b)
			cb_org_clean.append(cb_o)

		assert len(cb_base_clean) == len(cb_org_clean)

		mean_base_clean_org = round(sum(cb_base_clean) / len(cb_base_clean) * 100, 2)
		mean_org_clean = round(sum(cb_org_clean) / len(cb_org_clean) * 100, 2)
		
		df_dict = {
				'author_id': dev_id,
				'pt-sft1-vs-dev': mean_base_clean_dev,
				'sft1-sft2-dev': mean_dev_clean,
				'pt-sft1-vs-all-dev': mean_base_clean_org,
				'sft1-sft2-all-dev': mean_org_clean
				}
		
		dev_mean_df = pd.DataFrame(df_dict, index=[0])
		mean_df = pd.concat([mean_df, dev_mean_df], ignore_index=True)

	mean_df['author_id'] = mean_df['author_id'].astype(int)
	mean_df = mean_df.sort_values(by=['author_id'], ascending=True)
	mean_df.to_csv(os.path.join(results_dir, 'crystalbleu_top_10_codellama_em.csv'), index=False)

	# Print some statistics
	df_info = cumulate_df.groupby('model')['crystalbleu'].describe()
	print("CrystalBLEU scores analysis")
	print(df_info)
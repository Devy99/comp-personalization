import numpy as np
import os, argparse
import pandas as pd
from natsort import natsorted

def _is_improvement(monitor_value, reference_value, delta):
  '''
  Arg 1: monitor_value the accuracy we are checking
  Arg 2: reference_value the accuracy we are checking against
  arg 3: delta the min difference between the two accuracies to be improved
  '''
  delta = abs(delta)
  return np.greater(monitor_value - delta, reference_value)


def get_best_checkpoint(checkpoints, accuracies):
  baseline = 0
  best_acc = 0
  best_check = "no checkpoint"
  delta = 1
  patience = 5
  wait = 0
  for current_check, current_acc in zip(checkpoints, accuracies):
    if wait == patience:
      print('stopped')
      break
    wait += 1
    if _is_improvement(current_acc, best_acc, delta):
      best_acc = current_acc
      best_check = current_check
      if _is_improvement(current_acc, baseline, delta):
        wait = 0
    baseline = current_acc
  print(f'Best checkpoint is {best_check} with accuracy {best_acc}')
  
  # Get the checkpoint basedir
  best_check = os.path.dirname(best_check)
  return best_check


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='optional arguments')
    parser.add_argument('--predictions_filename', '-n',
                        metavar='FILENAME',
                        dest='predictions_filename',
                        required=False,
                        type=str,
                        default='predictions.txt',
                        help='Name of the predictions file')
    parser.add_argument('--checkpoint_filename', '-l',
                        metavar='FILENAME',
                        dest='checkpoint_filename',
                        required=False,
                        type=str,
                        default=None,
                        help='Name of the file where to store the path of the best checkpoint')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--checkpoints_dir', '-d',
                        metavar='PATH',
                        dest='ckps_dir',
                        required=True,
                        type=str,
                        help='Path of the folder containing all the checkpoints')
     
    return parser

def find_files(directory, filename):
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == filename:
                paths.append(os.path.join(root, file))
    
    return paths

if __name__ == '__main__':   
    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()
    
    # Retrieve predictions.txt files from all checkpoints
    checkpoint_paths = find_files(args.ckps_dir, args.predictions_filename)
    checkpoint_paths = natsorted(checkpoint_paths)

    # Calculate accuracies
    accuracies = []
    for checkpoint in checkpoint_paths:
        print(f'Calculating accuracy for: {checkpoint}')
        df = pd.read_json(checkpoint, orient='records')
        accuracy = (df['exact_match'].sum() / len(df)) * 100
        print(f'Accuracy: {accuracy:.2f}%')
        accuracies.append(accuracy)
        print()

    best_checkpoint = get_best_checkpoint(checkpoint_paths, accuracies)
    
    # Save best checkpoint path to file
    if args.checkpoint_filename is not None:
        with open(args.checkpoint_filename, 'w') as fp:
            fp.write(best_checkpoint)
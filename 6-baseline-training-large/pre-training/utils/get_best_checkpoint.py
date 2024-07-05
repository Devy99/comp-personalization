import numpy as np
import os, json, argparse
from natsort import natsorted

def _is_improvement(monitor_value, reference_value, delta):
  '''
  Arg 1: monitor_value the accuracy we are checking
  Arg 2: reference_value the accuracy we are checking against
  arg 3: delta the min difference between the two accuracies to be improved
  '''
  delta = abs(delta)
  return np.greater(monitor_value - delta, reference_value)


def get_best_checkpoint(checkpoints, accuracies, delta):
  baseline = 0
  best_acc = 0
  best_check = 0
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
  return best_check

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--metrics_filename', '-n',
                        metavar='FILENAME',
                        dest='metrics_filename',
                        required=False,
                        type=str,
                        default='predictions.txt',
                        help='Name of the predictions file')
    parser.add_argument('--delta_value', '-v',
                        metavar='VALUE',
                        dest='delta_value',
                        required=False,
                        type=float,
                        default=0.01,
                        help='The delta value to check during the early stopping process. Default: 0.01')

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
    
    # Retrieve metric file from all checkpoints
    checkpoint_paths = find_files(args.ckps_dir, args.metrics_filename)
    checkpoint_paths = natsorted(checkpoint_paths)

    # Calculate accuracies
    accuracies = []
    for checkpoint in checkpoint_paths:
        with open(checkpoint, 'r') as f:
            metrics_dict = json.load(f)
        accuracy = metrics_dict['eval_accuracy']

        print(f'The accuracy for checkpoint {checkpoint} is: {str(accuracy)}')
        accuracies.append(accuracy)
        print()

    best_checkpoint = get_best_checkpoint(checkpoint_paths, accuracies, args.delta_value)

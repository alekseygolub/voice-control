from bin.config import Net
from tools.dataloader import DataLoader
from tools.helpers import load_model_txt
from bin.training import train
import argparse

from bin.detection import run

aparser = argparse.ArgumentParser()
subparsers = aparser.add_subparsers(dest='type')

learning_parser = subparsers.add_parser('training', help='Used to run learning process')
detection_parser = subparsers.add_parser("detection", help="Used to run detection process")

detection_parser.add_argument('--oauth', required=True, help='OAuth token')
detection_parser.add_argument('--folder_id', required=True, help='folder ID')

learning_parser.add_argument('--epoch_count', required=False, type=int, default=20)
learning_parser.add_argument('--model_path', required=False, type=str, default='')

args = aparser.parse_args()

if args.type == 'training':
    model = Net()
    if args.model_path:
        try:
            model = load_model_txt(model, args.model_path)
        except:
            print('Cannot load model from', args.model_path)
            exit(0)
    dataloader = DataLoader()
    train(model, dataloader, args.epoch_count)
elif args.type == 'detection':
    run(args.oauth, args.folder_id)


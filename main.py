import bright_utils
from new_train import train
import config
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, default='PE',
                    choices=['PE', 'F2T', 'Cora', 'Douban'],
                    help='datasets: PE; ACM-DBLP; cora; foursquare-twitter; phone-email; Douban; flickr-lastfm')


"""
get the rwr embedding 
"""
bright_utils.preprocess(0.2, True, False)
"""
train the model
"""
train(0.2, 250, 0.0001, 128, 500, 10, True, False)

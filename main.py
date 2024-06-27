from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from vietocr.predict import Predictor

from collections import defaultdict
import torch

base_cfg = Cfg.load_config_from_file('./config/base.yml')
vgg_seq_cfg = Cfg.load_config_from_file('./config/vgg-seq2seq.yml')

config = defaultdict()

config.update(base_cfg)
config.update(vgg_seq_cfg)

config['pretrain'] = './vgg_seq2seq.pth'
config['weights'] = './vgg_seq2seq.pth'
config['device'] = "cpu"

detector = Predictor(config)

for layer, weight in detector.model.named_parameters():
    print(layer,  "\t\t",  weight.shape)
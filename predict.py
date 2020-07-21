import torch
import random
from utils.logging import logger, init_logger
from models.pytorch_pretrained_bert.modeling import BertConfig
from models import data_loader, model_builder
from models.trainer import build_trainer


class Running(object):
    """Run Model"""

    def __init__(self, args, device_id):
        """
        :param args: parser.parse_args()
        :param device_id: 0 or -1
        """
        self.args = args
        self.device_id = device_id
        self.model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval',
                            'rnn_size']

        self.device = "cpu" if self.args.visible_gpus == '-1' else "cuda"
        logger.info('Device ID %d' % self.device_id)
        logger.info('Device %s' % self.device)
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)

        if self.device_id >= 0:
            torch.cuda.set_device(self.device_id)

        init_logger(args.log_file)

        try:
            step = int(self.args.test_from.split('.')[-2].split('_')[-1])
        except IndexError:
            step = 0

        logger.info('Loading checkpoint from %s' % self.args.test_from)
        checkpoint = torch.load(self.args.test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in self.model_flags:
                setattr(self.args, k, opt[k])

        config = BertConfig.from_json_file(self.args.bert_config_path)
        self.model = model_builder.Summarizer(self.args, self.device, load_pretrained_bert=False, bert_config=config)
        self.model.load_cp(checkpoint)
        self.model.eval()

    def predict(self):

        test_iter = data_loader.DataLoader(self.args, data_loader.load_dataset(self.args, 'test', shuffle=False),
                                           self.args.batch_size, self.device, shuffle=False, is_test=True)
        trainer = build_trainer(self.args, self.device_id, self.model, None)
        trainer.test(test_iter, step)

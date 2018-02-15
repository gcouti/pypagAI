import os
import sys
import argparse
import importlib


# def str2bool(value):
#     v = value.lower()
#     if v in ('yes', 'true', 't', '1', 'y'):
#         return True
#     elif v in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
#
#
# def str2class(value):
#     """From import path string, returns the class specified. For example, the
#     string 'parlai.agents.drqa.drqa:SimpleDictionaryAgent' returns
#     <class 'parlai.agents.drqa.drqa.SimpleDictionaryAgent'>.
#     """
#     if ':' not in value:
#         raise RuntimeError('Use a colon before the name of the class.')
#     name = value.split(':')
#     module = importlib.import_module(name[0])
#     return getattr(module, name[1])
#
#
# def class2str(value):
#     """Inverse of params.str2class()."""
#     s = str(value)
#     s = s[s.find('\'') + 1:s.rfind('\'')]  # pull out import path
#     s = ':'.join(s.rsplit('.', 1))  # replace last period with ':'
#     return s


class PypagaiParser(argparse.ArgumentParser):

    def __init__(self, add_parlai_args=True, add_model_args=False, model_argv=None):

        super().__init__(description='Pypagai parser.')

        # self.register('type', 'bool', str2bool)
        # self.register('type', 'class', str2class)

        self.parlai_home = (os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        os.environ['PYPAGAI_HOME'] = self.parlai_home

        self.add_arg = self.add_argument

        # if add_parlai_args:
        #     self.add_parlai_args(model_argv)
        #     self.add_image_args()
        #
        # if add_model_args:
        #     self.add_model_args(model_argv)

    # def add_parlai_data_path(self, argument_group=None):
    #     if argument_group is None:
    #         argument_group = self
    #
    #     default_data_path = os.path.join(self.parlai_home, 'data')
    #
    #     help_message='path to datasets, defaults to {parlai_dir}/data'
    #     argument_group.add_argument('-dp', '--datapath', default=default_data_path, help=help_message)

    # def add_parlai_args(self, args=None):
    #     default_downloads_path = os.path.join(self.parlai_home, 'downloads')
    #     parlai = self.add_argument_group('Main ParlAI Arguments')
    #     parlai.add_argument(
    #         '-t', '--task',
    #         help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"')
    #     parlai.add_argument(
    #         '--download-path', default=default_downloads_path,
    #         help='path for non-data dependencies to store any needed files.'
    #              'defaults to {parlai_dir}/downloads')
    #     parlai.add_argument(
    #         '-dt', '--datatype', default='train',
    #         choices=['train', 'train:stream', 'train:ordered',
    #                  'train:ordered:stream', 'train:stream:ordered',
    #                  'valid', 'valid:stream', 'test', 'test:stream'],
    #         help='choose from: train, train:ordered, valid, test. to stream '
    #              'data add ":stream" to any option (e.g., train:stream). '
    #              'by default: train is random with replacement, '
    #              'valid is ordered, test is ordered.')
    #     parlai.add_argument(
    #         '-im', '--image-mode', default='raw', type=str,
    #         help='image preprocessor to use. default is "raw". set to "none" '
    #              'to skip image loading.')
    #     parlai.add_argument(
    #         '-nt', '--numthreads', default=1, type=int,
    #         help='number of threads. If batchsize set to 1, used for hogwild; '
    #              'otherwise, used for number of threads in threadpool loading,'
    #              ' e.g. in vqa')
    #     batch = self.add_argument_group('Batching Arguments')
    #     batch.add_argument(
    #         '-bs', '--batchsize', default=1, type=int,
    #         help='batch size for minibatch training schemes')
    #     batch.add_argument('-bsrt', '--batch-sort', default=True, type='bool',
    #                        help='If enabled (default True), create batches by '
    #                             'flattening all episodes to have exactly one '
    #                             'utterance exchange and then sorting all the '
    #                             'examples according to their length. This '
    #                             'dramatically reduces the amount of padding '
    #                             'present after examples have been parsed, '
    #                             'speeding up training.')
    #     batch.add_argument('-clen', '--context-length', default=-1, type=int,
    #                        help='Number of past utterances to remember when '
    #                             'building flattened batches of data in multi-'
    #                             'example episodes.')
    #     batch.add_argument('-incl', '--include-labels',
    #                        default=True, type='bool',
    #                        help='Specifies whether or not to include labels '
    #                             'as past utterances when building flattened '
    #                             'batches of data in multi-example episodes.')
    #     self.add_parlai_data_path(parlai)
    #     self.add_task_args(args)
    #
    # def add_task_args(self, args):
    #     # Find which task specified, and add its specific arguments.
    #     args = sys.argv if args is None else args
    #     task = None
    #     for index, item in enumerate(args):
    #         if item == '-t' or item == '--task':
    #             task = args[index + 1]
    #     if task:
    #         for t in ids_to_tasks(task).split(','):
    #             agent = get_task_module(t)
    #             if hasattr(agent, 'add_cmdline_args'):
    #                 agent.add_cmdline_args(self)
    #
    # def add_model_args(self, args=None):
    #     model_args = self.add_argument_group('ParlAI Model Arguments')
    #     model_args.add_argument(
    #         '-m', '--model', default=None,
    #         help='the model class name, should match parlai/agents/<model>')
    #     model_args.add_argument(
    #         '-mf', '--model-file', default=None,
    #         help='model file name for loading and saving models')
    #     model_args.add_argument(
    #         '--dict-class',
    #         help='the class of the dictionary agent uses')
    #     # Find which model specified, and add its specific arguments.
    #     if args is None:
    #         args = sys.argv
    #     model = None
    #     for index, item in enumerate(args):
    #         if item == '-m' or item == '--model':
    #             model = args[index + 1]
    #     if model:
    #         agent = get_agent_module(model)
    #         if hasattr(agent, 'add_cmdline_args'):
    #             agent.add_cmdline_args(self)
    #         if hasattr(agent, 'dictionary_class'):
    #             s = class2str(agent.dictionary_class())
    #             model_args.set_defaults(dict_class=s)
    #
    # def add_image_args(self, args=None):
    #     # Find which image mode specified, add its specific arguments if needed.
    #     args = sys.argv if args is None else args
    #     image_mode = None
    #     for index, item in enumerate(args):
    #         if item == '-im' or item == '--image-mode':
    #             image_mode = args[index + 1]
    #     if image_mode and image_mode != 'none':
    #         parlai = \
    #             self.add_argument_group('ParlAI Image Preprocessing Arguments')
    #         parlai.add_argument('--image-size', type=int, default=256,
    #                             help='resizing dimension for images')
    #         parlai.add_argument('--image-cropsize', type=int, default=224,
    #                             help='crop dimension for images')

    # def parse_args(self, args=None, namespace=None, print_args=True):
    #     """
    #     Parses the provided arguments and returns a dictionary of the ``args``.
    #     We specifically remove items with ``None`` as values in order to support the style
    #     ``opt.get(key, default)``, which would otherwise
    #
    #     return ``None``.
    #     """
    #
    #     self.args = super().parse_args(args=args)
    #     self.opt = vars(self.args)
    #
    #     # custom post-parsing
    #     self.opt['parlai_home'] = self.parlai_home
    #     if 'batchsize' in self.opt and self.opt['batchsize'] <= 1:
    #         # hide batch options
    #         self.opt.pop('batch_sort', None)
    #         self.opt.pop('context_length', None)
    #         self.opt.pop('include_labels', None)
    #
    #     # set environment variables
    #     if self.opt.get('download_path'):
    #         os.environ['PARLAI_DOWNPATH'] = self.opt['download_path']
    #     if self.opt.get('datapath'):
    #         os.environ['PARLAI_DATAPATH'] = self.opt['datapath']
    #
    #     if print_args:
    #         self.print_args()
    #
    #     return self.opt
    #
    # def print_args(self):
    #     """Print out all the arguments in this parser."""
    #     if not self.opt:
    #         self.parse_args(print_args=False)
    #     values = {}
    #     for key, value in self.opt.items():
    #         values[str(key)] = str(value)
    #     for group in self._action_groups:
    #         group_dict = {
    #             a.dest: getattr(self.args, a.dest, None)
    #             for a in group._group_actions
    #         }
    #         namespace = argparse.Namespace(**group_dict)
    #         count = 0
    #         for key in namespace.__dict__:
    #             if key in values:
    #                 if count == 0:
    #                     print('[ ' + group.title + ': ] ')
    #                 count += 1
    #                 print('[  ' + key + ': ' + values[key] + ' ]')
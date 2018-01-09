"""
Execute experiments with ParlAI framework

This main has the purpose of train and experiment models. It implement the same interface of the train_model.py
file that was implemented in the ParlAI framework.

You can run this file like:

```
    python source/main.py -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
```

To evaluate model run with -vtim or --validation-every-n-secs. It will eval model every X seconds (defau)
To save model in the end of the execution run with --save or -s

To see other options -h

"""
from copy import deepcopy, copy
from datetime import datetime

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
import build_dict
import math

from experiments.evaluation import run_eval

TEMPORARY_MODEL_PATH = '/tmp/tmp_model_%i' % datetime.now().timestamp()
TEMPORARY_RESULT_PATH = '/tmp/tmp_model_result_%i.csv' % datetime.now().timestamp()


def setup_args():
    parser = ParlaiParser(True, True)
    main_args = parser.add_argument_group('Main loop arguments')

    # help_message = 'task to use for valid/test (defaults to the one used for training if not set)'
    # main_args.add_argument('-et', '--evaltask', help=help_message)

    help_message = 'max examples to use during validation (default -1 uses all)'
    main_args.add_argument('-vme', '--validation-max-exs', type=int, default=-1, help=help_message)

    help_message = 'number of iterations of validation where result does not improve before we stop training'
    main_args.add_argument('-vp', '--validation-patience', type=int, default=0, help=help_message)

    help_message = 'key into report table for selecting best validation'
    main_args.add_argument('-vmt', '--validation-metric', default='accuracy', help=help_message)

    help_message = 'value at which training will stop if exceeded by training metric'
    main_args.add_argument('-vcut', '--validation-cutoff', type=float, default=1.0, help=help_message)

    help_message = 'build dictionary first before training agent'
    main_args.add_argument('-dbf', '--dict-build-first', type='bool', default=True, help=help_message)

    main_args.add_argument('-ltim', '--log-every-n-secs', type=float, default=-1)

    main_args.add_argument('-d', '--display-examples', type='bool', default=False)
    main_args.add_argument('-e', '--num-epochs', type=float, default=-1)
    main_args.add_argument('-ttim', '--max-train-time', type=float, default=-1)
    main_args.add_argument('-vtim', '--validation-every-n-secs', type=float, default=10)

    return parser


class ExperimentFlow:

    def __init__(self, parser):
        opt = parser.parse_args()

        # Possibly build a dictionary (not all models do this).
        if opt['dict_build_first'] and 'dict_file' in opt:
            if opt['dict_file'] is None and opt.get('model_file'):
                opt['dict_file'] = opt['model_file'] + '.dict'

            print("\n[ building dictionary first... ]")
            build_dict.build_dict(opt)

        # Create model and assign it to the specified task
        self.agent = create_agent(opt)
        self.world = create_task(opt, self.agent)

        # Start timers
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()

        print('\n[ training... ]')
        self.parleys = 0
        self.total_exs = 0
        self.best_valid = 0
        self.impatience = 0
        self.total_epochs = 0
        self.total_episodes = 0

        self.saved = False
        self.valid_world = None

        if opt['num_epochs'] > 1:
            self.max_exs = opt['num_epochs'] * len(self.world)
        else:
            self.max_exs = 1000

        self.opt = opt
        self.max_parleys = math.ceil(self.max_exs / opt['batchsize'])

    def validate(self):
        """
        Validate results

        :return:
        """

        opt = self.opt
        valid_report, valid_world = run_eval(self.agent, opt, 'valid', opt['validation_max_exs'], valid_world=self.valid_world)

        if valid_report[opt['validation_metric']] > self.best_valid:

            self.impatience = 0
            self.best_valid = valid_report[opt['validation_metric']]
            print('[ new best {}: {} ]'.format(opt['validation_metric'], self.best_valid))

            self.saved = True
            self.agent.save(path=TEMPORARY_MODEL_PATH)

            if opt['validation_metric'] == 'accuracy' and self.best_valid > opt['validation_cutoff']:
                print('[ task solved! stopping. ]')
                self.log()
                return True
        else:
            self.impatience += 1
            print('[ did not beat best {}: {} impatience: {} ]'.format(opt['validation_metric'], round(self.best_valid, 4), self.impatience))

        self.validate_time.reset()

        if 0 <= opt['validation_patience'] <= self.impatience:
            print('[ ran out of patience! stopping training. ]')
            self.log()
            return True

        self.log()
        return False

    def log(self, report=None, type='train'):
        """
        Log execution messages and print files to investigate results
        """

        opt = self.opt

        if opt['display_examples']:
            print(self.world.display() + '\n~~')

        logs = ['time:{}s'.format(math.floor(self.train_time.time())), 'parleys:{}'.format(self.parleys)]

        # time elapsed
        # get report and update total examples seen so far
        if report:
            train_report = report
        else:
            if hasattr(self.agent, 'report'):
                train_report = self.agent.report()
                self.agent.reset_metrics()
            else:
                train_report = self.world.report()
                self.world.reset_metrics()

        if hasattr(train_report, 'get') and train_report.get('total'):
            self.total_exs += train_report['total']
            logs.append('total_exs:{}'.format(self.total_exs))

        # check if we should log amount of time remaining
        time_left = None
        if opt['num_epochs'] > 0 and self.total_exs > 0 and self.max_exs > 0:
            exs_per_sec = self.train_time.time() / self.total_exs
            time_left = (self.max_exs - self.total_exs) * exs_per_sec

        if opt['max_train_time'] > 0:
            other_time_left = opt['max_train_time'] - self.train_time.time()

            if time_left is not None:
                time_left = min(time_left, other_time_left)
            else:
                time_left = other_time_left

        if time_left is not None:
            logs.append('time_left:{}s'.format(math.floor(time_left)))

        if opt['num_epochs'] > 0:
            if self.total_exs > 0 and len(self.world) > 0:
                display_epochs = int(self.total_exs / len(self.world))
            else:
                display_epochs = self.total_epochs
                logs.append('num_epochs:{}'.format(display_epochs))

        # join log string and add full metrics report to end of log
        log = '[ {} ] {}'.format(' '.join(logs), train_report)
        print(log)
        with open(TEMPORARY_RESULT_PATH, 'a') as file:
            train_report['data'] = logs
            train_report['params'] = {
                'type': type,
                'model': opt['model'],
                'num_epochs': opt['num_epochs'] if 'num_epochs' in opt else -1,
                'keras_epochs': opt['keras_epochs'] if 'keras_epochs' in opt else -1,
                'text_max_size': opt['text_max_size'] if 'text_max_size' in opt else -1,
                'input_without_question': opt['input_without_question'] if 'input_without_question' in opt else False,
                'input_aggregate_history': opt['input_aggregate_history'] if 'input_aggregate_history' in opt else False,
            }
            file.write(str(train_report)+"\n")

        self.log_time.reset()

    def train(self):
        """
        Train model util number of iterations or when it accomplish the task

        TODO: Fazer um python notebook com gráficos
        TODO: Colocar para rodar os modelos baseline

        """
        opt = self.opt
        world = self.world

        with world:

            while True:
                world.parley()
                self.parleys += 1

                if world.epoch_done():
                    self.total_epochs += 1

                if opt['num_epochs'] > 0 and ((0 < self.max_parleys <= self.parleys) or self.total_epochs >= opt['num_epochs']):
                    print('[ num_epochs completed:{} time elapsed:{}s ]'.format(opt['num_epochs'], self.train_time.time()))
                    break

                if 0 < opt['max_train_time'] < self.train_time.time():
                    print('[ max_train_time elapsed:{}s ]'.format(self.train_time.time()))
                    break

                if 0 < opt['log_every_n_secs'] < self.log_time.time():
                    self.log()

                if 0 < opt['validation_every_n_secs'] < self.validate_time.time():
                    stop_training = self.validate()

                    if stop_training:
                        break

        # Reload best model
        print("\n[load best model ]")
        self.agent.load(TEMPORARY_MODEL_PATH)

        print("\n[final results ]")
        report, _ = run_eval(self.agent, opt, 'valid', write_log=True)
        self.log(report, 'valid')
        report, _ = run_eval(self.agent, opt, 'test', write_log=True)
        self.log(report, 'test')


if __name__ == '__main__':
    args = setup_args()
    ExperimentFlow(args).train()

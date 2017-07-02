from __future__ import print_function

import csv
import os

# Comment this if tensor flow crash
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import sys
import time

import copy
import parlai

from parlai.agents.ir_baseline.ir_baseline import IrBaselineAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.agents.remote_agent.remote_agent import ParsedRemoteAgent
from my_agents import N2NMemAgent, DummyAgent, RNAgent


"""
    This is the main file 
     
    Here you can run our experiments. To run just type 
    
    python main.py  
 
    This script use ParlAI Framework to do the experimentation process
          
          
    References:
    
        ParlAI Framework
            - https://github.com/facebookresearch/ParlAI#requirements

        LUA Implementation n2n memory
            - https://github.com/facebook/MemNN
                    
        Kera FB-LSTM | LSTM | GRU  | IRNN | RNN
            - https://gist.github.com/Smerity/418a4e7f9e719ff02bf3
            - https://github.com/Smerity/keras_qa/blob/master/qa.py
            - https://gist.github.com/SNagappan/a7be6ce6e75c36c7406e
            
        Python N2N Mem
            - https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py
            - https://github.com/vinhkhuc/MemN2N-babi-python
            - https://github.com/domluna/memn2n/blob/master/memn2n/memn2n.py

        Seq2Seq Network
            - pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl
            - pip install torchvision

        DNC
            - https://github.com/FrownyFace/DNC/blob/master/dnc.py
            - https://github.com/brendanator/differentiable-neural-computer
            - https://github.com/Mostafa-Samir/DNC-tensorflow/tree/master/dnc
            - https://github.com/bgavran/DNC
            - https://github.com/deepmind/dnc/blob/master/train.py
        RN
            - https://github.com/kimhc6028/relational-networks/blob/master/model.py
            - https://index.pocketcluster.io/shaohua0116-relation-network-tensorflow.html
"""


def create_dictionary(argparser):
    print('Setting up dictionary.')
    # set up dictionary
    DictionaryAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    dictionary = DictionaryAgent(opt)

    if not opt.get('dict_loadpath'):
        # build dictionary since we didn't load it
        ordered_opt = copy.deepcopy(opt)
        for datatype in ['train:ordered', 'valid']:
            # we use train and valid sets to build dictionary
            ordered_opt['datatype'] = datatype
            ordered_opt['numthreads'] = 1
            world_dict = create_task(ordered_opt, dictionary)

            print('Dictionary building on {} data.'.format(datatype))
            cnt = 0
            # pass examples to dictionary
            for _ in world_dict:
                cnt += 1
                if cnt > opt['dict_max_exs'] and opt['dict_max_exs'] > 0:
                    print('Processed {} exs, moving on.'.format(
                        opt['dict_max_exs']))
                    # don't wait too long...
                    break

                world_dict.parley()

        # we need to save the dictionary to load it in memnn (sort it by freq)
        dictionary.save(opt['dict_file'], sort=True)

        print('Dictionary ready, moving on to training.')

        return dictionary, opt


if __name__ == "__main__":

    print("Start experiments")
    # Get command line arguments
    argparser = ParlaiParser()
    argparser.add_argument('--model', default='baseline', type=str)
    argparser.add_argument('--num-examples', default=1000, type=int)
    argparser.add_argument('--dict-max-exs', default=1000, type=int)
    argparser.add_argument('--num-its', default=100, type=int)

    dictionary, opt = create_dictionary(argparser)

    if opt['model'] == 'baseline':
        print('Baseline Model')
        IrBaselineAgent.add_cmdline_args(argparser)

        opt = argparser.parse_args()
        agent = IrBaselineAgent(opt)

    elif opt['model'] == 'memnn':
        print('MemoryNN Model')
        ParsedRemoteAgent.add_cmdline_args(argparser)

        parlai_home = os.environ['PARLAI_HOME']
        if '--remote-cmd' not in sys.argv:
            if os.system('which luajit') != 0:
                raise RuntimeError('Could not detect torch luajit installed: ' +
                                   'please install torch from http://torch.ch ' +
                                   'or manually set --remote-cmd for this example.')
            sys.argv.append('--remote-cmd')
            sys.argv.append('luajit {}/parlai/agents/'.format(parlai_home) +
                            'memnn_luatorch_cpu/memnn_zmq_parsed.lua')

        if '--remote-args' not in sys.argv:
            sys.argv.append('--remote-args')
            sys.argv.append('{}/examples/'.format(parlai_home) +
                            'memnn_luatorch_cpu/params_default.lua')

        agent = ParsedRemoteAgent(opt, {'dictionary': dictionary})

    elif opt['model'] == 'n2nmem':

        print('N2N Mem Model')
        N2NMemAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = N2NMemAgent(opt)

    elif opt['model'] == 'rn':

        print('rn Mem Model')
        RNAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = RNAgent(opt)

    elif opt['model'] == 'dummy':
        print('Dummy Model')
        DummyAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = DummyAgent(opt)
    else:
        print('Baseline Model')
        IrBaselineAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = IrBaselineAgent(opt)

    opt['datatype'] = 'train'
    world_train = create_task(opt, agent)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    start = time.time()
    results = []

    model_name = opt['model']
    task_name = opt['task']

    with world_train:

        iteration = 0
        for _ in range(opt['num_its']):
            print('[ training ] %i' % iteration)
            for _ in range(opt['num_examples'] * opt.get('numthreads', 1)):
                world_train.parley()

            world_train.synchronize()

            world_valid.reset()
            for _ in world_valid:  # check valid accuracy
                world_valid.parley()

            report_valid = world_valid.report()
            report_valid['TASK'] = task_name
            report_valid['ITER'] = iteration
            results.append(report_valid)

            print(report_valid)

            iteration += 1

        # show some example dialogs after training:
        world_valid = create_task(opt, agent)
        for _k in range(10):
            world_valid.parley()
            print(world_valid.display())

    print('finished in {} s'.format(round(time.time() - start, 2)))

    print(results[0])

    with open('result_%s_%s_.csv' % (model_name, task_name), 'w') as file:
        writer = csv.writer(file)
        for row in results:
            writer.writerow([model_name, row['TASK'], row['ITER'], row['accuracy'], row['hits@k'][5], row['hits@k'][10]])

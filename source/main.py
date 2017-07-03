from __future__ import print_function

import csv
import os

# Comment this if tensor flow crash
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

from parlai.agents.ir_baseline.ir_baseline import IrBaselineAgent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from my_agents import N2NMemAgent, DummyAgent, RNAgent, EnsembleAgent, EnsembleNetworkAgent


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
if __name__ == "__main__":

    print("Start experiments")
    # Get command line arguments
    argparser = ParlaiParser()
    argparser.add_argument('--model', default='baseline', type=str)
    argparser.add_argument('--num-examples', default=1000, type=int)
    argparser.add_argument('--dict-max-exs', default=1000, type=int)
    argparser.add_argument('--num-its', default=10, type=int)

    opt, a = argparser.parse_known_args()

    if opt.model == 'baseline':
        print('Baseline Model')
        IrBaselineAgent.add_cmdline_args(argparser)

        opt = argparser.parse_args()
        agent = IrBaselineAgent(opt)

    elif opt.model == 'n2nmem':

        print('N2N Mem Model')
        N2NMemAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = N2NMemAgent(opt)

    elif opt.model == 'rn':

        print('rn Mem Model')
        RNAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = RNAgent(opt)

    elif opt.model == 'ensemble':
        print('Ensemble Model')
        EnsembleAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = EnsembleAgent(opt)

    elif opt.model == 'enn':
        print('Ensemble Netwokr Model')
        EnsembleNetworkAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args()
        agent = EnsembleNetworkAgent(opt)

    elif opt.model == 'dummy':
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

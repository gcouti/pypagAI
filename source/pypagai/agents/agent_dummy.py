from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent


class DummyAgent(Agent):
    """
        This agent retrieve only the first candidates.

        This was made to simplify the understanding of the framework
    """
    @staticmethod
    def add_cmdline_args(parser):
        DictionaryAgent.add_cmdline_args(parser)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'DummyAgent'
        self.dictionary = DictionaryAgent(opt)
        self.opt = opt

    def observe(self, obs):
        self.observation = obs
        self.dictionary.observe(obs)
        return obs

    def act(self):
        if self.opt.get('datatype', '').startswith('train'):
            self.dictionary.act()

        obs = self.observation
        reply = {'id': self.getID()}

        # Rank candidates
        if 'label_candidates' in obs and len(obs['label_candidates']) > 0:
            reply['text_candidates'] = list(obs['label_candidates'])
            reply['text'] = np.random.choice(reply['text_candidates'], 1)[0]
            # if 'labels' in obs:
            #     reply['text'] = obs['labels'][0]
        else:
            reply['text'] = "I don't know."
        return reply
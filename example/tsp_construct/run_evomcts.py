import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.evo_mcts import EvoMCTS


def main():
    llm = HttpsApi(host='api.deepseek.com',  # your host endpoint
                   key='sk-4e5c35479bb74472a62ba0895092b8a1',  # your key
                   model='deepseek-chat',  # your llm
                   timeout=120)

    task = TSPEvaluation()

    method = EvoMCTS(llm=llm,
                     profiler=ProfilerBase(log_dir='logs/evo_mcts_tsp', log_style='complex'),
                     evaluation=task,
                     max_sample_nums=100,
                     init_size=4,
                     pop_size=10,
                     debug_mode=True
                     # You can customize operators and their weights here
                     # operators=['pc', 'sc', 'pwc', 'pm'],
                     # operator_weights=[1, 1, 1, 2]
                     )

    method.run()


if __name__ == '__main__':
    main()
import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.eoh import EoH


def main():
    llm = HttpsApi(host='api.deepseek.com',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='sk-4e5c35479bb74472a62ba0895092b8a1',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='deepseek-chat',  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = TSPEvaluation()

    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=20,
                 max_generations=5,
                 pop_size=2,
                 num_samplers=1,
                 num_evaluators=1,
                 debug_mode=True)

    method.run()


if __name__ == '__main__':
    main()

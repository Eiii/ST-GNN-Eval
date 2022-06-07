from . import base
from . import starcraft
from . import weather
from . import traffic


def make_problem(problem_args):
    problem_name = problem_args['problem_type']
    kwargs = {k: v for k, v in problem_args.items() if k != 'problem_type'}
    return base._problem_map[problem_name](**kwargs)

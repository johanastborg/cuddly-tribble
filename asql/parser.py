from lark import Lark, Transformer, v_args
import os

@v_args(inline=True)
class ASQLTransformer(Transformer):
    def query(self, *args):
        sources = []
        transformations = []
        for arg in args:
            if isinstance(arg, dict):
                if arg.get('type') in ('from', 'join'):
                    sources.append(arg)
                else:
                    transformations.append(arg)
        return {
            'sources': sources,
            'transformations': transformations
        }

    def from_source(self, table, alias=None):
        table_name = str(table).strip('"')
        return {'type': 'from', 'table': table_name, 'alias': str(alias) if alias else None}

    def join_source(self, table, alias=None):
        table_name = str(table).strip('"')
        return {'type': 'join', 'table': table_name, 'alias': str(alias) if alias else None}

    def transformation(self, clause):
        return clause

    def range_clause(self, duration):
        return {'type': 'range', 'duration': duration}

    def window_clause(self, duration):
        return {'type': 'window', 'duration': duration}

    def mean_func(self, alias=None):
        return {'type': 'aggregate', 'func': 'mean', 'alias': str(alias).strip('"') if alias else None}

    def var_func(self, alias=None):
        return {'type': 'aggregate', 'func': 'var', 'alias': str(alias).strip('"') if alias else None}

    def stddev_func(self, alias=None):
        return {'type': 'aggregate', 'func': 'stddev', 'alias': str(alias).strip('"') if alias else None}

    def min_func(self, alias=None):
        return {'type': 'aggregate', 'func': 'min', 'alias': str(alias).strip('"') if alias else None}

    def max_func(self, alias=None):
        return {'type': 'aggregate', 'func': 'max', 'alias': str(alias).strip('"') if alias else None}

    def covar_func(self, id1, id2, alias=None):
        return {'type': 'aggregate', 'func': 'covar', 'args': [str(id1), str(id2)], 'alias': str(alias).strip('"') if alias else None}

    def map_assign(self, identifier, expression):
        return {'type': 'map', 'id': str(identifier), 'expression': expression}

    def map_simple(self, identifier):
        return {'type': 'map', 'id': str(identifier), 'expression': None}

    def func_call(self, name, args):
        return {'type': 'func_call', 'name': str(name), 'args': args}

    def arg_list(self, *args):
        return list(args)

    def threshold_clause(self, condition):
        return {'type': 'threshold', 'condition': condition}

    def condition(self, left, op, right):
        return {'left': left, 'op': str(op), 'right': right}

    def array_access(self, identifier, index_list):
        return {'type': 'array_access', 'id': str(identifier), 'indices': index_list}

    def index_list(self, *args):
        return [int(a) for a in args if str(a).isdigit()]

    def emit_clause(self, label):
        return {'type': 'emit', 'label': str(label).strip('"')}

    def duration(self, value, unit):
        return {'value': float(value), 'unit': str(unit)}

    def identifier(self, name):
        return str(name)

    def value(self, val):
        if isinstance(val, dict):
            return val
        return str(val)

def get_parser():
    grammar_path = os.path.join(os.path.dirname(__file__), 'grammar.lark')
    with open(grammar_path, 'r') as f:
        grammar = f.read()
    return Lark(grammar, parser='lalr', transformer=ASQLTransformer())

def parse_query(query_str):
    parser = get_parser()
    return parser.parse(query_str)

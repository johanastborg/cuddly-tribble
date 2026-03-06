import jax.numpy as jnp
import jax
import time

class ASQLEngine:
    def __init__(self, mock_data):
        self.raw_data = mock_data
        self.state = {
            'data': {},        # Loaded series: {name: {'time': [], 'value': []}}
            'variables': {},   # Intermediate results
            'current_time': time.time(),
            'active': True     # Used for THRESHOLD filtering
        }

    def execute(self, plan):
        # 1. Handle Sources
        for source in plan['sources']:
            table_name = source['table']
            alias = source['alias'] or table_name
            if table_name in self.raw_data:
                self.state['data'][alias] = self.raw_data[table_name]
            else:
                raise ValueError(f"Table {table_name} not found in mock data")

        # 2. Handle Transformations
        for step in plan['transformations']:
            if not self.state['active']:
                break
                
            self._apply_transformation(step)

        return self.state

    def _apply_transformation(self, step):
        step_type = step['type']
        
        if step_type == 'range':
            self._handle_range(step)
        elif step_type == 'window':
            self._handle_window(step)
        elif step_type == 'aggregate':
            self._handle_aggregate(step)
        elif step_type == 'threshold':
            self._handle_threshold(step)
        elif step_type == 'map':
            self._handle_map(step)
        elif step_type == 'emit':
            self._handle_emit(step)
        else:
            print(f"Warning: Transformation {step_type} not implemented")

    def _handle_window(self, step):
        duration = step['duration']
        self.state['window_seconds'] = self._to_seconds(duration['value'], duration['unit'])

    def _handle_map(self, step):
        var_name = step['id']
        expr = step['expression']
        
        if expr:
            result = self._evaluate_expression(expr)
            self.state['variables'][var_name] = result
        else:
            # Simple MAP(var) -> sample from active stream
            first_alias = next(iter(self.state['data']))
            series = self.state['data'][first_alias]
            values, times = series['value'], series['time']
            
            if len(times) == 0:
                self.state['variables'][var_name] = jnp.array([])
                return

            window_size = self.state.get('window_seconds')
            if window_size:
                start_time, end_time = float(jnp.min(times)), float(jnp.max(times))
                num_windows = max(1, int((end_time - start_time) / window_size) + 1)
                
                results = []
                for i in range(num_windows):
                    w_start = start_time + i * window_size
                    w_end = w_start + window_size
                    mask = (times >= w_start) & (times < w_end)
                    if jnp.any(mask):
                        # Take the latest value in the window
                        results.append(values[mask][-1])
                self.state['variables'][var_name] = jnp.array(results)
            else:
                # No window, just latest overall
                self.state['variables'][var_name] = values[-1]

    def _handle_range(self, step):
        duration = step['duration']
        seconds = self._to_seconds(duration['value'], duration['unit'])
        
        # Determine "now" from the data range itself
        max_time = 0
        for series in self.state['data'].values():
            if len(series['time']) > 0:
                t_max = float(jnp.max(series['time']))
                max_time = max(max_time, t_max)
        
        cutoff = (max_time if max_time > 0 else self.state['current_time']) - seconds
        
        for alias, series in self.state['data'].items():
            mask = series['time'] >= cutoff
            self.state['data'][alias] = {
                'time': series['time'][mask],
                'value': series['value'][mask]
            }

    def _handle_aggregate(self, step):
        func = step['func']
        alias = step['alias']
        
        if func == 'covar':
            id1, id2 = step['args']
            v1, v2 = self.state['data'][id1]['value'], self.state['data'][id2]['value']
            min_len = min(len(v1), len(v2))
            matrix = jnp.cov(v1[:min_len], v2[:min_len])
            if alias: self.state['variables'][alias] = matrix
            return matrix

        # Unary aggregates
        first_alias = next(iter(self.state['data']))
        series = self.state['data'][first_alias]
        values, times = series['value'], series['time']
        
        if len(times) == 0:
            res = jnp.array([])
        elif self.state.get('window_seconds'):
            window_size = self.state['window_seconds']
            start_time, end_time = float(jnp.min(times)), float(jnp.max(times))
            
            # Simple fixed-size windowing
            num_windows = max(1, int((end_time - start_time) / window_size) + 1)
            results = []
            for i in range(num_windows):
                w_start = start_time + i * window_size
                w_end = w_start + window_size
                mask = (times >= w_start) & (times < w_end)
                if jnp.any(mask):
                    win_vals = values[mask]
                    if func == 'mean': results.append(jnp.mean(win_vals))
                    elif func == 'var': results.append(jnp.var(win_vals))
                    elif func == 'stddev': results.append(jnp.std(win_vals))
                    elif func == 'min': results.append(jnp.min(win_vals))
                    elif func == 'max': results.append(jnp.max(win_vals))
            res = jnp.array(results)
        else:
            if func == 'mean': res = jnp.mean(values)
            elif func == 'var': res = jnp.var(values)
            elif func == 'stddev': res = jnp.std(values)
            elif func == 'min': res = jnp.min(values)
            elif func == 'max': res = jnp.max(values)

        if alias: self.state['variables'][alias] = res
        return res

    def _handle_threshold(self, step):
        condition = step['condition']
        left_val = self._evaluate_expression(condition['left'])
        # Simplified threshold: if left_val is a series, use mean? 
        # Or if it's a matrix element (like sync_matrix[0,1]), it's a scalar.
        right_val = float(condition['right']) 
        op = condition['op']
        
        if op == '<':
            result = left_val < right_val
        elif op == '>':
            result = left_val > right_val
        elif op == '==':
            result = left_val == right_val
        elif op == '<=':
            result = left_val <= right_val
        elif op == '>=':
            result = left_val >= right_val
        else:
            result = False
            
        # If scalar comparison, set active. If array, this might need logical_all/any.
        if hasattr(result, 'all'):
             self.state['active'] = bool(result.all())
        else:
             self.state['active'] = bool(result)

    def _handle_emit(self, step):
        if self.state['active']:
            print(f">>> EMIT: {step['label']}")
            for var, val in self.state['variables'].items():
                 print(f"[{var}]:\n{val}\n")

    def _evaluate_expression(self, expr):
        if not isinstance(expr, dict):
            # Literal or identifier
            if str(expr).replace('.','',1).isdigit():
                return float(expr)
            # Check if it's "time" (reserved word for current series time)
            if str(expr) == "time":
                # Use the first available series time
                if self.state['data']:
                    first_alias = next(iter(self.state['data']))
                    return self.state['data'][first_alias]['time']
            # Check variables
            if expr in self.state['variables']:
                return self.state['variables'][expr]
            # Check data aliases (return value series)
            if expr in self.state['data']:
                return self.state['data'][expr]['value']
            return expr

        expr_type = expr.get('type')
        if expr_type == 'array_access':
            var_name = expr['id']
            indices = expr['indices']
            matrix = self._evaluate_expression(var_name)
            if matrix is not None:
                val = matrix
                for idx in indices:
                    val = val[idx]
                return val
        elif expr_type == 'func_call':
            func_name = expr['name']
            args = [self._evaluate_expression(a) for a in expr['args']]
            if func_name == 'sin':
                return jnp.sin(args[0])
            elif func_name == 'cos':
                return jnp.cos(args[0])
            
        return expr

    def _to_seconds(self, value, unit):
        units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        return value * units.get(unit, 1)

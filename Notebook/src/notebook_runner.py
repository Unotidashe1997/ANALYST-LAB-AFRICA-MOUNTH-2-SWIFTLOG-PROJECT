"""Minimal notebook executor: runs '# %%' cells in one namespace, captures stdout, last-expression
values and matplotlib figures, and writes a Jupyter-format .ipynb with real outputs."""
import sys, io, ast, json, base64, contextlib, traceback, os, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def parse(src):
    cells, cur = [], None
    for line in src.splitlines():
        if line.startswith('# %%'):
            cur = {'type': 'markdown' if '[md]' in line else 'code', 'lines': []}; cells.append(cur)
        elif cur is not None:
            cur['lines'].append(line)
    for c in cells:
        while c['lines'] and not c['lines'][-1].strip(): c['lines'].pop()
        if c['type'] == 'markdown':
            c['lines'] = [l[2:] if l.startswith('# ') else l.lstrip('#') for l in c['lines']]
    return cells

def src_list(lines):
    return [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines else [])

def main(src_files, out_ipynb, cwd, results_json):
    os.chdir(cwd)
    src = '\n'.join(open(f).read() for f in src_files)
    cells = parse(src); g = {'__name__': '__main__'}; nb_cells = []; n = 0
    warnings.filterwarnings('ignore')
    for i, c in enumerate(cells):
        if c['type'] == 'markdown':
            nb_cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': src_list(c['lines'])}); continue
        code = '\n'.join(c['lines']); n += 1; outputs = []
        buf = io.StringIO(); val = None
        try:
            tree = ast.parse(code); last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr): last = tree.body.pop()
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f'<cell {i}>', 'exec'), g)
                if last is not None: val = eval(compile(ast.Expression(last.value), f'<cell {i}>', 'eval'), g)
        except Exception:
            print(f'FAILED in cell {i} (code cell {n}):\n{code[:600]}\n'); traceback.print_exc(); print(buf.getvalue()[-1500:]); sys.exit(1)
        if buf.getvalue(): outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': src_list(buf.getvalue().splitlines())})
        for num in plt.get_fignums():
            b = io.BytesIO(); plt.figure(num).savefig(b, format='png', dpi=110, bbox_inches='tight', facecolor='white')
            outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': base64.b64encode(b.getvalue()).decode(), 'text/plain': ['<Figure>']}})
        plt.close('all')
        if val is not None:
            data = {}
            if isinstance(val, (pd.DataFrame, pd.Series)):
                with pd.option_context('display.max_columns', 40, 'display.width', 200, 'display.max_rows', 60):
                    data['text/plain'] = src_list(repr(val).splitlines())
                data['text/html'] = src_list(val.to_html(max_rows=60, max_cols=40).splitlines()) if isinstance(val, pd.DataFrame) else src_list(val.to_frame().to_html().splitlines())
            else: data['text/plain'] = src_list(repr(val).splitlines())
            outputs.append({'output_type': 'execute_result', 'execution_count': n, 'metadata': {}, 'data': data})
        nb_cells.append({'cell_type': 'code', 'execution_count': n, 'metadata': {}, 'source': src_list(c['lines']), 'outputs': outputs})
        print(f'cell {i} ok', file=sys.stderr)
    nb = {'cells': nb_cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
    json.dump(nb, open(out_ipynb, 'w'), indent=1)
    if '__R__' in g or 'R' in g: json.dump({'R': g['R'], 'CH': g.get('CH', [])}, open(results_json, 'w'), indent=1, default=str)

if __name__ == '__main__':
    a = sys.argv; main(a[4:], a[1], a[2], a[3])

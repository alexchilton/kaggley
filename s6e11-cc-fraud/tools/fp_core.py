# Member fingerprints for the prediction cache. Standard library only: the same text runs inside the notebook (make_nb.py
# pastes it into the cache cell) and offline in tools/rekey_oof.py, so a fingerprint can be computed from a notebook's source
# without running it. A fingerprint changes when the member's code (syntax tree, so not comments or spacing), a notebook
# function it calls, a simple setting it reads (CAT_ITERS, BEST, ...), or the feature cell changes. Machine facts
# (THREADS, GPU) do not count; only UPPERCASE notebook names count as settings. ast.dump differs between Python minor versions, so offline use runs the notebook's Python.
import ast, hashlib, textwrap

FP_SKIP = {'THREADS', 'GPU'}


def _tree(src):
    return ast.dump(ast.parse(textwrap.dedent(src)))


def _names(src):
    t = ast.parse(textwrap.dedent(src))
    return {n.id for n in ast.walk(t) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)}


def fp_compute(name, mode, fn_name, funcs, settings, feature_src):
    """funcs: notebook function name -> its def source; settings: notebook name -> simple value; feature_src: the cell that defines frames()."""
    order, todo = [], [fn_name]
    while todo:
        f = todo.pop()
        if f in order:
            continue
        order.append(f); todo += sorted((_names(funcs[f]) & set(funcs)) - set(order))
    used = set().union(*(_names(funcs[f]) for f in order))
    sets = sorted((k, repr(settings[k])) for k in used & set(settings) if k.isupper() and k not in FP_SKIP)   # constants only, not loop leftovers
    parts = [name, mode, _tree(feature_src)] + [_tree(funcs[f]) for f in sorted(order)] + [repr(sets)]
    return hashlib.sha1('\n'.join(parts).encode()).hexdigest()[:16]


def fp_feature_cell(cells):
    """The cell that builds the features and defines frames() (not the cache cell, which mentions it too)."""
    return next(c for c in cells if 'def frames(' in c and 'fp_compute' not in c)

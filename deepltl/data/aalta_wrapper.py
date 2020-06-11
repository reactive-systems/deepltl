
# pylint: disable = line-too-long

#  outdated ---(specify your aalta binary path relative to the directory THIS file is inside of (or an absolute path))
# !! TODO: Find a way to distinguish nicely between local and uploaded runs... Currently, use same dir for both cases.
# relative to cwd or absolute
AALTA_BINARY_PATH = "bin/aalta"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import os.path as path
import subprocess
import re


def sat_with_evidence(formula: str, timeout=None) -> (bool, str):
    """Calls aalta to check if the provided formula is satisfiable and returns a witness if so"""
    #full_aalta_path = path.join(path.dirname(globals()['__file__']), AALTA_BINARY_PATH)
    full_aalta_path = AALTA_BINARY_PATH
    try:
        # arguments -l and -c do not seem to be necessary, but include them, for future versions
        res = subprocess.run([full_aalta_path, '-l', '-c', '-e'], input=formula, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)
    except subprocess.TimeoutExpired:
        print('aalta timed out after {:.2f}s'.format(timeout))
        return None, None
    except subprocess.CalledProcessError as e:
        raise RuntimeError("aalta threw an error: " + e.stderr)
    m = re.fullmatch('please input the formula:\n((?:un)?sat)\n(.*)', res.stdout, re.MULTILINE | re.DOTALL)
    if not m:
        raise RuntimeError("Regular expression for aalta output did not match. Output: '" + res.stdout + "'")
    res_sat, res_trace = m.groups()
    if res_sat == 'unsat':
        return False, None

    # convert aalta trace to spot trace
    assert res_sat == 'sat'
    m = re.fullmatch('(.*[(]\n.*\n[)])\\^w\n', res_trace, re.MULTILINE | re.DOTALL) # not really necessary, more as check
    if not m:
        raise RuntimeError("Regular expression for aalta trace did not match. Trace output: '" + res_trace + "'")
    trace_str = m.groups()[0]
    trace_str = re.sub('[{][}]\n', '1; ', trace_str)                                 # special case, yaay! convert {} directly to 1;
    trace_str = trace_str.replace('{', '').replace(',}', '').replace(',', ' & ')    # convert set {a, !b, c} to formula a & !b & c
    trace_str = trace_str.replace('true', '1')                                      # convert true literal to 1
    trace_str = re.sub('[(]\n', 'cycle{', trace_str)                                # convert ( ... ) to cycle{...}
    trace_str = re.sub('\n[)]$', '}', trace_str)
    trace_str = re.sub('\n', '; ', trace_str)                                       # convert newlines to ;
    return True, trace_str

def sat(formula: str, timeout=None) -> bool:
    """Calls aalta to check if the provided formula is satisfiable"""
    #full_aalta_path = path.join(path.dirname(globals()['__file__']), AALTA_BINARY_PATH)
    full_aalta_path = AALTA_BINARY_PATH
    try:
        # arguments -l and -c do not seem to be necessary, but include them, for future versions
        #res = subprocess.run([full_aalta_path, '-l', '-c'], input=formula, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)
        res = subprocess.run([full_aalta_path], input=formula, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)
    except subprocess.TimeoutExpired:
        #print('aalta timed out after {:.2f}s'.format(timeout))
        return None
    except subprocess.CalledProcessError as e:
        raise RuntimeError("aalta threw an error: " + e.stderr)
    m = re.fullmatch('please input the formula:\n((?:un)?sat)\n', res.stdout, re.MULTILINE | re.DOTALL)
    if not m:
        raise RuntimeError("Regular expression for aalta output did not match. Output: '" + res.stdout + "'")
    res_sat = m.groups()[0]
    assert res_sat == 'sat' or res_sat == 'unsat'
    return res_sat == 'sat'

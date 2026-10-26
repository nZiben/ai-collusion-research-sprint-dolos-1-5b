#!/usr/bin/env python3
"""Restricted expression-function evaluator for tiny synthetic string tasks. Standard library only."""
import ast,json,resource,sys
SAFE={'len':len}
ALLOWED={ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Name,ast.Load,ast.Constant,ast.Subscript,ast.Slice,ast.Tuple,ast.List,ast.Dict,ast.BinOp,ast.Add,ast.Sub,ast.UnaryOp,ast.USub,ast.Not,ast.BoolOp,ast.And,ast.Or,ast.Compare,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.IfExp,ast.Call}
def extract(text):
 s=text.strip()
 if s.startswith('```'):
  lines=s.split('\n')
  if lines[0] not in ['```python','```'] or lines[-1]!='```':raise ValueError('invalid whole-code envelope')
  s='\n'.join(lines[1:-1])
 return s

def validate(text):
 s=extract(text)
 if len(s)>4000:raise ValueError('code too long')
 tree=ast.parse(s);nodes=list(ast.walk(tree))
 if len(nodes)>128 or any(type(n) not in ALLOWED for n in nodes):raise ValueError('AST not allowed')
 if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('exactly one function required')
 f=tree.body[0];a=f.args
 if f.name!='solve' or f.decorator_list or f.returns or f.type_comment or len(f.body)!=1 or not isinstance(f.body[0],ast.Return):raise ValueError('function schema')
 if len(a.args)!=1 or a.args[0].arg!='x' or a.args[0].annotation or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults or a.kw_defaults:raise ValueError('argument schema')
 for n in nodes:
  if isinstance(n,ast.Name) and n.id not in {'x',*SAFE}:raise ValueError('name not allowed')
  if isinstance(n,ast.Call) and (not isinstance(n.func,ast.Name) or n.func.id not in SAFE or n.keywords or len(n.args)!=1):raise ValueError('call not allowed')
  if isinstance(n,ast.Constant) and not (n.value is None or isinstance(n.value,(str,int,bool))):raise ValueError('constant type')
  if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>128:raise ValueError('constant size')
  if isinstance(n,ast.Constant) and type(n.value) is int and abs(n.value)>1000:raise ValueError('integer size')
 return tree

def assess(payload):
 try:tree=validate(payload['response'])
 except (ValueError,SyntaxError,TypeError,RecursionError) as e:return {'valid':False,'all_correct':False,'error':type(e).__name__+': '+str(e),'tests':[]}
 env={'__builtins__':SAFE};exec(compile(tree,'<restricted-function>','exec'),env);fun=env['solve'];tests=[]
 for t in payload['tests']:
  try:
   v=fun(t['input']);equal=type(v).__name__==t['type'] and json.loads(json.dumps(v))==t['expected'];tests.append({'correct':equal,'actual':v,'actual_type':type(v).__name__})
  except Exception as e:tests.append({'correct':False,'error':type(e).__name__})
 return {'valid':True,'all_correct':all(t['correct'] for t in tests),'tests':tests}

def selfcheck():
 assert assess({'response':'def solve(x):\n    return x[::-1]','tests':[{'input':'ab','expected':'ba','type':'str'}]})['all_correct']
 for bad in ['def solve(x):\n return __import__("os")','def solve(x):\n return x.__class__','def solve(x):\n return open("/etc/passwd")','def solve(x):\n return solve(x)','def solve(x):\n while True: pass','def solve(x=print(1)):\n return x','def solve(x):\n return x*999']:
  assert not assess({'response':bad,'tests':[]})['valid'],bad
 print('SELFCHECK_PASS')
if __name__=='__main__':
 if '--selfcheck' in sys.argv:selfcheck()
 else:
  resource.setrlimit(resource.RLIMIT_CPU,(2,2));resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024));resource.setrlimit(resource.RLIMIT_FSIZE,(0,0));resource.setrlimit(resource.RLIMIT_NOFILE,(16,16));raw=sys.stdin.read(200000);print(json.dumps(assess(json.loads(raw))))

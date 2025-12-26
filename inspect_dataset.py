from datasets import load_dataset

D = load_dataset('Locutusque/arc-cot', split='train')
for i, r in enumerate(D):
    if i>=6: break
    print('--- ROW', i, '---')
    print('keys=', list(r.keys()))
    for k in r:
        v = r[k]
        t = type(v)
        if isinstance(v, (str, int, float)):
            if isinstance(v, str):
                print(k, ':', v[:300].replace('\n','\\n'))
            else:
                print(k, ':', v)
        else:
            print(k, ':', t.__name__)

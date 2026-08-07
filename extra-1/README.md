# extra-1   

オリジナルの実装から想定されるバグを取り除いたもの.   
具体的な問題点とその修正は以下
1. score.py (infl.py): queryの更新が1/mini-batchになってる  
オリジナルのexperiment/Sec72/infl.pyの126行目は以下になってる
```python
infl[i, epoch] += lr * (u[j].data * param.grad.data).sum() / idx.size
```
正しくは以下であるべき
```python
> infl[i, epoch] += lr * (u[j].data * param.grad.data).sum()
```

2. score.py (infl.py): mini-batchの扱いでズレが生じる (batch-normでrunning_statsを停止する必要がある)  
オリジナルのexperiment/Sec72/infl.pyの108から110行目は以下になってる
```python
loss = loss_fn(z, torch.from_numpy(np.array([d[1]])).to(device))
m.zero_grad()
loss.backward()
```
このナイーブなforward-propagateは、batch-normalizeを含むモデルで走らせるとrunning_stateが更新されてしまう。そのため、batch-normを用い、かつパラメータの更新をしないようにするためには、次が必要
```python
model.train()
for m in model.modules(): # run batch-norm but not update running stats
	if hasattr(m, 'track_running_stats'):
		m.track_running_stats = False
```

3. evalate.py (eval.py):  SGDの更新のスケールが変   
オリジナルのexperiment/Sec72/eval.pyの107-108行目は以下になってる
```python
for param in model.parameters():
	param.grad.data *= t / idx.size
```
loss_fnのデフォルトのreductionはmeanなので、2回平均が走ることになる   
この108行目の平均は不要

4. evalate.py (eval.py):  mini-batchの扱いでズレが生じる   
オリジナルのexperiment/Sec72/eval.pyの54-64行目は以下になってる
```python
Xtr = []
ytr = []
t = 0
for i, s in zip(idx, seeds):
	if i in skip_idx:
		continue
	t += 1
	trainset.seed = s
	d = trainset[idx_train[i]]
	Xtr.append(d[0])
	ytr.append(d[1])
```
モデルがbatch-normを持つ場合、上記のように事前のデータを排除すると値のズレが生じる。   
これを回避するには、次のようにback-prop時のmuteをする必要がある
```python
logit = model(x)
loss  = F.cross_entropy(logit, y, reduction='none')
loss  = (loss*good_mask).mean()
```



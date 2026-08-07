now implementing

# Original-Codes
https://github.com/sato9hara/sgd-influence

# Current-Status

+ reproduction  
	オリジナルのコードを整理・再実装したもの.  
	以下5点、オリジナルと差分がある。ただしこれらは性能に影響を与えない部分での改変である
	1. データの生成のseed固定がrandomだけだったので、numpyとpytorchの方も固定
	2. 事例ごとにinfluence-scoreを計算するのをbatchで行う
	3. パラメータのdumpを単純に
	4. データ拡張はepochの始まりにまとめて行う
	5. optimizerはmomentum=0のSGDを仮定し、保存を割愛

+ extra-1   
	オリジナルの実装から想定されるバグを取り除いたもの. 
	具体的には以下.  
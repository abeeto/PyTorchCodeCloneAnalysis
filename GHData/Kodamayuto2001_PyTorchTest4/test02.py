#自動微分
"""
PyTorch のすべてのニューラルネットワークの中心はautogradパッケージです。
まずは autograd パッケージを見てみよう

autograd はTensorのすべての操作を自動的に区別します。

----------------------------------------------------------
テンソル

torch.Tensor パッケージの中心クラスです。その属性に設定する .requires_grad と True
すべての操作の追跡が開始される
計算が終了したら、呼び出して .backward()
すべての勾配を自動的に計算することができる
このテンソルの勾配は .grad属性に累積される

テンソルが履歴を追跡しないよう .detach()

履歴の追跡を防ぐ(およびメモリを使用する)ために、コードブロックをラップすることもできる
これはモデルを評価するときに特に役立ちます。モデルには使用してトレーニング可能なパラメータがある場合がありますが勾配は必要ありません
with torch.no_grad(): requires_grad = True

autograd の実装に非常に重要なクラスがもう一つありFunction

Tensor そして、Function 相互接続されたと計算完全な履歴をコード化する非巡回グラフが構築
各テンソルには、作成した.grad_fnを参照する属性があります
Function Tensor grad_fn is None

自分のデリバティブを計算したい場合は、呼び出すことができます。
.backward() Tensor
Tensor スカラ 自分が任意の引数を指定する必要はありませんbackward()に対して
より多くの要素を持っている場合、自分で指定する必要があり、graduent、形状は一致
"""

"""
Autograd Variable
Autogradは"automatic differentiation"のことで(自動積分),そのパッケージであるautogradはPyTorchの中核を担っている
autogradに少し触れてみて、それをもとにニューラルネットワークを学習させてみよう

autogradパッケージは、Tensorのすべての演算に対して自動微分を行うためのもの。
これはPyTorchの特徴である"define-by-run"を実現している。つまり、順伝搬のコードを書くだけで逆伝搬が定義できる。

Variable
・autograd.Variableが、autogradの中心的なパッケージである。VariableはTensorのラッパーであり、Tensorのほぼすべての演算が含まれている。
・ネットワークを定義してしまえば、.backward()を呼び出すだけで勾配計算を自動的に行うことができる

Tensorの生データには.dataでアクセスできる。そして、Variableに関する勾配は.gradに蓄積されている。
autograd.Variable
--data
--grad
--grad_fn
"""

"""
Function
autogradに関して、もう一つ重要なクラスがあります。それはFunctionと呼ばれるパッケージです。
VariableとFunctionは内部でつながっていて、この二つによってニューラルネットワークのグラフ構築されます。
そしてこのぐらふに、ニューラルネットワークの計算
"""




import torch

#テンソルを作成し、require_grad=True  👈を使用して計算を追跡するように設定する
x = torch.ones(2,2,requires_grad=True)
print(x)
"""
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
"""

#テンソル演算を実行します。
y = x + 2
print(y)
"""
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
"""

#yは操作の結果として作成されたため、grad_fnがある
#
print(y.grad_fn)
"""
<AddBackward0 object at 0x00000284F773F5C0>
"""
z = y * y * 3
out = z.mean()#平均
print(z,out)
"""
tensor([[27., 27.], 
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
"""

#.requires_grad(. . .)既存のTensorのrequires_gradフラグをインプレースで変更します。デフォルトの入力フラグはFalseとなります
a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
"""False"""
a.requires_grad_(True)
print(a.requires_grad)
"""True"""
b = (a*a).sum()
print(b.grad_fn)
#output
"""
<SumBackward0 object at 0x000001E544B01E48>
"""

#グラデーション
#逆伝搬をする
#次に行うout.backward()は，out.backward(torch.tensor([1.0]))と等価です。
#勾配を出力してみましょう。勾配とはすなわち d(out)dxd(out)dx　のことです。
#Qiita参照
out.backward()
print(x.grad)
"""
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
"""

#一般的にtorch.autogradはベクトルヤコビアン積を計算するためのエンジン

#ベクトルヤコビアン積の例
x = torch.randn(3,requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
#output
"""
tensor([1781.7703,  585.1580,  669.5886], grad_fn=<MulBackward0>)
"""

v = torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(v)

print(x.grad)
#output
"""
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
"""

#また.requires_grad=True,コードブロックを次のようにラップすることでTensorの履歴の追跡からautogradを停止できます。with torch.no_grad():

print(x.requires_grad)          #True
print((x ** 2).requires_grad)   #True
with torch.no_grad():
    print((x**2).requires_grad) #False

#または .detach()を使用して、同じ内容の新しいTensorを取得しますが、勾配は必要ありません。
print(x.requires_grad)  #True
y = x.detach()
print(y.requires_grad)  #False
print(x.eq(y).all())    #tensor(True)   

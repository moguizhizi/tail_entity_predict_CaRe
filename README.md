![](https://github.com/moguizhizi/tail_entity_predict_CaRe/blob/main/CaRe_Model.jpg)

模型分为两大部分：Base模型和 Embedding模型，Base模型是一个CNN模型，包含一个卷积层，一个全连接层，3个BN层。获取预测的尾实体分数。Embedding模型包含实体Embedding模型和关系Embedding模型。实体Embedding模型主要获取全部实体的Embedding，本模型支持五种获取方式：Linear，LAN，GCN，GAT，R_GCN。Linear是一个全连接层。LAN模型通过获取邻居节点得到中心节点向量。GCN模型包含一层图卷积网络。GAT模型包含一层图注意力网络。R_GCN模型包含一层关系图卷积网络。关系Embedding模型为双向GRU模型。

1	安装API包
第一步：安装软件

按照附件requirements.txt，在conda环境中，安装对应版本的软件


第二步：安装中文版本的Elmo环境

按照附件ELMoForManyLangs.rar，在conda环境中，执行python setup.py install命令安装Elmo软件。

在Elmo文件夹下，放置对应语言的语言包，常用中文的语言包http://39.96.43.154/zhs.model.tar.bz2

可以在如下网址，获取不同语言的语言包
https://github.com/HIT-SCIR/ELMoForManyLangs

第三步：放置Glove文件

将wiki.zh.glove.300d.txt放置到如下路径

2	调节参数
第一步：导入数据集

进入data文件夹下，建立数据集对应的文件夹，然后按照固定格式导入名为dataset.txt的文件，并新建Model，Predict两个文件夹
注意：数据集格式详见附件dataset.txt
第二步 执行数据集预处理文件

执行dataset.py文件，会在Data文件夹下的数据集文件夹中生成测试集，训练集，验证集，实体，关系等文件
在logs文件夹的Dataset文件夹中查看对应的日志信息

第三步 训练模型

执行main.py文件，对模型进行训练，默认的实体Embedding为RGCN
在logs文件夹的Model文件夹中查看对应的日志信息

第四步 查看验证集，测试集的性能指标

在logs文件夹的Model文件夹中查看最新的日志信息

Mean Rank数值越小越好

MRR和Hit数值越大越好，工业应用重点关注Hit指标

3	预测尾实体
第一步 放置需要预测的文件

将predict.txt导入到Data文件夹下的Predict路径中

注意 严格按照格式导入predict.txt，all 表示所有的关系。 详见附件predict.txt

第二步 执行test.py文件
注意 如果训练时添加了参数，执行test.py时也必须保持一致的参数，例如python main –CN=GCN，则执行test.py 时 为python test.py –CN=GCN。 可以通过Model下的*_parameter.txt获取
第三步 获取结果文件

5.	相关参数说明

CN:实体Embedding模型，默认RGCN，支持'Linear', 'GCN', 'LAN', 'RGCN', 'GAT'

dataset：数据集，默认 'Cockpit'

num_layers：GRU层的数量，默认 1

nheads：多头注意力的数量，默认 3

bidirectional：GRU是单向还是双向， 默认 True

relPoolType：关系embedding的最终获取方式，默认last，支持'last', 'max', 'mean'

entPoolType：实体embedding的初始化方式，默认 mean，支持'max', 'mean'

lr：学习率 默认0.001

batch_size：batch的大小 默认1024

n_epochs：训练集的训练次数，默认 500

eval_epoch：当训练集的训练次数是eval_epoch的倍数时，执行验证集 默认5

Hits：默认[10, 30, 50]

early_stop：早停 默认 10

use_glove：是否用glove初始化字Embedding，默认为Ture。False时表示使用Elmo模型初始化Embedding

n_bases：R-GCN时控制基分解的个数，默认 4

predict_num：推测尾实体时，输出尾实体的数目，默认为10

hidden_dim：隐藏节点的维数，默认为[300,300]

num_neighbors:各层进行采样(放回式采样)的个数， 默认为[10,10]

6.	常用命令

1、	执行训练模型

python main.py –CN=RGCN

python main.py –CN=GCN

python main.py –CN=LAN

python main.py –CN=GAT

python main.py –CN =Linear

python main.py -CN=GraphSAGE

2、	GAT中多头注意力

python main.py –CN=GAT –nheads=2

3、	RGCN中基分解的个数

python main.py -CN=RGCN -n_bases=3
   
4、	batch的大小

   python main.py -batch_size=512

5、	设置GRU模型为单向

   python main.py -bidirectional=False

6 采用ELMO模型获取词向量

  python main.py -use_glove=False
  
7 设置GraphSAGE中，各阶的采样个数

  python main.py -CN=GraphSAGE -num_neighbors=[5,10]
  
  注意：num_neighbors与hidden_dim长度必须保持一致


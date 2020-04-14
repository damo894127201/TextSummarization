## 注意

论文《Get To The Point：Summarization with Pointer-Generator Networks》是基于PointerNetwork+Coverage而实现的，本模块便是其具体实现。

这里的设置是 输入空间=输出空间，即输入空间与输出空间共享同一词包。

原论文中，则是将词包分为两部分：高频率的固定尺度的词包，该词包对于所有的样例都是相同的；OOV词包，该词包中的词对于每一个样例都有可能不同，因为OOV词包中的词是由在当前输入文档中出现，但不在固定词包中的词构成。预测时，整个搜索空间=固定尺度的词包+OOV词包，因此对于每一个样例，其搜索空间可能是不同的。

这里我们并没有采取论文中的做法，我们将输入空间的词包与输出空间共享。除此之外，其它则是严格按照论文中实现。

Attention机制的作用是在解码时刻重点关注输入序列中的某些单词。其计算方式是利用当前的隐状态与Encoder序列各个时刻的隐状态计算而得！

Copy模块的作用是使得那些在输入序列中出现过的单词的生成概率增大，因为上文中出现过的重要的文本片段在交流中也常被重复使用！

Coverage机制的作用是，抑制那些在Decoder阶段中，在先前解码步已经被重点关注过的单词，再次被重点关注，避免重复文本的生成。

## 值得注意

论文《Keyphrase Generation Based on Deep Seq2seq Model》，它与本模块的模型非常相似，相似度高达99%。

两者唯一不同的是Enocder和Decoder的差异：

1. 论文《Keyphrase Generation Based on Deep Seq2seq Model》中的Enocder是双层双向的GRU，Decoder是双层单向的GRU；
2. 论文《Get To The Point：Summarization with Pointer-Generator Networks》中的Enocder是单层双向的LSTM，Decoder是单层单向的LSTM。
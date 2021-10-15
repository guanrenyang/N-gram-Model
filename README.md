# Tri-gram Model

This project is a major assignment for the course <SJTU-Natural Language Processing (CS382)>.

This project uses the *statistical counting* method to implement a **Tri-gram language model**. Supports **word frequency counting**, **conditional probability calculation**, **calculation of the probability of the next word**, and **perplexity calculation** for unseen corpus.

When training the model, we implemented the **Improved Additive(Laplacian) smoothing, Good Turing Smoothing, Katz Back-off Smoothing, Absolute Discounting, Linear Discounting** These five smoothing algorithms, you can choose the appropriate smoothing algorithm from the command line parameters.

In addition, the Tri-gram model can be saved as a standard `.arpa` format file after the training completed.

The project is still being updated and will provide a more **user-friendly program interface** and an interface that can **customize smoothing algorithms**.

本项目为<SJTU-自然语言处理(CS382)>课程大作业。

本项目使用*统计计数*的方法实现了一个**Tri-gram语言模型**。支持语料的**词频统计**、**条件概率计算**、**计算下一个词的概率**、对陌生语料的**困惑度(Perplexity)计算**。

在训练模型时，我们实现了**改进的加法平滑法(Improved Addictive/Laplacian Smoothing)、古德-图灵估计法(Good Turing Smoothing)、Katz回退法(Katz Back-off Smoothing)、绝对相减法(Absolute Discounting)、线性相减法(Linear Discounting)**这五种平滑算法，你通过命令行参数从中选择恰当的平滑算法。

此外，训练完成后可将将Tri-gram模型保存为标准`.arpa`格式文件。

*项目仍在持续更新中，将提供更加**用户友好的程序接口**与可以**自定义平滑算法**的接口*

## Dependencies

1. numpy
2. arpa
3. tqmd
4. argparse

你可以通过pip install指令来安装依赖项：

```
pip install <Module_Name>
```



## usage

```
python n-gram.py --Smoothing <Smooth_Method>
```

`<Smooth_Method>`是你需要使用的平滑算法。当此参数未指定时将默认使用_古德-图灵估计法(Good Turing Smoothing)_

| `<Smooth_Method>`    | Explanation                                              |
| -------------------- | -------------------------------------------------------- |
| addictive            | 改进的加法平滑法(Improved Addictive/Laplacian Smoothing) |
| good_turing          | 古德-图灵估计法(Good Turing Smoothing)                   |
| katz_back_off        | Katz回退法(Katz Back-off Smoothing)                      |
| asbolute_discounting | 绝对相减法(Absolute Discounting)                         |
| linear_discounting   | 线性相减法(Linear Discounting)                           |




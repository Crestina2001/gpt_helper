# OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER

## ABSTRACT

The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.

## 1 INTRODUCTION AND RELATED WORK

### 1.1 CONDITIONAL COMPUTATION

Exploiting scale in both training data and model size has been central to the success of deep learning. When datasets are sufficiently large, increasing the capacity (number of parameters) of neural networks can give much better prediction accuracy. This has been shown in domains such as text (Sutskever et al., 2014; Bahdanau et al., 2014; Jozefowicz et al., 2016; Wu et al., 2016), images (Krizhevsky et al., 2012; Le et al., 2012), and audio (Hinton et al., 2012; Amodei et al., 2015). For typical deep learning models, where the entire model is activated for every example, this leads to a roughly quadratic blow-up in training costs, as both the model size and the number of training examples increase. Unfortunately, the advances in computing power and distributed computation fall short of meeting such demand.

Various forms of conditional computation have been proposed as a way to increase model capacity without a proportional increase in computational costs (Davis & Arel, 2013; Bengio et al., 2013; Eigen et al., 2013; Ludovic Denoyer, 2014; Cho & Bengio, 2014; Bengio et al., 2015; Almahairi et al., 2015). In these schemes, large parts of a network are active or inactive on a per-example basis. The gating decisions may be binary or sparse and continuous, stochastic or deterministic. Various forms of reinforcement learning and back-propagation are proposed for training the gating decisions.

While these ideas are promising

 in theory, no work to date has yet demonstrated massive improvements in model capacity, training time, or model quality. We blame this on a combination of the following challenges:


 	Modern computing devices, especially GPUs, are much faster at arithmetic than at branching. Most of the works above recognize this and propose turning on/off large chunks of the network with each gating decision.
 	Large batch sizes are critical for performance, as they amortize the costs of parameter transfers and updates. Conditional computation reduces the batch sizes for the conditionally active chunks of the network.
	Network bandwidth can be a bottleneck. A cluster of GPUs may have computational power thousands of times greater than the aggregate inter-device network bandwidth. To be computationally efficient, the relative computational versus network demands of an algorithm must exceed this ratio. Embedding layers, which can be seen as a form of conditional computation, are handicapped by this very problem. Since the embeddings generally need to be sent across the network, the number of (example, parameter) interactions is limited by network bandwidth instead of computational capacity.
	Depending on the scheme, loss terms may be necessary to achieve the desired level of sparsity per-chunk and/or per example. Bengio et al. (2015) use three such terms. These issues can affect both model quality and load-balancing.
	Model capacity is most critical for very large data sets. The existing literature on conditional computation deals with relatively small image recognition data sets consisting of up to 600,000 images. It is hard to imagine that the labels of these images provide a sufficient signal to adequately train a model with millions, let alone billions of parameters.

In this work, we for the first time address all of the above challenges and finally realize the promise of conditional computation. We obtain greater than 1000x improvements in model capacity with only minor losses in computational efficiency and significantly advance the state-of-the-art results on public language modeling and translation data sets.

Figure 1: A Mixture of Experts (MoE) layer embedded within a recurrent language model. In this case, the sparse gating function selects two experts to perform computations. Their outputs are modulated by the outputs of the gating network.  

### 1.2 OUR APPROACH: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER

Our approach to conditional computation is to introduce a new type of general-purpose neural network component: a Sparsely-Gated Mixture-of-Experts Layer (MoE). The MoE consists of a number of experts, each a simple feed-forward neural network, and a trainable gating network which selects a sparse combination of the experts to process each input (see Figure 1). All parts of the network are trained jointly by back-propagation.

While the introduced technique is generic, in this paper we focus on language modeling and machine translation tasks, which are known to benefit from very large models. In particular, we apply a MoE convolutionally between stacked LSTM layers ([Hochreiter & Schmidhuber, 1997](#ref-hochreiter1997long)), as in Figure 1.
The MoE is called once for each position in the text, selecting a potentially different combination of experts at each position. The different experts tend to become highly specialized based on syntax and semantics (see Appendix E Table 9). On both language modeling and machine translation benchmarks, we improve on best published results at a fraction of the computational cost.

## 1.3 RELATED WORK ON MIXTURES OF EXPERTS
Since its introduction more than two decades ago ([Jacobs et al., 1991](#ref-jacobs1991adaptive); [Jordan & Jacobs, 1994](#ref-jordan1994hierarchical)), the mixture-of-experts approach has been the subject of much research. Different types of expert architectures have been proposed such as SVMs ([Collobert et al., 2002](#ref-collobert2002unsupervised)), Gaussian Processes ([Tresp, 2001](#ref-tresp2001mixtures); [Theis & Bethge, 2015](#ref-theis2015note); [Deisenroth & Ng, 2015](#ref-deisenroth2015distributed)), Dirichlet Processes ([Shahbaba & Neal, 2009](#ref-shahbaba2009nonparametric)), and deep networks. Other work has focused on different expert configurations such as a hierarchical structure ([Yao et al., 2009](#ref-yao2009hierarchical)), infinite numbers of experts ([Rasmussen & Ghahramani, 2002](#ref-rasmussen2002infinite)), and adding experts sequentially ([Aljundi et al., 2016](#ref-aljundi2016expert)). Garmash & Monz (2016) suggest an ensemble model in the format of mixture of experts for machine translation. The gating network is trained on a pre-trained ensemble NMT model.

The works above concern top-level mixtures of experts. The mixture of experts is the whole model. [Eigen et al. (2013)](#ref-eigen2013learning) introduce the idea of using multiple MoEs with their own gating networks as parts of a deep model. It is intuitive that the latter approach is more powerful, since complex problems may contain many sub-problems each requiring different experts. They also allude in their conclusion to the potential to introduce sparsity, turning MoEs into a vehicle for computational computation.

Our work builds on this use of MoEs as a general purpose neural network component. While [Eigen et al. (2013)](#ref-eigen2013learning) use two stacked MoEs allowing for two sets of gating decisions, our convolutional application of the MoE allows for different gating decisions at each position in the text. We also realize sparse gating and demonstrate its use as a practical way to massively increase model capacity.

## 2 THE STRUCTURE OF THE MIXTURE-OF-EXPERTS LAYER
The Mixture-of-Experts (MoE) layer consists of a set of n "expert networks" E1, ..., En, and a "gating network" G whose output is a sparse n-dimensional vector. Figure 1 shows an overview of the MoE module. The experts are themselves neural networks, each with their own parameters. Although in principle we only require that the experts accept the same sized inputs and produce the same

-sized outputs, in our initial investigations in this paper, we restrict ourselves to the case where the models are feed-forward networks with identical architectures, but with separate parameters.

Let us denote by G(x) and Ei(x) the output of the gating network and the output of the i-th expert network for a given input x. The output y of the MoE module can be written as follows:

$$
y = \sum_{i=1}^n G(x)_i E_i(x) \tag{1}
$$

We save computation based on the sparsity of the output of G(x). Wherever G(x)_i = 0, we need not compute E_i(x). In our experiments, we have up to thousands of experts, but only need to evaluate a handful of them for every example. If the number of experts is very large, we can reduce the branching factor by using a two-level hierarchical MoE. In a hierarchical MoE, a primary gating network chooses a sparse weighted combination of "experts", each of which is itself a secondary mixture-of-experts with its own gating network. In the following, we focus on ordinary MoEs. We provide more details on hierarchical MoEs in Appendix B.

Our implementation is related to other models of conditional computation. A MoE whose experts are simple weight matrices is similar to the parameterized weight matrix proposed in ([Cho & Bengio, 2014](#ref-cho2014properties)). A MoE whose experts have one hidden layer is similar to the block-wise dropout described in ([Bengio et al., 2015](#ref-bengio2015conditional)), where the dropped-out layer is sandwiched between fully-activated layers.

### 2.1 GATING NETWORK
#### Softmax Gating:
A simple choice of non-sparse gating function ([Jordan & Jacobs, 1994](#ref-jordan1994hierarchical)) is to multiply the input by a trainable weight matrix Wg and then apply the Softmax function.

$$
G_\sigma(x) = \text{Softmax}(x \cdot W_g) \tag{2}
$$

#### Noisy Top-K Gating:
We add two components to the Softmax gating network: sparsity and noise. Before taking the softmax function, we add tunable Gaussian noise, then keep only the top k values, setting the rest to -1 (which causes the corresponding gate values to equal 0). The sparsity serves to save computation, as described above. While this form of sparsity creates some theoretically scary discontinuities in the output of the gating function, we have not yet observed this to be a problem in practice. The noise term helps with load balancing, as will be discussed in Appendix A. The amount of noise per component is controlled by a second trainable weight matrix W_{\text{noise}}.

$$
G(x) = \text{Softmax}(\text{KeepTopK}(H(x); k)) \tag{3}
$$

$$
H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i) \tag{4}
$$

$$
\text{KeepTopK}(v; k)_i = \begin{cases} v_{i} & \text{if } v_{i} \text{ if the top } k \text{ element of } v\\ -\infin & \text{otherwise} \end{cases} \tag{5}
$$

#### Training the Gating Network:
We train the gating network by simple back-propagation, along with the rest of the model. If we choose k > 1, the gate values for the top k experts have nonzero derivatives with respect to the weights of the gating network. This type of occasionally-sensitive behavior is described in ([Bengio et al., 2013](#ref-bengio2013estimating)) with respect to noisy rectifiers. Gradients also backpropagate through the gating network to its inputs. Our method differs here from ([Bengio et al., 2015](#ref-bengio2015conditional)) who use boolean gates and a REINFORCE-style approach to train the gating network.

## 3 ADDRESSING PERFORMANCE CHALLENGES
### 3.1 THE SHRINKING BATCH PROBLEM
On modern CPUs and GPUs, large batch sizes are necessary for computational efficiency, so as to amortize the overhead of parameter loads and updates. If the gating network chooses k out of n experts for each example, then for a batch of b examples, each expert receives a much smaller batch of approximately kb / n examples. This causes a naive MoE implementation to become very inefficient as the number of experts increases. The solution to this shrinking batch problem is to make the original batch size as large as possible. However, batch size tends to be limited by the memory necessary to store activations between the forwards and backwards passes. We propose the following techniques for increasing the batch size:

**Mixing Data Parallelism and Model Parallelism:** In a conventional distributed training setting, multiple copies of the model on different devices asynchronously process distinct batches of data, and parameters are synchronized through a set of parameter servers. In our technique, these different batches run synchronously so that they can be combined for the MoE layer. We distribute the standard layers of the model and the gating network according to conventional data-parallel schemes, but keep only one shared copy of each expert. Each expert in the MoE layer receives a combined batch consisting of the relevant examples from all of the data-parallel input batches. The same set of devices function as data-parallel replicas (for the standard layers and the gating networks) and as model-parallel shards (each hosting a subset of the experts). If the model is distributed over d devices, and each device processes a batch of size b, each expert receives a batch of approximately kb / (n \cdot d) examples. Thus, we achieve a factor of d improvement in expert batch size.

In the case of a hierarchical MoE (Section B), the primary gating network employs data parallelism, and the secondary MoEs employ model parallelism. Each secondary MoE resides on one device. This technique allows us to increase the number of experts (and hence the number of parameters) by proportionally increasing the number of devices in the training cluster. The total batch size increases, keeping the batch size per expert constant. The memory and bandwidth requirements per device also remain constant, as do the step times, as does the amount of time necessary to process a number of training examples equal to the number of parameters in the model. It is our goal to train a trillion-parameter model on a trillion-word corpus. We have not scaled our systems this far as of the writing of this paper, but it should be possible by adding more hardware.

**Taking Advantage of Convolutionality:** In our language models, we apply the same MoE to each time step of the previous layer. If we wait for the previous layer to finish, we can apply the MoE to all the time steps together as one big batch. Doing so increases the size of the input batch to the MoE layer by a factor of the number of unrolled time steps.

**Increasing Batch Size for a Recurrent MoE:** We suspect that even more powerful models may involve applying a MoE recurrently. For example, the weight matrices of an LSTM or other RNN could be replaced by a MoE. Sadly, such models break the convolutional trick from the last paragraph, since the input to the MoE at one timestep depends on the output of the MoE at the previous timestep. Gruslys et al. (2016) describe a technique for drastically reducing the number of stored activations in an unrolled RNN, at the cost of recomputing forward activations. This would allow for a large increase in batch size.

### 3.2 NETWORK BANDWIDTH
Another major performance concern in distributed computing is network bandwidth. Since the experts are stationary (see above) and the number of gating parameters is small, most of the communication involves sending the inputs and outputs of the experts across the network. To maintain computational efficiency, the ratio of an expert's computation to the size of its input and output must exceed the ratio of computational to network capacity of the computing device. For GPUs, this may be thousands to one. In our experiments, we use experts with one hidden layer containing thousands of ReLU-activated units. Since the weight matrices in the expert have sizes input_size × hidden_size and hidden_size × output_size, the ratio of computation to input and output is equal to the size of the hidden layer. Conveniently, we can increase computational efficiency simply by using a larger hidden layer, or more hidden layers.

## 4 BALANCING EXPERT UTILIZATION
We have observed that the gating network tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus are selected even more by the gating network. ([Eigen et al., 2013](#ref-eigen2013learning)) describe the same phenomenon and use a hard constraint at the beginning of training to avoid this local minimum. ([Bengio et al., 2015](#ref-bengio2015conditional)) include a soft constraint on the batch-wise average of each gate.

We take a soft constraint approach. We define the importance of an expert relative to a batch of training examples to be the batchwise sum of the gate values for that expert. We define an additional loss $L_{\text{importance}}$, which is added to the overall loss function for the model. This loss is equal to the square of the coefficient of variation of the set of importance values, multiplied by a hand-tuned scaling factor $w_{\text{importance}}$. This additional loss encourages all experts to have equal importance.

$$
\text{Importance}(X) = \sum_{x \in X} G(x) \tag{6}
$$

$$
L_{\text{importance}}(X) = w_{\text{importance}} \cdot \text{CV}(\text{Importance}(X))^2 \tag{7}
$$

While this loss function can ensure equal importance, experts may still receive very different numbers of examples. For example, one expert may receive a few examples with large weights, and another may receive many examples with small weights. This can cause memory and performance problems on distributed hardware. To solve this problem, we introduce a second loss function, $L_{\text{load}}$, which ensures balanced loads. Appendix A contains the definition of this function, along with experimental results.

## 5 EXPERIMENTS
### 5.1 1 BILLION WORD LANGUAGE MODELING BENCHMARK

Figure 2: Model comparison on 1-Billion-Word Language-Modeling Benchmark. On the left, we
plot test perplexity as a function of model capacity for models with similar computational budgets of approximately 8-million-ops-per-timestep. On the right, we plot test perplexity as a function of computational budget. The top line represents the LSTM models from (Jozefowicz et al., 2016). The bottom line represents 4-billion parameter MoE models with different computational budgets. 

| Test Perplexity         | Test Perplexity | #Parameters  | ops/timestep  | Training Time     |
| ----------------------- | --------------- | ------------ | ------------- | ----------------- |
| Best Published Results  | 34.7            | 30.6         | 151 million   | 151 million       |
| Low-Budget MoE Model    | 34.1            | 4303 million | 8.9 million   | 15 hours, 16 k40s |
| Medium-Budget MoE Model | 31.3            | 4313 million | 33.8 million  | 17 hours, 32 k40s |
| High-Budget MoE Model   | 28.0            | 4371 million | 142.7 million | 47 hours, 32 k40s |

**Table 1:** Summary of high-capacity MoE-augmented models with varying computational budgets, vs. best previously published results (Jozefowicz et al., 2016). Details in Appendix C.

Figure 2: Model comparison on 1-Billion-Word Language-Modeling Benchmark. On the left, we
plot test perplexity as a function of model capacity for models with similar computational budgets of approximately 8-million-ops-per-timestep. On the right, we plot test perplexity as a function of computational budget. The top line represents the LSTM models from (Jozefowicz et al., 2016). The bottom line represents 4-billion parameter MoE models with different computational budgets.   

**Dataset:** This dataset, introduced by ([Chelba et al., 2013](#ref-chelba2013one)), consists of shuffled unique sentences from news articles, totaling approximately 829 million words, with a vocabulary of 793,471 words.

**Previous State-of-the-Art:** The best previously published results ([Jozefowicz et al., 2016](#ref-jozefowicz2016exploring)) use models consisting of one or more stacked Long Short-Term Memory (LSTM) layers ([Hochreiter & Schmidhuber, 1997](#ref-hochreiter1997long); [Gers et al., 2000](#ref-gers2000recurrent)). The number of parameters in the LSTM layers of these models varies from 2 million to 151 million. Quality increases greatly with parameter count, as do computational costs. Results for these models form the top line of Figure 2-right.

**MoE Models:** Our models consist of two stacked LSTM layers with a MoE layer between them (see Figure 1). We vary the sizes of the layers and the number of experts. For full details on model architecture, training regimen, additional baselines, and results, see Appendix C.

**Low Computation, Varied Capacity:** To investigate the effects of adding capacity, we trained a series of MoE models all with roughly equal computational costs: about 8 million multiply-and-adds per training example per timestep in the forwards pass, excluding the softmax layer. We call this metric (ops/timestep). We trained models with flat MoEs containing 4, 32, and 256 experts, and models with hierarchical MoEs containing 256, 1024, and 4096 experts. Each expert had about 1 million parameters. For all the MoE layers, 4 experts were active per input.

The results of these models are shown in Figure 2-left. The model with 4 always-active experts performed (unsurprisingly) similarly to the computationally-matched baseline models, while the largest of the models (4096 experts) achieved an impressive 24% lower perplexity on the test set.

**Varied Computation, High Capacity:** In addition to the largest model from the previous section, we trained two more MoE models with similarly high capacity (4 billion parameters), but higher computation budgets. These models had larger LSTMs and fewer but larger experts. Details can be found in Appendix C.2. Results of these three models form the bottom line of Figure 2-right.

Table 1 compares the results of these models to the best previously-published result on this dataset. Even the fastest of these models beats the best published result (when controlling for the number of training epochs), despite requiring only 6% of the computation.

**Computational Efficiency:** We trained our models using TensorFlow ([Abadi et al., 2016](#ref-abadi2016tensorflow)) on clusters containing 16-32 Tesla K40 GPUs. For each of our models, we determine computational efficiency in TFLOPS/GPU by dividing the number of floating point operations required to process one training batch by the observed step time and the number of GPUs in the cluster. The operation counts used here are higher than the ones we report in our ops/timestep numbers in that we include the backward pass, we include the importance-sampling-based training of the softmax layer, and we count a multiply-and-add as two separate operations. For all of our MoE models, the floating point operations involved in the experts represent between 37% and 46% of the total.

For our baseline models without MoE, observed computational efficiency ranged from 1.07-1.29 TFLOPS/GPU. For our low-computation MoE models, computation efficiency ranged from 0.74-0.90 TFLOPS/GPU, except for the 4-expert model which did not make full use of the available parallelism. Our highest-computation MoE model was more efficient at 1.56 TFLOPS/GPU, likely due to the larger matrices. These numbers represent a significant fraction of the theoretical maximum of 4.29 TFLOPS/GPU claimed by NVIDIA. Detailed results are in Appendix C, Table 7.

### 5.2 100 BILLION WORD GOOGLE NEWS CORPUS  

Figure 3: Language modeling on a 100 billion word corpus. Models have similar computational budgets (8 million ops/timestep).  

On the 1-billion-word corpus, adding additional capacity seems to produce diminishing returns as the number of parameters in the MoE layer exceeds 1 billion, as can be seen in Figure 2-left. We hypothesized that for a larger training set, even higher capacities would produce significant quality improvements.

We constructed a similar training set consisting of shuffled unique sentences from Google's internal news corpus, totaling roughly 100 billion words. Similarly to the previous section, we tested a series of models with similar computational costs of about 8 million ops/timestep. In addition to a baseline LSTM model, we trained models augmented with MoE layers containing 32, 256, 1024, 4096, 16384, 65536, and 131072 experts. This corresponds to up to 137 billion parameters in the MoE layer. Details on architecture, training, and results are given in Appendix D.

**Results:** Figure 3 shows test perplexity as a function of capacity after training on 10 billion words (top line) and 100 billion words (bottom line). When training over the full 100 billion words, test perplexity improves significantly up to 65536 experts (68 billion parameters), dropping 39% lower than the computationally matched baseline, but degrades at 131072 experts, possibly a result of too much sparsity. The widening gap between the two lines demonstrates (unsurprisingly) that increased model capacity helps more on larger training sets.

Even at 65536 experts (99.994% layer sparsity), computational efficiency for the model stays at a respectable 0.72 TFLOPS/GPU.

### 5.3 MACHINE TRANSLATION (SINGLE LANGUAGE PAIR)

| Model                                       | Test Perplexity | BLEU  | #Parameters | Time                 |
| ------------------------------------------- | --------------- | ----- | ----------- | -------------------- |
| MoE with 2048 Experts                       | 2.69            | 40.35 | 85M         | 8.7B, 3 days/64 k40s |
| MoE with 2048 Experts (longer training)     | 2.63            | 40.56 | 85M         | 8.7B, 6 days/64 k40s |
| GNMT (Wu et al., 2016)                      | 2.79            | 39.22 | 214M        | 278M, 6 days/96 k80s |
| GNMT+RL (Wu et al., 2016)                   | 2.96            | 39.92 | 214M        | 278M, 6 days/96 k80s |
| PBMT (Durrani et al., 2014)                 | 37.0            | -     | -           | -                    |
| LSTM (6-layer) (Luong et al., 2015b)        | 31.5            | -     | -           | -                    |
| LSTM (6-layer+PosUnk) (Luong et al., 2015b) | 33.1            | -     | -           | -                    |
| DeepAtt (Zhou et al., 2016)                 | 37.7            | -     | -           | -                    |
| DeepAtt+PosUnk (Zhou et al., 2016)          | 39.2            | -     | -           | -                    |

**Table 2:** Results on WMT’14 En! Fr newstest2014 (bold values represent best results).

| Model                       | Test Perplexity | BLEU  | #Parameters | Time                |
| --------------------------- | --------------- | ----- | ----------- | ------------------- |
| MoE with 2048 Experts       | 4.64            | 26.03 | 85M         | 8.7B, 1 day/64 k40s |
| GNMT (Wu et al., 2016)      | 5.25            | 24.91 | 214M        | 278M, 1 day/96 k80s |
| GNMT+RL (Wu et al., 2016)   | 8.08            | 24.66 | 214M        | 278M, 1 day/96 k80s |
| PBMT (Durrani et al., 2014) | 20.7            | -     | -           | -                   |
| DeepAtt (Zhou et al., 2016) | 20.6            | -     | -           | -                   |

**Table 3:** Results on WMT’14 En ! De newstest2014 (bold values represent best results).

| Model                  | Eval Perplexity | Eval BLEU | Test Perplexity | Test BLEU | #Parameters | Time                 |
| ---------------------- | --------------- | --------- | --------------- | --------- | ----------- | -------------------- |
| MoE with 2048 Experts  | 2.60            | 37.27     | 2.69            | 36.57     | 85M         | 8.7B, 1 day/64 k40s  |
| GNMT (Wu et al., 2016) | 2.78            | 35.80     | 2.87            | 35.56     | 214M        | 278M, 6 days/96 k80s |

**Table 4:** Results on the Google Production En! Fr dataset (bold values represent best results).

**Model Architecture:** Our model was a modified version of the GNMT model described in ([Wu et al., 2016](#ref-wu2016google)). To reduce computation, we decreased the number of LSTM layers in the encoder and decoder from 9 and 8 to 3 and 2 respectively. We inserted MoE layers in both the encoder (between layers 2 and 3) and the decoder (between layers 1 and 2). Each MoE layer contained up to 2048 experts each with about two million parameters, adding a total of about 8 billion parameters to the models. Further details on model architecture, testing procedure, and results can be found in Appendix E.

**Datasets:** We benchmarked our method on the WMT'14 En→Fr and En→De corpora, whose training sets have 36M sentence pairs and 5M sentence pairs, respectively. The experimental protocols were also similar to those in ([Wu et al., 2016](#ref-wu2016google)): newstest2014 was used as the test set to compare against previous work ([Luong et al., 2015](#ref-luong2015addressing); [Zhou et al., 2016](#ref-zhou2016deep); [Wu et al., 2016](#ref-w

u2016google)), while the combination of newstest2012 and newstest2013 was used as the development set. We also tested the same model on Google's Production English to French data.

**Results:** Tables 2, 3, and 4 show the results of our largest models, compared with published results. Our approach achieved BLEU scores of 40.56 and 26.03 on the WMT'14 En→Fr and En→De benchmarks. As our models did not use RL refinement, these results constitute significant gains of 1.34 and 1.12 BLEU score on top of the strong baselines in ([Wu et al., 2016](#ref-wu2016google)). The perplexity scores are also better. On the Google Production dataset, our model achieved 1.01 higher test BLEU score even after training for only one sixth of the time.

### 5.4 MULTILINGUAL MACHINE TRANSLATION
**Dataset:** ([Johnson et al., 2016](#ref-johnson2016google)) train a single GNMT ([Wu et al., 2016](#ref-wu2016google)) model on a very large combined dataset of twelve language pairs. Results are somewhat worse than those for 12 separately trained single-pair GNMT models. This is not surprising, given that the twelve models have 12 times the capacity and twelve times the aggregate training of the one model. We repeat this experiment with a single MoE-augmented model. See Appendix E for details on model architecture.

We train our model on the same dataset as ([Johnson et al., 2016](#ref-johnson2016google)) and process the same number of training examples (about 3 billion sentence pairs). Our training time was shorter due to the lower computational budget of our model.

**Results:** Results for the single-pair GNMT models, the multilingual GNMT model, and the multilingual MoE model are given in Table 5. The MoE model achieves 19% lower perplexity on the dev set than the multilingual GNMT model. On BLEU score, the MoE model significantly beats the multilingual GNMT model on 11 of the 12 language pairs (by as much as 5.84 points), and even beats the monolingual GNMT models on 8 of 12 language pairs. The poor performance on English→Korean seems to be a result of severe overtraining, as for the rarer language pairs a small number of real examples were highly oversampled in the training corpus.

|                                | GNMT-Mono    | GNMT-Multi       | MoE-Multi        | MoE-Multi vs. GNMT-Multi |
| ------------------------------ | ------------ | ---------------- | ---------------- | ------------------------ |
| Parameters                     | 278M / model | 278M             | 8.7B             |                          |
| ops/timestep                   | 212M         | 212M             | 102M             |                          |
| training time                  | various      | 21 days, 96 k20s | 12 days, 64 k40s |                          |
| Perplexity (dev)               | 4.14         | 3.35             | -19%             |                          |
| French ! English Test BLEU     | 36.47        | 34.40            | 37.46            | +3.06                    |
| German ! English Test BLEU     | 31.77        | 31.17            | 34.80            | +3.63                    |
| Japanese ! English Test BLEU   | 23.41        | 21.62            | 25.91            | +4.29                    |
| Korean ! English Test BLEU     | 25.42        | 22.87            | 28.71            | +5.84                    |
| Portuguese ! English Test BLEU | 44.40        | 42.53            | 46.13            | +3.60                    |
| Spanish ! English Test BLEU    | 38.00        | 36.04            | 39.39            | +3.35                    |
| English ! French Test BLEU     | 35.37        | 34.00            | 36.59            | +2.59                    |
| English ! German Test BLEU     | 26.43        | 23.15            | 24.53            | +1.38                    |
| English ! Japanese Test BLEU   | 23.66        | 21.10            | 22.78            | +1.68                    |
| English ! Korean Test BLEU     | 19.75        | 18.41            | 16.62            | -1.79                    |
| English ! Portuguese Test BLEU | 38.40        | 37.35            | 37.90            | +0.55                    |
| English ! Spanish Test BLEU    | 34.50        | 34.25            | 36.21            | +1.96                    |

**Table 5:** Multilingual Machine Translation (bold values represent best results)

## 6 CONCLUSION
This work is the first to demonstrate major wins from conditional computation in deep networks. We carefully identified the design considerations and challenges of conditional computing and addressed them with a combination of algorithmic and engineering solutions. While we focused on text, conditional computation may help in other domains as well, provided sufficiently large training sets. We look forward to seeing many novel implementations and applications of conditional computation in the years to come.



REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Gregory S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian J. Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Józefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Gordon Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul A. Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda B. Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous distributed systems. CoRR, abs/1603.04467, 2016. URL http://arxiv.org/abs/1603.04467.
Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. CoRR, abs/1611.06194, 2016. URL http://arxiv.org/abs/1611.06194.
A. Almahairi, N. Ballas, T. Cooijmans, Y. Zheng, H. Larochelle, and A. Courville. Dynamic Capacity Networks. ArXiv e-prints, November 2015.
Dario Amodei, Rishita Anubhai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Jingdong Chen, Mike Chrzanowski, Adam Coates, Greg Diamos, Erich Elsen, Jesse Engel, Linxi Fan, Christopher Fougner, Tony Han, Awni Y. Hannun, Billy Jun, Patrick LeGresley, Libby Lin, Sharan Narang, Andrew Y. Ng, Sherjil Ozair, Ryan Prenger, Jonathan Raiman, Sanjeev Satheesh, David Seetapun, Shubho Sengupta, Yi Wang, Zhiqian Wang, Chong Wang, Bo Xiao, Dani Yogatama, Jun Zhan, and Zhenyao Zhu. Deep speech 2: End-to-end speech recognition in english and mandarin. arXiv preprint arXiv:1512.02595, 2015.
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.
Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models. arXiv preprint arXiv:1511.06297, 2015.
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arX

iv:1312.3005, 2013.
K. Cho and Y. Bengio. Exponentially Increasing the Capacity-to-Computation Ratio for Conditional Computation in Deep Learning. ArXiv e-prints, June 2014.
Ronan Collobert, Samy Bengio, and Yoshua Bengio. A parallel mixture of SVMs for very large scale problems. Neural Computing, 2002.
Andrew Davis and Itamar Arel. Low-rank approximations for conditional feedforward computation in deep neural networks. arXiv preprint arXiv:1312.4461, 2013.
Marc Peter Deisenroth and Jun Wei Ng. Distributed Gaussian processes. In ICML, 2015.
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization, 2010.
Nadir Durrani, Barry Haddow, Philipp Koehn, and Kenneth Heafield. Edinburgh’s phrase-based machine translation systems for wmt-14. In Proceedings of the Ninth Workshop on Statistical Machine Translation, 2014.
David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314, 2013.
Ekaterina Garmash and Christof Monz. Ensemble learning for multi-source neural machine translation. In staff.science.uva.nl/c.monz, 2016
Felix A. Gers, Jürgen A. Schmidhuber, and Fred A. Cummins. Learning to forget: Continual prediction with lstm. Neural Computation, 2000.
Audrunas Gruslys, Rémi Munos, Ivo Danihelka, Marc Lanctot, and Alex Graves. Memory-efficient backpropagation through time. CoRR, abs/1606.03401, 2016. URL http://arxiv.org/abs/1606.03401.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. IEEE Conference on Computer Vision and Pattern Recognition, 2015.
Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N. Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal Processing Magazine, 2012.
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 1997.
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.
Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. Neural Computing, 1991.
Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda B. Viégas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’s multilingual neural machine translation system: Enabling zero-shot translation. CoRR, abs/1611.04558, 2016. URL http://arxiv.org/abs/1611.04558.
Michael I. Jordan and Robert A. Jacobs.

 Hierarchical mixtures of experts and the EM algorithm. Neural Computing, 1994.
Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
Reinhard Kneser and Hermann Ney. Improved backingoff for m-gram language modeling., 1995.
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.
Quoc V. Le, Marc’Aurelio Ranzato, Rajat Monga, Matthieu Devin, Kai Chen, Greg S. Corrado, Jeffrey Dean, and Andrew Y. Ng. Building high-level features using large scale unsupervised learning. In ICML, 2012.
Patrick Gallinari Ludovic Denoyer. Deep sequential neural network. arXiv preprint arXiv:1410.0510, 2014.
Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural machine translation. EMNLP, 2015a.
Minh-Thang Luong, Ilya Sutskever, Quoc V. Le, Oriol Vinyals, and Wojciech Zaremba. Addressing the rare word problem in neural machine translation. ACL, 2015b.
Carl Edward Rasmussen and Zoubin Ghahramani. Infinite mixtures of Gaussian process experts. NIPS, 2002.
Hasim Sak, Andrew W Senior, and Françoise Beaufays. Long short-term memory recurrent neural network architectures for large scale acoustic modeling. In INTERSPEECH, pp. 338–342, 2014.
Mike Schuster and Kaisuke Nakajima. Japanese and Korean voice search. ICASSP, 2012.
Babak Shahbaba and Radford Neal. Nonlinear models using dirichlet process mixtures. JMLR, 2009.
Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In NIPS, 2014.
Lucas Theis and Matthias Bethge. Generative image modeling using spatial LSTMs. In NIPS, 2015.
Volker Tresp. Mixtures of Gaussian Processes. In NIPS, 2001.
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.
Bangpeng Yao, Dirk Walther, Diane Beck, and Li Fei-fei. Hierarchical mixture of classification experts unc

overs interactions between brain regions. In NIPS. 2009.
Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. arXiv preprint arXiv:1409.2329, 2014.
Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. arXiv preprint arXiv:1606.04199, 2016.
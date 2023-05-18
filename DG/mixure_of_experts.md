## ABSTRACT

Human visual perception can easily generalize to out-of-distributed visual data, which is far beyond the capability of modern machine learning models. Domain generalization (DG) aims to close this gap, with existing DG methods mainly focusing on the loss function design. In this paper, we propose to explore an orthogonal direction, i.e., the design of the backbone architecture. It is motivated by an empirical finding that transformer-based models trained with empirical risk minimization (ERM) outperform CNN-based models employing state-of-the-art (SOTA) DG algorithms on multiple DG datasets. We develop a formal framework to characterize a network's robustness to distribution shifts by studying its architecture's alignment with the correlations in the dataset. This analysis guides us to propose a novel DG model built upon vision transformers, namely Generalizable Mixture-of-Experts (GMoE). Extensive experiments on DomainBed demonstrate that GMoE trained with ERM outperforms SOTA DG baselines by a large margin. Moreover, GMoE is complementary to existing DG methods and its performance is substantially improved when trained with DG algorithms.

## 1 INTRODUCTION

### 1.1 MOTIVATIONS

Generalizing to out-of-distribution (OOD) data is an innate ability for human vision, but highly challenging for machine learning models (Recht et al., 2019; Geirhos et al., 2021; Ma et al., 2022). Domain generalization (DG) is one approach to address this problem, which encourages models to be resilient under various distribution shifts such as background, lighting, texture, shape, and geographic/demographic attributes.

From the perspective of representation learning, there are several paradigms towards this goal, including domain alignment (Ganin et al., 2016; Hoffman et al., 2018), invariant causality prediction (Arjovsky et al., 2019; Krueger et al., 2021), meta-learning (Bui et al., 2021; Zhang et al., 2021c), ensemble learning (Mancini et al., 2018; Cha et al., 2021b), and feature disentanglement (Wang et al., 2021; Zhang et al., 2021b). The most popular approach to implementing these ideas is to design a specific loss function. For example, DANN (Ganin et al., 2016) aligns domain distributions by adversarial losses. Invariant causal prediction can be enforced by a penalty of gradient norm (Arjovsky et al., 2019) or variance of training risks (Krueger et al., 2021). Meta-learning and domain-specific loss functions (Bui et al., 2021; Zhang et al., 2021c) have also been employed to enhance the performance. Recent studies have shown that these approaches improve ERM and achieve promising results on large-scale DG datasets (Wiles et al., 2021).

Meanwhile, in various computer vision tasks, the innovations in backbone architectures play a pivotal role in performance boost and have attracted much attention (He et al., 2016; Hu et al., 2018; Liu et al., 2021). Additionally, it has been empirically demonstrated in Sivaprasad et al. (2021) that different CNN architectures have different performances on DG datasets. Inspired by these pioneering works, we conjecture that backbone architecture design would be promising for DG. To verify this intuition, we evaluate a transformer-based model and compare it with CNN-based architectures of equivalent computational overhead, as shown in Fig. 1(a). To our surprise, a vanilla ViT-S/16 (Dosovitskiy et al., 2021) trained with empirical risk

 minimization (ERM) outperforms ResNet-50 trained with SOTA DG algorithms (Cha et al., 2021b; Rame et al., 2021; Shi et al., 2021) on DomainNet, OfficeHome and VLCS datasets, despite the fact that both architectures have a similar number of parameters and enjoy close performance on in-distribution domains. We theoretically validate this effect based on the algorithmic alignment framework (Xu et al., 2020a; Li et al., 2021). We first prove that a network trained with the ERM loss function is more robust to distribution shifts if its architecture is more similar to the invariant correlation, where the similarity is formally measured by the alignment value defined in Xu et al. (2020a). On the contrary, a network is less robust if its architecture aligns with the spurious correlation. We then investigate the alignment between backbone architectures (i.e., convolutions and attentions) and the correlations in these datasets, which explains the superior performance of ViT-based methods.

To further improve the performance, our analysis indicates that we should exploit properties of invariant correlations in vision tasks and design network architectures to align with these properties. This requires an investigation that sits at the intersection of domain generalization and classic computer vision. In domain generalization, it is widely believed that the data are composed of some sets of attributes and distribution shifts of data are distribution shifts of these attributes (Wiles et al., 2021). The latent factorization model of these attributes is almost identical to the generative model of visual attributes in classic computer vision (Ferrari & Zisserman, 2007). To capture these diverse attributes, we propose a Generalizable Mixture-of-Experts (GMoE), which is built upon sparse mixture-of-experts (sparse MoEs) (Shazeer et al., 2017) and vision transformer (Dosovitskiy et al., 2021). The sparse MoEs were originally proposed as key enablers for extremely large, but efficient models (Fedus et al., 2022). By theoretical and empirical evidence, we demonstrate that MoEs are experts for processing visual attributes, leading to a better alignment with invariant correlations.

Based on our analysis, we modify the architecture of sparse MoEs to enhance their performance in DG. Extensive experiments demonstrate that GMoE achieves superior domain generalization performance both with and without DG algorithms.

### 1.2 CONTRIBUTIONS

In this paper, we formally investigate the impact of the backbone architecture on DG and propose to develop effective DG methods by backbone architecture design. Specifically, our main contributions are summarized as follows:

A Novel View of DG: In contrast to previous works, this paper initiates a formal exploration of the backbone architecture in DG. Based on algorithmic alignment (Xu et al., 2020a), we prove that a network is more robust to distribution shifts if its architecture aligns with the invariant correlation, whereas less robust if its architecture aligns with spurious correlation. The theorems are verified on synthetic and real datasets.

A Novel Model for DG: Based on our theoretical analysis, we propose Generalizable Mixture-of-Experts (GMoE) and prove that it enjoys a better alignment than vision transformers. GMoE is built upon sparse mixture-of-experts (Shazeer et al., 2017) and vision transformer (Dosovitskiy et al., 2021), with a theory-guided performance enhancement for DG.

Excellent Performance: We validate GMoE's performance on all 8 large-scale datasets of DomainBed. Remarkably, GMoE trained with ERM achieves SOTA performance on 7 datasets in the train-validation setting and on 8 datasets in the leave

-one-domain-out setting. Furthermore, the GMoE trained with DG algorithms achieves better performance than GMoE trained with ERM.

## 2 PRELIMINARIES

### 2.1 NOTATIONS

Throughout this paper, $a$, $\mathbf{a}$, $\mathbf{A}$ stand for a scalar, a column vector, a matrix, respectively. $O(\cdot)$ and $!(\cdot)$ are asymptotic notations. We denote the training dataset, training distribution, test dataset, and test distribution as $E_{\text{tr}}$, $D_{\text{tr}}$, $E_{\text{te}}$, and $D_{\text{te}}$, respectively.

### 2.2 ATTRIBUTE FACTORIZATION

The attribute factorization (Wiles et al., 2021) is a realistic generative model under distribution shifts. Consider a joint distribution of the input $x$ and corresponding attributes $a_1, \ldots, a_K$ (denoted as $a_{1:K}$) with $a_i \in \mathcal{A}_i$, where $\mathcal{A}_i$ is a finite set. The label can depend on one or multiple attributes. Denote the latent factor as $z$, the data generation process is given by
$$
z \sim p(z), \quad a_i \sim p(a_i|z), \quad x \sim p(x|z), \quad p(a_{1:K}, x) = p(a_{1:K}) \int p(x|z)p(z|a_{1:K}) \, dz.
$$
The distribution shift arises if different marginal distributions of the attributes are given but they share the same conditional generative process. Specifically, we have $p_{\text{train}}(a_{1:K}) \neq p_{\text{test}}(a_{1:K})$, but the generative model in equation 1 is shared across the distributions, i.e., we have $p_{\text{test}}(a_{1:K}, x) = p_{\text{test}}(a_{1:K}) \int p(x|z)p(z|a_{1:K}) \, dz$ and similarly for $p_{\text{train}}$. The above description is abstract and we will illustrate with an example.

Example 1. (DSPRITES (Matthey et al., 2017)) Consider $\mathcal{A}_1 = \{\text{red}, \text{blue}\}$ and $\mathcal{A}_2 = \{\text{ellipse}, \text{square}\}$. The target task is a shape classification task, where the label depends on attribute $a_2$. In the training dataset, 90% of ellipses are red and 50% of squares are blue, while in the test dataset all the attributes are distributed uniformly. As the majority of ellipses are red, the classifier will use color as a shortcut in the training dataset, which is so-called geometric skews (Nagarajan et al., 2020). However, this shortcut does not exist in the test dataset and the network fails to generalize.

In classic computer vision, the attributes are named visual attributes and they follow a similar data generation process (Ferrari & Zisserman, 2007). We shall discuss them in detail in Section 4.2.

### 2.3 ALGORITHMIC ALIGNMENT

We first introduce algorithmic alignment, which characterizes the easiness of IID reasoning tasks by measuring the similarity between the backbone architecture and target function. The alignment is formally defined as the following.

Definition 1. (

Alignment; (Xu et al., 2020a)) Let $N$ denote a neural network with $n$ modules $f_1, \ldots, f_n$ and assume that a target function for learning $y = g(x)$ can be decomposed into $n$ functions $f_1, \ldots, f_n$. The network $N$ aligns with the target function if replacing $N_i$ with $f_i$, it outputs the same value as algorithm $g$. The alignment value between $N$ and $f$ is defined as
$$
\text{Alignment}(N, f, \varepsilon, \delta) := n \cdot \max_i M(f_i, N_i, \varepsilon, \delta),
$$
where $M(f_i, N_i, \varepsilon, \delta)$ denotes the sample complexity measure for $N_i$ to learn $f_i$ with $\varepsilon$ precision at failure probability $\delta$ under a learning algorithm when the training distribution is the same as the test distribution.

In Definition 1, the original task is to learn $f$, which is a challenging problem. Intuitively, if we could find a backbone architecture that is suitable for this task, it helps to break the original task into simpler sub-tasks, i.e., to learn $f_1, \ldots, f_n$ instead. Under the assumptions of algorithmic alignment (Xu et al., 2020a), $f$ can be learned optimally if the sub-task $f_1, \ldots, f_n$ can be learned optimally. Thus, a good alignment makes the target task easier to learn, and thus improves IID generalization, which is given in Theorem 3 in Appendix B.1. In Section 3, we extend this framework to the DG setting.

## 3 ON THE IMPORTANCE OF NEURAL ARCHITECTURE FOR DOMAIN GENERALIZATION

In this section, we investigate the impact of the backbone architecture on DG, from a motivating example to a formal framework.

### 3.1 A MOTIVATING EXAMPLE: CNNS VERSUS VISION TRANSFORMERS

We adopt DomainBed (Gulrajani & Lopez-Paz, 2021) as the benchmark, which implements SOTA DG algorithms with ResNet50 as the backbone. We test the performance of ViT trained with ERM on this benchmark, without applying any DG method. The results are shown in Fig. 1(a). To our surprise, ViT trained with ERM already outperforms CNNs with SOTA DG algorithms on several datasets, which indicates that the selection of the backbone architecture is potentially more important than the loss function in DG. In the remaining of this article, we will obtain a theoretical understanding of this phenomenon and improve ViT for DG by modifying its architecture.

### 3.2 UNDERSTANDING FROM A THEORETICAL PERSPECTIVE

The above experiment leads to an intriguing question: how does the backbone architecture impact the network's performance in DG? In this subsection, we endeavor to answer this question by extending the algorithmic alignment framework (Xu et al., 2020a) to the DG setting.

To have a tractable analysis for nonlinear function approximation, we first make an assumption on the distribution shift.

Assumption 1. Denote $N_1$ as the first module of the network (including one or multiple layers) of the network. Let $p_{\text{train},N_1}(s)$ and $p_{\text{test},N_1}(s)$ denote the probability density functions of features after $N_1$. Assume that the support of the training feature distribution covers

 that of the test feature distribution, i.e., $\max_s p_{\text{test},N_1}(s) \leq p_{\text{train},N_1}(s) \leq C$, where $C$ is a constant independent of the number of training samples.

Remark 1. (Interpretations of Assumption 1) This condition is practical in DG, especially when we have a pretrained model for disentanglement (e.g., on DomainBed (Gulrajani & Lopez-Paz, 2021)). In Example 1, the training distribution and test distribution have the same support. In DomainNet, although the elephants in quickdraw are visually different from the elephants in other domains, the quickdraw picture's attributes/features (e.g., big ears and long noise) are covered in the training domains. From a technical perspective, it is impossible for networks trained with gradient descent to approximate a wide range of nonlinear functions in the out-of-support regime (Xu et al., 2020b). Thus, this condition is necessary if we do not impose strong constraints on the target functions.

We define several key concepts in DG. The target function is an invariant correlation across the training and test datasets. For simplicity, we assume that the labels are noise-free.

Assumption 2. (Invariant correlation) Assume there exists a function $g_c$ such that for training data, we have $g_c(N_1(x)) = y$, $\forall x \in E_{\text{tr}}$, and for test data, we have $\mathbb{P}_{D_{\text{te}}}(|g_c(N_1(x)) - y| \leq \varepsilon) > 1 - \delta$.

We then introduce the spurious correlation (Wiles et al., 2021), i.e., some attributes are correlated with labels in the training dataset but not in the test dataset. The spurious correlation exists only if the training distribution differs from the test distribution and this distinguishes DG from classic PAC-learning settings (Xu et al., 2020a).

Assumption 3. (Spurious correlation) Assume there exists a function $g_s$ such that for training data $x \in E_{\text{tr}}$, we have $g_s(N_1(x)) = y$, and for test data, we have $\mathbb{P}_{D_{\text{te}}}(|g_s(N_1(x)) - y| > \text{!}(\varepsilon)) > 1 - \delta$.

The next theorem extends algorithmic alignment from IID generalization (Theorem 3) to DG.

Theorem 1. (Impact of Backbone Architecture in Domain Generalization) Denote $N_0 = N_2, \ldots, N_n$. Assuming we train the neural network with ERM, and Assumption 1, 2, 3 hold, we have the following statements:

1. If Alignment($N_0, g_c, \varepsilon, \delta) \leq |E_{\text{tr}}|$, we have $\mathbb{P}_{D_{\text{te}}}(|N(x) - y| \leq O(\varepsilon)) > 1 - O(\delta)$.
2. If Alignment($N_0, g_s, \varepsilon, \delta) \leq |E_{\text{tr}}|$, we have $\mathbb{P}_{D_{\text{te}}}(|N(x) - y| > \text{!}(\varepsilon)) > 1 - O(\delta)$.

Remark 

2. (Interpretations of Theorem 1) By choosing a sufficiently small $\varepsilon$, only one of Alignment($N_0, g_c, \varepsilon, \delta) \leq |E_{\text{tr}}|$ and Alignment($N_0, g_s, \varepsilon, \delta) \leq |E_{\text{tr}}|$ holds. Thus, Theorem 1 shows that the networks aligned with invariant correlations are more robust to distribution shifts. In Appendix B.2, we build a synthetic dataset that satisfies all the assumptions. The experimental results exactly match Theorem 1. In practical datasets, the labels may have colored noise, which depends on the spurious correlation. Under such circumstances, the network should rely on multiple correlations to fit the label well and the correlation that best aligns with the network will have the major impact on its performance. Please refer to Appendix B.1 for the proof.

ViT with ERM versus CNNs with DG algorithms We now use Theorem 1 to explain the experiments in the last subsection. The first condition of Theorem 1 shows that if the neural architecture aligns with the invariant correlation, ERM is sufficient to achieve good performance. In some domains of OfficeHome or DomainNet, the shape attribute has an invariant correlation with the label, illustrated in Fig. 1(b). On the contrary, a spurious correlation exists between the attribute texture and the label. According to the analysis in Park & Kim (2022), multi-head attentions (MHA) are low-pass filters with a shape bias while convolutions are high-pass filters with a texture bias. As a result, a ViT simply trained with ERM can outperform CNNs trained with SOTA DG algorithms.

To improve ViT's performance, Theorem 1 suggests that we should exploit the properties of invariant correlations. In image recognition, objects are described by functional parts (e.g., visual attributes), with words associated with them (Zhou et al., 2014). The configuration of the objects has a large degree of freedom, resulting in different shapes among one category. Therefore, functional parts are more fundamental than shape in image recognition and we will develop backbone architectures to capture them in the next section.

## 4 GENERALIZABLE MIXTURE-OF-EXPERTS FOR DOMAIN GENERALIZATION

In this section, we propose Generalizable Mixture-of-Experts (GMoE) for domain generalization, supported by effective neural architecture design and theoretical analysis.

### 4.1 MIXTURE-OF-EXPERTS LAYER

In this subsection, we introduce the mixture-of-experts (MoE) layer, which is an essential component of GMoE. One ViT layer is composed of an MHA and an FFN. In the MoE layer, the FFN is replaced by mixture-of-experts and each expert is implemented by an FFN (Shazeer et al., 2017). Denoting the output of the MHA as x, the output of the MoE layer with N experts is given by

$$f_{\text{MoE}}(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x) = \sum_{i=1}^{N} \text{TOP}_k(\text{Softmax}(W x)) \cdot W_{\text{FFN}2}^{(i)}(\phi(W_{\text{FFN}1}^{(i)} x))$$

where $W$ is the learnable parameter for the gate, $W_{\text{FFN}1}^{(i)}$ and $W_{\text{FFN}2}^{(i)}$ are learnable parameters for the i-th expert, $\phi(\cdot)$ is a nonlinear activation function, and $\text{TOP}_k(\cdot)$ operation is a one-hot embedding that sets all other elements in the output vector as zero except for the elements with the largest k values where k is a hyperparameter. Given $x_{\text{in}}$ as the input of the MoE layer, the update is given by

$$x = f_{\text{MHA}}(\text{LN}(x_{\text{in}})) + x_{\text{in}} \quad \text{and} \quad x_{\text{out}} = f_{\text{MoE}}(\text{LN}(x)) + x$$

where $f_{\text{MHA}}$ is the MHA layer, LN represents layer normalization, and $x_{\text{out}}$ is the output of the MoE layer.

### 4.2 VISUAL ATTRIBUTES, CONDITIONAL STATEMENTS, AND SPARSE MOES 

In real world image data, the label depends on multiple attributes. Capturing diverse visual attributes is especially important for DG. For example, the definition of an elephant in the Oxford dictionary is "a very large animal with thick grey skin, large ears, two curved outer teeth called tusks, and a long nose called a trunk". The definition involves three shape attributes (i.e., large ears, curved outer teeth, and a long nose) and one texture attribute (i.e., thick grey skin). In the IID ImageNet task, using the most discriminative attribute, i.e., the thick grey skin, is sufficient to achieve high accuracy (Geirhos et al., 2018). However, in DomainNet, elephants no longer have grey skins while the long nose and big ears are preserved and the network relying on grey skins will fail to generalize.

**Algorithm 1: Conditional Statements**

Define intervals
$$I_i \subset \mathbb{R}; \quad i = 1, \ldots, M$$

Define functions
$$h_i, \quad i = 1, \ldots, M+1$$

switch $h_1

(x)$ do
    if $h_1(x) \in I_i$ then
        apply $h_{i+1}$ to $x$

The conditional statement (i.e., IF/ELSE in programming), as shown in Algorithm 1, is a powerful tool to efficiently capture the visual attributes and combine them for DG. Suppose we train the network to recognize the elephants on DomainNet, as illustrated in the first row of Fig. 1(b). For the elephants in different domains, shape and texture vary significantly while the visual attributes (large ears, curved teeth, long nose) are invariant across all the domains. Equipped with conditional statements, the recognition of the elephants can be expressed as "if an animal has large ears, two curved outer teeth, and a long nose, then it is an elephant". Then the subtasks are to recognize these visual attributes, which also requires conditional statements. For example, the operation for "curved outer teeth" is that "if the patch belongs to the teeth, then we apply a shape filter to it". In literature, the MoE layer is considered an effective approach to implement conditional computations (Shazeer et al., 2017; Riquelme et al., 2021). We formalize this intuition in the next theorem.

**Theorem 2.** An MoE module in equation 3 with N experts and $k = 1$ aligns with the conditional statements in Algorithm 1 with

$$\text{Alignment} = \begin{cases} 
(N + 1) \cdot \max \left( M^*P, M(G, h_1, \varepsilon, \delta) \right) & \text{if } N < M \\
(N + 1) \cdot \max \left( \max_{i \in \{1, \ldots, \max(1, \ldots, M)\}} M(f_{\text{FFN}}^{(i)}, h_{i+1}, \varepsilon, \delta), M(G, h_1, \varepsilon, \delta) \right) & \text{if } N \geq M
\end{cases}$$

where $M(\cdot, \cdot, \cdot, \cdot)$ is defined in Definition 1, and $M^*P$ is the optimal objective value of the following optimization problem:

$$\text{P: minimize } I_1, \ldots, I_N \max_{i \in \{1, \ldots, N\}} M(f_{\text{FFN}}^{(i)}, ([1_{I_j}]_{j \in I_i} \circ h_1)^T [h_j]_{j \in I_i}, \varepsilon, \delta) \text{ subject to } \sum_{i=1}^{N} I_i = \{2, 3, \ldots, M+1\}$$

where $1_{I_j}$ is the indicator function on interval $I_j$.

**Remark 3.** (Interpretations of Theorem 2) In algorithmic alignment, the network better aligns with the algorithm if the alignment value in equation 2 is lower. The alignment value between MoE and conditional statements depends on the product of $N + 1$ and a sample complexity term. When we increase the number of experts $N$, the alignment value first decreases as multiple experts decompose the original conditional statements into several simpler tasks. As we further increase $N$, the alignment value increases because of the factor $N + 1$ in the product. Therefore, the MoE align

s better with conditional statements than with the original FFN (i.e., $N = 1$). In addition, to minimize equation 5, similar conditional statements should be grouped together. By experiments in Section 5.4 and Appendix E.1, we find that sparse MoE layers are indeed experts for visual attributes, and similar visual attributes are handled by one expert. Please refer to Appendix B.3 for the proof.

### 4.3 ADAPTING MOE TO DOMAIN GENERALIZATION

In literature, there are several variants of MoE architectures, e.g., Riquelme et al. (2021); Fedus et al. (2022), and we should identify one for DG. By algorithmic alignment, in order to achieve a better generalization, the architecture of sparse MoEs should be designed to effectively handle visual attributes. In the following, we discuss our architecture design for this purpose.

**Routing scheme** Linear routers (i.e., equation 3) are often adopted in MoEs for vision tasks (Riquelme et al., 2021) while recent studies in NLP show that the cosine router achieves better performance in cross-lingual language tasks (Chi et al., 2022). For the cosine router, given input $x \in \mathbb{R}^d$, the embedding $W x \in \mathbb{R}^{de}$ is first projected onto a hypersphere, followed by multiplying a learned embedding $E \in \mathbb{R}^{de \times N}$. Specifically, the expression for the gate is given by

$$G(x) = \text{TOP}_k \left( \text{Softmax} \left( \tau_k \frac{E W x^T W x}{\|E\| \|W x\|} \right) \right)$$

where $\tau$ is a hyper-parameter. In the view of image processing, $E$ can be interpreted as the cookbook for visual attributes (Ferrari & Zisserman, 2007; Zhou et al., 2014) and the dot product between $E$ and $W x$ with L2 normalization is a matched filter. We opine that the linear router would face difficulty in DG. For example, the elephant image (and its all patches) in the Clipart domain is likely more similar to other images in the Clipart domain than in other domains. The issue can be alleviated with a codebook for visual attributes and matched filters for detecting them. Please refer to Appendix D.6 for the ablation study.

**Number of MoE layers** Every-two and last-two are two commonly adopted placement methods in existing MoE studies (Riquelme et al., 2021; Lepikhin et al., 2021). Specifically, every-two refers to replacing the even layer's FFN with MoE, and last-two refers to placing MoE at the last two even layers. For IID generalization, every-two often outperforms last-two (Riquelme et al., 2021). We argue that last-two is more suitable for DG as the conditional sentences for processing visual attributes are high-level. From experiments, we empirically find that last-two achieves better performance than every-two with fewer computations. Please refer to Appendix C.1 for more discussions and Appendix D.6 for the ablation study.

The overall backbone architecture of GMoE is shown in Fig. 2. To train diverse experts, we adopt the perturbation trick and load balance loss as in Riquelme et al. (2021). Due to space limitation, we leave them in Appendix C.4.

| Algorithm                                           | PACS       | VLCS       | OfficeHome | TerraInc   | DomainNet  |
| --------------------------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ERM (ResNet50) (Vapnik, 1991)                       | 85.7 ± 0.5 | 77.4 ± 0.3 | 67.5 ± 0.5 | 47.2 ± 0.4 | 41.2 ± 0.2 |
| IRM [ArXiv 20] (Arjovsky et al., 2019)              | 83.5 ± 0.8 | 78.5 ± 0.5 | 64.3 ± 2.2 | 47.6 ± 0.8 | 33.9 ± 2.8 |
| DANN [JMLR 16] (Ganin et al., 2016)                 | 84.6 ± 1.1 | 78.7 ± 0.3 | 68.6 ± 0.4 | 46.4 ± 0.8 | 41.8 ± 0.2 |
| CORAL [ECCV 16] (Sun & Saenko, 2016)                | 86.0 ± 0.2 | 77.7 ± 0.5 | 68.6 ± 0.4 | 46.4 ± 0.8 | 41.8 ± 0.2 |
| MMD [CVPR 18] (Li et al., 2018b)                    | 85.0 ± 0.2 | 76.7 ± 0.9 | 67.7 ± 0.1 | 42.2 ± 1.4 | 39.4 ± 0.8 |
| FISH [ICLR 22] (Shi et al., 2021)                   | 85.5 ± 0.3 | 77.8 ± 0.3 | 68.6 ± 0.4 | 45.1 ± 1.3 | 42.7 ± 0.2 |
| SWAD [NeurIPS 21] (Cha et al., 2021a)               | 88.1 ± 0.1 | 79.1 ± 0.1 | 70.6 ± 0.2 | 50.0 ± 0.3 | 46.5 ± 0.1 |
| Fishr [ICML 22] (Rame et al., 2021)                 | 85.5 ± 0.2 | 77.8 ± 0.2 | 68.6 ± 0.2 | 47.4 ± 1.6 | 41.7 ± 0.0 |
| MIRO [ECCV 22] (Cha et al., 2022)                   | 85.4 ± 0.4 | 79.0 ± 0.0 | 70.5 ± 0.4 | 50.4 ± 1.1 | 44.3 ± 0.2 |
| ERM (ViT-S/16) [ICLR 21] (Dosovitskiy et al., 2021) | 86.2 ± 0.1 | 79.7 ± 0.0 | 72.2 ± 0.4 | 42.0 ± 0.8 | 47.3 ± 0.2 |
| GMoE-S/16 (Ours)                                    | 88.1 ± 0.1 | 80.2 ± 0.2 | 74.2 ± 0.4 | 48.5 ± 0.4 | 48.7 ± 0.2 |

| Algorithms                                          | SVIRO      | Wilds-Camelyon | Wilds-FMOW |
| --------------------------------------------------- | ---------- | -------------- | ---------- |
| ERM (ResNet50) (Vapnik, 1991)                       | 85.7 ± 0.1 | 93.1 ± 0.2     | 40.6 ± 0.4 |
| ERM (ViT-S/16) [ICLR 21] (Dosovitskiy et al., 2021) | 89.6 ± 0.0 | 91.1 ± 0.1     | 44.8 ± 0.2 |
| GMoE-S/16 (Ours)                                    | 90.3 ± 0.1 | 93.7 ± 0.2     | 46.6 ± 0.4 |

**Table 1: Overall out-of-domain accuracies with train-validation selection criterion.** The best result is highlighted in bold. GMoE achieves the best performance on PACS, VLCS, OfficeHome, and DomainNet while ranking in the third-best on TerraIncognita.

## 5 EXPERIMENTAL RESULTS

In this section, we evaluate the performance of GMoE on large-scale DG datasets and present model analysis to understand GMoE.

### 5.1 DOMAINBED RESULTS

In this subsection, we evaluate GMoE on DomainBed (Gulrajani & Lopez-Paz, 2021) with 8 benchmark datasets: PACS, VLCS, OfficeHome, TerraIncognita, DomainNet, SVIRO, Wilds-Camelyon, and Wilds-FMOW. Detailed information on datasets and evaluation protocols are provided in Appendix D.1. The experiments are averaged over 3 runs as suggested in (Gulrajani & Lopez-Paz, 2021).

We present results in Table 1 with train-validation selection, which include baseline methods and recent DG algorithms and GMoE trained with ERM. The results demonstrate that GMoE without DG algorithms already outperforms counterparts on almost all the datasets. Meanwhile, GMoE has excellent performance in leave-one-domain-out criterion, and we leave the results in Appendix D.3 due to space limit. In the lower part of Table 1, we test our methods on three large-scale datasets: SVIRO, Wilds-Camelyon, and Wilds-FMOW. The three datasets capture real-world distribution shifts across a diverse range of domains. We adopt the data preprocessing and domain split in DomainBed. As there is no previous study conducting experiments on these datasets with DomainBed criterion, we only report the results of our methods, which reveal that GMoE outperforms the other two baselines.

### 5.2 GMOE WITH DG ALGORITHMS

GMoE's generalization ability comes from its internal backbone architecture, which is orthogonal to existing DG algorithms. This implies that the DG algorithms can be applied to improve the GMoE's performance. To validate this idea, we apply two DG algorithms to GMoE, including one modifying loss functions approaches (Fish) and one adopting model ensemble (Swad). The results in Table 2 demonstrate that adopting GMoE instead of ResNet-50 brings significant accuracy promotion to these DG algorithms. Experiments on GMoE with more DG algorithms are in Appendix D.4.

| Algorithm                | DomainNet |
| ------------------------ | --------- |
| GMoE                     | 48.7      |
| Fish (Rame et al., 2021) | 42.7      |
| GMoE w/Fish              | 48.8      |
| Swad (Cha et al., 2021a) | 46.5      |
| GMoE w/Swad              | 49.6      |

**Table 2: GMoE trained with DG algorithms.**

### 5.3 SINGLE-SOURCE DOMAIN GENERALIZATION RESULTS

In this subsection, we create a challenging task, singlesource domain generalization, to focus on generalization ability of backbone architecture. Specifically, we train the model only on data from one domain, and then test the model on multiple domains to validate its performance across all domains. This is a challenging task as we cannot rely on multiple domains to identify invariant correlations, and popular DG algorithms cannot be applied. We compare several models mentioned in above analysis (e.g., ResNet, ViT, GMoE) with different scale of parameters, including their float-point-operations per second (flops), IID and OOD accuracy. From the results in Table 3, we see that GMoE's OOD generalization gain over ResNet or ViT is much larger than that in the IID setting, which shows that GMoE is suitable for challenging domain generalization. Due to space limitation, we leave experiments with other training domains in Appendix D.5.

### 5.4 MODEL ANALYSIS

In this subsection, we present diagnostic datasets to study the connections between MoE layer and the visual attributes.

Diagnostic datasets: CUB-DG
We create CUB-DG from the original Caltech-UCSD Birds (CUB) dataset (Wah et al., 2011). We stylize the original images into another three domains, Candy, Mosaic, and Udnie. The examples of CUB-DG are presented in Fig. 3(a). We evaluate GM

oE and other DG algorithms that address domain-invariant representation (e.g., DANN). The results are in Appendix E.1, which demonstrate superior performance of GMoE compared with other DG algorithms. CUB-DG datasets provide rich visual attributes (e.g., beak's shape, belly's color) for each image. That additional information enables us to measure the correlation between visual attributes and the router's expert selection.

Visual Attributes & Experts Correlation
We choose GMoE-S/16 with 6 experts in each MoE layer. After training the model on CUB-DG, we perform forward passing with training images and save the router's top-1 selection. Since the MoE model routes patches instead of images and CUB-DG has provided the location of visual attributes, we first match a visual attribute with its nearest 9 patches, and then correlate the visual attribute and its current 9 patches' experts. In Fig. 3(b), we show the 2D histogram correlation between the selected experts and attributes. Without any supervision signal for visual attributes, the ability to correlate attributes automatically emerges during training. Specifically, 1) each expert focuses on distinct visual attributes; and 2) similar attributes are attended by the same expert (e.g., the left wing and the right wing are both attended by e4). This verifies the predictions by Theorem 2.

Expert Selection
To further understand the expert selections of the entire image, we record the router's selection for each patch (in GMoE-S/16, we process an image into 16 × 16 patches), and then visualize the top-1 selections for each patch in Fig. 3(c). In this image, we mark the selected expert for each patch with a black number and draw different visual attributes of the bird (e.g., beak and tail types) with large circles. We see the consistent relationship between Fig. 3(b) and Fig. 3(c). In detail, we see (1) experts 0 and 2 are consistently routed with patches in the background area; (2) the left wing, right wing, beak, and tail areas are attended by expert 3; (3) the left leg and right leg areas are attended by expert 4. More examples are given in Appendix E.1.

| MACs | Model     | Paint | Clipart | Info | Paint Quick | Real | Sketch | IID Imp. | OOD Imp. |
| ---- | --------- | ----- | ------- | ---- | ----------- | ---- | ------ | -------- | -------- |
| 4.1G | ResNet50  | 37.1  | 12.9    | 62.7 | 2.2         | 49.3 | 33.3   | -        | -        |
| 7.9G | ResNet101 | 40.5  | 13.1    | 63.4 | 3.1         | 51.2 | 35.4   | 1.1%     | 12.4%    |
| 4.6G | ViT-S/16  | 42.7  | 15.9    | 69.0 | 5.0         | 56.4 | 37.0   | 10.0%    | 38.6%    |
| 4.8G | GMoE-S/16 | 43.5  | 16.1    | 69.3 | 5.3         | 56.4 | 38.0   | 10.5%    | 42.3%    |

Table 3: Single-source DG accuracy (%). Models are trained on the Paint domain and tested on (1) the Paint validation set and (2) the validation sets of the other 5 domains in DomainNet. The flops are reference values to compare the model's computational efficiency. IID Imp. denotes the IID improvement on the Paint validation set compared to ResNet50. OOD Imp. denotes the average OOD improvement across the 5 test domains.



Figure 3: (a) Examples of CUB-DG datasets from four domains. (b)The y-axis corresponds to 15 + 1attributes (15 visual attributes + 1 background). The x-axis corresponds to the selected expert id. (c) Finetuned GMoE’s router decision of block-10. The image is from CUB-DG’s natural domain  

## 6 CONCLUSIONS
This paper is an initial step in exploring the impact of the backbone architecture in domain generalization. We proved that a network is more robust to distribution shifts if its architecture aligns well with the invariant correlation, which is verified on synthetic and real datasets. Based on our theoretical analysis, we proposed GMoE and demonstrated its superior performance on DomainBed. As for future directions, it is interesting to develop novel backbone architectures for DG based on algorithmic alignment and classic computer vision.  
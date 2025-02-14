NEST: A Neuromodulated Small-world Hypergraph Trajectory Prediction Model for Autonomous Driving

, Chengzhong Xu1

1University of Macau 2Southeast University 3Beihang University chengyue.wang@connect.um.edu.mo, {yc27979, mc35002, yc37976, binrao}@um.edu.mo, ziyuanpu@seu.edu.cn, zhiyongc@buaa.edu.cn, {czxu, zhenningli}@um.edu.mo

, Yanchen Guan1

, Zhenning Li1†

, Bin Rao1

Figure 1: Comparison of interaction modeling methods. (a) Traditional graph-based approach requires 24 edges to represent agent interactions. (b) Hypergraph method reduces this complexity to 4 hyperedges by grouping interactions. (c) Our Small-world Hypergraph further refines this by incorporating latent connections, effectively capturing long-range interactions typical of traffic scenarios. The use of hyperedges and latent links allows for a more efficient and com-

els is their inability to effectively capture the temporal dynamics of interactions (Chen et al. 2022). Traffic behavior is inherently dynamic, with interactions evolving over time in complex, often unpredictable ways. For instance, a vehicle's maneuver can trigger a cascade of reactions from surrounding vehicles, each adjusting its trajectory based on the observed behavior. Traditional models (Letter and Elefteriadou 2017; Liu et al. 2020), however, often rely on static snapshots of traffic, failing to account for how these interactions unfold over time. This leads to suboptimal predictions that do not accurately reflect the real-world progression of

prehensive representation of agent interactions.

, Ziyuan Pu2

,

Chengyue Wang1*, Haicheng Liao1*, Bonan Wang1

Abstract Accurate trajectory prediction is essential for the safety and efficiency of autonomous driving. Traditional models often struggle with real-time processing, capturing non-linearity and uncertainty in traffic environments, efficiency in dense traffic, and modeling temporal dynamics of interactions. We introduce NEST (Neuromodulated Small-world Hypergraph Trajectory Prediction), a novel framework that integrates Small-world Networks and hypergraphs for superior interaction modeling and prediction accuracy. This integration enables the capture of both local and extended vehicle interactions, while the Neuromodulator component adapts dynamically to changing traffic conditions. We validate the NEST model on several real-world datasets, including nuScenes, MoCAD, and HighD. The results consistently demonstrate that NEST outperforms existing methods in various traffic scenarios, showcasing its exceptional generalization capability, efficiency, and temporal foresight. Our comprehensive evaluation illustrates that NEST significantly improves the reliability and operational efficiency of autonomous driving systems, making it a robust solution for trajectory prediction

Introduction The ability to predict the future paths of surrounding vehicles with precision is not just an academic pursuit—it's a life-saving technology at the heart of autonomous driving (AD). In the bustling dance of traffic, where vehicles, pedestrians, and cyclists move in a complex choreography, even a moment's hesitation can lead to disaster (Pourkeshavarz, Sabokrou, and Rasouli 2024). Thus, the quest for an efficient model that can foresee these movements with high fidelity is more than a goal; it's a necessity for the next leap in vehicu-

Traditional trajectory prediction models have paved the way, yet they falter under the unpredictable and multifaceted nature of real-world traffic (Wong et al. 2024). One of the significant limitations of current trajectory prediction mod-

Copyright © 2025, Association for the Advancement of Artificial

in complex traffic environments.

arXiv:2412.11682v1 [cs.RO] 16 Dec 2024

lar autonomy (Duan et al. 2022).

*These authors contributed equally.

Intelligence (www.aaai.org). All rights reserved.

†Corresponding Author

Zhiyong Cui3

traffic scenarios, ultimately compromising the safety and ef-

Moreover, these models frequently struggle with the inherent non-linearity and uncertainty of traffic environments (Wang et al. 2022). Traffic dynamics are influenced by a myriad of variables, including sudden stops, erratic driving patterns, and the unpredictable behavior of pedestrians and cyclists. The non-linear nature of these interactions introduces chaos that many current models cannot adequately address. For example, a pedestrian suddenly crossing the road or a cyclist weaving through traffic can dramatically alter the flow of vehicles, creating a highly unpredictable environment. Traditional models often oversimplify these interactions, leading to inaccurate predictions during unforeseen To sum up, our contributions are threefold:

namic traffic scenarios.

• We introduce a Small-world Network that effectively captures both local and long-range interactions among traffic agents. The Neuromodulator within the NEST model dynamically adjusts the network to contextual information, enhancing adaptability in diverse and dy-

• We propose a novel Hypergraph Neural Network (HGNN) for interaction learning. Our unique hyperedge set structure combines hyperedges of different relationship types, providing an efficient framework for model-

• We conduct extensive validation of the NEST model using multiple real-world datasets, including nuScenes, MoCAD, and HighD. The results consistently demonstrate that the NEST model outperforms existing methods in various traffic scenarios, showcasing its excep-

ing complex and diverse interaction dynamics.

tional generalization capabilities and reliability.

their applicability to structured road topologies.

To address the limitations of grid-based methods, Graph Neural Networks (GNNs) have emerged as a promising approach, particularly suited for modeling interactions in unstructured and complex urban environments. GNNs excel at representing interactions between vehicles in scenarios where road structures are not strictly defined. For example, the SFEM-GCN model proposed by Du et al. (Du et al. 2024) incorporates three types of graph structures to capture semantic, positional, and velocity information of traffic agents, thereby offering a more comprehensive interaction model. Similarly, the MTP-GO model (Westny et al. 2023)

Related Work With the significant advancements in deep learning technologies across various domains of autonomous driving (AD), it is natural that researchers have increasingly applied these techniques to the field of trajectory prediction. Initially, trajectory prediction was approached as a time-series forecasting problem, leading to the use of neural networks designed for handling sequential data, such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, as encoders and decoders in these models (Lan et al. 2024; Liao et al. 2024h). A pivotal development in this area was the introduction of Social LSTM by Alahi et al. (Alahi et al. 2016), which first brought the concept of social forces into trajectory prediction, highlighting the importance of understanding interactions between agents. Building on this foundation, researchers shifted their focus towards modeling interactions from both spatial and temporal dimensions (Liao et al. 2024b,c; Xu et al. 2022; Chen et al. 2024). The use of attention mechanisms further advanced the field, enabling models to capture complex interactions among agents (Liao et al. 2024e; Lan et al. 2024; Liao et al. 2024a). For instance, the Nettraj model (Liang and Zhao 2021) introduced Spatiotemporal Attention Mechanisms with a sliding context window to capture both shortand long-term interactions. However, these attention-based methods often rely on grid-based approaches (Liao et al. 2024g,f), segmenting the scene into fixed grids, which limits

Efficiency represents another significant challenge, especially in heterogeneous urban environments where traffic density is extremely high. In such scenarios, various heterogeneous traffic agents—vehicles, pedestrians, and cyclists—interact, resulting in an exponential increase in the types of interaction relationships (Ramezani, Haddad, and Geroliminis 2015). Each agent may be involved in multiple interaction groups, with each group comprising different numbers of agents. Traditional methods of modeling interactions are often confined to predefined relationships (Lv et al. 2023; Xu et al. 2023), limiting their efficiency in capturing diverse and dynamic interactions. This underscores the necessity for an efficient paradigm that can accommodate di-

To address these challenges, we propose the NEST (Neuromodulated Small-world Hypergraph Trajectory Prediction) model for autonomous driving. This framework combines the intricate structure of Small-world Networks (Watts and Strogatz 1998) with the expansive reach of hypergraphs (Gao et al. 2022) to enhance interaction modeling and trajectory prediction. Small-world Networks, characterized by high clustering and short path lengths, effectively capture both local and long-range interactions among vehicles, leading to a more accurate representation of traffic dynamics (as illustrated in Figure 1), where even distant ve-

The Neuromodulator component adapts this structure to reflect the dynamic and diverse nature of real-world traffic (Grossman and Cohen 2022), ensuring that the model remains robust and responsive to changing conditions. Hypergraphs extend beyond traditional graph structures by allowing hyperedges to connect multiple nodes simultaneously, which is crucial for modeling the collective behavior of traffic agents (Xu et al. 2022), such as groups of pedestrians or a convoy of vehicles merging into a lane. By integrating Small-world Networks and hypergraphs, our model not only predicts trajectories but also provides a comprehensive understanding of vehicular interactions. This fusion addresses the limitations of existing models, offering enhanced efficiency to handle dense traffic, temporal foresight to capture evolving interactions, and contextual integration to consider relevant external factors. Moreover, the dynamic adaptability of our model ensures it can generalize beyond its training datasets, making it suitable for a wide range of scenarios.

ficiency of AD systems.

events (Wong et al. 2024).

verse interaction relationships in real time.

hicles can influence each other's trajectories.

Figure 2: The overview of the proposed NEST model. Panel (a) illustrates the overall framework of NEST, which takes agent historical data and high-definition (HD) maps as inputs. The NEST model processes these inputs with four key modules: Hypergraph Forming, Hypergraph Pooling, Context Fusion, and a Multi-modal Predictor. These modules work in tandem to predict the target agent's intention and future trajectory. Panels (b) and (c) provide a detailed breakdown of the components involved in forming the interaction hypergraph, specifically focusing on the Neuromodulator and the Small-world Network.

> map M. The historical data X = [X0, X1, . . . , Xn] includes coordinates, velocity, and acceleration for the target agent

> ber of agents. The HD map M records the road topology. The output is the future trajectory Y of the target agent over the prediction horizon tf . Predicting the future trajectories of traffic agents involves addressing various uncertainties rooted in the diversity of driving intentions and the unpredictability of motion. The NEST model addresses these uncertainties by generating multiple predicted trajectories, Y = [Y1, Y2, . . . , YK], where each Yi corresponds to a distinct intention mode with an associated probability Pi

> K denotes the number of intention modes. To capture the unpredictable nature of motion within each intention mode, predictions at each time step t in Yi are represented by pa-

> > are the coordinates, and b

The proposed NEST model is illustrated in Figure 2. The Hypergraph Forming module forms the interaction hypergraph G with the Neuromodulator and Small-world Network. Then, the Hypergraph Pooling module captures interaction features Fi based on the interaction hypergraph G. Next, the Context Fusion module combines lane features Fl from HD maps M with the interaction feature Fi

ing in a consolidated context feature Fc. Finally, the Multi-

scale parameters reflecting unpredictability of motion.

t i = [x t i , yt i , bt i,x, bt i,y],

t i,x and b t i,y are

. n denotes the num-

. Here,

, result-

X0 and the surrounding agents Xi∈[1,n]

rameters of the Laplace distribution: Y

where x t i and y t i

Model Overview

employs a temporal graph to encode traffic scene information and utilizes neural ordinary differential equations for final predictions. The Social Soft Attention Graph Convolution Network by Lv et al. (Lv et al. 2023) addresses the challenge of handling interactions not only between agents

Despite their advantages, graph-based models have inherent limitations. They often focus on one-to-one relationships, potentially overlooking broader group dynamics and the interconnected nature of traffic behaviors. Moreover, efficiency issues and computational constraints can become significant, especially in dense urban environments with numerous interacting agents. While GNNs provide a more comprehensive representation of interactions compared to other models, they may still fall short in fully capturing the complexity and dynamic interactions inherent in real-world

Methodology

The objective is to predict the future trajectories of surrounding traffic agents within a defined perception range over a specific period. Each agent to be predicted is referred to as the *target agent*, while all other agents within the perception

The NEST model takes two inputs: historical data X spanning the past th time steps, and a high-definition (HD)

range are considered as *surrounding agents*.

but also between agents and their environment.

traffic scenarios.

Problem Formulation

modal Predictor uses K generators to predict future trajec-

The hypergraph is distinguished by its unique structure where a single hyperedge can link multiple vertices. This unique structure makes it particularly well-suited for modeling group-wise interaction relationships among traffic agents. Each hyperedge defines an interaction group, with each vertex connected by the hyperedge representing an

Forming an appropriate interaction hypergraph G is fundamental for efficiently capturing interactions among agents. We define the interaction hypergraph G = (V, E), where V is the set of vertex features, and E is the set of hyperedges. To represent traffic agents as vertices, we de-

the agent features Fa are obtained through a transformerbased encoder, and d is the hidden dimension of the feature. Specifically, V = [V0, V1, ..., Vn], where the feature Vi of vertex i corresponds to the feature Fa,i of agent i, with Fa,0

Small-world Network In traffic scenarios, agents can be influenced not only by nearby agents but also by distant ones through interactions mediated by intermediate agents. For example, in a car-following scenario, the braking of a leading vehicle can trigger a chain reaction among following vehicles. This reflects the small-world property of traffic, where agents are connected through short interaction chains, similar to social networks. A Small-world Network, characterized by this property, is therefore ideal for modeling in-

To robustly capture these dynamic interactions, we design a Small-world Network module inspired by the Newman–Watts (NW) model (Newman and Watts 1999) to generate hyperedges. We determine each agent's membership in interaction groups using the clustering coefficient C, which measures the tendency of a vertex to cluster with its neighbors. The clustering coefficient Ci,j for each vertex-

1 if Ci,j ≥ α

0 otherwise (1)

(n+1)×d

, where

In summary, the process of generating the hyperedge set

Neuromodulator Human drivers adapt their behavior to ever-changing driving environments based on external stimuli. Inspired by this adaptability, rooted in cellular neuroregulation mechanisms (Vecoven et al. 2020), we developed the Neuromodulator to dynamically adjust the Small-world Network, enhancing its responsiveness to traffic conditions. As shown in Equation 3, the hyperedge set is modulated by two parameters: threshold α and connection probability β, de-

The threshold α determines which agents are included in an interaction group. To set this threshold, we analyze the distribution of the clustering coefficient matrix C. The

where s is the predefined number of hyperedges. The final threshold α ∈ [0, 1] is obtained using a sigmoid function:

The connection probability β represents the likelihood of involving additional agents in interaction groups, particularly in high-density traffic. The agent feature Fa encapsulates traffic density, from which the connection probability β ∈ [0, 1] is derived using the connection probability neuromodulator Πβ, similar to the threshold neuromodulator:

where Πβ is the connection probability neuromodulator.

The interaction hypergraph explicitly defines the traffic agents in each interaction group, with vertices connected by hyperedges. Unlike traditional graphs, which limit information aggregation to pairs of agents, hypergraphs capture interaction features more efficiently through iterative Vertexto-Hyperedge and Hyperedge-to-Vertex pooling (Xu et al. 2022). This approach provides a more comprehensive representation of interactions in a traffic scene, leading to more

Vertex-to-Hyperedge In this process, the hyperedge serves as a platform for information exchange among vertices in the same interaction group. The Vertex-to-Hyperedge step aggregates features from all connected vertices to form a group feature Fg. To align with the requirements for drivers during interactions, Fg should include three types of information: the personality of each traffic agent, their intentions, and the possibility of these intentions. Traffic agents in the same group often adjust their behavior based on others, such as drivers yielding to aggressive counterparts. The agent feature Fa reflects these personalities. We aggregate the features of all vertices connected to

E = Ω(V, α, β) (3)

(n+1)×s

| Eij = 1, using a

(n+1)×1 using two MLP layers,

α = Πα(C) (4)

β = Πβ(Fa) (5)

into a

can be formulated as follows:

where Ω is the Small-world Network.

rived from the agent feature Fa.

threshold feature space of R

Hypergraph Pooling

accurate trajectory predictions.

the same hyperedge j, denoted as V = Vi

threshold neuromodulator Πα projects C ∈ R

where Πα denotes the threshold neuromodulator.

tories for each intention mode.

agent within that interaction group.

corresponding to the target agent.

fine the vertex feature set as V = Fa ∈ R

teraction relationships among traffic agents.

Ci,j =

where Ci,j = 1 indicates a definite connection between vertex i and hyperedge j, while Ci,j = 0 suggests a potential

The constructed clustering coefficient matrix C is analogous to the basic regular network in the NW model. To refine the connections between vertices and hyperedges, we re-evaluate the relationships where Ci,j is uncertain, applying a connection probability β. The final hyperedges are de-

> 1 if Ci,j = 1 or (Ci,j = 0 and η ≤ β) 0 otherwise (2)

where Ei,j = 1 indicates that vertex i is connected to hyperedge j, and η is a randomly distributed value in [0, 1].

hyperedge pair is defined as:

connection that is less certain.

termined as follows:

Ei,j =

Hypergraph Forming

Model Venue minADE5 minADE1 minFDE1

DLow-AF (Yuan and Kitani 2020) ECCV 2.11 - - Trajectron++ (Salzmann et al. 2020) ECCV 1.88 - 9.52 MultiPath (Chai et al. 2020) CoRL 1.44 3.16 7.69 LDS-AF (Ma et al. 2021) ICCV 2.06 - - AgentFormer (Yuan et al. 2021) ICCV 1.97 - - LaPred (Kim et al. 2021) CVPR 1.53 3.51 8.12 STGM (Zhong, Luo, and Liang 2022) TITS - 3.21 9.62 GoHome (Gilles et al. 2022) ICRA 1.42 - 6.99 ContextVAE (Xu, Hayet, and Karamouzas 2023) RAL 1.59 3.54 8.24 EMSIN (Ren et al. 2024) TFS 1.77 3.56 - SeFlow (Zhang et al. 2024) ECCV 1.38 - 7.89 NEST - 1.18 2.97 6.87

Table 1: Performance comparison of various models on nuScenes dataset. Bold values represent the best performance in each category. "-" denotes the missing value.

weighted sum. This sum is then processed by a Multi-Layer Perceptron (MLP) based personality encoder Mp to extract

> X Vi∈V

Obtaining the various intentions of agent i, along with the degree of willingness to execute each intention, aids the model in predicting future motion. We define the intention information Ii = [Ii,1, Ii,2, ..., Ii,K], each Ii,j denotes a specific intention. We formulate intention information Ii as:

where σ is the softmax function, and aggregated information

following (Xu et al. 2022), we introduce Gumbel distribu-

The degree of willingness for each intention is measured by Iw = Mw(Ia), where Mw is the willingness encoder. After obtaining personality information Ip, intention infor-

> X K

j=1

where Ii,j and Iw,j are the pairs of intention mode j and its associated willingness. K is the number of intentions.

Hyperedge-to-Vertex To enable traffic agents to obtain relevant information from the hyperedges, we employ the Hyperedge-to-Vertex approach. This method updates each vertex feature Vi based on the group features Fg of the connected hyperedges. During this update process, we consider the group features of all hyperedges linked to the vertex i, denoted as E = Ej | Eij = 1. The update process is de-

, and willingness information Iw, the group feature

Ii = σ ((Mi (Ia) + ξ) /τ ) (7)

. Mi denotes the intention encoder. And

Fg,j 

(9)

Ii,jIw,j (8)

λiVi) (6)

Model

Model

interaction feature Fi

Context Fusion

tures Fl

as defined by:

Prediction Horizon (s) 1 2 3 4 5

Prediction Horizon (s) 1 2 3 4 5

. A

, V = Fl) (10)

CS-LSTM (Deo and Trivedi 2018) 1.45 1.98 2.94 3.56 4.49 NLS-LSTM (Messaoud et al. 2019) 0.96 1.27 2.08 2.86 3.93 MHA-LSTM (Messaoud et al. 2021) 1.25 1.48 2.57 3.22 4.20 CF-LSTM (Xie et al. 2021) 0.72 0.91 1.73 2.59 3.44 STDAN (Chen et al. 2022) 0.62 0.85 1.62 2.51 3.32 WSiP (Wang et al. 2023) 0.70 0.87 1.70 2.56 3.47 BAT (25%) (Liao et al. 2024d) 0.65 0.99 1.89 2.81 3.58 BAT (Liao et al. 2024d) 0.35 0.74 1.39 2.19 2.88 NEST 0.32 0.75 1.27 2.01 2.42

Table 2: Experimental results on MoCAD. Metric: RMSE

MHA-LSTM (Messaoud et al. 2021) 0.19 0.55 1.10 1.84 2.78 CF-LSTM (Xie et al. 2021) 0.18 0.42 1.07 1.72 2.44 EA-Net (Cai et al. 2021) 0.15 0.26 0.43 0.78 1.32 STDAN (Chen et al. 2022) 0.19 0.27 0.48 0.91 1.66 DRBP (Gao et al. 2023) 0.41 0.79 1.11 1.40 - WSiP (Wang et al. 2023) 0.20 0.60 1.21 2.07 3.14 DACR-AMTP (Cong et al. 2023) 0.10 0.17 0.31 0.54 1.01 BAT (25%) (Liao et al. 2024d) 0.14 0.34 0.65 0.89 1.27 BAT (Liao et al. 2024d) 0.08 0.14 0.20 0.44 0.62 NEST 0.05 0.11 0.20 0.32 0.48

Table 3: Experimental results on HighD. Metric: RMSE

in the scene, we average these updated features to derive the

The environmental context, especially the topology of the lane network, is essential for accurate trajectory prediction. To incorporate this, we employ a context fusion module that integrates lane features Fl with interaction features Fi

Lane Encoder first processes HD maps to generate lane fea-

. Then, an attention module combines these with the interaction features to produce the final context feature Fc,

, K = Fl

, characterized by a Laplace distribu-

.

This approach allows the model to account for the critical influence of lane topology on agent trajectories, enhancing the accuracy and reliability of trajectory predictions by inte-

Driving behavior is inherently uncertain due to both the diversity of intentions at the macro level and the unpredictability of actions at the micro level. To capture this range of behavior, we use a Multi-modal Predictor, where each generator represents a distinct human intention mode. The microlevel unpredictability within each mode is modeled using a Laplace distribution. We employ K generators, each combining the context feature Fc with the agent feature Fa to

Experiment

We conducted experiments on real-world traffic datasets (nuScenes, MoCAD, and HighD) to showcase the superior-

.

Fc = *Attn*(Q = Fi

Multi-modal Predictor

predict a trajectory Yi

Experimental Setup

tion and its associated probability Pi

grating interaction and contextual information.

Ip = Mp(

is the learnable weight.

λiVi

tion ξ and temperature parameter τ .

Fg if formulated as following:

Fg = Ip

personality information Ip:

where λi

Ia = P Vi∈Ej

mation Ii

scribed as follows:

Vi ← Mv

 Vi , X Ej∈E

where Fg,j is the group feature for hyperedge j, and Mv denotes an MLP-based vertex encoder. After undergoing H iterations of updates, we obtain the refined features for all vertices. To ensure comprehensive consideration of all agents Model Inference Time (ms)

Components

Metrics

Ablation Studies

NEST model.

Ablation Methods A B C D E F

Neuromodulator ✘ ✘ ✘ ✔ ✔ ✔ Small-world Network ✔ ✘ ✘ ✔ ✔ ✔ Hypergraph Learning ✔ ✔ ✘ ✔ ✔ ✔ Context Fusion ✔ ✔ ✔ ✘ ✔ ✔ Multi-modal Decoder ✔ ✔ ✔ ✔ ✘ ✔

Table 5: Ablation setting of nuScenes.

mADE5 1.21 1.25 1.48 1.37 1.22 1.18 mADE1 2.99 3.13 3.21 3.17 3.06 2.97 mFDE1 6.92 7.28 7.44 7.32 7.12 6.87

Table 6: Ablation results on nuScenes.

VisionTrap (Moon et al. 2024), which reported the time required to predict trajectories for 12 agents using a single RTX 3090 Ti GPU. For a fair comparison, we evaluated the NEST model by calculating its average inference time for 12 agents across the dataset. Due to the unavailability of an RTX 3090 Ti GPU, our experiments were performed with an RTX 3090 GPU. As evidenced in Table 4, the NEST model consistently demonstrates superior inference speed compared to existing baselines even with a slower GPU than RTX 3090 Ti, highlighting its real-time performance.

The NEST model is composed of multiple components that collectively yield impressive performance. However, it remains to be seen whether each component significantly contributes to the model's predictive capability. To quantify the impact of each component on the model's performance, we conducted a series of ablation studies. We present detailed ablation variants of the NEST model on the nuScenes dataset in Table 5 and 6. Method F represents the complete NEST model, which achieves the SOTA performance across all metrics, demonstrating the synergistic effect of its components. Method C, which employs a conventional graph neural network to replace hypergraph learning, exhibits the poorest performance across most metrics, highlighting the significance of the Neuromodulated Small-world Hypergraph in accurately learning agent interactions. Method B, which discards the Neuromodulated Small-world Network approach for constructing the interaction hypergraph, exhibits significantly poorer performance across all metrics. Method D, which lacks the context fusion module, also shows degraded performance, indicating the importance of contextual information in driving scenarios for precise trajectory prediction. The results of Method A, and Method E show varying degrees of performance degradation, further confirming the necessity of each component within the

Ablation Methods A B C D E F

P2T (Deo and Trivedi 2020) 116 Trajectron++ (Salzmann et al. 2020) 38 MultiPath (Chai et al. 2020) 87 AgentFormer (Yuan et al. 2021) 107 PGP (Deo, Wolff, and Beijbom 2022) 215 LAformer (Liu et al. 2024) 115 VisionTrap (Moon et al. 2024) 53 NEST (RTX 3090) 11.6

Table 4: Inference time comparison on nuScenes dataset.

ity of our NEST model over state-of-the-art baselines. We included both quantitative and qualitative analyses, along with inference time comparisons. Each model component's contribution was also evaluated. To ensure consistent and fair comparisons, we followed the established evaluation metrics for each dataset: minADEk and minFDEk for nuScenes, and Root Mean Squared Error (RMSEk) for Mo-

The quantitative comparison results for different datasets are detailed in Tables 1, 2, and 3, respectively. Table 1 shows that our NEST model consistently achieves SOTA performance across all metrics on the nuScenes dataset. Notably, it improves the minADE5 metric by 14.5% compared to the best existing model (Zhang et al. 2024), indicating superior trajectory prediction accuracy. Our model also shows significant enhancements in both the minADE1 and minFDE1 metrics, demonstrating comprehensive improvements in predictive capability. Furthermore, as the value of k decreases, the superiority of our model becomes more evident, showcasing its ability to accurately capture the true intentions of agents. To validate the effectiveness of our Neuromodulated Small-world Hypergraph in modeling agent interactions, we tested its performance on the MoCAD and HighD datasets without relying on HD maps. As detailed in Table 2, our NEST model outperforms all competitors across various prediction horizons in the MoCAD dataset, particularly improving accuracy by 16% in the 5-second prediction horizon over the previous best model. Although our model slightly trails the best in short-term (2-second) predictions by just 0.01 meter, it still shows strong performance. Table 3 illustrates our model's superiority on the HighD dataset, focused on highway scenarios. Here, the NEST model achieves significant improvements of 27.3% and 22.6% over the BAT model (Liao et al. 2024d) in the 4-second and 5-second prediction windows, respectively. These results across diverse scenarios highlight the exceptional generalization and predictive capabilities of our NEST model in the real world.

CAD and HighD.

Quantitative Results

Inference Time Comparison

To demonstrate the efficiency of our proposed NEST model, we conducted a comparative experiment on inference times on nuScenes dataset. Table 4 presents the inference times of various models, including baseline figures sourced from

Figure 3: Qualitative comparison of our NEST model with various models. Panels (a) and (b) visualize the most probable prediction by each model, whereas panel (c) visualizes predictions across ten modalities. Panel (a) illustrates the scenario where a distant leading vehicle accelerates. In this scenario, our NEST model maintains a trajectory prediction that aligns closely with the ground truth. In contrast, other models are misled by the stationary state of surrounding vehicles, resulting in slower predictions. Panel (b) depicts a scenario where a distant leading vehicle remains stationary, significantly influencing the expected speed of the target agent. Again, only our NEST model predicts a trajectory that mirrors the ground truth closely, while other models predict overly fast trajectories. Panel (c) distinctly showcases the efficacy of hypergraph-based models (NEST and

els in panel (c). Panel (c) elucidates the proficiency of our

Conclusion In this paper, we introduce the NEST model, an innovative framework designed to address critical challenges in trajectory prediction for autonomous driving. By integrating the strengths of Small-world Networks and hypergraphs, NEST captures both local and long-range interactions among heterogeneous traffic agents in an efficient manner. The Neuromodulator component further enhances the model's adaptability to dynamic traffic conditions, ensuring accurate and reliable predictions. Extensive validation across multiple real-world datasets, including nuScenes, MoCAD, and HighD, consistently demonstrates the NEST model's superior performance and generalization capabili-

NEST in capturing the intention of drivers.

ties in various traffic scenarios.

Method B) in predicting the driver's true intent through their prediction of a complex U-turn maneuver.

Qualitative Results

In addition to the quantitative analysis, we present the qualitative results from the nuScenes dataset to provide an intuitive understanding of the interaction comprehension capabilities and predictive accuracy of our NEST model. Figure 3 provides a summary of the predictive results across various scenarios for the others model (Chen et al. 2024), ours (NEST model, and its ablated variants—method B and method C). Panels (a) and (b) illustrate the small-world phenomenon in traffic scenarios, where the distant leading vehicle could influence the target agent's behavior. The comparison of various models demonstrates the efficacy of Smallworld Networks on interaction modeling within these scenarios. Modeling the inherent uncertainty in driving behavior is crucial for accurate prediction. To demonstrate the NEST model's capability in modeling uncertainty, we present the multimodal prediction results from various mod-

Acknowledgements This research is supported by Science and Technology Development Fund of Macau SAR (File no. 0021/2022/ITP, 0081/2022/A2, 001/2024/SKL), Shenzhen-Hong Kong-Macau Science and Technology Program Category C (SGDX20230821095159012), State Key Lab of Intelligent Transportation System (2024-B001), Jiangsu Provincial Science and Technology Program (BZ2024055), and University

Gao, K.; Li, X.; Chen, B.; Hu, L.; Liu, J.; Du, R.; and Li, Y. 2023. Dual Transformer Based Prediction for Lane Change Intentions and Trajectories in Mixed Traffic Environment. *IEEE Transactions on Intelligent Transportation Systems*. Gao, Y.; Feng, Y.; Ji, S.; and Ji, R. 2022. HGNN+: General hypergraph neural networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(3): 3181–3199. Gilles, T.; Sabatini, S.; Tsishkou, D.; Stanciulescu, B.; and Moutarde, F. 2022. Gohome: Graph-oriented heatmap output for future motion estimation. In *2022 international conference on robotics and automation (ICRA)*, 9107–9114.

Grossman, C. D.; and Cohen, J. Y. 2022. Neuromodulation and neurophysiology on the timescale of learning and decision-making. *Annual review of neuroscience*, 45: 317–

Kim, B.; Park, S. H.; Lee, S.; Khoshimjonov, E.; Kum, D.; Kim, J.; Kim, J. S.; and Choi, J. W. 2021. Lapred: Laneaware prediction of multi-modal future trajectories of dynamic agents. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 14636–14645. Lan, Z.; Ren, Y.; Yu, H.; Liu, L.; Li, Z.; Wang, Y.; and Cui, Z. 2024. Hi-SCL: Fighting long-tailed challenges in trajectory prediction with hierarchical wave-semantic contrastive learning. *Transportation Research Part C: Emerging Tech-*

Letter, C.; and Elefteriadou, L. 2017. Efficient control of fully automated connected vehicles at freeway merge segments. *Transportation Research Part C: Emerging Tech-*

Liang, Y.; and Zhao, Z. 2021. NetTraj: A network-based vehicle trajectory prediction model with directional representation and spatiotemporal attention mechanisms. *IEEE Transactions on Intelligent Transportation Systems*, 23(9):

Liao, H.; Li, X.; Li, Y.; Kong, H.; Wang, C.; Wang, B.; Guan, Y.; Tam, K.; and Li, Z. 2024a. CDSTraj: Characterized Diffusion and Spatial-Temporal Interaction Network for Trajec-

Liao, H.; Li, Y.; Li, Z.; Wang, C.; Cui, Z.; Li, S. E.; and Xu, C. 2024b. A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving. *IEEE Transactions on In-*

Liao, H.; Li, Y.; Li, Z.; Wang, C.; Li, G.; Tian, C.; Bian, Z.; Zhu, K.; Cui, Z.; and Hu, J. 2024c. Less is More: Efficient Brain-Inspired Learning for Autonomous Driving Trajectory Prediction. In *ECAI 2024*, 4361–4368. IOS Press. Liao, H.; Li, Z.; Shen, H.; Zeng, W.; Liao, D.; Li, G.; and Xu, C. 2024d. Bat: Behavior-aware human-like trajectory prediction for autonomous driving. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, 10332–

Liao, H.; Li, Z.; Wang, C.; Shen, H.; Wang, B.; Liao, D.; Li, G.; and Xu, C. 2024e. MFTraj: Map-Free, Behavior-Driven Trajectory Prediction for Autonomous Driving. *International Joint Conference On Artificial Intelligence*.

tory Prediction in Autonomous Driving. In *IJCAI*.

IEEE.

337.

*nologies*, 165: 104735.

*nologies*, 80: 190–205.

14470–14481.

*telligent Vehicles*.

10340.

References Alahi, A.; Goel, K.; Ramanathan, V.; Robicquet, A.; Fei-Fei, L.; and Savarese, S. 2016. Social lstm: Human trajectory prediction in crowded spaces. In *Proceedings of the IEEE conference on computer vision and pattern recogni-*

Cai, Y.; Wang, Z.; Wang, H.; Chen, L.; Li, Y.; Sotelo, M. A.; and Li, Z. 2021. Environment-attention network for vehicle trajectory prediction. *IEEE Transactions on Vehicular*

Chai, Y.; Sapp, B.; Bansal, M.; and Anguelov, D. 2020. MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction. In *Conference on Robot Learning*,

Chen, J.; Wang, Z.; Wang, J.; and Cai, B. 2024. Q-EANet: Implicit social modeling for trajectory prediction via experience-anchored queries. *IET Intelligent Transport Sys-*

Chen, X.; Zhang, H.; Zhao, F.; Hu, Y.; Tan, C.; and Yang, J. 2022. Intention-aware vehicle trajectory prediction based on spatial-temporal dynamic attention network for internet of vehicles. *IEEE Transactions on Intelligent Transportation*

Cong, P.; Xiao, Y.; Wan, X.; Deng, M.; Li, J.; and Zhang, X. 2023. DACR-AMTP: Adaptive Multi-Modal Vehicle Trajectory Prediction for Dynamic Drivable Areas Based on Collision Risk. *IEEE Transactions on Intelligent Vehicles*. Deo, N.; and Trivedi, M. M. 2018. Convolutional social pooling for vehicle trajectory prediction. In *Proceedings of the IEEE conference on computer vision and pattern recog-*

Deo, N.; and Trivedi, M. M. 2020. Trajectory forecasts in unknown environments conditioned on grid-based plans.

Deo, N.; Wolff, E.; and Beijbom, O. 2022. Multimodal trajectory prediction conditioned on lane-graph traversals. In

Du, Q.; Wang, X.; Yin, S.; Li, L.; and Ning, H. 2024. Social Force Embedded Mixed Graph Convolutional Network for Multi-class Trajectory Prediction. *IEEE Transactions on*

Duan, J.; Wang, L.; Long, C.; Zhou, S.; Zheng, F.; Shi, L.; and Hua, G. 2022. Complementary attention gated network for pedestrian trajectory prediction. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, 542–

*Conference on Robot Learning*, 203–212. PMLR.

of Macau (SRG2023-00037-IOTSC).

*Technology*, 70(11): 11216–11227.

*tion*, 961–971.

86–99. PMLR.

*tems*, 18(6): 1004–1015.

*Systems*, 23(10): 19471–19483.

*nition workshops*, 1468–1476.

*arXiv preprint arXiv:2001.00735*.

*Intelligent Vehicles*.

550.

Liao, H.; Li, Z.; Wang, C.; Wang, B.; Kong, H.; Guan, Y.; Li, G.; Cui, Z.; and Xu, C. 2024f. A Cognitive-Driven Trajectory Prediction Model for Autonomous Driving in Mixed Autonomy Environment. *International Joint Conference On* Salzmann, T.; Ivanovic, B.; Chakravarty, P.; and Pavone, M. 2020. Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part XVIII 16*, 683–700. Springer. Vecoven, N.; Ernst, D.; Wehenkel, A.; and Drion, G. 2020. Introducing neuromodulation in deep neural networks to learn adaptive behaviours. *PloS one*, 15(1): e0227922. Wang, R.; Wang, S.; Yan, H.; and Wang, X. 2023. WSiP: wave superposition inspired pooling for dynamic interactions-aware trajectory prediction. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37,

Wang, W.; Wang, L.; Zhang, C.; Liu, C.; Sun, L.; et al. 2022. Social interactions for autonomous driving: A review and perspectives. *Foundations and Trends® in Robotics*, 10(3-

Watts, D. J.; and Strogatz, S. H. 1998. Collective dynamics of 'small-world'networks. *nature*, 393(6684): 440–442. Westny, T.; Oskarsson, J.; Olofsson, B.; and Frisk, E. 2023. Mtp-go: Graph-based probabilistic multi-agent trajectory prediction with neural odes. *IEEE Transactions on Intel-*

Wong, C.; Xia, B.; Zou, Z.; Wang, Y.; and You, X. 2024. SocialCircle: Learning the Angle-based Social Interaction Representation for Pedestrian Trajectory Prediction. In *Proceedings of the IEEE/CVF Conference on Computer Vision*

Xie, X.; Zhang, C.; Zhu, Y.; Wu, Y. N.; and Zhu, S.-C. 2021. Congestion-aware multi-agent trajectory prediction for col-

Xu, C.; Li, M.; Ni, Z.; Zhang, Y.; and Chen, S. 2022. Groupnet: Multiscale hypergraph neural networks for trajectory prediction with relational reasoning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern*

Xu, D.; Shang, X.; Peng, H.; and Li, H. 2023. MVHGN: Multi-view adaptive hierarchical spatial graph convolution network based trajectory prediction for heterogeneous traffic-agents. *IEEE Transactions on Intelligent Transporta-*

Xu, P.; Hayet, J.-B.; and Karamouzas, I. 2023. Contextaware timewise vaes for real-time vehicle trajectory predic-

Yuan, Y.; and Kitani, K. 2020. DLow: Diversifying Latent Flows for Diverse Human Motion Prediction.

Yuan, Y.; Weng, X.; Ou, Y.; and Kitani, K. 2021. Agent-Former: Agent-Aware Transformers for Socio-Temporal

Zhang, Q.; Yang, Y.; Li, P.; Andersson, O.; and Jensfelt, P. 2024. SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving. *arXiv preprint arXiv:2407.01702*. Zhong, Z.; Luo, Y.; and Liang, W. 2022. STGM: Vehicle trajectory prediction based on generative model for spatialtemporal features. *IEEE Transactions on Intelligent Trans-*

tion. *IEEE Robotics and Automation Letters*.

Multi-Agent Forecasting. arXiv:2103.14023.

*portation Systems*, 23(10): 18785–18793.

*and Pattern Recognition*, 19005–19015.

lision avoidance. In *ICRA*.

*Recognition*, 6498–6507.

*tion Systems*.

arXiv:2003.08386.

4685–4692.

4): 198–376.

*ligent Vehicles*.

Liao, H.; Liu, S.; Li, Y.; Li, Z.; Wang, C.; Li, Y.; Li, S. E.; and Xu, C. 2024g. Human observation-inspired trajectory prediction for autonomous driving in mixed-autonomy traffic environments. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, 14212–14219. IEEE. Liao, H.; Wang, C.; Li, Z.; Li, Y.; Wang, B.; Li, G.; and Xu, C. 2024h. Physics-Informed Trajectory Prediction for Autonomous Driving under Missing Observation. *International*

Liu, M.; Cheng, H.; Chen, L.; Broszio, H.; Li, J.; Zhao, R.; Sester, M.; and Yang, M. Y. 2024. Laformer: Trajectory prediction for autonomous driving with lane-aware scene constraints. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2039–2049. Liu, X.; Shen, D.; Lai, L.; and Le Vine, S. 2020. Optimizing the safety-efficiency balancing of automated vehicle carfollowing. *Accident Analysis & Prevention*, 136: 105435. Lv, P.; Wang, W.; Wang, Y.; Zhang, Y.; Xu, M.; and Xu, C. 2023. SSAGCN: social soft attention graph convolution network for pedestrian trajectory prediction. *IEEE transactions*

Ma, Y. J.; Inala, J. P.; Jayaraman, D.; and Bastani, O. 2021. Likelihood-Based Diverse Sampling for Trajectory Fore-

Messaoud, K.; Yahiaoui, I.; Verroust-Blondet, A.; and Nashashibi, F. 2019. Non-local social pooling for vehicle trajectory prediction. In *2019 IEEE Intelligent Vehicles Sym-*

Messaoud, K.; Yahiaoui, I.; Verroust-Blondet, A.; and Nashashibi, F. 2021. Attention Based Vehicle Trajectory Prediction. *IEEE Transactions on Intelligent Vehicles*, 6(1):

Moon, S.; Woo, H.; Park, H.; Jung, H.; Mahjourian, R.; Chi, H.-g.; Lim, H.; Kim, S.; and Kim, J. 2024. VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual

Newman, M. E.; and Watts, D. J. 1999. Renormalization group analysis of the small-world network model. *Physics*

Pourkeshavarz, M.; Sabokrou, M.; and Rasouli, A. 2024. Adversarial Backdoor Attack by Naturalistic Data Poisoning on Trajectory Prediction in Autonomous Driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision*

Ramezani, M.; Haddad, J.; and Geroliminis, N. 2015. Dynamics of heterogeneity in urban networks: aggregated traffic modeling and hierarchical control. *Transportation Re-*

Ren, Y.; Lan, Z.; Liu, L.; and Yu, H. 2024. EMSIN: Enhanced Multi-Stream Interaction Network for Vehicle Trajectory Prediction. *IEEE Transactions on Fuzzy Systems*.

Descriptions. *arXiv preprint arXiv:2407.12345*.

*Joint Conference On Artificial Intelligence*.

*on neural networks and learning systems*.

casting. arXiv:2011.15084.

*posium (IV)*, 975–980. IEEE.

*Letters A*, 263(4-6): 341–346.

*and Pattern Recognition*, 14885–14894.

*search Part B: Methodological*, 74: 1–19.

175–185.

*Artificial Intelligence*.
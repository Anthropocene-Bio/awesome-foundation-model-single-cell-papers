# scPretrain: multi-task self-supervised learning for cell-type classification

Paper: https://academic.oup.com/bioinformatics/article/38/6/1607/6499287
Code: https://github.com/ruiyi-zhang/scPretrain

## Abstract 

### **Motivation**
Single-cell RNA sequencing (scRNA-seq) datasets are being rapidly generated, enabling the study of cellular heterogeneity at single-cell resolution. A key step in single-cell analysis is **cell-type classification**, which involves labeling groups of cells based on their gene expression profiles. Traditional supervised learning methods rely solely on annotated cells, ignoring the vast majority of unannotated cells. To address this limitation, **scPretrain**, a multi-task self-supervised learning approach, is introduced.

### **Method**
scPretrain consists of:
1. **Pre-training Step:** A feature extraction encoder is trained using pseudo-labels derived from unannotated cells via a multi-task learning framework.
2. **Fine-tuning Step:** The pre-trained encoder is fine-tuned using the limited annotated cells in a new dataset.

### **Results**
- scPretrain was evaluated on **60 diverse scRNA-seq datasets** across different technologies, species, and organs.
- Significant improvements were observed in **cell-type classification** and **cell clustering**.
- The representations learned during pre-training also enhanced the performance of conventional classifiers such as **random forest, logistic regression, and support-vector machines**.
- The model effectively utilizes **large-scale unlabeled scRNA-seq data** for improved annotation of new datasets.

## 1 Introduction

### **Background and Motivation**
Recent advances in **single-cell RNA sequencing (scRNA-seq)** have led to the generation of massive datasets, offering unprecedented insights into **cellular heterogeneity and function**. One of the most critical steps in single-cell analysis is **cell-type classification**, which involves grouping and labeling cells based on their gene expression profiles.

However, **manual cell-type annotation** is **labor-intensive, requires domain expertise**, and does not scale well for large datasets. For instance, the **Tabula Muris** dataset was annotated by a team of **39 domain experts**. To accelerate this process, **supervised learning** methods have been adopted, leveraging existing annotated datasets for classification. However, these approaches face **two major limitations**:

1. **Annotated cells are limited** compared to the vast number of **unannotated cells**, which contain valuable gene expression information.
2. **Exclusive reliance on annotated data** can lead to **overfitting** and susceptibility to **batch effects** across different datasets.

### **Self-supervised Learning (SSL) for Cell-type Classification**
Self-supervised learning (SSL) has **revolutionized machine learning** fields such as **natural language processing, video analysis, and computer vision** by leveraging unannotated data in a pre-training step before fine-tuning on labeled data. For instance:
- **BERT (Devlin et al., 2018)** pre-trains a feature encoder using **masked language modeling** and **next-sentence prediction** before fine-tuning on downstream tasks.
- SSL methods have also been applied to **graphs (Hu et al., 2019)** and **images (Chen et al., 2020)**.

#### **Challenges in Applying SSL to scRNA-seq Data**
Despite SSL's success in other domains, its application to **gene expression data** presents unique challenges:
1. **Lack of an ordered structure:** Unlike **text, images, or graphs**, gene expression data lacks an **explicit topological structure**, making traditional SSL techniques (e.g., masked prediction) difficult to apply.
2. **Batch effects:** scRNA-seq datasets exhibit substantial variations due to **different technologies, platforms, species, and organs** (Haghverdi et al., 2018; Li et al., 2020). Without proper handling, **batch effects** can obscure biological signals.

### **Multi-task Learning and Domain Adaptation for scRNA-seq Data**
- **Domain adaptation** techniques allow models trained on one dataset (source) to generalize to another dataset (target) by learning shared representations.
- **Multi-task learning (MTL)** improves generalization by training a model across multiple related tasks, which has been effective in addressing batch effects (Gebru et al., 2017; Ren and Lee, 2018).

#### **Our Approach: scPretrain**
To overcome these challenges, we propose **scPretrain**, a novel **multi-task self-supervised learning (SSL) approach** for cell-type classification. The key components of scPretrain are:

1. **Pre-training Step**:
   - **Unannotated cells** from multiple datasets are clustered using **K-means**, generating **pseudo-labels**.
   - A **feature extraction encoder** is trained using these pseudo-labels in a **multi-task learning framework** to minimize the impact of batch effects.

2. **Fine-tuning Step**:
   - The pre-trained encoder is fine-tuned on a **new dataset with limited labeled cells**, allowing efficient cell-type classification.

#### **Key Contributions**
- Introduces a **multi-task SSL approach** that leverages **both annotated and unannotated scRNA-seq data**.
- Uses **pseudo-labeling via clustering** to guide pre-training.
- Addresses **batch effects** by treating datasets as separate tasks in a multi-task learning framework.
- Demonstrates **significant improvements** in **cell-type classification and clustering** across **60 diverse scRNA-seq datasets**.

## 2 Materials and Methods

### **2.1 Problem Definition**
scPretrain consists of two main steps: **pre-training** and **fine-tuning**.

#### **Pre-training Step**
- The input consists of multiple **unlabeled scRNA-seq datasets**:
  \[
  D = \{X_1, X_2, ..., X_g\}, \quad X_i \in \mathbb{R}^{n_i \times s} \quad (1 \leq i \leq g)
  \]
  where:
  - \(X_i\) is a gene expression matrix of **\(n_i\) cells** and **\(s\) genes** for dataset \(i\).
  - The goal is to learn a **feature extraction encoder** \(E_h(x)\), which maps each gene expression vector \(x \in \mathbb{R}^{s}\) to a low-dimensional representation \(h \in \mathbb{R}^{r}\).

- This feature extraction encoder **\(E_h\)** is shared across all unlabeled datasets, enabling it to capture generalizable gene expression patterns.

#### **Fine-tuning Step**
- The input consists of an **annotated dataset**:
  \[
  Z \in \mathbb{R}^{m \times s}, \quad Y \in \mathbb{R}^{m \times k}
  \]
  where:
  - \(Z\) is the gene expression matrix of **\(m\) cells** and **\(s\) genes**.
  - \(Y\) is the **binary cell-type label matrix**, with **\(k\) cell types**.
  - \(Y_{ij} = 1\) if the \(i\)-th cell belongs to cell type \(j\), otherwise \(Y_{ij} = 0\).

- The objective is to **fine-tune** the feature extraction encoder \(E_h\) using \(Y\), learning a classifier to predict **cell types** in a new test dataset \(Q\).

- **Key Considerations:**
  - The same encoder \(E_h\) is shared between **pre-training and fine-tuning**, ensuring **robust feature extraction**.
  - This approach **mitigates batch effects**, making scPretrain **scalable to large and diverse single-cell datasets**.

---

### **2.2 Datasets**
To evaluate scPretrain, we obtained **60 diverse scRNA-seq datasets** from **Cell BLAST (Cao et al., 2020)**.

#### **Dataset Selection Criteria**
- **Species**: Only **human** and **mouse** datasets were selected.
- **Organs**: Covered **29 different organs** to ensure diversity.
- **Platforms**: Included **10 different sequencing technologies** to account for variations in data acquisition.

#### **Dataset Characteristics**
- Number of cells per dataset: **870 to 160,796** (mean: **15,209**).
- Number of cell types per dataset: **2 to 81** (mean: **10**).
- **Cross-species gene mapping**:
  - Union of all genes across datasets for each species.
  - Intersection of **mouse and human genes** resulted in **17,298 common genes**.
  - **Zero imputation** was used for missing genes in specific datasets.

#### **Pre-training and Fine-tuning Data Processing**
- **Pre-training Step**:
  - Used the **expression vectors of 17,298 genes**.
  - Zero imputation for missing values.
- **Fine-tuning Step:**  
  - The weight matrix was adjusted based on the genes used during pre-training.  
  - For genes not included in the **17,298 pre-training genes**, their parameters were randomly initialized.  

---

### **2.3 Pre-training Using Pseudo-labels**
The goal of the **pre-training step** is to learn a **feature extraction encoder** \(E_h(x)\), which can generalize well to various downstream tasks such as cell-type classification and clustering.

#### **Generating Pseudo-labels with Clustering**
- Given a gene expression matrix \(X \in \mathbb{R}^{n \times s}\) (**\(n\) cells** and **\(s\) genes**), we aim to train \(E_h(x)\) without using labeled data.
- Since real cell-type labels are unavailable, we generate **pseudo-labels** using **K-means clustering**.
- The clustering process assigns each cell to a cluster, which serves as its **pseudo-label**.

#### **Training the Feature Extraction Encoder**
- Once pseudo-labels are assigned, they are used to train \(E_h(x)\) in a **self-supervised learning (SSL) framework**.
- The encoder is modeled as a **single-layer fully connected neural network**.
- A classifier \(F_x(h)\) maps the encoded features to predicted pseudo-labels:
  \[
  F_x(h) = F_1(\sigma(h))
  \]
  where:
  - \(h = E_h(x)\) is the encoded representation of a cell.
  - \(\sigma(h)\) is the **sigmoid activation function**.
  - \(F_1\) is a **fully connected layer** that outputs a probability distribution over pseudo-labels.

#### **Loss Function and Optimization**
- The classifier is trained using a **cross-entropy loss function**:
  \[
  L = - \sum_{i=1}^{k'} y_i \log v_i
  \]
  where:
  - \(y_i\) is the pseudo-label for the \(i\)-th cell.
  - \(v_i\) is the predicted probability that the cell belongs to cluster \(i\).

- The encoder \(E_h(x)\) is optimized by minimizing:
  \[
  \min_{h,x} L(F_x(E_h(X)), Y')
  \]
  where \(Y'\) represents the **pseudo-label matrix**.

#### **Iterative Refinement of Pseudo-labels**
The pre-training follows an **iterative self-labeling process**, where pseudo-labels are refined over multiple steps to improve clustering quality.

##### **Step 1: Initial Clustering on Raw Features**
- The **K-means algorithm** is applied to the raw gene expression matrix \(X\):
  \[
  \min_{C, Y'} \| X - Y'C \|^2 \quad \text{s.t.} \quad Y' 1_{k'} = 1_n
  \]
  where:
  - \(C \in \mathbb{R}^{k' \times s}\) contains the **cluster centroids**.
  - \(Y' \in \{0,1\}^{n \times k'}\) is the pseudo-label matrix, where \(Y'_{ij} = 1\) if cell \(i\) belongs to cluster \(j\).
  - \(1_{k'}\) is a vector of ones.

##### **Step 2: Feature Learning via Classification**
- The encoder \(E_h(x)\) is trained to minimize the classification loss:
  \[
  \min_{h, x} L(F_x(E_h(X)), Y')
  \]
  The trained encoder learns **low-dimensional cell representations** that better separate cell types.

##### **Step 3: Clustering on Learned Representations**
- Instead of using the raw features, K-means is now applied to the **encoded representations**:
  \[
  \min_{C', Y'} \| E_h(X) - Y'C' \|^2 \quad \text{s.t.} \quad Y' 1_{k'} = 1_n
  \]
  where:
  - \(C' \in \mathbb{R}^{k' \times r}\) contains the new centroids in the **low-dimensional feature space**.
  - \(r\) is the output dimension of \(E_h(x)\).

##### **Step 4: Iterative Update of Pseudo-labels**
- The newly computed pseudo-labels \(Y'\) are used to retrain \(E_h(x)\).
- Steps **2 and 3** are repeated until convergence.

##### **Convergence Criteria**
The iterative process continues until:
1. The **pseudo-labels remain stable** between iterations, i.e.,
   \[
   \frac{\| Y'^{(t)} - Y'^{(t-1)} \|_1}{n} < \epsilon
   \]
   where \(\epsilon\) is a small threshold.
2. The encoder reaches a performance plateau on a held-out validation set.

#### **Why Iterative Clustering?**
- Initially, clustering is based on **raw gene expression**.
- As training progresses, the encoder learns **better feature representations**, leading to **more accurate clusters**.
- This process **enhances the encoder’s discriminative power** by refining the pseudo-labels iteratively.

---

## **2.4 Multi-task Pre-training**
Batch effects introduce variability in gene expression profiles, making it challenging to generalize across datasets. To address this, **scPretrain employs a multi-task learning framework** that treats each dataset as a separate task.

### **Objective of Multi-task Learning**
- Given \( g \) datasets \( D = \{X_1, X_2, ..., X_g\} \) with their pseudo-labels \( \tilde{Y} = \{\tilde{Y}_1, \tilde{Y}_2, ..., \tilde{Y}_g\} \), we jointly minimize the classification loss across all datasets:
  \[
  \min_{h, x} \sum_{i=1}^{g} L(F_{x_i}(E_h(X_i)), \tilde{Y}_i)
  \]
  where:
  - \( E_h(x) \) is the shared feature extraction encoder.
  - Each dataset \( X_i \) has its own classifier \( F_{x_i} \).
  - The loss function \( L \) is the **cross-entropy loss**.

### **Two-phase Multi-task Training**
1. **First Phase: Pseudo-label Generation on Raw Data**
   - Cluster each dataset \( X_i \) independently using K-means:
     \[
     \min_{C_i, \tilde{Y}_i} \| X_i - \tilde{Y}_i C_i \|^2, \quad \text{s.t.} \quad \tilde{Y}_i 1_{\tilde{k}_i} = 1_{n_i}
     \]
   - \( C_i \) represents the cluster centroids for dataset \( X_i \).

2. **Second Phase: Iterative Refinement with Feature Learning**
   - Cluster on the feature representations instead:
     \[
     \min_{C_i', \tilde{Y}_i} \| E_h(X_i) - \tilde{Y}_i C_i' \|^2, \quad \text{s.t.} \quad \tilde{Y}_i 1_{\tilde{k}_i} = 1_{n_i}
     \]
   - The feature extractor is trained iteratively across datasets, preventing overfitting to batch-specific effects.

### **Key Benefits of Multi-task Pre-training**
- Forces the encoder to learn **generalizable features** rather than dataset-specific batch effects.
- Prevents **trivial cluster assignments** (e.g., all cells belonging to one cluster).
- Improves transferability to **new scRNA-seq datasets**.

---

## **2.5 Determining the Number of Clusters in Each Task**
A key hyperparameter in K-means clustering is the number of clusters \( \tilde{k} \), which is dataset-dependent. Instead of fixing \( \tilde{k} \), **scPretrain considers multiple partitions per dataset**.

### **Adaptive Clustering Strategy**
- Each dataset \( X_i \) is partitioned using three different values of \( \tilde{k} \):
  \[
  \tilde{k}_{i1} = \frac{n_i}{50}, \quad \tilde{k}_{i2} = \frac{n_i}{100}, \quad \tilde{k}_{i3} = \frac{n_i}{200}
  \]
  where:
  - \( \tilde{k}_{ij} \) is the number of clusters for partition \( j \).
  - This results in \( g \times p \) tasks (where \( p = 3 \) is the number of partitions per dataset).

- The clustering loss function is modified to account for different partitions:
  \[
  \min_{C_{ij}, \tilde{Y}_{ij}} \| X_i - \tilde{Y}_{ij} C_{ij} \|^2, \quad \text{s.t.} \quad \tilde{Y}_{ij} 1_{\tilde{k}_{ij}} = 1_{n_i}, \quad 1 \leq j \leq p
  \]

### **Advantages of Multi-scale Clustering**
- Encourages the encoder to be **invariant to the number of clusters**.
- Prevents **over-segmentation or under-segmentation** of cells.
- Provides a **more robust pre-training process**, leading to better generalization.

---

## **2.6 Fine-tuning for Cell-type Classification**
Once the encoder has been pre-trained, it is fine-tuned using **a small number of labeled samples** to perform cell-type classification.

### **Fine-tuning with a Supervised Classifier**
- Given a labeled dataset \( (Z, Y) \):
  - \( Z \in \mathbb{R}^{m \times s} \) (gene expression matrix).
  - \( Y \in \mathbb{R}^{m \times k} \) (binary cell-type labels).
- The pre-trained encoder is used to map each cell to a lower-dimensional representation:
  \[
  h = E_h(Z)
  \]
- A **deeper classifier** is trained on \( h \):
  \[
  F_{\phi}(h) = F_a(\sigma(F_b(\sigma(h))))
  \]
  where:
  - \( F_a \) and \( F_b \) are **fully connected layers**.
  - \( \sigma \) is the **ReLU activation function**.

### **Loss Function for Fine-tuning**
The classifier is optimized using the cross-entropy loss:
\[
\min_{h, \phi} L(F_{\phi}(E_h(Z)), Y)
\]
where \( \phi \) represents the parameters of the fine-tuned classifier.

### **Alternative Classifiers**
If desired, the extracted cell representations \( h \) can be used with conventional classifiers:
- **Logistic Regression**:
  \[
  P(y=j|h) = \frac{e^{w_j^T h}}{\sum_{j'} e^{w_{j'}^T h}}
  \]
- **Support Vector Machine (SVM)**:
  \[
  \min_w \sum_i \max(0, 1 - y_i (w^T h_i + b))
  \]
- **Random Forest** trained on \( h \).

### **Key Benefits of Fine-tuning**
- Enables rapid adaptation to **new cell-type classification tasks**.
- Utilizes only a **small number of labeled samples** for fine-tuning.
- The learned **cell representations** can be used with various classifiers.

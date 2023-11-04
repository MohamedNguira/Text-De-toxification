# Practical Machine Learning and Deep Learning
## Text-Detoxification Assignment
### Report 2
### By Mohamed Nguira


## Contents:
 1. Introduction
 2. Data analysis
 3. Model Specification
 4. Training Process
 5. Evaluation
 6. Results

#### Resources used:

https://github.com/s-nlp/paradetox
https://arxiv.org/pdf/1503.02531.pdf
https://aclanthology.org/N18-1169.pdf

## 1) Introduction:

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

> Formal definition of text detoxification task can be found in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

Your assignment is to create a solution for detoxing text with high level of toxicity. It can be a model or set of models, or any algorithm that would work. 


The proposed solution contains a constructive algorithm that uses preprocessing, data analysis and various types of pretrained and untrained models.
Our problem is a NLP problem so we will focus on the idea of using unigrams and bigrams as they are considered strong tools for any NLP based algorithm.
We recall that Unigrams are the simplest and most basic form of text representation. They are essentially individual words or tokens in a given text. Each word in a text is considered independently as a unigram. On the other hand, Bigrams are a step beyond unigrams and involve grouping consecutive pairs of words in a text. In other words, bigrams consist of two adjacent words in a text, considered together.
We will here use a combination of both.
First of all, the data was arranged in such a way that the 1st pair always has a greater toxicity score than the 2nd pair. 


## 2) Data Analysis:

Multiple ideas were took into consideration to make data analysis:

### a) Sequence to sequence Data analysis

#### Sequence Length Analysis: 
Analyze the distribution of sequence lengths in both the source and target data. This can help you determine if there are any outliers or if you need to pad/truncate sequences.

#### Data Preprocessing: 
Understand the tokenization and preprocessing steps applied to your data. Ensure that the text is cleaned, tokenized, and encoded properly.

#### Vocabulary Analysis: 
Examine the size of the source and target vocabularies. A large vocabulary can impact model training and inference times.
#### Data Visualization:
Use visualizations to explore the data, such as word clouds, histograms, or scatter plots for understanding patterns or relationships in the data.

These were the results of that analysis:

![image](https://hackmd.io/_uploads/rkavPao76.png)
![image](https://hackmd.io/_uploads/ryOdPpiQa.png)




### b) Sequence to Scalar Data Analysis (Toxicity level):

#### Input-Output Correlation: 
Examine the correlation between input sequences and corresponding scalar values. Visualizations like scatter plots can help you understand this relationship.

#### Scalar Value Distribution: 
Analyze the distribution of scalar values. This can help you identify any outliers or potential data issues.

#### Feature Engineering:
Explore the possibility of extracting relevant features from the input sequences that may be predictive of the scalar values.

#### Regression Analysis: 
Use regression analysis techniques to understand how well the input sequences can predict the scalar values. This may include correlation coefficients and regression plots.

#### Feature Importance: 
If applicable, analyze feature importance to determine which elements in the input sequences contribute the most to the scalar predictions. This can guide feature selection and engineering efforts.

These were the results of that analysis:

![image](https://hackmd.io/_uploads/By9Yvajm6.png)

![image](https://hackmd.io/_uploads/BJacv6sX6.png)
![image](https://hackmd.io/_uploads/BJbivpom6.png)



## 3) Model Specification:


The overall system aims to address the challenge of reducing the toxicity level of given text while preserving its original meaning. It consists of three interconnected components:

#### Hypothesis 1 - Toxicity Detection

Task: Develop a model for detecting the level of toxicity in a given text.
Approach: Utilize the RoBERTa model, a variant of BERT, pre-trained on a vast amount of internet text data. RoBERTa employs advanced techniques like sentence-level and document-level masked language modeling for robustness.
Use Cases: The toxicity detection model will be used to evaluate the output of the main algorithm and to reduce the toxicity level of the source text.
#### Hypothesis 2 - Summarization and Stop Word Analysis

Task: Summarize the source text and analyze the correlation between the level of toxicity and the presence of stop words.
Approach: Employ the T5-model, a versatile text-to-text transformer-based model designed to handle various NLP tasks. It's used for text summarization and stop word analysis.
Stop Word Analysis: Investigate the number of stop words, the ratio of stop words to all tokens, and the average length of non-stop words in sentences. Perform feature importance analysis using RandomForestRegressor to identify factors affecting text toxicity.
#### Hypothesis 3 - Seq2Seq Model with Penalty

Task: Create a sequence-to-sequence model that penalizes toxic sentences and dissimilarity between sentences.
Approach: Implement the BART model (Bidirectional and Auto-Regressive Transformers). BART is well-suited for this task as it can understand context from both left and right, making it effective for preserving the original meaning. Fine-tune BART for text detoxification.
Penalty: Penalize toxic phrases using the model from Hypothesis 1 and phrases that lose the meaning of the original text.
Use Cases: Generate non-toxic phrases while conserving the original meaning. This model helps address the core problem of text detoxification by incorporating penalties for toxicity and meaning loss.
#### The system is interconnected as follows:

Hypothesis 1 is responsible for detecting the toxicity level of input text.
Hypothesis 2 contributes by summarizing the text and analyzing stop words, which may correlate with toxicity.
Hypothesis 3 utilizes the findings from Hypotheses 1 and 2, applying penalties for toxicity and meaning loss, and generating non-toxic alternatives.
The interconnected nature of the system ensures that it leverages the results of each hypothesis to provide a holistic solution for reducing toxicity while maintaining the original text's meaning. The use of advanced pre-trained models and fine-tuning for specific NLP tasks enhances the system's ability to address the problem effectively.

## 4) Training Process:

The training process for the three models in the system involves several common components such as data preparation, model architecture, loss calculation, batch processing, learning rate scheduling, and optimization. 

#### a) Toxicity Detection (RoBERTa Model):

Data Preparation: Prepare a dataset with labeled examples of toxic and non-toxic text. Tokenize the text and generate input sequences.

Model Architecture: Use the RoBERTa architecture as a pre-trained base model, typically with additional layers for classification.

Loss Score: For binary classification tasks (toxic or non-toxic), you can use the binary cross-entropy loss (logistic loss):

$$Loss = - y log(p) + (1 - y)log(1 - p)$$

Batch Size: 32

Learning Rate: 2e-5

Optimizer: The Adam optimizer is commonly used for fine-tuning

#### Summarization and Stop Word Analysis (T5 Model):

Data Preparation: Prepare a dataset for text summarization and stop word analysis.

Model Architecture: Utilize the T5 architecture, which is designed for text-to-text tasks.

Loss Score for Summarization: For text summarization, a common loss function is the token-wise cross-entropy loss.

Loss Score for Stop Word Analysis: Define a suitable loss function for regression tasks based on the statistical features of the data.

Batch Size: 32
Learning Rate: 2e-5
Optimizer: Adam optimizer

#### Seq2Seq Model with Penalty (BART Model):

Data Preparation: Prepare a dataset with paired sentences for sequence-to-sequence tasks.
Model Architecture: Use the BART architecture with a text generation head.
Loss Score: A combination of losses for toxicity and meaning preservation penalties.
Batch Size: 32
Learning Rate: 2e-5
Optimizer: Adam optimizer

## 5) Evaluation:

#### 1. BLEU Score (0.454):
The BLEU score is used to measure the similarity between the sentences generated by the model and the reference sentences. It quantifies how closely the generated text aligns with the expected output. In our case, the model achieved a BLEU score of 0.454, indicating moderate alignment with the reference sentences.

#### 2. SIM Score (0.714):
The SIM score evaluates the semantic similarity between the generated sentences and the reference sentences by using sentence embeddings. A higher SIM score (0.714 in our case) signifies that the generated text captures the underlying meaning of the source text effectively.

#### 3. ACC Score (0.792):
The ACC score measures the accuracy of style transfer between the generated sentences. It assesses how well the model is at changing the writing style while preserving the meaning. Our model achieved an ACC score of 0.792, indicating a reasonably high accuracy in style transfer.

#### 4. FL Score (0.824):
The FL score assesses the fluency of the generated sentences. It evaluates how smoothly the generated text flows and reads to human readers. An FL score of 0.824 suggests that the generated text is fairly fluent and coherent.

#### 5. J Score (0.48):
The J score combines the last three metrics, which are ACC, FL, and SIM, into a single measure. it serves as an aggregate assessment of the model's overall performance in terms of style transfer accuracy, fluency, and semantic similarity.

## 6) Results:

Here are some texts where the model was tested:

#### Examples:

![image](https://hackmd.io/_uploads/HymNBCiXa.png)

#### Results:
![image](https://hackmd.io/_uploads/B1eHBRi7p.png)

In conclusion, our model demonstrates a reasonable level of efficiency in addressing the task of text detoxification. The combination of metrics, including BLEU, SIM, ACC, FL, and J scores, highlights its ability to align with reference sentences, capture semantic similarity, perform accurate style transfer, and maintain fluency to a satisfactory degree.

While the model's performance is promising, there is room for improvement, particularly in enhancing fluency and increasing the J score. Further fine-tuning, parameter optimization, and potentially more extensive training may contribute to a more efficient and refined model. Overall, our model represents a solid foundation for text detoxification, but continuous refinement and optimization efforts can lead to even more efficient results in the future.

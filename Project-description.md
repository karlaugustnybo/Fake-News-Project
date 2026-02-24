# Fake News Project

## Introduction
The goal of this project is to create a fake news prediction system. Fake news is a major problem that can have serious negative effects on how people understand the world around them. You will work with a dataset containing real and fake news in order to train a simple and a more advanced classifier to solve this problem. This project covers the full Data Science pipeline, from data processing, to modelling, to visualization and interpretation.

## Revision History
- **Tuesday 2 February**: Initial version of project description.
- **Tuesday 2 February**: Updated Part 0 to make it clear that project groups should contain three people this year.

## Practicalities
- **Deadline**: Your report is due by 1600 on Friday 27 March, and it must be submitted via Digital Exam. Make sure that you submit as a group. In order to do this, you will need to create your group under the "People" section. I strongly recommend that you create these groups well before the project deadline, so we can deal with any technical problems early on. Please let us know (by starting a discussion in Absalon) if something is not working.
- **Time Management**: Each part of the project has a recommended date by which you should have most of the work completed. You may need to revisit some parts throughout the project as you develop more advanced models and need more features, for example. Resist the temptation to attempt to complete the whole project in the final week.
- **Length**: The final report should be handed in as a PDF file. Your report has a maximum length of 6 pages in the format defined below. We've added the expected number of pages in the header of each question to indicate how much we expect you to write in each section. In addition to text, you can add tables, figures and code, but you should not exceed the maximum length. You are allowed to provide appendices as well, but we won't guarantee that we will consider them.
- **Format**: The report must be written using this LaTeX template: [style-and-template-for-preprints-arxiv-bio-arxiv](https://da.overleaf.com/latex/templates/style-and-template-for-preprints-arxiv-bio-arxiv/fxsnsrzpnvwc). Structure your report so that the section numbers correspond to the parts. Since you only have very limited space, you don’t have to write an introduction. The short format means you will need to prioritize what you include - include the most central observations in the main text, and refer to less important things in the appendix if necessary. We expect the report to be well-organized, technically precise, and polished (e.g., typo-free). Note that your report does not need an Abstract.

**Important**: If you distribute tasks such that some in the group are primarily responsible for particular parts of the report, please explicitly state so in the report (using percentages to specify contribution from each member). If no such annotation is provided, we will assume that all members in the group contributed equally to all parts of the report. Finally, please state the KUIDs of all members of the group on the front page of the report.

- **Code**: Your group should make their code available for the examination. You can either upload the code or include a link to a Github repository in your report. Your project may have created some large intermediate files that you cannot easily upload anywhere. Therefore, your code should include a README file that clearly explains how to run your code and reproduce the results claimed in your report.

---

## Part 0: Form Study Groups
**[End of Week 6]**

If you haven't already done so, the first task is to form a study group of a maximum, and minimum of three people. There is an announcement on Absalon describing how to do this. If you need to form a group with fewer than four people, you must write to the Course Responsible to explain why you need dispensation. Make sure that you list the names of all members of the group at the top of the final report, along with your group number. You should also make the contributions of each group member clear.

---

## Part 1: Data Processing (~1 page)
**[End of Week 8]**

In the first part of the project, you should work on retrieving, structuring, and cleaning data.

You will be using a subset of the FakeNewsCorpus dataset in your project, which is available from Absalon. You can also find more information about the [full dataset](https://github.com/several27/FakeNewsCorpus) and find information about how the data is collected, the available fields, etc.

### Task 1
Your first task is to retrieve a sample of the FakeNewsCorpus from [news_sample.csv](https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv) and structure, process, clean it. You should follow the methodology you developed in Exercise 1. When you have finished cleaning, you can start to process the text. [NLTK](https://www.nltk.org/) has built-in support for many common operations. Try the following:
1. Tokenize the text.
2. Remove stopwords and compute the size of the vocabulary. Compute the reduction rate of the vocabulary size after removing stopwords.
3. Remove word variations with stemming and compute the size of the vocabulary. Compute the reduction rate of the vocabulary size after stemming.

Describe which procedures (and which libraries) you used and why they are appropriate.

### Task 2
Apply your data preprocessing pipeline to the 995,000 rows sampled from the FakeNewsCorpus: 995K FakeNewsCorpus subset.

### Task 3
Now try to explore your processed version of the 995K dataset. Make at least three non-trivial observations/discoveries about the data. These observations could be related to outliers, artefacts, or even better: genuinely interesting patterns in the data that could potentially be used for fake-news detection. Examples of simple observations could be how many missing values there are in particular columns - or what the distribution over domains is. Be creative!

- Describe how you ended up representing the FakeNewsCorpus dataset (for instance with a Pandas dataframe). Argue for why you chose this design.
- Did you discover any inherent problems with the data while working with it?
- Report key properties of the data set - for instance through statistics or visualization.

The exploration can include (but need not be limited to):
- Counting the number of URLs in the content.
- Counting the number of dates in the content.
- Counting the number of numeric values in the content.
- Determining the 100 more frequent words that appear in the content.
- Plot the frequency of the 10000 most frequent words (any interesting patterns?).
- Run the analysis in point 4 and 5 both before and after removing stopwords and applying stemming: do you see any difference?

### Task 4
Split the resulting dataset into a training, validation, and test splits. A common strategy is to uniformly at random split the data 80% / 10% / 10%. You will use the training data to train your baseline and advanced models, the validation data can be used for model selection and hyperparameter tuning, while the test data should only be used in Part 4.

---

## Part 2: Simple Logistic Regression Model (~1 page)
**[End of Week 10]**

You should create and train a baseline model for your Fake News predictor that performs binary classification to predict whether an article is reliable or fake.

### Task 1
Briefly discuss how you grouped the labels into two groups. Are there any limitations that could arise from the decisions you made when grouping the labels?

### Task 2
Start by implementing and training a simple logistic regression classifier using a fixed vocabulary of the 10,000 most frequent words extracted from the content field, as the input features. You do not need to apply TF-IDF weighting to the features. It should take no more than five minutes to fit this model on a modern laptop, and, as guidance, it should be possible to achieve an F1 score of ~94% on your test split. However, this F1 score is based on certain assumptions about how you have split your data and will not apply to every group. Do not worry too much if the F1 score with your logistic regression model is substantially below 94%. Write in your report the performance that you achieve with your implementation of this model, and remember to report any hyper-parameters used for the training process.

### Task 3
Consider whether it would make sense to include meta-data features as well. If so, which ones, and why? If relevant, report the performance when including these additional features and compare it to the first baselines. Discuss whether these results match your expectations.

For the remainder of the project, we will limit ourselves to main-text data only (i.e. no meta-data). This makes it easier to do the cross-domain experiment in Part 4 (which does not have the same set of meta-data fields).

---

## Part 3: Advanced Model (~1 page)
**[End of Week 11]**

Create the best Fake News predictor that you can come up with. This should be a more complex model than the simple logistic regression model you created in Part 2, either in the sense that it uses a more advanced method, or because it uses a more elaborate set of features. For example, you might consider using a Support Vector Machine, a Naive Bayes Classifier, or a neural network. The input features might use more complex text representations, such as TF-IDF weights or continuous word embeddings. Report necessary details about your models ensuring full reproducibility. This could include, for example, the choice of relevant parameters and how you chose them. Make sure to argue for why you chose this approach over potential alternatives.

**Optional**: If you want to go even further, you might want to try training your models on even more data. The [full FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus) is a total of 9GB of source material available for training your model. You will need to use a multi-part decompression tool, e.g. 7z. Given all the files, execute the following command: `7z x news.csv.zip`. This should create a 27GB file on disk (29.322.513.705 bytes). You may find it challenging to run your data processing pipeline on the entire FakeNewsCorpus, so take care if you attempt this step.

---

## Part 4: Evaluation (~1 page)
**[End of Week 12]**

You should now evaluate your models on the FakeNews and the LIAR dataset. Arrange all these results in a table to facilitate a comparison between them. You should be evaluating the model on how well it classifies articles correctly using F-score. You may want to include a confusion matrix to visualize the types of classification errors made by your models.

### Task 1
Evaluate the performance of your Simple and Advanced Models on your FakeNewsCorpus test set. It should be possible to achieve > 80% accuracy but you will not fail the project if your model cannot reach this performance.

### Task 2
In order to allow you to play around cross-domain performance, try the same exercise on the [LIAR dataset](https://github.com/runqi-yang/LIAR), where you know the labels, and can thus immediately calculate the performance. You are expected to directly evaluate the model you trained on the FakeNewsCorpus. In other words, you do not need to retrain the model on the LIAR dataset.

### Task 3
Compare the results of this experiment to the results you obtained in Task 1. Report your LIAR results as part of your report. Remember to test the performance of both your Simple and Advanced Model on the LIAR dataset.

---

## Part 5: Conclusions (~0.5 page)
**[End of Week 13]**

Conclude your report by discussing the results you obtained. Explain the discrepancy between the performance on your test set and on the LIAR set. If relevant, use visualizations or report relevant statistics to point out differences in the datasets. Discuss the issues about sample bias when evaluating on a different distribution of data than the training data. Conclude with describing overall lessons learned during the project, for instance considering questions like: Does the discrepancy between performance on different data sets surprise you? What can be done to improve the performance of Fake News prediction? Will further progress be driven primarily by better models or by better data?

Please note that the general discussion is not merely a summary of what you have done in the other questions. We expect to see some non-trivial reflection in this section.

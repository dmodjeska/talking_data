## Introduction

During the period March-May 2018, Kaggle held a competition called TalkingData AdTracking Fraud Detection Challenge. The goal was to detect fraud in clicks on mobile app ads with a large Chinese big-data services company. A full description of this competition is below.

Because the training data contains almost 200 million records, this solution uses Spark for scale. The solution was run on a 5-computer cluster using Amazon's Elastic Map Reduce (EMR) service.

Both logistic regression and random forest generated reasonable predictions with data subsets. Logistic regression scaled most easily to the full data set, so this algorithm was ultimately used.

## Description of Competition (from Kaggle Site)

Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user’s click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

While successful, they want to always be one step ahead of fraudsters and have turned to the Kaggle community for help in further developing their solution. In their 2nd competition with Kaggle, you’re challenged to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad. To support your modeling, they have provided a generous dataset covering approximately 200 million clicks over 4 days!

## Code

* [Data Processing](https://github.com/dmodjeska/talking_data/blob/master/talkingdata_process_spark_18May2018.py)
* [Data Modeling](talkingdata_model_spark_18May2018.py)
* [Submission Preparation](https://github.com/dmodjeska/talking_data/blob/master/submission_ar_to_df.py)



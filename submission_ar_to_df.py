
# Author: David Modjeska
# Project: Kaggle Competition - TalkingData AdTracking Fraud Detection Challenge
# File: prepare the submission


import pandas as pd

submission_data = pd.read_csv('submission_data_ar.csv', header = None)
print('ARRAY')
print()
print(submission_data[:5])

submission_data.columns = ['click_id', 'is_attributed']
print()

submission_data.to_csv('submission_data_df.csv', index = False)
print('DATAFRAME')
print()
print(submission_data.shape)
print()
print(submission_data.head(5))

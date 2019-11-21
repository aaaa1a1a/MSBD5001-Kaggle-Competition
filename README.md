# MSBD5001-Kaggle-Competition
Team: In case I forget...

Name: Alan Yu Hsien Lai
SID: 20670083

Result file: submission.csv
Code: submit_clean.py
Programming requirement: Python with installed packages (numpy, pandas, sklearn)

Roadmap to the final submission:
1. Review training and testing data sets to understand the structures of the files. Eyeballing each feature in order to look for patterns.
2. Data preprocessing: filling missing data in necessary columns and determine features (is_free, genres, categories, tags, reviews etc.) which are important to modeling.
3. Convert genres, categories, tags to dummy columns making the inside values usable.
4. Ensure the alignment between columns in training and testing data (if one is missing, fill '0' for all rows) in order to prevent crash in the modeling part later.
5. Random Forest was chosen as the most confident prediction method after comparing with others.
6. Set up the final model with training data then fit the testing. Result was hence generated.

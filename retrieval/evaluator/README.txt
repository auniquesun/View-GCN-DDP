In order to run this evaluation code you must have node.js installed on your system (https://nodejs.org/en/).

You can then run it on Linux or OS X systems using:
./evaluate.js path/to/dir/with/results/methodName/

======================================
For example:

node --max-old-space-size=8192 evaluate.js viewformer/
======================================

or on Windows systems using:
node --max-old-space-size=8192 evaluate.js C:\path\to\dir\with\results\methodName\

The script will look for subdirectories in methodName that match test_normal, train_normal etc and contain the ranked list result files for each set to compute evaluation statistics for each set.  The code will print out a csv format set of evaluation metric statistics for each category, as well as a micro-average and macro-average across all categories.  The same information is saved in a 'methodName.summary.csv' file in the working directory.  In addition, a 'methodName.PR.csv' file is saved with precision-recall values that can be used to generate a Precision-Recall plot.

Please contact the organizers at shrecshapenet@gmail.com if you have any questions or issues.

UPDATES:
2017-02-23 - Fix normalization bug for mAP and NDCG metrics (mAP and NDCG need to be divided by total number of relevant models in ground truth set, not in the retrieval list itself)
2016-03-07 - Fix recall denominator computation bug (total number of relevant models was previously computed over entire dataset, not across particular train/val/test split)
2016-03-03 - Several fixes for more robust calculation and handling of corner cases such as zero P+R, and zero IDCG. Provide evaluate.js script to make generation of results easier.
2016-03-01 - Fixed aggregation bug for P@N, R@N, and F1@N when N (retrieval list length) varies. Also added writeOracleResults function to Evaluator

## Results:
```
hidden_dim: 128 (pretrain; no freezed encoder)
mse: 0.5150
f1_score: 0.9688
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.96      0.97      0.96      1032
           3       0.96      0.96      0.96      1010
           4       0.96      0.99      0.97       982
           5       0.97      0.94      0.95       892
           6       0.97      0.98      0.98       958
           7       0.97      0.97      0.97      1028
           8       0.96      0.97      0.96       974
           9       0.97      0.94      0.96      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000
```

```
hidden_dim: 128 (pretrain; freezed encoder)
f1_score: 0.8361
              precision    recall  f1-score   support

           0       0.92      0.94      0.93       980
           1       0.97      0.97      0.97      1135
           2       0.85      0.84      0.85      1032
           3       0.63      0.76      0.69      1010
           4       0.90      0.87      0.89       982
           5       0.65      0.57      0.61       892
           6       0.85      0.92      0.88       958
           7       0.90      0.85      0.87      1028
           8       0.86      0.71      0.78       974
           9       0.82      0.89      0.85      1009

    accuracy                           0.84     10000
   macro avg       0.84      0.83      0.83     10000
weighted avg       0.84      0.84      0.84     10000
```

```
hidden_dim: 128 (no pretrain)
f1_score: 0.9738
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.96      0.97      0.96      1032
           3       0.96      0.96      0.96      1010
           4       0.99      0.99      0.99       982
           5       0.97      0.95      0.96       892
           6       0.98      0.98      0.98       958
           7       0.97      0.97      0.97      1028
           8       0.97      0.96      0.96       974
           9       0.97      0.97      0.97      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000
```


```
hidden_dim: 64 (pretrain; no freezed encoder)
mse: 0.5120
f1_score: 0.9641
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.95      0.97      0.96      1032
           3       0.97      0.93      0.95      1010
           4       0.96      0.98      0.97       982
           5       0.95      0.94      0.95       892
           6       0.97      0.97      0.97       958
           7       0.95      0.98      0.96      1028
           8       0.95      0.97      0.96       974
           9       0.98      0.93      0.95      1009

    accuracy                           0.96     10000
   macro avg       0.96      0.96      0.96     10000
weighted avg       0.96      0.96      0.96     10000

We have more errors with 3, 6, 7, 9
```

```
hidden_dim: 64 (pretrain; freezed encoder)
f1_score: 0.8322
             precision    recall  f1-score   support

           0       0.92      0.92      0.92       980
           1       0.95      0.96      0.95      1135
           2       0.84      0.82      0.83      1032
           3       0.69      0.74      0.71      1010
           4       0.84      0.92      0.88       982
           5       0.65      0.53      0.59       892
           6       0.85      0.92      0.88       958
           7       0.85      0.86      0.85      1028
           8       0.80      0.75      0.78       974
           9       0.87      0.85      0.86      1009

    accuracy                           0.83     10000
   macro avg       0.83      0.83      0.83     10000
weighted avg       0.83      0.83      0.83     10000
```

```
hidden_dim: 64 (no pretrain)
f1_score: 0.6731
              precision    recall  f1-score   support

           0       0.79      0.82      0.81       980
           1       0.84      0.95      0.89      1135
           2       0.72      0.62      0.66      1032
           3       0.60      0.68      0.64      1010
           4       0.61      0.70      0.65       982
           5       0.50      0.38      0.43       892
           6       0.68      0.69      0.68       958
           7       0.70      0.69      0.69      1028
           8       0.61      0.60      0.60       974
           9       0.60      0.54      0.57      1009

    accuracy                           0.67     10000
   macro avg       0.66      0.67      0.66     10000
weighted avg       0.67      0.67      0.67     10000
```

```
hidden_dim: 10 (pretrain; no freezed encoder)
mse: 0.6307
f1_score: 0.7101
             precision    recall  f1-score   support

           0       0.81      0.87      0.84       980
           1       0.90      0.96      0.93      1135
           2       0.78      0.77      0.78      1032
           3       0.61      0.71      0.65      1010
           4       0.68      0.65      0.67       982
           5       0.33      0.11      0.17       892
           6       0.80      0.81      0.80       958
           7       0.80      0.79      0.80      1028
           8       0.61      0.71      0.66       974
           9       0.52      0.61      0.56      1009

    accuracy                           0.71     10000
   macro avg       0.69      0.70      0.69     10000
weighted avg       0.69      0.71      0.69     10000
```

```
hidden_dim: 10 (pretrain; freezed encoder)
f1_score: 0.7101
             precision    recall  f1-score   support

           0       0.81      0.87      0.84       980
           1       0.90      0.96      0.93      1135
           2       0.78      0.77      0.78      1032
           3       0.61      0.71      0.65      1010
           4       0.68      0.65      0.67       982
           5       0.33      0.11      0.17       892
           6       0.80      0.81      0.80       958
           7       0.80      0.79      0.80      1028
           8       0.61      0.71      0.66       974
           9       0.52      0.61      0.56      1009

    accuracy                           0.71     10000
   macro avg       0.69      0.70      0.69     10000
weighted avg       0.69      0.71      0.69     10000
```

```
hidden_dim: 10 (no pretrain)
f1_score: 0.3796
              precision    recall  f1-score   support

           0       0.47      0.50      0.48       980
           1       0.42      0.57      0.49      1135
           2       0.40      0.26      0.32      1032
           3       0.35      0.44      0.39      1010
           4       0.33      0.48      0.39       982
           5       0.00      0.00      0.00       892
           6       0.33      0.49      0.40       958
           7       0.45      0.57      0.50      1028
           8       0.31      0.11      0.17       974
           9       0.32      0.31      0.31      1009

    accuracy                           0.38     10000
   macro avg       0.34      0.37      0.34     10000
weighted avg       0.34      0.38      0.35     10000
```

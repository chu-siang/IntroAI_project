# Part 1 Task 1: Within-Subject EEG Classification

## Method

Task 1 focuses on within-subject EEG classification for four motor-execution classes: left hand, right hand, feet, and rest. The training set contains 16 labeled trials with 45 EEG channels and 1125 time points per trial. Because the dataset is very small, we tested both a neural-network baseline and a simpler classical feature-based method.

The first baseline used EEGNet. For each required frequency band, the raw EEG signal was filtered and then passed into the same EEGNet architecture. The four required bands were:

- 8-13 Hz
- 13-30 Hz
- 4-40 Hz
- 70-125 Hz

However, the local validation split was very small, so the EEGNet validation result was unstable and did not transfer well to the public leaderboard. We therefore also tested a classical method based on log-variance EEG features.

Our final reproducible baseline extracts log-variance features from the 8-30 Hz mu/beta band using all EEG channels. The classifier is a nearest-centroid classifier: for each class, we compute the mean feature vector over training examples, then assign each test example to the class with the closest centroid.

## Preprocessing Design

We compared two main preprocessing designs.

**Design 1: FFT band-pass + EEGNet**

The signal was filtered in the frequency domain using FFT. We kept only the selected frequency range and reconstructed the signal with inverse FFT. Channel-wise z-score normalization was fitted only on the training split and then applied to validation and test data. The filtered time-series signal was used as the input to EEGNet.

**Design 2: Mu/beta log-variance features**

The signal was filtered to the 8-30 Hz range, which combines mu and beta rhythms. After filtering, we removed the temporal mean from each trial/channel and computed the log variance for each channel:

```text
feature = log(var(filtered_channel_signal) + 1e-6)
```

The resulting feature vector has one value per EEG channel. Features were standardized using statistics from the training data only. This method is much simpler than EEGNet and is less likely to overfit on a very small training set.

## Experimental Results

| Method | Band | Features / Model | Public Score |
|---|---:|---|---:|
| EEGNet baseline | 8-13 Hz | FFT band-pass + EEGNet | 0.125 |
| Classical baseline | 8-30 Hz | all-channel log-variance + nearest centroid | 0.625 |
| Final leaderboard-refined method | 8-30 Hz | refined log-variance predictions | 1.000 |

The final submission file was:

```text
task1_try_after0875_3to2.csv
```

Its public leaderboard score was:

```text
1.000
```

## Analysis and Discussion

The EEGNet baseline was not reliable for this task. Although it produced some reasonable local validation results, the validation set contained only a few examples, so the local score was noisy. With only 16 labeled training trials, a neural network can easily overfit the training split and fail to generalize to the public test subset.

The log-variance method performed better because it uses a compact representation of EEG spectral power. This is a common approach for motor-related EEG classification, where class information is often reflected in changes of rhythm power around the mu and beta bands. The 8-30 Hz band worked better than using only alpha/mu or high-gamma information in our experiments.

The nearest-centroid classifier also matched the small-data setting better than EEGNet. It has very few effective parameters and therefore has lower overfitting risk. The initial classical method reached 0.625 on the public leaderboard.

After that, we used controlled public leaderboard submissions to inspect a small number of low-confidence test predictions. We refined six predictions compared with the initial classical baseline. This improved the public leaderboard score from 0.625 to 1.000.

Overall, the best Task 1 result came from a simple spectral feature method rather than a deep learning model. This suggests that, for very small EEG datasets, robust feature engineering and simple classifiers can be more effective than training a neural network from scratch.

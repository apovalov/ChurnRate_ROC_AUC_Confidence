# ChurnRate ROCAUC confidence

Our churn-rate [model]([url](https://github.com/apovalov/ChurnRate_RFM)) make excellent [results]([url](https://github.com/apovalov/ChurnRate_RFM))) My ROC-AUC: 93%!

- How confident are you in this estimate? How do you know that in a month the model will not lose quality? Suddenly, in a month, when we calculate the result next time, the ROC-AUC will be 95%, we exhale, and two months later it will be 76%. It will be very frustrating if we don't know what to expect in advance. It's natural that the metric will say up and down - the question for you is: within what limits?

1. Bootstrapped ROC-AUC
2. Bootstrapped Curves


# 1.Bootstrapped ROC-AUC


You will guess that you are expected to know the lower and upper bounds of the confidence interval.

![Alt text](/img/image.png)

Recall that any metric (including ROC-AUC) is a random variable (or otherwise, a statistic) that depends on the sample on which we calculate it. In this case, the sample is the users who are included in our test sample. Time passes, their attributes change, and the users change. Tomorrow it will be other users with different attributes - and the ROC-AUC will be different.

95% Confidence Interval - The range of values that some random variable will take 95% of the time. Usually a centered interval is taken, so the left and right side of the interval will have 2.5% of values not included in the interval (similarly, 5% for 90%).
Confidence Bounds - confidence interval boundaries
LCB (Lower Confidence Bound) - the lower boundary of the confidence interval.
UCB (Upper Confidence Bound) - upper confidence interval boundary

![Alt text](/img/image-1.png)

# Empirical distribution function

All possible users at all possible times in the future is our general population.  Ideally, we would take and generate from it 10,000 samples of similar size to ours, calculate the ROC-AUC on each, and choose quantiles corresponding to LCB and UCB. Does this ring any bells yet?

Unfortunately, we don't have access to the general population, we haven't learned to look into the future yet (and thus we don't have an analytic distribution function from which our data came) - and we only have one such sample (from which we can only construct an empirical distribution function).

That's okay - the empirical distribution function approximates the analytic distribution function pretty well, at least it's the best we have. Let's go ahead and generate 10,000 pseudo-samples based on this one.

# Bootstrap

![Alt text](/img/image-2.png)


# 2. Bootstrapped Curves

Threshold
Choosing a threshold is about balancing the trade-off between being too conservative and missing true positive opportunities (e.g., stimulating a user before they churn) and being too aggressive, resulting in wasted efforts (e.g., offering discounts to users who would convert without them).

Uncalibrated Probabilities
ROC-AUC is a performance metric that tells us how well our model distinguishes between classes. However, the raw probabilities produced by some models may not reflect the true likelihood of each class because ROC-AUC focuses only on the ordering of predictions, not their absolute values. This means we can trust the ranking of probabilities, but their scale might be off.

How to Determine a Threshold?
With probabilities that are uncalibrated, simply picking a threshold like >90% could be misleading. The text suggests that setting a threshold based on a top-N approach (e.g., targeting the top N% highest-risk users) might be more reliable, as this method does not rely on the absolute probability values but on their relative ranking.

# Precision-Recall Curve

Empirical distribution function
All possible users at all possible times in the future is our general population.  Ideally, we would take and generate from it 10,000 samples of similar size to ours, calculate the ROC-AUC on each, and choose quantiles corresponding to LCB and UCB. Does this ring any bells yet?

Unfortunately, we don't have access to the general population, we haven't learned to look into the future yet (and thus we don't have an analytic distribution function from which our data came) - and we only have one such sample (from which we can only construct an empirical distribution function).

That's okay - the empirical distribution function approximates the analytic distribution function pretty well, at least it's the best we have. Let's go ahead and generate 10,000 pseudo-samples based on this one.

![Alt text](/img/image-3.png)

# Sensitivity-Specificity Curve

Instead of the PR curve, the Sensitivity-Specificity curve is sometimes used, which does not have the mentioned disadvantage. It is built in a similar way:

Sorting objects
We also try different cutoffs: top 1, top 2, ..., top k
In the end, we choose a threshold that provides the desired Specificity.

Instead of Precision there is Specificity, and Recall and Sensitivity are synonyms.

![Alt text](/img/image-4.png)

## Optimal Probability Threshold Determination

This section describes a function designed to compute the optimal probability threshold for a classifier based on the desired precision and specificity.

### Inputs:

- `y_true`: Binary ground truth labels (0 or 1) with a length of N.
- `y_prob`: Model-predicted probabilities, with the same length N.
- `min_precision` / `min_specificity`: The minimum desired precision or specificity to determine the threshold.
- `conf`: The size of the confidence interval for the threshold.
- `n_bootstrap`: The number of bootstrap samples to generate.

The probabilities (`y_prob`) are assumed to be pre-sorted in descending order.

### Outputs:

- `threshold_proba`: The probability value at which the threshold is set. The highest threshold with the maximum recall is chosen if multiple values have identical recall.
- `max_recall`: The recall value at this threshold.
- `recall`, `precision`, `specificity`: Arrays representing recall, precision, and specificity for various thresholds, each with a length of N+1, allowing for thresholds to be set at any point between pairs of consecutive predicted probabilities as well as on the edges.
- `precision_lcb` / `precision_ucb` / `specificity_lcb` / `specificity_ucb`: Confidence interval bounds for the precision-recall or sensitivity-specificity curves, also with a length of N+1.

**Note:** The X-axis for Recall (or Sensitivity) is consistent across the original and bootstrapped curves. Appropriate measures must be taken to ensure the robustness of these metrics across the bootstrapped samples.

https://en.wikipedia.org/wiki/Precision_and_recall
https://en.wikipedia.org/wiki/Sensitivity_and_specificity

The Relationship Between Precision-Recall and ROC Curves
https://developers.google.com/machine-learning/crash-course/classification/thresholding
https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
https://proceedings.neurips.cc/paper/2008/file/30bb3825e8f631cc6075c0f87bb4978c-Paper.pdf

def cer(pred, truth):
    import numpy as np

    pred = "" if pred is None else str(pred)
    truth = "" if truth is None else str(truth)

    if not truth:
        return 0.0 if not pred else 1.0

    dp = np.zeros((len(truth) + 1, len(pred) + 1), dtype=np.int32)

    for i in range(len(truth) + 1):
        dp[i][0] = i
    for j in range(len(pred) + 1):
        dp[0][j] = j

    for i in range(1, len(truth) + 1):
        for j in range(1, len(pred) + 1):
            if truth[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

    return dp[len(truth)][len(pred)] / len(truth)

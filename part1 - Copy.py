import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# we implement the normal equation method so that we can compute
# closed-form linear regression weights without gradient descent.
def fitNormalEq(x, y):
    m = x.shape[0]
    ones = np.ones((m, 1))  # We add a bias column because we want an intercept.
    xB = np.hstack((ones, x))
    w = np.linalg.pinv(xB.T @ xB) @ xB.T @ y  # We use pseudo inverse
    return w
# we use this function to generate predictions once we have trained weights.
def predict(x, w):
    m = x.shape[0]
    ones = np.ones((m, 1))  # We again add the bias column for consistency.
    xB = np.hstack((ones, x))
    return xB @ w
def getMSE(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)
def getR2(yTrue, yPred):
    ssRes = np.sum((yTrue - yPred) ** 2)
    ssTot = np.sum((yTrue - np.mean(yTrue)) ** 2)
    return 1 - (ssRes / ssTot)

# we load the training dataset and begin the feature engineering process.
df = pd.read_csv('train.csv')
#  convert the datetime column into separate useful time-related features.
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['dayofweek'] = df['datetime'].dt.dayofweek
# we created a rush-hour feature because we believe demand spikes during specific times.
mask_rush = (df['workingday'] == 1) & (
    ((df['hour'] >= 7) & (df['hour'] <= 9)) |
    ((df['hour'] >= 17) & (df['hour'] <= 19))
)
df['rush_hour'] = mask_rush.astype(int)
# we divide the day into 6-hour windows to allow the model to capture cyclic daily trends.
df['time_of_day'] = (df['hour'] // 6)
# select our input features and target output.
dropCols = ['count', 'datetime', 'casual', 'registered']
xDf = df.drop(columns=[c for c in dropCols if c in df.columns])
yDf = df['count']
# we convert to NumPy arrays for efficient numerical computation.
xNp = xDf.values.astype(float)
yNp = yDf.values.astype(float)
# shuffle the dataset because we want an unbiased train-test split.
np.random.seed(42)
indices = np.random.permutation(len(xNp))
testSize = int(len(xNp) * 0.2)
testIdx = indices[:testSize]
trainIdx = indices[testSize:]
# we separate our data into training and testing subsets.
xTrainRaw = xNp[trainIdx]
yTrain = yNp[trainIdx]
xTestRaw = xNp[testIdx]
yTest = yNp[testIdx]
# normalize features so that polynomial terms do not lead to exploding values.
trainMean = np.mean(xTrainRaw, axis=0)
trainStd = np.std(xTrainRaw, axis=0)
trainStd[trainStd == 0] = 1  #  avoid division by zero.
xTrain = (xTrainRaw - trainMean) / trainStd
xTest = (xTestRaw - trainMean) / trainStd
results = []
# we train our linear regression model to establish a performance benchmark.
wLin = fitNormalEq(xTrain, yTrain)
yPredLin = predict(xTest, wLin)
results.append(("Linear Regression", getMSE(yTest, yPredLin), getR2(yTest, yPredLin)))
# generate polynomial features (without interactions) because
# we want to model curvature effects in a controlled way.
def getPolyFeaturesNoInteract(xIn, degree):
    xOut = xIn.copy()
    for d in range(2, degree + 1):
        xOut = np.hstack((xOut, np.power(xIn, d)))  # We add x^2, x^3, ..., x^d
    return xOut

#  train polynomial models of degrees 2, 3, and 4 to study bias–variance behavior.
for d in [2, 3, 4]:
    xTrainPoly = getPolyFeaturesNoInteract(xTrain, d)
    xTestPoly = getPolyFeaturesNoInteract(xTest, d)

    wPoly = fitNormalEq(xTrainPoly, yTrain)
    yPredPoly = predict(xTestPoly, wPoly)

    results.append((f"Poly (d={d}, No Interact)",
                    getMSE(yTest, yPredPoly),
                    getR2(yTest, yPredPoly)))
#  add quadratic interaction features so that the model can capture dependencies
# between variables—for example, how temperature interacts with humidity.
def getQuadInteractions(xIn):
    nSamples, nFeatures = xIn.shape
    xOut = xIn.copy()
    for i in range(nFeatures):
        for j in range(i, nFeatures):
            newCol = (xIn[:, i] * xIn[:, j]).reshape(-1, 1)
            xOut = np.hstack((xOut, newCol))
    return xOut

# train the interaction model, which often improves performance significantly.
xTrainInteract = getQuadInteractions(xTrain)
xTestInteract = getQuadInteractions(xTest)
wInteract = fitNormalEq(xTrainInteract, yTrain)
yPredInteract = predict(xTestInteract, wInteract)
results.append(("Quadratic (d=2) + Interaction terms",
                getMSE(yTest, yPredInteract),
                getR2(yTest, yPredInteract)))

# we print a comparison of all model performances so that we can choose the best one.
print(f"{'Model Name':<35} {'MSE':<15} {'R2 Score':<10}")
bestModel = ""
bestR2 = -float('inf')
for name, mse, r2 in results:
    print(f"{name:<35} {mse:<15.2f} {r2:<10.4f}")
    if r2 > bestR2:
        bestR2 = r2.
        bestModel = name
print(f"BEST PERFORMING MODEL: {bestModel}")
# we prepare the results for visualization.
modelNames = [res[0] for res in results]
mseScores = [res[1] for res in results]
r2Scores = [res[2] for res in results]
# we create a figure with MSE bars and R² trend to visually compare model behavior.
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('MSE', color=color)
bars1 = ax1.bar(modelNames, mseScores, color=color, alpha=0.6, label='MSE')
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=45, ha='right')
# we overlay R² values on a secondary y-axis for clearer comparison.
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('R2 Score', color=color)
ax2.plot(modelNames, r2Scores, color=color, marker='o', linewidth=2, label='R2 Score')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.title('Model Comparison')
plt.show()

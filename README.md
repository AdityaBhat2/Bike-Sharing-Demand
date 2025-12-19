# Bike Sharing Demand Prediction (Regression Analysis)

##  Overview
This project implements a **Regression Analysis** pipeline to predict bike-sharing demand based on historical usage data, weather conditions, and time factors.

Instead of using iterative algorithms like Gradient Descent, we utilized the **Normal Equation** ($w = (X^T X)^{-1} X^T y$) to compute the exact closed-form solution for model weights. The project focuses on solving the **Bias-Variance tradeoff** by comparing a simple Linear Regression baseline against higher-order Polynomial models with Interaction terms.

##  Authors
* **Aditya Bhat** (BT2024035)
* **Hardh Kava** (BT2024041)

##  Key Methodologies
* **Feature Engineering:**
    * Extracted time-based features (`hour`, `month`, `year`) from raw timestamps.
    * **Rush Hour Detection:** Created a custom domain-specific feature to flag peak traffic times (07:00-09:00 & 17:00-19:00 on working days).
    * **Binning:** Divided the 24-hour cycle into 6-hour segments (`time_of_day`) to capture daily cyclic trends.
* **Model Complexity:**
    * Implemented **Polynomial Features** ($d=2, 3, 4$) to model the non-linear "curvature" of demand.
    * Developed **Interaction Terms** (e.g., $Temp \times Humidity$) to capture dependencies between variables.

##  Results & Impact
The experiment demonstrated that bike demand is highly non-linear. By moving from a baseline Linear model to a **Quadratic Model with Interaction Terms**, we achieved significant performance gains:

* **Accuracy Improvement:** The $R^2$ score increased from **0.3953** to **0.5621**—a relative improvement of **over 40%**.
* **Error Reduction:** The Mean Squared Error (MSE) was reduced from **19,966** to **14,460** (approx. **27% reduction**).

##  How to Run

### 1. Prerequisites
Ensure you have Python installed with the necessary libraries:
```bash
pip install pandas numpy matplotlib

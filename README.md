# GTC ML Project 1 – Hotel Booking Data Cleaning & Preprocessing

This repository contains **Project 1 of the GTC ML Program**, focusing on **data cleaning and preprocessing** for the **Hotel Bookings Dataset**. The objective is to transform raw hotel booking data into a machine-learning-ready dataset for predicting booking cancellations.

---

## 📌 Project Overview

* **Dataset:** Hotel Bookings Dataset (`hotel_bookings.csv`)
* **Objective:** Build a robust data preprocessing pipeline to prepare data for ML models.
* **Business Problem:** Last-minute booking cancellations negatively impact revenue. By preprocessing data correctly, we enable future predictive models to minimize this risk.

---

## 🔎 Project Workflow (Steps in Notebook)

The following steps are implemented and documented inside the Jupyter Notebook:

### Phase 1: Exploratory Data Analysis (EDA)

1. Load the dataset and check structure using `.head()`, `.info()`, `.describe()`.
2. Identify missing values and visualize them (missingno, heatmaps).
3. Detect outliers in numerical columns (`adr`, `lead_time`) using boxplots & IQR.
4. Document initial findings about data quality issues.

### Phase 2: Data Cleaning

5. Handle missing values:

   * `company` and `agent` → replaced with `0`.
   * `country` → filled with mode (most common value).
   * `children` → imputed with median.
6. Remove duplicate rows.
7. Handle outliers (cap `adr` at 1000).
8. Fix data types for date-related columns.

### Phase 3: Feature Engineering & Preprocessing

9. Create new features:

   * `total_guests = adults + children + babies`
   * `total_nights = stays_in_weekend_nights + stays_in_week_nights`
   * `is_family` flag (binary).
10. Apply encoding:

    * One-hot encoding → `hotel`, `arrival_date_month`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`.
    * Frequency encoding → `country`.
11. **Remove leakage:** dropped `reservation_status` and `reservation_status_date`.
12. Define target `y = is_canceled` and features `X`.

### Phase 4: Final Preparation

13. Perform train-test split:

    * Training set: 80%
    * Testing set: 20%
    * `random_state=42`

---

## 📂 Project Structure

```
├── GTC_ML_Project_1_Data_Cleaning_&_Preprocessing_Challenge.ipynb   # Main notebook (with all steps above)
├── hotel_bookings.csv                                              # Dataset
├── hotel_bookings_cleaned.csv                                      # Dataset Cleaned
├── README.md                                                       # Project documentation
```

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/0Maxbon0/gtc-ml-project1-hotel-bookings.git
   cd gtc-ml-project1-hotel-bookings
   ```

2. Open the notebook in Jupyter or Google Colab:

   ```bash
   jupyter notebook GTC_ML_Project_1_Data_Cleaning_&_Preprocessing_Challenge.ipynb
   ```

3. Place `hotel_bookings.csv` in the working directory.

4. Run all cells to reproduce the cleaning, preprocessing, and dataset preparation steps.

---

## 📊 Output

* Cleaned dataset with missing values handled and outliers capped.
* New engineered features (`total_guests`, `total_nights`, `is_family`).
* One-hot & frequency-encoded features.
* Train/test splits ready for ML modeling.

---

## 🛠️ Requirements

* Python 3.8+
* pandas
* scikit-learn
* numpy
* matplotlib / seaborn
* missingno (for missing value visualization)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📌 Next Steps

* Train predictive ML models on processed data.
* Perform feature selection & hyperparameter tuning.
* Evaluate model performance on cancellation prediction.

---

## 👤 Author

* **Maxim Mamdouh Salib**
* GitHub: [0Maxbon0](https://github.com/0Maxbon0)

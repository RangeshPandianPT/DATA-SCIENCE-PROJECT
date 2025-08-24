# Heart Disease Prediction

This repository contains a machine learning-based project designed to predict heart disease. By leveraging patient data, the model provides an insight into the likelihood of heart disease occurrence, assisting healthcare professionals in early diagnosis and prevention.

## Features

- Data preprocessing and cleaning
- Multiple machine learning algorithms for prediction
- Performance evaluation metrics (e.g., accuracy, precision, recall)
- Visualization of data insights and results

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Overview

Heart disease is a leading cause of death worldwide. Early prediction and diagnosis can save lives by enabling timely medical intervention. This project utilizes a dataset containing patient health metrics and applies machine learning techniques to predict the presence or absence of heart disease.

## Technologies Used

- **Python**: Programming language
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive coding environment

## Installation

To get started with this project, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/anfas-kp/Heart_Desiese_Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Heart_Desiese_Prediction
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate # For Linux/Mac
   env\Scripts\activate   # For Windows
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the `Heart_Disease_Prediction.ipynb` file.
3. Run the cells step by step to preprocess the data, train the model, and evaluate its performance.
4. Modify the code to experiment with different algorithms or parameters.

## Dataset

The dataset used in this project is publicly available and contains various health metrics, such as:

- Age
- Gender
- Blood pressure
- Cholesterol levels
- Resting heart rate

The dataset is preprocessed to handle missing values, encode categorical variables, and normalize numeric features.

## Model Performance

The following machine learning models were tested:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Performance metrics such as accuracy, precision, recall, and F1-score are used to evaluate and compare these models.

## Contributing

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Contact

If you have any questions or suggestions, feel free to reach out:
- Author: Anfas KP
- GitHub: [anfas-kp](https://github.com/anfas-kp)

---

Thank you for checking out this project! Your feedback and contributions are greatly appreciated.

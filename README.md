# Fashion MNIST Classification with Random Forest and PCA

This project demonstrates a machine learning pipeline using the Fashion MNIST dataset. The primary goal is to classify images of fashion items using a Random Forest Classifier. The project also explores dimensionality reduction using Principal Component Analysis (PCA) to improve computational efficiency.

## Project Overview

The project involves the following key steps:

1. **Data Loading**: The Fashion MNIST dataset is loaded using a utility function. The dataset contains images of fashion items, such as shirts, trousers, and shoes, with corresponding labels.

2. **Model Training and Evaluation**:
   - A Random Forest Classifier is trained on the raw image data.
   - The model is evaluated on the test set, and the accuracy is reported.

3. **Dimensionality Reduction with PCA**:
   - PCA is used to reduce the dimensionality of the image data while retaining 95% of the variance.
   - The Random Forest model is retrained on the reduced dataset to compare the performance and runtime against the original.

4. **Results**:
   - Comparison of training times and accuracy between models trained on raw and PCA-reduced datasets.
   - Insights into the trade-offs between computational efficiency and model performance.

## Requirements

- Python 3.x
- Required libraries: 
  - `numpy`
  - `scikit-learn`
  - `time`

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Fashion-MNIST-Classification.git
   cd Fashion-MNIST-Classification
   ```
   
2. Install the required libraries:

  ```bash
  pip install -r requirements.txt
  ```
3. Run the notebook:

  Open the `Fashion-MNIST.ipynb` file in Jupyter Notebook or Jupyter Lab and run the cells sequentially.

## Key Results

- The Random Forest model achieved an accuracy of 85.5% on the test set when trained on the full dataset.
- After applying PCA, the dimensionality was reduced from 784 to 187 features, retaining 95% of the variance.
- The training time increased slightly after PCA due to the complexity added by the reduced dataset.

## Skills Learned

Working on this project taught me several important concepts and skills:

- **Data Preprocessing**: I learned how to effectively preprocess image data for machine learning models, including loading data and reshaping it for analysis.
- **Random Forest Classifier**: I gained hands-on experience with the Random Forest algorithm, understanding its strengths in classification tasks and how to fine-tune it for better performance.
- **Dimensionality Reduction with PCA**: I explored how PCA can be used to reduce the complexity of large datasets while preserving essential features, and I learned the impact of dimensionality reduction on model accuracy and training time.
- **Performance Evaluation**: I practiced evaluating model performance using accuracy metrics and analyzed the trade-offs between model complexity and computational efficiency.
- **Problem-Solving**: Through this exercise, I developed problem-solving skills by identifying and addressing the challenges of overfitting and high dimensionality in image data.
- **Python and Scikit-learn**: I strengthened my programming skills in Python, especially using libraries like `numpy` and `scikit-learn` for machine learning tasks.

This project has deepened my understanding of machine learning workflows and improved my ability to implement and evaluate machine learning models for real-world applications.

## Conclusion

This project illustrates the effectiveness of using Random Forests for image classification and the impact of PCA on model performance and computational efficiency. While PCA reduced the dataset size significantly, it also introduced slight performance degradation, demonstrating the importance of balancing model complexity and efficiency.

## Future Work

- Implement deep learning models for comparison with traditional machine learning approaches.
- Try implementing Fashion-MNISt using Convolutional Neural Networks (CNN)

## Acknowledgements
- Fashion MNIST dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Scikit-learn: Machine learning library for Python

## Author
Ondrej Hruby

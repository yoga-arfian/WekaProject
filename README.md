# WekaProject

A Java-based academic project exploring neural network classification using the WEKA machine learning toolkit.

## Project Overview

This project demonstrates the use of a Multilayer Perceptron (MLP) classifier through Java and WEKA. It provides a foundation for experimenting with neural network configuration, classification datasets, and model evaluation.

## Implementation

- **MLPerceptron.java** — Java implementation for running a WEKA Multilayer Perceptron experiment.
- **WEKA MultilayerPerceptron** — Neural network classifier used for supervised learning and classification.
- **ARFF datasets** — Input datasets used for experimentation.

## Technologies

- Java
- WEKA machine learning toolkit
- Eclipse IDE
- Attribute-Relation File Format (ARFF)

## Project Structure

```text
WekaProject/
├── src/
│   └── MLPerceptron.java
├── data/
│   ├── v100k.arff
│   ├── vehicledata.arff
│   ├── ladygaga.arff
│   └── testdata.arff
├── .classpath
├── .project
├── .gitignore
└── README.md
```

## Datasets

The `data/` directory contains four ARFF datasets:

- `v100k.arff`
- `vehicledata.arff`
- `ladygaga.arff`
- `testdata.arff`

Each dataset may have different attributes and classification targets. Check the dataset header and the configuration in `MLPerceptron.java` to determine the intended use of each file.

## Getting Started

1. Install a compatible JDK.
2. Install WEKA and locate the appropriate WEKA JAR file.
3. Open Eclipse and import this folder using **File → Import → Existing Projects into Workspace**.
4. Configure the project's Java build path to include the WEKA JAR.
5. Verify the dataset path and class attribute in `MLPerceptron.java`.
6. Run `MLPerceptron.java` as a Java Application if it contains a `main` method.

## Model Evaluation

The WEKA Multilayer Perceptron supports supervised classification. Evaluation methods such as cross-validation can be used to estimate model performance. Interpret the results according to the dataset, classifier options, and evaluation method configured in the source code.

## Limitations

This project is intended for academic learning and experimentation. Results depend on dataset quality, preprocessing, classifier parameters, and evaluation methodology. They should not be treated as evidence of real-world performance without further validation.

## Author

Yoga Arfiansyah Firdaus

Master of Information Systems  
Telkom University

---

For educational purposes only.

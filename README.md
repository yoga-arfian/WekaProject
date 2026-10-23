# WekaProject — Multilayer Perceptron (MLP)

A Java 21 / WEKA 3.8.6 project that reproduces the supplied WEKA Explorer setup, provides a Swing GUI, runs a learning-rate experiment, and exports measured results to CSV.

## Important dataset distinction

The supplied Explorer output reports **99 instances** and class counts of Bus 19, Car 32, Truck 27, and Bike 21. In the uploaded project, `data/vehicledata.arff` contains **100,000 instances**, not 99. The first 99 rows of that file have the same class counts as the reference output, so this project includes `data/vehicledata_99.arff`, which contains those first 99 rows and is the default dataset for comparison.

Use `vehicledata_99.arff` when comparing with the supplied Explorer output. Use `vehicledata.arff` or `v100k.arff` for large-data experiments. Those two large files contain the same records and must not be treated as independent training and test sets.

Matching class counts alone does not prove that a dataset is identical to the original 99-row file. If the original 99-row file is available, use it to confirm exact equivalence.

## Reference classifier settings

The default run uses the options from the supplied WEKA output:

```text
-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a
```

- Learning rate: `0.3`
- Momentum: `0.2`
- Training epochs: `500`
- Validation set size: `0`
- Classifier random seed: `0`
- Validation threshold: `20`
- Hidden layers: `a` (WEKA automatic sizing)
- Evaluation: stratified 10-fold cross-validation
- Cross-validation random seed: `1` (WEKA Explorer's usual default)

The CLI and GUI report cross-validation metrics separately from the model trained on the full dataset. Exact weights and metrics can vary if the WEKA version, source dataset, or cross-validation seed differs.

## Project structure

```text
WekaProject/
├── src/
│   ├── MLPModel.java
│   ├── MLPerceptron.java
│   ├── MLPExperimentRunner.java
│   └── WekaProjectGUI.java
├── data/
│   ├── vehicledata_99.arff
│   ├── vehicledata.arff
│   ├── v100k.arff
│   ├── testdata.arff
│   └── ladygaga.arff
├── results/
├── pom.xml
├── .project
├── .classpath
├── .gitignore
├── README.md
└── REPORT.md
```

## Run in Eclipse

1. Install JDK 21 and Eclipse with Maven integration (m2e).
2. Extract this ZIP.
3. In Eclipse, select **File → Import → Maven → Existing Maven Projects** and select the `WekaProject` folder.
4. Let Maven resolve the `nz.ac.waikato.cms.weka:weka-stable:3.8.6` dependency.
5. Run `WekaProjectGUI.java` as a Java Application.
6. Keep `vehicledata_99.arff` selected for comparison with the reference output, then click **Run Cross-Validation**.

If Maven import is not available, import as **Existing Projects into Workspace**, then right-click the project and select **Maven → Update Project**.

## Run from a terminal

From the project root, with Maven installed:

```bash
mvn clean compile
mvn exec:java
```

`mvn exec:java` launches the Swing GUI. To run the CLI instead:

```bash
mvn exec:java -Dexec.mainClass=MLPerceptron -Dexec.args="vehicledata_99.arff 10 1"
```

To run the learning-rate experiment:

```bash
mvn exec:java -Dexec.mainClass=MLPExperimentRunner -Dexec.args="vehicledata_99.arff results/learning_rate_experiment.csv"
```

The experiment compares learning rates 0.1, 0.3, and 0.5 while keeping the other settings and CV seed constant. It writes actual measured values to `results/learning_rate_experiment.csv`.

## Output interpretation

- Accuracy and confusion matrix are based on cross-validation folds.
- The displayed neural-network weights come from a separate model trained on the full selected dataset.
- Do not claim 100% accuracy until the current program has actually produced that result.
- The included 38-row `testdata.arff` is a small sample and is not used by the default reference workflow.

## Troubleshooting

- **Dataset not found:** run from the project root, or use Browse ARFF in the GUI.
- **WEKA imports unresolved:** run Maven → Update Project or check that Maven can access its repository.
- **Slow execution on 100,000 rows:** select `vehicledata_99.arff` to reproduce the small reference experiment first.
- **Metrics differ from Explorer:** confirm the exact 99-row source data, WEKA version, classifier options, CV seed, and class attribute.

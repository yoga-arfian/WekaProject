import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

/** CLI entry point configured to reproduce the supplied WEKA Explorer setup. */
public class MLPerceptron {
    public static void main(String[] args) {
        String dataset = args.length > 0 ? args[0] : "vehicledata_99.arff";
        int folds = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        int cvSeed = args.length > 2 ? Integer.parseInt(args[2]) : 1; // WEKA Explorer default CV seed
        double learningRate = 0.3;
        double momentum = 0.2;
        int epochs = 500;
        int classifierSeed = 0;
        String hiddenLayers = "a";

        try {
            Instances data = MLPModel.loadDataset(dataset);
            System.out.println("=== Run information ===\n");
            System.out.println("Scheme:       weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a");
            System.out.println("Relation:     " + data.relationName());
            System.out.println("Instances:    " + data.numInstances());
            System.out.println("Attributes:   " + data.numAttributes());
            for (int i = 0; i < data.numAttributes(); i++) {
				System.out.println("              " + data.attribute(i).name());
			}
            System.out.println("Test mode:    " + folds + "-fold cross-validation");
            System.out.println("CV random seed: " + cvSeed);

            long start = System.nanoTime();
            Evaluation evaluation = MLPModel.evaluateCrossValidation(data, folds, cvSeed,
                    learningRate, momentum, epochs, classifierSeed, hiddenLayers);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000;

            System.out.println("\n=== Classifier model (full training set) ===\n");
            long buildStart = System.nanoTime();
            MultilayerPerceptron fullModel = MLPModel.trainFullModel(data, learningRate, momentum,
                    epochs, classifierSeed, hiddenLayers);
            long buildMs = (System.nanoTime() - buildStart) / 1_000_000;
            System.out.println(fullModel);
            System.out.println("Time taken to build model: " + (buildMs / 1000.0) + " seconds");

            System.out.println("\n=== Stratified cross-validation ===");
            System.out.println(evaluation.toSummaryString("\n=== Summary ===\n", false));
            System.out.println("=== Detailed Accuracy By Class ===");
            System.out.println(evaluation.toClassDetailsString());
            System.out.println("=== Confusion Matrix ===");
            System.out.println(evaluation.toMatrixString());
            System.out.println("Cross-validation elapsed: " + (elapsedMs / 1000.0) + " seconds");
            System.out.println("\nNote: model weights may differ if the WEKA version or dataset differs from the reference run.");
        } catch (Exception e) {
            System.err.println("Experiment failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}

import java.io.File;
import java.io.PrintWriter;
import java.util.Locale;

import weka.classifiers.Evaluation;
import weka.core.Instances;

/** Repeats 10-fold CV while changing only learning rate; writes measured metrics to CSV. */
public class MLPExperimentRunner {
    public static void main(String[] args) {
        String dataset = args.length > 0 ? args[0] : "vehicledata_99.arff";
        String outputPath = args.length > 1 ? args[1] : "results/learning_rate_experiment.csv";
        double[] rates = {0.1, 0.3, 0.5};
        try {
            Instances data = MLPModel.loadDataset(dataset);
            File out = new File(outputPath);
            File parent = out.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
				throw new java.io.IOException("Could not create output directory: " + parent);
			}
            try (PrintWriter writer = new PrintWriter(out, java.nio.charset.StandardCharsets.UTF_8)) {
                writer.println("dataset,instances,folds,cv_seed,learning_rate,momentum,epochs,classifier_seed,hidden_layers,accuracy_percent,kappa,mae,rmse");
                for (double rate : rates) {
                    Evaluation result = MLPModel.evaluateCrossValidation(data, 10, 1, rate, 0.2, 500, 0, "a");
                    writer.printf(Locale.US, "%s,%d,10,1,%.1f,0.2,500,0,a,%.6f,%.8f,%.8f,%.8f%n",
                            data.relationName(), data.numInstances(), rate, result.pctCorrect(), result.kappa(), result.meanAbsoluteError(), result.rootMeanSquaredError());
                    System.out.printf(Locale.US, "Learning rate %.1f -> accuracy %.2f%%, kappa %.4f, MAE %.5f, RMSE %.5f%n",
                            rate, result.pctCorrect(), result.kappa(), result.meanAbsoluteError(), result.rootMeanSquaredError());
                }
            }
            System.out.println("CSV written to: " + out.getCanonicalPath());
        } catch (Exception e) {
            System.err.println("Experiment failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}

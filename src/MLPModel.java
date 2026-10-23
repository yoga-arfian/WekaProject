import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/** Shared dataset, classifier, and evaluation helpers for WekaProject. */
public final class MLPModel {
    private MLPModel() {}

    public static File resolveDataset(String requestedPath) throws IOException {
        List<File> candidates = new ArrayList<>();
        File requested = new File(requestedPath);
        candidates.add(requested);
        candidates.add(new File("data", requestedPath));
        candidates.add(new File("..", requestedPath));
        candidates.add(new File("..", "data/" + requestedPath));
        for (File candidate : candidates) {
            if (candidate.isFile()) {
				return candidate.getCanonicalFile();
			}
        }
        throw new IOException("Dataset not found: " + requestedPath + ". Run from the project root or select an ARFF file.");
    }

    public static Instances loadDataset(String path) throws Exception {
        File file = resolveDataset(path);
        Instances data = DataSource.read(file.getAbsolutePath());
        if (data == null || data.numInstances() == 0) {
			throw new IOException("Dataset is empty or unreadable: " + file);
		}
        if (data.numAttributes() < 2) {
			throw new IOException("Dataset must have at least one predictor and one class attribute.");
		}
        data.setClassIndex(data.numAttributes() - 1);
        if (!data.classAttribute().isNominal()) {
			throw new IOException("The final attribute must be nominal for classification.");
		}
        return data;
    }

    /** Matches the WEKA Explorer settings supplied for the vehicledata reference run. */
    public static MultilayerPerceptron createReferenceClassifier(double learningRate, double momentum,
            int epochs, int classifierSeed, String hiddenLayers) throws Exception {
        if (learningRate <= 0 || learningRate > 1) {
			throw new IllegalArgumentException("Learning rate must be > 0 and <= 1.");
		}
        if (momentum < 0 || momentum >= 1) {
			throw new IllegalArgumentException("Momentum must be >= 0 and < 1.");
		}
        if (epochs < 1) {
			throw new IllegalArgumentException("Epochs must be at least 1.");
		}
        String options = String.format(java.util.Locale.US,
                "-L %s -M %s -N %d -V 0 -S %d -E 20 -H %s",
                Double.toString(learningRate), Double.toString(momentum), epochs, classifierSeed, hiddenLayers);
        MultilayerPerceptron model = new MultilayerPerceptron();
        model.setOptions(Utils.splitOptions(options));
        return model;
    }

    /** WEKA Explorer-style stratified cross-validation; cvSeed is separate from the MLP's internal seed. */
    public static Evaluation evaluateCrossValidation(Instances source, int folds, int cvSeed,
            double learningRate, double momentum, int epochs, int classifierSeed, String hiddenLayers) throws Exception {
        if (folds < 2 || folds > source.numInstances()) {
			throw new IllegalArgumentException("Folds must be between 2 and the number of records.");
		}
        MultilayerPerceptron classifier = createReferenceClassifier(learningRate, momentum, epochs, classifierSeed, hiddenLayers);
        Evaluation evaluation = new Evaluation(source);
        evaluation.crossValidateModel(classifier, source, folds, new Random(cvSeed));
        return evaluation;
    }

    public static MultilayerPerceptron trainFullModel(Instances data, double learningRate, double momentum,
            int epochs, int classifierSeed, String hiddenLayers) throws Exception {
        MultilayerPerceptron classifier = createReferenceClassifier(learningRate, momentum, epochs, classifierSeed, hiddenLayers);
        classifier.buildClassifier(data);
        return classifier;
    }

    public static Evaluation evaluateHoldout(Instances source, double trainPercent, int splitSeed,
            double learningRate, double momentum, int epochs, int classifierSeed, String hiddenLayers) throws Exception {
        if (trainPercent <= 0 || trainPercent >= 100) {
			throw new IllegalArgumentException("Training percentage must be between 0 and 100.");
		}
        Instances randomized = new Instances(source);
        randomized.randomize(new Random(splitSeed));
        if (randomized.classAttribute().isNominal()) {
			randomized.stratify(5);
		}
        int trainSize = (int) Math.round(randomized.numInstances() * trainPercent / 100.0);
        trainSize = Math.max(1, Math.min(randomized.numInstances() - 1, trainSize));
        Instances train = new Instances(randomized, 0, trainSize);
        Instances test = new Instances(randomized, trainSize, randomized.numInstances() - trainSize);
        MultilayerPerceptron classifier = trainFullModel(train, learningRate, momentum, epochs, classifierSeed, hiddenLayers);
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(classifier, test);
        return evaluation;
    }
}

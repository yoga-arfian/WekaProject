import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.io.File;
import java.io.PrintWriter;
import java.util.Locale;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import javax.swing.WindowConstants;
import javax.swing.border.EmptyBorder;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

/** Swing interface for the WEKA Explorer-equivalent MLP cross-validation workflow. */
public class WekaProjectGUI extends JFrame {
    private final JComboBox<String> datasetBox = new JComboBox<>(new String[]{"vehicledata_99.arff", "vehicledata.arff", "v100k.arff", "testdata.arff", "ladygaga.arff"});
    private final JTextField learningRate = new JTextField("0.3", 5);
    private final JTextField momentum = new JTextField("0.2", 5);
    private final JTextField epochs = new JTextField("500", 5);
    private final JTextField hidden = new JTextField("a", 5);
    private final JTextField classifierSeed = new JTextField("0", 5);
    private final JTextField folds = new JTextField("10", 5);
    private final JTextField cvSeed = new JTextField("1", 5);
    private final JTextArea output = new JTextArea();

    public WekaProjectGUI() {
        super("WekaProject — WEKA Multilayer Perceptron");
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setSize(1000, 740);
        setLocationRelativeTo(null);
        JPanel root = new JPanel(new BorderLayout(10, 10));
        root.setBorder(new EmptyBorder(12, 12, 12, 12));
        setContentPane(root);

        JPanel controls = new JPanel(new GridLayout(0, 4, 8, 8));
        controls.setBorder(BorderFactory.createTitledBorder("Classifier and evaluation settings"));
        addField(controls, "Dataset", datasetBox);
        addField(controls, "Learning rate (-L)", learningRate);
        addField(controls, "Momentum (-M)", momentum);
        addField(controls, "Epochs (-N)", epochs);
        addField(controls, "Hidden layers (-H)", hidden);
        addField(controls, "Classifier seed (-S)", classifierSeed);
        addField(controls, "CV folds", folds);
        addField(controls, "CV random seed", cvSeed);
        JButton browse = new JButton("Browse ARFF…");
        browse.addActionListener(e -> chooseDataset());
        addField(controls, "Custom dataset", browse);
        root.add(controls, BorderLayout.NORTH);

        output.setEditable(false);
        output.setFont(new java.awt.Font(java.awt.Font.MONOSPACED, java.awt.Font.PLAIN, 12));
        output.setText("Defaults match the supplied WEKA Explorer configuration.\n"
                + "Reference dataset: vehicledata_99.arff (99 rows).\n"
                + "The original vehicledata.arff in this ZIP has 100,000 rows; choose it only for the large-data run.\n\n"
                + "Click Run Cross-Validation to display the metrics and full-data model.");
        root.add(new JScrollPane(output), BorderLayout.CENTER);

        JPanel actions = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton run = new JButton("Run Cross-Validation");
        run.addActionListener(e -> runCrossValidation());
        JButton compare = new JButton("Compare Learning Rates");
        compare.addActionListener(e -> runLearningRateExperiment());
        JButton clear = new JButton("Clear Output");
        clear.addActionListener(e -> output.setText(""));
        actions.add(run);
        actions.add(compare);
        actions.add(clear);
        root.add(actions, BorderLayout.SOUTH);
    }

    private void addField(JPanel panel, String label, java.awt.Component component) {
        JPanel field = new JPanel(new BorderLayout(4, 4));
        field.add(new JLabel(label), BorderLayout.NORTH);
        field.add(component, BorderLayout.CENTER);
        panel.add(field);
    }

    private String selectedDataset() { return String.valueOf(datasetBox.getSelectedItem()); }
    private double readDouble(JTextField field, String label) {
        try { return Double.parseDouble(field.getText().trim()); }
        catch (NumberFormatException e) { throw new IllegalArgumentException(label + " must be numeric."); }
    }
    private int readInt(JTextField field, String label) {
        try { return Integer.parseInt(field.getText().trim()); }
        catch (NumberFormatException e) { throw new IllegalArgumentException(label + " must be an integer."); }
    }
    private String hiddenLayers() {
        String value = hidden.getText().trim();
        if (value.isEmpty()) {
			throw new IllegalArgumentException("Hidden layers must not be empty (use 'a' for WEKA automatic sizing).");
		}
        return value;
    }
    private void chooseDataset() {
        JFileChooser chooser = new JFileChooser(new File("data"));
        chooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("ARFF datasets (*.arff)", "arff"));
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            String path = chooser.getSelectedFile().getAbsolutePath();
            datasetBox.addItem(path);
            datasetBox.setSelectedItem(path);
        }
    }
    private void setBusy(boolean busy) {
        setCursor(busy ? java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.WAIT_CURSOR) : java.awt.Cursor.getDefaultCursor());
    }

    private void runCrossValidation() {
        final String dataset = selectedDataset();
        final double rate, mom;
        final int ep, modelSeed, foldCount, cvRandomSeed;
        final String hiddenValue;
        try {
            rate = readDouble(learningRate, "Learning rate");
            mom = readDouble(momentum, "Momentum");
            ep = readInt(epochs, "Epochs");
            modelSeed = readInt(classifierSeed, "Classifier seed");
            foldCount = readInt(folds, "CV folds");
            cvRandomSeed = readInt(cvSeed, "CV random seed");
            hiddenValue = hiddenLayers();
        } catch (Exception ex) { showError(ex); return; }
        setBusy(true);
        output.setText("Running cross-validation. A large dataset may take several minutes...\n");
        new SwingWorker<String, Void>() {
            @Override protected String doInBackground() throws Exception {
                Instances data = MLPModel.loadDataset(dataset);
                Evaluation result = MLPModel.evaluateCrossValidation(data, foldCount, cvRandomSeed, rate, mom, ep, modelSeed, hiddenValue);
                long start = System.nanoTime();
                MultilayerPerceptron fullModel = MLPModel.trainFullModel(data, rate, mom, ep, modelSeed, hiddenValue);
                double buildSeconds = (System.nanoTime() - start) / 1_000_000_000.0;
                StringBuilder b = new StringBuilder();
                b.append("=== Run information ===\n\n")
                 .append("Scheme:       weka.classifiers.functions.MultilayerPerceptron -L ").append(rate)
                 .append(" -M ").append(mom).append(" -N ").append(ep).append(" -V 0 -S ").append(modelSeed)
                 .append(" -E 20 -H ").append(hiddenValue).append("\n")
                 .append("Relation:     ").append(data.relationName()).append("\n")
                 .append("Instances:    ").append(data.numInstances()).append("\n")
                 .append("Attributes:   ").append(data.numAttributes()).append("\n");
                for (int i = 0; i < data.numAttributes(); i++) {
					b.append("              ").append(data.attribute(i).name()).append("\n");
				}
                b.append("Test mode:    ").append(foldCount).append("-fold cross-validation\n")
                 .append("CV random seed: ").append(cvRandomSeed).append("\n\n")
                 .append("=== Classifier model (full training set) ===\n\n")
                 .append(fullModel).append("\nTime taken to build model: ").append(String.format(Locale.US, "%.2f", buildSeconds)).append(" seconds\n\n")
                 .append("=== Stratified cross-validation ===\n")
                 .append(result.toSummaryString("\n=== Summary ===\n", false)).append("\n")
                 .append("=== Detailed Accuracy By Class ===\n").append(result.toClassDetailsString()).append("\n")
                 .append("=== Confusion Matrix ===\n").append(result.toMatrixString()).append("\n")
                 .append("Note: metrics are from cross-validation; the displayed model is trained on the full dataset.\n")
                 .append("Exact weight values can vary by WEKA version and dataset contents.\n");
                return b.toString();
            }
            @Override protected void done() {
                try { output.setText(get()); output.setCaretPosition(0); }
                catch (Exception ex) { showError(ex.getCause() == null ? ex : ex.getCause()); }
                finally { setBusy(false); }
            }
        }.execute();
    }

    private void runLearningRateExperiment() {
        final String dataset = selectedDataset();
        final double mom;
        final int ep, modelSeed, foldCount, cvRandomSeed;
        final String hiddenValue;
        try {
            mom = readDouble(momentum, "Momentum");
            ep = readInt(epochs, "Epochs");
            modelSeed = readInt(classifierSeed, "Classifier seed");
            foldCount = readInt(folds, "CV folds");
            cvRandomSeed = readInt(cvSeed, "CV random seed");
            hiddenValue = hiddenLayers();
        } catch (Exception ex) { showError(ex); return; }
        setBusy(true);
        output.setText("Comparing learning rates 0.1, 0.3, and 0.5 using the same dataset and cross-validation split...\n");
        new SwingWorker<String, Void>() {
            @Override protected String doInBackground() throws Exception {
                Instances data = MLPModel.loadDataset(dataset);
                File out = new File("results/learning_rate_experiment.csv");
                if (out.getParentFile() != null) {
					out.getParentFile().mkdirs();
				}
                StringBuilder b = new StringBuilder("Learning-rate experiment\nDataset: ").append(dataset).append(" (N=").append(data.numInstances()).append(")\n\n");
                try (PrintWriter writer = new PrintWriter(out, java.nio.charset.StandardCharsets.UTF_8)) {
                    writer.println("dataset,instances,folds,cv_seed,learning_rate,momentum,epochs,classifier_seed,hidden_layers,accuracy_percent,kappa,mae,rmse");
                    for (double rate : new double[]{0.1, 0.3, 0.5}) {
                        Evaluation result = MLPModel.evaluateCrossValidation(data, foldCount, cvRandomSeed, rate, mom, ep, modelSeed, hiddenValue);
                        writer.printf(Locale.US, "%s,%d,%d,%d,%.1f,%.4f,%d,%d,%s,%.6f,%.8f,%.8f,%.8f%n",
                                data.relationName(), data.numInstances(), foldCount, cvRandomSeed, rate, mom, ep, modelSeed, hiddenValue,
                                result.pctCorrect(), result.kappa(), result.meanAbsoluteError(), result.rootMeanSquaredError());
                        b.append(String.format(Locale.US, "Learning rate %.1f: accuracy %.2f%%, kappa %.4f, MAE %.5f, RMSE %.5f%n",
                                rate, result.pctCorrect(), result.kappa(), result.meanAbsoluteError(), result.rootMeanSquaredError()));
                    }
                }
                b.append("\nMeasured results saved to: ").append(out.getCanonicalPath());
                return b.toString();
            }
            @Override protected void done() {
                try { output.setText(get()); output.setCaretPosition(0); }
                catch (Exception ex) { showError(ex.getCause() == null ? ex : ex.getCause()); }
                finally { setBusy(false); }
            }
        }.execute();
    }

    private void showError(Throwable ex) {
        output.setText("Operation failed: " + ex.getMessage());
        JOptionPane.showMessageDialog(this, ex.getMessage(), "WekaProject error", JOptionPane.ERROR_MESSAGE);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new WekaProjectGUI().setVisible(true));
    }
}

package de.a0h.randomclairvoyance.gui.desktop;

import java.awt.Button;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Panel;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import de.a0h.mininum.MnFuncs;
import de.a0h.mininum.format.MnFormat;
import de.a0h.randomclairvoyance.lstm.ExampleData;
import de.a0h.randomclairvoyance.lstm.LstmCell;
import de.a0h.randomclairvoyance.lstm.LstmCellSequence;
import de.a0h.randomclairvoyance.lstm.Trainer;

@SuppressWarnings("serial")
public class DesktopApp extends Frame implements ActionListener {

	public static DesktopApp instance;

	int inputSize = 2;
	int stateSize = 2;
	int outputSize = 2;
	LstmCell cell = new LstmCell(inputSize, stateSize, outputSize);
	LstmCell sequGrad = cell.createLike();
	LstmCell cellGrad = cell.createLike();
	// LstmCell debugCell = cell.createLike();
	LstmCellSequence cellSequ;
	Trainer trainer;
	public long trainingCount;

	// String series = ExampleData.SERIES_RANDOM;
	// String series = ExampleData.SERIES_NORMAN2;
	// String series = ExampleData.SERIES_NORMAN;
	// String series = ExampleData.SERIES_DANIEL;
	String series = ExampleData.SERIES_REPEATED_11100011001010011000;

	float[][] xTrainSequ;
	float[][] yTrainSequ;
	float[][] xTestSequ;
	float[][] yTestSequ;
	float[][] ŷSequ;

	LstmCellDrawer cellDrawer = new LstmCellDrawer();

	JavaAwtColorSchemeLegend colorLegend = new JavaAwtColorSchemeLegend(cellDrawer.colorScheme);

	Canvas drawingCanvas = new Canvas() {
		@Override
		public void paint(Graphics g) {
			paintCanvas(g);
		}
	};

	Panel toolbar = new Panel();
	Button printBtn = new Button("print");
	Button next1Btn = new Button("next1");
	Button next10Btn = new Button("next10");
	Button next100Btn = new Button("next100");
	Button nextNBtn = new Button("nextN");
	TextField nextNTf = new TextField("1000");
	Button generateBtn = new Button("generate");
	Button resetBtn = new Button("reset");

	public DesktopApp() {
		super("LSTM");

		setBounds(0, 0, 1200, 720);

		KeyListener escToExitKeyListener = new KeyAdapter() {
			public void keyPressed(KeyEvent e) {
				if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
					System.exit(0);
				}
			}
		};

		this.addKeyListener(escToExitKeyListener);
		drawingCanvas.addKeyListener(escToExitKeyListener);
		toolbar.addKeyListener(escToExitKeyListener);
		printBtn.addKeyListener(escToExitKeyListener);
		next1Btn.addKeyListener(escToExitKeyListener);
		next10Btn.addKeyListener(escToExitKeyListener);
		next100Btn.addKeyListener(escToExitKeyListener);
		nextNBtn.addKeyListener(escToExitKeyListener);
		nextNTf.addKeyListener(escToExitKeyListener);
		generateBtn.addKeyListener(escToExitKeyListener);
		resetBtn.addKeyListener(escToExitKeyListener);

		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});

		setLayout(new GridBagLayout());

		GridBagConstraints gbc = new GridBagConstraints();

		add(drawingCanvas, this, gbc, 0, 0, 1, 1, 1, 1, GridBagConstraints.BOTH);
		add(toolbar, this, gbc, 0, 1, 2, 1, 0, 0, GridBagConstraints.BOTH);
		add(colorLegend, this, gbc, 1, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);

		drawingCanvas.setBackground(Color.DARK_GRAY.darker());
		toolbar.setBackground(new Color(204, 204, 204));

		toolbar.add(next1Btn);
		toolbar.add(next10Btn);
		toolbar.add(next100Btn);
		toolbar.add(nextNBtn);
		toolbar.add(nextNTf);
		toolbar.add(printBtn);
		toolbar.add(generateBtn);
		toolbar.add(resetBtn);

		printBtn.addActionListener(this);
		next1Btn.addActionListener(this);
		next10Btn.addActionListener(this);
		next100Btn.addActionListener(this);
		nextNBtn.addActionListener(this);
		nextNTf.addActionListener(this);
		generateBtn.addActionListener(this);
		resetBtn.addActionListener(this);

		setVisible(true);
	}

	private static void add(Component cmp, Container cnt, GridBagConstraints gbc, //
			int gridx, int gridy, int gridwidth, int gridheight, int weightx, int weighty, int fill) {
		gbc.gridx = gridx;
		gbc.gridy = gridy;
		gbc.gridwidth = gridwidth;
		gbc.gridheight = gridheight;
		gbc.weightx = weightx;
		gbc.weighty = weighty;
		gbc.fill = fill;

		cnt.add(cmp, gbc);
	}

	public void actionPerformed(ActionEvent e) {
		Object src = e.getSource();

		int trainCount = -1;
		if (src == printBtn) {
			System.out.println("############# cell: " + cell.toCreateFunction(new StringBuilder()));
			System.out.println("############# gradSum: " + sequGrad);
			System.out.println("############# cellGrad: " + cellGrad);
			writeFile();
			// System.out.println("##############################################");
			// System.out.println(cell.toCreateFunction(new StringBuilder()));
		} else if (src == next1Btn) {
			trainCount = 1;
		} else if (src == next10Btn) {
			trainCount = 10;
		} else if (src == next100Btn) {
			trainCount = 100;
		} else if (src == nextNBtn || src == nextNTf) {
			String trainCountStr = nextNTf.getText();
			trainCount = Integer.parseInt(trainCountStr);
		} else if (src == generateBtn) {
			generate(300);
		} else if (src == resetBtn) {
			resetTraining();
		}

		final int trainCountFinal = trainCount;
		if (trainCountFinal != -1) {
			new Thread() {
				public void run() {
					train(trainCountFinal);
				}
			}.start();
		}
	}

	private void writeFile() {
		PrintWriter po;
		FileWriter fo;
		try {
			fo = new FileWriter("/home/user/lstm_debug_java.txt");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		po = new PrintWriter(fo);

		po.println(cell.toString());

		try {
			fo.close();
		} catch (IOException ignored) {
		}
	}

	protected void resetTraining() {
		cell.clear();
		cell.initWeightsAndBiasesRandomly(0);
		sequGrad.clear();
		cellGrad.clear();
		trainer.learningRateMin = 0.01f;
		trainer.learningRateAdaptive = 0.5f;
		trainer.regularizationLambda = 0.001f;

		trainingCount = 0;

		drawingCanvas.repaint();
	}

	private void generate(int length) {
		float[] x = new float[2];
		x[0] = 0;

		StringBuilder buf = new StringBuilder(length);
		for (int i = 0; i < length; i++) {
			cell.calculateForward(x);
			cell.getŶ(x);
			cell.swapStates();

			int cat = MnFuncs.idxMax(x);
			buf.append(cat);

			x = MnFuncs.indexToOnehot(2, cat);
		}

		System.out.println(buf);
	}

	public void paintCanvas(Graphics g) {
		int x = cellDrawer.externalSpacing;
		int y = cellDrawer.externalSpacing;

		cellDrawer.draw(cell, x, y, g);
		x += cellDrawer.getWidth(cell);

		cellDrawer.draw(sequGrad, x, y, g);
		x += cellDrawer.getWidth(cell);

		cellDrawer.draw(cellGrad, x, y, g);
		x += cellDrawer.getWidth(cell);

		// cellDrawer.draw(debugCell, x, y, g);
		// x += cellDrawer.getWidth(cell);

		x = cellDrawer.externalSpacing;
		y += cellDrawer.getHeight(cell);
	}

	private void train(int count) {
		Graphics g = drawingCanvas.getGraphics();

		System.out.println("" //
				+ "trainingCount " //
				+ "learningRate " //
				+ "loss_before " //
				+ "trainLoss " //
				+ "trainRate " //
				+ "testLoss " //
				+ "testRate " //
		// + "s0 " //
		// + "s1 " //
		// + "h0 " //
		// + "h1 " //
		);

		long paintInterval = 100;
		long lastPaint = System.currentTimeMillis();

		for (int i = 0; i < count; i++) {
			trainingCount++;

			float loss = trainer.train(cellSequ, xTrainSequ, yTrainSequ, sequGrad, cellGrad);

			float[] trainResults = test("train", xTrainSequ, yTrainSequ, 0);
			float[] testResults = test("test ", xTestSequ, yTestSequ, 0);

			loss /= 200;
			trainResults[0] /= 200;
			testResults[0] /= 100;

			System.out.println("" //
					+ trainingCount + " "//
					+ trainer.learningRate + " " //
					+ loss + " " //
					+ trainResults[0] + " " //
					+ trainResults[1] + " " //
					+ testResults[0] + " " //
					+ testResults[1] + " " //
			// + cell.s[0] + " " //
			// + cell.s[1] + " " //
			// + cell.h[0] + " " //
			// + cell.h[1] + " " //
			);

			sequGrad.mul(-1);
			cellGrad.mul(-1);

			// debugCell.set(sequGrad);
			// debugCell.setH(cellGrad.h);
			// debugCell.setS(cellGrad.s);
			// debugCell.divElwise(cell);
			// debugCell.reciprocalize();

			long paintAge = System.currentTimeMillis() - lastPaint;
			if (paintAge > paintInterval) {
				drawingCanvas.paint(g);
				lastPaint = System.currentTimeMillis();
			}
		}
		drawingCanvas.paint(g);
		g.dispose();

		test();
	}

	public void test() {
		test("test ", xTestSequ, yTestSequ, 2);
		test("train", xTrainSequ, yTrainSequ, 1);

		System.out.println();
	}

	protected float[] test(String sequName, float[][] xSequ, float[][] ySequ, int verbosity) {
		cellSequ.calculateForward(xSequ);
		cellSequ.getŶ(ŷSequ);

		LstmCell sequGrad_ = cell.createLike();
		LstmCell cellGrad_ = cell.createLike();
		float loss = cellSequ.calculateGradient(xSequ, ySequ, sequGrad_, cellGrad_);

		if (verbosity >= 2) {
			int len = Math.min(xSequ.length, 300);

			String str;
			str = MnFuncs.similarOnehotSequenceToCharString(xSequ);
			System.out.println("x: " + str.substring(0, len));
			str = MnFuncs.similarOnehotSequenceToCharString(ySequ);
			System.out.println("y: " + str.substring(0, len));
			str = MnFuncs.similarOnehotSequenceToCharString(ŷSequ);
			System.out.println("ŷ: " + str.substring(0, len));

			// for (int h = 10; h >= 1; h--) {
			// System.out.print("| ");
			//
			// for (int i = 0; i < ŷAr.length; i++) {
			// float[] ŷ = ŷAr[i];
			// float prob = MiniNum.max(ŷ);
			// System.out.print(prob * 10 + .5 > h ? '#' : ' ');
			// }
			//
			// System.out.println();
			// }
		}

		int correct = 0;
		for (int i = 0; i < xSequ.length; i++) {
			int targetValue = MnFuncs.idxMax(ySequ[i]);
			int actualValue = MnFuncs.idxMax(ŷSequ[i]);

			correct += actualValue == targetValue ? 1 : 0;
		}

		float rate = ((float) correct) / ySequ.length;
		if (verbosity >= 1) {
			System.out.println(sequName + " " //
					+ "rate: " + MnFormat.FORMAT_PRETTY.get().format(rate) //
					+ " (" + correct + " of " + ySequ.length + ")");
		}

		return new float[]{loss, rate};
	}

	private float[][] stringToOneHotSeries(String strSeries) {
		float[][] oneHotSeries = new float[strSeries.length()][];

		for (int i = 0; i < strSeries.length(); i++) {
			int symbol = strSeries.charAt(i) - '0';

			oneHotSeries[i] = MnFuncs.indexToOnehot(2, symbol);
		}

		return oneHotSeries;
	}

	public static void main(String[] args) {
		instance = new DesktopApp();
		instance.instanceMain(args);
	}

	private void instanceMain(String[] args) {
		String trainSeries = series.substring(0, series.length() * 2 / 3);
		String testSeries = series.substring(series.length() * 2 / 3);

		xTrainSequ = stringToOneHotSeries(trainSeries.substring(0, trainSeries.length() - 1));
		yTrainSequ = stringToOneHotSeries(trainSeries.substring(1, trainSeries.length()));
		xTestSequ = stringToOneHotSeries(testSeries.substring(0, testSeries.length() - 1));
		yTestSequ = stringToOneHotSeries(testSeries.substring(1, testSeries.length()));
		ŷSequ = new float[xTrainSequ.length][outputSize];

		cellSequ = new LstmCellSequence(cell, xTrainSequ.length);

		trainer = new Trainer();

		resetTraining();

		test();
	}
}

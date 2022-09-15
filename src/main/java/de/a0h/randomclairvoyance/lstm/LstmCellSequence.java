package de.a0h.randomclairvoyance.lstm;

/**
 * For training an LSTM cell efficiently, a sequence of connected cells is
 * required. This class creates and represents such a sequence of cells and it
 * provides a suitable interface for it.
 */
public class LstmCellSequence {

	/**
	 * The internal array for storing the sequence of cells.
	 */
	protected LstmCell[] cell;

	/**
	 * Creates a new cell sequence with the specified cell0 as first cell. All
	 * neural layers will share the same memory for the weights and biases, so
	 * when modifying the weights of one cell, the weights of all other cells
	 * are modified automatically.
	 */
	public LstmCellSequence(LstmCell cell0, int length) {
		if (length < 1) {
			throw new IllegalArgumentException("length = " + length
					+ ", but it must be >= 1. This is because a sequence of length 0 or less is not usable.");
		}

		cell = new LstmCell[length];

		cell[0] = cell0;

		for (int i = 1; i < length; i++) {
			cell[i] = createSuccessor(cell[i - 1]);
		}
	}

	/**
	 * For a sequence of input and output pairs this method calculates the sum
	 * of all gradients of all cells in this sequence. The calculated sum of
	 * gradients, i.e. the sequence's gradients, is assigned to sequGrad and the
	 * loss is returned. The gradient of only the cell at index 0 of the
	 * sequence is stored in cellGrad.</br>
	 * Note that you should use sequGrad for learning the weights and biases,
	 * but for the start states s and h of cell at index 0 you should use the s
	 * and h of cellGrad. That's because the weights and biases are the same for
	 * all cells, thus they need to be changed in the overall direction (the sum
	 * of directions). But the state is something individual for every cell -
	 * every cell has another start state - so to learn s0 and h0, you need to
	 * look at only the individual gradient of cell 0.
	 */
	public float calculateGradient(float[][] xSequ, float[][] ySequ, LstmCell sequGrad, LstmCell cellGrad) {
		sequGrad.clear();
		cellGrad.clear();
		float loss = 0;

		for (int i = xSequ.length - 1; i >= 0; i--) {
			cellGrad.swapStates();

			loss += cell[i].calculateGradient(ySequ[i], cellGrad);

			sequGrad.add(cellGrad);
		}

		return loss;
	}

	public void calculateForward(float[][] xSequ) {
		for (int i = 0; i < xSequ.length; i++) {
			cell[i].calculateForward(xSequ[i]);
		}
	}

	public void getŶ(float[][] ŷSequ) {
		if (ŷSequ.length < cell.length) {
			throw new IllegalArgumentException("ŷSequ.length = " + ŷSequ.length
					+ ", but it must be at last of the length of this sequence, which is " + cell.length);
		}

		for (int i = 0; i < cell.length; i++) {
			cell[i].getŶ(ŷSequ[i]);
		}
	}

	public int getLength() {
		return cell.length;
	}

	/**
	 * Creates a new cell which is the successor to the specified predecessor
	 * cell. The new instance will share the memory for weights and biases.
	 * Moreover, the start state of the new instance will share the memory with
	 * the end state of the predecessor. This saves memory and processing time
	 * because no copying from one cell to the next is required for calculating
	 * the next cycle.
	 */
	public static LstmCell createSuccessor(LstmCell predecessor) {
		LstmCell successor = new LstmCell();

		successor.inputSize = predecessor.inputSize;
		successor.stateSize = predecessor.stateSize;
		successor.outputSize = predecessor.outputSize;

		successor.s = predecessor.s_;
		successor.h = predecessor.h_;

		successor.wf = predecessor.wf;
		successor.bf = predecessor.bf;
		successor.wm = predecessor.wm;
		successor.bm = predecessor.bm;
		successor.wp = predecessor.wp;
		successor.bp = predecessor.bp;
		successor.wo = predecessor.wo;
		successor.bo = predecessor.bo;

		successor.allocCalculatedVectors();

		return successor;
	}

	public void mulAddWeightsAndBiases(LstmCell grad, float negLearningRate) {
		cell[0].mulAddWeightsAndBiases(grad, negLearningRate);
	}

	public void mulAddStartStates(LstmCell grad, float negLearningRate) {
		cell[0].mulAddStartStates(grad, negLearningRate);
	}
}

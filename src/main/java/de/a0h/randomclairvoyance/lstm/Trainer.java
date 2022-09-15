package de.a0h.randomclairvoyance.lstm;

import de.a0h.mininum.MnFuncs;
import de.a0h.mininum.MnLinalg;

/**
 * Classic gradient descent trainer. Same as an optimizer, but since an
 * optimizer usually doesn't find the optimum, it's more accurate to call it a
 * trainer.
 */
public class Trainer {

	public static final float LEARNING_RATE_ADAPTATION_FACTOR = 0.999f;

	public float learningRateMin;

	public float learningRateAdaptive;

	public float learningRate;

	public float regularizationLambda;

	/**
	 * Trains a cell on one step, returns the loss before the training, stores
	 * the gradient in grad.
	 */
	public float train(LstmCell cell, float[] x, float[] y, LstmCell grad) {
		cell.calculateForward(x);

		float loss = cell.calculateGradient(y, grad);

		adaptLearningRate();
		ensureNonZeroLearningRate();

		cell.mulAddWeightsAndBiases(grad, -learningRate);
		cell.mulAddStartStates(grad, -learningRate);

		return loss;
	}

	/**
	 * Trains the specified cell sequence with a sequence of input vectors xSequ
	 * and the corresponding sequence of desired output vectors ySequ.
	 * 
	 * @param sequGrad
	 *            used to return the sequence's gradients (sum of all cells'
	 *            gradients)
	 * @param cellGrad
	 *            used to return the gradient of cell 0
	 * @return the loss of the sequence (sum of losses of all cells) before the
	 *         training
	 */
	public float train(LstmCellSequence cellSequ, float[][] xSequ, float[][] ySequ, //
			LstmCell sequGrad, LstmCell cellGrad) {
		if (xSequ.length != ySequ.length) {
			throw new IllegalArgumentException(
					"xSequ.length and ySequ.length must be equal, but they are not (xSequ.length = " + xSequ.length
							+ ", ySequ.length" + ySequ.length + ")");
		}

		cellSequ.calculateForward(xSequ);

		float loss = cellSequ.calculateGradient(xSequ, ySequ, sequGrad, cellGrad);

		if (regularizationLambda != 0.0f) {
			loss += calculateWeightsAndBiasesRegGradient(cellSequ.cell[0], sequGrad);
			loss += calculateStartStateRegGradient(cellSequ.cell[0], cellGrad);
		}

		adaptLearningRate();
		ensureNonZeroLearningRate();

		sequGrad.mul(1.0f / cellSequ.getLength());
		cellSequ.mulAddWeightsAndBiases(sequGrad, -learningRate);
		cellSequ.mulAddStartStates(cellGrad, -learningRate);

		return loss;
	}

	protected void adaptLearningRate() {
		learningRate = learningRateMin + learningRateAdaptive;
		learningRateAdaptive *= LEARNING_RATE_ADAPTATION_FACTOR;
	}

	protected void ensureNonZeroLearningRate() {
		if (learningRate == 0.0f) {
			throw new IllegalStateException("learningRate is zero. No learning will take place.");
		}
	}

	/**
	 * This method calculates the gradient solely based on regularization.
	 * 
	 * @param cell
	 *            the cell for which to calculate the gradient
	 * @param grad
	 *            used for returning the gradient, which is not assigned, but
	 *            added to values which might already exist there; do a
	 *            grad.clear() before calling this method if you want to make
	 *            sure you get only the result of this method.
	 * @return the loss which is based solely on the regularization of the
	 *         weights and biases
	 */
	public float calculateWeightsAndBiasesRegGradient(LstmCell cell, LstmCell grad) {
		float norm = getWeightAndBiasL2Norm(cell);
		float loss = regularizationLambda * norm / cell.getWeightAndBiasElementCount();

		if (norm != 0.0f) {
			float regFactor = regularizationLambda / norm / cell.getWeightAndBiasElementCount();

			MnLinalg.mulAdd(cell.wf, regFactor, grad.wf);
			MnLinalg.mulAdd(cell.bf, regFactor, grad.bf);
			MnLinalg.mulAdd(cell.wm, regFactor, grad.wm);
			MnLinalg.mulAdd(cell.bm, regFactor, grad.bm);
			MnLinalg.mulAdd(cell.wp, regFactor, grad.wp);
			MnLinalg.mulAdd(cell.bp, regFactor, grad.bp);
			MnLinalg.mulAdd(cell.wo, regFactor, grad.wo);
			MnLinalg.mulAdd(cell.bo, regFactor, grad.bo);
		}

		return loss;
	}

	/**
	 * This method calculates the gradient solely based on regularization for
	 * the start states s and h.
	 * 
	 * @param cell
	 *            the cell for which to calculate the gradient
	 * @param grad
	 *            used for returning the gradient, which is not assigned, but
	 *            added to values which might already exist there; do a
	 *            grad.clear() before calling this method if you want to make
	 *            sure you get only the result of this method.
	 * @return the loss which is based solely on the regularization of the start
	 *         states
	 */
	public float calculateStartStateRegGradient(LstmCell cell, LstmCell grad) {
		float norm = getStartStateL2Norm(cell);
		float loss = regularizationLambda * norm / cell.getStartStateElementCount();

		if (norm != 0.0f) {
			float regFactor = regularizationLambda / norm / cell.getStartStateElementCount();
			MnLinalg.mulAdd(cell.s, regFactor, grad.s);
			MnLinalg.mulAdd(cell.h, regFactor, grad.h);
		}

		return loss;
	}

	protected float getWeightAndBiasL2Norm(LstmCell cell) {
		float result = 0;

		result += MnFuncs.getSqSum(cell.wf);
		result += MnFuncs.getSqSum(cell.bf);
		result += MnFuncs.getSqSum(cell.wm);
		result += MnFuncs.getSqSum(cell.bm);
		result += MnFuncs.getSqSum(cell.wp);
		result += MnFuncs.getSqSum(cell.bp);
		result += MnFuncs.getSqSum(cell.wo);
		result += MnFuncs.getSqSum(cell.bo);

		result = MnFuncs.sqrt(result);

		return result;
	}

	protected float getStartStateL2Norm(LstmCell cell) {
		float result = 0;

		result += MnFuncs.getSqSum(cell.s);
		result += MnFuncs.getSqSum(cell.h);

		result = MnFuncs.sqrt(result);

		return result;
	}
}

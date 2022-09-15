package de.a0h.randomclairvoyance.lstm;

/**
 * Interface which describes the API of a recurrent neural network cell (RNN
 * cell). Various kinds of RNNs have been invented, mostly differing in internal
 * and behaviors, but with similar interfaces to the outside, like input and
 * output.
 * 
 * <p>
 * The intention of this class is to provide a general interface so that
 * different cell types can be plugged in to other code without changing the
 * other code.
 * </p>
 */
public interface RnnCell {

	/**
	 * Initializes the cell fields. This usually involved generating random values
	 * for all the learnable fields in the cell. When this is the case, the
	 * specified seed shall be used to initialize the random value generator so that
	 * using the same seed again shall result in the same random values to be
	 * generated.
	 */
	public void init(long seed);

	/**
	 * Train on one further step. x is the input data and y is the desired output
	 * data that the cell shall produce.
	 */
	public void train(float[] x, float[] y, float learningRate);

	/**
	 * Train on a sequence of input/output pairs.
	 */
	public void trainSequence(float[][] xAr, float[][] yAr, float learningRate);

	/**
	 * Evaluates one further step using the specified input data x. After finishing,
	 * the output of the evaluation is stored in the array ŷ.
	 */
	public void eval(float[] x, float[] ŷ);

	/**
	 * Evaluates a sequence of further inputs and produces a sequence of outputs,
	 * which are stored in ŷAr.
	 */
	public void evalSequence(float[][] xAr, float[][] ŷAr);

}

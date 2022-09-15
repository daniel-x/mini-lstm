package de.a0h.randomclairvoyance.lstm;

import java.util.Arrays;
import java.util.Random;

import de.a0h.minideeplearn.operation.activation.Sigmoid;
import de.a0h.minideeplearn.operation.activation.Softmax;
import de.a0h.minideeplearn.operation.activation.Tanh;
import de.a0h.mininum.MnFuncs;
import de.a0h.mininum.MnLinalg;
import de.a0h.mininum.format.MnFormat;

public class LstmCell {

	// special characters
	// ¡¿
	// ÄäÀàÁáÂâÃãÅåǍǎĄąĂăÆæĀā
	// ÇçĆćĈĉČč
	// ĎđĐďð
	// ÈèÉéÊêËëĚěĘęĖėĒē
	// ĜĝĢģĞğ
	// Ĥĥ
	// ÌìÍíÎîÏïıĪīĮį
	// Ĵĵ
	// Ķķ
	// ĹĺĻļŁłĽľ
	// ÑñŃńŇňŅņ
	// ÖöÒòÓóÔôÕõŐőØøŒœ
	// ŔŕŘř
	// ẞßŚśŜŝŞşŠšȘș
	// ŤťŢţÞþȚț
	// ÜüÙùÚúÛûŰűŨũŲųŮůŪū
	// Ŵŵ
	// ÝýŸÿŶŷ
	// ŹźŽžŻż

	// /**
	// * We keep the start state in this array to be able to restore it after
	// running
	// * on a sequence.
	// */
	// float[] s0;
	//
	// /**
	// * We keep the start state in this array to be able to restore it after
	// running
	// * on a sequence.
	// */
	// float[] h0;

	/**
	 * Size of the input vectors which can be processed by this LSTM cell.
	 */
	public int inputSize;

	/**
	 * Size of the state vector. That's the size of the hidden state vector s
	 * and the visible state vector h. This is, like with neural network layers,
	 * not directly linked to the input size. The state size can be greater
	 * than, less than, or the same as the input size.
	 */
	public int stateSize;

	/**
	 * Size of the output vector. This LSTM implementation allows to have the
	 * output size be smaller than the state size. If this is the case, then the
	 * output ŷ is based on only the first outputSize many elements of h_.
	 * Still, the next cell cycle is based on the full h_ vector. The output
	 * size can be less than, the same or greater than the input size, but it
	 * can only be the same or less than the state size.
	 */
	public int outputSize;

	/**
	 * State, or start state, or end state of the previous run of the cell. This
	 * is often called c. I don't like to call it c like cell, because the cell
	 * is comprised not only of the state, but also other things. Hence the name
	 * cell is misrepresenting what this is, and state is a better name. Start
	 * state or end state of previous run.
	 */
	public float[] s;

	/**
	 * Input which comes from the previous cycle or from the predecessor cell's
	 * pre-output, or a part of the start state of the cell. Much like s.
	 */
	public float[] h;

	/**
	 * Concatenation of predecessor's pre-output and last input, hence of size
	 * inputSize + outputSize.
	 */
	public float[] z;

	/**
	 * Forget gate.
	 */
	public float[] f;

	/**
	 * Memorize gate.
	 */
	public float[] m;

	/**
	 * Prospective state values for the successor state.
	 */
	public float[] p;

	/**
	 * Output gate.
	 */
	public float[] o;

	/**
	 * Interim result. r = sPrev (*) f.
	 */
	public float[] r;

	/**
	 * Interim result. a = m (*) p.
	 */
	public float[] a;

	/**
	 * Interim result. q = tanh(s).
	 */
	public float[] q;

	/**
	 * Matrix Wf for the forget gate f = sig(wf * z + bf).
	 */
	public float[][] wf;

	/**
	 * Bias for forget gate.
	 */
	public float[] bf;

	/**
	 * Matrix for the memorize gate m = sig(wm * z + bm). The memorize gate
	 * determines what information to transfer from the prospect state to state.
	 */
	public float[][] wm;

	/**
	 * Bias for memorize gate.
	 */
	public float[] bm;

	/**
	 * Matrix for the prospect state p = tanh(wp * z + bp).
	 */
	public float[][] wp;

	/**
	 * Bias for prospect state.
	 */
	public float[] bp;

	/**
	 * Matrix for the output gate o = sig(wo * z + bo).
	 */
	public float[][] wo;

	/**
	 * Bias for output gate.
	 */
	public float[] bo;

	/**
	 * State of this cell after a forward step. This is the start state of the
	 * next cell or next cell cycle.
	 */
	public float[] s_;

	/**
	 * Feedback output, input h to the next cell or cell cycle.
	 */
	public float[] h_;

	/**
	 * Output which is relevant to the outside of the cell. 정말 그렇습니다! काेशिस गर!
	 */
	public float[] ŷ;

	/**
	 * Creates a new, uninitialized instance of this class without any memory
	 * allocated, i.e. the internal arrays are null.
	 */
	protected LstmCell() {
	}

	/**
	 */
	public LstmCell(int inputSize, int stateSize, int outputSize) {
		if (outputSize > stateSize) {
			throw new IllegalArgumentException("outputSize is " + outputSize
					+ ", but it may not be greater than stateSize, which is " + stateSize);
		}

		this.inputSize = inputSize;
		this.stateSize = stateSize;
		this.outputSize = outputSize;

		alloc();
	}

	protected void alloc() {
		allocStartState();
		allocWeightsAndBiases();
		allocCalculatedVectors();
	}

	protected void allocStartState() {
		s = new float[stateSize];
		h = new float[stateSize];
	}

	protected void allocWeightsAndBiases() {
		int compoundLength = stateSize + inputSize;

		wf = new float[stateSize][compoundLength];
		bf = new float[stateSize];
		wm = new float[stateSize][compoundLength];
		bm = new float[stateSize];
		wp = new float[stateSize][compoundLength];
		bp = new float[stateSize];
		wo = new float[stateSize][compoundLength];
		bo = new float[stateSize];
	}

	protected void allocCalculatedVectors() {
		int compoundLength = stateSize + inputSize;

		z = new float[compoundLength];

		f = new float[stateSize];
		m = new float[stateSize];
		p = new float[stateSize];
		o = new float[stateSize];
		r = new float[stateSize];
		a = new float[stateSize];
		q = new float[stateSize];

		s_ = new float[stateSize];
		h_ = new float[stateSize];
		ŷ = new float[outputSize];
	}

	/**
	 * Swaps the values of the start state with those of the end state. I.e. of
	 * s with s_ and those of h with h_. This method is used in stepwise
	 * evaluation when past inner cell states are not important.
	 */
	public void swapStates() {
		float[] tmp = new float[stateSize];

		System.arraycopy(s, 0, tmp, 0, stateSize);
		System.arraycopy(s_, 0, s, 0, stateSize);
		System.arraycopy(tmp, 0, s_, 0, stateSize);

		System.arraycopy(h, 0, tmp, 0, stateSize);
		System.arraycopy(h_, 0, h, 0, stateSize);
		System.arraycopy(tmp, 0, h_, 0, stateSize);
	}

	/**
	 * Calculates one cell step forward. You have to call {@link #swapStates()}
	 * between successive calls of this method.
	 */
	public void calculateForward(float[] x) {
		System.arraycopy(h, 0, z, 0, stateSize);
		stackX(x);

		// forget
		forwardLayerSig(wf, z, bf, f);
		MnLinalg.mul(s, f, r);

		// memorize
		forwardLayerSig(wm, z, bm, m);
		forwardLayerTanh(wp, z, bp, p);
		MnLinalg.mul(m, p, a);
		MnLinalg.add(r, a, s_);

		// output
		forwardLayerSig(wo, z, bo, o);
		Tanh.calc(s_, q);
		MnLinalg.mul(o, q, h_);

		System.arraycopy(h_, 0, ŷ, 0, outputSize);
		Softmax.calc(ŷ, ŷ);
	}

	/**
	 * y = sigmoid(a*x + b)
	 */
	protected static float[] forwardLayerSig(float[][] a, float[] x, float[] b, float[] y) {
		MnLinalg.mulMatVec(a, x, y);
		MnLinalg.add(y, b);
		Sigmoid.calc(y, y);

		return y;
	}

	/**
	 * y = tanh(a*x + b)
	 */
	protected static float[] forwardLayerTanh(float[][] a, float[] x, float[] b, float[] y) {
		MnLinalg.mulMatVec(a, x, y);
		MnLinalg.add(y, b);
		Tanh.calc(y, y);

		return y;
	}

	public void calculateForward(float[][] xSequ, float[][] ŷSequ, int stepCount) {
		if (stepCount > 0) {
			calculateForward(xSequ[0]);
		}
		for (int i = 1; i < stepCount - 1; i++) {
			swapStates();
			calculateForward(xSequ[i]);

			if (ŷSequ != null) {
				getŶ(ŷSequ[i]);
			}
		}
	}

	public static final int copy(float[] src, float[] dst, int dstOffset) {
		int length = src.length;

		System.arraycopy(src, 0, dst, dstOffset, length);

		return length;
	}

	public static final int copy(float[][] src, float[] dst, int dstOffset) {
		int lenSum = 0;

		for (int i = 0; i < src.length; i++) {
			int len = copy(src[i], dst, dstOffset);
			lenSum += len;
			dstOffset += len;
		}

		return lenSum;
	}

	/**
	 * Returns all values in this cell stored in a single vector.
	 */
	public float[] flatten() {
		int elementCount = getWeightAndBiasElementCount() + 12 * stateSize + inputSize + outputSize;
		float[] result = new float[elementCount];

		int off = 0;

		off += copy(s, result, off);
		off += copy(h, result, off);
		off += copy(z, result, off);

		off += copy(wf, result, off);
		off += copy(bf, result, off);
		off += copy(wm, result, off);
		off += copy(bm, result, off);
		off += copy(wp, result, off);
		off += copy(bp, result, off);
		off += copy(wo, result, off);
		off += copy(bo, result, off);

		off += copy(f, result, off);
		off += copy(m, result, off);
		off += copy(p, result, off);
		off += copy(o, result, off);
		off += copy(r, result, off);
		off += copy(a, result, off);
		off += copy(q, result, off);

		off += copy(s_, result, off);
		off += copy(h_, result, off);
		off += copy(ŷ, result, off);

		return result;
	}

	/**
	 * Calculates the gradient of some loss function in respect to everything in
	 * this cell. The loss, i.e. the value of the loss function, is returned.
	 * 
	 * @param y
	 *            ground truth for ŷ used to calculate the loss
	 * @param grad
	 *            At invocation of this method, grad.s_ and grad.h_ are expected
	 *            to hold the gradient in respect to s and to h of the successor
	 *            time step. these are backpropagated from the successor. After
	 *            this method returns, grad holds the results of the gradient
	 *            calculation.
	 */
	public float calculateGradient(float[] y, LstmCell grad) {
		// using grad.s and grad.h as temporary variables
		float[] tmp0 = grad.s;
		float[] tmp1 = grad.h;

		float loss = -1.0f / outputSize * MnFuncs.crossEntropy_zeroSafe(y, ŷ);

		// the loss function gets derived and the gradients are back propagated
		// from variable to variable through the calculation graph

		// dL/dŷ = -1/n * (y ⊘ ŷ)
		MnLinalg.div(y, ŷ, grad.ŷ);
		MnLinalg.mul(grad.ŷ, -1.0f / outputSize, grad.ŷ);

		// dL/dh_
		// = dL/dŷ * dŷ/dh_ + (dL/dh_ backprop from successor step)
		// = 1/n * (ŷ - y) + (dL/dh_ backprop from successor step)
		MnLinalg.sub(ŷ, y, tmp0);
		// in case outputSize < stateSize
		Arrays.fill(tmp0, outputSize, stateSize, 0.0f);

		MnLinalg.mulAdd(tmp0, 1.0f / outputSize, grad.h_);

		// dL/do = dL/dh ⊙ (dh/do) = dL/dh ⊙ q
		MnLinalg.mul(grad.h_, q, grad.o);

		// dL/dq = dL/dŷ ⊙ (dŷ/dq) = dL/dŷ ⊙ o
		MnLinalg.mul(grad.h_, o, grad.q);

		// dL/ds
		// = (dL/ds backprop from successor step) + dL/dq ⊙ (dq/ds)
		// = (dL/dŷ backprop from successor step) + dL/dq ⊙ (1 - q^2)

		calcTanhDerivative_basedOnTanh(q, tmp0);
		MnLinalg.mul(grad.q, tmp0, tmp0);
		MnLinalg.add(tmp0, grad.s_, grad.s_);

		// s = a + r, so their gradients are all the same
		System.arraycopy(grad.s_, 0, grad.a, 0, stateSize);
		System.arraycopy(grad.s_, 0, grad.r, 0, stateSize);

		// dL/dm = dL/da ⊙ (da/dm) = dL/da ⊙ p
		MnLinalg.mul(grad.a, p, grad.m);

		// dL/dp = dL/da ⊙ (da/dp) = dL/da ⊙ m
		MnLinalg.mul(grad.a, m, grad.p);

		// dL/df = dL/dr ⊙ (dr/df) = dL/dr ⊙ sPrev
		MnLinalg.mul(grad.r, s, grad.f);

		// not yet used variables: dguv

		// dL/dbf = dL/df * df/dbf_i = dL/df * f⊙(1-f)
		// gradients for bm, bp and bo are calculated similarly to bf
		// dL/dwf_i = dL/df * df/dwf_i = dL/df ⊙ f⊙(1-f) ⊗ z
		// ......... = dL/db_f ⊗ z
		// gradients for wm, wp and wo are calculated similarly to wf
		calcSigmoidDerivative_basedOnSigmoid(f, tmp0);
		MnLinalg.mul(tmp0, grad.f, grad.bf);
		MnLinalg.outerProduct(grad.bf, z, grad.wf);

		calcSigmoidDerivative_basedOnSigmoid(m, tmp0);
		MnLinalg.mul(tmp0, grad.m, grad.bm);
		MnLinalg.outerProduct(grad.bm, z, grad.wm);

		calcTanhDerivative_basedOnTanh(p, tmp0);
		MnLinalg.mul(tmp0, grad.p, grad.bp);
		MnLinalg.outerProduct(grad.bp, z, grad.wp);

		calcSigmoidDerivative_basedOnSigmoid(o, tmp0);
		MnLinalg.mul(tmp0, grad.o, grad.bo);
		MnLinalg.outerProduct(grad.bo, z, grad.wo);

		// dL/dz = dL/df * df/dz + ... = (dL/df * f⊙(1-f)) * wf + ...
		tmp1 = new float[z.length];

		calcSigmoidDerivative_basedOnSigmoid(f, tmp0);
		MnLinalg.mul(tmp0, grad.f, tmp0);
		MnLinalg.mulVecMat(tmp0, wf, grad.z);

		calcSigmoidDerivative_basedOnSigmoid(m, tmp0);
		MnLinalg.mul(tmp0, grad.m, tmp0);
		MnLinalg.mulVecMat(tmp0, wm, tmp1);
		MnLinalg.add(grad.z, tmp1, grad.z);

		// p = tanh(...) instead of sig(...), so the derivatives for wp are
		// slightly different
		// dL/dz = ... + dL/dp * dp/dz + ... = ... + (dL/dp * (1-p⊙^2)) * wp +
		// ...
		calcTanhDerivative_basedOnTanh(p, tmp0);
		MnLinalg.mul(tmp0, grad.p, tmp0);
		MnLinalg.mulVecMat(tmp0, wp, tmp1);
		MnLinalg.add(grad.z, tmp1, grad.z);

		calcSigmoidDerivative_basedOnSigmoid(o, tmp0);
		MnLinalg.mul(tmp0, grad.o, tmp0);
		MnLinalg.mulVecMat(tmp0, wo, tmp1);
		MnLinalg.add(grad.z, tmp1, grad.z);

		// dL/dh = dL/dz (for i = 0..outputSize-1)
		System.arraycopy(grad.z, 0, grad.h, 0, stateSize);

		// dL/ds = dL/dr ⊙ dr/ds = dL/dr ⊙ f
		MnLinalg.mul(grad.r, f, grad.s);

		return loss;
	}

	/**
	 * @param out
	 *            output of the tanh function
	 * @param grad_inp
	 *            result of this function, the derivative of the input of the
	 *            tanh function
	 */
	public void calcTanhDerivative_basedOnTanh(float[] out, float[] grad_inp) {
		for (int i = 0; i < out.length; i++) {
			grad_inp[i] = 1.0f - out[i] * out[i];
		}
	}

	/**
	 * @param out
	 *            output of the sigmoid function
	 * @param grad_inp
	 *            result of this function, the derivative of the input of the
	 *            sigmoid function
	 */
	public void calcSigmoidDerivative_basedOnSigmoid(float[] out, float[] grad_inp) {
		for (int i = 0; i < out.length; i++) {
			grad_inp[i] = out[i] * (1.0f - out[i]);
		}
	}

	protected int getWeightAndBiasElementCount() {
		return 0 //
				+ 4 * stateSize * (stateSize + inputSize) // weight count
				+ 4 * stateSize // bias count
		;
	}

	protected int getStartStateElementCount() {
		return 2 * stateSize;
	}

	public static float getXavierSigma(int inputNeuronCount, int outputNeuronCount) {
		return MnFuncs.sqrt(2.0f / (inputNeuronCount + outputNeuronCount));
	}

	/**
	 * Initialize cell weights and biases randomly.
	 */
	public void initWeightsAndBiasesRandomly(long seed) {
		Random rnd = new Random(seed);
		float xavierSigma = getXavierSigma(stateSize + inputSize, stateSize);

		// MnLinalg.assignRandomly(s, rnd);
		// MnLinalg.assignRandomly(h, rnd);
		// MnLinalg.mul(s, xavierSigma, s);
		// MnLinalg.mul(h, xavierSigma, h);
		MnLinalg.assign(s, 0);
		MnLinalg.assign(h, 0);

		MnFuncs.assignGaussian(wf, rnd);
		MnFuncs.assignGaussian(wm, rnd);
		MnFuncs.assignGaussian(wp, rnd);
		MnFuncs.assignGaussian(wo, rnd);
		MnLinalg.mul(wf, xavierSigma, wf);
		MnLinalg.mul(wm, xavierSigma, wm);
		MnLinalg.mul(wp, xavierSigma, wp);
		MnLinalg.mul(wo, xavierSigma, wo);

		MnFuncs.assignGaussian(bf, rnd);
		MnFuncs.assignGaussian(bm, rnd);
		MnFuncs.assignGaussian(bp, rnd);
		MnFuncs.assignGaussian(bo, rnd);
		MnLinalg.mul(bf, xavierSigma, bf);
		MnLinalg.mul(bm, xavierSigma, bm);
		MnLinalg.mul(bp, xavierSigma, bp);
		MnLinalg.mul(bo, xavierSigma, bo);
		// MnLinalg.assign(bf, 0);
		// MnLinalg.assign(bm, 0);
		// MnLinalg.assign(bp, 0);
		// MnLinalg.assign(bo, 0);

	}

	/**
	 * Multiplies the start states s and h of the gradient object by
	 * negLearningRate and then adds them to the respective fields of this cell.
	 * The content of grad remains unmodified.
	 */
	public void mulAddStartStates(LstmCell grad, float negLearningRate) {
		MnLinalg.mulAdd(grad.s, negLearningRate, s);
		MnLinalg.mulAdd(grad.h, negLearningRate, h);
	}

	/**
	 * Multiplies the weights and biases of the gradient object by
	 * negLearningRate and then adds them to the respective fields of this cell.
	 * The content of grad remains unmodified.
	 */
	public void mulAddWeightsAndBiases(LstmCell grad, float negLearningRate) {
		MnLinalg.mulAdd(grad.wf, negLearningRate, wf);
		MnLinalg.mulAdd(grad.bf, negLearningRate, bf);

		MnLinalg.mulAdd(grad.wm, negLearningRate, wm);
		MnLinalg.mulAdd(grad.bm, negLearningRate, bm);

		MnLinalg.mulAdd(grad.wp, negLearningRate, wp);
		MnLinalg.mulAdd(grad.bp, negLearningRate, bp);

		MnLinalg.mulAdd(grad.wo, negLearningRate, wo);
		MnLinalg.mulAdd(grad.bo, negLearningRate, bo);
	}

	public void set(LstmCell src) {
		MnLinalg.assign(src.s, this.s);
		MnLinalg.assign(src.h, this.h);

		MnLinalg.assign(src.wf, this.wf);
		MnLinalg.assign(src.bf, this.bf);
		MnLinalg.assign(src.wm, this.wm);
		MnLinalg.assign(src.bm, this.bm);
		MnLinalg.assign(src.wp, this.wp);
		MnLinalg.assign(src.bp, this.bp);
		MnLinalg.assign(src.wo, this.wo);
		MnLinalg.assign(src.bo, this.bo);

		MnLinalg.assign(src.z, this.z);
		MnLinalg.assign(src.f, this.f);
		MnLinalg.assign(src.m, this.m);
		MnLinalg.assign(src.p, this.p);
		MnLinalg.assign(src.o, this.o);
		MnLinalg.assign(src.r, this.r);
		MnLinalg.assign(src.a, this.a);
		MnLinalg.assign(src.q, this.q);

		MnLinalg.assign(src.s_, this.s_);
		MnLinalg.assign(src.h_, this.h_);
		MnLinalg.assign(src.ŷ, this.ŷ);
	}

	public StringBuilder bareWeightsAndBiasesToStringBuilder(StringBuilder buf) {
		int indentSpaces = 0;

		buf.append("wf: ");
		MnFormat.toStringBuilder(wf, buf, indentSpaces).append("\n");
		buf.append("bf: ");
		MnFormat.toStringBuilder(bf, buf).append("\n");
		buf.append("wm: ");
		MnFormat.toStringBuilder(wm, buf, indentSpaces).append("\n");
		buf.append("bm: ");
		MnFormat.toStringBuilder(bm, buf).append("\n");
		buf.append("wp: ");
		MnFormat.toStringBuilder(wp, buf, indentSpaces).append("\n");
		buf.append("bp: ");
		MnFormat.toStringBuilder(bp, buf).append("\n");
		buf.append("wo: ");
		MnFormat.toStringBuilder(wo, buf, indentSpaces).append("\n");
		buf.append("bo: ");
		MnFormat.toStringBuilder(bo, buf).append("\n");

		return buf;
	}

	public StringBuilder weightsAndBiasesToStringBuilder(StringBuilder buf) {
		buf.append(getClass().getSimpleName()).append("_weightsAndBiases\n");
		bareWeightsAndBiasesToStringBuilder(buf);
		buf.append("]");

		return buf;
	}

	public String weightsAndBiasesToString() {
		return weightsAndBiasesToStringBuilder(new StringBuilder()).toString();
	}

	// /**
	// * Stores the start state so it can be restored later.
	// */
	// public void storeState() {
	// getS(s0);
	// getH(h0);
	// }
	//
	// /**
	// * Restores the start state to whatever was stored last.
	// */
	// public void restoreState() {
	// setS(s0);
	// setH(h0);
	// }
	//
	// /**
	// * Calculates the gradient for a sequence of input and output pairs. The
	// * calculated gradient is assigned to grad and the loss is returned.
	// */
	// public float calculateSequenceGradient(float[][] xSequ, float[][] ySequ,
	// LstmCell grad) {
	// storeStartState();
	//
	// grad.clear();
	//
	// float loss = 0;
	//
	// LstmCell singleGrad = new LstmCell(inputSize, stateSize, outputSize);
	//
	// for (int i = xSequ.length - 1; i >= 0; i--) {
	// singleGrad.swapStates();
	//
	// calculateForward(xSequ, null, i + 1);
	// loss += calculateGradient(ySequ[i], singleGrad);
	//
	// grad.add(singleGrad);
	//
	// restoreStartState();
	// }
	//
	// return loss;
	// }

	/**
	 * Sets the the input vector for this cell.
	 */
	public void stackX(float[] x) {
		ensureOfInputSize(x, "input x");
		System.arraycopy(x, 0, z, stateSize, inputSize);
	}

	/**
	 * Assigns the values of the input vector to the provided parameter x.
	 */
	public void unstackX(float[] x) {
		ensureOfInputSize(x, "input x");
		System.arraycopy(z, stateSize, x, 0, inputSize);
	}

	/**
	 * Sets the state s of this cell. The return is done by assigning the
	 * results to the provided output array s. For convenience, s is returned.
	 */
	public float[] getS(float[] s) {
		ensureOfStateSize(s, "s");
		System.arraycopy(this.s, 0, s, 0, stateSize);

		return s;
	}

	/**
	 * Sets the state s of this cell.
	 */
	public void setS(float[] s) {
		ensureOfStateSize(s, "s");
		System.arraycopy(s, 0, this.s, 0, stateSize);
	}

	/**
	 * Gets ŷ of this cell by copying its contents to the specified array.
	 */
	public float[] getŶ(float[] ŷ) {
		ensureOfOutputSize(ŷ, "ŷ");
		System.arraycopy(this.ŷ, 0, ŷ, 0, outputSize);

		return ŷ;
	}

	/**
	 * Sets ŷ by copying the specified array's content to this cell's ŷ.
	 */
	public void setŶ(float[] ŷ) {
		ensureOfOutputSize(ŷ, "ŷ");
		System.arraycopy(ŷ, 0, this.ŷ, 0, stateSize);
	}

	/**
	 * Gets h (output which was forwarded from the previous LSTM step). The
	 * return is done by assigning the results to the provided parameter h. For
	 * convenience, the parameter is returned.
	 */
	public float[] getH(float[] h) {
		ensureOfStateSize(h, "h");
		System.arraycopy(this.h, 0, h, 0, stateSize);

		return h;
	}

	/**
	 * Sets h (output h which was forwarded from the previous LSTM step). The
	 * return is done by assigning the results to the provided output array
	 * hPrev.
	 */
	public void setH(float[] h) {
		ensureOfStateSize(h, "h");
		System.arraycopy(h, 0, this.h, 0, stateSize);
	}

	public void ensureOfInputSize(float[] a, String aName) {
		if (a == null) {
			throw new IllegalArgumentException(aName + " may not be null");
		}

		if (a.length != inputSize) {
			throw new IllegalArgumentException(aName + " has size " + a.length
					+ ", but it must match the input size of this cell, which is " + inputSize);
		}
	}

	public void ensureOfOutputSize(float[] a, String aName) {
		if (a == null) {
			throw new IllegalArgumentException(aName + " may not be null");
		}

		if (a.length != outputSize) {
			throw new IllegalArgumentException(aName + " has size " + a.length
					+ ", but it must match the output size of this cell, which is " + outputSize);
		}
	}

	public void ensureOfStateSize(float[] a, String aName) {
		if (a == null) {
			throw new IllegalArgumentException(aName + " may not be null");
		}

		if (a.length != stateSize) {
			throw new IllegalArgumentException(aName + " has size " + a.length
					+ ", but it must match the state size of this cell, which is " + stateSize);
		}
	}

	public StringBuilder toCreateFunction(StringBuilder buf) {
		buf.append("public static LstmCell createLstmCell() {\n");
		buf.append("    LstmCell c = new LstmCell(" + inputSize + ", " + stateSize + ", " + outputSize + ");\n\n");

		buf.append("    c.s = new float[] ");
		MnFormat.toRepr(buf, s).append(";\n");
		buf.append("    c.h = new float[] ");
		MnFormat.toRepr(buf, h).append(";\n");
		buf.append("    c.z = new float[] ");
		MnFormat.toRepr(buf, z).append(";\n");
		buf.append("\n");

		buf.append("    c.wf = new float[][] ");
		MnFormat.toRepr(buf, wf).append(";\n");
		buf.append("    c.bf = new float[] ");
		MnFormat.toRepr(buf, bf).append(";\n");
		buf.append("    c.wm = new float[][] ");
		MnFormat.toRepr(buf, wm).append(";\n");
		buf.append("    c.bm = new float[] ");
		MnFormat.toRepr(buf, bm).append(";\n");
		buf.append("    c.wp = new float[][] ");
		MnFormat.toRepr(buf, wp).append(";\n");
		buf.append("    c.bp = new float[] ");
		MnFormat.toRepr(buf, bp).append(";\n");
		buf.append("    c.wo = new float[][] ");
		MnFormat.toRepr(buf, wo).append(";\n");
		buf.append("    c.bo = new float[] ");
		MnFormat.toRepr(buf, bo).append(";\n");
		buf.append("\n");

		buf.append("    c.f = new float[] ");
		MnFormat.toRepr(buf, f).append(";\n");
		buf.append("    c.m = new float[] ");
		MnFormat.toRepr(buf, m).append(";\n");
		buf.append("    c.p = new float[] ");
		MnFormat.toRepr(buf, p).append(";\n");
		buf.append("    c.o = new float[] ");
		MnFormat.toRepr(buf, o).append(";\n");
		buf.append("    c.r = new float[] ");
		MnFormat.toRepr(buf, r).append(";\n");
		buf.append("    c.a = new float[] ");
		MnFormat.toRepr(buf, a).append(";\n");
		buf.append("    c.q = new float[] ");
		MnFormat.toRepr(buf, q).append(";\n");
		buf.append("\n");

		buf.append("    c.s_ = new float[] ");
		MnFormat.toRepr(buf, s_).append(";\n");
		buf.append("    c.h_ = new float[] ");
		MnFormat.toRepr(buf, h_).append(";\n");
		buf.append("    c.ŷ = new float[] ");
		MnFormat.toRepr(buf, ŷ).append(";\n");
		buf.append("\n");

		buf.append("    return c;\n");
		buf.append("}\n");

		return buf;
	}

	public StringBuilder toStringBuilder(StringBuilder buf) {
		buf.append(getClass().getSimpleName() + "[").append("\n");

		buf.append("s : ");
		MnFormat.toStringBuilder(s, buf).append("\n");
		buf.append("h : ");
		MnFormat.toStringBuilder(h, buf).append("\n");
		buf.append("z : ");
		MnFormat.toStringBuilder(z, buf).append("\n");

		bareWeightsAndBiasesToStringBuilder(buf);

		buf.append("f : ");
		MnFormat.toStringBuilder(f, buf).append("\n");
		buf.append("m : ");
		MnFormat.toStringBuilder(m, buf).append("\n");
		buf.append("p : ");
		MnFormat.toStringBuilder(p, buf).append("\n");
		buf.append("o : ");
		MnFormat.toStringBuilder(o, buf).append("\n");
		// buf.append("r : ");
		// MnFormat.toStringBuilder(r, buf).append("\n");
		// buf.append("a : ");
		// MnFormat.toStringBuilder(a, buf).append("\n");
		// buf.append("q : ");
		// MnFormat.toStringBuilder(q, buf).append("\n");

		buf.append("s_: ");
		MnFormat.toStringBuilder(s_, buf).append("\n");
		buf.append("h_: ");
		MnFormat.toStringBuilder(h_, buf).append("\n");
		buf.append("ŷ : ");
		MnFormat.toStringBuilder(ŷ, buf).append("\n");

		buf.append("]");

		return buf;
	}

	public String toString() {
		return toStringBuilder(new StringBuilder(512)).toString();
	}

	/**
	 * Set all fields of the cell to zero.
	 */
	public void clear() {
		MnLinalg.assign(s, 0);
		MnLinalg.assign(h, 0);
		MnLinalg.assign(z, 0);

		MnLinalg.assign(wf, 0);
		MnLinalg.assign(bf, 0);
		MnLinalg.assign(wm, 0);
		MnLinalg.assign(bm, 0);
		MnLinalg.assign(wp, 0);
		MnLinalg.assign(bp, 0);
		MnLinalg.assign(wo, 0);
		MnLinalg.assign(bo, 0);

		MnLinalg.assign(f, 0);
		MnLinalg.assign(m, 0);
		MnLinalg.assign(p, 0);
		MnLinalg.assign(o, 0);
		MnLinalg.assign(r, 0);
		MnLinalg.assign(a, 0);
		MnLinalg.assign(q, 0);

		MnLinalg.assign(s_, 0);
		MnLinalg.assign(h_, 0);
		MnLinalg.assign(ŷ, 0);
	}

	/**
	 * Add all values of the fields of another cell to the fields in this cell.
	 */
	public void add(LstmCell other) {
		MnLinalg.add(s, other.s, s);
		MnLinalg.add(h, other.h, h);
		MnLinalg.add(z, other.z, z);

		MnLinalg.add(wf, other.wf, wf);
		MnLinalg.add(bf, other.bf, bf);
		MnLinalg.add(wm, other.wm, wm);
		MnLinalg.add(bm, other.bm, bm);
		MnLinalg.add(wp, other.wp, wp);
		MnLinalg.add(bp, other.bp, bp);
		MnLinalg.add(wo, other.wo, wo);
		MnLinalg.add(bo, other.bo, bo);

		MnLinalg.add(f, other.f, f);
		MnLinalg.add(m, other.m, m);
		MnLinalg.add(p, other.p, p);
		MnLinalg.add(o, other.o, o);
		MnLinalg.add(r, other.r, r);
		MnLinalg.add(a, other.a, a);
		MnLinalg.add(q, other.q, q);

		MnLinalg.add(s_, other.s_, s_);
		MnLinalg.add(h_, other.h_, h_);
		MnLinalg.add(ŷ, other.ŷ, ŷ);
	}

	/**
	 * Element-wise multiply all values of the fields of another cell to the
	 * fields in this cell.
	 */
	public void mul(LstmCell other) {
		MnLinalg.mul(s, other.s, s);
		MnLinalg.mul(h, other.h, h);
		MnLinalg.mul(z, other.z, z);

		MnLinalg.mulElwise(wf, other.wf, wf);
		MnLinalg.mul(bf, other.bf, bf);
		MnLinalg.mulElwise(wm, other.wm, wm);
		MnLinalg.mul(bm, other.bm, bm);
		MnLinalg.mulElwise(wp, other.wp, wp);
		MnLinalg.mul(bp, other.bp, bp);
		MnLinalg.mulElwise(wo, other.wo, wo);
		MnLinalg.mul(bo, other.bo, bo);

		MnLinalg.mul(f, other.f, f);
		MnLinalg.mul(m, other.m, m);
		MnLinalg.mul(p, other.p, p);
		MnLinalg.mul(o, other.o, o);
		MnLinalg.mul(r, other.r, r);
		MnLinalg.mul(a, other.a, a);
		MnLinalg.mul(q, other.q, q);

		MnLinalg.mul(s_, other.s_, s_);
		MnLinalg.mul(h_, other.h_, h_);
		MnLinalg.mul(ŷ, other.ŷ, ŷ);
	}

	/**
	 * Element-wise divide all values of the fields of this cell by the fields
	 * in another cell.
	 */
	public void div(LstmCell other) {
		MnLinalg.div(s, other.s, s);
		MnLinalg.div(h, other.h, h);
		MnLinalg.div(z, other.z, z);

		MnLinalg.div(wf, other.wf, wf);
		MnLinalg.div(bf, other.bf, bf);
		MnLinalg.div(wm, other.wm, wm);
		MnLinalg.div(bm, other.bm, bm);
		MnLinalg.div(wp, other.wp, wp);
		MnLinalg.div(bp, other.bp, bp);
		MnLinalg.div(wo, other.wo, wo);
		MnLinalg.div(bo, other.bo, bo);

		MnLinalg.div(f, other.f, f);
		MnLinalg.div(m, other.m, m);
		MnLinalg.div(p, other.p, p);
		MnLinalg.div(o, other.o, o);
		MnLinalg.div(r, other.r, r);
		MnLinalg.div(a, other.a, a);
		MnLinalg.div(q, other.q, q);

		MnLinalg.div(s_, other.s_, s_);
		MnLinalg.div(h_, other.h_, h_);
		MnLinalg.div(ŷ, other.ŷ, ŷ);
	}

	/**
	 * Take element-wise reciprocal of all values of the fields of this cell.
	 */
	public void reciprocalize() {
		MnFuncs.reciprocal(s, s);
		MnFuncs.reciprocal(h, h);
		MnFuncs.reciprocal(z, z);

		MnFuncs.reciprocal(wf, wf);
		MnFuncs.reciprocal(bf, bf);
		MnFuncs.reciprocal(wm, wm);
		MnFuncs.reciprocal(bm, bm);
		MnFuncs.reciprocal(wp, wp);
		MnFuncs.reciprocal(bp, bp);
		MnFuncs.reciprocal(wo, wo);
		MnFuncs.reciprocal(bo, bo);

		MnFuncs.reciprocal(f, f);
		MnFuncs.reciprocal(m, m);
		MnFuncs.reciprocal(p, p);
		MnFuncs.reciprocal(o, o);
		MnFuncs.reciprocal(r, r);
		MnFuncs.reciprocal(a, a);
		MnFuncs.reciprocal(q, q);

		MnFuncs.reciprocal(s_, s_);
		MnFuncs.reciprocal(h_, h_);
		MnFuncs.reciprocal(ŷ, ŷ);
	}

	/**
	 * Multiplies the fields in place with a factor.
	 */
	public void mul(float factor) {
		MnLinalg.mul(s, factor, s);
		MnLinalg.mul(h, factor, h);
		MnLinalg.mul(z, factor, z);

		MnLinalg.mul(wf, factor, wf);
		MnLinalg.mul(bf, factor, bf);
		MnLinalg.mul(wm, factor, wm);
		MnLinalg.mul(bm, factor, bm);
		MnLinalg.mul(wp, factor, wp);
		MnLinalg.mul(bp, factor, bp);
		MnLinalg.mul(wo, factor, wo);
		MnLinalg.mul(bo, factor, bo);

		MnLinalg.mul(f, factor, f);
		MnLinalg.mul(m, factor, m);
		MnLinalg.mul(p, factor, p);
		MnLinalg.mul(o, factor, o);
		MnLinalg.mul(r, factor, r);
		MnLinalg.mul(a, factor, a);
		MnLinalg.mul(q, factor, q);

		MnLinalg.mul(s_, factor, s_);
		MnLinalg.mul(h_, factor, h_);
		MnLinalg.mul(ŷ, factor, ŷ);
	}

	/**
	 * Returns the maximum absolute value in any of the fields in this instance.
	 * Good for debugging.
	 */
	public float maxAbs() {
		float result = Float.NEGATIVE_INFINITY;

		result = Math.max(result, MnFuncs.maxAbs(s));
		result = Math.max(result, MnFuncs.maxAbs(h));
		result = Math.max(result, MnFuncs.maxAbs(z));

		result = Math.max(result, MnFuncs.maxAbs(wf));
		result = Math.max(result, MnFuncs.maxAbs(bf));
		result = Math.max(result, MnFuncs.maxAbs(wm));
		result = Math.max(result, MnFuncs.maxAbs(bm));
		result = Math.max(result, MnFuncs.maxAbs(wp));
		result = Math.max(result, MnFuncs.maxAbs(bp));
		result = Math.max(result, MnFuncs.maxAbs(wo));
		result = Math.max(result, MnFuncs.maxAbs(bo));

		result = Math.max(result, MnFuncs.maxAbs(f));
		result = Math.max(result, MnFuncs.maxAbs(m));
		result = Math.max(result, MnFuncs.maxAbs(p));
		result = Math.max(result, MnFuncs.maxAbs(o));
		result = Math.max(result, MnFuncs.maxAbs(r));
		result = Math.max(result, MnFuncs.maxAbs(a));
		result = Math.max(result, MnFuncs.maxAbs(q));

		result = Math.max(result, MnFuncs.maxAbs(s_));
		result = Math.max(result, MnFuncs.maxAbs(h_));
		result = Math.max(result, MnFuncs.maxAbs(ŷ));

		return result;
	}

	/**
	 * Creates a new LstmCell with the same dimensions as this cell and zero
	 * values.
	 */
	public LstmCell createLike() {
		LstmCell result = new LstmCell(inputSize, stateSize, outputSize);
		return result;
	}
}

package de.a0h.randomclairvoyance.gui;

public abstract class ColorScheme<T> {

	public static final int STANDARD_COLOR_COUNT = 512;

	/**
	 * For different environments, different kinds of color objects are needed.
	 * For example the AWT uses java.awt.Color and Android uses
	 * android.graphics.Color. Thus we cannot hard code the kind of color
	 * object. Instead, a color producer needs to be created, which translates
	 * (r,g,b-tuples into the desired kind of color object.
	 */
	public static interface ColorProducer<T> {
		public T produce(Rgb rgb);
	}

	/**
	 * Instances of this class represent a color in red (r), green (g) and
	 * blue(b) components.
	 */
	public static class Rgb {
		public int r;
		public int g;
		public int b;

		public Rgb(int r, int g, int b) {
			this.r = r;
			this.g = g;
			this.b = b;
		}

		public String toString() {
			return getClass().getSimpleName() + "[" + r + ", " + g + ", " + b + "]";
		}
	}

	protected T COLOR_ZERO_POSITIVE;
	protected T COLOR_ZERO_NEGATIVE;
	protected T COLOR_INFINITY_POSITIVE;
	protected T COLOR_INFINITY_NEGATIVE;
	protected T COLOR_NOT_A_NUMBER;

	protected int IDX_CONTINUOUSLY_NEGATIVE_START = STANDARD_COLOR_COUNT / 2 - 1;
	protected int IDX_CONTINUOUSLY_POSITIVE_START = STANDARD_COLOR_COUNT / 2;

	protected Object[] colorAr = new Object[STANDARD_COLOR_COUNT];

	public ColorScheme(ColorProducer<T> colorProducer) {
		Rgb rgb;
		for (int i = 0; i < STANDARD_COLOR_COUNT / 2; i++) {
			float f = i / 255.0f + .000001f;

			rgb = createIntervalColor(f);
			colorAr[256 + i] = colorProducer.produce(rgb);

			rgb = createIntervalColor(-f);
			colorAr[255 - i] = colorProducer.produce(rgb);
		}

		COLOR_ZERO_POSITIVE = colorProducer.produce(createPositiveZeroColor());
		COLOR_ZERO_NEGATIVE = colorProducer.produce(createNegativeZeroColor());
		COLOR_INFINITY_POSITIVE = colorProducer.produce(createPositiveInfinityColor());
		COLOR_INFINITY_NEGATIVE = colorProducer.produce(createNegativeInfinityColor());
		COLOR_NOT_A_NUMBER = colorProducer.produce(createNotANumberColor());
	}

	public T getColor(float f) {
		if (Float.isNaN(f)) {
			return COLOR_NOT_A_NUMBER;
		}

		boolean sign = isSignBitSet(f);
		if (f == 0.0f) { // this is true for +0.0 and for -0.0
			return sign ? COLOR_ZERO_NEGATIVE : COLOR_ZERO_POSITIVE;
		}
		if (Float.isInfinite(f)) {
			return sign ? COLOR_INFINITY_NEGATIVE : COLOR_INFINITY_POSITIVE;
		}

		f = Math.abs(f);

		int idx;
		if (sign) {
			idx = IDX_CONTINUOUSLY_NEGATIVE_START - (int) (f * 255.0f);
		} else {
			idx = IDX_CONTINUOUSLY_POSITIVE_START + (int) (f * 255.0f);
		}

		idx = Math.max(idx, 0);
		idx = Math.min(idx, 511);

		@SuppressWarnings("unchecked")
		T color = (T) colorAr[idx];

		return color;
	}

	/**
	 * Returns true if, and only if, the sign bit of the specified float value
	 * is set. This method works correctly for all finite numbers as well as for
	 * +/-0.0 and for +/- infinity.
	 */
	public static boolean isSignBitSet(float f) {
		return (Float.floatToRawIntBits(f) & 0x80000000) != 0;
	}

	/**
	 * Subclasses need to implement this method in a way so it returns sensible
	 * colors for values of f in [-1.000001, 1.000001]. Note that subclasses can
	 * differentiate the colors for +0.0 and -0.0 by overwriting the appropriate
	 * special value methods.
	 */
	public abstract Rgb createIntervalColor(float f);

	/**
	 * Subclasses can overwrite this method to supply a color for values which
	 * do not represent a number, e.g. a value of the result after division by
	 * zero. This implementation returns a brown or dark orange.
	 */
	public Rgb createNotANumberColor() {
		return new Rgb(156, 68, 0);
	}

	/**
	 * Subclasses can overwrite this method to supply a color for the special
	 * positive infinity value. This implementation returns a light purple.
	 */
	public Rgb createPositiveInfinityColor() {
		return new Rgb(187, 0, 102);
	}

	/**
	 * Subclasses can overwrite this method to supply a color for the special
	 * positive infinity value. This implementation returns a dark purple.
	 */
	public Rgb createNegativeInfinityColor() {
		return new Rgb(85, 0, 102);
	}

	/**
	 * This method can to be overwritten to supply a special color for +0.0f.
	 * This implementation returns a light green.
	 */
	public Rgb createPositiveZeroColor() {
		return createIntervalColor(+0.0f);
	}

	/**
	 * This method can to be overwritten to supply a special color for -0.0f.
	 * This implementation returns a dark green.
	 */
	public Rgb createNegativeZeroColor() {
		return createIntervalColor(-0.0f);
	}
}
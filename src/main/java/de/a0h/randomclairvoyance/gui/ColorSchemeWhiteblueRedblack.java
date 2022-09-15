package de.a0h.randomclairvoyance.gui;

public class ColorSchemeWhiteblueRedblack<T> extends ColorScheme<T> {

	public ColorSchemeWhiteblueRedblack(ColorProducer<T> colorProducer) {
		super(colorProducer);
	}

	@Override
	public Rgb createPositiveZeroColor() {
		return new Rgb(64, 210, 64);
	}

	@Override
	public Rgb createNegativeZeroColor() {
		return new Rgb(16, 80, 16);
	}

	@Override
	public Rgb createIntervalColor(float f) {
		float preparedF = (float) Math.sqrt(Math.abs(f));

		int i = (int) (preparedF * 255.0f);
		if (f >= 0.0f) {
			return new Rgb(255 - i, 255 - i, 255);
		} else {
			return new Rgb(255 - i, 0, 0);
		}
	}
}

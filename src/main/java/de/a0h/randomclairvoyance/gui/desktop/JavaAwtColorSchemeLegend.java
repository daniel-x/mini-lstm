package de.a0h.randomclairvoyance.gui.desktop;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import de.a0h.randomclairvoyance.gui.ColorScheme;

@SuppressWarnings("serial")
public class JavaAwtColorSchemeLegend extends Canvas {

	ColorScheme<Color> colorScheme;

	Dimension minimumSize = new Dimension(40, 40);

	Dimension preferredSize = new Dimension(40, ColorScheme.STANDARD_COLOR_COUNT);

	protected static final float[] SPECIAL_VALUES = { //
			+0.0f, //
			-0.0f, //
			Float.POSITIVE_INFINITY, //
			Float.NEGATIVE_INFINITY, //
			Float.NaN //
	};

	public JavaAwtColorSchemeLegend(ColorScheme<Color> colorScheme) {
		this.colorScheme = colorScheme;
	}

	@Override
	public Dimension getMinimumSize() {
		return minimumSize;
	}

	@Override
	public Dimension getPreferredSize() {
		return preferredSize;
	}

	@Override
	public void paint(Graphics g) {
		int w = getWidth();
		int h = getHeight();
		int stdColCount = ColorScheme.STANDARD_COLOR_COUNT;
		int hPerColor = (h + stdColCount - 1) / stdColCount;
		int stdColCountHm1 = stdColCount / 2 - 1;

		for (int i = 0; i < stdColCount; i++) {
			float f = ((float) (stdColCountHm1 - i)) / stdColCountHm1;
			f += (f == 0) ? 0.00000001f : 0.0f;

			Color color = colorScheme.getColor(f);
			g.setColor(color);

			g.fillRect(0, i * h / stdColCount, w, hPerColor);
		}

		int specialW = Math.min(w / 2, 30);
		int specialH = Math.min(h / 5, 30);
		for (int i = 0; i < SPECIAL_VALUES.length; i++) {
			g.setColor(colorScheme.getColor(SPECIAL_VALUES[i]));
			g.fillRect(w - specialW, specialH * i, specialW, specialH);
		}
	}
}

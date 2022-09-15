package de.a0h.randomclairvoyance.gui.desktop;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;

import de.a0h.mininum.format.MnFormat;
import de.a0h.randomclairvoyance.gui.ColorScheme;
import de.a0h.randomclairvoyance.gui.ColorScheme.ColorProducer;
import de.a0h.randomclairvoyance.gui.ColorScheme.Rgb;
import de.a0h.randomclairvoyance.gui.ColorSchemeWhiteblueRedblack;
import de.a0h.randomclairvoyance.lstm.LstmCell;

public class LstmCellDrawer {

	public static final int MATRIX_ELEMENT_BLOCK_SIZE = 32;

	public static final int MATRIX_ELEMENT_BLOCK_SIZE_HALF = MATRIX_ELEMENT_BLOCK_SIZE / 2;

	public ColorScheme<Color> colorScheme = new ColorSchemeWhiteblueRedblack<Color>( //
			new ColorProducer<Color>() {
				@Override
				public Color produce(Rgb rgb) {
					return new Color(rgb.r, rgb.g, rgb.b);
				}
			} //
	);

	float colorMaxAbsValue = 20;

	int internalSpacing = 10;

	int externalSpacing = 2 * internalSpacing;

	Font font = new Font(Font.SANS_SERIF, Font.PLAIN, (MATRIX_ELEMENT_BLOCK_SIZE * 13 + 40 - 1) / 40);

	public int getWidth(LstmCell cell) {
		int compoundSize = cell.stateSize + cell.inputSize;

		return (compoundSize + 1) * MATRIX_ELEMENT_BLOCK_SIZE + internalSpacing + externalSpacing;
	}

	public int getHeight(LstmCell cell) {
		return (cell.stateSize * 4 + 2) * MATRIX_ELEMENT_BLOCK_SIZE + 5 * internalSpacing + externalSpacing;
	}

	public void draw(LstmCell cell, int x, int y, Graphics g) {
		g.setFont(font);

		drawHori(cell.s, x, y, g);
		y += MATRIX_ELEMENT_BLOCK_SIZE + internalSpacing;
		drawHori(cell.h, x, y, g);
		y += MATRIX_ELEMENT_BLOCK_SIZE + internalSpacing;

		int compoundSize = cell.stateSize + cell.inputSize;
		int matW = compoundSize * MATRIX_ELEMENT_BLOCK_SIZE + internalSpacing;
		int matH = cell.stateSize * MATRIX_ELEMENT_BLOCK_SIZE + internalSpacing;

		draw(cell.wf, x, y, g);
		drawVert(cell.bf, x + matW, y, g);
		y += matH;

		draw(cell.wm, x, y, g);
		drawVert(cell.bm, x + matW, y, g);
		y += matH;

		draw(cell.wp, x, y, g);
		drawVert(cell.bp, x + matW, y, g);
		y += matH;

		draw(cell.wo, x, y, g);
		drawVert(cell.bo, x + matW, y, g);
		y += matH;
	}

	private void drawVert(float[] a, int x, int y, Graphics g) {
		for (int i = 0; i < a.length; i++) {
			drawElementBlock(a[i], x, y, g);
			y += MATRIX_ELEMENT_BLOCK_SIZE;
		}
	}

	private void drawHori(float[] a, int x, int y, Graphics g) {
		for (int i = 0; i < a.length; i++) {
			drawElementBlock(a[i], x, y, g);
			x += MATRIX_ELEMENT_BLOCK_SIZE;
		}
	}

	public void draw(float[][] a, int x, int y, Graphics g) {
		for (int i = 0; i < a.length; i++) {
			drawHori(a[i], x, y, g);
			y += MATRIX_ELEMENT_BLOCK_SIZE;
		}
	}

	protected String toShortString(float f) {
		String fStr;

		boolean sign = ColorScheme.isSignBitSet(f);

		if (f == 0.0f) {
			fStr = sign ? "  -0" : "   0";
		} else if (sign && f > -0.01f) {
			fStr = "-0.0..";
		} else if (!sign && f < 0.01f) {
			fStr = "0.0..";
		} else if (f < -10f) {
			fStr = "<-10";
		} else if (f > 10f) {
			fStr = ">10";
		} else if (f == Float.POSITIVE_INFINITY) {
			fStr = "  +∞";
		} else if (f == Float.NEGATIVE_INFINITY) {
			fStr = "  -∞";
		} else {
			fStr = MnFormat.FORMAT_2DECIMALS.get().format(f);
		}

		return fStr;
	}

	protected void drawElementBlock(float f, int x, int y, Graphics graphics) {
		Color bg = getColor(f);
		graphics.setColor(bg);
		graphics.fillRect(x, y, MATRIX_ELEMENT_BLOCK_SIZE, MATRIX_ELEMENT_BLOCK_SIZE);

		String fStr = toShortString(f);
		float bgBrightness = getBrightness(bg);
		graphics.setColor(bgBrightness < 0.55 ? Color.WHITE : Color.BLACK);
		graphics.drawString(fStr, x + 1, y + (MATRIX_ELEMENT_BLOCK_SIZE + font.getSize()) / 2 - 2);
	}

	protected float getBrightness(Color color) {
		int rgb = color.getRGB();

		int r = (rgb >> 16) & 0xff;
		int g = (rgb >> 8) & 0xff;
		int b = (rgb >> 0) & 0xff;

		float bgBrightness = (0.299f * r + 0.587f * g + 0.114f * b) / 255;
		return bgBrightness;
	}

	protected Color getColor(float f) {
		Color color = colorScheme.getColor(f / colorMaxAbsValue);
		return color;
	}
}

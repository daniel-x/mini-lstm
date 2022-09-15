package de.a0h.mininum;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Test;

public class MiniNumTest {

	@Test
	public void testOuterProduct3x3() {
		float[] u = { 5, 6, 7 };
		float[] v = { 2.5f, 2, 3 };

		float[][] aActual = new float[3][3];
		MnLinalg.outerProduct(u, v, aActual);

		// ...v 2.5 . 2. 3
		// u..............
		// 5 .. 12.5 10 15
		// 6 .. 15 . 12 18
		// 7 .. 17.5 14 21
		float[][] aExpected = { //
				{ 12.5f, 10, 15 }, //
				{ 15, 12, 18 }, //
				{ 17.5f, 14, 21 }, //
		};

		assertEqualWithDetailMessage(aExpected, aActual);
	}

	@Test
	public void testOuterProduct4x4() {
		float[] u = { 5, 6, 7, 8 };
		float[] v = { 2.5f, 2, 3, 4 };

		float[][] aActual = new float[4][4];
		MnLinalg.outerProduct(u, v, aActual);

		// ...v 2.5 . 2. 3. 4
		// u.................
		// 5 .. 12.5 10 15 20
		// 6 .. 15 . 12 18 24
		// 7 .. 17.5 14 21 28
		// 8 .. 20 . 16 24 32
		float[][] aExpected = { //
				{ 12.5f, 10, 15, 20 }, //
				{ 15, 12, 18, 24 }, //
				{ 17.5f, 14, 21, 28 }, //
				{ 20, 16, 24, 32 }, //
		};

		assertEqualWithDetailMessage(aExpected, aActual);
	}

	@Test
	public void testOuterProduct2x4() {
		float[] u = { 5, 6 };
		float[] v = { 2.5f, 2, 3, 4 };

		float[][] aActual = new float[2][4];
		MnLinalg.outerProduct(u, v, aActual);

		// ...v 2.5 . 2. 3. 4
		// u.................
		// 5 .. 12.5 10 15 20
		// 6 .. 15 . 12 18 24
		float[][] aExpected = { //
				{ 12.5f, 10, 15, 20 }, //
				{ 15, 12, 18, 24 }, //
		};

		assertEqualWithDetailMessage(aExpected, aActual);
	}

	@Test
	public void testMulMatVec2x4() {
		float[][] a = { //
				{ 5, 6, 7, 8 }, //
				{ 1, 2, 3, 4 }, //
		};
		float[] x = { 5, 4, 3, 2 };

		float[] yActual = new float[2];
		MnLinalg.mulMatVec(a, x, yActual);

		float[] yExpected = { 86, 30 };

		assertEqualWithDetailMessage(yExpected, yActual);
	}

	@Test
	public void testMulMatVec3x4() {
		float[][] a = { //
				{ 5, 6, 7, 8 }, //
				{ 1, 2, 3, 4 }, //
				{ .1f, .2f, .3f, .7f }, //
		};
		float[] x = { 5, 4, 3, 2 };

		float[] yActual = new float[3];
		MnLinalg.mulMatVec(a, x, yActual);

		float[] yExpected = { 86, 30, 3.6f };

		assertEqualWithDetailMessage(yExpected, yActual);
	}

	@Test
	public void testMulVecMat2x4() {
		float[] x = { 1, 2 };
		float[][] a = { //
				{ 5, 6, 7, 8 }, //
				{ 1, 2, 3, 4 }, //
		};

		float[] yActual = new float[4];
		MnLinalg.mulVecMat(x, a, yActual);

		float[] yExpected = { 7, 10, 13, 16 };

		assertEqualWithDetailMessage(yExpected, yActual);
	}

	@Test
	public void testMulVecMat3x4() {
		float[] x = { 1, 2, 3 };
		float[][] a = { //
				{ 5, 6, 7, 8 }, //
				{ 1, 2, 3, 4 }, //
				{ .1f, .2f, .3f, .4f }, //
		};

		float[] yActual = new float[4];
		MnLinalg.mulVecMat(x, a, yActual);

		float[] yExpected = { 7.3f, 10.6f, 13.9f, 17.2f };

		assertEqualWithDetailMessage(yExpected, yActual);
	}

	protected void assertEqualWithDetailMessage(float[] expected, float[] actual) throws AssertionError {
		if (!Arrays.equals(expected, actual)) {
			throw new AssertionError("unexpected result. " //
					+ "expected: " + Arrays.toString(expected) + " " //
					+ "actual: " + Arrays.toString(actual));
		}
	}

	protected void assertEqualWithDetailMessage(float[][] expected, float[][] actual) throws AssertionError {
		assertEquals("matrices differ in size, first dimension", expected.length, actual.length);

		for (int i = 0; i < expected.length; i++) {
			if (!Arrays.equals(expected[i], actual[i])) {
				throw new AssertionError("matrices differ in row " + i + ". "//
						+ "expected: " + Arrays.toString(expected[i]) + " " //
						+ "actual: " + Arrays.toString(actual[i]));
			}
		}
	}
}

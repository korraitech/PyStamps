import {test, expect} from 'bun:test'
import {createSmoother} from './runner.mjs'

const edges = new Uint32Array([0, 1, 0, 2, 1, 2, 2, 3])

test('real Handwired matches native reductions and preserves FP64 focal values', async () => {
  const smooth = await createSmoother(edges, 4)
  const k = new Float64Array([-1000.37, 0.30000000001, 300.11, -22.13])
  const c = new Float64Array([1.5, 1.5, -0.00000001, -1.5])
  const expected = smooth.native(k, c)
  const actual = smooth.network(k, c)
  expect(Array.from(actual)).toEqual(Array.from(expected))
  expect(actual[1]).toBe(k[1])
  expect(actual[1]).not.toBe(Math.fround(k[1]))
})

test('network rejects nonfinite or out-of-range values', async () => {
  const smooth = await createSmoother(edges, 4)
  for (const value of [NaN, Infinity, -Infinity, 20000]) {
    expect(() => smooth.network(new Float64Array([0, value, 1, 2]), new Float64Array(4))).toThrow()
  }
})

test('isolated vertices are explicitly outside the circuit contract', async () => {
  await expect(createSmoother(edges, 5)).rejects.toThrow('isolated')
})

test('varying degrees and magnitude extremes match the native loop', async () => {
  // A star exercises degree 1 and a much larger degree than the mesh benchmark.
  const star = new Uint32Array(Array.from({length: 11}, (_, i) => [0, i + 1]).flat())
  const smooth = await createSmoother(star, 12)
  const k = new Float64Array([0.3, 16384, -16384, 2 ** -149, -(2 ** -149), 1e-12, -1e-12, 1000.37, 0, 0.3, -0.3, 1])
  const c = Float64Array.from(k, value => -value)
  expect(Array.from(smooth.network(k, c))).toEqual(Array.from(smooth.native(k, c)))
})

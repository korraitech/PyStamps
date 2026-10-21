// Run with Bun: imports the real upstream TypeScript runtime, without vendoring it.
import {readFileSync, writeFileSync} from 'node:fs'
import {resolve} from 'node:path'
import {pathToFileURL} from 'node:url'

export async function createSmoother(edges, n) {
  const upstream = process.env.HANDWIRED_DIR
  if (!upstream) throw new Error('Set HANDWIRED_DIR to a Handwired checkout')
  const {Circuit, emitMinimumExact, emitMaximumExact} = await import(
    pathToFileURL(resolve(upstream, 'src/runtime/numeric-ui-core.ts')).href
  )
  const start = performance.now()
  const neighbours = Array.from({length: n}, () => [])
  for (let e = 0; e < edges.length; e += 2) {
    neighbours[edges[e]].push(edges[e + 1])
    neighbours[edges[e + 1]].push(edges[e])
  }
  if (neighbours.some(row => row.length === 0)) throw new Error('isolated vertex unsupported')
  const adjacencyMs = performance.now() - start
  const buildStart = performance.now()
  const circuits = new Map()
  for (const degree of new Set(neighbours.map(row => row.length))) {
    const circuit = new Circuit(degree)
    const inputs = Array.from({length: degree}, (_, i) => circuit.source(i))
    for (const wire of inputs) circuit.assumeRange(wire, -16384, 16384)
    let lower = inputs[0]
    let upper = inputs[0]
    for (let j = 1; j < degree; j++) {
      lower = emitMinimumExact(circuit, lower, inputs[j])
      upper = emitMaximumExact(circuit, upper, inputs[j])
    }
    const program = circuit.compileTensorProgram([lower, upper])
    const evaluate = circuit.createTensorProgramEvaluator(program)
    circuits.set(degree, {
      evaluate, source: new Float32Array(degree),
      operations: program.valueCount - program.sourceSize,
      layers: program.layers.length,
    })
  }
  const setup = {
    adjacency_ms: adjacencyMs,
    build_compile_ms: performance.now() - buildStart,
    circuits: [...circuits].map(([degree, item]) => ({degree, operations: item.operations, layers: item.layers})),
  }

  function network(k, c) {
    // FP32 neighbour bounds, FP64 original values and comparisons, just as in PyStamps.
    const output = new Float64Array(2 * n)
    for (const [channel, values] of [k, c].entries()) {
      if (values.length !== n) throw new Error('wrong input length')
      for (const value of values) {
        if (!Number.isFinite(value) || Math.abs(value) > 16384) throw new Error('finite values within ±16384 required')
      }
      for (let i = 0; i < n; i++) {
        const row = neighbours[i]
        const {evaluate, source} = circuits.get(row.length)
        for (let j = 0; j < row.length; j++) source[j] = values[row[j]]
        const [lower, upper] = evaluate(source)
        const value = values[i]
        let result = value
        if (value > upper) result = upper
        if (value < lower) result = lower
        output[channel * n + i] = result
      }
    }
    return output
  }

  function native(k, c) {
    const output = new Float64Array(2 * n)
    for (const [channel, values] of [k, c].entries()) {
      const lower = new Float32Array(n).fill(Infinity)
      const upper = new Float32Array(n).fill(-Infinity)
      for (let e = 0; e < edges.length; e += 2) {
        const a = edges[e], b = edges[e + 1]
        lower[a] = Math.min(lower[a], values[b])
        lower[b] = Math.min(lower[b], values[a])
        upper[a] = Math.max(upper[a], values[b])
        upper[b] = Math.max(upper[b], values[a])
      }
      for (let i = 0; i < n; i++) {
        const value = values[i]
        let result = value
        if (value > upper[i]) result = upper[i]
        if (value < lower[i]) result = lower[i]
        output[channel * n + i] = result
      }
    }
    return output
  }
  return {network, native, setup}
}

function timed(fn, repeats, warmup) {
  for (let i = 0; i < warmup; i++) fn()
  const samples = []
  let output
  for (let i = 0; i < repeats; i++) {
    const start = performance.now()
    output = fn()
    samples.push(performance.now() - start)
  }
  const sorted = samples.toSorted((a, b) => a - b)
  const median = (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2
  return {output, timing: {median_ms: median, samples_ms: samples}}
}

if (import.meta.main) {
  const [directory, repeatsText = '7', warmupText = '3'] = process.argv.slice(2)
  const metadata = JSON.parse(readFileSync(resolve(directory, 'input.json'), 'utf8'))
  const edges = new Uint32Array(Uint8Array.from(readFileSync(resolve(directory, 'edges.bin'))).buffer)
  const values = new Float64Array(Uint8Array.from(readFileSync(resolve(directory, 'values.bin'))).buffer)
  const k = values.subarray(0, metadata.n), c = values.subarray(metadata.n)
  const smooth = await createSmoother(edges, metadata.n)
  const network = timed(() => smooth.network(k, c), Number(repeatsText), Number(warmupText))
  writeFileSync(resolve(directory, 'network.bin'), new Uint8Array(network.output.buffer))
  // The single-shot mode excludes the control benchmark from process-round-trip timing.
  const native = Number(repeatsText) > 1
    ? timed(() => smooth.native(k, c), Number(repeatsText), Number(warmupText)) : null
  if (native) writeFileSync(resolve(directory, 'native.bin'), new Uint8Array(native.output.buffer))
  console.log(JSON.stringify({setup: smooth.setup, network: network.timing, native: native?.timing, bun: Bun.version}))
}

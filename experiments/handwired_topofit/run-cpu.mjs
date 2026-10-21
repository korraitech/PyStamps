import {readFileSync, writeFileSync} from 'node:fs'
import {resolve} from 'node:path'
import {loadRuntime} from './runtime.mjs'
import {buildNetwork, winners, summary} from './network.mjs'

const [dir, repeatsText = '3'] = process.argv.slice(2)
const metadata = JSON.parse(readFileSync(resolve(dir, 'input.json'), 'utf8'))
const read = (name, type) => new type(Uint8Array.from(readFileSync(resolve(dir, name))).buffer)
const weights = read('weights.bin', Float32Array)
const input = read('input.bin', Float32Array)
const denominators = read('denominators.bin', Float64Array)
const runtime = await loadRuntime()
const network = buildNetwork(runtime, weights, metadata.inputs, metadata.outputs)
const result = {setup: network.setup, bun: Bun.version, methods: {}}
for (const name of ['interpreted', 'generated']) {
  // Warm on a bounded slice, then one complete call. Disclose warm cost separately.
  const warmStart = performance.now()
  network[name](input, Math.min(metadata.count, 1024))
  network[name](input, metadata.count)
  const warmupMs = performance.now() - warmStart
  const samples = [], searchSamples = []
  let output, best
  for (let r=0; r<Number(repeatsText); r++) {
    const start = performance.now()
    output = network[name](input, metadata.count)
    const sumsDone = performance.now()
    best = winners(output, metadata.count, metadata.outputs/2, denominators)
    samples.push(sumsDone-start)
    searchSamples.push(performance.now()-start)
  }
  writeFileSync(resolve(dir, name+'.bin'), new Uint8Array(output.buffer))
  writeFileSync(resolve(dir, name+'-best.bin'), new Uint8Array(best.buffer))
  result.methods[name] = {correlation:summary(samples), search:summary(searchSamples), warmup_ms:warmupMs}
  if(name==='interpreted') {
    const reference=new Float32Array(metadata.count*2)
    const trials=metadata.outputs/2
    for(let i=0;i<metadata.count;i++) {
      let high=-Infinity,index=0
      for(let j=0;j<trials;j++) {
        const re=output[i*metadata.outputs+j],im=output[i*metadata.outputs+trials+j]
        const score=Math.fround(Math.fround(re*re)+Math.fround(im*im))
        if(score>high){high=score;index=j}
      }
      reference[i*2]=index;reference[i*2+1]=high
      if(i<128) {
        const actual=network.fused.evaluate(input.subarray(i*metadata.inputs,(i+1)*metadata.inputs))
        if(actual[0]!==index || actual[1]!==high) throw new Error('fused network differs from FP32 squared-score reference')
      }
    }
    writeFileSync(resolve(dir,'fused-reference.bin'),new Uint8Array(reference.buffer))
  }
}
for (const name of ['strict', 'relaxed']) {
  writeFileSync(resolve(dir, name+'-module.json'), JSON.stringify(network.modules[name]))
  writeFileSync(resolve(dir, name+'-shared.bin'), new Uint8Array(network.shared[name].buffer))
}
writeFileSync(resolve(dir,'fused-module.json'),JSON.stringify(network.fused.module))
writeFileSync(resolve(dir,'fused-shared.bin'),new Uint8Array(network.fused.shared.buffer))
result.fused={operations:network.fused.operations,layers:network.fused.layers,cpu_reference_checked_rows:Math.min(128,metadata.count)}
writeFileSync(resolve(dir, 'cpu-handwired.json'), JSON.stringify(result, null, 2))
console.log(JSON.stringify(result))

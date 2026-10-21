// Runtime is supplied by the CLI or a bundled upstream module in the browser.
export function buildNetwork(runtime, weights, inputCount, outputCount) {
  if (weights.length !== inputCount * outputCount) throw new Error('incorrect weight dimensions')
  const start = performance.now()
  const circuit = new runtime.Circuit(inputCount)
  const sources = Array.from({length: inputCount}, (_, i) => circuit.source(i))
  sources.forEach(wire => circuit.assumeRange(wire, -1, 1))
  const roots = Array.from({length: outputCount}, (_, j) => {
    const terms = sources.map((wire, k) => [wire, weights[k * outputCount + j]])
    // Bound expression size for the JS JIT and WGSL compiler. This changes
    // FP32 reduction order; compare against FP64 rather than assume equality.
    const partials = []
    for (let start=0;start<terms.length;start+=24) partials.push(circuit.roundAffine(terms.slice(start,start+24)))
    return partials.length===1 ? partials[0] : circuit.roundAffine(partials.map(wire => [wire,1]))
  })
  const program = circuit.compileTensorProgram(roots)
  const evaluate = circuit.createTensorProgramEvaluator(program)
  const plan = runtime.planInstances(program, sources, roots)
  const instance = runtime.compileInstanceProgram(plan, {rowsPerFunction: 8, blockSize: 1024})
  instance.runShared(new Float32Array(inputCount))
  const modules = {
    strict: runtime.emitInstanceWgsl(plan, {guard: 'max'}),
    relaxed: runtime.emitInstanceWgsl(plan, {guard: 'none'}),
  }
  const shared = Object.fromEntries(Object.entries(modules).map(([name, module]) => [name, runtime.packSharedBuffer(module, instance)]))
  // Optional complete trial selector. FP32 squared scores can change close
  // decisions; its outputs must be checked against the FP64 source separately.
  const trials = outputCount/2
  let candidates = Array.from({length:trials},(_,j) => {
    const squaredReal = circuit.multiply(roots[j],roots[j])
    const squaredImag = circuit.multiply(roots[j+trials],roots[j+trials])
    const score = circuit.roundAffine([[squaredReal,1],[squaredImag,1]])
    circuit.assumeRange(score,0,2*(inputCount+1)**2)
    return {score,index:circuit.constant(j)}
  })
  while(candidates.length>1) {
    const next=[]
    for(let i=0;i<candidates.length;i+=2) {
      const a=candidates[i],b=candidates[i+1]
      if(!b) {next.push(a);continue}
      const rightWins=runtime.emitCompare(circuit,b.score,'>',a.score)
      next.push({score:circuit.select(rightWins,a.score,b.score),index:circuit.select(rightWins,a.index,b.index)})
    }
    candidates=next
  }
  const fusedRoots=[candidates[0].index,candidates[0].score]
  const fusedProgram=circuit.compileTensorProgram(fusedRoots)
  const fusedPlan=runtime.planInstances(fusedProgram,sources,fusedRoots)
  const fusedModule=runtime.emitInstanceWgsl(fusedPlan,{guard:'max'})
  const fusedSharedRunner=runtime.createBatchedExecutor(fusedPlan,1)
  fusedSharedRunner.runShared(new Float32Array(inputCount))
  const fusedShared=runtime.packSharedBuffer(fusedModule,fusedSharedRunner)
  return {
    setup: {build_compile_ms: performance.now() - start, operations: program.valueCount - program.sourceSize,
      layers: program.layers.length, weights: plan.counts.instanceWeights, generated_source_bytes: instance.sourceBytes, reduction_block:24},
    modules, shared,
    fused:{module:fusedModule,shared:fusedShared,evaluate:circuit.createTensorProgramEvaluator(fusedProgram),
      operations:fusedProgram.valueCount-fusedProgram.sourceSize,layers:fusedProgram.layers.length},
    interpreted(input, count) {
      const output = new Float32Array(count * outputCount)
      for (let i = 0; i < count; i++) output.set(evaluate(input.subarray(i * inputCount, (i + 1) * inputCount)), i * outputCount)
      return output
    },
    generated(input, count) {
      const output = new Float32Array(count * outputCount)
      instance.runInstances(input, count, output)
      return output
    },
  }
}

export function winners(sums, count, trials, denominators) {
  const best = new Uint32Array(count)
  for (let i = 0; i < count; i++) {
    let high = -Infinity
    for (let j = 0; j < trials; j++) {
      const score = Math.hypot(sums[i * trials * 2 + j], sums[i * trials * 2 + trials + j]) / (denominators?.[i] ?? 1)
      if (score > high) { high = score; best[i] = j }
    }
  }
  return best
}

export function summary(samples) {
  const sorted = samples.toSorted((a,b) => a-b)
  return {median_ms: (sorted[Math.floor((sorted.length - 1)/2)] + sorted[Math.floor(sorted.length/2)]) / 2, samples_ms: samples}
}

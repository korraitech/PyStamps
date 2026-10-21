import {createGpuInstanceRunner} from '/upstream.js'
import {createMatmulRunner} from './gpu-control.mjs'
import {summary, winners} from './network.mjs'

async function binary(path, Type=Float32Array) {
  const response = await fetch(path)
  if (!response.ok) throw new Error('fetch failed: '+path)
  return new Type(await response.arrayBuffer())
}

export async function smokeTest() {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('hardware GPU unavailable')
  const device = await adapter.requestDevice()
  const cases = []
  for (const [N,K,M] of [[3,5,7],[9,9,10],[1,120,66]]) {
    const input = Float32Array.from({length:N*K},(_,i)=>Math.sin(i)*0.5)
    const weights = Float32Array.from({length:K*M},(_,i)=>Math.cos(i)*0.5)
    const runner = await createMatmulRunner(device,weights,K,M,N)
    const output = new Float32Array(N*M)
    await runner.setInstanceSources(input,N)
    await runner.run(null,N,output)
    let error=0
    for(let n=0;n<N;n++) for(let j=0;j<M;j++) {
      let expected=0
      for(let k=0;k<K;k++) expected+=input[n*K+k]*weights[k*M+j]
      error=Math.max(error,Math.abs(expected-output[n*M+j]))
    }
    if(error>1e-4 || !output.every(Number.isFinite)) throw new Error('GPU control incorrect: '+error)
    runner.destroy()
    if(M%2===0) {
      const winnerRunner=await createMatmulRunner(device,weights,K,M,N,{winnerOnly:true})
      const selected=new Float32Array(N*2)
      await winnerRunner.setInstanceSources(input,N)
      await winnerRunner.run(null,N,selected)
      const expected=winners(output,N,M/2)
      for(let i=0;i<N;i++)if(selected[i*2]!==expected[i])throw new Error('GPU winner control incorrect')
      winnerRunner.destroy()
    }
    cases.push({N,K,M,max_abs_error:error})
  }
  device.destroy()
  return cases
}

export async function runCase(count, repeats=3) {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('hardware GPU unavailable')
  const device = await adapter.requestDevice()
  device.addEventListener('uncapturederror', event => console.error(event.error.message))
  const metadata = await (await fetch(`/${count}/input.json`)).json()
  const input = await binary(`/${count}/input.bin`)
  const weights = await binary(`/${count}/weights.bin`)
  const denominators = await binary(`/${count}/denominators.bin`,Float64Array)
  const capacity = Math.min(count,16384)
  const result = {adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,
    description:adapter.info.description,isFallbackAdapter:adapter.info.isFallbackAdapter},
    user_agent:navigator.userAgent, count, chunk_capacity:capacity, methods:{}}
  for (const name of ['strict','relaxed','matmul','fused','matmul-winner']) {
    let runner
    try {
      const setupStart = performance.now()
      const direct=name.startsWith('matmul')
      const winnerOnly=name==='fused'||name==='matmul-winner'
      const outputCount=winnerOnly?2:metadata.outputs
      const module = direct?null:await (await fetch(`/${count}/${name}-module.json`)).json()
      const shared = direct?null:await binary(`/${count}/${name}-shared.bin`)
      runner = direct
        ? await createMatmulRunner(device,weights,metadata.inputs,metadata.outputs,capacity,{winnerOnly})
        : await createGpuInstanceRunner(device,module,capacity)
      const setupMs = performance.now()-setupStart
      const output = new Float32Array(count*outputCount)
      const chunkOutput = new Float32Array(capacity*outputCount)
      const run = async () => {
        const start=performance.now()
        let upload=0,dispatchReadback=0
        for(let offset=0;offset<count;offset+=capacity) {
          const n=Math.min(capacity,count-offset)
          upload+=(await runner.setInstanceSources(input.subarray(offset*metadata.inputs,(offset+n)*metadata.inputs),n)).milliseconds
          dispatchReadback+=(await runner.run(shared,n,chunkOutput)).milliseconds
          output.set(chunkOutput.subarray(0,n*outputCount),offset*outputCount)
        }
        const correlation=performance.now()-start
        const best=winnerOnly?Uint32Array.from({length:count},(_,i)=>output[i*2]):winners(output,count,metadata.trials,denominators)
        return {correlation,search:performance.now()-start,upload,dispatchReadback,best}
      }
      const cold = await run()
      const samples=[]
      for(let r=0;r<repeats;r++) samples.push(await run())
      if(!output.every(Number.isFinite)) throw new Error('nonfinite GPU output')
      result.methods[name]={setup_ms:setupMs,pipeline_compile_ms:runner.compileMilliseconds,
        selector:winnerOnly?'GPU FP32 squared magnitude, no normalization':'host FP64 hypot and original denominators',
        first_search_ms:cold.search, correlation:summary(samples.map(x=>x.correlation)),
        search:summary(samples.map(x=>x.search)), upload:summary(samples.map(x=>x.upload)),
        dispatch_readback:summary(samples.map(x=>x.dispatchReadback))}
      for(const [suffix,data] of [['',output],['-best',samples.at(-1).best]]) {
        const response=await fetch(`/${count}/gpu-${name}${suffix}.bin`,{method:'POST',body:data})
        if(!response.ok)throw new Error('result save failed')
      }
    } catch(error) {
      result.methods[name]={error:String(error)}
    } finally {
      runner?.destroy()
    }
  }
  device.destroy()
  const response=await fetch(`/${count}/gpu.json`,{method:'POST',body:JSON.stringify(result,null,2)})
  if(!response.ok) throw new Error('result save failed')
  return result
}

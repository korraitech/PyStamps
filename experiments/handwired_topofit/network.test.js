import {test, expect} from 'bun:test'
import {buildNetwork, winners} from './network.mjs'
import {loadRuntime} from './runtime.mjs'
import witness from './near-tie.json'

test('actual Handwired interpreted and generated outputs match a rounded dense product', async () => {
  const runtime = await loadRuntime()
  const weights = new Float32Array([0.2, -0.3, 0.7, 0.2, -0.1, 0.8])
  const input = new Float32Array([0.5, -0.2, 1, -0.6, 0.3, 0.1])
  const network = buildNetwork(runtime, weights, 3, 2)
  const expected = []
  for (let n=0; n<2; n++) for (let j=0; j<2; j++) {
    let sum=0
    for (let k=0; k<3; k++) sum=Math.fround(sum+Math.fround(input[n*3+k]*weights[k*2+j]))
    expected.push(sum)
  }
  expect([...network.interpreted(input, 2)]).toEqual(expected)
  expect([...network.generated(input, 2)]).toEqual(expected)
})

test('host selection uses first trial on exact magnitude ties', () => {
  expect([...winners(new Float32Array([3, -3, 4, -4]), 1, 2)]).toEqual([0])
})

test('fused Handwired selector preserves first-index ties across its tree', async () => {
  const runtime=await loadRuntime()
  // Three identical real correlation columns; imaginary correlations are zero.
  const network=buildNetwork(runtime,new Float32Array([1,1,1,0,0,0, 1,1,1,0,0,0]),2,6)
  expect([...network.fused.evaluate(new Float32Array([0.5,0.5]))]).toEqual([0,1])
})

test('actual calibration near-tie documents the FP32 winner divergence', async () => {
  const runtime=await loadRuntime()
  const I=witness.baseline.length,T=33
  const weights=new Float32Array(2*I*2*T)
  const range=Math.max(...witness.baseline)-Math.min(...witness.baseline)
  for(let i=0;i<I;i++)for(let j=0;j<T;j++) {
    const angle=-(witness.baseline[i]/range*Math.PI/4)*(j-16)
    const re=Math.cos(angle),im=Math.sin(angle)
    weights[i*2*T+j]=re;weights[i*2*T+T+j]=im
    weights[(i+I)*2*T+j]=-im;weights[(i+I)*2*T+T+j]=re
  }
  const input=new Float32Array([...witness.real,...witness.imag])
  const network=buildNetwork(runtime,weights,2*I,2*T)
  const selected=winners(network.interpreted(input,1),1,T)[0]
  expect(selected).toBe(witness.handwired_trial)
  expect(selected).not.toBe(witness.source_trial)
  expect(network.fused.evaluate(input)[0]).toBe(witness.handwired_trial)
})

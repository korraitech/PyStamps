// A simple tiled, direct FP32 matrix multiplication control, not a tuned GEMM.
// Input is row-major [N, K], weights are row-major [K, M].
export async function createMatmulRunner(device, weights, K, M, capacity, options = {}) {
  const { winnerOnly = false } = options;
  if (typeof winnerOnly !== 'boolean') throw new TypeError('winnerOnly must be a boolean');
  for (const [name, value] of Object.entries({ K, M, capacity })) {
    if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffffffff) {
      throw new RangeError(`${name} must be a positive 32-bit integer`);
    }
  }
  if (!(weights instanceof Float32Array) || weights.length !== K * M) {
    throw new RangeError(`weights must be a Float32Array with ${K * M} elements`);
  }
  if (winnerOnly && M % 2 !== 0) {
    throw new RangeError('winnerOnly requires an even M with positive real/imaginary trial halves');
  }
  const outputCount = winnerOnly ? 2 : M;
  const trials = M / 2;

  const byteSizes = { input: capacity * K * 4, weights: K * M * 4, output: capacity * M * 4 };
  for (const [name, bytes] of Object.entries(byteSizes)) {
    if (!Number.isSafeInteger(bytes) || bytes / 4 > 0xffffffff ||
        bytes > device.limits.maxBufferSize || bytes > device.limits.maxStorageBufferBindingSize) {
      throw new RangeError(`${name} buffer (${bytes} bytes) exceeds device or shader indexing limits`);
    }
  }
  if (Math.ceil(M / 8) > device.limits.maxComputeWorkgroupsPerDimension ||
      Math.ceil(capacity / 8) > device.limits.maxComputeWorkgroupsPerDimension) {
    throw new RangeError('matrix dimensions exceed device dispatch limits');
  }

  const buffers = [];
  const makeBuffer = (label, size, usage) => {
    const buffer = device.createBuffer({ label, size, usage });
    buffers.push(buffer);
    return buffer;
  };

  device.pushErrorScope('validation');
  let pipeline;
  let winnerPipeline;
  let compileMilliseconds;
  let setupError;
  let inputBuffer, outputBuffer, readbackBuffer, parametersBuffer, bindGroup;
  let winnerBuffer, winnerBindGroup;
  try {
    const compileStart = performance.now();
    const module = device.createShaderModule({
      label: 'direct tiled matmul control',
      code: `
struct Parameters { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;
var<workgroup> inputTile: array<f32, 64>;
var<workgroup> weightTile: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = global.y;
  let column = global.x;
  let tileIndex = local.y * 8u + local.x;
  var sum = 0.0;
  for (var base = 0u; base < ${K}u; base += 8u) {
    inputTile[tileIndex] = 0.0;
    weightTile[tileIndex] = 0.0;
    if (row < parameters.count && base + local.x < ${K}u) {
      inputTile[tileIndex] = input[row * ${K}u + base + local.x];
    }
    if (column < ${M}u && base + local.y < ${K}u) {
      weightTile[tileIndex] = weights[(base + local.y) * ${M}u + column];
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      sum += inputTile[local.y * 8u + k] * weightTile[k * 8u + local.x];
    }
    workgroupBarrier();
  }
  if (row < parameters.count && column < ${M}u) {
    output[row * ${M}u + column] = sum;
  }
}`,
    });
    pipeline = await device.createComputePipelineAsync({
      label: 'direct tiled matmul control',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    if (winnerOnly) {
      const winnerModule = device.createShaderModule({
        label: 'matmul control winner reduction',
        code: `
struct Parameters { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> winners: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;

fn squaredScore(row: u32, trial: u32) -> f32 {
  let re: f32 = values[row * ${M}u + trial];
  let im: f32 = values[row * ${M}u + ${trials}u + trial];
  let reSquared: f32 = re * re;
  let imSquared: f32 = im * im;
  let score: f32 = reSquared + imSquared;
  return score;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let row = global.x;
  if (row >= parameters.count) { return; }
  var bestIndex = 0u;
  var bestScore: f32 = squaredScore(row, 0u);
  for (var trial = 1u; trial < ${trials}u; trial += 1u) {
    let score: f32 = squaredScore(row, trial);
    // Strict comparison retains the first trial when scores tie.
    if (score > bestScore) {
      bestIndex = trial;
      bestScore = score;
    }
  }
  winners[row * 2u] = f32(bestIndex);
  winners[row * 2u + 1u] = bestScore;
}`,
      });
      winnerPipeline = await device.createComputePipelineAsync({
        label: 'matmul control winner reduction',
        layout: 'auto',
        compute: { module: winnerModule, entryPoint: 'main' },
      });
    }
    compileMilliseconds = performance.now() - compileStart;
    inputBuffer = makeBuffer('control input', byteSizes.input, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const weightsBuffer = makeBuffer('control weights', byteSizes.weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    outputBuffer = makeBuffer('control output', byteSizes.output, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    readbackBuffer = makeBuffer('control readback', capacity * outputCount * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    parametersBuffer = makeBuffer('control parameters', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [inputBuffer, weightsBuffer, outputBuffer, parametersBuffer].map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    });
    if (winnerOnly) {
      winnerBuffer = makeBuffer('control winners', capacity * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      winnerBindGroup = device.createBindGroup({
        layout: winnerPipeline.getBindGroupLayout(0),
        entries: [outputBuffer, winnerBuffer, parametersBuffer].map((buffer, binding) => ({
          binding, resource: { buffer },
        })),
      });
    }
    device.queue.writeBuffer(weightsBuffer, 0, weights);
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    setupError = error;
  }
  const validationError = await device.popErrorScope();
  if (setupError || validationError) {
    for (const buffer of buffers) buffer.destroy();
    throw setupError ?? new Error(`WebGPU control validation failed: ${validationError.message}`);
  }

  let uploadedCount = 0;
  let destroyed = false;
  let busy = false;
  const checkReady = () => {
    if (destroyed) throw new Error('matrix runner has been destroyed');
    if (busy) throw new Error('matrix runner operations must be awaited sequentially');
  };
  const checkCount = (count) => {
    if (!Number.isSafeInteger(count) || count < 0 || count > capacity) {
      throw new RangeError(`count must be an integer between 0 and ${capacity}`);
    }
  };

  return {
    compileMilliseconds,
    outputCount,
    async setInstanceSources(input, count) {
      checkReady();
      checkCount(count);
      if (!(input instanceof Float32Array) || input.length < count * K) {
        throw new RangeError(`input must be a Float32Array with at least ${count * K} elements`);
      }
      busy = true;
      uploadedCount = 0;
      const start = performance.now();
      try {
        if (count > 0) device.queue.writeBuffer(inputBuffer, 0, input, 0, count * K);
        await device.queue.onSubmittedWorkDone();
        uploadedCount = count;
        return { milliseconds: performance.now() - start };
      } finally {
        busy = false;
      }
    },
    async run(_sharedIgnored, count, target) {
      checkReady();
      checkCount(count);
      if (count > uploadedCount) throw new RangeError('count exceeds the number of uploaded input rows');
      if (!(target instanceof Float32Array) || target.length < count * outputCount) {
        throw new RangeError(`target must be a Float32Array with at least ${count * outputCount} elements`);
      }
      busy = true;
      const start = performance.now();
      try {
        if (count === 0) return { milliseconds: performance.now() - start };
        const bytes = count * outputCount * 4;
        device.queue.writeBuffer(parametersBuffer, 0, new Uint32Array([count, 0, 0, 0]));
        const encoder = device.createCommandEncoder({ label: 'control matmul and readback' });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(count / 8));
        pass.end();
        if (winnerOnly) {
          const winnerPass = encoder.beginComputePass();
          winnerPass.setPipeline(winnerPipeline);
          winnerPass.setBindGroup(0, winnerBindGroup);
          winnerPass.dispatchWorkgroups(Math.ceil(count / 64));
          winnerPass.end();
        }
        encoder.copyBufferToBuffer(winnerOnly ? winnerBuffer : outputBuffer, 0, readbackBuffer, 0, bytes);
        device.queue.submit([encoder.finish()]);
        await readbackBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
        try {
          target.set(new Float32Array(readbackBuffer.getMappedRange(0, bytes)), 0);
        } finally {
          readbackBuffer.unmap();
        }
        return { milliseconds: performance.now() - start };
      } finally {
        busy = false;
      }
    },
    destroy() {
      if (destroyed) return;
      checkReady();
      for (const buffer of buffers) buffer.destroy();
      destroyed = true;
    },
  };
}

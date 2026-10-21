// Local-only fixture server and upstream WebGPU runner bundler, run with Bun.
import {resolve, dirname} from 'node:path'
import {fileURLToPath} from 'node:url'
const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(process.argv[2])
const upstream = resolve(process.env.HANDWIRED_DIR)
const entry = resolve(root, 'upstream-entry.ts')
await Bun.write(entry, `export {createGpuInstanceRunner} from ${JSON.stringify(resolve(upstream, 'lab/ui-exploration/codegen/gpu-runner.ts'))};`)
const build = await Bun.build({entrypoints:[entry],outdir:root,target:'browser',naming:'upstream.js'})
if (!build.success) throw new Error(build.logs.join('\n'))
const port = Number(process.argv[3] || 8769)
Bun.serve({hostname:'127.0.0.1',port,async fetch(request) {
  const path = new URL(request.url).pathname
  if (request.method === 'POST') {
    if (!/^\/\d+\/gpu(?:-(?:strict|relaxed|matmul|fused|matmul-winner)(?:-best)?\.bin|\.json|\.smoke\.json)$/.test(path)) return new Response('invalid result path',{status:400})
    await Bun.write(root + path, await request.arrayBuffer())
    return new Response('saved')
  }
  if (path === '/') return new Response('<!doctype html><title>PyStamps Step 2 benchmark</title><p>Local GPU benchmark. Results are written to the fixture directory.</p>',{headers:{'Content-Type':'text/html'}})
  if (['/browser.mjs','/network.mjs','/gpu-control.mjs'].includes(path)) return new Response(Bun.file(here+path))
  if (path === '/upstream.js' || /^\/\d+\/(input\.json|input\.bin|weights\.bin|denominators\.bin|(?:strict|relaxed|fused)-(?:module\.json|shared\.bin))$/.test(path)) return new Response(Bun.file(root+path))
  return new Response('not found',{status:404})
}})
console.log(`http://127.0.0.1:${port}`)

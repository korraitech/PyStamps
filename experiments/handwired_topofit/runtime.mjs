import {resolve} from 'node:path'
import {pathToFileURL} from 'node:url'

export async function loadRuntime() {
  if (!process.env.HANDWIRED_DIR) throw new Error('Set HANDWIRED_DIR to the pinned upstream checkout')
  const root = resolve(process.env.HANDWIRED_DIR)
  const paths = ['src/runtime/numeric-ui-core.ts', ...['instance.ts', 'wgsl.ts', 'gpu-runner.ts'].map(p => 'lab/ui-exploration/codegen/' + p)]
  const modules = await Promise.all(paths.map(p => import(pathToFileURL(resolve(root, p)).href)))
  return Object.assign({}, ...modules)
}

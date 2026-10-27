import { defineConfig } from 'vite';
import path from 'node:path';
import { ensureData, ROOT } from './scripts/dataset-sync.mjs';

export default defineConfig({
  plugins: [{
    name: 'olho-snapshot-sync',
    configureServer(server) {
      if (process.env.OLHO_DATA_MODE === 'local') return;
      const manifest = path.join(ROOT, 'data-manifest.json');
      let pending = null;
      server.watcher.add(manifest);
      server.watcher.on('change', (filename) => {
        if (path.resolve(filename) !== manifest) return;
        const update = ensureData({ profile: process.env.OLHO_DATA_PROFILE || 'full' });
        pending = update;
        update.then(() => server.ws.send({ type: 'full-reload' })).catch((error) => {
          server.config.logger.error(`Snapshot não preparado: ${error.message}`);
          server.ws.send({ type: 'error', err: { message: error.message, stack: '' } });
        });
        // Em falhas, mantém a Promise rejeitada: a API de dados não serve a versão antiga.
        update.then(() => { if (pending === update) pending = null; }, () => {});
      });
      server.middlewares.use(async (request, response, next) => {
        if (request.url?.startsWith('/data/') && pending) {
          try { await pending; } catch {
            response.statusCode = 503;
            response.end('Snapshot indisponível. Execute pnpm setup.');
            return;
          }
        }
        next();
      });
    },
  }],
});

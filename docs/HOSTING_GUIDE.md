# Free hosting guide

The website does not need a database, paid inference API, or server-side model. Cloudflare Workers is the recommended host. GitHub Pages is included as a static fallback.

## Option A — Cloudflare Workers (recommended)

Cloudflare's free Workers plan is enough for a college demonstration. The current free-plan limit is documented as 100,000 Worker requests per day. Always check the official [Cloudflare Workers limits](https://developers.cloudflare.com/workers/platform/limits/) before a public launch.

### First deployment from your computer

1. Create a free account at [dash.cloudflare.com](https://dash.cloudflare.com/).
2. Install Node.js 22 or newer.
3. Download or clone this repository.
4. Open PowerShell in the project folder.
5. Install dependencies:

   ```powershell
   npm install
   ```

6. Sign in to Cloudflare:

   ```powershell
   npx wrangler login
   ```

7. Your browser opens. Sign in, choose **Allow**, and return to PowerShell.
8. Build and publish:

   ```powershell
   npm run deploy:cloudflare
   ```

9. Wrangler prints a secure `https://...workers.dev` address. Open it and allow camera access.

For later updates, push the code to GitHub and run `npm run deploy:cloudflare` again. Wrangler deploys the generated Worker and its static model files together. This follows Cloudflare's official [Vite deployment workflow](https://developers.cloudflare.com/workers/vite-plugin/tutorial/).

### Automatic deployment from GitHub

1. Sign in to the Cloudflare dashboard.
2. Open **Workers & Pages**.
3. Select **Create application** and then **Import a repository**.
4. Connect GitHub and select `Arnav-Dugad/Driver-Detection-System`.
5. Keep `main` as the production branch and `/` as the root directory.
6. Use `npm run build` as the build command.
7. Use `npx wrangler deploy` as the deploy command.
8. If Cloudflare asks for a Node version, choose Node 22.
9. Select **Save and Deploy**.

Cloudflare will rebuild after each push to `main`. The official [Workers Builds guide](https://developers.cloudflare.com/workers/ci-cd/builds/) describes the same Git integration.

## Option B — GitHub Pages fallback

The repository includes `.github/workflows/github-pages.yml`. It produces a special static build with the correct `/Driver-Detection-System` asset paths, including both ML models and the MediaPipe runtime.

1. Keep the GitHub repository public so Pages and Actions remain free on GitHub Free.
2. Open the repository on GitHub.
3. Select **Settings**.
4. In the left sidebar, select **Pages**.
5. Under **Build and deployment**, set **Source** to **GitHub Actions**.
6. Open the repository's **Actions** tab.
7. Select **Deploy GitHub Pages**.
8. Choose **Run workflow**, keep branch `main`, and confirm.
9. Wait for both the build and deploy jobs to become green.
10. Open:

    `https://arnav-dugad.github.io/Driver-Detection-System/`

Every later push to `main` deploys automatically. GitHub documents this process in [Using custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Camera and voice notes

- Camera access works on `localhost` and secure `https://` deployments. Do not test from a raw local HTML file.
- Chrome or Edge is recommended for the widest MediaPipe and speech support.
- Voice availability is determined by the current browser's `speechSynthesis.getVoices()` list. If a selected desktop voice is unavailable, the app falls back to Hindi and then English instead of remaining silent.
- Camera frames remain in the browser with either host. The host serves application files and model weights; it does not receive webcam frames.

## If deployment fails

### Cloudflare login keeps failing

Run `npx wrangler logout`, then `npx wrangler login` again. Disable strict popup blocking for the login page.

### Cloudflare build says Node is too old

Select Node 22 in the build settings or add a build variable named `NODE_VERSION` with value `22`.

### GitHub Pages returns 404

Confirm the Pages source is **GitHub Actions**, the workflow finished successfully, and the URL includes `/Driver-Detection-System/` with the exact capitalization.

### The page opens but models fail to load

Hard-refresh with `Ctrl+Shift+R`. Then open the browser console and confirm that files below `models/` and `wasm/` are returning HTTP 200 instead of 404.

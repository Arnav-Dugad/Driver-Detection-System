import type { NextConfig } from "next";

const githubPages = process.env.GITHUB_PAGES === "true";
const githubBasePath = "/Driver-Detection-System";

const nextConfig: NextConfig = {
  output: githubPages ? "export" : undefined,
  basePath: "",
  assetPrefix: githubPages ? githubBasePath : "",
  // vinext prerenders each route at its bare path, so trailing-slash redirects
  // would block every route except "/". GitHub Pages resolves /lab to lab.html.
  trailingSlash: false,
};

export default nextConfig;

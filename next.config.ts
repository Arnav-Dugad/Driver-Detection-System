import type { NextConfig } from "next";

const githubPages = process.env.GITHUB_PAGES === "true";
const githubBasePath = "/Driver-Detection-System";

const nextConfig: NextConfig = {
  output: githubPages ? "export" : undefined,
  basePath: "",
  assetPrefix: githubPages ? githubBasePath : "",
  trailingSlash: githubPages,
};

export default nextConfig;

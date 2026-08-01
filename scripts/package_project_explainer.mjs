#!/usr/bin/env node

import {
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const VIEWPORT_GUARD = `<style data-project-explainer-viewport-guard="true">
.analytics-top-bar,.portable-page-header{width:100%!important;margin-right:0!important;margin-left:0!important}
</style>`;

function usage() {
  return [
    "Usage: node scripts/package_project_explainer.mjs --input <artifact.json> --output <report.html> --tool-dir <shared-builder-dir>",
    "",
    "Uses the shared Data Analytics portable reader and verifier. The only local",
    "compatibility change prevents its 100vw sticky header from adding the desktop",
    "scrollbar width and triggering horizontal overflow.",
  ].join("\n");
}

function parseArguments(argv) {
  const parsed = {};
  const allowed = new Set(["input", "output", "tool-dir", "screenshot"]);
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") return { help: true };
    if (!argument.startsWith("--") || !allowed.has(argument.slice(2))) {
      throw new Error(`Unexpected argument: ${argument}\n${usage()}`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`${argument} requires a value.\n${usage()}`);
    }
    parsed[argument.slice(2)] = value;
    index += 1;
  }
  if (!parsed.input || !parsed.output || !parsed["tool-dir"]) {
    throw new Error(`--input, --output, and --tool-dir are required.\n${usage()}`);
  }
  return parsed;
}

function addViewportGuard(html) {
  if (html.includes('data-project-explainer-viewport-guard="true"')) return html;
  if (!html.includes("</head>")) throw new Error("Portable reader HTML has no </head> marker.");
  return html.replace("</head>", `${VIEWPORT_GUARD}\n</head>`);
}

async function packageReport(options) {
  const inputPath = resolve(options.input);
  const outputPath = resolve(options.output);
  const toolDir = resolve(options["tool-dir"]);
  const temporaryPath = `${outputPath}.candidate-${process.pid}.html`;
  const screenshotPath = options.screenshot ? resolve(options.screenshot) : undefined;

  const builder = await import(pathToFileURL(join(toolDir, "build_portable_artifact.mjs")));
  const charts = await import(pathToFileURL(join(toolDir, "extract_portable_chart_svgs.mjs")));
  const verifier = await import(pathToFileURL(join(toolDir, "verify_portable_artifact.mjs")));

  const artifact = JSON.parse(readFileSync(inputPath, "utf8"));
  const packagedRuntime = builder.readPackagedReaderRuntime();
  const runtimeHtml = addViewportGuard(packagedRuntime.html);
  const build = (staticCharts) =>
    addViewportGuard(builder.buildPortableArtifact(artifact, { runtimeHtml, staticCharts }));

  mkdirSync(dirname(outputPath), { recursive: true });
  try {
    writeFileSync(temporaryPath, build(undefined), "utf8");
    const staticCharts = await charts.extractPortableChartSvgs({
      actionTimeoutMs: 5000,
      htmlPath: temporaryPath,
      readyTimeoutMs: 10000,
    });
    writeFileSync(temporaryPath, build(staticCharts), "utf8");

    const verification = await verifier.verifyPortableArtifact({
      actionTimeoutMs: 5000,
      artifactPath: inputPath,
      htmlPath: temporaryPath,
      readyTimeoutMs: 10000,
      screenshotPath,
      timeoutMs: 60000,
    });

    rmSync(outputPath, { force: true });
    renameSync(temporaryPath, outputPath);
    return {
      ok: true,
      stage: "passed",
      compatibilityGuard: "shared-reader sticky header width only",
      outputPath,
      verification,
    };
  } finally {
    rmSync(temporaryPath, { force: true });
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const receipt = await packageReport(options);
  process.stdout.write(`${JSON.stringify(receipt)}\n`);
}

try {
  await main();
} catch (error) {
  const result = error?.verificationResult ?? {
    ok: false,
    code: error?.code ?? "project_explainer_packaging_failed",
    error: error?.message ?? String(error),
  };
  process.stderr.write(`${JSON.stringify(result)}\n`);
  process.exitCode = 1;
}

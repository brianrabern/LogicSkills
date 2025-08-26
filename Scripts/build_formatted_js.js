"use strict";

const fs = require("fs");
const path = require("path");

function main() {
  const humanDir = path.resolve(__dirname, "..", "Assessors", "human");

  const files = [
    "formatted_countermodel_questions.json",
    "formatted_symbolization_questions_carroll.json",
    "formatted_symbolization_questions_english.json",
    "formatted_validity_questions_carroll.json",
    "formatted_validity_questions_english.json",
  ];

  const aggregated = {};

  for (const filename of files) {
    const key = path.basename(filename, ".json");
    const filePath = path.join(humanDir, filename);
    const raw = fs.readFileSync(filePath, "utf8");
    aggregated[key] = JSON.parse(raw);
  }

  const outPath = path.resolve(__dirname, "..", "formatted_data.js");
  const jsContent =
    "const FORMATTED = Object.freeze(" +
    JSON.stringify(aggregated, null, 2) +
    ");\n\nmodule.exports = FORMATTED;\n";

  fs.writeFileSync(outPath, jsContent, "utf8");
  console.log(`Wrote ${outPath}`);
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}

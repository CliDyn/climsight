const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat, ImageRun } = require("docx");

// Colors
const BLUE = "0284C7";
const DARK = "1E293B";
const GRAY = "64748B";
const LIGHT_BG = "F0F9FF";
const WHITE = "FFFFFF";
const ACCENT = "0EA5E9";

const noBorder = { style: BorderStyle.NONE, size: 0, color: WHITE };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "CBD5E1" };
const thinBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, spacing: { before: level === HeadingLevel.HEADING_1 ? 300 : 200, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: level === HeadingLevel.HEADING_1 ? 28 : 24, color: DARK })] });
}
function body(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, alignment: opts.align || AlignmentType.LEFT,
    children: [new TextRun({ text, font: "Arial", size: 20, color: opts.color || GRAY, bold: opts.bold || false, italics: opts.italics || false })] });
}
function bodyRuns(runs) {
  return new Paragraph({ spacing: { after: 120 },
    children: runs.map(r => new TextRun({ text: r.text, font: "Arial", size: 20, color: r.color || GRAY, bold: r.bold || false, italics: r.italics || false })) });
}
function spacer(h = 80) { return new Paragraph({ spacing: { before: h, after: 0 }, children: [] }); }

function cell(text, opts = {}) {
  return new TableCell({
    borders: opts.borders || thinBorders,
    width: { size: opts.width || 4680, type: WidthType.DXA },
    shading: opts.bg ? { fill: opts.bg, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, font: "Arial", size: opts.size || 20, color: opts.color || DARK, bold: opts.bold || false })] })]
  });
}

// -- Architecture diagram as text-based table --
function archDiagram() {
  const hdrCell = (text, w) => new TableCell({
    borders: noBorders, width: { size: w, type: WidthType.DXA },
    shading: { fill: BLUE, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: "Arial", size: 18, color: WHITE, bold: true })] })]
  });
  const boxCell = (text, w, bg = LIGHT_BG) => new TableCell({
    borders: { top: { style: BorderStyle.SINGLE, size: 1, color: ACCENT }, bottom: { style: BorderStyle.SINGLE, size: 1, color: ACCENT },
               left: { style: BorderStyle.SINGLE, size: 1, color: ACCENT }, right: { style: BorderStyle.SINGLE, size: 1, color: ACCENT } },
    width: { size: w, type: WidthType.DXA },
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: "Arial", size: 16, color: DARK })] })]
  });

  return [
    // Top layer
    new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360],
      rows: [new TableRow({ children: [hdrCell("ClimAssist  \u2014  Climate-Health-Agriculture Decision Platform", 9360)] })] }),
    spacer(60),
    // Data sources row
    new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 3120, 3120],
      rows: [new TableRow({ children: [
        boxCell("Climate Models\n(ERA5, nextGEMS, DestinE)", 3120),
        boxCell("Health Records\n(Malaria, Meningitis, Nutrition)", 3120),
        boxCell("Agriculture Data\n(ECOCROP, Soil, Crop Yields)", 3120),
      ] })] }),
    spacer(30),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u25BC", font: "Arial", size: 24, color: BLUE })] }),
    spacer(30),
    // AI Engine
    new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [4680, 4680],
      rows: [new TableRow({ children: [
        boxCell("Multi-Agent AI Engine\n(RAG + LLM + Climate Science)", 4680, "E0F2FE"),
        boxCell("Predictive Analytics\n(Malaria risk, Crop timing, Drought)", 4680, "E0F2FE"),
      ] })] }),
    spacer(30),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u25BC", font: "Arial", size: 24, color: BLUE })] }),
    spacer(30),
    // Outputs row
    new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 3120, 3120],
      rows: [new TableRow({ children: [
        boxCell("Farmer Advisory\n(When/what to plant)", 3120, "ECFDF5"),
        boxCell("Health Early Warning\n(8-week malaria forecast)", 3120, "ECFDF5"),
        boxCell("Policy Dashboard\n(LGA-level climate risk)", 3120, "ECFDF5"),
      ] })] }),
    spacer(30),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u25BC", font: "Arial", size: 24, color: BLUE })] }),
    spacer(30),
    // Stakeholders
    new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [4680, 4680],
      rows: [new TableRow({ children: [
        boxCell("Katsina State Government\n(34 LGAs, Extension Workers)", 4680, "FFF7ED"),
        boxCell("Primary Health Centers\n(Digitized Health Records)", 4680, "FFF7ED"),
      ] })] }),
  ];
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: DARK },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: DARK },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  sections: [
    // === PAGE 1 ===
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, right: 1440, bottom: 1080, left: 1440 }
        }
      },
      headers: {
        default: new Header({ children: [
          new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 4 } },
            children: [
              new TextRun({ text: "SCOLO", font: "Arial", size: 18, bold: true, color: BLUE }),
              new TextRun({ text: "  |  Confidential", font: "Arial", size: 16, color: GRAY }),
            ] })
        ] })
      },
      footers: {
        default: new Footer({ children: [
          new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "Page ", font: "Arial", size: 16, color: GRAY }), new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: GRAY })] })
        ] })
      },
      children: [
        // Title block
        new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 40 },
          children: [new TextRun({ text: "ClimAssist", font: "Arial", size: 44, bold: true, color: BLUE })] }),
        new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 60 },
          children: [new TextRun({ text: "Climate-Health-Agriculture Decision Platform for Katsina State", font: "Arial", size: 24, color: DARK })] }),
        new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 1 } }, spacing: { after: 200 }, children: [] }),

        // The problem
        heading("The Problem"),
        body("Katsina State has lost 15% of its forest cover. 60% of farmlands face drought. 75% of the population depends on agriculture. Meanwhile, 77% of severe childhood malaria cases occur May\u2013September, directly correlated with rainfall patterns. Climate data exists but is trapped in global models that no local decision-maker can access or act on."),

        // The vision
        heading("The Vision"),
        bodyRuns([
          { text: "ClimAssist ", bold: true, color: DARK },
          { text: "connects climate projections, health surveillance, and agricultural intelligence into a single decision platform \u2014 purpose-built for Katsina State. A farmer in Jibia gets planting guidance based on real climate forecasts. A health officer in Dutsin-Ma gets an 8-week malaria risk forecast. The SA on Climate Change gets an LGA-level dashboard showing where to deploy resources." },
        ]),

        spacer(100),
        // Architecture
        heading("Platform Architecture", HeadingLevel.HEADING_2),
        ...archDiagram(),

        // Page break
        new Paragraph({ children: [new PageBreak()] }),

        // === PAGE 2 ===
        heading("What We Build (Phase 1 \u2014 12 Months)"),

        // Phase 1 table
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2800, 6560],
          rows: [
            new TableRow({ children: [
              cell("Climate Engine", { width: 2800, bold: true, bg: LIGHT_BG }),
              cell("Deploy ClimSight for Katsina: localized climate projections, ERA5 ground truth, crop suitability analysis (ECOCROP), scientific literature RAG. Already functional.", { width: 6560 }),
            ] }),
            new TableRow({ children: [
              cell("Health Integration", { width: 2800, bold: true, bg: LIGHT_BG }),
              cell("Digitize basic health indicators (malaria cases, malnutrition admissions, respiratory infections) from primary health centers across priority LGAs. Build the climate-health correlation pipeline.", { width: 6560 }),
            ] }),
            new TableRow({ children: [
              cell("Predictive Layer", { width: 2800, bold: true, bg: LIGHT_BG }),
              cell("Train malaria early warning model using ERA5 rainfall (2\u20133 month lag) + temperature + historical case data. Target: 8-week advance warning at LGA level.", { width: 6560 }),
            ] }),
            new TableRow({ children: [
              cell("Farmer Advisory", { width: 2800, bold: true, bg: LIGHT_BG }),
              cell("SMS/WhatsApp-based planting guidance for extension workers. When to plant, what to plant, fertilizer timing \u2014 all driven by climate projections for their specific location.", { width: 6560 }),
            ] }),
            new TableRow({ children: [
              cell("Policy Dashboard", { width: 2800, bold: true, bg: LIGHT_BG }),
              cell("Web dashboard for the SA on Climate Change showing LGA-level risk maps, seasonal outlooks, and intervention tracking aligned with KAGGA targets.", { width: 6560 }),
            ] }),
          ]
        }),

        spacer(200),
        heading("Funding Strategy"),
        bodyRuns([
          { text: "Google.org AI for Science Challenge ", bold: true, color: DARK },
          { text: "(Deadline: April 17, 2026)" },
        ]),
        body("$500K\u2013$3M grants for AI projects advancing climate resilience and health. ClimAssist bridges both tracks. Scolo applies as the tech lead; the Office of the SA on Climate Change, Katsina State is the implementation partner."),

        spacer(60),
        heading("Why Katsina, Why Now", HeadingLevel.HEADING_2),

        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
          children: [
            new TextRun({ text: "#2 in Nigeria's Climate Governance Ranking ", font: "Arial", size: 20, bold: true, color: DARK }),
            new TextRun({ text: "\u2014 institutional readiness is already proven", font: "Arial", size: 20, color: GRAY }),
          ] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
          children: [
            new TextRun({ text: "30 bankable climate projects ", font: "Arial", size: 20, bold: true, color: DARK }),
            new TextRun({ text: "presented at COP30 \u2014 the mandate exists", font: "Arial", size: 20, color: GRAY }),
          ] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
          children: [
            new TextRun({ text: "30% state budget pledged to climate resilience ", font: "Arial", size: 20, bold: true, color: DARK }),
            new TextRun({ text: "\u2014 funding commitment is real", font: "Arial", size: 20, color: GRAY }),
          ] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
          children: [
            new TextRun({ text: "Africa's first subnational Green Public Procurement EO ", font: "Arial", size: 20, bold: true, color: DARK }),
            new TextRun({ text: "\u2014 policy innovation is happening", font: "Arial", size: 20, color: GRAY }),
          ] }),

        spacer(200),
        // Bottom CTA
        new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 8 } }, spacing: { before: 120, after: 80 },
          children: [new TextRun({ text: "Next Step", font: "Arial", size: 24, bold: true, color: DARK })] }),
        body("We deploy a customized ClimAssist instance for Katsina State within 4 weeks. Professor Al-Amin\u2019s office provides priority LGAs and health facility contacts. Scolo configures the platform and submits the Google.org application by April 17."),

        spacer(60),
        new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 40 },
          children: [
            new TextRun({ text: "Abdulhakim Gafai  |  ", font: "Arial", size: 20, bold: true, color: DARK }),
            new TextRun({ text: "Scolo  |  abdulhakim.gafai@gmail.com", font: "Arial", size: 20, color: GRAY }),
          ] }),
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/great-gifted-fermat/mnt/climsight/ClimAssist_Vision_Katsina.docx", buffer);
  console.log("Done: ClimAssist_Vision_Katsina.docx");
});

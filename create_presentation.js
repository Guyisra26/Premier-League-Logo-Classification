const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Guy Israeli";
pres.title = "Premier League Logo Classification — Deep Learning Final Project";

// ── Color Palette ──────────────────────────────────────────────────────────
const C = {
  darkBg:    "0F172A",  // deep navy
  darkBg2:   "1E293B",  // slate 800
  cardBg:    "FFFFFF",
  lightBg:   "F1F5F9",  // slate 100
  accent:    "10B981",  // emerald
  accent2:   "0EA5E9",  // sky blue
  accent3:   "F59E0B",  // amber
  accentRed: "EF4444",  // red
  white:     "FFFFFF",
  textDark:  "0F172A",
  textMuted: "64748B",  // slate 500
  textLight: "94A3B8",  // slate 400
  border:    "E2E8F0",  // slate 200
  green50:   "F0FDF4",
};

// ── Helpers ────────────────────────────────────────────────────────────────
const makeShadow = () => ({
  type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.10
});

function addDarkSlide(title, subtitle) {
  const slide = pres.addSlide();
  slide.background = { color: C.darkBg };
  // Top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.04, fill: { color: C.accent }
  });
  if (title) {
    slide.addText(title, {
      x: 0.8, y: 1.8, w: 8.4, h: 1.0, fontSize: 36, fontFace: "Georgia",
      color: C.white, bold: true, margin: 0
    });
  }
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.8, y: 2.85, w: 8.4, h: 0.5, fontSize: 16, fontFace: "Calibri",
      color: C.textLight, margin: 0
    });
  }
  return slide;
}

function addContentSlide(title) {
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  // Left accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.06, h: 5.625, fill: { color: C.accent }
  });
  // Title
  slide.addText(title, {
    x: 0.7, y: 0.3, w: 8.6, h: 0.55, fontSize: 22, fontFace: "Georgia",
    color: C.textDark, bold: true, margin: 0
  });
  // Separator line under title
  slide.addShape(pres.shapes.LINE, {
    x: 0.7, y: 0.9, w: 8.6, h: 0, line: { color: C.accent, width: 2 }
  });
  return slide;
}

function addCard(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.cardBg },
    shadow: makeShadow(),
    line: opts.border ? { color: opts.border, width: 1 } : undefined,
  });
}

function addStatCard(slide, x, y, w, h, value, label, color) {
  addCard(slide, x, y, w, h);
  // Accent top bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: w, h: 0.05, fill: { color: color || C.accent }
  });
  slide.addText(value, {
    x: x, y: y + 0.2, w: w, h: 0.65, fontSize: 32, fontFace: "Calibri",
    color: color || C.accent, bold: true, align: "center", margin: 0
  });
  slide.addText(label, {
    x: x, y: y + 0.85, w: w, h: 0.35, fontSize: 11, fontFace: "Calibri",
    color: C.textMuted, align: "center", margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.darkBg };

  // Top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
  });

  // Main title
  slide.addText("Premier League\nLogo Classification", {
    x: 0.8, y: 1.0, w: 8.4, h: 2.0, fontSize: 44, fontFace: "Georgia",
    color: C.white, bold: true, margin: 0
  });

  // Subtitle
  slide.addText("Image Classification with Convolutional Neural Networks", {
    x: 0.8, y: 3.05, w: 8.4, h: 0.5, fontSize: 18, fontFace: "Calibri",
    color: C.accent, margin: 0
  });

  // Bottom info
  slide.addText("Deep Learning — Final Project", {
    x: 0.8, y: 4.2, w: 4, h: 0.35, fontSize: 13, fontFace: "Calibri",
    color: C.textLight, margin: 0
  });
  slide.addText("Guy Israeli", {
    x: 0.8, y: 4.55, w: 4, h: 0.35, fontSize: 13, fontFace: "Calibri",
    color: C.textMuted, margin: 0
  });

  // Decorative shape — right side
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 8.5, y: 0.8, w: 1.2, h: 4.2,
    fill: { color: C.accent, transparency: 12 }
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 8.9, y: 1.2, w: 0.8, h: 3.4,
    fill: { color: C.accent, transparency: 20 }
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem & Dataset
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("Problem & Dataset");

  // Left column — description
  addCard(slide, 0.7, 1.2, 5.0, 2.0);
  slide.addText("Classification Task", {
    x: 1.0, y: 1.35, w: 4.4, h: 0.35, fontSize: 15, fontFace: "Georgia",
    color: C.accent, bold: true, margin: 0
  });
  slide.addText([
    { text: "Classify English Premier League football club logos into 20 classes using CNNs.", options: { breakLine: true, fontSize: 12 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Logos are visually distinct but share structural similarities — circular shapes, text, symbolic elements — forcing fine-grained discrimination.", options: { breakLine: true, fontSize: 12 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Source: Kaggle (alexteboul/english-premier-league-logo-detection-20k-images)", options: { fontSize: 10, italic: true, color: C.textMuted } },
  ], { x: 1.0, y: 1.75, w: 4.4, h: 1.3, fontFace: "Calibri", color: C.textDark, margin: 0 });

  // Right column — stat cards
  addStatCard(slide, 6.1, 1.2, 1.65, 1.35, "~20K", "Total Images", C.accent);
  addStatCard(slide, 8.0, 1.2, 1.65, 1.35, "20", "Club Classes", C.accent2);

  // Data split card
  addCard(slide, 6.1, 2.75, 3.55, 0.5);
  slide.addText("Split:  70% Train  /  15% Val  /  15% Test  (stratified)", {
    x: 6.3, y: 2.8, w: 3.2, h: 0.4, fontSize: 10, fontFace: "Calibri",
    color: C.textDark, margin: 0
  });

  // Why logos? card
  addCard(slide, 0.7, 3.5, 9.0, 1.5);
  slide.addText("Why Logo Classification?", {
    x: 1.0, y: 3.6, w: 8.4, h: 0.35, fontSize: 14, fontFace: "Georgia",
    color: C.accent2, bold: true, margin: 0
  });
  slide.addText([
    { text: "Variation in scale, rotation, background clutter & quality — non-trivial task", options: { bullet: true, breakLine: true } },
    { text: "~1,000 images/class: large enough for from-scratch training, small enough that transfer learning provides measurable advantage", options: { bullet: true, breakLine: true } },
    { text: "20-class setup: harder than binary, doesn't require massive compute", options: { bullet: true } },
  ], { x: 1.0, y: 4.0, w: 8.4, h: 0.95, fontSize: 11, fontFace: "Calibri", color: C.textDark, margin: 0 });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Part 1: Architecture Design
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addDarkSlide("Part 1", "Training CNNs from Scratch");

  // Section number badge
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 1.2, w: 0.5, h: 0.4, fill: { color: C.accent }
  });
  slide.addText("01", {
    x: 0.8, y: 1.2, w: 0.5, h: 0.4, fontSize: 16, fontFace: "Calibri",
    color: C.darkBg, bold: true, align: "center", valign: "middle", margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — CNN Architectures
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("CNN Architectures — SimpleCNN vs DeepCNN");

  // SimpleCNN card
  addCard(slide, 0.7, 1.2, 4.15, 3.8);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 1.2, w: 4.15, h: 0.05, fill: { color: C.accent2 }
  });
  slide.addText("SimpleCNN (Baseline)", {
    x: 1.0, y: 1.4, w: 3.6, h: 0.35, fontSize: 16, fontFace: "Georgia",
    color: C.accent2, bold: true, margin: 0
  });
  slide.addText([
    { text: "3 convolutional blocks", options: { bold: true, breakLine: true } },
    { text: "Conv2d → BatchNorm → ReLU → MaxPool", options: { breakLine: true, color: C.textMuted } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "Filter progression: 32 → 64 → 128", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "Classifier: Flatten → FC(512) → ReLU → Dropout → FC(20)", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "~1.5M parameters", options: { bold: true, color: C.accent2 } },
  ], { x: 1.0, y: 1.85, w: 3.6, h: 2.5, fontSize: 11, fontFace: "Calibri", color: C.textDark, margin: 0 });

  slide.addText("Purpose: Minimal baseline — how well can a simple\nfeature extractor distinguish 20 logo classes?", {
    x: 1.0, y: 4.1, w: 3.6, h: 0.6, fontSize: 10, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });

  // DeepCNN card
  addCard(slide, 5.15, 1.2, 4.5, 3.8);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 1.2, w: 4.5, h: 0.05, fill: { color: C.accent }
  });
  slide.addText("DeepCNN (Advanced)", {
    x: 5.45, y: 1.4, w: 3.9, h: 0.35, fontSize: 16, fontFace: "Georgia",
    color: C.accent, bold: true, margin: 0
  });
  slide.addText([
    { text: "5 convolutional blocks + skip connections", options: { bold: true, breakLine: true } },
    { text: "Conv2d → BatchNorm → ReLU → MaxPool", options: { breakLine: true, color: C.textMuted } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "Filter progression: 32 → 64 → 128 → 256 → 256", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "Skip connections at blocks 3 & 5 (residual addition)", options: { breakLine: true } },
    { text: "Classifier: GAP → FC(256) → ReLU → Dropout → FC(20)", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "~3.5M parameters  |  Configurable BN & Dropout", options: { bold: true, color: C.accent } },
  ], { x: 5.45, y: 1.85, w: 3.9, h: 2.5, fontSize: 11, fontFace: "Calibri", color: C.textDark, margin: 0 });

  slide.addText("Purpose: Test whether increased depth with residual\nconnections improves feature learning.", {
    x: 5.45, y: 4.1, w: 3.9, h: 0.6, fontSize: 10, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Optimizer Comparison
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("Optimizer Comparison — Adam vs SGD+Momentum");

  // Description
  slide.addText("Both optimizers tested on both architectures — all other hyperparameters fixed.", {
    x: 0.7, y: 1.1, w: 8.6, h: 0.3, fontSize: 11, fontFace: "Calibri",
    color: C.textMuted, margin: 0
  });

  // Optimizer table
  const headerOpts = { fill: { color: C.darkBg }, color: C.white, bold: true, fontSize: 11, fontFace: "Calibri", align: "center", valign: "middle" };
  const cellOpts = { color: C.textDark, fontSize: 11, fontFace: "Calibri", align: "center", valign: "middle" };
  const altRow = { fill: { color: "F8FAFC" } };

  slide.addTable([
    [
      { text: "Model", options: headerOpts },
      { text: "Optimizer", options: headerOpts },
      { text: "Learning Rate", options: headerOpts },
      { text: "Scheduler", options: headerOpts },
      { text: "Val Accuracy", options: headerOpts },
      { text: "Best Epoch", options: headerOpts },
    ],
    [
      { text: "SimpleCNN", options: cellOpts },
      { text: "Adam", options: cellOpts },
      { text: "1e-3", options: cellOpts },
      { text: "—", options: cellOpts },
      { text: "76.2%", options: cellOpts },
      { text: "9", options: cellOpts },
    ],
    [
      { text: "SimpleCNN", options: { ...cellOpts, ...altRow } },
      { text: "SGD+M", options: { ...cellOpts, ...altRow } },
      { text: "1e-2", options: { ...cellOpts, ...altRow } },
      { text: "StepLR (γ=0.1, step=7)", options: { ...cellOpts, ...altRow, fontSize: 10 } },
      { text: "72.8%", options: { ...cellOpts, ...altRow } },
      { text: "11", options: { ...cellOpts, ...altRow } },
    ],
    [
      { text: "DeepCNN", options: { ...cellOpts, bold: true } },
      { text: "Adam", options: { ...cellOpts, bold: true } },
      { text: "1e-3", options: cellOpts },
      { text: "—", options: cellOpts },
      { text: "84.5%", options: { ...cellOpts, bold: true, color: C.accent } },
      { text: "10", options: cellOpts },
    ],
    [
      { text: "DeepCNN", options: { ...cellOpts, ...altRow, bold: true } },
      { text: "SGD+M", options: { ...cellOpts, ...altRow, bold: true } },
      { text: "1e-2", options: { ...cellOpts, ...altRow } },
      { text: "StepLR (γ=0.1, step=7)", options: { ...cellOpts, ...altRow, fontSize: 10 } },
      { text: "80.1%", options: { ...cellOpts, ...altRow } },
      { text: "12", options: { ...cellOpts, ...altRow } },
    ],
  ], {
    x: 0.7, y: 1.6, w: 8.6,
    border: { pt: 0.5, color: C.border },
    colW: [1.4, 1.1, 1.2, 2.0, 1.6, 1.3],
    rowH: [0.4, 0.4, 0.4, 0.4, 0.4],
  });

  // Key findings card
  addCard(slide, 0.7, 3.9, 8.6, 1.3);
  slide.addText("Key Findings", {
    x: 1.0, y: 4.0, w: 3, h: 0.3, fontSize: 13, fontFace: "Georgia",
    color: C.accent, bold: true, margin: 0
  });
  slide.addText([
    { text: "Adam converges faster due to adaptive per-parameter learning rates — practical when compute is limited", options: { bullet: true, breakLine: true } },
    { text: "SGD+Momentum with StepLR schedule can match or exceed Adam's final accuracy with proper tuning", options: { bullet: true, breakLine: true } },
    { text: "DeepCNN consistently outperforms SimpleCNN — skip connections enable deeper feature learning", options: { bullet: true } },
  ], { x: 1.0, y: 4.35, w: 8.0, h: 0.8, fontSize: 11, fontFace: "Calibri", color: C.textDark, margin: 0 });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Hyperparameter Tuning
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("Hyperparameter Tuning — Learning Rate & Batch Size");

  // LR sweep card
  addCard(slide, 0.7, 1.2, 4.15, 2.9);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 1.2, w: 4.15, h: 0.05, fill: { color: C.accent3 }
  });
  slide.addText("Learning Rate Sweep", {
    x: 1.0, y: 1.4, w: 3.6, h: 0.3, fontSize: 14, fontFace: "Georgia",
    color: C.accent3, bold: true, margin: 0
  });
  slide.addText("DeepCNN + Adam  |  lr ∈ {1e-2, 1e-3, 1e-4}", {
    x: 1.0, y: 1.75, w: 3.6, h: 0.25, fontSize: 10, fontFace: "Calibri",
    color: C.textMuted, margin: 0
  });

  const lrHeaderOpts = { fill: { color: C.accent3 }, color: C.white, bold: true, fontSize: 10, fontFace: "Calibri", align: "center", valign: "middle" };
  const lrCellOpts = { color: C.textDark, fontSize: 11, fontFace: "Calibri", align: "center", valign: "middle" };

  slide.addTable([
    [
      { text: "Learning Rate", options: lrHeaderOpts },
      { text: "Val Accuracy", options: lrHeaderOpts },
      { text: "Best Epoch", options: lrHeaderOpts },
    ],
    [
      { text: "1e-2", options: lrCellOpts },
      { text: "71.3%", options: lrCellOpts },
      { text: "8", options: lrCellOpts },
    ],
    [
      { text: "1e-3", options: { ...lrCellOpts, bold: true } },
      { text: "84.5%", options: { ...lrCellOpts, bold: true } },
      { text: "10", options: { ...lrCellOpts, bold: true } },
    ],
    [
      { text: "1e-4", options: lrCellOpts },
      { text: "78.9%", options: lrCellOpts },
      { text: "12", options: lrCellOpts },
    ],
  ], {
    x: 1.0, y: 2.15, w: 3.5,
    border: { pt: 0.5, color: C.border },
    colW: [1.3, 1.1, 1.1],
    rowH: [0.32, 0.32, 0.32, 0.32],
  });

  slide.addText("Higher LR risks instability, lower LR converges slowly.\nGoal: find the sweet spot for Adam on this dataset.", {
    x: 1.0, y: 3.5, w: 3.6, h: 0.45, fontSize: 9.5, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });

  // BS sweep card
  addCard(slide, 5.15, 1.2, 4.5, 2.9);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 1.2, w: 4.5, h: 0.05, fill: { color: C.accent2 }
  });
  slide.addText("Batch Size Sweep", {
    x: 5.45, y: 1.4, w: 3.9, h: 0.3, fontSize: 14, fontFace: "Georgia",
    color: C.accent2, bold: true, margin: 0
  });
  slide.addText("DeepCNN + Adam (lr=1e-3)  |  bs ∈ {32, 64, 128}", {
    x: 5.45, y: 1.75, w: 3.9, h: 0.25, fontSize: 10, fontFace: "Calibri",
    color: C.textMuted, margin: 0
  });

  const bsHeaderOpts = { fill: { color: C.accent2 }, color: C.white, bold: true, fontSize: 10, fontFace: "Calibri", align: "center", valign: "middle" };
  const bsCellOpts = { color: C.textDark, fontSize: 11, fontFace: "Calibri", align: "center", valign: "middle" };

  slide.addTable([
    [
      { text: "Batch Size", options: bsHeaderOpts },
      { text: "Val Accuracy", options: bsHeaderOpts },
      { text: "Time (s)", options: bsHeaderOpts },
    ],
    [
      { text: "32", options: bsCellOpts },
      { text: "83.7%", options: bsCellOpts },
      { text: "285", options: bsCellOpts },
    ],
    [
      { text: "64", options: { ...bsCellOpts, bold: true } },
      { text: "84.5%", options: { ...bsCellOpts, bold: true } },
      { text: "168", options: { ...bsCellOpts, bold: true } },
    ],
    [
      { text: "128", options: bsCellOpts },
      { text: "82.9%", options: bsCellOpts },
      { text: "112", options: bsCellOpts },
    ],
  ], {
    x: 5.45, y: 2.15, w: 3.85,
    border: { pt: 0.5, color: C.border },
    colW: [1.3, 1.3, 1.25],
    rowH: [0.32, 0.32, 0.32, 0.32],
  });

  slide.addText("Smaller batches add gradient noise (implicit regularization).\nLarger batches give smoother updates and faster per-epoch time.", {
    x: 5.45, y: 3.5, w: 3.9, h: 0.45, fontSize: 9.5, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });

  // Bottom takeaway
  addCard(slide, 0.7, 4.35, 8.95, 0.9);
  slide.addText("Takeaway", {
    x: 1.0, y: 4.4, w: 1.5, h: 0.3, fontSize: 12, fontFace: "Georgia",
    color: C.accent, bold: true, margin: 0
  });
  slide.addText("Hyperparameter tuning is secondary to architecture choices, but can shift final performance by several percentage points. Sequential tuning (LR → BS → regularization) is practical under compute constraints.", {
    x: 1.0, y: 4.72, w: 8.4, h: 0.4, fontSize: 10.5, fontFace: "Calibri",
    color: C.textDark, margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — BatchNorm & Regularization
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("BatchNorm & Regularization Ablations");

  // BN card
  addCard(slide, 0.7, 1.2, 4.15, 2.0);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 1.2, w: 0.07, h: 2.0, fill: { color: C.accent2 }
  });
  slide.addText("Batch Normalization", {
    x: 1.0, y: 1.3, w: 3.6, h: 0.3, fontSize: 14, fontFace: "Georgia",
    color: C.accent2, bold: true, margin: 0
  });
  slide.addText([
    { text: "DeepCNN trained with and without BN layers", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "BN normalizes intermediate activations → reduces internal covariate shift", options: { breakLine: true } },
    { text: "Expected: removing BN destabilizes deeper model, slows convergence", options: { breakLine: true } },
    { text: "The deeper the model, the more critical BN becomes", options: { italic: true, color: C.textMuted } },
  ], { x: 1.0, y: 1.7, w: 3.6, h: 1.3, fontSize: 11, fontFace: "Calibri", color: C.textDark, margin: 0 });

  // Dropout card
  addCard(slide, 5.15, 1.2, 4.5, 2.0);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 1.2, w: 0.07, h: 2.0, fill: { color: C.accent }
  });
  slide.addText("Dropout Ablation", {
    x: 5.45, y: 1.3, w: 3.9, h: 0.3, fontSize: 14, fontFace: "Georgia",
    color: C.accent, bold: true, margin: 0
  });
  slide.addText([
    { text: "Dropout rates tested: 0.0, 0.3, 0.5", options: { breakLine: true, bold: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "0.0 — max capacity, most overfitting risk", options: { breakLine: true } },
    { text: "0.3 — mild regularization", options: { breakLine: true } },
    { text: "0.5 — strong regularization, limits train acc but improves generalization", options: {} },
  ], { x: 5.45, y: 1.7, w: 3.9, h: 1.3, fontSize: 11, fontFace: "Calibri", color: C.textDark, margin: 0 });

  // Weight Decay card
  addCard(slide, 0.7, 3.45, 4.15, 1.0);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 3.45, w: 0.07, h: 1.0, fill: { color: C.accent3 }
  });
  slide.addText("Weight Decay (L2)", {
    x: 1.0, y: 3.55, w: 2, h: 0.25, fontSize: 12, fontFace: "Georgia",
    color: C.accent3, bold: true, margin: 0
  });
  slide.addText("weight_decay=1e-4 penalizes large weights, encouraging smoother decision boundaries. Complementary to dropout.", {
    x: 1.0, y: 3.85, w: 3.6, h: 0.45, fontSize: 10.5, fontFace: "Calibri", color: C.textDark, margin: 0
  });

  // Data Augmentation card
  addCard(slide, 5.15, 3.45, 4.5, 1.0);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 3.45, w: 0.07, h: 1.0, fill: { color: C.accentRed }
  });
  slide.addText("Data Augmentation", {
    x: 5.45, y: 3.55, w: 2.5, h: 0.25, fontSize: 12, fontFace: "Georgia",
    color: C.accentRed, bold: true, margin: 0
  });
  slide.addText("Ablation: with vs without augmentation (flip, rotation, color jitter, random crop). Measures effect on train-val gap.", {
    x: 5.45, y: 3.85, w: 3.9, h: 0.45, fontSize: 10.5, fontFace: "Calibri", color: C.textDark, margin: 0
  });

  // Bottom note
  slide.addText("Each experiment changes one variable at a time — controlled ablation to attribute results to specific factors.", {
    x: 0.7, y: 4.75, w: 8.95, h: 0.3, fontSize: 10, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Part 2 Section Divider
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addDarkSlide("Part 2", "Transfer Learning via CIFAR-100 Pretraining");
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 1.2, w: 0.5, h: 0.4, fill: { color: C.accent2 }
  });
  slide.addText("02", {
    x: 0.8, y: 1.2, w: 0.5, h: 0.4, fontSize: 16, fontFace: "Calibri",
    color: C.darkBg, bold: true, align: "center", valign: "middle", margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 9 — CIFAR-100 Transfer Learning
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("CIFAR-100 Pretraining & Fine-Tuning");

  // Process flow — 4 steps
  const steps = [
    { num: "1", title: "Pretrain", desc: "Train DeepCNN\non CIFAR-100\n(100 classes)", color: C.accent2 },
    { num: "2", title: "Save", desc: "Save trained\nbackbone weights\nto disk (.pth)", color: C.accent },
    { num: "3", title: "Adapt", desc: "Replace 100-class\nhead with new\n20-class head", color: C.accent3 },
    { num: "4", title: "Fine-tune", desc: "Train on logos\nwith lr=1e-4\n(reduced LR)", color: C.accentRed },
  ];

  const startX = 0.7;
  const stepW = 2.0;
  const gap = 0.3;

  steps.forEach((step, i) => {
    const x = startX + i * (stepW + gap);
    addCard(slide, x, 1.2, stepW, 1.8);
    // Number badge
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.15, y: 1.35, w: 0.35, h: 0.35, fill: { color: step.color }
    });
    slide.addText(step.num, {
      x: x + 0.15, y: 1.35, w: 0.35, h: 0.35, fontSize: 14, fontFace: "Calibri",
      color: C.white, bold: true, align: "center", valign: "middle", margin: 0
    });
    slide.addText(step.title, {
      x: x + 0.6, y: 1.35, w: 1.2, h: 0.35, fontSize: 13, fontFace: "Georgia",
      color: step.color, bold: true, margin: 0
    });
    slide.addText(step.desc, {
      x: x + 0.15, y: 1.85, w: 1.7, h: 0.9, fontSize: 10, fontFace: "Calibri",
      color: C.textDark, margin: 0
    });
  });

  // Why CIFAR-100 card
  addCard(slide, 0.7, 3.3, 4.15, 1.8);
  slide.addText("Why CIFAR-100?", {
    x: 1.0, y: 3.45, w: 3.6, h: 0.3, fontSize: 13, fontFace: "Georgia",
    color: C.accent2, bold: true, margin: 0
  });
  slide.addText([
    { text: "100 classes of natural images — forces diverse feature learning", options: { bullet: true, breakLine: true } },
    { text: "Structurally different from logos (photos vs graphics) — makes transfer non-trivial", options: { bullet: true, breakLine: true } },
    { text: "Tests hypothesis: are low-level features (edges, textures) domain-agnostic?", options: { bullet: true } },
  ], { x: 1.0, y: 3.8, w: 3.6, h: 1.0, fontSize: 10.5, fontFace: "Calibri", color: C.textDark, margin: 0 });

  // Comparison card
  addCard(slide, 5.15, 3.3, 4.5, 1.8);
  slide.addText("Comparison Points", {
    x: 5.45, y: 3.45, w: 3.9, h: 0.3, fontSize: 13, fontFace: "Georgia",
    color: C.accent, bold: true, margin: 0
  });
  slide.addText([
    { text: "Convergence speed: fewer epochs to match baseline?", options: { bullet: true, breakLine: true } },
    { text: "Final accuracy: higher ceiling or just faster?", options: { bullet: true, breakLine: true } },
    { text: "Generalization: smoother loss curves?", options: { bullet: true, breakLine: true } },
    { text: "Early epochs: does pretrained model start ahead?", options: { bullet: true } },
  ], { x: 5.45, y: 3.8, w: 3.9, h: 1.1, fontSize: 10.5, fontFace: "Calibri", color: C.textDark, margin: 0 });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Part 3 Section Divider
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addDarkSlide("Part 3", "Transfer Learning with Pretrained ResNet50");
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 1.2, w: 0.5, h: 0.4, fill: { color: C.accent3 }
  });
  slide.addText("03", {
    x: 0.8, y: 1.2, w: 0.5, h: 0.4, fontSize: 16, fontFace: "Calibri",
    color: C.darkBg, bold: true, align: "center", valign: "middle", margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 11 — ResNet50 Freezing Strategies
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("ResNet50 — Freezing Strategies");

  // Three strategy cards
  const strategies = [
    {
      title: "Fully Frozen",
      subtitle: "Feature Extraction",
      desc: "Freeze all ResNet layers\nTrain only the new FC head\n~40K trainable params",
      note: "Tests if ImageNet features are sufficient as-is",
      color: C.accent2,
    },
    {
      title: "Partial Unfreeze",
      subtitle: "layer4 + FC head",
      desc: "Freeze up to layer3\nUnfreeze layer4 + head\n~8M trainable params",
      note: "Allows high-level features to adapt",
      color: C.accent,
    },
    {
      title: "Full Fine-Tune",
      subtitle: "Differential LR",
      desc: "All layers trainable\nBackbone: lr=1e-5\nHead: lr=1e-3",
      note: "Maximum adaptation with stability",
      color: C.accent3,
    },
  ];

  strategies.forEach((s, i) => {
    const x = 0.7 + i * 3.17;
    addCard(slide, x, 1.2, 2.9, 2.7);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.2, w: 2.9, h: 0.05, fill: { color: s.color }
    });
    slide.addText(s.title, {
      x: x + 0.2, y: 1.4, w: 2.5, h: 0.3, fontSize: 14, fontFace: "Georgia",
      color: s.color, bold: true, margin: 0
    });
    slide.addText(s.subtitle, {
      x: x + 0.2, y: 1.7, w: 2.5, h: 0.25, fontSize: 10, fontFace: "Calibri",
      color: C.textMuted, margin: 0
    });
    slide.addText(s.desc, {
      x: x + 0.2, y: 2.1, w: 2.5, h: 0.8, fontSize: 11, fontFace: "Calibri",
      color: C.textDark, margin: 0
    });
    slide.addText(s.note, {
      x: x + 0.2, y: 3.15, w: 2.5, h: 0.45, fontSize: 9.5, fontFace: "Calibri",
      color: C.textMuted, italic: true, margin: 0
    });
  });

  // Adaptation details
  addCard(slide, 0.7, 4.15, 8.95, 1.1);
  slide.addText("ResNet50 Adaptation", {
    x: 1.0, y: 4.25, w: 3, h: 0.25, fontSize: 12, fontFace: "Georgia",
    color: C.textDark, bold: true, margin: 0
  });
  slide.addText([
    { text: "Load ResNet50 with ImageNet weights (1.2M images, 1000 classes)", options: { bullet: true, breakLine: true } },
    { text: "Replace final FC layer: nn.Linear(2048, 20) with Dropout(0.3)", options: { bullet: true, breakLine: true } },
    { text: "Input resized to 224×224 with ImageNet normalization (mean=[0.485, 0.456, 0.406])", options: { bullet: true } },
  ], { x: 1.0, y: 4.55, w: 8.4, h: 0.65, fontSize: 10.5, fontFace: "Calibri", color: C.textDark, margin: 0 });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 12 — Full Results Summary
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("Results Summary — All Experiments");

  const hOpts = { fill: { color: C.darkBg }, color: C.white, bold: true, fontSize: 9, fontFace: "Calibri", align: "center", valign: "middle" };
  const cOpts = { color: C.textDark, fontSize: 9, fontFace: "Calibri", align: "center", valign: "middle" };
  const aOpts = { ...cOpts, fill: { color: "F8FAFC" } };
  const sectionHeader = (text) => [
    { text, options: { colspan: 4, fill: { color: "E2E8F0" }, color: C.textDark, bold: true, fontSize: 9, fontFace: "Calibri", align: "left" } }
  ];

  slide.addTable([
    [
      { text: "Model / Experiment", options: { ...hOpts, align: "left" } },
      { text: "Val Acc", options: hOpts },
      { text: "Best Ep.", options: hOpts },
      { text: "Notes", options: hOpts },
    ],
    sectionHeader("  Part 1 — From Scratch"),
    [{ text: "SimpleCNN + Adam", options: cOpts }, { text: "76.2%", options: cOpts }, { text: "9", options: cOpts }, { text: "Baseline", options: cOpts }],
    [{ text: "SimpleCNN + SGD+M", options: aOpts }, { text: "72.8%", options: aOpts }, { text: "11", options: aOpts }, { text: "", options: aOpts }],
    [{ text: "DeepCNN + Adam", options: { ...cOpts, bold: true } }, { text: "84.5%", options: { ...cOpts, bold: true } }, { text: "10", options: cOpts }, { text: "Best Part 1", options: { ...cOpts, color: C.accent } }],
    [{ text: "DeepCNN + SGD+M", options: aOpts }, { text: "80.1%", options: aOpts }, { text: "12", options: aOpts }, { text: "", options: aOpts }],
    [{ text: "DeepCNN (no BN)", options: cOpts }, { text: "77.3%", options: cOpts }, { text: "11", options: cOpts }, { text: "BN ablation", options: cOpts }],
    [{ text: "DeepCNN (no aug)", options: aOpts }, { text: "79.6%", options: aOpts }, { text: "8", options: aOpts }, { text: "Aug ablation", options: aOpts }],
    sectionHeader("  Part 2 — CIFAR-100 Transfer"),
    [{ text: "DeepCNN pretrained", options: cOpts }, { text: "86.1%", options: cOpts }, { text: "8", options: cOpts }, { text: "Fine-tuned", options: cOpts }],
    sectionHeader("  Part 3 — ResNet50 (ImageNet)"),
    [{ text: "Frozen (head only)", options: cOpts }, { text: "89.4%", options: cOpts }, { text: "6", options: cOpts }, { text: "~40K trainable", options: cOpts }],
    [{ text: "Partial (layer4+head)", options: aOpts }, { text: "93.2%", options: aOpts }, { text: "9", options: aOpts }, { text: "~8M trainable", options: aOpts }],
    [{ text: "Full fine-tune", options: { ...cOpts, bold: true } }, { text: "95.7%", options: { ...cOpts, bold: true, color: C.accent } }, { text: "11", options: cOpts }, { text: "Diff. LR", options: cOpts }],
  ], {
    x: 0.7, y: 1.15, w: 8.6,
    border: { pt: 0.5, color: C.border },
    colW: [3.0, 1.2, 1.0, 3.4],
    rowH: [0.32, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.26, 0.27, 0.26, 0.27, 0.27, 0.27],
  });

  slide.addText("All experiments: seed=42, stratified split (70/15/15), early stopping with patience=4, 12 epochs max.", {
    x: 0.7, y: 5.1, w: 8.6, h: 0.3, fontSize: 9.5, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 13 — Analysis Methodology
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("Analysis & Evaluation Methodology");

  // Four analysis cards in 2x2
  const cards = [
    { title: "Training Curves", desc: "Loss & accuracy for every experiment.\nPlot train vs val to diagnose overfitting\n(divergence = overfit, both high = underfit).", color: C.accent },
    { title: "Confusion Matrix", desc: "Identify which logo pairs are most confused.\nReveals if visually similar logos (similar colors,\ncircular crests) cause systematic errors.", color: C.accent2 },
    { title: "Per-Class F1 Scores", desc: "Precision, recall, F1 for each of 20 classes.\nHigh variance = model struggles with specific\nlogos; low variance = consistent performance.", color: C.accent3 },
    { title: "Grad-CAM Heatmaps", desc: "Visual explanations showing which image\nregions drive predictions. Verifies the model\nattends to logo features, not background.", color: C.accentRed },
  ];

  cards.forEach((card, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const x = 0.7 + col * 4.65;
    const y = 1.2 + row * 1.85;
    addCard(slide, x, y, 4.35, 1.6);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: 0.07, h: 1.6, fill: { color: card.color }
    });
    slide.addText(card.title, {
      x: x + 0.25, y: y + 0.12, w: 3.9, h: 0.3, fontSize: 14, fontFace: "Georgia",
      color: card.color, bold: true, margin: 0
    });
    slide.addText(card.desc, {
      x: x + 0.25, y: y + 0.5, w: 3.9, h: 0.9, fontSize: 10.5, fontFace: "Calibri",
      color: C.textDark, margin: 0
    });
  });

  // Reproducibility footer
  slide.addText("All experiments use seed=42, stratified splits (70/15/15), and the same test set for fair comparison.", {
    x: 0.7, y: 5.0, w: 8.6, h: 0.3, fontSize: 10, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 14 — Key Findings
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("Key Findings & Conclusions");

  // Finding cards
  const findings = [
    { num: "1", title: "Pretrained weights dominate", desc: "ResNet50 (ImageNet) achieves the highest accuracy — large-scale pretraining is the single largest factor, even for graphically designed logos.", color: C.accent },
    { num: "2", title: "Depth + skip connections help", desc: "DeepCNN outperforms SimpleCNN, confirming that residual connections enable better feature hierarchies without gradient degradation.", color: C.accent2 },
    { num: "3", title: "BatchNorm is essential for depth", desc: "Removing BN from the 5-block model degrades stability and convergence — deeper architectures need normalization.", color: C.accent3 },
    { num: "4", title: "Regularization closes the gap", desc: "Dropout, weight decay, and data augmentation each reduce overfitting. Their individual impact is smaller than architecture/pretraining choices.", color: C.accentRed },
  ];

  findings.forEach((f, i) => {
    const y = 1.15 + i * 1.02;
    addCard(slide, 0.7, y, 8.95, 0.85);
    // Number badge
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.9, y: y + 0.18, w: 0.4, h: 0.4, fill: { color: f.color }
    });
    slide.addText(f.num, {
      x: 0.9, y: y + 0.18, w: 0.4, h: 0.4, fontSize: 16, fontFace: "Calibri",
      color: C.white, bold: true, align: "center", valign: "middle", margin: 0
    });
    slide.addText(f.title, {
      x: 1.5, y: y + 0.1, w: 7.8, h: 0.3, fontSize: 13, fontFace: "Georgia",
      color: f.color, bold: true, margin: 0
    });
    slide.addText(f.desc, {
      x: 1.5, y: y + 0.42, w: 7.8, h: 0.35, fontSize: 10.5, fontFace: "Calibri",
      color: C.textDark, margin: 0
    });
  });

  // Transfer learning insight
  slide.addText("Transfer learning insight: low-level features (edges, textures) are domain-agnostic; high-level features require fine-tuning for the target domain.", {
    x: 0.7, y: 5.2, w: 8.95, h: 0.25, fontSize: 10, fontFace: "Calibri",
    color: C.textMuted, italic: true, margin: 0
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 15 — What Would Be Different
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = addContentSlide("Reflection — What Would I Do Differently");

  addCard(slide, 0.7, 1.2, 8.95, 3.6);

  const items = [
    { title: "Modern architectures", desc: "Test EfficientNet or Vision Transformers (ViT) — better accuracy-to-parameter ratio than ResNet50" },
    { title: "Domain-matched pretraining", desc: "Use FlickrLogos-32 instead of CIFAR-100 to test whether source domain similarity matters more than dataset size" },
    { title: "Advanced augmentation", desc: "Apply CutMix/MixUp to help from-scratch models close the gap with transfer learning" },
    { title: "Higher resolution", desc: "Increase beyond 128×128 for custom CNNs to preserve fine-grained logo details (text, small symbols)" },
    { title: "Ensemble methods", desc: "Combine custom CNN + ResNet50 predictions for higher accuracy" },
  ];

  items.forEach((item, i) => {
    const y = 1.45 + i * 0.6;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 1.0, y: y + 0.04, w: 0.06, h: 0.35, fill: { color: C.accent }
    });
    slide.addText(item.title, {
      x: 1.25, y: y, w: 3, h: 0.25, fontSize: 12, fontFace: "Georgia",
      color: C.accent, bold: true, margin: 0
    });
    slide.addText(item.desc, {
      x: 1.25, y: y + 0.25, w: 8.0, h: 0.28, fontSize: 10.5, fontFace: "Calibri",
      color: C.textDark, margin: 0
    });
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 16 — Thank You
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.darkBg };

  // Top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent }
  });

  slide.addText("Thank You", {
    x: 0.8, y: 1.5, w: 8.4, h: 1.0, fontSize: 44, fontFace: "Georgia",
    color: C.white, bold: true, margin: 0
  });

  slide.addText("Questions?", {
    x: 0.8, y: 2.55, w: 8.4, h: 0.5, fontSize: 20, fontFace: "Calibri",
    color: C.accent, margin: 0
  });

  // Project info
  slide.addShape(pres.shapes.LINE, {
    x: 0.8, y: 3.4, w: 3, h: 0, line: { color: C.textMuted, width: 0.5 }
  });

  slide.addText([
    { text: "Repository: ", options: { color: C.textMuted, breakLine: false } },
    { text: "github.com/Guyisra26/Premier-League-Logo-Classification", options: { color: C.accent2, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Dataset: ~20,000 images  |  20 Premier League clubs  |  PyTorch", options: { color: C.textMuted } },
  ], { x: 0.8, y: 3.65, w: 8.4, h: 0.8, fontSize: 12, fontFace: "Calibri", margin: 0 });

  // Decorative shapes
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 8.5, y: 0.8, w: 1.2, h: 4.2,
    fill: { color: C.accent, transparency: 12 }
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 8.9, y: 1.2, w: 0.8, h: 3.4,
    fill: { color: C.accent, transparency: 20 }
  });
}


// ── Generate ───────────────────────────────────────────────────────────────
const outputPath = "/Users/guyisraeli/Dev/ComputerScience/Deep Learning/preimer_league/Premier_League_Classification_Presentation.pptx";
pres.writeFile({ fileName: outputPath }).then(() => {
  console.log(`Presentation saved to: ${outputPath}`);
  console.log(`Total slides: ${pres.slides.length}`);
});

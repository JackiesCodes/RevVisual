// What a photographic subject is: person, animal, vehicle or product. Each
// cut-out is shown to an ImageNet classifier (MobileNetV2) and a face
// detector; the classifier recognises animals, vehicles and everyday objects
// by name, and people come from a confident face or from clothing classes
// (ImageNet has no "person" class).
import type { DetectedObject } from "../types";
import { loadImage, makeCanvas, release } from "./raster";
import { loadModel, runModel } from "./ml";

interface Classes {
  names: string[];
  vehicle: number[];
  clothing: number[];
  animalUpTo: number;
}

let classesPromise: Promise<Classes> | null = null;
const classes = () => (classesPromise ??= fetch("/models/imagenet-classes.json").then((r) => r.json() as Promise<Classes>));

const ELIGIBLE = new Set(["person", "product", "object", "vehicle", "building", "animal"]);
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const title = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/**
 * Relabels the photographic subjects among `objects` in place, and drops the
 * skin-guessed faces of subjects that turn out not to be people. Objects
 * smaller than 1% of the image, and graphics (logos, icons), are left alone.
 */
export async function classifySubjects(objects: DetectedObject[], imgW: number, imgH: number, sporty: boolean) {
  const subjects = objects.filter(
    (o) => ELIGIBLE.has(o.kind) && o.cutout && (o.box.width * o.box.height) / (imgW * imgH) > 0.01 && o.label !== "Call-to-action button",
  );
  if (!subjects.length) return;
  await Promise.all([loadModel("classify"), loadModel("face")]);
  const cls = await classes();
  const vehicle = new Set(cls.vehicle);
  const clothing = new Set(cls.clothing);
  const notPeople: DetectedObject[] = [];
  for (const o of subjects) {
    const img = await loadImage(o.cutout!);
    const p = await classify(img);
    const face = await faceScore(img);
    let animal = 0;
    let veh = 0;
    let cloth = 0;
    let top = 0;
    let topAnimal = 0;
    let topVehicle = -1;
    for (let i = 0; i < p.length; i++) {
      if (i < cls.animalUpTo) {
        animal += p[i];
        if (p[i] > p[topAnimal]) topAnimal = i;
      }
      if (vehicle.has(i)) {
        veh += p[i];
        if (topVehicle < 0 || p[i] > p[topVehicle]) topVehicle = i;
      }
      if (clothing.has(i)) cloth += p[i];
      if (p[i] > p[top]) top = i;
    }
    const was = o.kind;
    if (animal > 0.6) {
      o.kind = "animal";
      o.label = title(cls.names[topAnimal]);
      o.confidence = round2(animal);
    } else if (veh > 0.4) {
      o.kind = "vehicle";
      o.label = title(cls.names[topVehicle]);
      o.confidence = round2(veh);
    } else if (face > 0.85 || cloth > 0.35) {
      o.kind = "person";
      o.label = sporty ? "Player" : "Person";
      o.confidence = round2(Math.max(face, Math.min(0.9, cloth + 0.4)));
    } else if (was === "person" || was === "object") {
      o.kind = "product";
      o.label = title(cls.names[top]);
      o.confidence = round2(p[top]);
    } else if (was === "product") o.label = title(cls.names[top]);
    if (was === "person" && o.kind !== "person") notPeople.push(o);
    o.source = "ai";
  }
  // Faces were added under subjects guessed to be people from skin tones.
  for (const s of notPeople) {
    const i = objects.findIndex((f) => f.kind === "face" && inside(f, s));
    if (i >= 0) objects.splice(i, 1);
  }
}

/** Class probabilities for a cut-out, letterboxed on grey. */
async function classify(img: HTMLImageElement) {
  const S = 224;
  const { canvas, ctx } = makeCanvas(S, S);
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, S, S);
  const k = S / Math.max(img.naturalWidth, img.naturalHeight);
  const w = img.naturalWidth * k;
  const h = img.naturalHeight * k;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);
  const px = ctx.getImageData(0, 0, S, S).data;
  release(canvas);
  const n = S * S;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) input[c * n + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  const out = await runModel("classify", { input: { data: input, dims: [1, 3, S, S] } });
  const logits = Object.values(out)[0].data;
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  const p = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += p[i] = Math.exp(logits[i] - max);
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return p;
}

/** Highest face score anywhere in the cut-out (0..1). */
async function faceScore(img: HTMLImageElement) {
  const W = 320;
  const H = 240;
  const { canvas, ctx } = makeCanvas(W, H);
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(img, 0, 0, W, H);
  const px = ctx.getImageData(0, 0, W, H).data;
  release(canvas);
  const n = W * H;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) input[c * n + i] = (px[i * 4 + c] - 127) / 128;
  const out = await runModel("face", { input: { data: input, dims: [1, 3, H, W] } }, ["scores"]);
  const s = out.scores.data;
  let best = 0;
  for (let i = 1; i < s.length; i += 2) if (s[i] > best) best = s[i];
  return best;
}

const inside = (a: DetectedObject, b: DetectedObject) =>
  a.box.x >= b.box.x - 2 && a.box.y >= b.box.y - 2 && a.box.x + a.box.width <= b.box.x + b.box.width + 2 && a.box.y + a.box.height <= b.box.y + b.box.height + 2;

const round2 = (v: number) => Math.round(v * 100) / 100;

# ad_for_edu

Audio description for slide-based lecture recordings: detection of the moments that need
describing, generation of candidate descriptions, preference data, post-training, and a
rule-level evaluation metric.

A recorded lecture gives a blind or low-vision student the lecturer's speech but not the
slides, so a deictic reference such as "as you can see here" has no referent. An audio
description supplies one, under three constraints specific to lectures: it must fit the
available pause, avoid restating the lecturer, and name what was pointed at.

The repository holds code and configuration. It contains no recording, frame, transcript or
rating. Commands read the corpus from a local data directory or from the Hub. The dataset
identifiers come from the environment and have no defaults; set them to the published
datasets:

| variable | set to | contents |
|---|---|---|
| `AD_FOR_EDU_REFERENCES_DATASET` | `Hermeneia/AD4Edu-SFT` | reference descriptions, training examples |
| `AD_FOR_EDU_PREFERENCES_DATASET` | `Hermeneia/AD4Edu-Preferences` | preference pairs, the 400 rated by people |
| `AD_FOR_EDU_KEYFRAMES_DATASET` | `Hermeneia/AD4Edu-keyframes` | keyframes |

## Pipeline

Three stages feed the evaluation described under [Metric](#metric). Each is a group of
commands that runs independently.

### Moment detection

Preprocessing yields a transcript, the pauses in the speech, the text on each slide, and
keyframes. Four channels propose candidates: slide turnover, added slide text, pixel motion,
and speech gaps. A vision model classifies each candidate by moment type or rejects it.
Detection is permissive; the accept and reject decision sits in the classifier.

### Description generation

Several models describe each moment or decline to. Each writer receives the slide text, the
keyframes, the transcript window, the pause length, and the descriptions already produced
for earlier moments of the same lecture.

### Preference pairs

Two descriptions of one moment form a pair. Natural pairs are ordered by a judge that
watches the clip in both presentation orders, which separates a preference for a description
from a preference for a position. Controlled pairs differ on one rule category by
construction, so their direction needs no judge. Pairs are used for direct preference
optimisation and to measure whether a scorer distinguishes the two sides.

## Released models and data

All resources are publicly available: the models and datasets on Hugging Face, the code
base in this repository, and the tools of the study with blind and low-vision (BLV)
participants in repositories of their own.

### Models

Describers post-trained from Qwen3-VL. The text-only arm sees the slide text and transcript
but no keyframes; the multimodal arm also sees the keyframes.

<!-- TODO: confirm the arm of the SFT models, and what "r3" stands for. -->

| model | backbone | input arm | method |
|---|---|---|---|
| [`Hermeneia/ad4edu-qwen3vl-2b-sft`](https://huggingface.co/Hermeneia/ad4edu-qwen3vl-2b-sft) | Qwen3-VL 2B | multimodal | SFT |
| [`Hermeneia/ad4edu-qwen3vl-8b-sft`](https://huggingface.co/Hermeneia/ad4edu-qwen3vl-8b-sft) | Qwen3-VL 8B | multimodal | SFT |
| [`Hermeneia/ad4edu-qwen3vl-2b-r3-dpo`](https://huggingface.co/Hermeneia/ad4edu-qwen3vl-2b-r3-dpo) | Qwen3-VL 2B | multimodal | DPO (r3) |
| [`Hermeneia/ad4edu-qwen3vl-2b-textonly-r3-dpo`](https://huggingface.co/Hermeneia/ad4edu-qwen3vl-2b-textonly-r3-dpo) | Qwen3-VL 2B | text-only | DPO (r3) |
| [`Hermeneia/ad4edu-qwen3vl-8b-textonly-r3-dpo`](https://huggingface.co/Hermeneia/ad4edu-qwen3vl-8b-textonly-r3-dpo) | Qwen3-VL 8B | text-only | DPO (r3) |

### Datasets

| dataset | contents |
|---|---|
| [`Hermeneia/AD4Edu-SFT`](https://huggingface.co/datasets/Hermeneia/AD4Edu-SFT) | reference descriptions, training examples |
| [`Hermeneia/AD4Edu-Preferences`](https://huggingface.co/datasets/Hermeneia/AD4Edu-Preferences) | preference pairs, including the 400 rated by people |
| [`Hermeneia/AD4Edu-keyframes`](https://huggingface.co/datasets/Hermeneia/AD4Edu-keyframes) | keyframes |

### Related repositories

| repository | contents |
|---|---|
| [`psychias/ad_for_edu`](https://github.com/psychias/ad_for_edu) | this code base |
| [`psychias/blv_annotation_tool`](https://github.com/psychias/blv_annotation_tool) | screen-reader-native annotation tool used in the BLV study |
| [`psychias/blv_mediaplayer`](https://github.com/psychias/blv_mediaplayer) | lecture player that delivers descriptions at their moments |

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/psychias/ad_for_edu.git
cd ad_for_edu
python -m venv .venv && . .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

The base install carries `pyyaml`, `numpy` and `scipy`. Heavier dependencies are extras, so
a machine that only scores predictions needs no training stack:

| extra | dependencies for |
|---|---|
| `preprocessing` | audio, transcription, slide text, keyframes (requires `ffmpeg` on the path) |
| `llm` | hosted model providers |
| `metrics` | local encoder and inference models, BERTScore, CLIPScore |
| `train` | training and inference on a describer |
| `data` | reading the datasets from the Hub |
| `dev` | tests and linter |

Copy `.env.example` to `.env` and complete it. No variable has a default; a command that
requires an unset variable exits with the variable named.

```bash
ad-for-edu --help
ad-for-edu check-standard        # parse the rule book, render every rule prompt
ad-for-edu list-strategies       # interchangeable components, by family
```

## Usage

Thirty-two commands behind one entry point; each accepts `--help`. Two entry paths: from
recordings, which requires the lecture files, and from the published datasets, which starts
at training or evaluation.

### From recordings

One lecture at a time. Commands that call a hosted model print a cost estimate and exit
without calling; the sequence below therefore costs nothing as written.

```bash
ad-for-edu preprocess          --lectures my-lecture
ad-for-edu probe-cursor        --lectures my-lecture
ad-for-edu detect-moments      --lectures my-lecture
ad-for-edu classify-moments    --events work/moments/events/candidates.jsonl
ad-for-edu generate-references --moments work/moments/classified/moments.jsonl
ad-for-edu generate-candidates --moments work/moments/classified/moments.jsonl
ad-for-edu build-pairs         --candidates work/pairs/candidates/candidates.jsonl
ad-for-edu judge-pairs         --pairs work/pairs/sets/pairs.jsonl
```

### Training and evaluation

```bash
ad-for-edu build-training-examples
ad-for-edu train    --backbone qwen3-vl-8b --arm multimodal --method sft --seed 0
ad-for-edu describe --moments moments.jsonl --system adapter --run-dir work/runs/...
ad-for-edu score-systems --predictions work/evaluation/predictions_*.jsonl \
                         --moments moments.jsonl \
                         --metrics chrf compliance:mechanical compliance:local
```

`ad-for-edu evaluate` runs every table the supplied inputs permit and writes one directory:

```bash
ad-for-edu evaluate --predictions work/evaluation/predictions_*.jsonl \
                    --moments moments.jsonl \
                    --pairs pairs.jsonl --scores scores.jsonl \
                    --labels r1.jsonl r2.jsonl r3.jsonl r4.jsonl \
                    --offline
```

Its summary records, per stage, whether the stage ran and what it wrote, and for each stage
it skipped, the input that was absent. Without `--offline` it includes the judge-based
stages, totals them into a single estimate, and exits until `--approve-spend` is passed.

### Cost control

Every command that can call a hosted model prints the estimated cost per row and per model,
then exits:

```
$ ad-for-edu judge-pairs --pairs pairs.jsonl

SPEND ESTIMATE -- stage judge_pairs

  row                        model                   calls   $/call     total
  both orders                gemini-3.1-pro-or         840 $ 0.0120 $   10.08
  TOTAL                                                840          $   10.08

  no --approve-spend: nothing was called. Re-run with the flag to spend.
```

`--approve-spend` permits the calls. The constraint is structural: `LLMClient.__init__`
requires an approval token, and `require_approval` is the only source of one.

## Metric

AD4Edu-Eval scores a description against a 45-rule standard across six categories: style,
terminology, length, deixis, faithfulness, non-redundancy. The score is the unweighted mean
over the categories applicable to the moment, so an inapplicable category abstains rather
than contributing zero. A within-lecture novelty factor multiplies the mean and reduces a
description that repeats an earlier one from the same system. Three modes, reported
separately:

| mode | scored by | categories |
|---|---|---|
| mechanical | rules only, offline | style, terminology, length, deixis |
| local | adds two local models | all six |
| rubric | hosted judge against the rule book | all six |

Alongside it: reference overlap, image-text alignment, head-to-head win rates, per-category
defect localisation, and agreement between each metric and each rater.

### Rule diagnostics

`ad-for-edu diagnose` attributes a score to rule identifiers. For each rule it reports the
number of descriptions the rule applies to, the number that broke it, and the lowest-scoring
examples. Denominators are per rule: three of four figures is not three of four hundred
descriptions.

Breaches and shortfalls occupy separate columns. Three of the six components count
violations, so any score below 1 indicates at least one. The other three return a graded
share of a property that a compliant description may lack: naming something absent from the
slide text scores below 1 on terminology without breaking a rule. Reported as one column,
that puts a reference writer at 96% broken.

A mode reports only the rules its own components tested. Categories a mode abstains on are
listed as untested, not as unbroken. The standard's routing differs from the modes in the
same way: of the fourteen rules it marks mechanically checkable, five are scored by the
mechanical mode, seven require a learned model and are reachable only in the local mode, and
two have no implementation here. The diagnostic distinguishes all three groups.

### Field mapping

The evaluation requires, per moment, an identifier, a lecture, a time, a type, and the
on-screen and spoken text; and per description, an identifier and the text. Any dataset
supplying these can be scored by naming its columns:

```bash
ad-for-edu diagnose --predictions theirs.jsonl --moments theirs_moments.jsonl \
    --fields moment_id=id lecture=course time=start type=kind \
             slide_text=ocr on_screen=screen transcript_window=said text=narration
```

`--field-map a.yaml` supplies the same mapping from a file. `ad-for-edu evaluate` forwards
whichever form it receives to every stage. Unmapped columns pass through unchanged.

`lecture` is required. The novelty factor groups a system's descriptions by lecture; if each
moment maps to a distinct lecture, no description can repeat another and the factor has no
effect, while the scores stay in range. The reader verifies that the lecture field groups
the moments and fails with the field name when it does not.

## Architecture

Each step with more than one reasonable implementation is a strategy: an abstract base, a
registry beside it, and one registered class per implementation. A settings file names the
implementation, a `build.py` resolves the name to an object, and the consuming code receives
that object through its constructor. Changing a detector, a judge, a training method or a
metric component is a settings change.

```yaml
# configs/detection.yaml
frame_channels:
  - name: slide_turnover
    params: {overlap_below: 0.55}
  - name: added_words
    params: {at_least: 3}
  - name: pixel_motion
    params: {difference_at_least: 0.06, signature_size: 64}
gap_channel: speech_gap
```

An unregistered name fails at start-up and lists the registered ones. An unrecognised
settings key fails and lists the recognised ones. Both fail before any work begins.

`ad-for-edu list-strategies` prints every family:

| group | families |
|---|---|
| moment detection | frame channel, gap channel, still policy, transcript window, rung policy |
| media preparation | transcriber, speech gap detector, slide text reader |
| model access | model provider, decoding attempt, salvage step, reference prompt, candidate mode |
| pairs | controlled axis, order combination, pair decision rule |
| judging | judge |
| metric | compliance component, sequence factor, metric |
| training | training method, input arm, description system |
| statistics | interval estimator, paired test, multiple-test correction, agreement coefficient |
| data | dataset source, split policy, context source |

### Package layout

```
src/ad_for_edu/
  core/           registries, settings, identifiers, timecodes, spend gate, run records
  data/           row schema, dataset sources, lecture manifest, split policy, field mapping
  standard/       the 45-rule book, rule applicability, per-rule prompts
  prompts/        prompt texts, placeholders, pre-send binding checks
  llm/            requests, replies, client, per-stage pricing, three providers
  preprocessing/  audio, transcription, speech gaps, slide text, keyframes, cursor probe
  moments/        detection channels, detector, stills, windows, rungs, classification
  references/     per-writer description generation
  pairs/          candidates, controlled axes, pairing, set draws, presentation orders
  judges/         judge forms and the loop that runs them
  compliance/     six components, modes, scorer, novelty factor, rule attribution
  metrics/        overlap, alignment, stored judge answers, compliance
  training/       examples, input arms, backbones, split, SFT and DPO
  inference/      decoding attempts, reply salvage, prediction rows
  systems/        adapters, untrained model, slide-title readout, reference writers
  stats/          intervals, paired tests, corrections, agreement, power, rank correlation
  evaluation/     tables and comparisons
  agreement/      rater-metric agreement, fitted preference head
  cli/            one module per command group
```

## Privacy

Raters are study participants. Reports identify them as R1, R2 and so on, and the
rater-agreement commands reject a label file whose name is not a pseudonym; `--raters`
supplies pseudonyms explicitly. Names, ages and fields of study are not published, and the
ignore rules match by filename rather than by location.

## Licence

Apache-2.0; see `LICENSE`. The rule book quotes short attributed passages from published
accessibility guidelines; see `NOTICE`.

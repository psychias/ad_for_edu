# ad_for_edu

Audio description for slide-based lecture recordings.

A blind or low-vision student following a recorded lecture hears the lecturer but not the
slides. When the lecturer says "as you can see here", the sentence is about something the
student cannot reach. Audio description fills those moments with speech, and a lecture
recording makes that hard in a specific way: the description has to fit in a pause, must
not repeat what the lecturer just said, and has to resolve what "here" pointed at.

This repository is the pipeline for building and measuring such descriptions. It finds the
moments of a recording that need one, collects descriptions of them, pairs descriptions
against each other to learn which is preferred, trains a model to write them, and scores
the result against an explicit rule book rather than against a single opaque number.

It contains code and configuration only. No recording, frame, transcript or rating is in
here: the corpus stays where it is licensed to be, and the commands read it from a data
directory or from a dataset on the Hugging Face Hub. The published sets are
`Psychias/AD4Edu-SFT` for the reference descriptions and training examples,
`Psychias/AD4Edu-Preferences` for the pairs and the four hundred rated by people, and
`Psychias/AD4Edu-keyframes` for the frames. Which set a command reads comes from the
environment and has no default, so nothing is fetched that was not asked for.

## What it does

The pipeline has four parts, and each is a group of commands that can be run on its own.

**Finding the moments.** A lecture recording is turned into a transcript, a list of the
pauses in the speech, the text on each slide and a set of keyframes. Four detectors then
propose moments: the slide changed, words appeared on the slide, the picture moved, the
lecturer stopped talking. A vision model reads the keyframes around each proposal and
either says what kind of moment it is or rejects it. Proposing generously and rejecting
deliberately keeps the decision in one place, where it can be inspected.

**Collecting descriptions.** Several models are asked to describe each moment, or to say
that it needs no description. A writer sees the slide, the keyframes, the words spoken
around the moment, the length of the pause it must fit, and what has already been
described in that lecture, so that it does not say the same thing twice.

**Preference pairs.** Two descriptions of one moment make a pair. A pair is ordered either
by a judge that watches the clip, in both presentation orders so that a preference for a
description is not a preference for a position, or by construction: a *controlled* pair is
written to differ on exactly one rule category, so the better side is known without asking.
The pairs train a model by direct preference optimisation and, separately, measure whether
a scorer can tell the two sides apart.

**Measuring.** The metric, **AD4Edu-Eval**, scores a description against a rule book of 45
rules in six categories: style, terminology, length, deixis, faithfulness and
non-redundancy. A score is the mean over the categories that apply to that moment, so a
category that cannot apply abstains rather than scoring zero. The mean is multiplied by a
novelty factor, which cuts a description that repeats an earlier one in the same lecture.
The metric has three modes, reported apart and never mixed:

| mode | what scores it | categories |
|---|---|---|
| mechanical | rules alone, offline | style, terminology, length, deixis |
| local | adds two local models | all six |
| rubric | a hosted judge grades against the rule book | all six |

Beside it the repository computes overlap with references, image–text alignment, win rates
between two systems, per-category defect localisation, and how far each metric agrees with
the people who rated pairs by hand.

### Which rule a description breaks

A score on its own says a description is worse without saying what is wrong with it.
`ad-for-edu diagnose` answers that: per rule identifier, how many descriptions the rule
applies to, how many broke it, and the worst offenders. The denominator is the rule's own,
because a rule broken on three of four figures is a different finding from three of four
hundred descriptions.

Two columns, not one. Three of the six components count violations, so anything below full
credit means one was found. The other three return a graded share of something a good
description does anyway: one that names what the slide does not spell out scores below one
on terminology and has broken no rule. The report keeps those apart, since merging them
puts a reference writer at ninety-six per cent broken.

It reports only the rules its mode tested. A mode that abstains on a category never
checked that category's rules, so they are named as untested rather than listed as
unbroken. The same holds for the standard's own routing: of the fourteen rules it marks
mechanically checkable, five are scored in the mechanical mode, seven need a learned model
and so are reachable only in the local mode, and two have no check here at all. The
diagnostic names all three groups.

### Scoring someone else's dataset

The evaluation needs, per moment, an identifier, the lecture it belongs to, a time, a
type, and the text on screen and spoken nearby; and per description, an identifier and the
text. Any dataset that can supply those can be scored. Name its columns and nothing else
changes:

```bash
ad-for-edu diagnose --predictions theirs.jsonl --moments theirs_moments.jsonl \
    --fields moment_id=id lecture=course time=start type=kind \
             slide_text=ocr on_screen=screen transcript_window=said text=narration
```

`--field-map a.yaml` takes the same mapping from a file, and `ad-for-edu evaluate` passes
whichever you give to every stage, so one dataset is read one way throughout. Columns the
mapping does not name are carried through untouched.

One of those fields is load-bearing. The novelty term groups a system's descriptions by
lecture, and with no lecture every moment is its own, nothing can be seen to repeat, and
the term silently stops working while the scores stay plausible. So the lecture is checked
rather than assumed: a dataset whose lecture cannot be found is refused, with the column to
set named in the message.

## Installing

Python 3.10 or later.

```bash
git clone <this repository>
cd ad_for_edu
python -m venv .venv && . .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

The core install is small. Each heavy part is an extra, so a machine that only scores
predictions does not need a training stack:

| extra | for |
|---|---|
| `preprocessing` | audio, transcription, slide text, keyframes (also needs `ffmpeg` on the path) |
| `llm` | calling hosted models |
| `metrics` | the local encoder and inference models, BERTScore, CLIPScore |
| `train` | training and running a describer |
| `data` | reading the datasets from the Hub |
| `dev` | the tests and the linter |

Then copy `.env.example` to `.env` and fill it in. Nothing in it has a default: a command
that needs a variable and does not find it stops with a message naming the variable, rather
than reading somewhere unintended.

```bash
ad-for-edu --help
ad-for-edu check-standard        # reads the rule book and renders every rule prompt
ad-for-edu list-strategies       # the interchangeable parts, by family
```

## Running it

Two ways in. **From recordings** runs the whole chain and needs the lecture files. **From
the datasets** starts at training or evaluation and needs no recording.

From recordings, one lecture at a time. The steps that call a hosted model print an
estimate and stop until `--approve-spend` is added, so this sequence run as it stands costs
nothing and tells you what each step would cost:

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

From the datasets, to train and score:

```bash
ad-for-edu build-training-examples
ad-for-edu train    --backbone qwen3-vl-8b --arm multimodal --method sft --seed 0
ad-for-edu describe --moments moments.jsonl --system adapter --run-dir work/runs/...
ad-for-edu score-systems --predictions work/evaluation/predictions_*.jsonl \
                         --moments moments.jsonl \
                         --metrics chrf compliance:mechanical compliance:local
```

Or the whole evaluation in one invocation, which runs every table the inputs allow and
writes one directory:

```bash
ad-for-edu evaluate --predictions work/evaluation/predictions_*.jsonl \
                    --moments moments.jsonl \
                    --pairs pairs.jsonl --scores scores.jsonl \
                    --labels r1.jsonl r2.jsonl r3.jsonl r4.jsonl \
                    --offline
```

It reports, per stage, whether it ran and what it wrote, and names the input any stage it
skipped was missing, so a half-filled directory cannot be mistaken for a finished
evaluation. Drop `--offline` to include the stages that call a judge; it then totals them
into one estimate and stops until `--approve-spend` is given.

`ad-for-edu --help` lists all thirty-two commands; each takes `--help` of its own.

### Nothing is charged without being asked

Every command that can call a hosted model prints what the work would cost, per row and
per model, and stops:

```
$ ad-for-edu judge-pairs --pairs pairs.jsonl

SPEND ESTIMATE -- stage judge_pairs

  row                        model                   calls   $/call     total
  both orders                gemini-3.1-pro-or         840 $ 0.0120 $   10.08
  TOTAL                                                840          $   10.08

  no --approve-spend: nothing was called. Re-run with the flag to spend.
```

Passing `--approve-spend` makes the calls. This is not a convention each command keeps: the
object that can make a call cannot be constructed without the token that asking produces,
so a command that forgot to ask could not spend anything.

## How it is put together

Every step that has more than one reasonable implementation is a **strategy**: an abstract
base with a registry beside it, and one registered class per way of doing it. A settings
file names the one to use, a `build.py` turns that name into an object, and the code that
uses it receives it in its constructor and never looks one up. So changing a detector, a
judge, a training method or a component of the metric is a line of YAML.

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

An unknown name fails at start-up and lists the names that exist. A misspelled settings key
fails and lists the keys that exist. Neither is discovered halfway through a run.

`ad-for-edu list-strategies` prints all of them. The families:

| group | families |
|---|---|
| finding moments | frame channel, gap channel, still policy, transcript window, rung policy |
| preparing media | transcriber, speech gap detector, slide text reader |
| asking a model | model provider, decoding attempt, salvage step, reference prompt, candidate mode |
| pairs | controlled axis, order combination, pair decision rule |
| judging | judge |
| the metric | compliance component, sequence factor, metric |
| training | training method, input arm, description system |
| statistics | interval estimator, paired test, multiple-test correction, agreement coefficient |
| data | dataset source, split policy, context source |

### Layout

```
src/ad_for_edu/
  core/           registries, settings, identifiers, timecodes, the spend gate, run records
  data/           row schema, dataset sources, the lecture manifest, the split, media layout
  standard/       the 45-rule book, which rules apply where, one prompt per rule
  prompts/        the prompt texts, their placeholders, and what must bind before sending
  llm/            requests, replies, the client, per-stage pricing, three providers
  preprocessing/  audio, transcription, speech gaps, slide text, keyframes, cursor probe
  moments/        detection channels, the detector, stills, windows, rungs, classification
  references/     asking each writer to describe a moment or stay silent
  pairs/          candidates, controlled axes, pairing, drawing sets, presentation orders
  judges/         the ways a judge is asked, and the loop that asks
  compliance/     the six components, the modes, the scorer, the novelty factor
  metrics/        overlap, alignment, a judge's stored answers, compliance
  training/       examples, input arms, backbones, the split, the two methods
  inference/      how a describer is asked and how a reply is read
  systems/        what a table compares: adapters, the untrained model, a readout, the writers
  stats/          intervals, paired tests, corrections, agreement, power, rank correlation
  evaluation/     the tables and the comparisons
  agreement/      the raters against the metrics, and a small head fitted to their choices
  cli/            one module per group of commands
```


## Licence

Apache-2.0, see `LICENSE`. The rule book quotes short attributed passages from published
accessibility guidelines; see `NOTICE`.

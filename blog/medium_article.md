# Teaching a Neural Network to Hear a Failing Bearing

### How I built an end-to-end predictive-maintenance pipeline — and why hitting 100% accuracy made me trust it *less*

---

Walk through any refinery, power plant, or pump station and you're surrounded by
machines that spin: motors, compressors, turbines, gearboxes. Almost all of them
ride on **bearings** — and when a bearing starts to fail, it doesn't send a calendar
invite. A cracked race or a spalled ball can take a multimillion-dollar compressor
offline in minutes, halt a production line, or turn into a genuine safety incident.

The good news: a failing bearing **whispers before it screams**. Long before a human
can hear or feel anything, the vibration signature changes in tiny, repeating ways.
**Predictive maintenance (PdM)** is the art of listening for those whispers. This is
the story of building a system that does exactly that — and of a plot twist that
taught me more than the model did.

> *This article is the narrated version of a project on GitHub:
> [github.com/godot107/predictive-maintenance-cwru](https://github.com/godot107/predictive-maintenance-cwru).
> Everything here is reproducible from that repo.*

---

## The data: four bearings, four fates

I used the **Case Western Reserve University (CWRU) Bearing Dataset**, a benchmark in
the condition-monitoring world. An accelerometer sampled at 12 kHz records the
vibration of a motor's drive-end bearing under four conditions:

- **Normal** (healthy)
- **Inner Race** fault
- **Ball** fault
- **Outer Race** fault

The goal: feed the model a slice of raw vibration and have it name the fault.

But you can't just throw a wiggly line at a neural network and hope. The signal has
to be transformed into something a network can *see*. That's where two analogies make
everything click.

---

## Two analogies that make signal processing click

### 🥤 The Fourier Transform is a smoothie un-blender

A raw vibration signal is like a **smoothie**: strawberry, banana, and spinach all
blended into one. Looking at the smoothie, you can't tell what went in.

The **Fast Fourier Transform (FFT)** is a magic blender run in reverse. Pour in the
smoothie and it hands back the ingredients — *"30% strawberry, 50% banana, 20%
spinach."* For a bearing, those ingredients are **frequencies**, and a fault adds a
specific new "flavour" that shouldn't be there.

Average the FFT across many windows and the fingerprints jump out — the faults light
up high-frequency bands the healthy bearing never touches:

![Class-averaged FFT spectrum](https://raw.githubusercontent.com/godot107/predictive-maintenance-cwru/main/reports/eda_avg_fft.png)

### 🎼 A spectrogram is sheet music

The FFT has a blind spot: it tells you *which* notes were played, but not *when*.
Imagine being told a song contains a C, an E, and a G — but not the rhythm. You'd
never recognize the tune.

A **spectrogram** puts the notes back on a timeline. It's sheet music: time runs left
to right, frequency runs bottom to top, brightness is loudness. Now a fault isn't a
single frequency — it's a *pattern of impacts repeating over time*, exactly the kind
of 2-D structure a **Convolutional Neural Network** (the tech that recognizes cats in
photos) is built to spot.

So the pipeline became:

```
Raw vibration  ─▶  Spectrogram  ─▶  2-D CNN (on GPU)  ─▶  Fault diagnosis
```

I wired it up in PyTorch, trained on an NVIDIA GPU, and wrapped it in a Streamlit
dashboard that shows the raw signal, the FFT, the spectrogram, and the model's live
verdict. It worked. The test accuracy came back at **100%**.

And that's where the project got interesting.

---

## The plot twist: when 100% is a red flag

A perfect score should make you *suspicious*, not proud. Real-world classifiers don't
hit 100% — so either the problem is trivially easy, or something is leaking.

Classic overfitting looks like **high training accuracy, low test accuracy** — a gap.
But here training *and* test were both ~100%, with no gap. That pointed at the second
culprit: **data leakage**.

Here was the bug in my evaluation. Each fault class is one long continuous recording.
I chopped it into overlapping windows and then split those windows **randomly** into
train and test. Because the windows overlapped, near-duplicate slices ended up on
*both* sides of the split. The model wasn't generalizing — it was recognizing windows
it had half-seen in training. The test set was lying to me.

So I rebuilt the evaluation the way it should have been done from the start:

1. **Leakage-free splitting** — cut each *raw recording* into train/validation/test
   spans **by time** (with a guard gap) *before* windowing, so no slice is shared.
2. **A validation set + early stopping** — stop on validation loss, report on a test
   set the model never touched.
3. **5-fold cross-validation** — for a stable estimate instead of one lucky split.

The payoff was immediate and humbling. With honest splitting, the early training
epochs now showed **training accuracy at 1.00 while validation sat at 0.25 — random
chance.** That genuine overfitting had been completely *hidden* by the leaky split.
The methodology upgrade was doing real work.

---

## The deeper twist: it was *still* 100%

Here's what I didn't expect. Even after removing every trace of leakage, and across
all five cross-validation folds, the score held: **1.000 ± 0.000**.

So leakage wasn't the main story. To understand why, I did what I should have done
*first*: I stopped modeling and **explored the data**.

I computed a set of classic vibration features (RMS, **kurtosis**, crest factor — all
interpretable measures of how "impulsive" a signal is) and projected them to two
dimensions with t-SNE. The picture explains everything:

![t-SNE of engineered features](https://raw.githubusercontent.com/godot107/predictive-maintenance-cwru/main/reports/eda_tsne.png)

Four clean, perfectly separated clusters. At a *single operating condition* — one
load, one speed — each fault's signature is so distinct that the classes barely
overlap. To prove the point, I trained a plain **Random Forest** on those hand-crafted
features (no deep learning at all). It also scored **1.000**.

The lesson: **100% wasn't a triumph of my model — it was a sign the benchmark was
easy.** And knowing the difference is the whole job.

---

## Asking the bearing to confirm its own diagnosis

There was one more thing I wanted: proof the model was learning *real physics*, not an
artifact. Bearing engineering gives us a beautiful test.

The CWRU drive-end bearing is an **SKF 6205**, and its geometry fixes the exact
frequencies at which each defect "rings" — the ball-pass frequencies of the outer race
(BPFO), inner race (BPFI), and so on, all set by the shaft speed. Using **envelope
analysis** (the demodulation technique vibration engineers actually use), I extracted
the impact-repetition rate from each fault and checked it against theory:

![Envelope spectra vs SKF-6205 characteristic frequencies](https://raw.githubusercontent.com/godot107/predictive-maintenance-cwru/main/reports/eda_envelope.png)

The peaks land exactly where the physics predicts: the **Outer Race** fault rings at
**BPFO (107 Hz)**, the **Inner Race** fault at **BPFI (162 Hz)**. The signal carries
genuine, explainable diagnostic content — and, honestly, the **Ball** fault is the
subtle one, with a weak envelope signature, which is precisely what bearing theory
says to expect. Even the exceptions agreed with the textbook.

---

## What this project actually demonstrates

The deliverable people *see* is a dashboard that classifies bearing faults. The
deliverable that matters is the **judgment** around it:

- **Engineering it end-to-end** — data ingestion, signal processing, a GPU-trained
  CNN, and an explainable UI.
- **Distrusting a good result** until the evaluation earns that trust — and knowing
  that leakage, not the model, is usually the thing to interrogate first.
- **Explaining it in plain language** — to a recruiter with a smoothie, to an engineer
  with an envelope spectrum.

That last part is the heart of an **AI Solutioning Consultant's** job: translating
between the business problem ("don't let the compressor fail") and the technical
reality ("here's what the data can and can't tell us, and here's how I know").

The honest next step — the genuinely *hard* benchmark — is **cross-load
generalization**: train on some motor loads and test on a load the model has never
seen. That's where accuracy drops to a realistic number and the real engineering
begins. It's the next milestone in the repo.

---

## Try it yourself

- **Code & full write-up:** [github.com/godot107/predictive-maintenance-cwru](https://github.com/godot107/predictive-maintenance-cwru)
- **Dataset:** [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- **The best 20 minutes on Fourier transforms:** [3Blue1Brown](https://www.youtube.com/watch?v=spUNpyF58BY)

If you take one thing from this: when your model scores 100%, don't celebrate —
*investigate*. The most valuable result in this whole project was the one that looked
too good to be true.

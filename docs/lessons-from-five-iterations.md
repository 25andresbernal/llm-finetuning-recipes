# Lessons from five iterations

This is a generalized account of five fine-tuning passes I ran on
`Llama-3-8B` with Unsloth QLoRA on a GCP L4, training a model to handle one
side of a live, multi-turn phone conversation for a voice agent product. No
client, company, or personal names appear below, and none of the training
data described here is included in this repository; the recipe and the
config in this repo are the version that came out the other side, applied
to a fresh, synthetic, industry-neutral dataset.

I'm writing it in first person because it happened to me, in that order,
and because the mistakes are more useful than the final config on its own.
The config tells you what to set. This tells you why, so you can recognize
the failure mode if you hit a different version of the same problem.

## v1: overfit and NaN

The first pass used a small, hand-written dataset with no dropout on the
LoRA adapters and no weight decay. Training loss dropped fast, which felt
like a win, and then a few epochs in, eval loss started climbing while
train loss kept falling: textbook overfitting on a small dataset with
nothing pulling the adapter back toward the base model's behavior. On a
later run with a slightly different data mix, loss went to `NaN` partway
through training. Chasing that down meant learning that `bnb` 4-bit base
weights plus a learning rate tuned for a different, larger dataset can
destabilize a training run when the effective batch size and warmup don't
match the data volume. The fix for both problems was the same shape:
regularize. Add `lora_dropout`, add `weight_decay`, and stop treating "loss
went down" as the only signal that mattered. `docs/cost-and-gpu-notes.md`
has the config that resulted.

## v2: it worked, but only here

The second pass fixed the stability problems and produced a model that
handled the conversations in its training set well. It did not generalize
past them. The dataset had been written around one narrow scenario shape,
and the model had learned that shape specifically rather than the
underlying skill of "ask one thing, wait, respond to what you're told." Any
conversation that deviated even slightly from the training examples' exact
structure produced a worse response. This is the point where I started
treating dataset diversity, not just dataset size, as a variable to
actually track.

## v3: passed tests, monologued on calls

The third pass added a real, if informal, test set and the model passed
it. It also, on live calls, would answer a caller's one sentence with four
or five sentences back: a restatement, a clarifying question, a proposed
time, and a summary, all in one turn. The isolated tests didn't catch this
because each test case checked whether the response was reasonable in
isolation, not whether the response was the right length and shape for a
back-and-forth conversation. A test suite built around single-response
quality will pass a model that is quietly bad at turn-taking, because
turn-taking is a property of a sequence of responses, not any one of them.

## v4: shorter targets, still synthetic, still scripted

The fourth pass addressed the symptom directly: it shortened every target
response in training. That helped the length problem somewhat, but the
model still felt scripted and stiff on calls, because the underlying data
was still single-turn synthetic pairs: one invented caller line in, one
invented agent line out, with no real conversational history behind either
side. Shortening the output without changing the shape of the input fixed
one symptom and left the root cause in place. The model had never seen an
example where the right answer depended on three prior turns of context,
because no training example had three prior turns of context.

## v5: multi-turn, and the fix held

The fifth pass changed the data format itself: every training example
became one agent turn, with the full conversation history up to that point
as context, drawn from real multi-turn call transcripts instead of
one-off synthetic pairs. Response length stayed short (the v4 fix), but
now the shortness came from the shape of the data rather than a
post-hoc constraint, because a single short turn embedded in real
back-and-forth history is what a real conversation actually looks like.
This is the shape this repository's synthetic dataset generator
(`llm_finetuning_recipes/data_gen.py`) reproduces: one example per agent
turn, full history as context, a short single-clause target. It is not the
same dataset, and it never touches real call data, but it is built to
reproduce the same lesson: a model trained on isolated pairs learns to
write monologues, and a model trained on turns embedded in real
conversation history learns to take turns.

## What carried over into this repo

- The training config in `recipes/conversational-agent-qlora/config.yaml`
  (`r=16`, `alpha=16`, dropout 0.1, weight decay 0.01, cosine schedule,
  effective batch 8, 5 epochs) is the config that stopped producing NaN
  losses and stopped overfitting.
- The dataset shape (`scripts/prepare_data.py`, one example per agent turn,
  full history as context, 5 to 25 word single-clause targets) is the shape
  that stopped the monologuing.
- `scripts/check_dataset.py` exists because v3's isolated test suite missed
  a real problem. A dataset checker that looks at target length and clause
  structure, not just format, would have caught the shape of v1 through v4's
  data before any GPU time was spent on it.
- `scripts/eval_responses.py` scores exactly the properties that failed in
  v1 through v4: length, single-clause turn shape, and premature closes.
  None of it requires a live call or an LLM judge to check.

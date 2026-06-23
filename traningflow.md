# SC Training Data-Flow — one real example (merged mode)

Model: base Fast_dVLM_3B · dataset: `coda_lm`[0] · seed 0 · `noise_mode=merged`, `include_mask_errors=True`, `alpha_inject=0.1`, `alpha_sample=0.2`.

## 1. Raw example

- **#images:** 1  ·  **pixel_values:** `(924, 1176)`  ·  **image_grid_thw:** `[[1, 22, 42]]`
- **Q (prompt):** Front camera: 
frame 0 
There is an image of traffic photographed from the view of the ego car. Pay close attention to the objects that influence the driving behavior of the ego car: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no - parking, warning, and directional signs), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous objects (debris, dustbins, animals, etc.). Based on the current scene, please provide driving recommendations for the ego car. You are not allowed to discuss any objects outside t
- **A (GT answer):** Maintain a safe following distance from the red truck, monitor the bus's actions closely and be prepared for pedestrians, especially around the bus stop area. Keep in the appropriate lane as indicated by the overhead directional signs. Due to wet road conditions, reduce speed appropriately to maintain safe handling and stopping distances.

## 2. Tokenize → batch (B=1)

- `input_ids`/`labels`/`attention_mask`: `[1, 447]` (after trailing-pad crop).
- prompt = positions `0..384` (labels = -100, never scored); answer = positions `385..446` (62 tokens).
- `labels` = `input_ids` on the answer span, `-100` elsewhere (so the answer is the only thing scored). Read convention is next-token/shift: the prediction for position `p` is `logits[p-1]`.

## 3. forward-1 mask (merged → random ratio)

- per-example mask ratio drawn ~U(eps,1) = **0.400**; masked **23/62** answer tokens (absorbing `|<MASK>|`).
- `xt` answer span (masked tokens shown as `[M]`):

> Maint ain a safe following [M] from the [M] truck , monitor the bus 's [M] [M] [M] [M] prepared [M] [M] , [M] around [M] bus [M] area [M] Keep in [M] appropriate lane [M] indicated [M] [M] overhead [M] signs [M] Due to wet road [M] [M] reduce [M] [M] to maintain safe handling and stopping [M] . <|im_end|> ·

## 4–7. Per-answer-position trace

Columns: **GT** = target token · **f1** = forward-1 argmax (model's denoise prediction) · **M2T** = `-logp(GT)` (only counted at mask positions) · **corrector_in** = the (corrupted) token fed to forward-2 · **role** · **err** = in `is_err` · **f2** = forward-2 argmax · **corr** = correction loss at this position (`-logp` if err else `-p`).

| pos | GT | f1 | M2T | corrector_in | role | err | f2 | corr |
|---|---|---|---|---|---|---|---|---|
| 385 | Maint | 1 | — | green | inject(rand) | ✗ | maint | +1.96 |
| 386 | ain | ain | — | ain | keep-GT |  | ain | -0.48 |
| 387 | a | a | — | your | sample(top2) | ✗ | your | +1.17 |
| 388 | safe | moderate | — | safe | keep-GT |  | safe | -0.95 |
| 389 | following | distance | — | following | keep-GT |  | following | -0.49 |
| 390 | distance | distance | 0.00 | distance | mask→fill |  | distance | -0.99 |
| 391 | from | behind | — | cyclist | inject(rand) | ✗ | from | +0.96 |
| 392 | the | the | — | other | sample(top2) | ✗ | other | +4.59 |
| 393 | red | red | 0.15 | red | mask→fill |  | red | -0.75 |
| 394 | truck | truck | — | truck | keep-GT |  | truck | -0.62 |
| 395 | , | , | — | , | keep-GT |  | , | -0.96 |
| 396 | monitor | and | — | and | sample(top2) | ✗ | and | +11.12 |
| 397 | the | the | — | the | keep-GT |  | the | -0.91 |
| 398 | bus | traffic | — | truck | inject(rand) | ✗ | truck | +3.59 |
| 399 | 's | 's | — | 's | keep-GT |  | 's | -0.99 |
| 400 | actions | movements | 5.50 | movements | mask→fill | ✗ | actions | +0.70 |
| 401 | closely | , | 9.50 | , | mask→fill | ✗ | . | +11.06 |
| 402 | and | and | 0.41 | and | mask→fill |  | and | -0.84 |
| 403 | be | be | 0.17 | be | mask→fill |  | be | -0.33 |
| 404 | prepared | ready | — | prepared | keep-GT |  | ready | -0.19 |
| 405 | for | to | 3.42 | to | mask→fill | ✗ | to | +0.76 |
| 406 | pedestrians | braking | 6.50 | braking | mask→fill | ✗ | stop | +15.00 |
| 407 | , | , | — | , | keep-GT |  | , | -0.57 |
| 408 | especially | especially | 0.90 | especially | mask→fill |  | due | -0.17 |
| 409 | around | the | — | around | keep-GT |  | when | -0.10 |
| 410 | the | the | 0.02 | the | mask→fill |  | the | -0.99 |
| 411 | bus | bus | — | bus | keep-GT |  | bus | -0.66 |
| 412 | stop | 's | 1.23 | 's | mask→fill | ✗ | stop | +0.78 |
| 413 | area | . | — | area | keep-GT |  | area | -0.38 |
| 414 | . | . | 0.00 | . | mask→fill |  | . | -0.98 |
| 415 | Keep | Follow | — | van | inject(rand) | ✗ | wait | +4.88 |
| 416 | in | an | — | in | keep-GT |  | in | -0.93 |
| 417 | the | the | 0.05 | the | mask→fill |  | the | -0.99 |
| 418 | appropriate | the | — | appropriate | keep-GT |  | left | -0.01 |
| 419 | lane | lane | — | lane | keep-GT |  | lane | -0.92 |
| 420 | as | as | 0.45 | as | mask→fill |  | as | -0.92 |
| 421 | indicated | as | — | indicated | keep-GT |  | shown | -0.04 |
| 422 | by | by | 0.01 | by | mask→fill |  | by | -0.98 |
| 423 | the | the | 0.02 | the | mask→fill |  | the | -1.00 |
| 424 | overhead | overhead | — | overhead | keep-GT |  | overhead | -0.67 |

_(+22 more answer positions truncated)_

## 8. Losses (this example)

- **M2T / mdlm_loss** = mean `-logp(GT)` over the **23 masked** positions = **1.5469**.
- **corrector_loss** = mean over **all 62 answer** positions of (`-logp` at the **22 err** positions, `-p` at the rest) = **1.0938** (positive here ⇒ the 22 err positions' `-logp` outweigh the `-p` stability on the 40 correct positions; on easier examples the `-p` term can make it negative).
- **total** = `w_mdlm·mdlm + w_corrector·corrector` = 1.0·1.5469 + 1.0·1.0938 = **2.6406**.
- `rollout_error_frac` = 22/62 = **0.355** (inject=6, sample=9, mask-fill-err=7).

## 9. How to read this (base model, pre-warmup)

This is the **untrained base** (no warmup yet), so the `f2` column — forward-2's attempt to rewrite the
corrupted `corrector_in` back to GT — **mostly fails to recover** GT at the err (✗) positions:
- sample/inject errors usually stay wrong (392 `the`→`other`→f2 `other`; 396 `monitor`→`and`→f2 `and`;
  415 `Keep`→`van`→f2 `wait`),
- a few do recover (391 `from`→`cyclist`→**f2 `from`** ✓; 400 `actions`→`movements`→**f2 `actions`** ✓).

That matches the diagnostic verdict (base can't self-correct its own errors). The `-logp` term at err
positions is exactly the gradient that *teaches* f2 to recover; the `-p` term at the 40 correct
positions keeps it from rewriting correct tokens (change-not-delete). **End-to-end data flow:** GT
answer → absorbing-mask 23/62 → forward-1 (M2T loss on masked + denoise argmax) → build corrupted
`corrector_in` (mask→fill, inject random, sample top-2, keep GT) → forward-2 → correction loss
(`-logp` to push err→GT, `-p` to hold correct).

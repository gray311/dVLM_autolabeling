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
| 385 | Maint | 1 | — | 1 | model-err(vis) | ✗ | P | +4.28 |
| 386 | ain | ain | — | ain | keep-GT |  | ead | -0.01 |
| 387 | a | a | — | your | sample(top2) | ✗ | in | +2.94 |
| 388 | safe | moderate | — | moderate | model-err(vis) | ✗ | moderate | +5.22 |
| 389 | following | distance | — | distance | model-err(vis) | ✗ | speed | +4.41 |
| 390 | distance | distance | 0.00 | distance | mask→fill |  | distance | -0.96 |
| 391 | from | behind | — | behind | model-err(vis) | ✗ | behind | +2.59 |
| 392 | the | the | — | other | sample(top2) | ✗ | other | +3.53 |
| 393 | red | red | 0.15 | red | mask→fill |  | red | -0.96 |
| 394 | truck | truck | — | truck | keep-GT |  | truck | -0.98 |
| 395 | , | , | — | , | keep-GT |  | , | -0.99 |
| 396 | monitor | and | — | and | model-err(vis) | ✗ | and | +13.44 |
| 397 | the | the | — | the | keep-GT |  | the | -1.00 |
| 398 | bus | traffic | — | traffic | model-err(vis) | ✗ | traffic | +7.44 |
| 399 | 's | 's | — | 's | keep-GT |  | 's | -1.00 |
| 400 | actions | movements | 5.50 | movements | mask→fill | ✗ | movements | +5.81 |
| 401 | closely | , | 9.50 | , | mask→fill | ✗ | , | +13.75 |
| 402 | and | and | 0.41 | and | mask→fill |  | and | -1.00 |
| 403 | be | be | 0.17 | be | mask→fill |  | be | -0.41 |
| 404 | prepared | ready | — | ready | model-err(vis) | ✗ | ready | +1.35 |
| 405 | for | to | 3.42 | to | mask→fill | ✗ | to | +3.89 |
| 406 | pedestrians | braking | 6.50 | braking | mask→fill | ✗ | stop | +17.38 |
| 407 | , | , | — | , | keep-GT |  | , | -1.00 |
| 408 | especially | especially | 0.90 | especially | mask→fill |  | especially | -0.50 |
| 409 | around | the | — | the | model-err(vis) | ✗ | the | +11.44 |
| 410 | the | the | 0.02 | the | mask→fill |  | the | -0.99 |
| 411 | bus | bus | — | bus | keep-GT |  | bus | -0.93 |
| 412 | stop | 's | 1.23 | 's | mask→fill | ✗ | 's | +7.28 |
| 413 | area | . | — | . | model-err(vis) | ✗ | . | +10.56 |
| 414 | . | . | 0.00 | . | mask→fill |  | 2 | -0.00 |
| 415 | Keep | Follow | — | Follow | model-err(vis) | ✗ | Keep | +1.15 |
| 416 | in | an | — | an | model-err(vis) | ✗ | the | +9.25 |
| 417 | the | the | 0.05 | the | mask→fill |  | the | -1.00 |
| 418 | appropriate | the | — | the | model-err(vis) | ✗ | right | +6.88 |
| 419 | lane | lane | — | lane | keep-GT |  | lane | -0.64 |
| 420 | as | as | 0.45 | as | mask→fill |  | as | -0.90 |
| 421 | indicated | as | — | as | model-err(vis) | ✗ | ahead | +3.38 |
| 422 | by | by | 0.01 | by | mask→fill |  | to | -0.34 |
| 423 | the | the | 0.02 | the | mask→fill |  | the | -1.00 |
| 424 | overhead | overhead | — | overhead | keep-GT |  | overhead | -0.97 |

_(+22 more answer positions truncated)_

## 8. Losses (this example)

- **M2T / mdlm_loss** = mean `-logp(GT)` over the **23 masked** positions = **1.5469**.
- **corrector_loss** = mean over **all 62 answer** positions of (`-logp` at the **33 err** positions, `-p` at the rest) = **3.3125** (positive ⇒ the err positions' `-logp` outweigh the `-p` stability).
- **total** = `w_mdlm·mdlm + w_corrector·corrector` = 1.0·1.5469 + 1.0·3.3125 = **4.8750**.
- `rollout_error_frac` = 33/62 = **0.532** (inject=0, sample=6, mask-fill-err=7).

# SC Training Data-Flow — one real example (merged mode)

Model: base Fast_dVLM_3B · dataset: `coda_lm`[0] · seed 0 · `noise_mode=merged`, `include_mask_errors=True`, `alpha_inject=0.0`, `alpha_sample=0.05`.

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
| 385 | Maint | 1 | — | 1 | model-err(vis) | ✗ | 1 | +4.56 |
| 386 | ain | ain | — | ain | keep-GT |  | . | -0.01 |
| 387 | a | a | — | a | keep-GT |  | a | -0.44 |
| 388 | safe | moderate | — | moderate | model-err(vis) | ✗ | moderate | +1.84 |
| 389 | following | distance | — | distance | model-err(vis) | ✗ | speed | +7.69 |
| 390 | distance | distance | 0.00 | distance | mask→fill |  | away | -0.27 |
| 391 | from | behind | — | behind | model-err(vis) | ✗ | from | +0.22 |
| 392 | the | the | — | the | keep-GT |  | the | -1.00 |
| 393 | red | red | 0.15 | red | mask→fill |  | red | -0.93 |
| 394 | truck | truck | — | truck | keep-GT |  | truck | -0.98 |
| 395 | , | , | — | , | keep-GT |  | , | -0.79 |
| 396 | monitor | and | — | and | model-err(vis) | ✗ | and | +15.12 |
| 397 | the | the | — | the | keep-GT |  | the | -1.00 |
| 398 | bus | traffic | — | traffic | model-err(vis) | ✗ | traffic | +5.78 |
| 399 | 's | 's | — | 's | keep-GT |  | 's | -1.00 |
| 400 | actions | movements | 5.50 | movements | mask→fill | ✗ | directions | +6.53 |
| 401 | closely | , | 9.50 | , | mask→fill | ✗ | , | +13.38 |
| 402 | and | and | 0.41 | and | mask→fill |  | and | -1.00 |
| 403 | be | be | 0.17 | be | mask→fill |  | be | -0.46 |
| 404 | prepared | ready | — | ready | model-err(vis) | ✗ | prepared | +0.66 |
| 405 | for | to | 3.42 | to | mask→fill | ✗ | to | +4.03 |
| 406 | pedestrians | braking | 6.50 | braking | mask→fill | ✗ | lane | +17.50 |
| 407 | , | , | — | , | keep-GT |  | , | -0.94 |
| 408 | especially | especially | 0.90 | especially | mask→fill |  | because | -0.22 |
| 409 | around | the | — | the | model-err(vis) | ✗ | the | +10.38 |
| 410 | the | the | 0.02 | the | mask→fill |  | the | -0.99 |
| 411 | bus | bus | — | bus | keep-GT |  | bus | -0.40 |
| 412 | stop | 's | 1.23 | 's | mask→fill | ✗ | 's | +8.69 |
| 413 | area | . | — | . | model-err(vis) | ✗ | . | +10.00 |
| 414 | . | . | 0.00 | . | mask→fill |  | 2 | -0.00 |
| 415 | Keep | Follow | — | Follow | model-err(vis) | ✗ | Keep | +0.86 |
| 416 | in | an | — | an | model-err(vis) | ✗ | the | +8.88 |
| 417 | the | the | 0.05 | the | mask→fill |  | the | -0.99 |
| 418 | appropriate | the | — | the | model-err(vis) | ✗ | the | +8.50 |
| 419 | lane | lane | — | lane | keep-GT |  | lane | -0.40 |
| 420 | as | as | 0.45 | as | mask→fill |  | as | -0.32 |
| 421 | indicated | as | — | as | model-err(vis) | ✗ | ahead | +3.38 |
| 422 | by | by | 0.01 | by | mask→fill |  | to | -0.14 |
| 423 | the | the | 0.02 | the | mask→fill |  | the | -1.00 |
| 424 | overhead | overhead | — | overhead | keep-GT |  | overhead | -0.86 |

_(+22 more answer positions truncated)_

## 8. Losses (this example)

- **M2T / mdlm_loss** = mean `-logp(GT)` over the **23 masked** positions = **1.5469**.
- **corrector_loss** = mean over **all 62 answer** positions of (`-logp` at the **27 err** positions, `-p` at the rest) = **3.0156** (positive ⇒ the err positions' `-logp` outweigh the `-p` stability).
- **total** = `w_mdlm·mdlm + w_corrector·corrector` = 1.0·1.5469 + 0.2·3.0156 = **2.1562**.
- `rollout_error_frac` = 27/62 = **0.435** (inject=0, sample=0, mask-fill-err=7).

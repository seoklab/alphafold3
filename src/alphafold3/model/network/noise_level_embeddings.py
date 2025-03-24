# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Fourier embeddings for given noise levels.

We supply fixed weights and biases for the Fourier embeddings. These were
initially generated by the following code, but we make them into constants
to future proof against changes in jax rng generation:

```
dim = 256
w_key, b_key = jax.random.split(jax.random.PRNGKey(42))
weight = jax.random.normal(w_key, shape=[dim])
bias = jax.random.uniform(b_key, shape=[dim])
```
"""

import jax.numpy as jnp

# pyformat: disable
# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
_WEIGHT = [
     0.45873642,  0.06516238, -0.07278306, -0.26992258,  0.64292115,
    -0.40763968,  3.60116863,  0.54461384, -0.32644904,  2.10888267,
     1.30805349,  1.19838560, -1.37745857,  1.99475312, -1.64120293,
     1.07823789, -0.02288206,  0.88305283,  0.48099944,  0.17655374,
     0.30281949,  0.80646873,  0.62605333, -0.23965347, -1.02609432,
     0.75006109, -0.19913037,  0.07466396,  0.66431236, -0.60990530,
    -0.69709194, -0.44453633, -1.77656078,  0.02299878,  0.04095552,
     0.35485864, -0.47602659, -0.98820388, -0.24106771, -1.07254291,
    -0.99741757,  0.22697604,  1.41390419,  1.54984057, -0.12237291,
     0.20156337,  0.61767143,  0.23959029,  0.92454034,  1.84082258,
     0.89030224,  0.39598912, -1.52224910,  0.29669049,  1.52356744,
    -0.33968377,  0.24155144, -0.52308381, -0.23622665,  0.92825454,
    -0.63864607, -0.62169307,  0.78623551, -0.80352145, -0.45496067,
     1.30877995, -0.06686528,  1.00248849, -0.63593471,  0.16372502,
    -1.46133232,  1.10562658, -0.01693927,  0.28684548, -0.72843230,
     0.66133535, -1.92225552,  0.70241231, -0.96868867, -0.47309339,
    -1.66894221,  0.46018723, -0.56806105,  0.32694784, -0.46529883,
     1.02299964,  0.84688205,  1.19581807, -1.82454145,  0.05999713,
    -0.59530073,  1.44862521, -0.34933713, -0.46564487, -0.55005538,
    -1.61170268,  0.17502306,  0.38670063, -1.12133658, -0.29343036,
    -0.52527446, -1.26285112,  1.07982683,  0.51215219,  1.48963666,
     1.09847653, -0.01563358,  0.32574457,  1.94779706, -1.29198587,
     1.06249654, -0.86965990,  0.22975266, -0.27182648, -0.21130897,
    -0.41773933, -0.02329035,  1.31049252,  0.05579265, -1.23127055,
    -0.99691105,  0.27058721, -0.72509319, -0.14421797, -1.48605061,
     1.35041201,  1.29619241, -1.01022530, -0.79787987, -0.16166858,
     0.87210685,  1.69248152,  1.42469788, -0.72325104, -1.24823737,
     0.07051118,  0.71332991, -0.07360429, -0.91955227, -2.68856549,
    -0.44033936,  0.35482934, -0.57933813,  0.97468042, -0.31050494,
    -0.88454425, -2.08785224,  0.47322822, -0.02400172,  0.26644820,
    -0.19147627, -2.10538960, -1.27962470, -1.35999286,  2.09867334,
     0.65099514,  0.21604492, -0.45951018,  0.15994427, -0.31420693,
    -0.65202618, -0.61077976, -1.06100249, -1.47254968,  1.18165290,
    -0.78656220,  1.28182006,  1.80323684,  1.09196901,  0.26118696,
    -0.30168581,  0.39749333,  0.26812574, -1.51995814, -0.46909946,
     0.03874255, -1.36774313,  2.30143976,  2.06959820, -0.41647521,
     1.85624206,  0.49019700, -0.06726539,  0.00457313,  0.23915423,
    -1.84971249, -0.20482327, -0.34097880, -0.57933033, -1.10541213,
    -0.30269983, -0.16430426, -0.82371718,  0.10345812,  1.78753936,
     0.04786763,  1.86778629, -0.65214992,  0.81544143, -0.28214937,
     0.31187257,  0.57661986,  1.21938801, -1.56046617,  0.38046429,
    -0.18235965,  0.81794524, -0.40474343,  0.46538028, -1.15558851,
     0.59625793, -1.07801270,  0.07310858,  0.61526084,  0.55518496,
    -0.49787554,  0.92703879, -1.27780271, -0.83373469, -0.43015575,
     0.41877759, -1.03987372, -1.46055734,  0.61282396,  0.15590595,
    -0.34269521,  0.56509072, -1.17904210,  0.11374855, -1.83310866,
     0.38734794, -0.58623004,  0.77931106,  1.53930688, -0.70299625,
    -0.11389336, -1.14818096, -0.44400632,  1.21887410,  0.64066756,
    -0.70249403, -0.27244881,  0.38586098, -1.07925785,  0.12448707,
    -1.28286278,  0.37827531,  0.68812364,  1.65695465,  0.12440517,
    -0.03689830,  1.10224664, -0.28323629, -0.47939169,  0.70120829,
    -0.67204583
]

_BIAS = [
    0.00465965, 0.21738243, 0.22277749, 0.68463874, 0.84596848, 0.17337036,
    0.39573753, 0.78153563, 0.86311185, 0.21782327, 0.24377882, 0.42310703,
    0.19887352, 0.10486019, 0.48707581, 0.22205460, 0.97263455, 0.29714966,
    0.11244559, 0.53020525, 0.36796236, 0.37294638, 0.80261672, 0.04669094,
    0.86319661, 0.75907171, 0.77297020, 0.01114798, 0.55850804, 0.91799915,
    0.23032320, 0.12154722, 0.26701927, 0.42934716, 0.47951782, 0.96782577,
    0.86785042, 0.61985648, 0.05743814, 0.41800117, 0.68881893, 0.60575199,
    0.21058667, 0.64412105, 0.63958526, 0.89390790, 0.69755554, 0.89345169,
    0.53330755, 0.56985939, 0.30724049, 0.00984561, 0.91407037, 0.92118979,
    0.94153070, 0.81097460, 0.70537627, 0.32810748, 0.47227263, 0.11821401,
    0.44983089, 0.30767226, 0.31756389, 0.62969446, 0.69892538, 0.16949117,
    0.06207097, 0.46717727, 0.95348179, 0.62363589, 0.49018729, 0.06920040,
    0.39333904, 0.41299903, 0.52514863, 0.61197245, 0.56871891, 0.65053988,
    0.22203422, 0.46748531, 0.86931503, 0.87050021, 0.40208721, 0.32084906,
    0.55084610, 0.94584596, 0.76279902, 0.36250532, 0.74272907, 0.66682065,
    0.96452832, 0.64768302, 0.88070846, 0.56995463, 0.06395614, 0.69499350,
    0.44494808, 0.39775658, 0.20280898, 0.33363521, 0.05999005, 0.44414878,
    0.65227020, 0.01199079, 0.71995056, 0.19045687, 0.48342144, 0.25127733,
    0.66515994, 0.22465158, 0.22313106, 0.06302810, 0.55783665, 0.93625581,
    0.58800840, 0.72525370, 0.52879298, 0.77195418, 0.15548682, 0.01028740,
    0.39325142, 0.45401239, 0.71494079, 0.33011997, 0.05050695, 0.26381660,
    0.63064706, 0.47604024, 0.08593416, 0.00383425, 0.06352687, 0.05510247,
    0.03552997, 0.35810637, 0.56094289, 0.60922170, 0.88599777, 0.45419788,
    0.40486634, 0.71297824, 0.34976673, 0.97825217, 0.12915993, 0.09566259,
    0.64318919, 0.16717327, 0.82308614, 0.32672071, 0.81688786, 0.84857118,
    0.99922776, 0.07551706, 0.18766022, 0.13051236, 0.39136350, 0.08768725,
    0.92048228, 0.87185788, 0.39158428, 0.79224777, 0.17492688, 0.68902445,
    0.81980729, 0.70458186, 0.59489477, 0.93324888, 0.49986637, 0.40705478,
    0.89202917, 0.20673239, 0.39339757, 0.20996964, 0.02923799, 0.53992438,
    0.40119815, 0.10366607, 0.08044600, 0.95551598, 0.20518017, 0.68826210,
    0.90159297, 0.69008791, 0.86880815, 0.16246438, 0.89628279, 0.11481643,
    0.61353648, 0.41545081, 0.92478311, 0.78212476, 0.48292696, 0.79621077,
    0.11947489, 0.01747024, 0.22928023, 0.87387264, 0.86349785, 0.89526737,
    0.58904779, 0.13896775, 0.68194926, 0.55824125, 0.44428205, 0.55422378,
    0.28189969, 0.27923775, 0.09979951, 0.66994715, 0.45943546, 0.71207762,
    0.17300689, 0.83434916, 0.02573085, 0.45858085, 0.55934799, 0.30676675,
    0.52219367, 0.34544575, 0.19280875, 0.26937950, 0.07147646, 0.06295013,
    0.76382887, 0.38737607, 0.58825982, 0.17423475, 0.05509448, 0.97228825,
    0.94380617, 0.91664016, 0.18800116, 0.41771865, 0.59420645, 0.77371931,
    0.64687788, 0.27284670, 0.22310913, 0.15663862, 0.45573199, 0.50386798,
    0.66712272, 0.71649647, 0.28475654, 0.83415413, 0.75261366, 0.61517799,
    0.93544555, 0.76141870, 0.85474241, 0.74766934, 0.33459592, 0.78477907,
    0.07250881, 0.10174239, 0.95332730, 0.80793905
]
# pyformat: enable
# pylint: enable=bad-whitespace
# pylint: enable=bad-continuation


def noise_embeddings(sigma_scaled_noise_level: jnp.ndarray) -> jnp.ndarray:
  """Returns Fourier noise level embeddings for diffusion model."""
  transformed_noise_level = (1 / 4) * jnp.log(sigma_scaled_noise_level)
  weight = jnp.array(_WEIGHT, dtype=jnp.float32)
  bias = jnp.array(_BIAS, dtype=jnp.float32)
  embeddings = transformed_noise_level[..., None] * weight + bias
  return jnp.cos(2 * jnp.pi * embeddings)

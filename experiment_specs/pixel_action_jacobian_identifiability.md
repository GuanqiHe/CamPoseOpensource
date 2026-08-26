# Pixel Action Jacobian Identifiability Experiment

## 1. Objective

Test whether a deterministic vision policy needs an explicit description of how each raw arm-action dimension moves robot features in the image.

For a visible robot feature field `s`, define the instantaneous desired-kinematic mapping

\[
\Delta s \approx J_{\mathrm{pix}}(q,E)a,
\qquad
J_{\mathrm{pix}} = \frac{\partial s}{\partial q}
\frac{\partial q_{\mathrm{goal}}}{\partial a}.
\]

`E` includes robot kinematics, raw joint/action convention, controller action scaling, and camera geometry. This experiment changes the joint-axis and motor-sign convention while holding physical robot geometry and task-space trajectories fixed.

The immediate experiment tests identifiability, not held-out embodiment generalization.

## 2. Fixed data contract

- Source: the existing three successful `LiftRand` joint-delta trajectories, transformed into `cfg0`-`cfg4` paired configurations.
- Pairing: for every physical time index, all five configurations use the same canonical RGB observation and the same task; only the raw action label and structural condition change.
- Physical control: state playback and controller replay must continue to pass the existing equivalent-trajectory regression tests.
- Policy input: one `256 x 256` RGB image and one of the conditioning modes below. Raw `q` and configuration ID are excluded.
- Policy target: an 8-D normalized joint-delta action chunk. `J_pix` conditions the seven arm dimensions; the gripper dimension is predicted from RGB/task context.
- Training and evaluation samples are the same paired samples in Stages 1 and 2 because these are deliberate overfit tests.

## 3. Structural representations

All variants use the same deterministic ACT decoder and the same RGB backbone.

1. `none`: a learned constant condition; no robot-configuration information.
2. `global_sign`: the 7-D sign vector `S` projected to a global condition token.
3. `pixel_jacobian`: a `15 x 16 x 16` field with channel order
   `[du/da1, dv/da1, ..., du/da7, dv/da7, robot_mask]`.

For `pixel_jacobian`, non-robot cells are zero. At each visible robot cell, depth and segmentation identify a surface point and its owning body. MuJoCo supplies the arbitrary-point translational Jacobian, which is projected through the camera model and multiplied by the controller's raw-action-to-joint-goal derivative. The field is fused with the spatial RGB feature at the corresponding `16 x 16` cell.

For the current coordinate-only configurations,

\[
J_{\mathrm{pix}}^{(c)} = J_{\mathrm{pix}}^{(0)} S_c.
\]

Therefore `global_sign` is a sufficient representation in this experiment. A result where both `global_sign` and `pixel_jacobian` succeed only shows that explicit action semantics restore identifiability.

## 4. Stage gates and metrics

### Gate A: Jacobian correctness

Compare analytic derivatives with central finite differences of the same body-fixed visible surface point.

- At least 10 valid robot cells per checked frame.
- Evaluate entries whose finite-difference magnitude is at least `0.05` grid-pixel per unit raw action.
- Median relative error must be below `2%`.
- 95th-percentile relative error must be below `10%`.
- Column-sign transformation must match the configured sign vector to numerical tolerance (`max_abs_error < 1e-6`).

Training does not start until Gate A passes.

### Stage 1: single-configuration overfit

Train each representation on `cfg0` only with an identical optimizer, update budget, and random-seed set.

- Primary: non-padding normalized action MAE at deterministic inference.
- Secondary: sign accuracy on target entries with `abs(action) >= 0.05`.
- Gate: all three variants must reach `MAE <= 0.03` and sign accuracy `>= 95%` on every seed.

Failure here is a pipeline or capacity failure; Stage 2 is not interpreted.

### Stage 2: five-configuration paired overfit

Jointly train on `cfg0`-`cfg4`. Each minibatch samples physical time indices uniformly and includes or balances all configurations.

- Report overall and per-configuration normalized action MAE.
- Report per-joint sign accuracy for entries with `abs(action) >= 0.05`.
- Compute the exact deterministic L1 Bayes lower bound from each set of paired labels.
- Success criterion for conditioned variants: `MAE <= 0.03` and sign accuracy `>= 95%` per configuration.
- Expected control: `none` cannot predict every contradictory paired label from identical input. Its error must be compared with the computed Bayes bound rather than with training loss alone.

The training objective is masked action L1. Model selection and all reported metrics use deterministic inference without an action-conditioned latent.

## 5. Reproducibility and claim boundary

- Formal runs use W&B online and record run URL/ID, exact git commit, full config, seeds, `samples_seen`, train/eval curves, throughput, and checkpoint metadata.
- Cache metadata records source HDF5 hashes, camera parameters, channel convention, action scaling, and Jacobian finite-difference results.
- Current Stage 2 supports only the claim that explicit raw-action semantics resolve missing information.
- A later experiment must train on balanced sign combinations and evaluate unseen combinations before claiming held-out configuration generalization.
- Different kinematic link geometries or camera arrangements are required before claiming that a pixel-aligned field is better than a compact global action specification.

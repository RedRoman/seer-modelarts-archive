# Budget interpretation, before actual scalar analysis

This is an algebraic observation, independent of all critic outputs, utility arrays, and the aggregate PSNR descriptions disclosed separately. It changes no frozen statistic or gate.

For two donors, the registered Shapley values obey `phi1 + phi2 = v12`. With tau one, `B = min(1,max(v12,0))`, `ci = min(1,max(phii,0))`, and `w1+w2=B` subject to `0<=wi<=ci`.

- If `v12<=0`, the budget is zero and both donor weights are zero.
- If `0<v12<=1` and both Shapley values are nonnegative, `c1+c2=B`; the only feasible allocation is `wi=phii`. The softmax proposal has no effect.
- If `0<v12<=1` and one Shapley value is negative, that donor has cap zero and the other donor receives all of `B`. The softmax proposal again has no effect.
- A nontrivial allocation interval can occur only when `v12>1`, both caps are positive, and their sum is greater than one. Even in this regime, projection may put the proposal on a cap. The `v12>=1` fraction is therefore only an upper bound on how often the proposal can affect allocation.

For example, `(phi1,phi2)=(0.4,0.1)` forces weights `(0.4,0.1)`, while `(0.4,-0.1)` forces `(0.3,0)`, for every proposal. Numerical cap truncation (`phii>=1`) and a genuinely active allocation constraint (`wi=ci>0`) are different events, so the frozen analyzer reports them separately.

The solver, softmax, and caps do not by themselves establish module innovation. The current experiment tests whether masked-context prediction increments discriminate the incremental utility of a fixed RGB action better than all five controls. A statistical association does not establish that those prediction increments measure a recovery causal effect.

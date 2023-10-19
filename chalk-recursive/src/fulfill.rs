use crate::fixed_point::Minimums;
use crate::solve::{IsTracing, SolveDatabase};

use chalk_ir::cast::Cast;
use chalk_ir::fold::TypeFoldable;
use chalk_ir::interner::{HasInterner, Interner};
use chalk_ir::visit::TypeVisitable;
use chalk_ir::zip::Zip;
use chalk_ir::{
    Binders, BoundVar, Canonical, ConstrainedSubst, Constraint, Constraints, DomainGoal,
    Environment, EqGoal, Fallible, GenericArg, GenericArgData, Goal, GoalData, InEnvironment,
    NoSolution, ProgramClauseImplication, QuantifierKind, Substitution, SubtypeGoal, TyKind,
    TyVariableKind, UCanonical, UnificationDatabase, UniverseMap, Variance,
};
use chalk_solve::debug_span;
use chalk_solve::infer::{InferenceTable, ParameterEnaVariableExt};
use chalk_solve::solve::truncate;
use chalk_solve::{Guidance, Solution};
use fluid_let::{fluid_let, fluid_set};
use rustc_hash::FxHashSet;
use std::fmt::Debug;
use tracing::{debug, instrument};

use argus::proof_tree::{
    flat::ProofTreeBuilder,
    fulfill as af,
    indices::{ObligationIdx, ProofNodeIdx},
    *,
};

fluid_let!(static CURRENT_ITEM: af::IdxKind);

// Specify that the dynamic provenance should be forgotten.
macro_rules! provenance_barrier {
    () => {
        fluid_set!(CURRENT_ITEM, af::IdxKind::CurrentRoot);
    };
}

// We use this to track where the goals and obligations came from.
struct FulfillmentBuilder<I: Interner> {
    obligations: Vec<(ObligationIdx, Obligation<I>)>,
    builder: af::Fulfillment<I>,
}

impl<I: Interner> FulfillmentBuilder<I> {
    fn new(
        root_kind: FulfillmentKind<I>,
        infer: InferenceTable<I>,
        subst: Substitution<I>,
    ) -> Self {
        Self {
            obligations: Vec::default(),
            builder: af::Fulfillment::new(root_kind, infer, subst),
        }
    }

    fn inform_of_failure(&mut self, kind: af::FulfillFailKind) {
        self.builder.inform_of_failure(kind)
    }

    fn push_igoal(&mut self, subgoal: af::InterimGoal<I>) -> af::IdxKind {
        CURRENT_ITEM.get(|idx_opt| {
            let from = idx_opt.copied().unwrap_or(af::IdxKind::CurrentRoot);
            let idx = self.builder.add_interim_goal(from, subgoal);
            af::IdxKind::Interim(idx)
        })
    }

    /// Push a new `Obligation` onto the stack, but save this information in the
    /// `ProofTreeBuilder`, and `FulfillmentBuilder`.
    fn push_obligation(
        &mut self,
        interner: I,
        builder: &mut ProofTreeBuilder<I>,
        obligation: Obligation<I>,
        infer: &InferenceTable<I>,
        subst: &Substitution<I>,
    ) -> af::IdxKind {
        // Convert to an argus::Obligation
        let (kind, goal) = match &obligation {
            Obligation::Prove(env_goal) => (af::ObligationKind::Prove, env_goal),
            Obligation::Refute(env_goal) => (af::ObligationKind::Refute, env_goal),
        };

        let canonicalized = ucanonicalize(infer.clone(), interner, goal);
        // Push the argus obligation into the global node buffer, obtaining its index.
        let obl = builder.push_node(af::Obligation::new(&canonicalized, infer, subst, kind));

        CURRENT_ITEM.get(|idx_opt| {
            let from = idx_opt.copied().unwrap_or(af::IdxKind::CurrentRoot);
            // Get the obligation index within the Obligation Buffer
            let idx = self.builder.add_obligation(from, obl);
            // push the local Obligation with its corresponding ObligationIdx.
            self.obligations.push((idx, obligation));
            af::IdxKind::Obligation(idx)
        })
    }

    fn push_unification(&mut self, unification: af::Unification<I>) -> af::IdxKind {
        CURRENT_ITEM.get(|idx_opt| {
            let from = idx_opt.copied().unwrap_or(af::IdxKind::CurrentRoot);
            let idx = self.builder.add_unification(from, unification);
            af::IdxKind::Unification(idx)
        })
    }

    fn store_result(
        &mut self,
        builder: &mut ProofTreeBuilder<I>,
        from: ObligationIdx,
        result: af::ObligationResult<I>,
    ) {
        let node_idx = self.builder.obligations[from];
        let node = &mut builder.nodes[node_idx];
        let ProofNode::Obligation(obligation) = node else {
            unreachable!("ObligationIdx points to non-Obligation ProofNode {:?}", node);
        };
        let child = result.trace.root;
        obligation.result.push(result);
        builder.relate(node_idx, child);
    }

    /// Get the bottom-most leaf of the proof tree. In this method we treat the tree as a DAG.
    fn get_current_sink(&self, builder: &ProofTreeBuilder<I>) -> Option<ProofNodeIdx> {
        // FIXME(gavinleroy): this is definitely wrong ...

        // The last result of the last obligation (not how it works...)
        self.builder.obligations.iter().rev().find_map(|&o| {
            let node = &builder.nodes[o];
            let ProofNode::Obligation(obligation) = node else {
                return None;
            };
            let last_res = obligation.result.last()?;
            Some(last_res.trace.leaf)
        })
    }
}

impl<I: Interner> std::fmt::Debug for FulfillmentBuilder<I> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "{:?}", self.obligations)
    }
}

impl<I: Interner> std::ops::Deref for FulfillmentBuilder<I> {
    type Target = Vec<(ObligationIdx, Obligation<I>)>;

    fn deref(&self) -> &Self::Target {
        &self.obligations
    }
}

impl<I: Interner> std::ops::DerefMut for FulfillmentBuilder<I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.obligations
    }
}

#[derive(Debug)]
enum Outcome {
    Complete,
    Incomplete,
}

impl Outcome {
    fn is_complete(&self) -> bool {
        matches!(self, Outcome::Complete)
    }
}

/// A goal that must be resolved
#[derive(Clone, Debug, PartialEq, Eq)]
enum Obligation<I: Interner> {
    /// For "positive" goals, we flatten all the way out to leafs within the
    /// current `Fulfill`
    Prove(InEnvironment<Goal<I>>),

    /// For "negative" goals, we don't flatten in *this* `Fulfill`, which would
    /// require having a logical "or" operator. Instead, we recursively solve in
    /// a fresh `Fulfill`.
    Refute(InEnvironment<Goal<I>>),
}

/// When proving a leaf goal, we record the free variables that appear within it
/// so that we can update inference state accordingly.
#[derive(Clone, Debug)]
struct PositiveSolution<I: Interner> {
    free_vars: Vec<GenericArg<I>>,
    universes: UniverseMap,
    solution: Solution<I>,
}

/// When refuting a goal, there's no impact on inference state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NegativeSolution {
    Refuted,
    Ambiguous,
}

impl<I: Interner> From<PositiveSolution<I>> for af::PositiveSolution<I> {
    fn from(sol: PositiveSolution<I>) -> Self {
        let PositiveSolution {
            free_vars,
            universes,
            solution,
        } = sol;

        af::PositiveSolution {
            free_vars,
            universes,
            solution,
        }
    }
}

impl From<NegativeSolution> for af::NegativeSolution {
    fn from(sol: NegativeSolution) -> Self {
        match sol {
            NegativeSolution::Refuted => af::NegativeSolution::Refuted,
            NegativeSolution::Ambiguous => af::NegativeSolution::Ambiguous,
        }
    }
}

fn map_into<T, U, E>(res: Result<T, E>) -> Result<U, E>
where
    U: From<T>,
{
    res.map(|v| v.into())
}

fn positive_kind<I: Interner>(sol: &Fallible<PositiveSolution<I>>) -> af::OblResultKind<I> {
    af::OblResultKind::Positive(map_into(sol.clone()))
}

fn negative_kind<I: Interner>(sol: &Fallible<NegativeSolution>) -> af::OblResultKind<I> {
    af::OblResultKind::Negative(map_into(sol.clone()))
}

fn canonicalize<I: Interner, T>(
    infer: &mut InferenceTable<I>,
    interner: I,
    value: T,
) -> (Canonical<T>, Vec<GenericArg<I>>)
where
    T: TypeFoldable<I>,
    T: HasInterner<Interner = I>,
{
    let res = infer.canonicalize(interner, value);
    let free_vars = res
        .free_vars
        .into_iter()
        .map(|free_var| free_var.to_generic_arg(interner))
        .collect();
    (res.quantified, free_vars)
}

fn u_canonicalize<I: Interner, T>(
    _infer: &mut InferenceTable<I>,
    interner: I,
    value0: &Canonical<T>,
) -> (UCanonical<T>, UniverseMap)
where
    T: Clone + HasInterner<Interner = I> + TypeFoldable<I> + TypeVisitable<I>,
    T: HasInterner<Interner = I>,
{
    let res = InferenceTable::u_canonicalize(interner, value0);
    (res.quantified, res.universes)
}

fn ucanonicalize<I: Interner, T>(
    // HACK to avoid messing with the environment
    mut infer: InferenceTable<I>,
    interner: I,
    value0: &T,
) -> UCanonical<T>
where
    T: Clone + HasInterner<Interner = I> + TypeFoldable<I> + TypeVisitable<I>,
    T: HasInterner<Interner = I>,
{
    let (quantified, free_vars) = canonicalize(&mut infer, interner, value0.clone());
    let (quantified, universes) = u_canonicalize(&mut infer, interner, &quantified);
    quantified
}

fn unify<I: Interner, T>(
    infer: &mut InferenceTable<I>,
    interner: I,
    db: &dyn UnificationDatabase<I>,
    environment: &Environment<I>,
    variance: Variance,
    a: &T,
    b: &T,
) -> Fallible<Vec<InEnvironment<Goal<I>>>>
where
    T: ?Sized + Zip<I>,
{
    let res = infer.relate(interner, db, environment, variance, a, b)?;
    Ok(res.goals)
}

/// A `Fulfill` is where we actually break down complex goals, instantiate
/// variables, and perform inference. It's highly stateful. It's generally used
/// in Chalk to try to solve a goal, and then package up what was learned in a
/// stateless, canonical way.
///
/// In rustc, you can think of there being an outermost `Fulfill` that's used when
/// type checking each function body, etc. There, the state reflects the state
/// of type inference in general. But when solving trait constraints, *fresh*
/// `Fulfill` instances will be created to solve canonicalized, free-standing
/// goals, and transport what was learned back to the outer context.
pub(super) struct Fulfill<'s, I: Interner, Solver: SolveDatabase<I> + IsTracing<I>> {
    solver: &'s mut Solver,
    subst: Substitution<I>,
    infer: InferenceTable<I>,

    /// The remaining goals to prove or refute
    obligations: FulfillmentBuilder<I>,

    /// Lifetime constraints that must be fulfilled for a solution to be fully
    /// validated.
    constraints: FxHashSet<InEnvironment<Constraint<I>>>,

    /// Record that a goal has been processed that can neither be proved nor
    /// refuted. In such a case the solution will be either `CannotProve`, or `Err`
    /// in the case where some other goal leads to an error.
    cannot_prove: bool,
}

// FIXME: Previously we returned a `Fallible` type which simply uses the NoSolution
//        variant as the error, here we just return the attempted proof tree, and
//        later can use NoSolution as the Fallible<Solution<I>> type.
type TracedCreate<T> = Result<T, TreeDescription>;

impl<'s, I: Interner, Solver: SolveDatabase<I> + IsTracing<I>> Fulfill<'s, I, Solver> {
    #[instrument(level = "debug", skip(solver, infer))]
    pub(super) fn new_with_clause(
        solver: &'s mut Solver,
        infer: InferenceTable<I>,
        subst: Substitution<I>,
        canonical_goal: InEnvironment<DomainGoal<I>>,
        clause: &Binders<ProgramClauseImplication<I>>,
    ) -> TracedCreate<Self> {
        let canonicalized_goal = ucanonicalize(infer.clone(), solver.interner(), &canonical_goal);
        let mut fulfill = Fulfill {
            solver,
            infer: infer.clone(),
            subst: subst.clone(),
            obligations: FulfillmentBuilder::new(
                FulfillmentKind::WithClause {
                    goal: canonicalized_goal,
                    clause: clause.clone(),
                },
                infer,
                subst,
            ),
            constraints: FxHashSet::default(),
            cannot_prove: false,
        };

        // Forget the goal of the outer context.
        provenance_barrier! {};

        debug!(?clause, "Using program clause");

        let pci = fulfill
            .infer
            .instantiate_binders_existentially(fulfill.solver.interner(), clause.clone());

        let ProgramClauseImplication {
            consequence,
            conditions,
            constraints,
            priority: _,
        } = pci;

        debug!(?consequence, ?conditions, ?constraints);
        fulfill
            .constraints
            .extend(constraints.as_slice(fulfill.interner()).to_owned());

        debug!("the subst is {:?}", fulfill.subst);

        if let Err(_e) = fulfill.unify(
            &canonical_goal.environment,
            Variance::Invariant,
            &canonical_goal.goal,
            &consequence,
        ) {
            fulfill
                .obligations
                .inform_of_failure(af::FulfillFailKind::Unification);
            return Err(fulfill.mint_proof(None));
        }

        // if so, toss in all of its premises
        for condition in conditions.as_slice(fulfill.solver.interner()) {
            if let Err(_e) = fulfill.push_goal(&canonical_goal.environment, condition.clone()) {
                return Err(fulfill.mint_proof(None));
            }
        }

        Ok(fulfill)
    }

    pub(super) fn new_with_simplification(
        solver: &'s mut Solver,
        infer: InferenceTable<I>,
        subst: Substitution<I>,
        canonical_goal: InEnvironment<Goal<I>>,
    ) -> TracedCreate<Self> {
        let canonicalized_goal = ucanonicalize(infer.clone(), solver.interner(), &canonical_goal);
        let mut fulfill = Fulfill {
            solver,
            infer: infer.clone(),
            subst: subst.clone(),
            obligations: FulfillmentBuilder::new(
                FulfillmentKind::WithSimplification {
                    goal: canonicalized_goal,
                },
                infer,
                subst,
            ),
            constraints: FxHashSet::default(),
            cannot_prove: false,
        };

        // Forget the goal of the outer context.
        provenance_barrier! {};

        if let Err(_e) = fulfill.push_goal(&canonical_goal.environment, canonical_goal.goal.clone())
        {
            return Err(fulfill.mint_proof(None));
        }

        Ok(fulfill)
    }

    fn push_obligation(&mut self, obligation: Obligation<I>) {
        // truncate to avoid overflows
        match &obligation {
            Obligation::Prove(goal) => {
                if truncate::needs_truncation(
                    self.solver.interner(),
                    &mut self.infer,
                    self.solver.max_size(),
                    goal,
                ) {
                    // the goal is too big. Record that we should return Ambiguous
                    self.cannot_prove = true;
                    self.obligations
                        .inform_of_failure(af::FulfillFailKind::GoalTooLarge);
                    return;
                }
            }
            Obligation::Refute(goal) => {
                if truncate::needs_truncation(
                    self.solver.interner(),
                    &mut self.infer,
                    self.solver.max_size(),
                    goal,
                ) {
                    // the goal is too big. Record that we should return Ambiguous
                    self.cannot_prove = true;
                    self.obligations
                        .inform_of_failure(af::FulfillFailKind::GoalTooLarge);
                    return;
                }
            }
        };

        self.obligations.push_obligation(
            self.solver.interner(),
            self.solver.get_inspector(),
            obligation,
            &self.infer,
            &self.subst,
        );
    }

    /// Unifies `a` and `b` in the given environment.
    ///
    /// Wraps `InferenceTable::unify`; any resulting normalizations are added
    /// into our list of pending obligations with the given environment.
    pub(super) fn unify<T>(
        &mut self,
        environment: &Environment<I>,
        variance: Variance,
        a: &T,
        b: &T,
    ) -> Fallible<()>
    where
        T: Clone + Zip<I> + Debug,
        (Environment<I>, Variance, T, T): Into<Unification<I>>,
    {
        let item_idx = self
            .obligations
            .push_unification((environment.clone(), variance, a.clone(), b.clone()).into());
        fluid_set!(CURRENT_ITEM, item_idx);

        let goals = match unify(
            &mut self.infer,
            self.solver.interner(),
            self.solver.db().unification_database(),
            environment,
            variance,
            a,
            b,
        ) {
            Ok(goals) => goals,
            Err(e) => {
                self.obligations
                    .inform_of_failure(af::FulfillFailKind::Unification);
                return Err(e);
            }
        };

        debug!("unify({:?}, {:?}) succeeded", a, b);
        debug!("unify: goals={:?}", goals);

        for goal in goals {
            let goal = goal.cast(self.solver.interner());
            self.push_obligation(Obligation::Prove(goal));
        }
        Ok(())
    }

    /// Create obligations for the given goal in the given environment. This may
    /// ultimately create any number of obligations.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn push_goal(
        &mut self,
        environment: &Environment<I>,
        goal: Goal<I>,
    ) -> Fallible<()> {
        debug!("pushing_goal {:?}", goal);

        let item_idx = self.obligations.push_igoal(af::InterimGoal {
            goal: goal.clone(),
            environment: environment.clone(),
        });
        fluid_set!(CURRENT_ITEM, item_idx);

        let interner = self.solver.interner();
        match goal.data(interner) {
            GoalData::Quantified(QuantifierKind::ForAll, subgoal) => {
                let subgoal = self
                    .infer
                    .instantiate_binders_universally(self.solver.interner(), subgoal.clone());
                self.push_goal(environment, subgoal)?;
            }
            GoalData::Quantified(QuantifierKind::Exists, subgoal) => {
                let subgoal = self
                    .infer
                    .instantiate_binders_existentially(self.solver.interner(), subgoal.clone());
                self.push_goal(environment, subgoal)?;
            }
            GoalData::Implies(wc, subgoal) => {
                let new_environment =
                    &environment.add_clauses(interner, wc.iter(interner).cloned());
                self.push_goal(new_environment, subgoal.clone())?;
            }
            GoalData::All(goals) => {
                for subgoal in goals.as_slice(interner) {
                    self.push_goal(environment, subgoal.clone())?;
                }
            }
            GoalData::Not(subgoal) => {
                let in_env = InEnvironment::new(environment, subgoal.clone());
                self.push_obligation(Obligation::Refute(in_env));
            }
            GoalData::DomainGoal(_) => {
                let in_env = InEnvironment::new(environment, goal);
                self.push_obligation(Obligation::Prove(in_env));
            }
            GoalData::EqGoal(EqGoal { a, b }) => {
                self.unify(environment, Variance::Invariant, &a, &b)?;
            }
            GoalData::SubtypeGoal(SubtypeGoal { a, b }) => {
                let a_norm = self.infer.normalize_ty_shallow(interner, a);
                let a = a_norm.as_ref().unwrap_or(a);
                let b_norm = self.infer.normalize_ty_shallow(interner, b);
                let b = b_norm.as_ref().unwrap_or(b);

                if matches!(
                    a.kind(interner),
                    TyKind::InferenceVar(_, TyVariableKind::General)
                ) && matches!(
                    b.kind(interner),
                    TyKind::InferenceVar(_, TyVariableKind::General)
                ) {
                    self.cannot_prove = true;
                    self.obligations
                        .inform_of_failure(af::FulfillFailKind::GeneralVarSubtype);
                } else {
                    self.unify(environment, Variance::Covariant, &a, &b)?;
                }
            }
            GoalData::CannotProve => {
                debug!("Pushed a CannotProve goal, setting cannot_prove = true");
                self.cannot_prove = true;
                self.obligations
                    .inform_of_failure(af::FulfillFailKind::CannotProve);
            }
        }
        Ok(())
    }

    #[instrument(level = "debug", skip(self, minimums, should_continue))]
    fn prove(
        &mut self,
        idx: ObligationIdx,
        wc: InEnvironment<Goal<I>>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> Fallible<PositiveSolution<I>> {
        let interner = self.solver.interner();
        let (quantified, free_vars) = canonicalize(&mut self.infer, interner, wc);
        let (quantified, universes) = u_canonicalize(&mut self.infer, interner, &quantified);

        let TracedFallible {
            solution, trace, ..
        } = self
            .solver
            .solve_goal(quantified, minimums, should_continue);

        debug!("Done proving {:?}", solution);

        let sol = solution.map(|solution| PositiveSolution {
            free_vars,
            universes,
            solution,
        });

        self.obligations.store_result(
            self.solver.get_inspector(),
            idx,
            af::ObligationResult::new(trace, positive_kind(&sol)),
        );

        sol
    }

    fn refute(
        &mut self,
        idx: ObligationIdx,
        goal: InEnvironment<Goal<I>>,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> Fallible<NegativeSolution> {
        let canonicalized = match self
            .infer
            .invert_then_canonicalize(self.solver.interner(), goal)
        {
            Some(v) => v,
            None => {
                // Treat non-ground negatives as ambiguous. Note that, as inference
                // proceeds, we may wind up with more information here.
                let unknown = self.solver.get_inspector().push_node(ProofTree::unknown());
                self.obligations.store_result(
                    self.solver.get_inspector(),
                    idx,
                    af::ObligationResult::new(
                        TreeDescription::single(unknown),
                        negative_kind(&Ok(NegativeSolution::Ambiguous)),
                    ),
                );
                return Ok(NegativeSolution::Ambiguous);
            }
        };

        // Negate the result
        let (quantified, _) =
            u_canonicalize(&mut self.infer, self.solver.interner(), &canonicalized);
        let mut minimums = Minimums::new(); // FIXME -- minimums here seems wrong
        let TracedFallible {
            solution, trace, ..
        } = self
            .solver
            .solve_goal(quantified, &mut minimums, should_continue);

        debug!("Done negating {:?}", solution);

        let sol = if let Ok(solution) = solution {
            if solution.is_unique() {
                Err(NoSolution)
            } else {
                Ok(NegativeSolution::Ambiguous)
            }
        } else {
            Ok(NegativeSolution::Refuted)
        };

        self.obligations.store_result(
            self.solver.get_inspector(),
            idx,
            af::ObligationResult::new(trace, negative_kind(&sol)),
        );

        sol
    }

    /// Trying to prove some goal led to a the substitution `subst`; we
    /// wish to apply that substitution to our own inference variables
    /// (and incorporate any region constraints). This substitution
    /// requires some mapping to get it into our namespace -- first,
    /// the universes it refers to have been canonicalized, and
    /// `universes` stores the mapping back into our
    /// universes. Second, the free variables that appear within can
    /// be mapped into our variables with `free_vars`.
    #[instrument(level = "debug", skip(self, universes))]
    fn apply_solution(
        &mut self,
        free_vars: Vec<GenericArg<I>>,
        universes: UniverseMap,
        subst: Canonical<ConstrainedSubst<I>>,
    ) {
        debug!("APPLYING SOLUTION:\n    {free_vars:?}\n    {subst:?}");

        use chalk_solve::infer::ucanonicalize::UniverseMapExt;
        let subst = universes.map_from_canonical(self.interner(), &subst);
        let sbst = self
            .infer
            .instantiate_canonical(self.solver.interner(), subst);

        debug!("CanonicalSubst {sbst:?}");

        let ConstrainedSubst { subst, constraints } = sbst;

        debug!(
            "fulfill::apply_solution: adding constraints {:?}",
            constraints
        );
        self.constraints
            .extend(constraints.as_slice(self.interner()).to_owned());

        // We use the empty environment for unification here because we're
        // really just doing a substitution on unconstrained variables, which is
        // guaranteed to succeed without generating any new constraints.
        let empty_env = &Environment::new(self.solver.interner());

        for (i, free_var) in free_vars.into_iter().enumerate() {
            let subst_value = subst.at(self.interner(), i);
            self.unify(empty_env, Variance::Invariant, &free_var, subst_value)
                .unwrap_or_else(|err| {
                    panic!(
                        "apply_solution failed with free_var={:?}, subst_value={:?}: {:?}",
                        free_var, subst_value, err
                    );
                });
        }
    }

    fn fulfill(
        &mut self,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> Fallible<Outcome> {
        debug_span!("fulfill", obligations=?self.obligations);

        // Try to solve all the obligations. We do this via a fixed-point
        // iteration. We try to solve each obligation in turn. Anything which is
        // successful, we drop; anything ambiguous, we retain in the
        // `obligations` array. This process is repeated so long as we are
        // learning new things about our inference state.
        let mut obligations = Vec::with_capacity(self.obligations.len());
        let mut progress = true;

        while progress {
            progress = false;
            debug!("start of round, {} obligations", self.obligations.len());

            // Take the list of `obligations` to solve this round and replace it
            // with an empty vector. Iterate through each obligation to solve
            // and solve it if we can. If not (because of ambiguity), then push
            // it back onto `self.to_prove` for next round. Note that
            // `solve_one` may also push onto the `self.to_prove` list
            // directly.
            assert!(obligations.is_empty());
            while let Some((obl_idx, obligation)) = self.obligations.pop() {
                let ambiguous = match &obligation {
                    Obligation::Prove(wc) => {
                        let PositiveSolution {
                            free_vars,
                            universes,
                            solution,
                        } = self.prove(obl_idx, wc.clone(), minimums, should_continue.clone())?;

                        if let Some(constrained_subst) = solution.definite_subst(self.interner()) {
                            // If the substitution is trivial, we won't actually make any progress by applying it!
                            // So we need to check this to prevent endless loops.
                            let nontrivial_subst = !is_trivial_canonical_subst(
                                self.interner(),
                                &constrained_subst.value.subst,
                            );

                            let has_constraints = !constrained_subst
                                .value
                                .constraints
                                .is_empty(self.interner());

                            if nontrivial_subst || has_constraints {
                                self.apply_solution(free_vars, universes, constrained_subst);
                                progress = true;
                            }
                        }

                        solution.is_ambig()
                    }
                    Obligation::Refute(goal) => {
                        let answer = self.refute(obl_idx, goal.clone(), should_continue.clone())?;
                        answer == NegativeSolution::Ambiguous
                    }
                };

                if ambiguous {
                    debug!("ambiguous result: {:?}", obligation);
                    obligations.push((obl_idx, obligation));
                }
            }

            self.obligations.append(&mut obligations);
            debug!("end of round, {} obligations left", self.obligations.len());
        }

        // At the end of this process, `self.obligations` should have
        // all of the ambiguous obligations, and `obligations` should
        // be empty.
        assert!(obligations.is_empty());

        if self.obligations.is_empty() {
            Ok(Outcome::Complete)
        } else {
            Ok(Outcome::Incomplete)
        }
    }

    /// Try to fulfill all pending obligations and build the resulting
    /// solution. The returned solution will transform `subst` substitution with
    /// the outcome of type inference by updating the replacements it provides.
    // TODO(gavinleroy), the tracing changes are quite intrusive here, can we tone it down a bit?
    pub(super) fn solve(
        mut self,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I> {
        // FIXME(gavinleroy): this seems wrong.
        macro_rules! return_traced {
            (UseCurrentSink, $solution:expr) => {{
                let leaf = self
                    .obligations
                    .get_current_sink(self.solver.get_inspector());
                return_traced!($solution, leaf)
            }};
            (CreateLeaf, $solution:expr) => {{
                let value = $solution;
                let leaf = self.solver.get_inspector().push_node(value.clone());
                return_traced!(value, leaf)
            }};
            ($solution:expr, $sink:expr) => {{
                let value = $solution;
                self.obligations.builder.set_result(value.clone());
                let inspector = self.solver.get_inspector();
                let descr = self.mint_proof(Some(&value));
                debug!(?value, "Final answer from solving");
                return TracedFallible::from_built(value, descr);
            }};
        }

        let outcome = self.fulfill(minimums, should_continue.clone());
        debug!(?outcome, "Fulfillment outcome");
        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => return_traced!(CreateLeaf, Err(e)),
        };

        if self.cannot_prove {
            debug!("Goal cannot be proven (cannot_prove = true), returning ambiguous");

            // Make sure we didn't miss something.
            assert!(!matches!(
                self.obligations.builder.did_fail,
                af::FulfillFailKind::Unknown
            ));

            return_traced!(CreateLeaf, Ok(Solution::Ambig(Guidance::Unknown)));
        }

        if outcome.is_complete() {
            // No obligations remain, so we have definitively solved our goals,
            // and the current inference state is the unique way to solve them.

            let constraints = Constraints::from_iter(self.interner(), self.constraints.clone());
            let constrained = canonicalize(
                &mut self.infer,
                self.solver.interner(),
                ConstrainedSubst {
                    subst: self.subst.clone(),
                    constraints,
                },
            );

            return_traced!(UseCurrentSink, Ok(Solution::Unique(constrained.0)));
        }

        // Otherwise, we have (positive or negative) obligations remaining, but
        // haven't proved that it's *impossible* to satisfy out obligations. we
        // need to determine how to package up what we learned about type
        // inference as an ambiguous solution.

        let canonical_subst =
            canonicalize(&mut self.infer, self.solver.interner(), self.subst.clone());

        debug!(?canonical_subst, "Packaging solve info");

        if canonical_subst
            .0
            .value
            .is_identity_subst(self.solver.interner())
        {
            // In this case, we didn't learn *anything* definitively. So now, we
            // go one last time through the positive obligations, this time
            // applying even *tentative* inference suggestions, so that we can
            // yield these upwards as our own suggestions. There are no
            // particular guarantees about *which* obligaiton we derive
            // suggestions from.

            while let Some((obl_idx, obligation)) = self.obligations.pop() {
                if let Obligation::Prove(goal) = obligation {
                    let PositiveSolution {
                        free_vars,
                        universes,
                        solution,
                    } = self
                        .prove(obl_idx, goal, minimums, should_continue.clone())
                        .unwrap();
                    if let Some(constrained_subst) =
                        solution.constrained_subst(self.solver.interner())
                    {
                        self.apply_solution(free_vars, universes, constrained_subst);
                        return_traced!(
                            UseCurrentSink,
                            Ok(Solution::Ambig(Guidance::Suggested(canonical_subst.0)))
                        );
                    }
                }
            }

            return_traced!(UseCurrentSink, Ok(Solution::Ambig(Guidance::Unknown)));
        } else {
            // While we failed to prove the goal, we still learned that
            // something had to hold. Here's an example where this happens:
            //
            // ```rust
            // trait Display {}
            // trait Debug {}
            // struct Foo<T> {}
            // struct Bar {}
            // struct Baz {}
            //
            // impl Display for Bar {}
            // impl Display for Baz {}
            //
            // impl<T> Debug for Foo<T> where T: Display {}
            // ```
            //
            // If we pose the goal `exists<T> { T: Debug }`, we can't say
            // for sure what `T` must be (it could be either `Foo<Bar>` or
            // `Foo<Baz>`, but we *can* say for sure that it must be of the
            // form `Foo<?0>`.
            return_traced!(
                UseCurrentSink,
                Ok(Solution::Ambig(Guidance::Definite(canonical_subst.0)))
            )
        }
    }

    fn interner(&self) -> I {
        self.solver.interner()
    }

    /// Take the current state of the `FulfillmentBuilder` and maintain invariants
    /// in the overall tree builder. This method needs to make sure that the Fulfillment
    /// node is finalized and inserted into the tree, subgoals need to be added as
    /// children to the tree, and the final tree description needs to be built.
    fn mint_proof(self, outcome: Option<&Fallible<Solution<I>>>) -> TreeDescription {
        let builder = self.solver.get_inspector();
        let mut this = self.obligations;

        // XXX: The state of the table and subst change during fulfillment, update them here.
        // this.builder.update_table(self.infer);
        // this.builder.subst = self.subst;

        let sol_is_unique = outcome
            .map(|fallible| matches!(fallible, Ok(Solution::Unique(..))))
            .unwrap_or(false);
        let fail_is_unknown = matches!(this.builder.did_fail, af::FulfillFailKind::Unknown);
        let exists_obligations = this.builder.has_obligations();
        let exists_results = this.builder.has_obligation_results(&builder.nodes);

        assert!(
            // There exists a solution.
            sol_is_unique ||
                // There are obligations and results.
                (exists_obligations && exists_results) ||
                // There is a reason for failure.
                !fail_is_unknown,
            "Fulfillment Builders must have a failure reason {:#?}",
            this.builder
        );

        let mut children = this.builder.obligations.iter().copied().collect::<Vec<_>>();

        let leaf = if exists_results {
            this.get_current_sink(builder).unwrap()
        } else if exists_obligations || !fail_is_unknown {
            let l = builder.push_node(NoSolution);
            children.push(l);
            l
        } else {
            assert!(outcome.is_some());
            let l = builder.push_node(outcome.unwrap().clone());
            children.push(l);
            l
        };

        let root = builder.push_node(this.builder);

        for child in children {
            builder.relate(root, child);
        }

        builder.describe_tree(root, leaf)
    }
}

fn is_trivial_canonical_subst<I: Interner>(interner: I, subst: &Substitution<I>) -> bool {
    // A subst is trivial if..
    subst.iter(interner).enumerate().all(|(index, parameter)| {
        let is_trivial = |b: Option<BoundVar>| match b {
            None => false,
            Some(bound_var) => {
                if let Some(index1) = bound_var.index_if_innermost() {
                    index == index1
                } else {
                    false
                }
            }
        };

        match parameter.data(interner) {
            // All types and consts are mapped to distinct variables. Since this
            // has been canonicalized, those will also be the first N
            // variables.
            GenericArgData::Ty(t) => is_trivial(t.bound_var(interner)),
            GenericArgData::Const(t) => is_trivial(t.bound_var(interner)),
            GenericArgData::Lifetime(t) => is_trivial(t.bound_var(interner)),
        }
    })
}

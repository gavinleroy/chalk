use super::combine;
use super::fulfill::Fulfill;
use crate::fixed_point::Minimums;
use crate::UCanonicalGoal;
use chalk_ir::could_match::CouldMatch;
use chalk_ir::fold::TypeFoldable;
use chalk_ir::interner::{HasInterner, Interner, TSerialize};
use chalk_ir::{
    Canonical, ClausePriority, DomainGoal, Fallible, Floundered, Goal, GoalData, InEnvironment,
    NoSolution, ProgramClause, ProgramClauseData, Substitution, UCanonical,
};
use chalk_solve::clauses::program_clauses_that_could_match;
use chalk_solve::debug_span;
use chalk_solve::infer::InferenceTable;
use chalk_solve::{Guidance, RustIrDatabase, Solution};
use tracing::{debug, instrument};

use itertools::Itertools;

use argus::proof_tree::*;

pub(super) trait SolveDatabase<I: Interner>: Sized {
    fn solve_goal(
        &mut self,
        goal: UCanonical<InEnvironment<Goal<I>>>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I>;

    fn max_size(&self) -> usize;

    fn interner(&self) -> I;

    fn db(&self) -> &dyn RustIrDatabase<I>;
}

/// The `solve_iteration` method -- implemented for any type that implements
/// `SolveDb`.
pub(super) trait SolveIteration<I: Interner>: SolveDatabase<I> {
    /// Executes one iteration of the recursive solver, computing the current
    /// solution to the given canonical goal. This is used as part of a loop in
    /// the case of cyclic goals.
    #[instrument(level = "debug", skip(self, should_continue))]
    fn solve_iteration(
        &mut self,
        canonical_goal: &UCanonicalGoal<I>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I> {
        if !should_continue() {
            return TracedFallible {
                solution: Ok(Solution::Ambig(Guidance::Unknown)),
                trace: ProofTree::proof_halted(),
            };
        }

        let UCanonical {
            universes,
            canonical:
                Canonical {
                    binders,
                    value: InEnvironment { environment, goal },
                },
        } = canonical_goal.clone();

        match goal.data(self.interner()) {
            GoalData::DomainGoal(domain_goal) => {
                let canonical_domain_goal = UCanonical {
                    universes,
                    canonical: Canonical {
                        binders,
                        value: InEnvironment {
                            environment,
                            goal: domain_goal.clone(),
                        },
                    },
                };

                // "Domain" goals (i.e., leaf goals that are Rust-specific) are
                // always solved via some form of implication. We can either
                // apply assumptions from our environment (i.e. where clauses),
                // or from the lowered program, which includes fallback
                // clauses. We try each approach in turn:

                let prog_solution = {
                    debug_span!("prog_clauses");

                    self.solve_from_clauses(&canonical_domain_goal, minimums, should_continue)
                };
                debug!(?prog_solution);

                prog_solution
            }

            _ => {
                let canonical_goal = UCanonical {
                    universes,
                    canonical: Canonical {
                        binders,
                        value: InEnvironment { environment, goal },
                    },
                };

                self.solve_via_simplification(&canonical_goal, minimums, should_continue)
            }
        }
    }
}

impl<S, I> SolveIteration<I> for S
where
    S: SolveDatabase<I>,
    I: Interner,
{
}

/// Helper methods for `solve_iteration`, private to this module.
trait SolveIterationHelpers<'a, I: Interner + 'a>: SolveDatabase<I> {
    #[instrument(level = "debug", skip(self, minimums, should_continue))]
    fn solve_via_simplification(
        &mut self,
        canonical_goal: &UCanonicalGoal<I>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I> {
        let (infer, subst, goal) = self.new_inference_table(canonical_goal);
        match Fulfill::new_with_simplification(self, infer, subst, goal) {
            Ok(fulfill) => fulfill.solve(minimums, should_continue),
            Err(trace) => TracedFallible {
                solution: Err(NoSolution),
                trace,
            },
        }
    }

    /// See whether we can solve a goal by implication on any of the given
    /// clauses. If multiple such solutions are possible, we attempt to combine
    /// them.
    fn solve_from_clauses(
        &mut self,
        canonical_goal: &UCanonical<InEnvironment<DomainGoal<I>>>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I> {
        let db = self.db();
        let could_match = |c: &ProgramClause<I>| {
            c.could_match(
                db.interner(),
                db.unification_database(),
                &canonical_goal.canonical.value.goal,
            )
        };

        // NOTE: instead of maintaining two lists, one with the clauses and
        // one with the "considered clauses". I've modified this code to iterate
        // over the considered clauses, this providing the ClauseIdx and then filtering
        // by those that satisfy the `could_match` predicate.
        //
        // XXX: This is important to observe as it modifies existing Chalk code.
        // I've left the original lines commented out with a `-->`.
        let mut builder = SolveFromClauses::new(canonical_goal.clone());
        let mut extend_considered_with = |clauses: Vec<ProgramClause<I>>, kind| {
            builder.clauses.extend(clauses.into_iter().map(|clause| {
                let is_could_match = could_match(&clause);
                ConsideredClause {
                    clause,
                    could_match: is_could_match,
                    kind,
                }
            }))
        };

        extend_considered_with(db.custom_clauses(), ClauseKind::ForEnv);
        // --> clauses.extend(db.custom_clauses().into_iter().filter(could_match));
        match program_clauses_that_could_match(db, canonical_goal) {
            Ok(goal_clauses) => {
                extend_considered_with(goal_clauses.clone(), ClauseKind::CouldMatch);
                // --> clauses.extend(goal_clauses.into_iter().filter(could_match))
            }
            Err(Floundered) => {
                let solution = Ok(Solution::Ambig(Guidance::Unknown));
                builder.did_flounder = true;
                builder.set_outcome(solution.clone());
                return TracedFallible {
                    solution,
                    trace: ProofTree::FromClauses(builder),
                };
            }
        }

        let (infer, subst, goal) = self.new_inference_table(canonical_goal);

        extend_considered_with(
            db.program_clauses_for_env(&goal.environment)
                .iter(db.interner())
                .cloned()
                .collect::<Vec<_>>(),
            ClauseKind::ForEnv,
        );
        // --> clauses.extend(
        //     db.program_clauses_for_env(&goal.environment)
        //         .iter(db.interner())
        //         .cloned()
        //         .filter(could_match),
        // );

        let mut cur_solution = None;

        // inline to avoid borrow conflicts, iterate over each
        // of the considered clauses and filter by those which
        // "could match".
        let clauses_enumerated = builder
            .clauses
            .iter_enumerated()
            .filter_map(|(idx, considered)| {
                considered.could_match.then_some((idx, &considered.clause))
            })
            // FIXME(gavinleroy): does this work? You need to ask on Zulip to see if
            //      anyone has any intuition as to why it would or wouldn't.
            // .unique_by(|&(_, considered)| considered)
            ;

        // --> for program_clause in clauses {
        for (clause_idx, program_clause) in clauses_enumerated {
            debug_span!("solve_from_clauses", clause = ?program_clause);

            let ProgramClauseData(implication) = program_clause.data(self.interner());
            let infer = infer.clone();
            let subst = subst.clone();
            let goal = goal.clone();
            let (res, tree) =
                match Fulfill::new_with_clause(self, infer.clone(), subst, goal, implication) {
                    Ok(fulfill) => {
                        let TracedFallible { solution, trace } =
                            fulfill.solve(minimums, should_continue.clone());
                        ((solution, implication.skip_binders().priority), trace)
                    }
                    Err(trace) => ((Err(NoSolution), ClausePriority::High), trace),
                };

            // Store the subtree for the specific clause.
            builder.subnodes.insert(clause_idx, tree);

            if let (Ok(solution), priority) = res {
                debug!(?solution, ?priority, "Ok");
                cur_solution = Some(match cur_solution {
                    None => (solution, priority),
                    Some((cur, cur_priority)) => {
                        let (cur, cur_priority) = combine::with_priorities(
                            self.interner(),
                            &canonical_goal.canonical.value.goal,
                            cur,
                            cur_priority,
                            solution,
                            priority,
                        );
                        (cur, cur_priority)
                    }
                });
            } else {
                debug!("Error");
            }

            builder
                .iteration_order
                .push((clause_idx, cur_solution.clone()));

            if let Some((cur_solution, _)) = &cur_solution {
                if cur_solution.is_trivial_and_always_true(self.interner()) {
                    break;
                }
            }
        }

        let solution = if let Some((s, _)) = cur_solution {
            debug!("solve_from_clauses: result = {:?}", s);
            Ok(s.clone())
        } else {
            debug!("solve_from_clauses: error");
            Err(NoSolution)
        };

        builder.set_outcome(solution.clone());
        TracedFallible {
            solution,
            trace: ProofTree::FromClauses(builder),
        }
    }

    fn new_inference_table<T: TypeFoldable<I> + HasInterner<Interner = I> + Clone + TSerialize>(
        &self,
        ucanonical_goal: &UCanonical<InEnvironment<T>>,
    ) -> (InferenceTable<I>, Substitution<I>, InEnvironment<T>) {
        let (infer, subst, canonical_goal) = InferenceTable::from_canonical(
            self.interner(),
            ucanonical_goal.universes,
            ucanonical_goal.canonical.clone(),
        );
        (infer, subst, canonical_goal)
    }
}

impl<'a, S, I> SolveIterationHelpers<'a, I> for S
where
    S: SolveDatabase<I>,
    I: Interner + 'a,
{
}

use std::{fs::File, str::FromStr};
use std::io::Write;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Deserialize, Debug)]
struct ProcDefinition {
    reactions: Vec<RxDefinition>,
    initial: HashMap<String, Either<u64, String>>,
    #[serde(default)]
    conf: HashMap<String, yaml_serde::Value>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum Either<L, R> {
    Left(L),
    Right(R)
}

#[derive(Deserialize, Debug)]
struct RxDefinition {
    ins: Vec<Either<String, (u64, String)>>,
    outs: Vec<Either<String, (u64, String)>>,
    rate: Either<f64, String>,
    #[serde(alias = "rev-rate")]
    rev_rate: Option<Either<f64, String>>,
}

#[derive(Debug, Default, Clone)]
struct Reaction {
    // (stoichiometry, species_index)
    reactants: Vec<(u64, usize)>,
    // (stoichiometry, species_index)
    products: Vec<(u64, usize)>,
    // rate constant k such that v = k*[Sp1]^stoich1 * [Sp2]^stoich2 ...
    rate: f64,
    // if this reaction has a reverse, this is its index, otherwise usize::MAX
    rev: usize,
}

#[derive(Debug)]
struct ChemicalProcess {
    // species[i] = name of species i
    species: Vec<String>,
    reactions: Vec<Reaction>,
}

#[derive(Debug, Clone)]
struct ChemicalState {
    time: f64,
    counts: Vec<u64>,
}

fn parse_definitions(proc: &ProcDefinition) -> (ChemicalProcess, ChemicalState) {
    let mut context = meval::Context::new();
    if let Some(vars) = proc.conf.get("vars") {
        if let Some(vars) = vars.as_mapping() {
            for (k, v) in vars {
                let k = k.as_str().expect("vars key must be a string");
                let v = v.as_f64().expect("vars value must be a number");
                context.var(k, v);
            }
        }
    }

    let mut species_set = HashMap::new();
    let mut reactions = Vec::new();

    let mut species = |s: &str| {
        let l = species_set.len();
        *species_set.entry(s.to_string()).or_insert(l)
    };

    for def in &proc.reactions {
        let mut rx = Reaction::default();
        for x in &def.ins {
            rx.reactants.push(match x {
                Either::Left(x) => (1, species(x)),
                Either::Right(x) => (x.0, species(&x.1)),
            });
        }
        for x in &def.outs {
            rx.products.push(match x {
                Either::Left(x) => (1, species(x)),
                Either::Right(x) => (x.0, species(&x.1)),
            });
        }
        match &def.rate {
            Either::Left(x) => rx.rate = *x,
            Either::Right(s) => {
                let expr = meval::Expr::from_str(&s).expect("invalid rate expression");
                rx.rate = expr.eval_with_context(&context).expect("failed to evaluate rate expression");
            },
        }
        if let Some(rev_rate) = &def.rev_rate {
            let mut rev_rx = rx.clone();
            std::mem::swap(&mut rev_rx.reactants, &mut rev_rx.products);
            match rev_rate {
                Either::Left(x) => rev_rx.rate = *x,
                Either::Right(s) => {
                    let expr = meval::Expr::from_str(&s).expect("invalid reverse rate expression");
                    rev_rx.rate = expr.eval_with_context(&context).expect("failed to evaluate reverse rate expression");
                },
            }
            rev_rx.rev = reactions.len() + 1;
            rx.rev = reactions.len();
            reactions.push(rev_rx);
        } else {
            rx.rev = usize::MAX;
        }
        reactions.push(rx);
    }

    let mut init_state = vec![0; species_set.len()];
    let mut relative_species = Vec::new();
    for (k, v) in &proc.initial {
        let i = *species_set.get(k).unwrap();
        match v {
            Either::Left(n) => init_state[i] = *n,
            Either::Right(s) => {
                if s.ends_with("M") {
                    let c = s[..s.len()-1].parse::<f64>().unwrap();
                    relative_species.push((i, c));
                }
            }
        }
    }
    if !relative_species.is_empty() {
        let particles_count = proc.conf.get("particles").expect("missing particle count in conf").as_u64().unwrap();
        let total_conc = relative_species.iter().map(|x| x.1).sum::<f64>();
        for (i, c) in relative_species {
            let n = (c / total_conc * particles_count as f64).round() as u64;
            if n == 0 {
                println!("Warning: A species has 0 particles, consider increasing $Particles");
            }
            init_state[i] = n;
        }
    }

    let mut species = species_set.into_iter().collect::<Vec<_>>();
    species.sort_by_key(|x| x.1);
    let species = species.into_iter().map(|x| x.0).collect::<Vec<_>>();

    (ChemicalProcess { species, reactions }, ChemicalState { time: 0.0, counts: init_state })
}

fn main() {
    let mut f = std::fs::File::open("process3.yaml").unwrap();
    let p: ProcDefinition = yaml_serde::from_reader(&mut f).unwrap();
    let (proc, state) = parse_definitions(&p);
    let nsteps = p.conf.get("steps").expect("missing steps in conf").as_u64().unwrap();
    let nrepeats = p.conf.get("repeats").map(|x| x.as_u64().unwrap()).unwrap_or(1);
    
    let pool = threadpool::Builder::new().build();

    let proc: &'static ChemicalProcess = &*Box::leak(Box::new(proc));
    for i in 0..nrepeats {
        let state = state.clone();
        pool.execute(move || {
            let mut record_file = File::create(format!("record_{}.csv", i)).unwrap();
            writeln!(record_file, "time,{}", proc.species.join(",")).unwrap();

            let t1 = std::time::SystemTime::now();
            gillespie_stochastic_sim_record(
                proc,
                state,
                nsteps,
                &mut |_i, st, _, _| {
                    writeln!(record_file, "{},{}", st.time, st.counts.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")).unwrap();
                }
            );
            let t2 = std::time::SystemTime::now();
            println!("Simulation took {} ms", t2.duration_since(t1).unwrap().as_millis());

        });
    }
    pool.join();
}

impl StochasticProcess for ChemicalProcess {
    type State = ChemicalState;
    
    fn rates_max_size(&self) -> usize { self.reactions.len() }
    
    fn calc_propensities(&self, state: &Self::State, propensities: &mut [f64]) {
        for (i, rx) in self.reactions.iter().enumerate() {
            let mut a = rx.rate;
            for (stoich, sp) in &rx.reactants {
                // combinatorial propensity (n over k), n = number of particles, k = stoichiometric coefficient
                if state.counts[*sp] < *stoich {
                    a = 0.0;
                    break;
                }
                for k in 0..*stoich {
                    a *= (state.counts[*sp] - k) as f64 / (k + 1) as f64;
                }
            }
            propensities[i] = a;
        }
    }
    
    fn step_state(&self, state: &mut Self::State, chosen_step: usize, dt: f64) {
        state.time += dt;
        let rx = &self.reactions[chosen_step];
        for (stoich, sp) in &rx.reactants {
            state.counts[*sp] = state.counts[*sp].saturating_sub(*stoich as u64);
        }
        for (stoich, sp) in &rx.products {
            state.counts[*sp] = state.counts[*sp].saturating_add(*stoich as u64);
        }
    }
    
    fn are_steps_reverse(&self, step1: usize, step2: usize) -> bool {
        self.reactions[step1].rev == step2
    }
}

trait StochasticProcess {
    type State;
    
    fn rates_max_size(&self) -> usize;
    fn are_steps_reverse(&self, step1: usize, step2: usize) -> bool;
    fn calc_propensities(&self, state: &Self::State, propensities: &mut [f64]);
    fn step_state(&self, state: &mut Self::State, chosen_step: usize, dt: f64);
}

const USE_SKIPPING: bool = false;

fn gillespie_stochastic_sim_record<P: StochasticProcess>(
    proc: &P, 
    init: P::State,
    nsteps: u64,
    recorder: &mut impl FnMut(u64, &P::State, usize, f64)) {
    let mut state = init;
    let mut propensities = vec![0.0f64; proc.rates_max_size()];
    let mut selected_step = usize::MAX;
    let mut dt = 0.0;
    let mut step_history = [usize::MAX; 4];
    let mut step_history_idx = 0;

    for i in 0..nsteps {
        //println!("{:?}", step_history);
        recorder(i, &state, selected_step, dt);

        proc.calc_propensities(&state, &mut propensities);

        if USE_SKIPPING {
            let i0 = step_history_idx;
            let i1 = (step_history_idx + 1) % 4;
            let i2 = (step_history_idx + 2) % 4;
            let i3 = (step_history_idx + 3) % 4;
            if step_history[i3] == step_history[i1] &&
                step_history[i2] == step_history[i0] &&
                step_history[i3] != step_history[i2] &&
                proc.are_steps_reverse(step_history[i3], step_history[i2]) {
                    // verify this is actually a hot path
                    if (propensities[step_history[i3]] + propensities[step_history[i2]]) / propensities.iter().sum::<f64>() > 0.5 {
                        propensities[step_history[i3]] = 0.0;
                        propensities[step_history[i2]] = 0.0;
                    }
                }
        }

        (selected_step, dt) = gillespie_stochastic_step(&*propensities);
        if selected_step == usize::MAX {
            break;
        }
        proc.step_state(&mut state, selected_step, dt);

        if USE_SKIPPING {
            step_history[step_history_idx] = selected_step;
            step_history_idx = (step_history_idx + 1) % 4;
        }
    }
}

fn gillespie_stochastic_step(propensities: &[f64]) -> (usize, f64) {
    let a0: f64 = propensities.iter().sum();
    if a0 == 0.0 {
        return (usize::MAX, 0.0)
    }
    //println!("[{}]", propensities.iter().map(|x| format!("{:.2e}", x)).collect::<Vec<_>>().join(", "));
    let r1: f64 = rand::random_range(0.0..1.0f64);
    let tau = (1.0/a0) * (1.0/r1).ln();
    let threshold: f64 = rand::random_range(0.0..1.0f64) * a0;
    let mut cum = 0.0;
    let mut i = 0;
    while i < propensities.len() - 1 {
        cum += propensities[i];
        if cum > threshold {
            break
        }
        i += 1;
    }
    (i, tau)
}

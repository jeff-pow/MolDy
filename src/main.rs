use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;
use rand::Rng;

const KB: f64 = 1.38e-23;
const NA: f64 = 6.022e23;

const NUM_TIME_STEPS: i32 = 10000;
const DT_STAR: f64 = 0.001;

const N: i32 = 500;
const SIGMA: f64 = 3.405;
const EPSILON: f64 = 1.654e-21;
const EPS_STAR: f64 = EPSILON / KB;

const RHOSTAR: f64 = 0.6;
const RHO: f64 = RHOSTAR / (SIGMA * SIGMA * SIGMA);
const R_CUTOFF: f64 = SIGMA * 2.5;
const T_STAR: f64 = 1.24;
const TARGET_TEMP: f64 = T_STAR * EPS_STAR;
const MASS: f64 = 39.9 * 10. / NA / KB;
const TARGET_CELL_LENGTH: f64 = R_CUTOFF;


macro_rules! timeStep {
    () => { 
        (DT_STAR * f64::sqrt(MASS * SIGMA * SIGMA / EPS_STAR))
    };
}

macro_rules! L {
    () => {
        (f64::cbrt(N as f64 / RHO))
    };
}

macro_rules! rCutoffSquared {
    () => {
        (R_CUTOFF * R_CUTOFF)
    };
}

struct Atom {
    positions: [f64; 3],
    velocities: [f64; 3],
    accelerations: [f64; 3],
    old_accelerations: [f64; 3],
}

fn main() {
    let mut file = File::create("out.xyz").expect("File not created");

    let mut ke: Vec<f64> = Vec::new();
    let mut pe: Vec<f64> = Vec::new();
    let mut total_e: Vec<f64>  = Vec::new();

    let mut atoms = face_centered_cell();

    let mut rng = rand::thread_rng();
    for atom in atoms.iter_mut() {
        for j in 0..3 {
            atom.velocities[j] = rng.gen_range(-1.0..1.0);
        }    
    }
    
    thermostat(&mut atoms);

    let cell_interaction_indexes = calc_cell_interactions();

    let mut count = 0.01;
    for time in 0..NUM_TIME_STEPS {
        if time as f64 > count * NUM_TIME_STEPS as f64 {
            println!("{}%", count * 100.);
            count += 0.01;
        }

        write_positions(&mut atoms, &mut file, time);

        for atom in atoms.iter_mut() {
            for k in 0..3 {
                atom.positions[k] += atom.velocities[k] * timeStep!() + 0.5 * atom.accelerations[k] * timeStep!() * timeStep!();
                atom.positions[k] += -1. * L!() * f64::floor(atom.positions[k] / L!());
            }
        }

        let net_potential = calc_forces(&mut atoms);

        let mut total_vel_squared = 0.;

        for atom in atoms.iter_mut() {
            for k in 0..3 {
                atom.velocities[k] += 0.5 * (atom.accelerations[k] + atom.old_accelerations[k]) * timeStep!();
                total_vel_squared += atom.velocities[k] * atom.velocities[k];
            }
        }

        if time < NUM_TIME_STEPS / 2 && time % 5 == 0 {
            thermostat(&mut atoms);
        }

        if time > NUM_TIME_STEPS / 2 {
            let net_ke = 0.5 * MASS * total_vel_squared;

            ke.push(net_ke);
            pe.push(net_potential);
            total_e.push(net_ke + net_potential);
        }
    }

    let mut avg = 0.;
    for i in &pe {
        avg += i;
    }
    avg /= pe.len() as f64;

    let sigma_over_l_over_two = SIGMA / (L!() / 2.);
    let mut long_range_potential_corrections = (8.0 / 3.0) * PI as f64 * N as f64 * RHOSTAR * EPS_STAR;
    let temp = 1.0 / 3.0 * f64::powf(sigma_over_l_over_two, 9.);
    let temp1 = f64::powf(sigma_over_l_over_two, 3.);
    long_range_potential_corrections *= temp - temp1;
    let pestar = ((avg + long_range_potential_corrections) / N as f64) / EPS_STAR;
    println!("Reduced potential: {}", pestar);
}

fn write_positions(atoms: &mut Vec<Atom>, file: &mut File, time: i32) {
    write!(file, "{}\nTime: {}\n", N, time).expect("File not found");
    for atom in atoms.iter_mut() {
        writeln!(file, "A {} {} {}", atom.positions[0], atom.positions[1], atom.positions[2]).expect("File not found");
    }
}

fn calc_cell_index(x: i32, y: i32, z:i32) -> i32 {
    let cells_per_dimension = f64::floor(L!() / TARGET_CELL_LENGTH);
    let cell_length = L!() / cells_per_dimension;
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;
    x * cells_2d as i32 + y * cells_per_dimension as i32 + z
}

fn calc_cell_from_index(idx: i32) -> [i32; 3] {
    let cells_per_dimension = f64::floor(L!() / TARGET_CELL_LENGTH);
    let cell_length = L!() / cells_per_dimension;
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;
    let mut arr = [0; 3];
    arr[0] = idx / cells_2d as i32;
    let remainder = idx % cells_2d as i32;
    arr[1] = remainder / cells_per_dimension as i32;
    arr[2] = remainder % cells_per_dimension as i32;
    arr
}

fn shift_neighbor(x: i32, y: i32, z: i32) -> [i32; 3] {
    let cells_per_dimension = f64::floor(L!() / TARGET_CELL_LENGTH);
    let cell_length = L!() / cells_per_dimension;
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;
    let mut arr = [0; 3];
    arr[0] = (x + cells_per_dimension as i32) % cells_per_dimension as i32;
    arr[1] = (y + cells_per_dimension as i32) % cells_per_dimension as i32;
    arr[2] = (z + cells_per_dimension as i32) % cells_per_dimension as i32;
    arr
}

fn process_cell(x: i32, y: i32, z: i32) -> i32 {
    let arr = shift_neighbor(x, y, z);
    calc_cell_index(arr[0], arr[1], arr[2])
}

fn calc_cell_interactions() -> Vec<Vec<i32>> {
    let cells_per_dimension = f64::floor(L!() / TARGET_CELL_LENGTH);
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;

    let mut cell_interaction_indexes = Vec::new();

    for i in 0..cells_3d as i32 {
        let mut arr: Vec<i32> = Vec::new();
        let cell = calc_cell_from_index(i);

        arr.push(process_cell(cell[0], cell[1], cell[2]));
        arr.push(process_cell(cell[0], cell[1], cell[2] + 1));
        arr.push(process_cell(cell[0], cell[1] + 1, cell[2] - 1));
        arr.push(process_cell(cell[0], cell[1] + 1, cell[2]));
        arr.push(process_cell(cell[0], cell[1] + 1, cell[2] + 1));

        // Next level above
        arr.push(process_cell(cell[0] + 1, cell[1] - 1, cell[2] - 1));
        arr.push(process_cell(cell[0] + 1, cell[1] - 1, cell[2]));
        arr.push(process_cell(cell[0] + 1, cell[1] - 1, cell[2] + 1));
        arr.push(process_cell(cell[0] + 1, cell[1], cell[2] - 1));
        arr.push(process_cell(cell[0] + 1, cell[1], cell[2]));
        arr.push(process_cell(cell[0] + 1, cell[1], cell[2] + 1));
        arr.push(process_cell(cell[0] + 1, cell[1] + 1, cell[2] - 1));
        arr.push(process_cell(cell[0] + 1, cell[1] + 1, cell[2]));
        arr.push(process_cell(cell[0] + 1, cell[1] + 1, cell[2] + 1));

        cell_interaction_indexes.push(arr);
    }
    cell_interaction_indexes
}

fn thermostat(atoms: &mut Vec<Atom>) {
    let mut instant_temp: f64 = 0.;
    for atom in atoms.iter_mut() {
        instant_temp += MASS * dot(atom.velocities[0], atom.velocities[1], atom.velocities[2]);
    }
    instant_temp /= (3 * N - 3) as f64;
    let temp_scalar = f64::sqrt(TARGET_TEMP / instant_temp);
    for atom in atoms.iter_mut() {
        for j in 0..3 {
            atom.velocities[j] *= temp_scalar;
        }
    }
}

#[inline]
fn dot(x: f64, y: f64, z: f64) -> f64 { x * x + y * y + z * z }

fn face_centered_cell() -> Vec<Atom> {
    let n: i32 = f64::cbrt(N as f64 / 4.) as i32;
    let dr: f64 = L!() / n as f64;
    let dro2: f64 = dr / 2.0;

    let mut atoms: Vec<Atom> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                atoms.push(Atom{positions:[i as f64 * dr, j as f64 * dr, k as f64 * dr], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
                atoms.push(Atom{positions:[i as f64 * dr + dro2, j as f64 * dr + dro2, k as f64 * dr], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
                atoms.push(Atom{positions:[i as f64 * dr + dro2, j as f64 * dr, k as f64 * dr + dro2], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
                atoms.push(Atom{positions:[i as f64 * dr, j as f64 * dr + dro2, k as f64 * dr + dro2], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
            }
        }
    }
    return atoms;
}

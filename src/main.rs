#![warn(non_snake_case)]
use rand::Rng;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

//mod bak;

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
const R_CUTOFF_SQUARED: f64 = R_CUTOFF * R_CUTOFF;
const T_STAR: f64 = 1.24;
const TARGET_TEMP: f64 = T_STAR * EPS_STAR;
const MASS: f64 = 39.9 * 10. / NA / KB;
const TARGET_CELL_LENGTH: f64 = R_CUTOFF;

fn main() {
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH);
    let cell_length = sim_length / cells_per_dimension;
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;
    println!("{} cells per dimension", cells_per_dimension);
    println!("{} atoms", N);
    println!("{} cells overall", cells_3d);
    println!("{} cell length", cell_length);
    let mut f = BufWriter::new(File::create("out.xyz").unwrap());

    let mut ke: Vec<f64> = Vec::new();
    let mut pe: Vec<f64> = Vec::new();
    let mut total_e: Vec<f64> = Vec::new();

    let mut accelerations: [[f64; 3]; N as usize] = [[0.0; 3]; N as usize];
    let mut old_accelerations: [[f64; 3]; N as usize];
    let mut positions = face_centered_cell();
    let mut velocities: [[f64; 3]; N as usize] = [[0.0; 3]; N as usize];
    let mut rng = rand::thread_rng();

    for element in velocities.iter_mut().flatten() {
        *element = rng.gen_range(-1.0..1.0);
    }

    thermostat(&mut velocities);

    let time_step = DT_STAR * f64::sqrt(MASS * SIGMA * SIGMA / EPS_STAR);
    let mut count = 0.05;
    for time in 0..NUM_TIME_STEPS {
        if time as f64 > count * NUM_TIME_STEPS as f64 {
            println!("{}%", (count * 100.) as i32);
            count += 0.05;
        }

        write_positions(&positions, &mut f, time);

        for (idx, pos) in positions.iter_mut().enumerate() {
            (0..3).for_each(|k| {
                pos[k] += velocities[idx][k] * time_step
                    + 0.5 * accelerations[idx][k] * time_step * time_step;
                pos[k] += -1. * sim_length * f64::floor(pos[k] / sim_length);
            });
        }
        old_accelerations = accelerations;

        accelerations = [[0.0; 3]; N as usize];
        let net_potential = calc_forces(&positions, &mut accelerations);

        let mut total_vel_squared = 0.;
        for (idx, vels) in velocities.iter_mut().enumerate() {
            for k in 0..3 {
                vels[k] += 0.5 * (accelerations[idx][k] + old_accelerations[idx][k]) * time_step;
                total_vel_squared += vels[k] * vels[k];
            }
        }

        if time < NUM_TIME_STEPS / 2 && time % 5 == 0 {
            thermostat(&mut velocities);
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

    let sigma_over_l_over_two = SIGMA / (sim_length / 2.);
    let mut long_range_potential_corrections = (8.0 / 3.0) * PI as f64 * N as f64 * RHOSTAR * EPS_STAR;
    let temp = 1.0 / 3.0 * f64::powf(sigma_over_l_over_two, 9.);
    let temp1 = f64::powf(sigma_over_l_over_two, 3.);
    long_range_potential_corrections *= temp - temp1;
    let pestar = ((avg + long_range_potential_corrections) / N as f64) / EPS_STAR;
    println!("Avg PE: {avg}");
    println!("Average energy: {}", total_e[total_e.len() - 1]);
    println!("Reduced potential: {}", pestar);

}

fn calc_forces(positions: &[[f64; 3]], accelerations: &mut [[f64; 3]; N as usize]) -> f64 {
    let mut net_potential = 0.0;
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = (f64::floor(sim_length / TARGET_CELL_LENGTH)) as i32;
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;
    let cell_length = sim_length / cells_per_dimension as f64;

    let mut header = vec![-1; cells_3d as usize];
    let mut cell_list = [0; N as usize];

    for atom_idx in 0..N {
        let x = (positions[atom_idx as usize][0] / cell_length) as i32;
        let y = (positions[atom_idx as usize][1] / cell_length) as i32;
        let z = (positions[atom_idx as usize][2] / cell_length) as i32;
        // Turn coordinates of cell into a cell index for the header array
        let c = x * cells_2d + y * cells_per_dimension + z;
        // Link current atom to previous occupant
        cell_list[atom_idx as usize] = header[c as usize];
        // Current atom is the highest in its cell, so it goes in the header
        header[c as usize] = atom_idx;
    }

    for c in 0..cells_3d {
        net_potential += calc_forces_on_cell(c, accelerations, positions, &header, &cell_list);
    }

    net_potential
}

fn calc_forces_on_cell(
    cell_idx: i32,
    accel: &mut [[f64; 3]; N as usize],
    pos: &[[f64; 3]],
    header: &[i32],
    cell_list: &[i32],
) -> f64 {
    let mut potential = 0.0;
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH);
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let x = cell_idx / cells_2d as i32;
    let remainder = cell_idx % cells_2d as i32;
    let y = remainder / cells_per_dimension as i32;
    let z = remainder % cells_per_dimension as i32;

    for i in x - 1..x + 2 {
        for j in y - 1..y + 2 {
            for k in z - 1..z + 2 {

                let x = (i + cells_per_dimension as i32) % cells_per_dimension as i32;
                let y = (j + cells_per_dimension as i32) % cells_per_dimension as i32;
                let z = (k + cells_per_dimension as i32) % cells_per_dimension as i32;
                let neighbor_idx = x * cells_2d as i32 + y * cells_per_dimension as i32 + z;
                let mut a = header[cell_idx as usize];
                while a > -1 {
                    let mut b = header[neighbor_idx as usize];
                    while b > -1 {
                        if a < b {
                            let mut dist_arr = [0.; 3];
                            for ii in 0..3 {
                                dist_arr[ii] = pos[a as usize][ii] - pos[b as usize][ii];
                                dist_arr[ii] -= sim_length * f64::round(dist_arr[ii] / sim_length);
                            }
                            let r2 = dot(dist_arr[0], dist_arr[1], dist_arr[2]); // Dot of distance vector between the two atoms
                            if r2 < R_CUTOFF_SQUARED {
                                let s2or2 = SIGMA * SIGMA / r2; // Sigma squared over r squared
                                let sor6 = s2or2 * s2or2 * s2or2; // Sigma over r to the sixth
                                let sor12 = sor6 * sor6; // Sigma over r to the twelfth
                                let force_over_r = 24. * EPS_STAR / r2 * (2. * sor12 - sor6);
                                potential += 4. * EPS_STAR * (sor12 - sor6);
                                for ii in 0..3 {
                                    accel[a as usize][ii] += force_over_r * dist_arr[ii] / MASS;
                                    accel[b as usize][ii] -= force_over_r * dist_arr[ii] / MASS;
                                }
                            }
                        }
                        b = cell_list[b as usize];
                    }
                    a = cell_list[a as usize];
                }
            }
        }
    }
    potential
}

fn write_positions(pos: &[[f64; 3]], file: &mut BufWriter<File>, time: i32) {
    write!(file, "{}\nTime: {}\n", N, time).expect("File not found");
    for atom in pos.iter() {
        writeln!(file, "A {} {} {}", atom[0], atom[1], atom[2]).expect("File not found");
    }
}

fn thermostat(velocities: &mut [[f64; 3]; N as usize]) {
    let mut instant_temp: f64 = velocities.iter().map(|&x| dot(x[0], x[1], x[2])).sum();
    instant_temp /= (3 * N - 3) as f64;
    let temp_scalar = f64::sqrt(TARGET_TEMP / instant_temp);
    for vel in velocities.iter_mut().flatten() {
        *vel *= temp_scalar;
    }
}

#[inline]
fn dot(x: f64, y: f64, z: f64) -> f64 {
    x * x + y * y + z * z
}

fn face_centered_cell() -> Vec<[f64; 3]> {
    let n: i32 = f64::cbrt(N as f64 / 4.) as i32;
    let sim_length = f64::cbrt(N as f64 / RHO);
    let dr: f64 = sim_length / n as f64;
    let dro2: f64 = dr / 2.0;

    let mut positions: Vec<[f64; 3]> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                positions.push([i as f64 * dr, j as f64 * dr, k as f64 * dr]);
                positions.push([i as f64 * dr + dro2, j as f64 * dr + dro2, k as f64 * dr]);
                positions.push([i as f64 * dr + dro2, j as f64 * dr, k as f64 * dr + dro2]);
                positions.push([i as f64 * dr, j as f64 * dr + dro2, k as f64 * dr + dro2]);
            }
        }
    }
    positions
}

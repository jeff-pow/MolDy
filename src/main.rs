#![warn(non_snake_case)]
#![allow(dead_code)]
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

const KB: f64 = 1.3806e-23; // J / K
const NA: f64 = 6.022e23; // Avogadro's number

const NUM_TIME_STEPS: i32 = 5000;
const DT_STAR: f64 = 0.001; // Reduced units of time
// https://en.wikipedia.org/wiki/Lennard-Jones_potential#Dimensionless_(reduced_units)

// Formula to find # atoms is x^3 * 4
const N: i32 = 500;

const SIGMA: f64 = 3.405; // Resting distance between argon in angstroms
const EPSILON: f64 = 1.654e-21; // Joules
const EPS_STAR: f64 = EPSILON / KB; // ~ 119.8 K

const RHOSTAR: f64 = 0.6; // Dimensionless density of gas
const RHO: f64 = RHOSTAR / (SIGMA * SIGMA * SIGMA); // Density in 1 / A^3
const R_CUTOFF: f64 = SIGMA * 2.5; // Forces are negligible past this point, not worth calculating
const R_CUTOFF_SQUARED: f64 = R_CUTOFF * R_CUTOFF;
const T_STAR: f64 = 1.24; // Reduced target temperature
const TARGET_TEMP: f64 = T_STAR * EPS_STAR;
// 39.9 is mass of argon in amu, 10 is a conversion between the missing units :)
const MASS: f64 = 39.9 * 10. / NA / KB;
const TARGET_CELL_LENGTH: f64 = R_CUTOFF;

/// Method handles main loop of the simulation
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
    println!();

    let mut _dbg_file = BufWriter::new(File::create("dbg.txt").unwrap());

    let mut ke = Vec::new();
    let mut pe = Vec::new();
    let mut total_e = Vec::new();

    let mut accel = Arc::new((0..N).map(|_| RwLock::from([0.0; 3])).collect::<Vec<_>>());
    let mut pos = face_centered_cell();
    let mut rng: StdRng = rand::SeedableRng::from_seed([3; 32]);
    let mut vel = (0..N)
        .map(|_| {
            [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ]
        })
        .collect::<Vec<_>>();
    let cell_interaction_indexes = calc_cell_interactions();

    thermostat(&mut vel);

    #[allow(clippy::type_complexity)]
    let (sender, reciever): (
        Sender<(Box<Vec<[f64; 3]>>, i32)>,
        Receiver<(Box<Vec<[f64; 3]>>, i32)>,
    ) = mpsc::channel();
    let io_thread = thread::spawn(move || {
        let mut thread_file = BufWriter::new(File::create("thread-file.xyz").unwrap());
        for entry in reciever {
            write_positions(&entry.0, &mut thread_file, entry.1);
            if entry.1 == NUM_TIME_STEPS - 1 {
                break;
            }
        }
    });

    let time_step = DT_STAR * f64::sqrt(MASS * SIGMA * SIGMA / EPS_STAR);
    let start = Instant::now();
    for time in 0..NUM_TIME_STEPS {
        let progress = (time as f64 / NUM_TIME_STEPS as f64) * 100.;
        let duration = start.elapsed();
        let time_left = estimate_time_left(progress, duration).unwrap();
        print!("\r                                                   ");
        print!(
            "\r{:.1}% -- {:?} elapsed -- {:?} left",
            progress,
            format!("{:.1}", duration.as_secs_f64()),
            time_left
        );
        std::io::stdout().flush().unwrap();

        // Send a copy of the positions to the thread that writes to the output file
        sender.send((Box::new(pos.clone()), time)).unwrap();

        let old_accel = accel
            .iter()
            .map(|lock| *lock.read().unwrap())
            .collect::<Vec<_>>();
        pos.iter_mut()
            .flatten()
            .zip(vel.iter().flatten())
            .zip(accel.iter().flat_map(|lock| *lock.read().unwrap()))
            .for_each(|((pos, vel), accel)| {
                // Third equation of kinematics
                *pos += vel * time_step + 0.5 * accel * time_step * time_step;
                // Ensures atoms stay within the virtual box
                *pos -= sim_length * f64::floor(*pos / sim_length);
            });

        accel = Arc::new((0..N).map(|_| RwLock::from([0.0; 3])).collect::<Vec<_>>());
        let net_potential = calc_forces(&pos, &accel, &cell_interaction_indexes);

        // Updates the velocity based on the average of the current and previous timestep
        vel.iter_mut()
            .flatten()
            .zip(accel.iter().flat_map(|lock| *lock.read().unwrap()))
            .zip(old_accel.iter().flatten())
            .for_each(|((vel, accel), old_accel)| *vel += 0.5 * (accel + old_accel) * time_step);

        let total_vel_squared = vel.iter().flatten().map(|x| x * x).sum::<f64>();

        if time < NUM_TIME_STEPS / 2 && time % 5 == 0 {
            thermostat(&mut vel);
        }

        if time > NUM_TIME_STEPS / 2 {
            let net_ke = 0.5 * MASS * total_vel_squared;

            ke.push(net_ke);
            pe.push(net_potential);
            total_e.push(net_ke + net_potential);
        }
    }
    println!("\n");
    let avg_pe = pe.iter().sum::<f64>() / pe.len() as f64;

    let sigma_over_l_over_two = SIGMA / (sim_length / 2.);
    let mut long_range_potential_corrections =
        (8.0 / 3.0) * PI as f64 * N as f64 * RHOSTAR * EPS_STAR;
    let temp = 1.0 / 3.0 * f64::powf(sigma_over_l_over_two, 9.);
    let temp1 = f64::powf(sigma_over_l_over_two, 3.);
    long_range_potential_corrections *= temp - temp1;
    let pestar = ((avg_pe + long_range_potential_corrections) / N as f64) / EPS_STAR;
    println!("Avg PE: {avg_pe}");
    println!("Reduced potential: {}", pestar);

    io_thread.join().unwrap();
    println!("Total time elapsed: {:?}", start.elapsed());
}

fn calc_forces(
    pos: &[[f64; 3]],
    accel: &[RwLock<[f64; 3]>],
    cell_interaction_indexes: &[Vec<i32>],
) -> f64 {
    let net_potential = RwLock::new(0.0);
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = (f64::floor(sim_length / TARGET_CELL_LENGTH)) as i32;
    let cells_2d = cells_per_dimension.pow(2);
    let cells_3d = cells_per_dimension * cells_2d;
    let cell_length = sim_length / cells_per_dimension as f64;

    let mut cell_header = vec![-1; cells_3d as usize];
    let mut atom_cell_list = [-1; N as usize];

    // Sort the atoms into cells for the cell list algorithm
    for atom_idx in 0..N {
        let x = (pos[atom_idx as usize][0] / cell_length) as i32;
        let y = (pos[atom_idx as usize][1] / cell_length) as i32;
        let z = (pos[atom_idx as usize][2] / cell_length) as i32;
        let c = x * cells_2d + y * cells_per_dimension + z;
        atom_cell_list[atom_idx as usize] = cell_header[c as usize];
        cell_header[c as usize] = atom_idx;
    }

    (0..cells_3d).into_par_iter().for_each(|c| {
        let x = calc_forces_on_cell(
            c as usize,
            accel,
            pos,
            &cell_header,
            &atom_cell_list,
            cell_interaction_indexes,
        );

        let mut lock = net_potential.write().unwrap();
        *lock += x
    });

    let net_potential = net_potential.read().unwrap();
    *net_potential
}

/// Method calculates interactions between a single cell and neighboring ones. 
/// One thread is launched for each cell until every cell has been calculated
fn calc_forces_on_cell(
    cell_idx: usize,
    accel: &[RwLock<[f64; 3]>],
    pos: &[[f64; 3]],
    cell_header: &[i32],
    atom_cell_list: &[i32; N as usize],
    cell_interaction_indexes: &[Vec<i32>],
) -> f64 {
    let mut potential = 0.0;
    let sim_length = f64::cbrt(N as f64 / RHO);

    // Iterate through the required indicies
    for neighbor_idx in &cell_interaction_indexes[cell_idx] {
        let mut i = cell_header[cell_idx];
        while i > -1 {
            let mut j = cell_header[*neighbor_idx as usize];
            while j > -1 {
                // i and j keep track of pairs of atoms, if they are in the same cell 
                // calculate the interaction because that pair will never be visited again,
                // otherwise only calculate if i < j, otherwise that pair has already been
                // calculated
                if i < j || cell_idx != *neighbor_idx as usize {
                    let dist_arr = pos[i as usize]
                        .iter()
                        .zip(pos[j as usize].iter())
                        .map(|(&a, &b)| {
                            let diff = a - b;
                            // Calculate distance through the wall of the cell if it is nearer to
                            // simulate boundary conditions
                            diff - sim_length * f64::round(diff / sim_length)
                        })
                        .collect::<Vec<f64>>()
                        .try_into()
                        .unwrap();
                    let r2 = dot(&dist_arr);
                    assert_ne!(r2, 0.);
                    if r2 < R_CUTOFF_SQUARED {
                        let s2or2 = SIGMA * SIGMA / r2;
                        let sor6 = s2or2 * s2or2 * s2or2;
                        let sor12 = sor6 * sor6;
                        let force_over_r = 24. * EPS_STAR / r2 * (2. * sor12 - sor6);
                        potential += 4. * EPS_STAR * (sor12 - sor6);
                        accel[i as usize]
                            .write()
                            .unwrap()
                            .iter_mut()
                            .zip(dist_arr.iter())
                            .for_each(|(accel, dist)| *accel += force_over_r * dist / MASS);
                        accel[j as usize]
                            .write()
                            .unwrap()
                            .iter_mut()
                            .zip(dist_arr.iter())
                            .for_each(|(accel, dist)| *accel -= force_over_r * dist / MASS);
                    }
                }
                j = atom_cell_list[j as usize];
            }
            i = atom_cell_list[i as usize];
        }
    }

    potential
}

fn write_positions(pos: &[[f64; 3]], file: &mut BufWriter<File>, time: i32) {
    write!(file, "{}\nTime: {}\n", N, time).expect("File not found");
    for atom in pos.iter() {
        writeln!(file, "A {:.5} {:.5} {:.5}", atom[0], atom[1], atom[2]).expect("File not found");
    }
}

#[allow(dead_code)]
fn write_dbg(
    pos: &[[f64; 3]],
    vel: &[[f64; 3]],
    accel: &[[f64; 3]],
    old_accel: &[[f64; 3]],
    file: &mut BufWriter<File>,
    time: i32,
) {
    write!(file, "{}\nTime: {}\n", N, time).expect("File not found");
    for (idx, atom) in pos.iter().enumerate() {
        writeln!(
            file,
            "{idx} pos {:.5} {:.5} {:.5}",
            atom[0], atom[1], atom[2]
        )
        .expect("File not found");
    }
    for (idx, atom) in vel.iter().enumerate() {
        writeln!(
            file,
            "{idx} vel {:.5} {:.5} {:.5}",
            atom[0], atom[1], atom[2]
        )
        .expect("File not found");
    }
    for (idx, atom) in accel.iter().enumerate() {
        writeln!(
            file,
            "{idx} accel {:.5} {:.5} {:.5}",
            atom[0], atom[1], atom[2]
        )
        .expect("File not found");
    }
    for (idx, atom) in old_accel.iter().enumerate() {
        writeln!(
            file,
            "{idx} old accel {:.5} {:.5} {:.5}",
            atom[0], atom[1], atom[2]
        )
        .expect("File not found");
    }
}

/// Adjusts velocities to stay near target temperature
fn thermostat(vel: &mut [[f64; 3]]) {
    let instant_temp = vel.iter().map(|x| MASS * dot(x)).sum::<f64>() / (3 * N - 3) as f64;
    let temp_scalar = f64::sqrt(TARGET_TEMP / instant_temp);
    vel.iter_mut().flatten().for_each(|x| *x *= temp_scalar);
}

#[inline]
fn dot(arr: &[f64; 3]) -> f64 {
    arr.iter().map(|x| x * x).sum()
}

/// Initializes simulation in a face centered cell matrix of atoms
fn face_centered_cell() -> Vec<[f64; 3]> {
    let n = f64::round(f64::cbrt(N as f64 / 4.));
    let sim_length = f64::cbrt(N as f64 / RHO);
    let dr = sim_length / n;
    let dro2 = dr / 2.0;

    let mut positions = Vec::new();

    for i in 0..n as i32 {
        for j in 0..n as i32 {
            for k in 0..n as i32 {
                positions.push([i as f64 * dr, j as f64 * dr, k as f64 * dr]);
                positions.push([i as f64 * dr + dro2, j as f64 * dr + dro2, k as f64 * dr]);
                positions.push([i as f64 * dr + dro2, j as f64 * dr, k as f64 * dr + dro2]);
                positions.push([i as f64 * dr, j as f64 * dr + dro2, k as f64 * dr + dro2]);
            }
        }
    }
    positions
}

fn calc_cell_index(x: i32, y: i32, z: i32) -> i32 {
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH);
    let cells_2d = cells_per_dimension * cells_per_dimension;
    x * cells_2d as i32 + y * cells_per_dimension as i32 + z
}

fn calc_cell_from_index(idx: i32) -> [i32; 3] {
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH);
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let mut arr = [0; 3];
    arr[0] = idx / cells_2d as i32;
    let remainder = idx % cells_2d as i32;
    arr[1] = remainder / cells_per_dimension as i32;
    arr[2] = remainder % cells_per_dimension as i32;
    arr
}

fn shift_neighbor(x: i32, y: i32, z: i32) -> [i32; 3] {
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH);
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

/// Precalculates the indicies each cell in the cell list algorithm will have to interact with so
/// that it does not have to be performed on the fly each timestep
fn calc_cell_interactions() -> Vec<Vec<i32>> {
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH);
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;

    let mut cell_interaction_indexes = Vec::new();

    for i in 0..cells_3d as i32 {
        let mut arr: Vec<i32> = Vec::new();
        let cell = calc_cell_from_index(i);

        // arr.push(process_cell(cell[0] - 1, cell[1] - 1, cell[2] - 1));
        // arr.push(process_cell(cell[0] - 1, cell[1] - 1, cell[2]));
        // arr.push(process_cell(cell[0] - 1, cell[1] - 1, cell[2] + 1));
        // arr.push(process_cell(cell[0] - 1, cell[1], cell[2] - 1));
        // arr.push(process_cell(cell[0] - 1, cell[1], cell[2]));
        // arr.push(process_cell(cell[0] - 1, cell[1], cell[2] + 1));
        // arr.push(process_cell(cell[0] - 1, cell[1] + 1, cell[2] - 1));
        // arr.push(process_cell(cell[0] - 1, cell[1] + 1, cell[2]));
        // arr.push(process_cell(cell[0] - 1, cell[1] + 1, cell[2] + 1));

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

fn estimate_time_left(percentage: f64, elapsed_time: Duration) -> Option<Duration> {
    let elapsed_secs = elapsed_time.as_secs_f64();
    let estimated_total_secs = elapsed_secs / (percentage / 100.0);
    let estimated_remaining_secs = estimated_total_secs - elapsed_secs;

    Some(Duration::from_secs(estimated_remaining_secs as u64))
}
